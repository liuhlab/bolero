import pathlib
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from memelite import tomtom
from modiscolite.core import Seqlet, TrackSet
from modiscolite.extract_seqlets import extract_seqlets


def _read_npz(npz_path: str | np.ndarray) -> np.ndarray:
    """Read a .npz file and return the data as a dictionary."""
    if isinstance(npz_path, np.ndarray):
        return npz_path
    with np.load(npz_path) as data:
        arr = data["arr_0"].transpose(0, 2, 1)
    return arr


class SeqletTomtom:
    """Class to perform motif comparison using Tomtom."""

    def __init__(
        self,
        motif_db,
        sliding_window_size=21,
        flank_size=10,
        min_metacluster_size=100,
        target_seqlet_fdr=0.05,
        min_passing_windows_frac=0.03,
        max_passing_windows_frac=0.2,
        weak_threshold_for_counting_sign=0.8,
        n_jobs=-1,
        tomtom_n_nearest=10,
        chunk_size=10000,
        verbose=True,
    ):
        self.motif_db = motif_db

        # Extract seqlet parameters
        self.sliding_window_size = sliding_window_size
        self.flank_size = flank_size
        self.min_metacluster_size = min_metacluster_size
        self.target_seqlet_fdr = target_seqlet_fdr
        self.min_passing_windows_frac = min_passing_windows_frac
        self.max_passing_windows_frac = max_passing_windows_frac
        self.weak_threshold_for_counting_sign = weak_threshold_for_counting_sign

        # TOMTOM parameters
        self.n_jobs = n_jobs
        self.tomtom_n_nearest = tomtom_n_nearest
        self.chunk_size = chunk_size

        self.verbose = verbose

    @staticmethod
    def _load_data(
        one_hot_path: str | np.ndarray,
        hypothetical_contribs_path: str | np.ndarray,
        regions: pd.DataFrame | None = None,
    ):
        one_hot = _read_npz(one_hot_path)
        hypothetical_contribs = _read_npz(hypothetical_contribs_path)

        assert one_hot.shape == hypothetical_contribs.shape, (
            f"Shape mismatch: one_hot and hypothetical_contribs must "
            f"have the same shape (n, seq_len, 4), but got one_hot "
            f"{one_hot.shape} and hypothetical_contribs {hypothetical_contribs.shape}."
        )
        assert one_hot.ndim == 3 and one_hot.shape[2] == 4, (
            f"Expected one_hot and hypothetical_contribs to have "
            f"shape (n, seq_len, 4), but got one_hot {one_hot.shape} "
            f"and hypothetical_contribs {hypothetical_contribs.shape}."
        )
        if regions is not None:
            assert len(regions) == one_hot.shape[0], (
                f"Number of regions ({len(regions)}) must match the number of examples "
                f"in one_hot and hypothetical_contribs ({one_hot.shape[0]})."
            )
        return one_hot, hypothetical_contribs, regions

    def _extract_seqlets(self, one_hot: str, hypothetical_contribs: str):
        contrib_scores = np.multiply(one_hot, hypothetical_contribs)

        track_set = TrackSet(
            one_hot=one_hot,
            contrib_scores=contrib_scores,
            hypothetical_contribs=hypothetical_contribs,
        )

        seqlet_coords, threshold = extract_seqlets(
            attribution_scores=contrib_scores.sum(axis=2),
            window_size=self.sliding_window_size,
            flank=self.flank_size,
            suppress=(int(0.5 * self.sliding_window_size) + self.flank_size),
            target_fdr=self.target_seqlet_fdr,
            min_passing_windows_frac=self.min_passing_windows_frac,
            max_passing_windows_frac=self.max_passing_windows_frac,
            weak_threshold_for_counting_sign=self.weak_threshold_for_counting_sign,
        )

        seqlets = track_set.create_seqlets(seqlet_coords)

        use_seqlets = []
        seqlet_signs = []
        for seqlet in seqlets:
            flank = int(0.5 * (len(seqlet) - self.sliding_window_size))
            attr = np.sum(seqlet.contrib_scores[flank:-flank])

            if attr > threshold:
                seqlet_signs.append(1)
                use_seqlets.append(seqlet)
            elif attr < -threshold:
                seqlet_signs.append(-1)
                seqlet.contrib_scores = -seqlet.contrib_scores
                seqlet.hypothetical_contribs = -seqlet.hypothetical_contribs
                use_seqlets.append(seqlet)
        return use_seqlets, seqlet_signs

    def adjust_seqlets_idx_(self, seqlets: list[Seqlet], chunk_start: int):
        """
        Adjust seqlet indices to account for the chunk start.
        This is necessary to maintain global indexing across chunks.
        """
        for seqlet in seqlets:
            seqlet.example_idx += chunk_start
        return

    def _tomtom(self, seqlets: list[Seqlet]) -> xr.DataArray:
        pwms = [m.pwm.values.T.astype("float32") for m in self.motif_db.motifs]
        scores = [seqlet.contrib_scores.T.astype("float32") for seqlet in seqlets]
        score_names = [seqlet.string for seqlet in seqlets]

        results = tomtom(Qs=scores, Ts=pwms, n_nearest=10, n_jobs=self.n_jobs)

        value_type = [
            "p_values",
            "scores",
            "offsets",
            "overlaps",
            "strands",
            "idxs",
        ]
        results = xr.DataArray(
            results.transpose(1, 2, 0).astype("float32"),
            dims=["seqlet", "motif_rank", "value_type"],
            coords={"value_type": value_type, "seqlet": score_names},
        )
        return results

    def _seqlet_to_da(self, seqlets: list) -> xr.DataArray:
        """Convert seqlets to an xarray DataArray."""
        data = np.array([seqlet.contrib_scores.T for seqlet in seqlets]).astype(
            "float16"
        )
        coords = {
            "seqlet": [seqlet.string for seqlet in seqlets],
            "base": ["A", "C", "G", "T"],
        }
        score = xr.DataArray(data, dims=["seqlet", "base", "position"], coords=coords)
        seq = np.array([seqlet.sequence.T for seqlet in seqlets]).astype("bool")
        seq = xr.DataArray(seq, dims=["seqlet", "base", "position"], coords=coords)
        return score, seq

    def _annotate_regions(self, chunk_ds, regions):
        seqlet_regions = []
        attr_region_pos = chunk_ds["seqlet"].values.copy()
        for seqlet_name in chunk_ds["seqlet"].values:
            seq_id, qstart, qend = map(int, seqlet_name.split("_"))
            chrom, rstart, *_ = regions.iloc[seq_id]
            gstart = rstart + qstart
            gend = rstart + qend
            seqlet_region = f"{chrom}:{gstart}-{gend}"
            seqlet_regions.append(seqlet_region)
        chunk_ds = chunk_ds.assign_coords(seqlet=pd.Index(seqlet_regions))

        attr_region_pos = pd.Series(attr_region_pos, index=chunk_ds.get_index("seqlet"))
        chunk_ds["attr_region"] = attr_region_pos
        return chunk_ds

    def _run_chunk(
        self, chunk_start, one_hot, hypothetical_contribs, output_dir, regions
    ):
        output_path = output_dir / f"chunk_{chunk_start}.zarr"
        temp_path = output_dir / f"chunk_{chunk_start}.zarr_tmp"
        if output_path.exists():
            return

        use_seqlets, seqlet_signs = self._extract_seqlets(
            one_hot, hypothetical_contribs
        )
        use_seqlets = self.adjust_seqlets_idx_(use_seqlets, chunk_start)
        if self.verbose:
            print(f"Extracted {len(use_seqlets)} seqlets from chunk {chunk_start}.")

        ds = {}
        ds["seqlets_tomtom"] = self._tomtom(use_seqlets)
        ds["seqlets_score"], ds["seqlets_seq"] = self._seqlet_to_da(use_seqlets)
        ds = xr.Dataset(ds)

        # add sign
        seqlet_signs = pd.Series(seqlet_signs, index=ds.get_index("seqlet"))
        ds["seqlets_sign"] = seqlet_signs

        # add motif annotations
        pwm_annot = pd.Series({m.motif_id: m.name for m in self.motif_db.motifs})
        pwm_annot.index.name = "motif_id"
        ds["motif_name"] = pwm_annot

        if regions is not None:
            # Annotate seqlets with genomic regions
            ds = self._annotate_regions(ds, regions)

        # save temporarily
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds.chunk({"seqlet": self.chunk_size * 3}).to_zarr(temp_path, mode="w")
        temp_path.rename(output_path)
        return ds

    def run(
        self,
        one_hot: str | np.ndarray,
        hypothetical_contribs: str | np.ndarray,
        output_dir: str,
        regions: pd.DataFrame | None = None,
    ):
        """
        Run the entire Modisco Tomtom pipeline.

        First using modisco extract seqlets from the dna and attribution scores,
        then perform Tomtom motif comparison, and finally save the results to a Zarr file.

        one_hot and hypothetical_contribs can be either paths to .npz files or numpy arrays.
        Shape of both should be (n, seq_len, 4), where n is the number of examples,
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        success_flag = output_dir / ".success"
        if success_flag.exists():
            if self.verbose:
                print(f"Output directory {output_dir} already processed. Skipping.")
            return

        one_hot, hypothetical_contribs, regions = self._load_data(
            one_hot, hypothetical_contribs, regions
        )
        if self.verbose:
            print(f"Loaded data with shape: {one_hot.shape} (n_regions, seq_len, 4)")

        n = one_hot.shape[0]
        n_chunks = n // self.chunk_size + 1
        chunk_size = n // n_chunks + 1
        for cid, chunk_start in enumerate(range(0, n, chunk_size)):
            if self.verbose:
                print(f"Processing chunk {cid + 1}/{n_chunks}")
            chunk_end = min(chunk_start + chunk_size, n)
            one_hot_chunk = one_hot[chunk_start:chunk_end]
            hypothetical_contribs_chunk = hypothetical_contribs[chunk_start:chunk_end]

            # Run the Tomtom comparison for this chunk
            self._run_chunk(
                chunk_start=chunk_start,
                one_hot=one_hot_chunk,
                hypothetical_contribs=hypothetical_contribs_chunk,
                output_dir=output_dir,
                regions=regions,
            )

        # Create a success flag file
        success_flag.touch()
        return
