import pathlib
import time
import warnings
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
import ray
import xarray as xr
from memelite import tomtom as _tomtom
from modiscolite.core import Seqlet as _Seqlet
from modiscolite.core import TrackSet as _TrackSet
from modiscolite.extract_seqlets import extract_seqlets
from modiscolite.io import save_hdf5
from modiscolite.tfmodisco import seqlets_to_patterns

from bolero.pp.genome import Genome

from .modisco import ModiscoHDF

# These motifs contain base pairs with low IC, which induce large number of
# FP hits in TOMTOM with very low p-value but visually unmatched hits.
# If these motifs are included, they will dominate most of top 1st hit.
# There is an issue about this: https://github.com/jmschrei/tangermeme/issues/28
MOTIF_BLACKLIST = ["MA1929.2", "MA1930.2", "MA1978.2"]


class Seqlet(_Seqlet):
    def __init__(self, name, sequence, contrib_scores, hypothetical_contribs):
        assert sequence.shape[1] == 4, "sequence dim 1 must be size 4"
        assert contrib_scores.shape[1] == 4, "contrib_scores dim 1 must be size 4"
        assert (
            hypothetical_contribs.shape[1] == 4
        ), "hypothetical_contribs dim 1 must be size 4"

        self.name = name
        self.sequence = sequence
        self.contrib_scores = contrib_scores
        self.hypothetical_contribs = hypothetical_contribs

        example_idx, start, end = map(int, name.split("_"))
        self.example_idx = example_idx
        self.start = start
        self.end = end
        self.is_revcomp = False


class TrackSet(_TrackSet):
    def __init__(self, one_hot, contrib_scores, hypothetical_contribs):
        assert one_hot.shape[2] == 4, "one_hot dim 2 must be size 4"
        assert contrib_scores.shape[2] == 4, "contrib_scores dim 2 must be size 4"
        assert (
            hypothetical_contribs.shape[2] == 4
        ), "hypothetical_contribs dim 2 must be size 4"

        super().__init__(one_hot, contrib_scores, hypothetical_contribs)


def _read_npz(npz_path: str | np.ndarray) -> np.ndarray:
    """Read a .npz file and return the data as a dictionary."""
    if isinstance(npz_path, np.ndarray):
        return npz_path
    with np.load(npz_path) as data:
        arr = data["arr_0"].transpose(0, 2, 1)
    return arr


def tomtom(scores, pwms, tomtom_n_nearest=10, n_jobs=-1, score_names=None):
    """Perform Tomtom motif comparison."""
    for _s in scores:
        assert _s.ndim == 2, f"scores must be 2D, but got {_s.ndim}D"
        assert (
            _s.shape[0] == 4
        ), f"scores shape must be (4, seq_len), but got {_s.shape}"
    for _p in pwms:
        assert _p.ndim == 2, f"pwms must be 2D, but got {_p.ndim}D"
        assert _p.shape[0] == 4, f"pwms shape must be (4, seq_len), but got {_p.shape}"

    results = _tomtom(Qs=scores, Ts=pwms, n_nearest=tomtom_n_nearest, n_jobs=n_jobs)

    value_type = [
        "p_values",
        "scores",
        "offsets",
        "overlaps",
        "strands",
        "idxs",
    ]
    coords = {
        "value_type": value_type,
    }
    if score_names is not None:
        coords["seqlet"] = score_names
    results = xr.DataArray(
        results.transpose(1, 2, 0).astype("float32"),
        dims=["seqlet", "motif_rank", "value_type"],
        coords=coords,
    )
    return results


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
        pseudobulk_ids: pd.Series | np.ndarray | None = None,
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
        if pseudobulk_ids is not None:
            assert len(pseudobulk_ids) == one_hot.shape[0], (
                f"Number of pseudobulk_ids ({len(pseudobulk_ids)}) must match "
                f"the number of examples ({one_hot.shape[0]})."
            )
            if not isinstance(pseudobulk_ids, pd.Series):
                pseudobulk_ids = pd.Series(pseudobulk_ids)
        return one_hot, hypothetical_contribs, regions, pseudobulk_ids

    def _extract_seqlets(self, one_hot: str, hypothetical_contribs: str):
        contrib_scores = np.multiply(one_hot, hypothetical_contribs)

        track_set = TrackSet(
            one_hot=one_hot,  # shape (n, seq_len, 4)
            contrib_scores=contrib_scores,  # shape (n, seq_len, 4)
            hypothetical_contribs=hypothetical_contribs,  # shape (n, seq_len, 4)
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

    def _get_pwms_and_motif_infos(
        self, blacklist: list[str] = MOTIF_BLACKLIST
    ) -> tuple[list[np.ndarray], list[str]]:
        high_ic_cutoff = 0.1
        step = 3

        # Cut PWMs at >= 3 consecutive low IC bases, then select the longest continuous region.
        # This is to prevent tomtom getting false positive hits when using motifs with low IC regions.
        pwms = []
        motif_infos = OrderedDict()
        for motif in self.motif_db.motifs:
            if motif.motif_id in blacklist:
                continue
            ic_flag = motif.pwm_info_content() > high_ic_cutoff
            g = nx.Graph()
            edges = set()
            for _step in range(1, step + 1):
                for start in range(0, ic_flag.size - _step):
                    start_flag = ic_flag[start]
                    end_flag = ic_flag[start + _step]
                    if start_flag and end_flag:
                        edges.add((start, start + _step))
            edges = list(edges)
            g.add_edges_from(edges)
            comps = []
            comp_sizes = []
            for comp in nx.components.connected_components(g):
                comps.append(comp)
                comp_sizes.append(len(comp))
            locs = sorted(comps[np.argmax(comp_sizes)])
            use_pos = ic_flag[
                max(min(locs) - 1, 0) : min(max(locs) + 1, ic_flag.size)
            ].index
            pwms.append(motif.pwm.loc[use_pos].values.T.astype("float32"))
            motif_infos[motif.motif_id] = motif.motif_name
        return pwms, motif_infos

    def tomtom(
        self, seqlets: list[Seqlet], blacklist: list[str] = MOTIF_BLACKLIST
    ) -> xr.DataArray:
        """Perform Tomtom motif comparison on seqlets."""
        pwms, motif_infos = self._get_pwms_and_motif_infos(blacklist)

        scores = [seqlet.contrib_scores.T.astype("float32") for seqlet in seqlets]
        score_names = [seqlet.string for seqlet in seqlets]

        results = tomtom(
            scores=scores,
            pwms=pwms,
            tomtom_n_nearest=self.tomtom_n_nearest,
            n_jobs=self.n_jobs,
            score_names=score_names,
        )
        results.attrs["motif_info"] = motif_infos
        return results

    def simple_tomtom(
        self,
        attr_scores: np.ndarray,
        score_names: list[str] = None,
        blacklist: list[str] = MOTIF_BLACKLIST,
    ) -> xr.DataArray:
        """
        Perform Tomtom motif comparison on attribute scores.

        Parameters
        ----------
        attr_scores : np.ndarray
            Attribute scores. Shape (n, 4, seq_len).
        score_names : list[str], optional
            Score names. Default is None.

        Returns
        -------
        xr.DataArray
            Tomtom results. Shape (n, motif_rank, value_type).
            value_type: ["p_values", "scores", "offsets", "overlaps", "strands", "idxs"].
        """
        pwms, motif_infos = self._get_pwms_and_motif_infos(blacklist)

        if isinstance(attr_scores, np.ndarray):
            # assum attr_scores is np.ndarray, shape (n_regions, 4, seq_len)
            scores = list(attr_scores.astype("float32"))
        else:
            scores = attr_scores

        results = tomtom(
            scores=scores,
            pwms=pwms,
            tomtom_n_nearest=self.tomtom_n_nearest,
            n_jobs=self.n_jobs,
            score_names=score_names,
        )
        results.attrs["motif_info"] = motif_infos
        return results

    def annotate_seqlet_ds(
        self,
        seqlet_ds: xr.Dataset,
        seqlet_info: pd.DataFrame,
        flank_seqlet_size: int = 2,
    ) -> pd.DataFrame:
        """
        Annotate a tengermeme extracted seqlet dataset with motif information.

        Parameters
        ----------
        seqlet_ds : xr.Dataset
            Seqlet dataset.
        seqlet_info : pd.DataFrame
            Seqlet information.
        flank_seqlet_size : int, optional
            Flank size of the seqlet cluster. Default is 2.
        """
        # use the actual seqlet region to do tomtom
        attr_scores = []
        for (seqlet_start, seqlet_end, flank_start), seqlet_attr in zip(
            seqlet_info[["start", "end", "flank_start"]].values,
            seqlet_ds["seqlets_attr"].values.astype("float32"),
        ):
            rel_start = seqlet_start - flank_seqlet_size - flank_start
            rel_start = max(rel_start, 0)
            rel_end = seqlet_end + flank_seqlet_size - flank_start
            rel_end = min(rel_end, seqlet_attr.shape[1])
            use_attr = seqlet_attr[:, rel_start:rel_end]
            attr_scores.append(use_attr)

        score_names = seqlet_ds.get_index("seqlet")
        result = self.simple_tomtom(
            attr_scores=attr_scores,
            score_names=score_names,
        )
        motif_top_hits = result.sel(motif_rank=0).to_pandas()
        motifs = pd.Series(result.attrs["motif_info"])
        motif_top_hits["motif_id"] = (
            motif_top_hits["idxs"].astype(int).map(lambda i: motifs.index[i])
        )
        motif_top_hits["motif_name"] = (
            motif_top_hits["idxs"].astype(int).map(lambda i: motifs.values[i])
        )
        return motif_top_hits

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

    def _annotate_regions(self, chunk_ds, regions, pseudobulk_ids=None):
        seqlet_regions = []
        attr_region_pos = chunk_ds["seqlet"].values.copy()
        pseudobulk_list = []  # Add this
        for seqlet_name in chunk_ds["seqlet"].values:
            seq_id, qstart, qend = map(int, seqlet_name.split("_"))
            chrom, rstart, *_ = regions.iloc[seq_id]
            gstart = rstart + qstart
            gend = rstart + qend
            seqlet_region = f"{chrom}:{gstart}-{gend}"
            seqlet_regions.append(seqlet_region)
            # Add pseudobulk ID
            if pseudobulk_ids is not None:
                pseudobulk_list.append(pseudobulk_ids[seq_id])
        chunk_ds = chunk_ds.assign_coords(seqlet=pd.Index(seqlet_regions))

        attr_region_pos = pd.Series(attr_region_pos, index=chunk_ds.get_index("seqlet"))
        chunk_ds["attr_region"] = attr_region_pos

        # Add pseudobulk IDs
        if pseudobulk_ids is not None:
            chunk_ds["pseudobulk_id"] = pd.Series(
                pseudobulk_list, index=chunk_ds.get_index("seqlet")
            )

        return chunk_ds

    def _run_chunk(
        self,
        chunk_start,
        one_hot,
        hypothetical_contribs,
        output_dir,
        regions,
        pseudobulk_ids=None,
    ):
        output_path = output_dir / f"chunk_{chunk_start}.zarr"
        temp_path = output_dir / f"chunk_{chunk_start}.zarr_tmp"
        if output_path.exists():
            return

        use_seqlets, seqlet_signs = self._extract_seqlets(
            one_hot, hypothetical_contribs
        )
        self.adjust_seqlets_idx_(use_seqlets, chunk_start)
        if self.verbose:
            print(f"Extracted {len(use_seqlets)} seqlets from chunk {chunk_start}.")

        ds = {}
        ds["seqlets_tomtom"] = self.tomtom(use_seqlets)
        ds["seqlets_score"], ds["seqlets_seq"] = self._seqlet_to_da(use_seqlets)
        ds = xr.Dataset(ds)

        # add sign
        seqlet_signs = pd.Series(seqlet_signs, index=ds.get_index("seqlet"))
        ds["seqlets_sign"] = seqlet_signs

        # add motif annotations
        motif_id_to_name = ds["seqlets_tomtom"].attrs["motif_info"]
        pwm_annot = pd.Series(motif_id_to_name)
        pwm_annot.index.name = "motif_id"
        ds["motif_name"] = pwm_annot

        if regions is not None:
            # Annotate seqlets with genomic regions
            chunk_pseudobulk = None
            if pseudobulk_ids is not None:
                chunk_end = chunk_start + one_hot.shape[0]
                chunk_pseudobulk = pseudobulk_ids.iloc[chunk_start:chunk_end]
            ds = self._annotate_regions(ds, regions, chunk_pseudobulk)
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
        pseudobulk_ids: pd.Series | np.ndarray | None = None,
    ):
        """
        Run the entire Modisco Tomtom pipeline.

        First using modisco extract seqlets from the dna and attribution scores,
        then perform Tomtom motif comparison, and finally save the results to a Zarr file.

        Zarr file structure:
        Dimensions:
        - seqlet, motif_id, base, position, value_type, motif_rank
        Coordinates:
        - seqlet: Seqlet name. If regions are provided, the seqlet name will be the genomic region.
        - motif_id: Motif ID.
        - base: DNA base A, C, G, T.
        - value_type: Type of value from Tomtom ["p_values", "scores", "offsets", "overlaps", "strands", "idxs"].
        Data variables:
        - attr_region: relative attr region coords of each seqlets: {input_region_idx}_{rel_start}_{rel_end} (seqlet,)
        - motif_name: motif_name (motif_id,)
        - seqlets_score: attr score for each seqlet (seqlet, base, position)
        - seqlets_seq: DNA one hot for each seqlet (seqlet, base, position)
        - seqlets_sign: sign of the seqlet (seqlet,)
        - seqlets_tomtom: Tomtom results for each seqlet (seqlet, motif_rank, value_type)

        Parameters
        ----------
        one_hot : str | np.ndarray
            Path to .npz file or numpy array of one-hot encoded DNA sequences. Shape (n, seq_len, 4).
        hypothetical_contribs : str | np.ndarray
            Path to .npz file or numpy array of hypothetical contribution scores. Shape (n, seq_len, 4).
        output_dir : str
            Path to the output directory.
        regions : pd.DataFrame | None, optional
            DataFrame of regions. If provided, the seqlet index will be annotated with the regions.
            First three columns should be "chrom", "start", "end".

        Returns
        -------
        None
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        success_flag = output_dir / ".success"
        if success_flag.exists():
            if self.verbose:
                print(f"Output directory {output_dir} already processed. Skipping.")
            return

        one_hot, hypothetical_contribs, regions, pseudobulk_ids = self._load_data(
            one_hot, hypothetical_contribs, regions, pseudobulk_ids
        )
        if self.verbose:
            print(f"Loaded data with shape: {one_hot.shape} (n_regions, seq_len, 4)")

        if regions is not None:
            assert isinstance(regions, pd.DataFrame), "regions must be a DataFrame"
            assert regions.shape[0] == len(
                one_hot
            ), "regions must have the same number of rows as one_hot"
        if pseudobulk_ids is not None:
            pseudobulk_ids = list(pseudobulk_ids)

        n = len(one_hot)
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
                pseudobulk_ids=pseudobulk_ids,
            )

        # Create a success flag file
        success_flag.touch()
        return


@ray.remote
def _dump_seqlets_per_cluster(
    tomtom_dir, attr_zarr_dir, save_name, output_dir, motif_cluster
):
    output_dir = pathlib.Path(output_dir)
    flag_path = pathlib.Path(f"{output_dir}/_finished/{save_name}")
    if flag_path.exists():
        return

    tomtom_dir = pathlib.Path(tomtom_dir)
    attr_zarr_dir = pathlib.Path(attr_zarr_dir)

    zarr_paths = list(tomtom_dir.glob("chunk_*.zarr"))
    dataset = xr.open_mfdataset(
        zarr_paths,
        concat_dim="seqlet",
        combine="nested",
        engine="zarr",
        data_vars="minimal",
    )

    # assign seqlet to cluster based on top TOMTOM hit
    top_motif = (
        dataset["seqlets_tomtom"]
        .sel(value_type="idxs", motif_rank=0)
        .to_pandas()
        .astype(int)
    )
    motif_ids = dataset.get_index("motif_id").tolist()
    top_motif_cluster = top_motif.map(lambda idx: motif_ids[idx]).map(motif_cluster)

    # group seqlets by motif cluster and save to each dir
    dataset = dataset.drop_duplicates(dim="seqlet")

    score_da = dataset["seqlets_score"].load()
    seq_da = dataset["seqlets_seq"].load()
    attr_region = dataset["attr_region"].load()
    seqlet_sign = dataset["seqlets_sign"].load()
    pseudobulk_id = dataset.get("pseudobulk_id", None)
    if pseudobulk_id is not None:
        pseudobulk_id = pseudobulk_id.load()

    for cluster, seqlets in top_motif_cluster.groupby(top_motif_cluster):
        seqlets = seqlets.index
        seq = seq_da.sel(seqlet=seqlets).values
        attr = score_da.sel(seqlet=seqlets).values
        attr_regions = attr_region.sel(seqlet=seqlets).values
        signs = seqlet_sign.sel(seqlet=seqlets).values
        if pseudobulk_id is not None:
            pbulk_ids = pseudobulk_id.sel(seqlet=seqlets).values
        for sign in [-1, 1]:
            sign_sel = signs == sign
            _seq = seq[sign_sel]
            _attr = attr[sign_sel] * sign
            _attr_regions = attr_regions[sign_sel]
            save_dict = {
                "seq": _seq,
                "attr": _attr,
                "attr_region": _attr_regions,
                "attr_zarr_dir": np.array([str(attr_zarr_dir)]),
            }

            if pseudobulk_id is not None:
                save_dict["pseudobulk_id"] = pbulk_ids[sign_sel]

            np.savez_compressed(
                f"{output_dir}/{cluster}/{save_name}_{sign}", **save_dict
            )
    flag_path.touch()
    return


@ray.remote
def _sample_seqlets_from_npz(npz_path, sample_prob, sign):
    seqlets = []
    seqs = np.load(npz_path)["seq"]
    seqs = seqs.transpose(0, 2, 1)  # shape (n, seq_len, 4)
    attrs = np.load(npz_path)["attr"] * sign
    attrs = attrs.astype(np.float32).transpose(0, 2, 1)  # shape (n, seq_len, 4)
    names = np.load(npz_path, allow_pickle=True)["attr_region"]
    use_rows = (
        pd.Series(names).sample(frac=sample_prob, replace=True).sort_index().index
    )

    for row_idx in use_rows:
        seq = seqs[row_idx]
        attr = attrs[row_idx]
        name = names[row_idx]
        seqlet = _Seqlet(name, seq, attr, attr)
        seqlets.append(seqlet)

    attr_zarr_dir = np.load(npz_path, allow_pickle=True)["attr_zarr_dir"][0]
    attr_dataset = xr.open_zarr(attr_zarr_dir)

    use_attr_rows = pd.Index({int(n.split("_")[0]) for n in names[use_rows]})
    idx_remap = {old_row: idx for idx, old_row in enumerate(use_attr_rows)}
    for seqlet in seqlets:
        seqlet.example_idx = idx_remap[seqlet.example_idx]

    use_attr_dataset = attr_dataset.isel(region=use_attr_rows)
    use_attr = use_attr_dataset["__dna__:attr"].values
    use_attr_regions = use_attr_dataset.get_index("region")
    return seqlets, use_attr, use_attr_regions


class TFModiscoOnMotifCluster:
    @classmethod
    def dump_seqlets_by_motif_cluster(
        cls,
        attr_prefix_dict: dict[str, str],
        output_dir: str,
        motif_cluster: pd.Series,
    ):
        """
        Dump seqlets by motif cluster.

        This function will read the Tomtom results, assign seqlets to motif clusters,
        and save the seqlets and their attributes to separate directories for each cluster.

        Parameters
        ----------
        attr_prefix_dict : dict[str, str]
            Dictionary mapping dataset keys to path prefix of the attribute zarr files.
            tomtom_dir = "{dir_prefix}.seqlet_tomtom.zarr"
            attr_zarr_dir = "{dir_prefix}.attr.zarr"
        output_dir : str
            Directory where the seqlets will be saved.
        motif_cluster : pd.Series
            Series mapping motif IDs to their respective clusters.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for c in motif_cluster.unique():
            pathlib.Path(f"{output_dir}/{c}").mkdir(exist_ok=True)
        flag_dir = f"{output_dir}/_finished"
        flag_dir = pathlib.Path(flag_dir)
        flag_dir.mkdir(exist_ok=True)

        tasks = []
        for save_name, dir_prefix in attr_prefix_dict.items():
            tomtom_dir = f"{dir_prefix}.seqlet_tomtom.zarr"
            attr_zarr_dir = f"{dir_prefix}.attr.zarr"
            t = _dump_seqlets_per_cluster.remote(
                tomtom_dir, attr_zarr_dir, save_name, output_dir, motif_cluster
            )
            tasks.append(t)
        _ = ray.get(tasks)
        return

    def __init__(
        self,
        genome,
        seqlet_dir: str,
        max_seqlets: int = 10000,
        pos_only: bool = True,
        n_leiden_runs: int = 24,
        verbose: bool = False,
    ):
        self.seqlet_dir = pathlib.Path(seqlet_dir)
        self.max_seqlets = max_seqlets
        self.genome = Genome(genome) if isinstance(genome, str) else genome
        self.motif_db = self.genome.get_motif_db()

        self.cluster_dir_dict = {
            p.name: p for p in self.seqlet_dir.glob("MotifCluster*")
        }
        self.pos_only = pos_only

        self.verbose = verbose
        self.modisco_kwargs = {
            "min_overlap_while_sliding": 0.7,
            "nearest_neighbors_to_compute": 500,
            "affmat_correlation_threshold": 0.15,
            "tsne_perplexity": 10.0,
            "n_leiden_iterations": 2,
            "n_leiden_runs": n_leiden_runs,
            "frac_support_to_trim_to": 0.2,
            "min_num_to_trim_to": 30,
            "trim_to_window_size": 30,
            "initial_flank_to_add": 10,
            "final_flank_to_add": 0,
            "prob_and_pertrack_sim_merge_thresholds": [
                (0.8, 0.8),
                (0.5, 0.85),
                (0.2, 0.9),
            ],
            "prob_and_pertrack_sim_dealbreaker_thresholds": [
                (0.4, 0.75),
                (0.2, 0.8),
                (0.1, 0.85),
                (0.0, 0.9),
            ],
            "subcluster_perplexity": 50,
            "merging_max_seqlets_subsample": 200,
            "final_min_cluster_size": 20,
            "min_ic_in_window": 0.6,
            "min_ic_windowsize": 6,
            "ppm_pseudocount": 0.001,
            "skip_subpattern": True,
            "verbose": verbose,
        }
        return

    def _get_seqlet_sample_prob(self, motif_cluster, sign):
        cluster_dir = self.cluster_dir_dict[motif_cluster]
        npz_paths = list(cluster_dir.glob(f"*_{sign}.npz"))
        all_regions = []
        for npz_path in npz_paths:
            attr_regions = np.load(npz_path, allow_pickle=True)["attr_region"]
            all_regions.extend(attr_regions)
        total_seqlets = len(all_regions)
        sample_prob = min(self.max_seqlets / total_seqlets, 1)
        return sample_prob

    def _prepare_cluster(self, cluster_name: str, sign):
        sample_prob = self._get_seqlet_sample_prob(cluster_name, sign)
        cluster_dir = self.cluster_dir_dict[cluster_name]
        npz_paths = list(cluster_dir.glob(f"*_{sign}.npz"))

        tasks = []
        for npz_path in npz_paths:
            task = _sample_seqlets_from_npz.remote(npz_path, sample_prob, sign)
            tasks.append(task)

        all_seqlets = []
        all_attr = []
        all_dna = []
        cur_idx = 0
        for seqlets, use_attr, use_attr_regions in ray.get(tasks):
            all_attr.append(use_attr)
            use_dna_onehot = self.genome.get_regions_one_hot(use_attr_regions)
            all_dna.append(use_dna_onehot)

            # update seqlets idx
            for seqlet in seqlets:
                seqlet.example_idx += cur_idx
            cur_idx += use_attr.shape[0]

            all_seqlets.extend(seqlets)

        all_attr = np.concatenate(all_attr).astype(np.float32).transpose(0, 2, 1)
        all_dna = np.concatenate(all_dna).transpose(0, 2, 1)
        all_dna = all_dna.transpose(0, 2, 1)
        track_set = TrackSet(
            one_hot=all_dna, contrib_scores=all_attr, hypothetical_contribs=all_attr
        )
        return track_set, all_seqlets

    def _single_cluster_patterns(self, cluster_name: str, sign: int):
        track_set, all_seqlets = self._prepare_cluster(cluster_name, sign)
        if len(all_seqlets) == 0:
            return []

        patterns = self.run_seqlets_to_patterns(all_seqlets, track_set)
        return patterns

    def run_seqlets_to_patterns(
        self, seqlets: list[Seqlet], track_set: TrackSet, **kwargs
    ):
        """
        Run seqlets to patterns step using modisco.
        """
        patterns = seqlets_to_patterns(
            seqlets=seqlets,
            track_set=track_set,
            **self.modisco_kwargs,
            **kwargs,
        )
        return patterns

    def run(self, cluster_sel=None):
        """Get patterns for all motif clusters."""
        print(f"Running TFModisco on {len(self.cluster_dir_dict)} motif clusters.")
        print(f"Sampling {self.max_seqlets} per motif cluster")
        for cluster_name, cluster_dir in self.cluster_dir_dict.items():
            if cluster_sel is not None and cluster_name not in cluster_sel:
                continue

            for sign in [-1, 1]:
                if self.pos_only and sign == -1:
                    continue

            sign_name = "pos" if sign == 1 else "neg"
            save_path = f"{cluster_dir}/patterns_{sign_name}.h5"
            if pathlib.Path(save_path).exists():
                continue

            print(f"Processing cluster {cluster_name} with sign {sign}")
            time_start = time.time()
            patterns = self._single_cluster_patterns(cluster_name, sign)
            save_hdf5(
                filename=save_path,
                pos_patterns=patterns,
                neg_patterns=None,
                window_size=400,
            )
            time_end = time.time()
            used_min = int((time_end - time_start) / 60)
            print(f"Found {len(patterns)} patterns in {used_min} minutes")
        return

    def annotate_and_aggregate(self, p_value_cutoff=1e-4, corrcoef_cutoff=0.5):
        """
        Annotate and aggregate patterns from all motif clusters.
        This will run TOMTOM on the patterns and return a DataFrame with annotations.
        """
        aggregator = AggregateModiscoResults(
            modisco_dir=self.seqlet_dir,
            motif_db=self.motif_db,
            n_jobs=16,
            p_value_cutoff=p_value_cutoff,
            corrcoef_cutoff=corrcoef_cutoff,
        )
        pattern_annot = aggregator.run()
        print(f"Annotated {len(pattern_annot)} patterns after filtering.")
        return pattern_annot


class AggregateModiscoResults:
    def __init__(
        self,
        modisco_dir: str,
        motif_db,
        n_jobs=16,
        p_value_cutoff=1e-4,
        corrcoef_cutoff=0.5,
    ):
        self.modisco_dir = pathlib.Path(modisco_dir)
        self.motif_db = motif_db
        self.n_jobs = n_jobs
        self.p_value_cutoff = p_value_cutoff
        self.corrcoef_cutoff = corrcoef_cutoff

        self.patterns = self._aggregate_patterns()

    def _aggregate_patterns(self):
        patterns = []
        h5_path_dict = {
            p.parent.name: p for p in self.modisco_dir.glob("MotifCluster*/pattern*.h5")
        }
        for key, path in h5_path_dict.items():
            modisco_hdf = ModiscoHDF(path)
            for pattern in modisco_hdf.pos_patterns:
                pattern.name = f"{key}.pos.{pattern.name}"
                pattern.sign = 1
                patterns.append(pattern)
            for pattern in modisco_hdf.neg_patterns:
                pattern.name = f"{key}.neg.{pattern.name}"
                pattern.sign = -1
                patterns.append(pattern)
        return patterns

    def annotate_patterns(self):
        """Annotate patterns using TOMTOM."""
        patterns = self.patterns
        use_motifs = [
            m for m in self.motif_db.motifs if m.motif_id not in MOTIF_BLACKLIST
        ]
        pwms = [m.pwm.values.T.astype("float32") for m in use_motifs]
        seqlets = [p.to_seqlet() for p in patterns]
        pattern_names = [p.name for p in patterns]
        scores = [seqlet.contrib_scores.T.astype("float32") for seqlet in seqlets]
        scores = [s if s.sum() > 0 else s * -1 for s in scores]

        results = tomtom(
            scores=scores,
            pwms=pwms,
            tomtom_n_nearest=10,
            n_jobs=self.n_jobs,
            score_names=pattern_names,
        )

        pattern_to_tf = (
            results.sel(value_type="idxs", motif_rank=0).to_pandas().astype(int)
        )
        pattern_annot = pd.DataFrame(
            {
                "motif_id": pattern_to_tf.map(lambda idx: use_motifs[idx].motif_id),
                "motif_name": pattern_to_tf.map(lambda idx: use_motifs[idx].motif_name),
                "p_values": results.sel(
                    value_type="p_values", motif_rank=0
                ).to_pandas(),
                "scores": results.sel(value_type="scores", motif_rank=0).to_pandas(),
                "offsets": results.sel(value_type="offsets", motif_rank=0)
                .to_pandas()
                .astype("int16"),
                "overlaps": results.sel(value_type="overlaps", motif_rank=0)
                .to_pandas()
                .astype("int16"),
                "strands": results.sel(value_type="strands", motif_rank=0)
                .to_pandas()
                .astype("int16"),
            }
        )
        pattern_annot["motif_cluster"] = pattern_annot["motif_id"].map(
            self.motif_db.motif_cluster
        )
        pattern_annot["p_values"] = pattern_annot["p_values"].clip(lower=0)
        pattern_annot["-lgp"] = -np.log10(pattern_annot["p_values"] + 1e-10)
        pattern_annot["origin_cluster"] = pattern_annot.index.map(
            lambda i: i.split(".")[0]
        )
        pattern_annot = pattern_annot[
            (pattern_annot["offsets"] < 0)
            & (pattern_annot["p_values"] < self.p_value_cutoff)
            & (pattern_annot["motif_cluster"] == pattern_annot["origin_cluster"])
        ].copy()
        del pattern_annot["origin_cluster"]

        pattern_annot = self.add_pattern_corr(pattern_annot)
        return pattern_annot

    def add_pattern_corr(self, pattern_annot):
        """
        For each TOMTOM hits, further calculate correlation with motif PWM.
        tomtom implementation doesn't support corr, so calculate it here as a post filtering step.
        """
        pattern_dict = {p.name: p for p in self.patterns}
        corr_col = {}
        for pattern_name, info in pattern_annot.iterrows():
            pattern = pattern_dict[pattern_name]
            ms = -info["offsets"]
            motif = self.motif_db.motif_id_dict[info["motif_id"]]
            motif_pwm = motif.pwm.values
            if info["strands"] == 1:
                motif_pwm = motif_pwm[::-1, ::-1]
            motif_len = motif_pwm.shape[0]
            me = ms + motif_len
            pattern_hit_scores = pattern.contrib_scores[ms:me]
            hit_len = pattern_hit_scores.shape[0]

            # in case hit is partial, pad even base to the hit to match motif size
            if hit_len < motif_len:
                to_pad = np.zeros((motif_len - hit_len, 4))
                pattern_hit_scores = np.concatenate([pattern_hit_scores, to_pad])
            if pattern_hit_scores.sum() < 0:
                pattern_hit_scores *= -1
            corr = np.corrcoef(motif_pwm.ravel(), pattern_hit_scores.ravel())[0, 1]
            corr_col[pattern_name] = corr
        pattern_annot["corrcoef"] = pd.Series(corr_col)
        pattern_annot = pattern_annot[
            pattern_annot["corrcoef"] > self.corrcoef_cutoff
        ].copy()
        return pattern_annot

    def run(self):
        """
        Run the aggregation and save the results to a file.
        """
        pattern_annot = self.annotate_patterns()
        annot_path = self.modisco_dir / "all_patterns_annot.filtered.feather"
        pattern_annot.to_feather(annot_path)

        remain_patterns = [p for p in self.patterns if p.name in pattern_annot.index]
        save_path = self.modisco_dir / "all_patterns.filtered.h5"
        save_hdf5(
            filename=save_path,
            pos_patterns=[p for p in remain_patterns if p.sign == 1],
            neg_patterns=[p for p in remain_patterns if p.sign == -1],
            window_size=400,
        )
        return pattern_annot
