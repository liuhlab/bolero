import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import ray
import xarray as xr
from memelite import tomtom
from modiscolite.core import Seqlet, TrackSet
from modiscolite.extract_seqlets import extract_seqlets
from modiscolite.io import save_hdf5
from modiscolite.tfmodisco import seqlets_to_patterns

from bolero.pp.genome import Genome


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
        self.adjust_seqlets_idx_(use_seqlets, chunk_start)
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

    for cluster, seqlets in top_motif_cluster.groupby(top_motif_cluster):
        seqlets = seqlets.index
        seq = seq_da.sel(seqlet=seqlets).values
        attr = score_da.sel(seqlet=seqlets).values
        attr_regions = attr_region.sel(seqlet=seqlets).values
        signs = seqlet_sign.sel(seqlet=seqlets).values
        for sign in [-1, 1]:
            sign_sel = signs == sign
            _seq = seq[sign_sel]
            _attr = attr[sign_sel] * sign
            _attr_regions = attr_regions[sign_sel]
            np.savez_compressed(
                f"{output_dir}/{cluster}/{save_name}_{sign}",
                seq=_seq,
                attr=_attr,
                attr_region=_attr_regions,
                attr_zarr_dir=np.array([str(attr_zarr_dir)]),
            )
    flag_path.touch()
    return


class _Seqlet(Seqlet):
    def __init__(self, name, sequence, contrib_scores, hypothetical_contribs):
        self.sequence = sequence
        self.contrib_scores = contrib_scores
        self.hypothetical_contribs = hypothetical_contribs

        example_idx, start, end = map(int, name.split("_"))
        self.example_idx = example_idx
        self.start = start
        self.end = end
        self.is_revcomp = False


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
        _ = self.genome.genome_one_hot  # trigger genome loading

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

        patterns = seqlets_to_patterns(
            seqlets=all_seqlets,
            track_set=track_set,
            track_signs=sign,
            **self.modisco_kwargs,
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
