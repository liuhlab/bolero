import pathlib
import tempfile

import joblib
import numpy as np
import pandas as pd
import ray
import snapatac2 as snap
from pyfaidx import Fasta
from scipy.sparse import coo_matrix, csr_matrix, vstack

from bolero import Genome
from bolero.pp.seq import one_hot_encoding


@ray.remote
def _dump_grouped_csr(
    sample, adata_path, key, pseudobulk_order, groupping, output_path
):
    """
    This function sum up single cell csr_matrix into pseudobulk csr_matrix
    It is a worker function that only handles one chromosome (key) in one adata_path
    """
    adata = snap.read(adata_path, backed="r")
    adata_bcs = pd.Index(adata.obs_names)

    cell_row_to_bulk_row = np.zeros(adata_bcs.size, dtype="int32") - 1
    for bulk_row_id, group in enumerate(pseudobulk_order):
        # we intersect adata obs_names with each group's bcs,
        # this means cells are hard assigned to groups, one cell only occurs in one group
        bool_row_sel = adata_bcs.isin(groupping.get((sample, group), []))
        if bool_row_sel.sum() == 0:
            continue
        cell_row_to_bulk_row[bool_row_sel] = bulk_row_id
    merge_plan = {
        cell_row: bulk_row
        for cell_row, bulk_row in enumerate(cell_row_to_bulk_row)
        if bulk_row != -1
    }
    if len(merge_plan) == 0:
        raise ValueError(f"Merge plan is empty for adata path: {adata_path}")

    merger = CSRRowMerge(
        merge_plan,
        n_input=adata_bcs.size,
        n_output=len(pseudobulk_order),
        dtype="float32",
    )
    group_mat = merger(adata.obsm[key].tocsr())
    joblib.dump(group_mat, output_path)
    return


class CSRRowMerge:
    def __init__(self, merge_plan: dict, n_input=None, n_output=None, dtype="float32"):
        """
        merge_plan: dict mapping each input row index to an output row index.
        """
        self.merge_plan = merge_plan
        n_input = max(merge_plan.keys()) + 1 if n_input is None else n_input
        n_output = max(merge_plan.values()) + 1 if n_output is None else n_output
        self.n_input, self.n_output = n_input, n_output

        # check if multimap
        try:
            multimap = isinstance(list(merge_plan.values())[0], list)
        except IndexError as e:
            print(merge_plan)
            raise ValueError(
                "merge_plan is empty, please provide a valid merge plan"
            ) from e
        # P is the auxiliary matrix to merge input rows into output rows
        rows = []
        cols = []
        for row, col in self.merge_plan.items():
            if multimap:
                # if multimap, one cell (row) can be assigned to multiple groups (col)
                for _col in col:
                    rows.append(_col)
                    cols.append(row)
            else:
                rows.append(col)
                cols.append(row)
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.ones_like(rows, dtype=dtype)
        self.P = coo_matrix((data, (rows, cols)), shape=(n_output, n_input)).tocsr()
        self.dtype = dtype

    def __call__(self, A, subset_plan: None | np.ndarray = None):
        """
        Apply the merge plan to the input matrix A.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Input sparse matrix to be merged. shape (n_cells, n_bins)
        subset_plan : list, optional
            If provided, only pseudobulk rows selected by this plan will be merged.
            P will be (n_selected_pseudobulks, n_cells) and only selected pseudobulks will be merged.
            If None, P will be (n_pseudobulks, n_cells) and all pseudobulks will be merged.
        """
        assert A.shape[0] == self.n_input
        if subset_plan is not None:
            P = self.P[subset_plan, :]
        else:
            P = self.P
        return P.dot(A.tocsr()).astype(self.dtype)


class AdataPseudobulkMerger:
    def __init__(
        self,
        pseudobulk_meta,
        adata_path_dict,
        output_dir,
        sparse_format="csc",
        use_chroms=None,
    ):
        """
        Merge single-cell by genome sparse matrix into pseudobulk by genome sparse matrix.
        This is a preparing step for generating parquet dataset with GenomeChunkDatasetGenerator class.

        single-cell matrix stored in snapatac2 adata files provided in adata_path_dict.
        single-cell to pseudobulk mapping stored in pseudobulk_meta.
        Output data are organized by chromosomes and stored in a single directory.

        It took ~30min to merge 173k cell into 1.3k unbalanced pseudobulks, using 60 cpus, memory max < 30GB.
        The last csc step is slow and unparalleled, can parallel if memory is enough.

        Parameters
        ----------
        pseudobulk_meta : str
            Path to the pseudobulk metadata file (feather format).
            It should have cell_id as index, and three columns:
            - sample (match with adata_path_dict)
            - groups (grouping for pseudobulk)
            - bc (barcode for each cell, match with each snap adata file)
        adata_path_dict : dict
            Dictionary mapping sample names to their corresponding snap adata file paths.
        output_dir : str
            Directory to save the merged pseudobulk data. Final file will be:
        sparse_format : str
            The sparse format to save the merged data. Options are 'csr' or 'csc'.
        use_chroms : list, optional
            List of chromosome names to use for merging.
            If None, all chromosomes in snap file will be used.
        """
        self.output_dir = pathlib.Path(output_dir)

        # Prepare merge info
        if not isinstance(pseudobulk_meta, pd.DataFrame):
            pseudobulk_meta = pd.read_feather(pseudobulk_meta)
        self.pseudobulk_meta = pseudobulk_meta
        self.pseudobulk_meta.columns = ["sample", "groups", "bc"]

        n_sample = self.pseudobulk_meta["sample"].nunique()
        n_groups = self.pseudobulk_meta["groups"].nunique()
        n_cells = self.pseudobulk_meta.shape[0]
        print(
            f"pseudobulk meta table: {n_sample} samples, {n_groups} groups, {n_cells} cells"
        )
        self.pseudobulk_order = sorted(self.pseudobulk_meta["groups"].unique())

        self.groupping = {
            (sample, group): df["bc"].tolist()
            for (sample, group), df in self.pseudobulk_meta.groupby(
                ["sample", "groups"], observed=True
            )
        }
        self.adata_path_dict = adata_path_dict
        samples_in_table = self.pseudobulk_meta["sample"].unique()
        assert all(
            k in samples_in_table for k in adata_path_dict.keys()
        ), "not all samples in pseudobulk meta table"

        adata = snap.read(list(adata_path_dict.values())[0], backed="r")
        self.chrom_keys = [k for k in adata.obsm.keys() if k.startswith("insertion_")]
        if use_chroms is not None:
            use_chrom_keys = [f"insertion_{c}" for c in use_chroms]
            self.chrom_keys = [k for k in self.chrom_keys if k in use_chrom_keys]
            print(f"Using chrom_keys: {self.chrom_keys}")
        adata.close()

        self.sparse_format = sparse_format
        assert self.sparse_format in [
            "csr",
            "csc",
        ], f"unsupported sparse format {sparse_format}"
        return

    def _save_metadata(self):
        joblib.dump(
            self.pseudobulk_order, self.output_dir / "pseudobulk_order.list.joblib"
        )
        joblib.dump(self.chrom_keys, self.output_dir / "chrom_keys.list.joblib")
        self.pseudobulk_meta.to_feather(self.output_dir / "pseudobulk_metadata.feather")

    def _extract_group_sum_for_single_adata(self, temp_dir, chrom_key):
        """
        Sum up cells into pseudobulks for a single chrom and adata_path,
        save into group:csr_mat dict for each adata_path.
        """
        fs = []
        for sample, adata_path in self.adata_path_dict.items():
            output_path = temp_dir / f"{sample}_{chrom_key}"
            f = _dump_grouped_csr.remote(
                sample=sample,
                adata_path=adata_path,
                key=chrom_key,
                pseudobulk_order=self.pseudobulk_order,
                groupping=self.groupping,
                output_path=output_path,
            )
            fs.append(f)
        _ = ray.get(fs)
        return

    def _merge_adata_for_single_chrom(self, temp_dir, chrom_key):
        """
        After each adata's pseudobulk-by-chromosome is calculated separately
        we sum up all adata into final pseudobulk-by-chromosome matrix
        """
        print(f"dump {chrom_key} groups merge")
        to_del = []
        group_datas = []
        merge_plan = {}
        row_cum = 0
        for sample in self.adata_path_dict.keys():
            output_path = temp_dir / f"{sample}_{chrom_key}"
            to_del.append(output_path)
            sample_data = joblib.load(output_path)
            group_datas.append(sample_data)
            for row in range(sample_data.shape[0]):
                merge_plan[row + row_cum] = row
            row_cum += sample_data.shape[0]
        merger = CSRRowMerge(
            merge_plan, n_input=row_cum, n_output=len(self.pseudobulk_order)
        )
        group_datas = merger(vstack(group_datas))

        group_datas.data = np.clip(group_datas.data, 0, np.iinfo("uint16").max)
        group_datas = group_datas.astype("uint16")

        if self.sparse_format == "csc":
            group_datas = group_datas.tocsc()

        temp_path = self.output_dir / f"{chrom_key}.{self.sparse_format}.temp.joblib"
        final_path = self.output_dir / f"{chrom_key}.{self.sparse_format}.joblib"
        joblib.dump(group_datas, temp_path)
        temp_path.rename(final_path)

        for path in to_del:
            if pathlib.Path(path).exists():
                pathlib.Path(path).unlink()
        return

    def merge(self):
        """Merge all chroms for all samples and groups."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)

            for chrom_key in self.chrom_keys:
                chrom_path = (
                    self.output_dir / f"{chrom_key}.{self.sparse_format}.joblib"
                )
                if chrom_path.exists():
                    print(f"{chrom_key} already exists, skip")
                    continue

                self._extract_group_sum_for_single_adata(temp_dir, chrom_key)
                self._merge_adata_for_single_chrom(temp_dir, chrom_key)
        return


class PseudobulkAdata:
    def __init__(self, pseudobulk_dir, genome, catch_chroms=False):
        """
        Load the pseudobulk data from the specified directory.

        Parameters
        ----------
        pseudobulk_dir : str
            Directory containing the merged pseudobulk data files.

        """
        self.pseudobulk_dir = pathlib.Path(pseudobulk_dir)
        self.pseudobulk_order = joblib.load(
            self.pseudobulk_dir / "pseudobulk_order.list.joblib"
        )
        self.pseudobulk_meta = pd.read_feather(
            self.pseudobulk_dir / "pseudobulk_metadata.feather"
        )
        self.chrom_keys = joblib.load(self.pseudobulk_dir / "chrom_keys.list.joblib")
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome
        self.catch_chroms = catch_chroms
        self.chrom_cache = {}

    def get_chrom(self, chrom_key, mat_type="csc"):
        """Get the pseudobulk whole chromosome sparse matrix."""
        cache_key = (chrom_key, mat_type)
        if cache_key in self.chrom_cache:
            return self.chrom_cache[cache_key]

        assert chrom_key in self.chrom_keys, f"chrom_key {chrom_key} not found"
        assert mat_type in ["csr", "csc"], f"mat_type {mat_type} not supported"

        data = joblib.load(self.pseudobulk_dir / f"{chrom_key}.{mat_type}.joblib")
        if self.catch_chroms:
            self.chrom_cache[cache_key] = data
        return data

    def clean_cache(self):
        """Clear the cached chromosome data."""
        self.chrom_cache.clear()


# ---------------------------------------------------------------------------
# Fragment file import
#
# Adapted from scPrinter's ``scp.pp.import_fragments`` (a wrapper around
# ``snapatac2.pp.import_data``). Two behaviours are preserved:
#
# 1. Auto-detection of the Tn5 shift already applied to a fragment file, by
#    matching the nucleotide-frequency profile around fragment ends against a
#    reference profile (``detect_shift``).
# 2. Conversion of snapatac2's fragment records into per-chromosome, single-base
#    insertion count matrices stored as ``insertion_{chrom}`` in ``.obsm``
#    (``_frags_to_insertions``) -- the representation the rest of bolero
#    consumes (see ``AdataPseudobulkMerger`` / ``SnapAnnDataDataset`` above).
#
# Unlike scPrinter's version this handles exactly one fragment file per call
# (no internal multiprocessing) and reuses bolero's ``Genome`` and
# ``one_hot_encoding`` helpers instead of scPrinter's.
# ---------------------------------------------------------------------------

# Reference nucleotide-frequency profiles (40 bp window, ACGT order) around the
# +/- ends of a correctly +4/-5 shifted fragment file. ``detect_shift`` aligns
# an observed profile to these to recover the shift. Copied verbatim from
# scPrinter so a detected shift matches scPrinter's.
_REF_FORWARD_BIAS = np.array(
    [
        [0.22801, 0.267627, 0.270118, 0.234245],
        [0.230582, 0.263092, 0.266591, 0.239735],
        [0.230966, 0.263395, 0.277853, 0.227786],
        [0.237373, 0.254811, 0.268621, 0.239195],
        [0.227756, 0.262171, 0.281433, 0.22864],
        [0.229343, 0.267518, 0.266358, 0.236781],
        [0.233688, 0.261873, 0.268756, 0.235683],
        [0.233246, 0.258417, 0.267473, 0.240864],
        [0.233571, 0.250868, 0.278438, 0.237123],
        [0.210302, 0.265267, 0.273532, 0.250899],
        [0.197471, 0.270346, 0.276262, 0.255921],
        [0.239908, 0.224086, 0.353718, 0.182288],
        [0.209381, 0.325079, 0.290731, 0.174809],
        [0.27844, 0.226113, 0.273372, 0.222075],
        [0.174853, 0.40936, 0.237924, 0.177863],
        [0.345859, 0.149262, 0.228952, 0.275927],
        [0.165938, 0.270514, 0.480596, 0.082952],
        [0.180195, 0.29281, 0.221314, 0.305681],
        [0.179562, 0.332162, 0.237392, 0.250884],
        [0.125249, 0.409938, 0.152056, 0.312757],
        [0.334499, 0.169781, 0.172222, 0.323498],
        [0.300765, 0.151369, 0.411777, 0.136089],
        [0.255409, 0.225268, 0.345623, 0.1737],
        [0.294189, 0.221911, 0.303888, 0.180012],
        [0.125051, 0.422991, 0.270467, 0.181491],
        [0.25149, 0.249151, 0.18284, 0.316519],
        [0.190253, 0.229807, 0.393018, 0.186922],
        [0.222202, 0.265248, 0.242275, 0.270275],
        [0.171303, 0.27846, 0.331291, 0.218946],
        [0.182575, 0.338698, 0.238956, 0.239771],
        [0.249742, 0.270574, 0.275662, 0.204022],
        [0.242147, 0.26794, 0.272813, 0.2171],
        [0.238624, 0.26772, 0.261579, 0.232077],
        [0.234333, 0.259208, 0.264034, 0.242425],
        [0.234719, 0.257802, 0.269948, 0.237531],
        [0.232542, 0.259099, 0.274311, 0.234048],
        [0.228187, 0.266636, 0.273797, 0.23138],
        [0.231669, 0.263802, 0.26339, 0.241139],
        [0.228852, 0.26637, 0.274964, 0.229814],
        [0.23085, 0.260809, 0.266385, 0.241956],
    ]
)
_REF_REVERSE_BIAS = np.array(
    [
        [0.227808, 0.264997, 0.272057, 0.235138],
        [0.229786, 0.261645, 0.267999, 0.24057],
        [0.229542, 0.265141, 0.277305, 0.228012],
        [0.2364, 0.25577, 0.267527, 0.240303],
        [0.228397, 0.261634, 0.278268, 0.231701],
        [0.227913, 0.266164, 0.267063, 0.23886],
        [0.233518, 0.263004, 0.268147, 0.235331],
        [0.231852, 0.259135, 0.267172, 0.241841],
        [0.233683, 0.251586, 0.27724, 0.237491],
        [0.209089, 0.266759, 0.272593, 0.251559],
        [0.200028, 0.266882, 0.283073, 0.250017],
        [0.235364, 0.228948, 0.348828, 0.18686],
        [0.213109, 0.320979, 0.290091, 0.175821],
        [0.271552, 0.238546, 0.267749, 0.222153],
        [0.186768, 0.377292, 0.242154, 0.193786],
        [0.312909, 0.176144, 0.256335, 0.254612],
        [0.176285, 0.258176, 0.440994, 0.124545],
        [0.174572, 0.297044, 0.226605, 0.301779],
        [0.174146, 0.338917, 0.232244, 0.254693],
        [0.129944, 0.406257, 0.1538, 0.309999],
        [0.329279, 0.166597, 0.17534, 0.328784],
        [0.301616, 0.149981, 0.416916, 0.131487],
        [0.251985, 0.230058, 0.337205, 0.180752],
        [0.300088, 0.215177, 0.299529, 0.185206],
        [0.083786, 0.462517, 0.282588, 0.171109],
        [0.274189, 0.22066, 0.155771, 0.34938],
        [0.174265, 0.225577, 0.425875, 0.174283],
        [0.219996, 0.271626, 0.230628, 0.27775],
        [0.171276, 0.28058, 0.333658, 0.214486],
        [0.178319, 0.342787, 0.233408, 0.245486],
        [0.255848, 0.265675, 0.27829, 0.200187],
        [0.241528, 0.267603, 0.273153, 0.217716],
        [0.238694, 0.266755, 0.261775, 0.232776],
        [0.234914, 0.2593, 0.26413, 0.241656],
        [0.234639, 0.258018, 0.270217, 0.237126],
        [0.231615, 0.258526, 0.275489, 0.23437],
        [0.226413, 0.269257, 0.273205, 0.231125],
        [0.230697, 0.264718, 0.261399, 0.243186],
        [0.228284, 0.265556, 0.273413, 0.232747],
        [0.230259, 0.260566, 0.268176, 0.240999],
    ]
)


def _circular_shift(matrix, shift):
    """Circularly shift ``matrix`` along axis 0 by ``shift`` rows."""
    return np.concatenate((matrix[-shift:, :], matrix[:-shift, :]), axis=0)


def _mse(a, b):
    """Mean squared error between two equally shaped arrays."""
    return np.mean((a - b) ** 2)


def _circular_detect(ref_bias, query_bias):
    """
    Return the circular shift (bp) best aligning ``query_bias`` to ``ref_bias``.

    A warning is printed if even the best alignment is a poor fit, which usually
    means the input was not a genuine paired-end fragment file.
    """
    assert np.shape(ref_bias) == np.shape(
        query_bias
    ), "bias matrices must have the same shape"
    context_radius = int(np.shape(ref_bias)[0] / 2)
    shift_mse = [
        _mse(_circular_shift(ref_bias, shift), query_bias)
        for shift in range(-context_radius, context_radius)
    ]
    min_mse = np.min(shift_mse)
    if min_mse > 0.002:
        print(
            "[import_fragments] warning: shift detection may be inaccurate "
            f"(minimum MSE {min_mse:.4g}); check the input file format"
        )
    return int(np.argmin(shift_mse) - context_radius)


def _get_nucleotide_freq(beds, genome, context_radius=20, paired=True):
    """
    Mean one-hot nucleotide frequency in a window around fragment +/- ends.

    Returns two ``(2 * context_radius, 4)`` arrays (forward, reverse) with ACGT
    columns, matching the layout of ``_REF_FORWARD_BIAS`` / ``_REF_REVERSE_BIAS``.
    """
    fasta = Fasta(str(genome.fasta_path))
    forward_bias, reverse_bias = [], []
    for frag in np.asarray(beds):
        chrom, start, end = frag[0], int(frag[1]), int(frag[2])
        if chrom not in genome.chrom_sizes:
            continue
        strands = ["+", "-"] if paired else [str(frag[5])]
        if "+" in strands:
            ctx = one_hot_encoding(
                fasta[chrom][
                    start - context_radius : start + context_radius
                ].seq.upper()
            )
            if ctx.shape[0] == 2 * context_radius:
                forward_bias.append(ctx)
        if "-" in strands:
            ctx = one_hot_encoding(
                fasta[chrom][end - context_radius : end + context_radius].seq.upper()
            )
            if ctx.shape[0] == 2 * context_radius:
                reverse_bias.append(ctx)
    return np.mean(forward_bias, axis=0), np.mean(reverse_bias, axis=0)


def detect_shift(fragment_path, genome, n_scan=1_000_000, n_sample=10_000):
    """
    Detect the +/- strand Tn5 shift already applied to a fragment file.

    The shift is inferred from the nucleotide composition around the fragment
    ends (the Tn5 cut site has a characteristic bias), by aligning the observed
    profile to a reference profile of a correctly +4/-5 shifted file.

    Parameters
    ----------
    fragment_path : str or Path
        Path to the (optionally gzipped) fragment file, tab-separated BED-like.
    genome : bolero.pp.Genome
        Genome providing ``fasta_path`` and ``chrom_sizes``.
    n_scan : int
        Number of leading rows to read from the file before sampling.
    n_sample : int
        Number of fragments to sample for the estimate.

    Returns
    -------
    tuple of int
        ``(plus_shift, minus_shift)`` -- the shift already applied to the left
        (+) and right (-) fragment ends. The standard pipeline yields ``(4, -5)``.
    """
    frags = pd.read_csv(fragment_path, sep="\t", header=None, comment="#", nrows=n_scan)
    frags = frags.sample(min(n_sample, len(frags)))
    single_end = frags.shape[1] >= 6 and str(frags.iloc[0, 5]) in ("+", "-")
    print(
        "[import_fragments] detecting shift in "
        f"{'single-end' if single_end else 'paired-end'} fragment file"
    )
    forward_bias, reverse_bias = _get_nucleotide_freq(
        frags, genome, paired=not single_end
    )
    plus_shift = 4 - _circular_detect(_REF_FORWARD_BIAS, forward_bias)
    minus_shift = -5 - _circular_detect(_REF_REVERSE_BIAS, reverse_bias)
    return plus_shift, minus_shift


def _frags_to_insertions(
    data,
    extra_plus_shift=0,
    extra_minus_shift=0,
    split=True,
    to_csc=False,
    sel_chrom=None,
):
    """
    Convert snapatac2 fragment records into single-base insertion matrices.

    Reads ``data.obsm['fragment_paired']`` (or ``'fragment_single'``) and writes,
    for each chromosome, a ``cell x base`` insertion count matrix into
    ``data.obsm['insertion_{chrom}']`` (or a single ``'insertion'`` when
    ``split=False``). ``extra_plus_shift`` / ``extra_minus_shift`` are the
    residual shifts applied to the left/right ends to reach the true cut site.
    """
    if "fragment_paired" in data.obsm:
        x = data.obsm["fragment_paired"]
        insertion = csr_matrix(
            (
                np.ones(len(x.indices) * 2, dtype="uint16"),
                np.stack(
                    [
                        x.indices + extra_plus_shift,
                        x.indices + x.data + extra_minus_shift,
                    ],
                    axis=-1,
                ).reshape(-1),
                x.indptr * 2,
            ),
            shape=x.shape,
        )
    elif "fragment_single" in data.obsm:
        # snapatac2 stores single-end reads as start index + signed read length
        # (positive = + strand, negative = - strand); the shift is not applied.
        x = data.obsm["fragment_single"]
        indices = np.copy(x.indices)
        mask = x.data > 0
        indices[mask] += extra_plus_shift
        indices[~mask] += 1 + extra_minus_shift
        insertion = csr_matrix(
            (np.ones(len(x.indices), dtype="uint16"), indices, x.indptr),
            shape=x.shape,
        )
    else:
        raise ValueError(
            "no fragment data in data.obsm "
            "(expected 'fragment_paired' or 'fragment_single')"
        )
    insertion.sort_indices()
    insertion.sum_duplicates()

    if not split:
        data.obsm["insertion"] = insertion
        return data

    ref = data.uns["reference_sequences"]
    seq_name = list(ref["reference_seq_name"])
    seq_len = np.asarray(ref["reference_seq_length"]).astype("int64")
    bounds = np.concatenate([[0], np.cumsum(seq_len)]).astype("int64")
    for i, chrom in enumerate(seq_name):
        if sel_chrom is not None and chrom not in sel_chrom:
            continue
        chrom_ins = insertion[:, bounds[i] : bounds[i + 1]]
        data.obsm[f"insertion_{chrom}"] = chrom_ins.tocsc() if to_csc else chrom_ins
    return data


def import_fragments(
    fragment_path,
    savename,
    genome,
    *,
    barcodes=None,
    sample_name=None,
    plus_shift=4,
    minus_shift=-5,
    auto_detect_shift=True,
    min_num_fragments=200,
    sorted_by_barcode=False,
    is_paired=True,
    to_csc=False,
    **kwargs,
):
    """
    Import one ATAC fragment file into a snapatac2 backed AnnData.

    Thin single-file wrapper around ``snapatac2.pp.import_fragments`` that (1)
    auto-detects the Tn5 shift already applied to the file, (2) imports the
    fragments, and (3) stores per-chromosome single-base insertion count
    matrices (``insertion_{chrom}`` in ``.obsm``) that the rest of bolero
    consumes. Adapted from scPrinter's ``scp.pp.import_fragments`` but limited
    to a single file per call (no internal multiprocessing).

    Parameters
    ----------
    fragment_path : str or Path
        Path to the (optionally gzipped) fragment file.
    savename : str or Path
        Output path for the backed snapatac2 AnnData (hdf5).
    genome : bolero.pp.Genome or str
        Genome object (or a name to construct one) providing ``chrom_sizes``,
        ``fasta_path`` and ``name``.
    barcodes : list of str or Path, optional
        Barcode whitelist passed to snapatac2 as ``whitelist`` (a list of
        barcodes, or a path to a file with one barcode per line). ``None`` keeps
        every barcode passing ``min_num_fragments``.
    sample_name : str, optional
        If given, stored in ``obs['sample']`` (useful to distinguish samples).
    plus_shift, minus_shift : int
        The shift already applied to the left (+) and right (-) fragment ends.
        Ignored when ``auto_detect_shift`` is True. Defaults ``4`` / ``-5`` match
        the standard pipeline.
    auto_detect_shift : bool
        If True (default), detect ``plus_shift`` / ``minus_shift`` from the file
        (overrides the passed values).
    min_num_fragments : int
        Minimum fragments per barcode to keep a cell.
    sorted_by_barcode : bool
        Whether the input is sorted by barcode. Default False (10x fragment
        files are coordinate-sorted).
    is_paired : bool
        Whether the fragment file is paired-end. Default True.
    to_csc : bool
        Store the per-chromosome insertion matrices as CSC instead of CSR.
        Default False. These matrices are ``cell x chromosome-length`` (up to
        ~2.5e8 columns), so a CSC index is huge (its indptr length equals the
        chromosome length) and ~80x slower to write than CSR for the same file
        size. CSR (the default) is also the format the pseudobulk-merge path
        (``AdataPseudobulkMerger``) consumes. Set True only if you specifically
        need CSC-backed storage.
    **kwargs
        Forwarded to ``snapatac2.pp.import_fragments`` (e.g. ``n_jobs``,
        ``chrM``, ``chunk_size``).

    Returns
    -------
    str
        ``savename``, the path to the written AnnData (closed on return).
    """
    fragment_path = str(fragment_path)
    savename = str(savename)
    if isinstance(genome, str):
        genome = Genome(genome)

    # historical snapatac2 kwargs that newer versions no longer accept
    for k in ("min_tsse", "low_memory"):
        kwargs.pop(k, None)

    if auto_detect_shift:
        plus_shift, minus_shift = detect_shift(fragment_path, genome)
        print(
            f"[import_fragments] auto-detected plus_shift={plus_shift}, "
            f"minus_shift={minus_shift} for {fragment_path} "
            "(set auto_detect_shift=False to use the values you passed)"
        )
    # snapatac2 records insertions at the raw fragment ends; this residual shift
    # moves them onto the true Tn5 cut site (standard pipeline is +4 / -5).
    extra_plus_shift = 4 - plus_shift
    extra_minus_shift = -5 - minus_shift

    chrom_sizes = {str(c): int(s) for c, s in genome.chrom_sizes.items()}
    # snapatac2 2.9.0 renamed import_data -> import_fragments and dropped the
    # shift_left / shift_right args: it now records the raw fragment ends, and
    # the Tn5 shift is applied downstream in _frags_to_insertions.
    data = snap.pp.import_fragments(
        fragment_path,
        chrom_sizes,
        is_paired=is_paired,
        file=savename,
        whitelist=barcodes,
        min_num_fragments=min_num_fragments,
        sorted_by_barcode=sorted_by_barcode,
        **kwargs,
    )
    n_cells = len(data.obs_names)
    data.obs["frag_path"] = [fragment_path] * n_cells
    if sample_name is not None:
        data.obs["sample"] = [str(sample_name)] * n_cells

    _frags_to_insertions(
        data,
        extra_plus_shift=extra_plus_shift,
        extra_minus_shift=extra_minus_shift,
        split=True,
        to_csc=to_csc,
    )
    data.uns["genome"] = genome.name
    data.close()
    return savename
