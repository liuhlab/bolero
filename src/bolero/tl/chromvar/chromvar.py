import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sparse
from anndata import AnnData
from pynndescent import NNDescent
from scipy.sparse import csr_matrix as scipy_csr_matrix
from tqdm.auto import trange


def create_bins_and_sample_background(trans_norm_mat, bs, w, niterations):
    """
    Translated from the chromVAR R package.

    Parameters
    ----------
    trans_norm_mat
    bs
    w
    niterations
    """
    # Create bins
    bins1 = np.linspace(np.min(trans_norm_mat[:, 0]), np.max(trans_norm_mat[:, 0]), bs)
    bins2 = np.linspace(np.min(trans_norm_mat[:, 1]), np.max(trans_norm_mat[:, 1]), bs)

    # Create bin_data
    bin_data = np.array(np.meshgrid(bins1, bins2)).T.reshape(-1, 2)

    # Calculate Euclidean distances
    bin_dist = scipy.spatial.distance.cdist(bin_data, bin_data, "euclidean")

    # Calculate probabilities
    bin_p = scipy.stats.norm.pdf(bin_dist, 0, w)
    # Find nearest bin membership for each point in trans_norm_mat
    print("NNDescent", bin_data.shape)
    index = NNDescent(bin_data, metric="euclidean", n_neighbors=1, n_jobs=1)
    indices, _ = index.query(trans_norm_mat, 1)
    # distance = scipy.spatial.distance.cdist(trans_norm_mat, bin_data)
    # indices = np.argmin(distance, axis=1)
    bin_membership = indices.flatten()
    # Calculate bin density
    unique, counts = np.unique(bin_membership, return_counts=True)
    bin_density = np.ones(bs**2)  # init with one pseudocount
    bin_density[unique] = counts

    # Sample background peaks
    # This assumes bg_sample_helper is defined as per previous instruction
    background_peaks = bg_sample_helper(bin_membership, bin_p, bin_density, niterations)

    return background_peaks


def bg_sample_helper(bin_membership, bin_p, bin_density, niterations):
    """Sample background peaks from the bin membership."""
    n = len(bin_membership)
    out = np.zeros((n, niterations), dtype=int)

    for i in trange(len(bin_density), desc="Sampling background peaks"):
        ix = np.where(bin_membership == i)[0]
        if len(ix) == 0:  # Skip if no members in bin
            continue
        p_tmp = bin_p[i,]
        p = (p_tmp / bin_density)[bin_membership]
        p /= p.sum()
        # Sampling with replacement according to probabilities
        sampled_indices = np.random.choice(
            np.arange(len(p)), size=niterations * len(ix), replace=True, p=p
        )
        out[ix, :] = sampled_indices.reshape((len(ix), niterations))

    return out


def sample_bg_peaks(
    reads_per_peak,
    gc_bias,
    niterations=250,
    w=0.1,
    bs=50,
):
    """
    This function samples background peaks for chromVAR analysis in single-cell ATAC-seq data.
    """
    assert np.min(reads_per_peak) > 0, "Some peaks have no reads"
    reads_per_peak = np.log10(reads_per_peak)
    reads_per_peak = np.array(reads_per_peak).reshape(-1)

    mat = np.array([reads_per_peak, gc_bias])
    chol_cov_mat = np.linalg.cholesky(np.cov(mat))
    trans_norm_mat = scipy.linalg.solve_triangular(
        a=chol_cov_mat, b=mat, lower=True
    ).transpose()

    print("Sampling nearest neighbors")
    knn_idx = create_bins_and_sample_background(
        trans_norm_mat, bs=bs, w=w, niterations=niterations
    )
    return knn_idx


def _parse_peak_regions(adata):
    """Parse peak coordinates from an AnnData into a BED-like DataFrame.

    Uses ``adata.var[["Chromosome", "Start", "End"]]`` when present, otherwise parses
    ``adata.var_names`` written as ``"chrom:start-end"``.

    Parameters
    ----------
    adata : anndata.AnnData
        Peak-by-cell (or cell-by-peak) object whose ``var`` indexes the peaks.

    Returns
    -------
    pandas.DataFrame
        Columns ``Chromosome``, ``Start``, ``End``, ``Name`` indexed by ``adata.var_names``.
    """
    var = adata.var
    if {"Chromosome", "Start", "End"}.issubset(var.columns):
        bed = var[["Chromosome", "Start", "End"]].copy()
        bed["Start"] = bed["Start"].astype(int)
        bed["End"] = bed["End"].astype(int)
    else:
        chroms, starts, ends = [], [], []
        for name in adata.var_names.astype(str):
            chrom, sep, coords = name.rpartition(":")
            start, dash, end = coords.partition("-")
            if not (sep and dash):
                raise ValueError(
                    f"Cannot parse peak coordinate from var name {name!r}; expected "
                    "'chrom:start-end' or Chromosome/Start/End columns in adata.var."
                )
            chroms.append(chrom)
            starts.append(int(start))
            ends.append(int(end))
        bed = pd.DataFrame(
            {"Chromosome": chroms, "Start": starts, "End": ends},
            index=adata.var_names,
        )
    bed["Name"] = adata.var_names.astype(str)
    return bed


def get_peak_bias(adata, genome, base_order=None):
    """
    Compute per-peak GC content and the overall background nucleotide frequency.

    This is the bolero-native replacement for scPrinter's ``get_peak_bias``: it fills
    ``adata.var["gc_content"]`` (used, together with per-peak read depth, to sample
    matched background peaks via :func:`sample_bg_peaks`) and ``adata.uns["bg_freq"]``
    (the genome-wide-over-peaks A/C/G/T frequency used as the MOODS background for
    motif matching in :func:`scan_peak_motifs`).

    Peak sequences are read directly from the genome FASTA (``genome.fasta_path``) with
    ``pyfaidx``; only A/C/G/T are counted (``N`` and other symbols are ignored).

    Parameters
    ----------
    adata : anndata.AnnData
        Peak object; peaks are taken from ``adata.var`` (see :func:`_parse_peak_regions`).
    genome : bolero.pp.genome.Genome
        Genome providing ``fasta_path`` for the same assembly the peaks are on.
    base_order : str, optional
        Base order for ``bg_freq``. Defaults to ``bolero.pp.seq.DEFAULT_ONE_HOT_ORDER``
        (``"ACGT"``), matching the order expected by MOODS.

    Returns
    -------
    None
        ``adata`` is modified in place (``var["gc_content"]``, ``uns["bg_freq"]``).
    """
    from pyfaidx import Fasta

    from bolero.pp.seq import DEFAULT_ONE_HOT_ORDER

    order = base_order or DEFAULT_ONE_HOT_ORDER
    base_to_idx = {b: i for i, b in enumerate(order)}

    bed = _parse_peak_regions(adata)
    counts = np.zeros((bed.shape[0], 4), dtype=np.int64)
    fasta = Fasta(str(genome.fasta_path), sequence_always_upper=True)
    try:
        chroms = bed["Chromosome"].to_numpy()
        starts = bed["Start"].to_numpy()
        ends = bed["End"].to_numpy()
        for i in trange(bed.shape[0], desc="Reading peak sequences"):
            seq = str(fasta[chroms[i]][int(starts[i]) : int(ends[i])])
            for base, idx in base_to_idx.items():
                counts[i, idx] = seq.count(base)
    finally:
        fasta.close()

    total = counts.sum(axis=1)
    gc = (counts[:, base_to_idx["G"]] + counts[:, base_to_idx["C"]]) / np.maximum(
        total, 1
    )
    adata.var["gc_content"] = pd.Series(gc, index=adata.var_names)

    bg = counts.sum(axis=0).astype("float64")
    adata.uns["bg_freq"] = bg / bg.sum()
    return


def scan_peak_motifs(
    adata,
    genome,
    motif_path=None,
    motif_db="JASPAR2024_CORE_vertebrates",
    bg=None,
    pvalue=5e-5,
    pseudocount=0.8,
    n_jobs=16,
    mode="motifmatchr",
    verbose=True,
):
    """
    Match motifs against each peak and store the hit matrix on ``adata``.

    This is the bolero-native replacement for scPrinter's ``Motifs.chromvar_scan``: it
    scans the peaks with :class:`bolero.tl.motif.scan.Motifs` (a MOODS motifmatchr port)
    and fills ``adata.varm["motif_match"]`` (a ``float32`` peak-by-motif 0/1 matrix) and
    ``adata.uns["motif_name"]`` (the JASPAR matrix ids), the inputs
    :func:`compute_deviations` needs.

    Parameters
    ----------
    adata : anndata.AnnData
        Peak object; peaks are taken from ``adata.var`` (see :func:`_parse_peak_regions`).
    genome : bolero.pp.genome.Genome
        Genome providing ``fasta_path`` for the same assembly the peaks are on.
    motif_path : str or pathlib.Path, optional
        JASPAR-format PFM file. If ``None``, a cached copy of ``motif_db`` is fetched via
        :func:`bolero.tl.motif.jaspar.get_jaspar_motif_file`.
    motif_db : str, optional
        JASPAR database key used when ``motif_path`` is ``None``. Default
        ``"JASPAR2024_CORE_vertebrates"``.
    bg : tuple of float or str, optional
        Background nucleotide frequency (A/C/G/T). If ``None``, uses
        ``adata.uns["bg_freq"]`` when available (from :func:`get_peak_bias`), else
        ``"even"``.
    pvalue : float, optional
        Motif match p-value threshold. Default ``5e-5`` (motifmatchr default).
    pseudocount : float, optional
        Pseudocount for the PFMs. Default ``0.8`` (motifmatchr default).
    n_jobs : int, optional
        Number of processes for scanning. Default ``16``.
    mode : {"motifmatchr", "moods"}, optional
        Motif-matching mode. Default ``"motifmatchr"``.
    verbose : bool, optional
        Whether to show a scanning progress bar. Default ``True``.

    Returns
    -------
    None
        ``adata`` is modified in place (``varm["motif_match"]``, ``uns["motif_name"]``).
    """
    from bolero.tl.motif.scan import Motifs

    if motif_path is None:
        from bolero.tl.motif.jaspar import get_jaspar_motif_file

        motif_path = get_jaspar_motif_file(motif_db)

    if bg is None:
        bg = adata.uns["bg_freq"] if "bg_freq" in adata.uns else "even"
    if not isinstance(bg, str):
        bg = tuple(float(x) for x in np.asarray(bg).reshape(-1))

    def _jaspar_id(name):
        # header names look like "MA0478.2\tFOSL2"; keep the JASPAR matrix id
        return name.split("\t")[0].split(" ")[0]

    motifs = Motifs(
        str(motif_path),
        str(genome.fasta_path),
        bg=bg,
        motif_name_func=_jaspar_id,
        n_jobs=n_jobs,
        pseudocount=pseudocount,
        pvalue=pvalue,
        mode=mode,
    )
    motifs.prep_scanner(None, pseudocount=pseudocount, pvalue=pvalue)

    bed = _parse_peak_regions(adata)
    hits = motifs.scan(bed, verbose=verbose)["hit"]
    # align to adata.var order and store
    hits = hits.loc[adata.var_names.astype(str)]
    adata.varm["motif_match"] = hits.values.astype("float32")
    adata.uns["motif_name"] = np.asarray(hits.columns)
    return


def scipy_to_cupy_sparse(sparse_matrix):
    """
    A function that converts a SciPy sparse matrix to a CuPy sparse matrix. Only supports CSR matrices now.

    Parameters
    ----------
    sparse_matrix
    """
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

    if not isinstance(sparse_matrix, scipy_csr_matrix):
        raise ValueError("Input matrix must be a SciPy CSR matrix")

    # Get the CSR components of the SciPy sparse matrix
    data = sparse_matrix.data.astype("float")
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    shape = sparse_matrix.shape

    # Convert the components to CuPy arrays
    data_cp = cp.array(data)
    indices_cp = cp.array(indices)
    indptr_cp = cp.array(indptr)

    # Create a CuPy CSR matrix with these components
    cupy_sparse_matrix = cupy_csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
    return cupy_sparse_matrix


def compute_deviations(adata, chunk_size: int = 10000, device="cuda"):
    """
    Computes the deviation of motif matches from the background for each cell.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing the count matrix, background peaks, and motif match information.
    chunk_size : int, optional
        The size of chunks to process the data in. Default is 10000. It is recommended to set this
        such that there's no GPU memory overflow. Although our implementation would tolerate that,
        it will be much slower if there's overflow.
    device : str, optional
        The device to use for computation. Can be either "cuda" for GPU or "cpu" for CPU. Default is "cuda".

    Returns
    -------
    dev : AnnData
        The AnnData object containing the deviation values for each cell and motif match.
    """
    assert (
        "bg_peaks" in adata.varm_keys()
    ), "Cannot find background peaks in the input object, please first run get_bg_peaks!"
    if device == "cuda":
        import cupy as backend
    else:
        import numpy as backend

    print("Computing expectation reads per cell and peak...")
    if "expectation_var" not in adata.var:
        expectation_var = backend.asarray(
            adata.X.sum(0), dtype=backend.float32
        ).reshape((1, adata.X.shape[1]))
    else:
        print("precomputed expectation_var")
        expectation_var = backend.asarray(
            adata.var["expectation_var"], dtype=backend.float32
        ).reshape((1, adata.X.shape[1]))
    expectation_var /= expectation_var.sum()
    if "expectation_obs" not in adata.obs:
        expectation_obs = np.asarray(adata.X.sum(1), dtype=np.float32).reshape(
            (adata.X.shape[0], 1)
        )
    else:
        print("precomputed expectation_obs")
        expectation_obs = np.asarray(
            adata.obs["expectation_obs"], dtype=np.float32
        ).reshape((adata.X.shape[0], 1))

    motif_match = backend.asarray(adata.varm["motif_match"], dtype=backend.float32)

    obs_dev = np.zeros((adata.n_obs, motif_match.shape[1]), dtype=np.float32)
    n_bg_peaks = adata.varm["bg_peaks"].shape[1]
    # bg_dev = np.zeros((n_bg_peaks, adata.n_obs, motif_match.shape[1]), dtype=np.float32)
    mean_bg_dev = np.zeros_like(obs_dev)
    std_bg_dev = np.zeros_like(obs_dev)

    print("Computing deviations...")
    for start in range(0, adata.n_obs, chunk_size):
        end = min(start + chunk_size, adata.n_obs)
        temp_adata = adata[start:end].copy()
        X_chunk = temp_adata.X
        expectation_obs_chunk = backend.asarray(expectation_obs[start:end])
        if sparse.isspmatrix(X_chunk):
            if device == "cuda":
                X_chunk = scipy_to_cupy_sparse(X_chunk)
            else:
                X_chunk = X_chunk.tocsr()
        else:
            X_chunk = backend.array(X_chunk)
        res = _compute_deviations(
            motif_match,
            X_chunk,
            expectation_obs_chunk,
            expectation_var,
            device=device,
        )
        obs_dev[start:end, :] = res.get() if device == "cuda" else res
        bg_dev_chunk = np.zeros(
            (n_bg_peaks, end - start, motif_match.shape[1]), dtype=np.float32
        )
        for i in trange(n_bg_peaks):
            bg_peak_idx = backend.array(adata.varm["bg_peaks"][:, i]).flatten()
            bg_motif_match = motif_match[bg_peak_idx, :]
            res = _compute_deviations(
                bg_motif_match,
                X_chunk,
                expectation_obs_chunk,
                expectation_var,
                device=device,
            )
            bg_dev_chunk[i, :, :] = res.get() if device == "cuda" else res
        mean_bg_dev[start:end, :] = np.mean(bg_dev_chunk, axis=0)
        std_bg_dev[start:end, :] = np.std(bg_dev_chunk, axis=0)
        del temp_adata, X_chunk
    print("Finish computing deviations...")

    # mean_bg_dev = np.mean(bg_dev, axis=0)
    # std_bg_dev = np.std(bg_dev, axis=0)
    dev = (obs_dev - mean_bg_dev) / std_bg_dev
    dev = np.nan_to_num(dev, nan=0.0)

    dev = AnnData(
        dev.astype("float32"), obs=adata.obs.copy()
    )  # Convert back to CPU for AnnData compatibility
    dev.var_names = adata.uns["motif_name"]
    return dev


def _compute_deviations(motif_match, count, expectation_obs, expectation_var, device):
    if device == "cuda":
        import cupy as backend
    else:
        import numpy as backend

    observed = count.dot(motif_match)
    expected = expectation_obs.dot(expectation_var.dot(motif_match))
    out = backend.zeros_like(expected)
    backend.divide(observed - expected, expected, out=out)
    out[expected == 0] = 0
    return out
