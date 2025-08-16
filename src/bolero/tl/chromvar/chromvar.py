import numpy as np
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
        dev, dtype="float32", obs=adata.obs.copy()
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
