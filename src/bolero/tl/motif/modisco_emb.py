import numpy as np
import pandas as pd
from modiscolite import affinitymat
from modiscolite.tfmodisco import _filter_by_correlation
from openTSNE import TSNE, affinity


def _tsne(dist_mat, perplexity=10, n_jobs=1, n_components=2):
    # Create a custom affinity object
    custom_affinity = affinity.PerplexityBasedNN(
        data=dist_mat,
        perplexity=perplexity,
        metric="precomputed",
        method="pynndescent",
        n_jobs=n_jobs,
    )

    # Initialize and run TSNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        initialization="spectral",
        metric="precomputed",
        n_jobs=n_jobs,
    )
    embeddings = tsne.fit(affinities=custom_affinity)
    return embeddings, tsne


def modisco_seqlets_embedding(
    seqlets,
    nearest_neighbors_to_compute=500,
    sign="pos",
    min_overlap_while_sliding=0.7,
    affmat_correlation_threshold=0.15,
    tsne_perplexity=10.0,
    corr_filter=False,
    n_jobs=1,
    n_components=2,
):
    """
    Calculate affinity matrix between seqlets and embed them in 2D space using t-SNE.

    Adapted from modisco.tfmodisco.seqlets_to_patterns function.
    https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/tfmodisco.py

    Parameters
    ----------
    seqlets : list of modisco.core.Seqlet
        List of seqlets to embed.
    nearest_neighbors_to_compute : int, optional
        Number of nearest neighbors to compute for cosine similarity as an approximate start,
        by default 500.
    sign : str, optional
        Sign of the cosine similarity, by default "pos".
    min_overlap_while_sliding : float, optional
        Minimum overlap while sliding, by default 0.7.
    affmat_correlation_threshold : float, optional
        Threshold for filtering by correlation, by default 0.15.
    tsne_perplexity : float, optional
        Perplexity for t-SNE, by default 10.0.
    corr_filter : bool, optional
        Whether to filter seqlet by correlation, by default False.
    n_jobs : int, optional
        Number of parallel jobs, by default 1.
    n_components : int, optional
        Number of components for t-SNE, by default 2.

    Returns
    -------
    dist_mat : np.ndarray
        Distance matrix between seqlets.
    embeddings : pd.DataFrame
        Embeddings of seqlets in 2D space.
    tsne : openTSNE.TSNE
        t-SNE object.
    """
    assert sign.lower() in ["pos", "neg"], "sign must be either 'pos' or 'neg'"
    sign = -1 if sign.lower() == "neg" else 1

    # Step 1: Generate coarse resolution
    print("1. Generating coarse representation")
    coarse_affmat_nn, seqlet_neighbors = affinitymat.cosine_similarity_from_seqlets(
        seqlets=seqlets, n_neighbors=nearest_neighbors_to_compute, sign=sign
    )

    # Step 2: Generate fine representation
    print("2. Generating fine representation")
    fine_affmat_nn = affinitymat.jaccard_from_seqlets(
        seqlets=seqlets,
        seqlet_neighbors=seqlet_neighbors,
        min_overlap=min_overlap_while_sliding,
    )

    # Optional Step 3: Filter by correlation
    if corr_filter:
        print("3. Filtering by correlation")
        seqlets, seqlet_neighbors, fine_affmat_nn = _filter_by_correlation(
            seqlets,
            seqlet_neighbors,
            coarse_affmat_nn,
            fine_affmat_nn,
            affmat_correlation_threshold,
        )

    # Step 4: Create distance matrix
    print("4. Creating distance matrix")
    n_seqlet = len(seqlet_neighbors)
    dist_mat = np.ones((n_seqlet, n_seqlet))

    for row, (aff, neighbor) in enumerate(zip(fine_affmat_nn, seqlet_neighbors)):
        dist_mat[row, neighbor] -= aff

    # Step 5: Run t-SNE
    print("5. Running t-SNE")
    embeddings, tsne = _tsne(
        dist_mat, perplexity=tsne_perplexity, n_jobs=n_jobs, n_components=n_components
    )
    seqlet_names = [seqlet.name for seqlet in seqlets]
    embeddings = pd.DataFrame(embeddings, index=seqlet_names)
    return dist_mat, embeddings, tsne
