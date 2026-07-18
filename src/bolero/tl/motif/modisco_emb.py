import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from modiscolite import affinitymat
from modiscolite.tfmodisco import _filter_by_correlation
from openTSNE import TSNEEmbedding, affinity, initialization

"""
The function tsne is from cytograph2 package.

https://github.com/linnarsson-lab/cytograph2/blob/master/cytograph/embedding/art_of_tsne.py

The idea behind that is based on :cite:p:`Kobak2019` with T-SNE algorithm implemented in
[openTSNE](https://opentsne.readthedocs.io/en/latest/) :cite:p:`Policar2019`.

"""


def _tsne(
    X: np.ndarray,
    metric: str | Callable = "precomputed",
    exaggeration: float = -1,
    perplexity: int = 30,
    n_jobs: int = -1,
) -> TSNEEmbedding:
    """
    Calculate T-SNE embedding with the openTSNE package.

    Implementation of Dmitry Kobak and Philipp Berens
    "The art of using t-SNE for single-cell transcriptomics" based on openTSNE.
    See https://doi.org/10.1038/s41467-019-13056-x | www.nature.com/naturecommunications

    Parameters
    ----------
    X
        The data matrix of shape (n_cells, n_genes) i.e. (n_samples, n_features)
    metric
        Any metric allowed by PyNNDescent (default: 'euclidean')
    exaggeration
        The exaggeration to use for the embedding
    perplexity
        The perplexity to use for the embedding
    n_jobs
        Number of CPUs to use

    Returns
    -------
    The embedding as an opentsne.TSNEEmbedding object (which can be cast to an np.ndarray)
    """
    n = X.shape[0]
    if n > 100_000:
        if exaggeration == -1:
            exaggeration = 1 + n / 333_333
        # Subsample, optimize, then add the remaining cells and optimize again
        # Also, use exaggeration == 4
        logging.info(f"Creating subset of {n // 40} elements")
        # Subsample and run a regular art_of_tsne on the subset
        indices = np.random.permutation(n)
        reverse = np.argsort(indices)
        X_sample, X_rest = X[indices[: n // 40]], X[indices[n // 40 :]]
        logging.info("Embedding subset")
        Z_sample = _tsne(X_sample, metric=metric)

        logging.info(
            f"Preparing partial initial embedding of the {n - n // 40} remaining elements"
        )
        if isinstance(Z_sample.affinities, affinity.Multiscale):
            rest_init = Z_sample.prepare_partial(
                X_rest, k=1, perplexities=[1 / 3, 1 / 3]
            )
        else:
            rest_init = Z_sample.prepare_partial(X_rest, k=1, perplexity=1 / 3)
        logging.info("Combining the initial embeddings, and standardizing")
        init_full = np.vstack((Z_sample, rest_init))[reverse]
        init_full = init_full / (np.std(init_full[:, 0]) * 10000)

        logging.info("Creating multiscale affinities")
        affinities = affinity.PerplexityBasedNN(
            X, perplexity=perplexity, metric=metric, method="approx", n_jobs=n_jobs
        )
        logging.info("Creating TSNE embedding")
        Z = TSNEEmbedding(
            init_full, affinities, negative_gradient_method="fft", n_jobs=n_jobs
        )
        logging.info("Optimizing, stage 1")
        Z.optimize(
            n_iter=250,
            inplace=True,
            exaggeration=12,
            momentum=0.5,
            learning_rate=n / 12,
            n_jobs=n_jobs,
        )
        logging.info("Optimizing, stage 2")
        Z.optimize(
            n_iter=750,
            inplace=True,
            exaggeration=exaggeration,
            momentum=0.8,
            learning_rate=n / 12,
            n_jobs=n_jobs,
        )
    elif n > 3_000:
        if exaggeration == -1:
            exaggeration = 1
        # Use multiscale perplexity
        affinities_multiscale_mixture = affinity.Multiscale(
            X,
            perplexities=[perplexity, n / 100],
            metric=metric,
            method="approx",
            n_jobs=n_jobs,
        )
        init = initialization.pca(X)
        Z = TSNEEmbedding(
            init,
            affinities_multiscale_mixture,
            negative_gradient_method="fft",
            n_jobs=n_jobs,
        )
        Z.optimize(
            n_iter=250,
            inplace=True,
            exaggeration=12,
            momentum=0.5,
            learning_rate=n / 12,
            n_jobs=n_jobs,
        )
        Z.optimize(
            n_iter=750,
            inplace=True,
            exaggeration=exaggeration,
            momentum=0.8,
            learning_rate=n / 12,
            n_jobs=n_jobs,
        )
    else:
        if exaggeration == -1:
            exaggeration = 1
        # Just a plain TSNE with high learning rate
        lr = max(200, n / 12)
        aff = affinity.PerplexityBasedNN(
            X, perplexity=perplexity, metric=metric, method="approx", n_jobs=n_jobs
        )
        init = initialization.pca(X)
        Z = TSNEEmbedding(
            init, aff, learning_rate=lr, n_jobs=n_jobs, negative_gradient_method="fft"
        )
        Z.optimize(250, exaggeration=12, momentum=0.5, inplace=True, n_jobs=n_jobs)
        Z.optimize(
            750, exaggeration=exaggeration, momentum=0.8, inplace=True, n_jobs=n_jobs
        )
    return np.array(Z)


def modisco_seqlets_embedding(
    seqlets,
    nearest_neighbors_to_compute=500,
    sign="pos",
    min_overlap_while_sliding=0.7,
    affmat_correlation_threshold=0.15,
    tsne_perplexity=30.0,
    corr_filter=False,
    n_jobs=1,
    tsne=True,
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
    tsne : bool, optional
        Whether to compute the 2D t-SNE embedding, by default True.

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
    else:
        print("3. Skipping correlation filtering")

    # Step 4: Create distance matrix
    print("4. Creating distance matrix")
    n_seqlet = len(seqlet_neighbors)
    dist_mat = np.ones((n_seqlet, n_seqlet))

    for row, (aff, neighbor) in enumerate(
        zip(fine_affmat_nn, seqlet_neighbors, strict=False)
    ):
        dist_mat[row, neighbor] -= aff

    # Step 5: Run t-SNE
    if tsne:
        print("5. Running t-SNE")
        embeddings = _tsne(
            dist_mat,
            perplexity=tsne_perplexity,
            n_jobs=n_jobs,
            metric="precomputed",
        )
        seqlet_names = [seqlet.name for seqlet in seqlets]
        embeddings = pd.DataFrame(embeddings, index=seqlet_names)
        return dist_mat, embeddings
    else:
        return dist_mat, None
