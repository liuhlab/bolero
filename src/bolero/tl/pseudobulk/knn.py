from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynndescent
import ray
import seaborn as sns
from geosketch import gs
from scipy.stats import gamma


def _trigger_jit():
    # trigger jit compilation
    pynndescent.NNDescent(np.ones((5, 5)), n_neighbors=1, n_jobs=-1).prepare()

    # the pynndescent must be compiled before ray.init(), otherwise it will cause error says:
    # TypingError: Failed in nopython mode pipeline (step: nopython frontend)
    # Failed in nopython mode pipeline (step: nopython frontend)
    # Untyped global name 'print': Cannot determine Numba type of <class 'function'>
    # Version info when writing this comment:
    # ray.__verion__ '2.12.0'
    # pynnDescent.__version__ '0.5.12'
    # numba.__version__ '0.59.1'
    # numpy.__version__ '1.26.4'
    return


_trigger_jit()


def _mask_far_neighbors(idx, dist, max_quantile):
    use_n = min(dist.shape[1], 5)
    _radius = dist[:, :use_n].ravel()

    _radius = _radius[np.isfinite(_radius)]
    if len(_radius) < 100:
        return idx

    # mask "far" cells based on overall distances distribution using Gamma 95%
    alpha_hat, loc_hat, beta_hat = gamma.fit(_radius)
    max_allowed_radius = gamma.ppf(max_quantile, alpha_hat, loc=loc_hat, scale=beta_hat)

    # if the distance is larger than max_allowed_radius, make the index point to the first cell,
    # so these "far" cells will not be added to bag later
    masked_idx = np.where(dist > max_allowed_radius, idx[:, [0]], idx)
    return masked_idx


def _overlap_distance(a, b):
    set_a = set(a)
    set_b = set(b)
    a_and_b = set_a & set_b
    return 1 - len(a_and_b) / min(len(set_a), len(set_b))


def _run_nndescent(train_data, query_data, k, n_jobs, max_dist_q):
    _index = pynndescent.NNDescent(
        train_data,
        n_neighbors=k,
        n_jobs=n_jobs,
        parallel_batch_queries=True,
        low_memory=False,
    )
    nn_idx, nn_dist = _index.query(query_data, k=k)
    if max_dist_q is not None:
        nn_idx = _mask_far_neighbors(idx=nn_idx, dist=nn_dist, max_quantile=max_dist_q)
    return nn_idx, nn_dist, _index


def _two_step_pseudobulk(embedding, k, oversample, n_jobs, max_dist_q):
    """
    Two-step pseudobulk generation algorithm
    """
    # both for the number of sketch cells and the number of cells in each mini pseudobulk
    k_sqrt = np.round(np.sqrt(k)).astype(int)

    # mini bulk cells
    cell_index = embedding.index
    n_mini_pseudobulk = np.ceil(embedding.shape[0] / k_sqrt * oversample).astype(int)
    minibulk_cells = pd.Index(
        np.random.choice(cell_index, n_mini_pseudobulk, replace=False)
    )
    minibulk_embedding = embedding.loc[minibulk_cells]

    # generate mini-bulk to cell map
    remaining_cells = cell_index.copy()
    minibulk_to_cells = OrderedDict()
    minibulk_nn_idx, _, _ = _run_nndescent(
        train_data=embedding.values,
        query_data=minibulk_embedding.values,
        k=k_sqrt,
        n_jobs=n_jobs,
        max_dist_q=max_dist_q,
    )
    for _cell, _nn_idx in zip(minibulk_cells, minibulk_nn_idx):
        minibulk_to_cells[_cell] = set(cell_index[_nn_idx])
        remaining_cells = remaining_cells.difference(minibulk_to_cells[_cell])

    # generte pseudobulk to sketch map
    pseudobulk_to_minibulk = OrderedDict()
    pseudobulk_nn_idx, _, minibulk_index = _run_nndescent(
        train_data=minibulk_embedding.values,
        query_data=minibulk_embedding.values,
        k=int(k_sqrt * 3),
        n_jobs=n_jobs,
        max_dist_q=max_dist_q,
    )

    # assign all remaining cells to its closest mini-bulk
    _remain_cells_embedding = embedding.loc[remaining_cells]
    _remain_cells_to_minibulk_nn_idx, _ = minibulk_index.query(
        _remain_cells_embedding.values, k=k
    )
    for _cell, _nn_idx in zip(remaining_cells, _remain_cells_to_minibulk_nn_idx):
        minibulk_to_cells[minibulk_cells[_nn_idx[0]]].add(_cell)

    # assign all mini-bulks to pseudobulk, make sure each pseudobulk has at least k cells
    for pseudobulk_cell, _minibulk_idx in zip(minibulk_cells, pseudobulk_nn_idx):
        use_cells = set()
        for pos, _minibulk in enumerate(_minibulk_idx):  # noqa: B007
            use_cells.update(minibulk_to_cells[minibulk_cells[_minibulk]])
            if len(use_cells) >= k:
                break
        pseudobulk_to_minibulk[pseudobulk_cell] = set(
            minibulk_cells[_minibulk_idx[: pos + 1]]
        )

    return pseudobulk_to_minibulk, minibulk_to_cells


@ray.remote
def _gs_parallel(embedding, k):
    idx = gs(embedding, k, replace=False)
    return idx


def geosketch_prioritize(embedding, n_steps=100):
    """
    Prioritize cells based on geosketch sampling

    Parameters
    ----------
    embedding : pd.DataFrame
        Cell embedding matrix with cells in index
    n_steps : int
        Number of steps to sample

    Returns
    -------
    priority : pd.Series
        Prioritized cells with index as cell name and value as priority
    """
    total = embedding.shape[0]
    steps = np.unique(np.round((np.arange(1, total, total / n_steps)).astype(int)))
    embedding_remote = ray.put(embedding.values)

    futures = []
    for s in steps:
        f = _gs_parallel.remote(embedding_remote, s)
        futures.append(f)

    idx_col = []
    for idx in ray.get(futures):
        idx_col.extend(idx)
    counts = pd.Series(idx_col).value_counts()

    priority = pd.Series(counts.values, index=embedding.index[counts.index])
    priority = priority.reindex(embedding.index).fillna(0).astype(int)
    return priority


def _merge_pseudobulk(
    cell_embedding, pseudobulk_to_minibulk, minibulk_to_cells, max_overlap, n_jobs
):
    minibulk_embedding = cell_embedding.loc[pseudobulk_to_minibulk.keys()]
    priority = geosketch_prioritize(minibulk_embedding)

    final_pseudobulk_to_minibulk = {}
    remaining_minibulks = pd.Index(minibulk_to_cells.keys())
    while priority.size > 0:
        # select pseudobulk to add to final collection based on priority
        _pseudobulk_cell = priority.idxmax()
        _minibulk_cells = pseudobulk_to_minibulk[_pseudobulk_cell]
        final_pseudobulk_to_minibulk[_pseudobulk_cell] = _minibulk_cells
        remaining_minibulks = remaining_minibulks.difference(_minibulk_cells)

        # remove pseudobulk cells that have > max_overlap with the current pseudobulk
        # overlap distance is calculated between their minibulk cells
        for _cell in priority.index.tolist():
            _temp_minibulk_cells = pseudobulk_to_minibulk[_cell]
            _cells = set()
            for _minibulk in _minibulk_cells:
                _cells.update(minibulk_to_cells[_minibulk])
            _temp_cells = set()
            for _minibulk in _temp_minibulk_cells:
                _temp_cells.update(minibulk_to_cells[_minibulk])
            dist = _overlap_distance(_cells, _temp_cells)
            if dist < (1 - max_overlap):
                priority.drop(_cell, inplace=True)

    # assign all remaining mini-bulks to its closest final pseudobulk
    _remain_minibulks_embedding = minibulk_embedding.loc[remaining_minibulks]
    _pseudobulk_cells = pd.Index(final_pseudobulk_to_minibulk.keys())
    _pseudobulk_embedding = minibulk_embedding.loc[_pseudobulk_cells]
    _remain_minibulks_to_pseudobulk_nn_idx, _, _ = _run_nndescent(
        train_data=_pseudobulk_embedding.values,
        query_data=_remain_minibulks_embedding.values,
        k=1,
        n_jobs=n_jobs,
        max_dist_q=None,
    )
    for _cell, _nn_idx in zip(
        remaining_minibulks, _remain_minibulks_to_pseudobulk_nn_idx
    ):
        final_pseudobulk_to_minibulk[_pseudobulk_cells[_nn_idx[0]]].add(_cell)

    final_pseudobulk = {}
    # cells assigned to the current pseudobulk
    for _pseudobulk_cell, _minibulk_cells in final_pseudobulk_to_minibulk.items():
        cells = []
        for _cell in _minibulk_cells:
            cells.extend(minibulk_to_cells[_cell])
        cells = set(cells)
        final_pseudobulk[_pseudobulk_cell] = cells
    return final_pseudobulk, final_pseudobulk_to_minibulk


def get_pseudobulk_adata(
    adata,
    obsm="X_pca",
    min_cells=1500,
    min_frags=5e6,
    frag_col=None,
    oversample=2,
    n_jobs=-1,
    max_dist_q=None,
    max_overlap=0.1,
    restrictions=None,
    save_internal=False,
):
    """
    Generate cell pseudobulk for adata

    The pseudobulk generation algorithm is based on two-step pseudobulk generation algorithm
    let k_sqrt = sqrt(pseudobulk_size), then the algorithm is as follows:
    1. run geosketch to select k_sqrt of sketch cells, run NNDescent on sketch cells only.
    2. run geosketch to select n_pseudobulk of pseudobulk sketch cells, merge sketch cells based on their NN into n_pseudobulk.
    3. run NNDescent on all cells, and query the sketch cells to generate mini-psuedobulk,
       in this step, far cells with large distance will be masked, which is determined by the max_dist_q parameter.
    4. generate pseudobulk by merging mini-pseudobulks based on their sketch cell overlap.
    5. the raw pseudobulk will be merged based on their overlap distance and hierarchical clustering,
       the merge_cutoff parameter is used to control the merging process.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    obsm : str
        Key in adata.obsm to use for pseudobulk generation
    min_cells : int
        Minimum number of cells in each pseudobulk
    min_frags : float
        Minimum number of fragments in each pseudobulk, this is calculated on average level, not on each individual pseudobulk
    frag_col : str
        Key in adata.obs to use for fragment count, if None, min_frags will be ignored
    oversample : float
        Oversample number of pseudobulks to cover more unique cells
    n_jobs : int
        Number of jobs to run in pyNNDescent
    max_dist_q : float
        Maximum Gamma quantile for masking far neighbors
    merge_cutoff : float
        Distance cutoff for merging pseudobulks, the larger the value, the more pseudobulks will be merged
    restrictions : str or list
        First group all cells based on restriction columns (in adata.obs), then generate pseudobulk for each group.
        if None, generate pseudobulk for all cells without restrictions.
    save_internal : bool
        Save internal pseudobulk to minibulk mapping and minibulk to cell mapping for visualization and troubleshooting

    Returns
    -------
    adata.uns[name] : dict
        Pseudobulk to cells map
    """
    _trigger_jit()

    total_cell_embedding = pd.DataFrame(adata.obsm[obsm], index=adata.obs_names)
    cell_embedding_dict = {}
    if restrictions is None:
        cell_embedding_dict["pseudobulk"] = total_cell_embedding
    else:
        if isinstance(restrictions, str):
            restrictions = [restrictions]
        assert all(
            r in adata.obs.columns for r in restrictions
        ), "Not all restrictions are in adata.obs.columns"
        for group, cells in adata.obs.groupby(restrictions):
            name = "-".join(group)
            cell_embedding_dict[name] = total_cell_embedding.loc[cells.index]

    total_psuedobulk = {}
    pseudobulk_to_minibulk = {}
    minibulk_to_cells = {}
    for name, cell_embedding in cell_embedding_dict.items():
        if frag_col is not None:
            cell_ave_frags = adata.obs[frag_col].mean()
            pseudo_bulk_size = max(min_frags / cell_ave_frags, min_cells)
        else:
            pseudo_bulk_size = min_cells
        pseudo_bulk_size = min(int(pseudo_bulk_size), int(min_cells * 3))

        n_pseudobulk = np.round(
            cell_embedding.shape[0] / pseudo_bulk_size * oversample
        ).astype(int)
        if n_pseudobulk <= 1:
            total_psuedobulk[f"{name}_pseudobulk0"] = set(cell_embedding.index)
        else:
            _pseudobulk_to_minibulk, _minibulk_to_cells = _two_step_pseudobulk(
                embedding=cell_embedding,
                k=pseudo_bulk_size,
                oversample=oversample,
                n_jobs=n_jobs,
                max_dist_q=max_dist_q,
            )
            _merged_pseudobulk, _pseudobulk_to_minibulk = _merge_pseudobulk(
                cell_embedding=cell_embedding,
                pseudobulk_to_minibulk=_pseudobulk_to_minibulk,
                minibulk_to_cells=_minibulk_to_cells,
                max_overlap=max_overlap,
                n_jobs=n_jobs,
            )
            total_psuedobulk.update(
                {
                    f"{name}_pseudobulk{i}": v
                    for i, (_, v) in enumerate(_merged_pseudobulk.items())
                }
            )
            pseudobulk_to_minibulk.update(_pseudobulk_to_minibulk)
            minibulk_to_cells.update(_minibulk_to_cells)

    # merge pseudobulk, priority is calculated
    adata.uns["pseudobulk"] = total_psuedobulk
    if save_internal:
        adata.uns["pseudobulk_to_minibulk"] = pseudobulk_to_minibulk
        adata.uns["minibulk_to_cells"] = minibulk_to_cells
    return


def plot_pseudobulk_qc(adata, coord_base, restrictions=None):
    """
    Plot pseudobulk quality control (QC) metrics.

    Parameters
    ----------
        adata (AnnData): Annotated data matrix.
        coord_base (str): Key for the coordinates in `adata.obsm` to be plotted.
        restrictions (str or list, optional): Key(s) for the categorical variable(s) in `adata.obs` to be used for restricting the plot. Defaults to None.

    Returns
    -------
        fig (matplotlib.figure.Figure): The created figure.
        axes (numpy.ndarray): Array of axes objects.

    """
    pseudobulk_to_cells = adata.uns["pseudobulk"]
    pseudobulk_to_minibulk = adata.uns["pseudobulk_to_minibulk"]
    minibulk_to_cells = adata.uns["minibulk_to_cells"]
    adata.obs["x"] = adata.obsm[coord_base][:, 0]
    adata.obs["y"] = adata.obsm[coord_base][:, 1]

    fig, axes = plt.subplots(figsize=(12, 3), ncols=4, dpi=200)

    _cl = [list(pseudobulk_to_minibulk.keys()), list(minibulk_to_cells.keys())]
    titles = [
        f"{len(pseudobulk_to_minibulk)} pseudobulk seeds",
        f"{len(minibulk_to_cells)} mini-bulk seeds",
    ]
    for cells, ax, title in zip(_cl, axes, titles):
        sns.scatterplot(
            ax=ax,
            data=adata.obs,
            x="x",
            y="y",
            s=1,
            linewidth=0,
            color="grey",
            rasterized=True,
        )
        sns.scatterplot(
            ax=ax,
            data=adata.obs[adata.obs_names.isin(cells)],
            x="x",
            y="y",
            s=3,
            linewidth=0,
            color="red",
            rasterized=True,
        )
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    ax = axes[2]
    ax.set_title("Restrictions", fontsize=8)
    if restrictions is not None:
        if isinstance(restrictions, str):
            restrictions = [restrictions]
        _cate = adata.obs[restrictions].apply(lambda x: "-".join(x), axis=1)
    else:
        _cate = None
    sns.scatterplot(
        ax=ax,
        data=adata.obs,
        x="x",
        y="y",
        s=1,
        linewidth=0,
        hue=_cate,
        legend=None,
        rasterized=True,
    )
    ax.axis("off")

    cell_include_count = pd.Series(0, index=adata.obs_names)
    for v in pseudobulk_to_cells.values():
        cell_include_count.loc[list(v)] += 1
    ax = axes[3]
    ax.set_title("Cell inclusion count", fontsize=8)
    sns.scatterplot(
        ax=ax,
        data=adata.obs,
        x="x",
        y="y",
        s=0.1,
        linewidth=0,
        hue=cell_include_count,
        legend=None,
        hue_norm=(0, 5),
        rasterized=True,
    )
    ax.axis("off")
    return fig, axes


def plot_pseudobulks(adata, coord_base, nrows=5, ncols=5):
    """
    Plot pseudobulks.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    coord_base : str
        Key for the basis in `adata.obsm` to use for plotting.
    nrows : int, optional
        Number of rows in the subplot grid. Default is 5.
    ncols : int, optional
        Number of columns in the subplot grid. Default is 5.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : numpy.ndarray
        Array of Axes objects representing the subplots.

    """
    pseudobulk_to_cells = adata.uns["pseudobulk"]
    adata.obs["x"] = adata.obsm[coord_base][:, 0]
    adata.obs["y"] = adata.obsm[coord_base][:, 1]

    sel_pseudobulks = np.random.choice(
        list(pseudobulk_to_cells.keys()),
        min(len(pseudobulk_to_cells), 64),
        replace=False,
    )

    fig, axes = plt.subplots(
        figsize=(ncols * 3, nrows * 3), dpi=200, ncols=ncols, nrows=nrows
    )

    for name, ax in zip(sel_pseudobulks, axes.ravel()):
        sns.scatterplot(
            ax=ax,
            data=adata.obs,
            x="x",
            y="y",
            s=1,
            linewidth=0,
            color="grey",
            rasterized=True,
        )
        sns.scatterplot(
            ax=ax,
            data=adata.obs[adata.obs_names.isin(pseudobulk_to_cells[name])],
            x="x",
            y="y",
            s=2,
            linewidth=0,
            color="red",
            rasterized=True,
        )
        ax.axis("off")
        ax.set_title(name, fontsize=8)
    return fig, axes


def plot_pseudobulk_overlaps(adata):
    """
    Plot the pseudobulk overlaps as a clustermap.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.

    Returns
    -------
    g : seaborn.ClusterGrid
        The clustermap object representing the pseudobulk overlaps.
    """
    pseudobulk_to_cells = adata.uns["pseudobulk"]

    n = len(pseudobulk_to_cells)
    overlap = np.zeros(shape=(n, n))
    for i, a in enumerate(adata.uns["pseudobulk"].values()):
        for j, b in enumerate(adata.uns["pseudobulk"].values()):
            d = _overlap_distance(a, b)
            overlap[i, j] = d
    overlap = pd.DataFrame(
        overlap, index=pseudobulk_to_cells.keys(), columns=pseudobulk_to_cells.keys()
    )
    g = sns.clustermap(overlap)
    return g
