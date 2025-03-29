import numpy as np
import pandas as pd

try:
    import SEACells
except ImportError as e:
    raise ImportError(
        "SEACells is not installed. Please install it from https://github.com/dpeerlab/SEACells."
    ) from e


def calc_meta_cells(adata, meta_fold, obsm, n_waypoint_eigs=10):
    """
    Meta cell calculation using SEACells.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the data.
    meta_fold : int
        The number of folds to group cells into.
    obsm : str
        The key in adata.obsm to use for the kernel matrix.
    n_waypoint_eigs : int
        The number of waypoint eigenvectors to use.
    """
    n_cells = adata.shape[0]
    n_meta_cells = max(n_cells // meta_fold, 10)
    print(f"Group {n_cells} cells to {n_meta_cells} meta cells")
    model = SEACells.core.SEACells(
        adata,
        build_kernel_on=obsm,
        n_SEACells=n_meta_cells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=False,
        convergence_epsilon=1e-5,
        verbose=False,
    )
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=50)
    assign = model.get_hard_assignments()
    return model, assign


def run_meta_cells(
    adata,
    obsm,
    groupby=None,
    max_fragments=3000000,
    n_fragment_key="n_fragments",
    meta_fold=75,
):
    """
    Run meta cell calculation on the given AnnData object.
    This function groups cells by the specified groupby key and calculates meta cells for each group.
    It also handles the case where the number of fragments exceeds the specified maximum number of fragments.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the data.
    groupby : str
        The key in adata.obs to group cells by.
    obsm : str
        The key in adata.obsm to use for the kernel matrix.
    max_fragments : int
        The maximum number of fragments allowed for each meta cell.
    n_fragment_key : str
        The key in adata.obs to use for the number of fragments.
    meta_fold : int
        The number of folds to group cells into.

    Returns
    -------
    pd.Series
        A series containing the meta cell assignments for each cell.
    """
    if groupby is None:
        _, assign = calc_meta_cells(adata, meta_fold, obsm)
        total_assign = assign["SEACell"]
    else:
        total_assign = []
        for group, sub_df in adata.obs.groupby(groupby, observed=True):
            ct_adata = adata[sub_df.index].copy()
            _, assign = calc_meta_cells(ct_adata, meta_fold, obsm)
            assign = group + "+" + assign["SEACell"]
            total_assign.append(assign)
        total_assign = pd.concat(total_assign)

    if max_fragments is None:
        return total_assign

    # for meta cell with fragments > max_fragments,
    # random split into equal sized folds so the meta cell fragment drop below max_fragments
    split_assign = []
    for meta_cell, cell_frags in adata.obs[n_fragment_key].groupby(total_assign):
        split_fold = int(np.ceil(cell_frags.sum() / max_fragments))
        cell_frags = cell_frags.sample(frac=1, replace=False)
        if split_fold > 1:
            new_assign = pd.Series(
                {
                    c: f"{meta_cell}.{idx % split_fold}"
                    for idx, c in enumerate(cell_frags.index)
                }
            )
        else:
            new_assign = pd.Series(
                [meta_cell] * cell_frags.size, index=cell_frags.index
            )
        split_assign.append(new_assign)
    split_assign = pd.concat(split_assign)
    return split_assign
