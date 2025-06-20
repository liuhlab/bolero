import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

try:
    import SEACells
except ImportError as e:
    raise ImportError(
        "SEACells is not installed. Please install it from https://github.com/dpeerlab/SEACells."
    ) from e


def calc_meta_cells(
    adata, meta_fold, obsm, min_seacells=10, n_waypoint_eigs=10, verbose=False
):
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
    n_waypoint_eigs = min(min_seacells, n_waypoint_eigs)
    n_cells = adata.shape[0]
    n_meta_cells = max(n_cells // meta_fold, min_seacells)
    if verbose:
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
    try:
        model.fit(min_iter=10, max_iter=80)
    except RuntimeWarning:
        # seacell raise error when not converge
        # This seems to happen some times when the total population is very small
        print("SEACell not converge, using the last fit")
        pass
    assign = model.get_hard_assignments()
    return model, assign


def kmeans_split(
    df: pd.DataFrame, desired_cluster_size: int, random_state=42
) -> list[pd.DataFrame]:
    """
    Recursively split a cell-by-embedding dataframe into smaller chunks using KMeans,
    such that each chunk is roughly of the desired_cluster_size.

    Parameters
    ----------
    - df: pandas DataFrame, rows are cells, columns are embeddings
    - desired_cluster_size: int, approximate size for each final cluster
    - random_state: int, for reproducibility

    Returns
    -------
    - List of pandas DataFrames, each with rows <= desired_cluster_size * 2
    """
    if len(df) <= 2 * desired_cluster_size:
        return [df]

    # Apply KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(df.values)

    # Split based on labels
    df0 = df[labels == 0]
    df1 = df[labels == 1]

    # Recurse
    return kmeans_split(
        df0, desired_cluster_size, random_state=random_state
    ) + kmeans_split(df1, desired_cluster_size, random_state=random_state)


def run_meta_cells(
    adata,
    obsm,
    groupby=None,
    max_fragments=3000000,
    group_size_cutoff=200,
    min_seacells_per_group=5,
    large_group_split_threshold=3000,
    n_fragment_key="n_fragments",
    meta_fold=75,
    random_state=0,
    verbose=False,
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
    group_size_cutoff : int, optional
        If specified, groups with fewer cells than this cutoff will be skipped.
        If None, all groups will be processed regardless of size.
    min_seacells_per_group : int
        The minimum number of SEACells to use for each group. If the group has fewer cells, it will use that number.
        This prevents errors when a group is too small to form a meta cell.
    large_group_split_threshold : int
        If the group size is larger than this threshold, it will be randomly split into smaller groups.
    n_fragment_key : str
        The key in adata.obs to use for the number of fragments.
    meta_fold : int
        The number of folds to group cells into.

    Returns
    -------
    pd.Series
        A series containing the meta cell assignments for each cell.
    """
    if group_size_cutoff is None:
        group_size_cutoff = 1

    if groupby is None:
        _, assign = calc_meta_cells(adata, meta_fold, obsm)
        total_assign = assign["SEACell"]
    else:
        total_assign = []
        n_groups = adata.obs[groupby].nunique()
        for group, sub_df in tqdm(
            adata.obs.groupby(groupby, observed=True),
            total=n_groups,
            desc="Calculating meta cells",
        ):
            n_cells = sub_df.shape[0]
            if sub_df.shape[0] < group_size_cutoff:
                if verbose:
                    print(
                        f"Skipping group '{group}' with {n_cells} cells: "
                        f"not enough cells to form meta cells (need at least 2)."
                    )
                # If the group has fewer cells than needed for meta cells, use the group name as the assignment
                assign = pd.Series([group] * n_cells, index=sub_df.index)
                total_assign.append(assign)
                continue

            if n_cells > large_group_split_threshold * 2:
                # split the group into smaller groups using KMeans
                cell_emb = pd.DataFrame(
                    adata[sub_df.index].obsm[obsm].copy(), index=sub_df.index
                )
                cell_splits = kmeans_split(
                    cell_emb,
                    desired_cluster_size=large_group_split_threshold,
                    random_state=random_state,
                )
                if verbose:
                    print(
                        f"Splitting group '{group}' into {len(cell_splits)} smaller groups."
                    )

                for idx, split_df in enumerate(cell_splits):
                    split_cells = adata.obs_names.intersection(split_df.index)
                    ct_adata = adata[split_cells].copy()
                    _group = f"{group}.{idx}"
                    if verbose:
                        print(
                            f"Run meta cell for {_group} with adata shape {ct_adata.shape}"
                        )
                    _, assign = calc_meta_cells(
                        ct_adata,
                        meta_fold=meta_fold,
                        obsm=obsm,
                        min_seacells=min_seacells_per_group,
                    )
                    assign = _group + "+" + assign["SEACell"]
                    total_assign.append(assign)
            else:
                ct_adata = adata[sub_df.index].copy()
                if verbose:
                    print(
                        f"Run meta cell for {group} with adata shape {ct_adata.shape}"
                    )
                _, assign = calc_meta_cells(
                    ct_adata,
                    meta_fold=meta_fold,
                    obsm=obsm,
                    min_seacells=min_seacells_per_group,
                )
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
