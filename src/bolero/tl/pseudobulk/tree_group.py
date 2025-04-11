from itertools import product

import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree


def _gather_node_info(node, node_dict, leaf_cell_counts):
    """
    Gather information about the tree nodes recursively.
    """
    if node.is_leaf():
        node_dict[node.id] = {
            "left": None,
            "right": None,
            "child_list": [node.id],
            "cell_count": leaf_cell_counts[node.id],
        }
    else:
        _gather_node_info(node.left, node_dict, leaf_cell_counts)
        _gather_node_info(node.right, node_dict, leaf_cell_counts)

        left_id = node.left.id
        right_id = node.right.id
        child_list = (
            node_dict[left_id]["child_list"] + node_dict[right_id]["child_list"]
        )
        cell_count = (
            node_dict[left_id]["cell_count"] + node_dict[right_id]["cell_count"]
        )

        node_dict[node.id] = {
            "left": left_id,
            "right": right_id,
            "child_list": child_list,
            "cell_count": cell_count,
        }


def _top_down_partition(node_id, node_dict, threshold):
    """
    Recursively partition the tree into clusters until either side of the branch has cells < threshold.
    """
    info = node_dict[node_id]

    # If leaf, it's a final cluster
    if info["left"] is None and info["right"] is None:
        return [node_id]

    left_id = info["left"]
    right_id = info["right"]
    left_count = node_dict[left_id]["cell_count"]
    right_count = node_dict[right_id]["cell_count"]

    # If either child is below threshold, stop splitting
    if left_count < threshold or right_count < threshold:
        return [node_id]
    else:
        # Otherwise, keep splitting both children
        clusters_left = _top_down_partition(left_id, node_dict, threshold)
        clusters_right = _top_down_partition(right_id, node_dict, threshold)
        return clusters_left + clusters_right


def _tree_based_cat_split(embedding, cell_counts, min_cell_count=200, method="ward"):
    """
    Tree-based categorical split for cells based on their embeddings.
    This function uses hierarchical clustering to partition the cells into groups
    based on their similarity in the embedding space and the specified minimum cell count.
    """
    # 1) Build tree
    Z = linkage(embedding.values, method="ward")
    root_node = to_tree(Z, rd=False)

    # 2) Build leaf_cell_counts
    leaf_cell_counts = {i: cell_counts[c] for i, c in enumerate(embedding.index)}

    # 3) Gather node info
    node_dict = {}
    _gather_node_info(root_node, node_dict, leaf_cell_counts)

    # 4) Top-down partitioning
    final_cluster_ids = _top_down_partition(root_node.id, node_dict, min_cell_count)

    # 5) Get the final clusters
    # cat_groups [list of lists] of cell ids in embedding.index
    cat_groups = [node_dict[cid]["child_list"] for cid in final_cluster_ids]
    cats = embedding.index.tolist()
    cat_groups = [[cats[i] for i in gl] for gl in cat_groups]
    return cat_groups


def _flatten_dict(d, prefix=()):
    """
    Make hierarchical dict to flat list of tuples.
    """
    result = []
    for key, value in d.items():
        new_prefix = prefix + (key,)
        if isinstance(value, dict):
            result.extend(_flatten_dict(value, new_prefix))
        else:
            result.append(new_prefix + (value,))
    return result


def prepare_multi_level_categorical_groups(
    cell_meta,
    embedding,
    group_cols,
    min_cell_count=500,
    dendrogram_method="ward",
    _is_top=True,
):
    """
    Group multiple levels of categorical variables into a single categorical variable
    by using a tree-based approach on cell embedding, with a minimum cell count threshold.

    Specifically, this function:
    1. Iteratively groups cells based on the each level of categorical variables.
    2. Within each level, it uses hierarchical clustering on group average embeddings
       to split data into groups if the resulting group size passes the minimum cell count threshold.
    3. It returns a final cell-to-group mapping and a dictionary of group-to-categorical mappings.

    Parameters
    ----------
    cell_meta : pd.DataFrame
        DataFrame containing cell metadata with categorical variables.
    embedding : pd.DataFrame
        DataFrame containing cell embeddings. Index should match with cell_meta.
    group_cols : list
        List of categorical variable names to group by, in order of priority.
    min_cell_count : int
        Minimum number of cells required in a group for further splitting.
    dendrogram_method : str
        Method to use for hierarchical clustering. Default is 'ward'.
    _is_top : bool
        Flag to indicate if this is the top-level call. Do not set this manually.
        It is used internally for recursive calls.

    Returns
    -------
    cell_to_group : pd.Series
        Series mapping each cell to its corresponding group name.
    group_to_cells : dict
        Dictionary mapping group names to lists of cell indices.
    """
    if _is_top:
        print(f"Cell metadata has {cell_meta.shape[0]} cells")
        print(f"Cell embedding has {embedding.shape[0]} cells")
        cell_meta = cell_meta.dropna(subset=group_cols)
        cells = cell_meta.index.intersection(embedding.index)
        print(f"Intersection index has {cells.size} cells")
        cell_meta = cell_meta.reindex(cells)
        embedding = embedding.reindex(cells)

    level = group_cols[0]
    remain_group_cols = group_cols[1:]
    group_emb = embedding.groupby(cell_meta[level], observed=True).mean()
    cell_counts = cell_meta.value_counts(level).reindex(group_emb.index)

    if group_emb.shape[0] > 1:
        groups = _tree_based_cat_split(
            embedding=group_emb,
            cell_counts=cell_counts,
            min_cell_count=min_cell_count,
            method=dendrogram_method,
        )
    else:
        # only one group, no need to merge
        groups = [group_emb.index.tolist()]

    if len(remain_group_cols) > 0:
        group_to_cells = {}
        for group in groups:
            use_cells = cell_meta[level].isin(group)
            key = tuple(group)

            group_to_cells[key] = prepare_multi_level_categorical_groups(
                cell_meta.loc[use_cells].copy(),
                embedding.loc[use_cells].copy(),
                group_cols=remain_group_cols,
                _is_top=False,
            )
    else:
        group_to_cells = {
            tuple(group): cell_meta.index[cell_meta[level].isin(group)].copy()
            for group in groups
        }
    to_return = group_to_cells

    if _is_top:
        group_to_cells = _flatten_dict(group_to_cells)

        cell_to_group = []
        group_to_cats = {}
        for group_id, (*cats, cells) in enumerate(group_to_cells):
            gname = f"group{group_id}"
            cell_to_group.append(pd.Series([gname] * cells.size, index=cells))
            group_to_cats[gname] = cats
        cell_to_group = pd.concat(cell_to_group).astype("category")
        to_return = cell_to_group, group_to_cats
    return to_return


def get_cell_to_group(
    cell_metadata: pd.DataFrame,
    groupby_cols: list,
    group_to_categories: dict[list],
):
    """
    Create cell to group mapping based on group categories and cell metadata.

    Parameters
    ----------
    cell_metadata : pd.DataFrame
        DataFrame containing cell metadata with categorical variables.
    groupby_cols : list
        List of categorical variable names to group by.
    group_to_categories : dict[list]
        Group to category combination created by the prepare_multi_level_categorical_groups function.
        The keys are group names and the values are lists of categories.

    Returns
    -------
    cell_to_group : pd.Series
        Series mapping each cell to its corresponding group name.
    """
    group_to_cells = {
        k: v.index for k, v in cell_metadata.groupby(groupby_cols, observed=True)
    }

    cell_to_group = []
    for group, cats in group_to_categories.items():
        for cond in product(*cats):
            cells = group_to_cells.get(cond, None)
            if cells is not None:
                gs = pd.Series(group, cells)
                cell_to_group.append(gs)
    cell_to_group = pd.concat(cell_to_group)
    return cell_to_group
