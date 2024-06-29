import gzip
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack


def compressed_bytes_to_array(bytes: bytes, dtype: str) -> np.ndarray:
    """
    Decompress bytes and convert to numpy array.

    Parameters
    ----------
    bytes : bytes
        The compressed bytes to be decompressed.
    dtype : str
        The data type of the resulting numpy array.

    Returns
    -------
    np.ndarray
        The decompressed numpy array.
    """
    return np.frombuffer(gzip.decompress(bytes), dtype=dtype)


class CompressedBytesToTensor:
    def __init__(self):
        """
        Convert all the prefix dataset into tensor (csr_matrix or numpy.ndarray).

        Two types of keys are expected in the input data_dict:

        1. csr_matrix:
        Each prefix should have four keys, which are:
        - "{prefix}:data+float32"
        - "{prefix}:indices+uint32"
        - "{prefix}:indptr+uint32"
        - "{prefix}:shape+uint32"
        2. numpy.ndarray:
        Each prefix should have two keys, which are:
        - "{prefix}:data+float32"
        - "{prefix}:shape+uint32"

        The value of these keys are gzip compressed bytes.

        The output will be a dict of ndarray or csr_matrix with shape of (row, base), key is the prefix, original keys will be removed.
        Row id is in the original order, recorded in dataset_dir/row_names.joblib.
        """
        self.prefixs = set()
        return

    def _bytes_to_array(self, data_dict):
        bytes_keys = [k for k, v in data_dict.items() if isinstance(v, bytes)]
        for key in bytes_keys:
            prefix, name_and_dtype = key.split(":")
            name, dtype = name_and_dtype.split("+")
            data_dict[f"{prefix}:{name}"] = compressed_bytes_to_array(
                data_dict.pop(key), dtype=dtype
            )
            self.prefixs.add(prefix)
        return data_dict

    def _make_tensor(self, data_dict):
        for prefix in self.prefixs:
            data = data_dict.pop(f"{prefix}:data")
            shape = data_dict.pop(f"{prefix}:shape")
            try:
                indices = data_dict.pop(f"{prefix}:indices")
                indptr = data_dict.pop(f"{prefix}:indptr")
                _data = csr_matrix((data, indices, indptr), shape=shape)
            except KeyError:
                _data = data.reshape(shape)
            data_dict[prefix] = _data
        return data_dict

    def __call__(self, data_dict: Dict[str, bytes]) -> Dict[str, np.ndarray]:
        """Perform the transformation."""
        # for each raw data key in binary format:
        #     input data is stored in bytes
        #     this function turn all the bytes data into numpy array
        data_dict = self._bytes_to_array(data_dict)

        # for each prefix:
        #     the parts of csr_matrix or ndarray is pop out and stored back as a complete tensor
        data_dict = self._make_tensor(data_dict)
        return data_dict


class GeneratePseudobulk:
    """
    Transform meta region data into bulk region data.
    """

    def __init__(
        self,
        n_pseudobulks=10,
        return_rows=False,
        inplace=False,
        bypass_keys=None,
        **name_to_pseudobulker,
    ):
        self.name_to_pseudobulker = name_to_pseudobulker
        self.n_pseudobulks = n_pseudobulks
        self.return_rows = return_rows
        self.inplace = inplace

        self.bypass_keys = ["region"]
        if bypass_keys is not None:
            if bypass_keys is str:
                self.bypass_keys.append(bypass_keys)
            else:
                self.bypass_keys.extend(list(bypass_keys))
        self._input_prefix = set()
        return

    def _get_pseudo_bulks(self, data_dict, output_prefix, pseudobulker):
        bulk_data_dict = {}

        _per_prefix_bulk_data = defaultdict(list)
        embedding_data = []
        rows_col = []
        pseudobulk_ids = []
        # merge rows (cell or sample) to bulk and also get embedding data
        for bulk_idx, (
            rows,  # rows is pd.Index
            prefix_to_rows,
            row_embedding,
            pseudobulk_id,
        ) in enumerate(pseudobulker.take(self.n_pseudobulks)):
            embedding_data.append(row_embedding)
            pseudobulk_ids.append(pseudobulk_id)
            rows_col.append(rows)
            found_row_count = 0
            for prefix, prefix_rows in prefix_to_rows.items():
                self._input_prefix.add(prefix)
                # prefix_rows is bool array
                # some pseudo-bulks may not have any rows for a prefix
                found_n = prefix_rows.sum()
                if found_n == 0:
                    continue
                found_row_count += found_n

                # row_by_base is a csr_matrix of shape (n_rows, region_length)
                row_by_base = data_dict.get(prefix, None)
                if row_by_base is None:
                    print(f"Prefix {prefix} not found in data_dict")
                    continue

                _bulk_values = csr_matrix(row_by_base[prefix_rows].sum(axis=0).A1)
                _per_prefix_bulk_data[bulk_idx].append(_bulk_values)

            # check if all rows is found, otherwise print warning
            if found_row_count != len(rows):
                example_rows = list(rows)[:5]
                print(
                    f"Not all rows found for bulk {pseudobulk_id}, this might be due to prefix or row id mismatch. "
                    f"Rows in pseudobulk: {len(rows)}, Rows found: {found_row_count}, Example row ids: {example_rows}"
                )

        embedding_data = np.array(
            embedding_data, dtype=np.float32
        )  # shape: n_pseudobulks x n_features
        pseudobulk_ids = np.array(pseudobulk_ids)  # shape: n_pseudobulks

        # pseudobulks maybe less than self.n_pseudobulks
        actual_n_pseudobulks = embedding_data.shape[0]
        bulk_data = []
        for bulk_idx in range(actual_n_pseudobulks):
            bulk_data_list = _per_prefix_bulk_data[bulk_idx]
            if len(bulk_data_list) == 0:
                example_rows = list(rows_col[bulk_idx])[:5]
                raise ValueError(
                    f"No rows for bulk {bulk_idx}, this might be due to prefix or row id mismatch. "
                    f"Example rows: {example_rows}"
                )
            agg_bulk = csr_matrix(
                vstack(_per_prefix_bulk_data[bulk_idx]).sum(axis=0).A1
            )
            bulk_data.append(agg_bulk)
        bulk_data = vstack(bulk_data)
        bulk_data_dict[f"{output_prefix}:bulk_data"] = bulk_data
        bulk_data_dict[f"{output_prefix}:embedding_data"] = embedding_data
        bulk_data_dict[f"{output_prefix}:pseudobulk_ids"] = pseudobulk_ids
        if self.return_rows:
            bulk_data_dict[f"{output_prefix}:rows"] = rows_col
        return bulk_data_dict, actual_n_pseudobulks

    def __call__(self, data_dict: Dict[str, bytes]) -> List[Dict[str, np.ndarray]]:
        """Generate pseudobulks for each output prefix."""
        if self.inplace:
            bulk_data_col = data_dict
        else:
            # only copy the region info
            bulk_data_col = {}

        for output_prefix, pseudobulker in self.name_to_pseudobulker.items():
            bulk_data_dict, actual_n_pseudobulks = self._get_pseudo_bulks(
                data_dict=data_dict,
                output_prefix=output_prefix,
                pseudobulker=pseudobulker,
            )
            bulk_data_col.update(bulk_data_dict)

        list_of_dicts = []
        for i in range(actual_n_pseudobulks):
            _dict = {k: v[i] for k, v in bulk_data_col.items()}
            for key in self.bypass_keys:
                # repeat shared data for each output pseudobulk
                if key in data_dict:
                    _dict[key] = deepcopy(data_dict[key])
            list_of_dicts.append(_dict)
        return list_of_dicts


class GenerateRegions:
    def __init__(
        self,
        bed,
        meta_region_overlap,
        action_keys,
    ):
        self.meta_region_overlap = meta_region_overlap

        assert isinstance(bed, pd.DataFrame), "bed should be a pandas DataFrame"
        assert bed.columns[:3].tolist() == ["Chromosome", "Start", "End"]
        self.bed: pd.DataFrame = bed

        self.action_keys = action_keys
        return

    def _select_relevant_regions(self, data_dict):
        dict_region = data_dict.pop("region")
        chrom, coords = dict_region.split(":")
        start, end = map(int, coords.split("-"))

        use_bed = self.bed[
            (self.bed["Chromosome"] == chrom)
            & (self.bed["Start"] >= start)
            & (self.bed["Start"] <= end - self.meta_region_overlap)
            & (self.bed["End"] <= end)
        ]
        offset = start
        return use_bed, offset

    def __call__(self, data_dict: Dict[str, bytes]) -> List[Dict[str, np.ndarray]]:
        """Generate regions for each meta region."""
        use_bed, offset = self._select_relevant_regions(data_dict)

        list_of_dicts = []
        for _, (chrom, start, end, *_) in use_bed.iterrows():
            data_col = {}
            data_col["region"] = f"{chrom}:{start}-{end}"
            for key, value in data_dict.items():
                if key in self.action_keys:
                    rstart = start - offset
                    rend = end - offset
                    rvalue = value[..., rstart:rend]
                    try:
                        rvalue = rvalue.toarray()
                    except AttributeError:
                        rvalue = rvalue.copy()
                    data_col[key] = rvalue
                else:
                    data_col[key] = deepcopy(value)
            list_of_dicts.append(data_col)
        return list_of_dicts


class FilterRegions:
    def __init__(
        self, cov_filter_key, min_cov=10, max_cov=1e5, low_cov_ratio=0.1, cov_func=None
    ):
        self.cov_filter_key = cov_filter_key
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio
        if cov_func is None:

            def _sum_all(data):
                # sum over all dims except the first one
                return data.sum(axis=tuple(range(1, data.ndim)))

            self.cov_func = _sum_all
        else:
            self.cov_func = cov_func
        return

    def __call__(self, batch: dict):
        """Filter regions based on coverage."""
        data = batch[self.cov_filter_key]

        region_sum = self.cov_func(data)

        use_rows = (region_sum > self.min_cov) & (region_sum < self.max_cov)

        # add some low coverage regions as negative samples
        low_cov_rows = np.where(region_sum <= self.min_cov)[0]
        choice_n = min(int(use_rows.sum() * self.low_cov_ratio), low_cov_rows.shape[0])
        choice_rows = np.random.choice(low_cov_rows, choice_n, replace=False)
        use_rows[choice_rows] = True

        if use_rows.sum() == 0:
            # keep at least one region
            use_rows[0] = True

        # apply filter to all keys
        batch = {
            k: v[use_rows, ...].copy()  # if v.ndim > 1 else v[use_rows]
            for k, v in batch.items()
        }
        return batch
