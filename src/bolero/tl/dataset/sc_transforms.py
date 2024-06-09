import gzip
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
import ray
from scipy.sparse import csr_matrix, vstack

from bolero.tl.pseudobulk.generator import (
    PseudobulkGenerator,
)


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


class scMetaRegionToBulkRegion:
    """
    Transform meta region data into bulk region data.

    Args:
        prefixs (str or List[str]): The prefix or list of prefixes for the meta region data.
        pseudobulk_to_rows (Dict[str, Dict[str, List[int]]]): A dictionary that maps each prefix to a dictionary of pseudobulk names and corresponding row indices.
        sample_regions (int): The number of regions to randomly sample if there are too many regions.
        min_cov (int): The minimum coverage threshold for filtering regions.
        max_cov (float): The maximum coverage threshold for filtering regions.
        low_cov_ratio (float): The ratio of low coverage regions to spike.

    Returns
    -------
        List[Dict[str, np.ndarray]]: A list of dictionaries containing the bulk region data for each final region.

    """

    def __init__(
        self,
        prefixs,
        sample_regions=200,
        min_cov=10,
        max_cov=1e5,
        low_cov_ratio=0.1,
        n_pseudobulks=10,
        return_cells=False,
        **kwargs,
    ):
        if isinstance(prefixs, str):
            prefixs = [prefixs]
        self.prefixs = prefixs

        if "bigwig" in self.prefixs:
            self.bigwig_prefix = "bigwig"
            self.prefixs = [p for p in self.prefixs if p != self.bigwig_prefix]
        else:
            self.bigwig_prefix = None

        self.pseudobulker: PseudobulkGenerator = (
            PseudobulkGenerator.prepare_pseudobulker(**kwargs)
        )
        self.n_pseudobulks = n_pseudobulks

        self.sample_regions = sample_regions
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio
        self.return_cells = return_cells
        return

    def _bytes_to_array(self, data_dict):
        bytes_keys = [k for k, v in data_dict.items() if isinstance(v, bytes)]
        for key in bytes_keys:
            prefix, name_and_dtype = key.split(":")
            name, dtype = name_and_dtype.split("+")
            data_dict[f"{prefix}:{name}"] = compressed_bytes_to_array(
                data_dict.pop(key), dtype=dtype
            )
        return data_dict

    def _make_csr(self, data_dict):
        _prefixs = self.prefixs.copy()
        if self.bigwig_prefix is not None:
            _prefixs.append(self.bigwig_prefix)

        for prefix in _prefixs:
            data = data_dict.pop(f"{prefix}:data")
            indices = data_dict.pop(f"{prefix}:indices")
            indptr = data_dict.pop(f"{prefix}:indptr")
            shape = data_dict.pop(f"{prefix}:shape")
            csr_data = csr_matrix((data, indices, indptr), shape=shape)
            data_dict[prefix] = csr_data
        return data_dict

    def _get_pseudo_bulks(self, data_dict, return_cells=False):
        _per_prefix_bulk_data = defaultdict(list)
        embedding_data = []
        cells_col = {}
        pseudobulk_ids = []
        # merge single cell to bulk and also get embedding data
        for bulk_idx, (
            cells,
            prefix_to_rows,
            cell_embedding,
            pseudobulk_id,
        ) in enumerate(self.pseudobulker.take(self.n_pseudobulks)):
            embedding_data.append(cell_embedding)
            pseudobulk_ids.append(pseudobulk_id)
            cells_col[bulk_idx] = cells
            for prefix in self.prefixs:
                cell_by_base = data_dict[prefix]
                rows = prefix_to_rows.get(prefix, [])

                # some pseudo-bulks may not have any cells for a prefix
                if len(rows) == 0:
                    continue

                _bulk_values = csr_matrix(cell_by_base[rows].sum(axis=0).A1)
                _per_prefix_bulk_data[bulk_idx].append(_bulk_values)
                # TODO: check if all cell is found, otherwise print warning
        embedding_data = np.array(embedding_data)
        pseudobulk_ids = np.array(pseudobulk_ids)

        # remove prefix csr_matrix from data_dict
        for prefix in self.prefixs:
            data_dict.pop(prefix)

        # pseudobulks maybe less than self.n_pseudobulks
        actual_n_pseudobulks = embedding_data.shape[0]
        bulk_data = []
        for bulk_idx in range(actual_n_pseudobulks):
            bulk_data_list = _per_prefix_bulk_data[bulk_idx]
            if len(bulk_data_list) == 0:
                example_cells = list(cells_col[bulk_idx])[:5]
                raise ValueError(
                    f"No cells for bulk {bulk_idx}, this might be due to prefix or cell id mismatch. "
                    f"Example cells: {example_cells}"
                )
            agg_bulk = csr_matrix(
                vstack(_per_prefix_bulk_data[bulk_idx]).sum(axis=0).A1
            )
            bulk_data.append(agg_bulk)
        bulk_data = vstack(bulk_data)
        data_dict["bulk_data"] = bulk_data
        data_dict["embedding_data"] = embedding_data
        data_dict["pseudobulk_ids"] = pseudobulk_ids

        if return_cells:
            data_dict["cells"] = cells_col
        return data_dict

    def _get_bigwig_data(self, data_dict):
        if self.bigwig_prefix is None:
            return {}

        bigwig_data = data_dict.pop(self.bigwig_prefix)
        name_order = data_dict.pop(f"{self.bigwig_prefix}:name_order").split("|")

        meta_region = data_dict.pop(f"{self.bigwig_prefix}:meta_region")
        chrom, coords = meta_region.split(":")
        meta_start, _ = map(int, coords.split("-"))
        relative_coords = data_dict.pop(
            f"{self.bigwig_prefix}:relative_coords"
        ).reshape(-1, 2)
        _bigwig_region_dict = defaultdict(dict)
        for row, name in enumerate(name_order):
            for rstart, rend in relative_coords:
                region = f"{chrom}:{meta_start+rstart}-{meta_start+rend}"
                _bigwig_region_dict[region][name] = (
                    bigwig_data[row, rstart:rend].toarray().ravel()
                )
        return _bigwig_region_dict

    def _get_regions(self, data_dict, bigwig_region_dict):
        # separate meta region data into regions
        _example_prefix = self.prefixs[0]
        relative_coords = data_dict[f"{_example_prefix}:relative_coords"].reshape(-1, 2)
        chrom, coords = data_dict[f"{_example_prefix}:meta_region"].split(":")
        meta_start, _ = map(int, coords.split("-"))

        # random sample regions if there are too many regions
        sample_regions = self.sample_regions
        if relative_coords.shape[0] > sample_regions:
            idx = np.random.choice(
                relative_coords.shape[0], sample_regions, replace=False
            )
            relative_coords = relative_coords[idx]

        # get bulk data for each region
        bulk_data = data_dict.pop("bulk_data")
        final_data = []
        min_cov = self.min_cov
        max_cov = self.max_cov
        low_cov_ratio = self.low_cov_ratio
        for rstart, rend in relative_coords:
            _region_bulk_data = bulk_data[:, rstart:rend].toarray()
            _region_bulk_sum = _region_bulk_data.sum(axis=1)

            # filter by coverage
            _sel_coverage = (_region_bulk_sum > min_cov) & (_region_bulk_sum < max_cov)

            # spiking low coverage region
            low_cov_rows = np.where(_region_bulk_sum <= min_cov)[0]
            choice_n = min(
                int(_sel_coverage.sum() * low_cov_ratio), low_cov_rows.shape[0]
            )
            choice_rows = np.random.choice(low_cov_rows, choice_n, replace=False)
            _sel_spiking_rows = np.zeros(_region_bulk_data.shape[0], dtype=bool)
            _sel_spiking_rows[choice_rows] = True

            # filter by coverage or spiking low coverage region
            use_rows = (
                (_region_bulk_sum > min_cov) & (_region_bulk_sum < max_cov)
            ) | _sel_spiking_rows
            if use_rows.sum() == 0:
                continue

            _region_bulk_data = _region_bulk_data[use_rows]
            _region_embedding_data = data_dict["embedding_data"][use_rows].copy()
            _pseudobulk_ids_data = data_dict["pseudobulk_ids"][use_rows].copy()
            if "cells" in data_dict:
                use_row_idx = np.where(use_rows)[0]
                use_cells = [data_dict["cells"][idx] for idx in use_row_idx]
            else:
                use_cells = None

            for data, embedding, pseudobulk_id in zip(
                _region_bulk_data, _region_embedding_data, _pseudobulk_ids_data
            ):
                region = f"{chrom}:{meta_start+rstart}-{meta_start+rend}"
                _bw_data = bigwig_region_dict.get(region, {})
                _data = {
                    "bulk_embedding": embedding,
                    "bulk_id": pseudobulk_id,
                    "bulk_data": data,
                    "region": region,
                }
                _data.update(_bw_data)
                if use_cells is not None:
                    _data["cells"] = use_cells.pop(0)
                final_data.append(_data)
        return final_data

    def __call__(self, data_dict: Dict[str, bytes]) -> List[Dict[str, np.ndarray]]:
        """Perform the transformation."""
        # for each raw data key in binary format:
        #     input data is stored in bytes
        #     this function turn all the bytes data into numpy array
        data_dict = self._bytes_to_array(data_dict)

        # for each prefix:
        #     the part of csr_matrix is then pop out and stored back as a complete csr_matrix
        #     the shape of csr_matrix is (n_cells, meta_region_length) where n_cells is the number of cells in each prefix
        data_dict = self._make_csr(data_dict)

        # for each pseudobulk:
        #     we then make pseudo bulks for each prefix, resulting in a csr_matrix of shape (n_pseudobulks, meta_region_length)
        #     multiple prefix is also combined into a single bulk_data in this step
        data_dict = self._get_pseudo_bulks(data_dict, return_cells=self.return_cells)

        # also process bigwig data and split to individual regions
        # bigwig region data is saved in bigwig_region_dict, key is region name and value is a dict of bigwig data for each region
        # each returned dict will contain the bigwig data for corresponding region
        if self.bigwig_prefix is not None:
            bigwig_region_dict = self._get_bigwig_data(data_dict)
        else:
            bigwig_region_dict = {}

        # for each final region:
        #     finally, we separate the meta region data into regions and get the bulk data for each region in the form of a list of dict
        #     region filter by coverage and spiking low coverage region is also performed here
        final_data = self._get_regions(data_dict, bigwig_region_dict)
        return final_data


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
        pseudobulker_and_names,
        n_pseudobulks=10,
        return_rows=False,
        inplace=False,
        bypass_keys=None,
    ):
        self.pseudobulker_and_names: list[tuple[PseudobulkGenerator, str]] = (
            pseudobulker_and_names
        )
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
        return bulk_data_dict

    def __call__(self, data_dict: Dict[str, bytes]) -> List[Dict[str, np.ndarray]]:
        """Generate pseudobulks for each output prefix."""
        if self.inplace:
            bulk_data_col = data_dict
        else:
            # only copy the region info
            bulk_data_col = {}

        for pseudobulker, output_prefix in self.pseudobulker_and_names:
            bulk_data_dict = self._get_pseudo_bulks(
                data_dict=data_dict,
                output_prefix=output_prefix,
                pseudobulker=pseudobulker,
            )
            bulk_data_col.update(bulk_data_dict)

        list_of_dicts = []
        for i in range(self.n_pseudobulks):
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
    def __init__(self, cov_filter_key, min_cov=10, max_cov=1e5, low_cov_ratio=0.1):
        self.cov_filter_key = cov_filter_key
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio
        return

    def __call__(self, batch: dict):
        """Filter regions based on coverage."""
        data = batch[self.cov_filter_key]

        # sum over all dims except the first one
        region_sum = data.sum(axis=tuple(range(1, data.ndim)))

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
        batch = {k: v[use_rows, ...].copy() for k, v in batch.items()}
        return batch


class FetchRegionOneHot:
    """Fetch the one-hot encoded DNA sequence from the genome."""

    def __init__(
        self,
        region_key: str = "region",
        output_key: str = "dna_one_hot",
        dtype: str = "float32",
    ) -> None:
        """
        Initialize the FetchRegionOneHot transform.

        Parameters
        ----------
        region_key : str, optional
            The key to access the region name in the data dictionary. Defaults to "Name".
        output_key : str, optional
            The key to store the one-hot encoded DNA in the data dictionary. Defaults to "dna_one_hot".
        dtype : str, optional
            The data type of the one-hot encoded DNA. Defaults to "float32".

        """
        self.region_key = region_key
        self.output_key = output_key
        self.dtype = dtype

    def __call__(self, data: dict, remote_genome_one_hot) -> dict:
        """
        Apply the FetchRegionOneHot transform to the input data.

        Parameters
        ----------
        data : dict
            The input data dictionary.

        Returns
        -------
        dict
            The modified data dictionary with the one-hot encoded DNA.
        """
        genome_one_hot = ray.get(remote_genome_one_hot)
        # shape: (batch, length, channel)
        one_hot = genome_one_hot.get_regions_one_hot(data[self.region_key])
        # change to (batch, channel, length)
        data[self.output_key] = np.moveaxis(one_hot.astype(self.dtype), -2, -1)
        return data
