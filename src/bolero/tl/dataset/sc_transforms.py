import gzip
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix, vstack

from bolero.tl.pseudobulk.generator import PseudobulkGenerator


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
        pseudobulker,
        sample_regions=200,
        min_cov=10,
        max_cov=1e5,
        low_cov_ratio=0.1,
        n_pseudobulks=10,
        return_cells=False,
    ):
        if isinstance(prefixs, str):
            prefixs = [prefixs]
        self.prefixs = prefixs

        if "bigwig" in self.prefixs:
            self.bigwig_prefix = "bigwig"
            self.prefixs = [p for p in self.prefixs if p != self.bigwig_prefix]
        else:
            self.bigwig_prefix = None

        self.pseudobulker: PseudobulkGenerator = pseudobulker
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
        # merge single cell to bulk and also get embedding data
        for bulk_idx, (cells, prefix_to_rows, cell_embedding) in enumerate(
            self.pseudobulker.take(self.n_pseudobulks)
        ):
            embedding_data.append(cell_embedding)
            cells_col[bulk_idx] = cells
            for prefix in self.prefixs:
                cell_by_base = data_dict[prefix]
                rows = prefix_to_rows.get(prefix, [])

                # some pseudo-bulks may not have any cells for a prefix
                if len(rows) == 0:
                    continue

                _bulk_values = csr_matrix(cell_by_base[rows].sum(axis=0).A1)
                _per_prefix_bulk_data[bulk_idx].append(_bulk_values)
        embedding_data = np.array(embedding_data)

        # remove prefix csr_matrix from data_dict
        for prefix in self.prefixs:
            data_dict.pop(prefix)

        bulk_data = []
        for bulk_idx in range(self.n_pseudobulks):
            bulk_data_list = _per_prefix_bulk_data[bulk_idx]
            if len(bulk_data_list) == 0:
                raise ValueError(
                    f"No cells for bulk {bulk_idx}, this might be due to prefix or cell id mismatch."
                )
            agg_bulk = csr_matrix(
                vstack(_per_prefix_bulk_data[bulk_idx]).sum(axis=0).A1
            )
            bulk_data.append(agg_bulk)
        bulk_data = vstack(bulk_data)
        data_dict["bulk_data"] = bulk_data
        data_dict["embedding_data"] = embedding_data

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
            if "cells" in data_dict:
                use_row_idx = np.where(use_rows)[0]
                use_cells = [data_dict["cells"][idx] for idx in use_row_idx]
            else:
                use_cells = None

            for data, embedding in zip(_region_bulk_data, _region_embedding_data):
                region = f"{chrom}:{meta_start+rstart}-{meta_start+rend}"
                _bw_data = bigwig_region_dict.get(region, {})
                _data = {
                    "bulk_embedding": embedding,
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
