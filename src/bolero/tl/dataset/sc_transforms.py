from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix, vstack


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
        pseudobulk_to_rows,
        sample_regions=200,
        min_cov=10,
        max_cov=1e5,
        low_cov_ratio=0.1,
    ):
        if isinstance(prefixs, str):
            prefixs = [prefixs]
        self.prefixs = prefixs
        self.pseudobulk_to_rows = pseudobulk_to_rows
        self.sample_regions = sample_regions
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio

        # bulk order
        bulk_order = set()
        for _dict in pseudobulk_to_rows.values():
            for name in _dict.keys():
                bulk_order.add(name)
        self.bulk_order = sorted(bulk_order)

    def _bytes_to_array(self, data_dict):
        bytes_keys = [k for k, v in data_dict.items() if isinstance(v, bytes)]
        for key in bytes_keys:
            prefix, name_and_dtype = key.split(":")
            name, dtype = name_and_dtype.split("+")
            data_dict[f"{prefix}:{name}"] = np.frombuffer(
                data_dict.pop(key), dtype=dtype
            )
        return data_dict

    def _make_csr(self, data_dict):
        for prefix in self.prefixs:
            data = data_dict.pop(f"{prefix}:data")
            indices = data_dict.pop(f"{prefix}:indices")
            indptr = data_dict.pop(f"{prefix}:indptr")
            shape = data_dict.pop(f"{prefix}:shape")
            csr_data = csr_matrix((data, indices, indptr), shape=shape)
            data_dict[prefix] = csr_data
        return data_dict

    def _get_pseudo_bulks(self, data_dict):
        _per_prefix_bulk_data = defaultdict(list)
        for prefix in self.prefixs:
            cell_by_base = data_dict.pop(prefix)
            _pseudobulk_to_rows = self.pseudobulk_to_rows[prefix]
            for name, rows in _pseudobulk_to_rows.items():
                _bulk_values = csr_matrix(cell_by_base[rows].sum(axis=0).A1)
                _per_prefix_bulk_data[name].append(_bulk_values)

        bulk_data = []
        for name in self.bulk_order:
            agg_bulk = csr_matrix(vstack(_per_prefix_bulk_data[name]).sum(axis=0).A1)
            bulk_data.append(agg_bulk)
        bulk_data = vstack(bulk_data)
        data_dict["bulk_data"] = bulk_data
        return data_dict

    def _get_regions(self, data_dict):
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
            _bulk_names = np.array(self.bulk_order)[use_rows]

            for name, data in zip(_bulk_names, _region_bulk_data):
                _data = {
                    "bulk_name": name,
                    "bulk_data": data,
                    "region": f"{chrom}:{meta_start+rstart}-{meta_start+rend}",
                }
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
        data_dict = self._get_pseudo_bulks(data_dict)

        # for each final region:
        #     finally, we separate the meta region data into regions and get the bulk data for each region in the form of a list of dict
        #     region filter by coverage and spiking low coverage region is also performed here
        final_data = self._get_regions(data_dict)
        return final_data
