import pathlib
from typing import Any, Dict, List, Union

import cooler
import h5py
import numpy as np
import pysam
from cooler.api import matrix
from cooler.core import (
    RangeSelector2D,
    region_to_extent,
)
from cooler.util import parse_region

from bolero.pp.genome_chunk_dataset import query_allc_region
from bolero.utils import understand_regions


def _open_allc(allc_path):
    handle = pysam.TabixFile(allc_path, mode="r")
    return handle


def _open_cool(cool_path):
    handle = cooler.Cooler(cool_path, mode="r")
    return handle


class FetchRegionALLCs:
    def __init__(
        self,
        allc_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        region_key: str = "region",
    ) -> None:
        """
        Initialize FetchRegionALLCs.

        Parameters
        ----------
        - allc_paths: Path(s) to the allc file(s).
        - region_key: Key in the data_dict that represents the region.

        Returns
        -------
        None
        """
        if isinstance(allc_paths, (str, pathlib.Path)):
            allc_paths = [allc_paths]
        self.allc_paths = allc_paths
        self.region_key = region_key
        self.allc_handles = [_open_allc(path) for path in allc_paths]

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch region ALLCs.

        Parameters
        ----------
        - data_dict: Dictionary containing the data.

        Returns
        -------
        Dictionary containing the updated data.
        """
        region_ = data_dict[self.region_key]
        if isinstance(region_, str):
            region_ = [region_]
        regions = understand_regions(region_)
        assert (regions["End"] - regions["Start"]).unique().shape[
            0
        ] == 1, "Regions must have the same length."

        n_regions = len(region_)
        region_length = regions["End"].iloc[0] - regions["Start"].iloc[0]
        n_allc = len(self.allc_paths)

        total_mc_values = np.zeros(
            shape=(n_regions, n_allc, region_length), dtype=np.float32
        )
        total_cov_values = np.zeros(
            shape=(n_regions, n_allc, region_length), dtype=np.float32
        )
        for idx, (_, (chrom, start, end, *_)) in enumerate(regions.iterrows()):
            for idy, allc_handle in enumerate(self.allc_handles):
                mc_values, cov_values = query_allc_region(
                    allc_handle, chrom, start, end
                )
                total_mc_values[idx, idy, :] = mc_values
                total_cov_values[idx, idy, :] = cov_values
        data_dict["mc_values"] = total_mc_values
        data_dict["cov_values"] = total_cov_values
        return data_dict

    def close(self) -> None:
        """
        Close allc handles.

        Returns
        -------
        None
        """
        for handle in self.allc_handles:
            handle.close()


class FetchRegionCools:
    def __init__(
        self,
        cool_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        resolution: int,
        region_key: str = "region",
    ) -> None:
        """
        Initialize FetchRegionALLCs.

        Parameters
        ----------
        - allc_paths: Path(s) to the allc file(s).
        - region_key: Key in the data_dict that represents the region.

        Returns
        -------
        None
        """
        if isinstance(cool_paths, (str, pathlib.Path)):
            cool_paths = [cool_paths]
        self.cool_paths = cool_paths
        self.region_key = region_key
        self.resolution = resolution
        self.cool_handles = [h5py.File(path) for path in cool_paths]
        self.cool_objects = [cooler.Cooler(path) for path in cool_paths]

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch region ALLCs.

        Parameters
        ----------
        - data_dict: Dictionary containing the data.

        Returns
        -------
        Dictionary containing the updated data.
        """
        region_ = data_dict[self.region_key]
        resolution = self.resolution
        if isinstance(region_, str):
            region_ = [region_]
        regions = understand_regions(region_, as_df=True)
        regions["Start"] = regions["Start"] // resolution * resolution
        regions["End"] = ((regions["End"] - 1) // resolution + 1) * resolution
        assert (regions["End"] - regions["Start"]).unique().shape[
            0
        ] == 1, "Regions must have the same length."

        n_regions = len(region_)
        region_length = (
            regions["End"].iloc[0] - regions["Start"].iloc[0]
        ) // resolution
        n_cool = len(self.cool_paths)

        total_values = np.zeros(
            shape=(n_regions, n_cool, region_length, region_length), dtype=np.float32
        )
        for idx, (_, (chrom, start, end, *_)) in enumerate(regions.iterrows()):
            for idy, (cool_handle, cool_object) in enumerate(
                zip(self.cool_handles, self.cool_objects)
            ):
                temp_values = self.query_cool_region(
                    cool_handle, cool_object, chrom, start, end
                )
                total_values[idx, idy, ...] = temp_values
        data_dict["values"] = total_values
        return data_dict

    def query_cool_region(self, cool_handle, cool_object, chrom, start, end):
        """Get region data from an COOL file handle."""
        # bin_start = start // resolution
        # bin_end = (end-1) // resolution + 1
        data = (
            self.matrix(h5=cool_handle, cool=cool_object)
            .fetch(f"{chrom}:{start}-{end}")
            .astype("float32")
        )
        return data

    def matrix(
        self,
        h5: h5py.File,
        cool: cooler.Cooler,
        sparse: bool = False,
        as_pixels: bool = False,
        chunksize: int = 10000000,
    ) -> RangeSelector2D:
        """
        Contact matrix selector

        Parameters
        ----------
        h5 : h5py.File
            The h5py file handle.
        cool : cooler.Cooler
            The cooler object.
        sparse : bool, optional
            Return a scipy.sparse.coo_matrix instead of a dense 2D numpy array.
            Default is False.
        as_pixels : bool, optional
            Return a DataFrame of the corresponding rows from the pixel table
            instead of a rectangular sparse matrix. Default is False.
        chunksize : int, optional
            The chunk size for fetching the matrix. Default is 10000000.

        Returns
        -------
        RangeSelector2D
            Matrix selector

        Notes
        -----
        If `as_pixels=True`, only data explicitly stored in the pixel table
        will be returned: if the cooler's storage mode is symmetric-upper,
        lower triangular elements will not be generated. If
        `as_pixels=False`, those missing non-zero elements will
        automatically be filled in.
        """
        balance = False
        join = True
        ignore_index = True
        divisive_weights = False
        field = None

        def _slice(field: str, i0: int, i1: int, j0: int, j1: int):
            grp = h5[cool.root]
            return matrix(
                grp,
                i0,
                i1,
                j0,
                j1,
                field,
                balance,
                sparse,
                as_pixels,
                join,
                ignore_index,
                divisive_weights,
                chunksize,
                cool._is_symm_upper,
            )

        def _fetch(region: str, region2: str = None) -> tuple[int, int, int, int]:
            grp = h5[cool.root]
            if region2 is None:
                region2 = region
            region1 = parse_region(region, cool._chromsizes)
            region2 = parse_region(region2, cool._chromsizes)
            i0, i1 = region_to_extent(grp, cool._chromids, region1, cool.binsize)
            j0, j1 = region_to_extent(grp, cool._chromids, region2, cool.binsize)
            return i0, i1, j0, j1

        return RangeSelector2D(field, _slice, _fetch, (cool._info["nbins"],) * 2)


# old code, temp save here
# class NotRayBatchRegionBigWig:
#     """Fetch the bigwig signal from the genome.

#     This class is not compatible with ray.data pipeline
#     because the pyBigWig's c code has some weird behavior when used with ray parallel.
#     Currently, I use this class after ray dataset's iter_batches, so it is processed in a single thread
#     To speed up, this class has its own cache
#     Do not use this class to fetch multiple bigwig, it will be slow and memory consuming. Prepare a ray dataset instead.

#     Args:
#         region_key (str): The key to access the region information in the data dictionary.
#         bw_dict (dict[str, str]): A dictionary mapping the signal name to the corresponding bigwig file path.
#         dtype (str): The data type of the fetched signal. Defaults to "float32".
#         torch (bool): Whether to convert the fetched signal to a torch tensor. Defaults to False.
#         device (str): The device to store the torch tensor. If None, it will try to use the available GPU. Defaults to None.

#     Attributes
#     ----------
#         region_key (str): The key to access the region information in the data dictionary.
#         bw_dict (dict[str, str]): A dictionary mapping the signal name to the corresponding bigwig file path.
#         dtype (str): The data type of the fetched signal.
#         torch (bool): Whether to convert the fetched signal to a torch tensor.
#         device (str): The device to store the torch tensor.

#     """

#     def __init__(
#         self,
#         region_key: str,
#         bw_dict: dict[str, str] = None,
#         dtype: str = "float32",
#         torch: bool = False,
#         device: str = None,
#     ):
#         self.region_key = region_key
#         self.bw_dict = bw_dict
#         self.dtype = dtype
#         self.torch = torch
#         if device is None:
#             self.device = try_gpu()
#         self.device = device

#     def __enter__(self):
#         # open bigwig files
#         self.bw_files = {k: pyBigWig.open(v) for k, v in self.bw_dict.items()}
#         self._region_cache = defaultdict(dict)
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         for bw in self.bw_files.values():
#             bw.close()
#         return

#     def _cached_fetch(self, name, bw, region):
#         try:
#             _value = self._region_cache[name][region]
#         except KeyError:
#             chrom, coords = region.split(":")
#             start, end = map(int, coords.split("-"))
#             _value = bw.values(chrom, start, end, numpy=True)
#             self._region_cache[name][region] = _value
#         return _value

#     def __call__(self, data_dict: dict) -> dict:
#         """Fetch the bigwig signal from the genome."""
#         for k, bw in self.bw_files.items():
#             _values = []
#             for region in data_dict[self.region_key]:
#                 _value = self._cached_fetch(k, bw, region)
#                 _values.append(_value)
#             data = np.nan_to_num(np.array(_values, dtype=self.dtype))
#             if self.torch:
#                 data = torch.tensor(data, device=self.device)
#             data_dict[k] = data
#         return data_dict
