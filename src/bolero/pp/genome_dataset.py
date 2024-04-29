
from typing import Any, Tuple, Optional, Union

import xarray as xr

import numpy as np
import pandas as pd
import pyranges as pr
import pyBigWig
import pathlib

from bolero.utils import parse_region_name, parse_region_names
from bolero.pp.utils import get_global_coords
import ray

@ray.remote
def _remote_isel(da, dim, sel):
    # try first sel to get shape
    data_list = []
    for slice_ in enumerate(sel):
        data_list.append(da.isel({dim: slice_}).values)
    return data_list


def _bw_values(bw, chrom, start, end):
    # inside bw, always keep numpy true
    _data = bw.values(chrom, start, end, numpy=True)
    return _data


def _bw_values_chunk(bw, regions, sparse):
    regions_data = []
    for _, (chrom, start, end, *_) in regions.iterrows():
        regions_data.append(_bw_values(bw=bw, chrom=chrom, start=start, end=end))
    regions_data = np.array(regions_data)
    regions_data.astype("float32", copy=False)
    np.nan_to_num(regions_data, copy=False)

    if sparse:
        regions_data = sparse.csr_matrix(regions_data)
    return regions_data


@ray.remote
def _remote_bw_values_chunk(bw_path, regions, sparse=False):
    with pyBigWig.open(bw_path) as bw:
        return _bw_values_chunk(bw, regions, sparse=sparse)


class GenomeWideDataset:
    """
    Represents a dataset containing genome-wide data.

    Attributes
    ----------
        None

    Methods
    -------
        get_region_data: Retrieves data for a specific genomic region.
        get_regions_data: Retrieves data for multiple genomic regions.

    """

    def __init__(self):
        return

    def get_region_data(self, chrom: str, start: int, end: int) -> Any:
        """
        Retrieves data for a specific genomic region.

        Args:
            chrom (str): The chromosome of the genomic region.
            start (int): The start position of the genomic region.
            end (int): The end position of the genomic region.

        Returns
        -------
            Any: The data for the specified genomic region.

        Raises
        ------
            NotImplementedError: This method should be implemented by a subclass.

        """
        raise NotImplementedError

    def get_regions_data(
        self, regions: list[Tuple[str, int, int]], chunk_size: Optional[int] = None
    ) -> Any:
        """
        Retrieves data for multiple genomic regions.

        Args:
            regions (List[Tuple[str, int, int]]): A list of genomic regions specified as tuples of (chromosome, start, end).
            chunk_size (Optional[int]): The size of each chunk of data to retrieve.

        Returns
        -------
            Any: The data for the specified genomic regions.

        Raises
        ------
            NotImplementedError: This method should be implemented by a subclass.

        """
        raise NotImplementedError


class GenomePositionZarr(GenomeWideDataset):
    """
    Represents a genomic position in a Zarr dataset.

    Parameters
    ----------
    - da (xarray.DataArray): The Zarr dataset.
    - offsets (dict): A dictionary containing the global start offsets for each chromosome.
    - load (bool): Whether to load the dataset into memory. Default is False.
    - pos_dim (str): The name of the position dimension. Default is "pos".

    Attributes
    ----------
    - da (xarray.DataArray): The Zarr dataset.
    - load (bool): Whether the dataset is loaded into memory.
    - pos_dim (str): The name of the position dimension.
    - offsets (dict): The global start offsets for each chromosome.
    - global_start (dict): The global start positions for each chromosome.
    - _remote_da (ray.ObjectRef): The remote reference to the dataset (if not loaded).

    Methods
    -------
    - get_region_data(chrom, start, end): Get the region data for a specific chromosome and range.
    - get_regions_data(regions_df): Get the region data for multiple regions specified in a DataFrame.
    """

    def __init__(self, da, offsets, load=False, pos_dim="pos"):
        super().__init__()
        self.da = da
        self.load = load
        if load:
            self.da.load()

        if "position" in da.dims:
            pos_dim = "position"
        assert pos_dim in da.dims
        self.da = self.da.rename({pos_dim: "pos"})
        self.pos_dim = pos_dim

        self.offsets = offsets
        self.global_start = self.offsets["global_start"].to_dict()

        if load:
            self._remote_da = None
        else:
            self._remote_da = ray.put(self.da)

    def get_region_data(self, chrom, start, end):
        """
        Get the region data for a specific chromosome and range.

        Parameters
        ----------
        - chrom (str): The chromosome name.
        - start (int): The start position of the region.
        - end (int): The end position of the region.

        Returns
        -------
        - region_data (numpy.ndarray): The region data as a NumPy array.
        """
        add_start = self.global_start[chrom]
        global_start = start + add_start
        global_end = end + add_start

        region_data = self.da.isel(pos=slice(global_start, global_end)).values
        return region_data

    def get_regions_data(self, regions, chunk_size=None):
        """
        Get the region data for multiple regions specified in a DataFrame.

        Parameters
        ----------
        - regions_df (pandas.DataFrame): A DataFrame containing the regions to retrieve.

        Returns
        -------
        - regions_data (numpy.ndarray): The region data as a NumPy array.
        """
        if isinstance(regions, pr.PyRanges):
            regions_df = regions.df
        elif isinstance(regions, pd.DataFrame):
            regions_df = regions
        else:
            raise ValueError("regions must be a PyRanges or DataFrame")

        global_coords = get_global_coords(
            chrom_offsets=self.offsets, region_bed_df=regions_df
        )

        # init an empty array, assume all regions have the same length
        n_regions = len(global_coords)
        region_size = global_coords[0, 1] - global_coords[0, 0]
        shape_list = [n_regions]
        for dim, size in self.da.sizes.items():
            if dim == "pos":
                shape_list.append(region_size)
            else:
                shape_list.append(size)

        regions_data = np.zeros(shape_list, dtype=self.da.dtype)
        if self.load:
            for i, (start, end) in enumerate(global_coords):
                _data = self.da.isel(pos=slice(start, end)).values
                regions_data[i] = _data
        else:
            chunk_size = regions_df.shape[0] if chunk_size is None else chunk_size
            futures = []
            chunk_slices = []
            for chunk_start in range(0, regions_df.shape[0], chunk_size):
                chunk_slice = slice(chunk_start, chunk_start + chunk_size)
                _slice_list = [
                    slice(start, end) for start, end in global_coords[chunk_slice]
                ]
                task = _remote_isel.remote(self._remote_da, "pos", _slice_list)
                futures.append(task)
                chunk_slices.append(chunk_slice)

            data_list = ray.get(futures)
            for chunk_slice, data in zip(chunk_slices, data_list):
                regions_data[chunk_slice] = data
        return regions_data


class GenomeRegionZarr(GenomeWideDataset):
    """
    Represents a genomic region in Zarr format.

    Parameters
    ----------
    da : xarray.DataArray
        The data array containing the genomic region.
    load : bool, optional
        Whether to load the data array into memory, by default False.
    region_dim : str, optional
        The name of the dimension representing the regions, by default "region".

    Attributes
    ----------
    da : xarray.DataArray
        The data array containing the genomic region.
    load : bool
        Whether the data array is loaded into memory.
    region_dim : str
        The name of the dimension representing the regions.
    _remote_da : ray.ObjectRef or None
        A reference to the remote data array if not loaded into memory, None otherwise.

    Methods
    -------
    get_region_data(region)
        Get the data for a specific region.
    get_regions_data(*regions)
        Get the data for multiple regions.

    """

    def __init__(self, da, load=False, region_dim="region"):
        super().__init__()
        self.da = da
        self.load = load
        if load:
            self.da = self.da.load()

        assert region_dim in self.da.dims
        self.da = self.da.rename({region_dim: "region"})
        self.region_dim = region_dim

        if load:
            self._remote_da = None
        else:
            self._remote_da = ray.put(self.da)

    def get_region_data(self, *args, **kwargs):
        """
        Get the data for a specific region.

        Parameters
        ----------
        region : int, slice, or str
            The region to retrieve the data for.

        Returns
        -------
        numpy.ndarray
            The data for the specified region.

        """
        if "chrom" in kwargs and "start" in kwargs and "end" in kwargs:
            chrom = kwargs["chrom"]
            start = kwargs["start"]
            end = kwargs["end"]
            region = f"{chrom}:{start}-{end}"
        else:
            if len(args) == 1:
                region = args[0]
            else:
                region = pd.Index(args)

        if isinstance(region, (int, slice)):
            region_data = self.da.isel(region=region).values
        else:
            region_data = self.da.sel(region=region).values
        return region_data

    def get_regions_data(self, regions, chunk_size=None):
        """
        Get the data for multiple regions.

        Parameters
        ----------
        regions : int, slice, or str
            The regions to retrieve the data for.

        Returns
        -------
        numpy.ndarray
            The data for the specified regions.

        """
        # chunk size is not really used here, just be consistent with other data classes
        _ = len(regions) if chunk_size is None else chunk_size

        if isinstance(regions, pr.PyRanges):
            regions_df = regions.df
        elif isinstance(regions, pd.DataFrame):
            regions_df = regions
        else:
            regions_df = None
        if regions_df is not None:
            if "Name" in regions_df.columns:
                regions = regions_df["Name"]
            else:
                regions = []
                for _, (chrom, start, end, *_) in regions_df.iterrows():
                    regions.append(f"{chrom}:{start}-{end}")

        _data = self.get_region_data(regions)
        return _data


class GenomeOneHotZarr(GenomePositionZarr):
    """
    A class for working with one-hot encoded genomic data stored in Zarr format.

    Parameters
    ----------
    ds_path : str
        The path to the Zarr dataset.
    load : bool, optional
        Whether to load the dataset into memory, by default True.

    Attributes
    ----------
    ds : xr.Dataset
        The Zarr dataset.
    one_hot : xr.DataArray
        The one-hot encoded genomic data.

    Methods
    -------
    __repr__()
        Returns a string representation of the Zarr dataset.
    get_region_one_hot(*args)
        Get the one-hot encoded representation of a genomic region.
    get_regions_one_hot(regions)
        Get the one-hot encoded representation of the given regions.

    """

    def __init__(self, ds_path, load=True):
        self.ds = xr.open_zarr(ds_path)
        self.one_hot = self.ds["X"]
        if load:
            print("Loading genome DNA one-hot encoding...")
            self.one_hot.load()
        super().__init__(
            da=self.one_hot,
            offsets=self.ds["offsets"].to_pandas(),
            load=load,
            pos_dim="pos",
        )

    def __repr__(self):
        """
        Returns a string representation of the Zarr dataset.

        Returns
        -------
        str
            The string representation of the Zarr dataset.

        """
        return self.ds.__repr__()

    def get_region_one_hot(self, *args):
        """
        Get the one-hot encoded representation of a genomic region.

        Parameters
        ----------
        args : tuple
            If a single argument is provided, it is assumed to be a region name
            and will be parsed into chromosome, start, and end coordinates.
            If three arguments are provided, they are assumed to be chromosome,
            start, and end coordinates directly.

        Returns
        -------
        region_one_hot : numpy.ndarray
            The one-hot encoded representation of the genomic region.

        Raises
        ------
        ValueError
            If the number of arguments is not 1 or 3.

        """
        if len(args) == 1:
            # assume it's a region name
            chrom, start, end = parse_region_name(args[0])
        elif len(args) == 3:
            # assume it's chrom, start, end
            chrom, start, end = args
        else:
            raise ValueError("args must be a region name or chrom, start, end")

        region_one_hot = self.get_region_data(chrom, start, end)
        return region_one_hot

    def get_regions_one_hot(self, regions):
        """
        Get the one-hot encoded representation of the given regions.

        Parameters
        ----------
        regions : pd.DataFrame or pr.PyRanges or str or list
            The regions to be encoded. It can be provided as a pandas DataFrame,
            a PyRanges object, a string representing a region name, or a list of region names.

        Returns
        -------
        np.ndarray
            The one-hot encoded representation of the regions.

        Raises
        ------
        AssertionError
            If the regions have different lengths.

        """
        # get global coords
        if isinstance(regions, pd.DataFrame):
            regions = regions[["Chromosome", "Start", "End"]]
        elif isinstance(regions, pr.PyRanges):
            regions = regions.df[["Chromosome", "Start", "End"]]
        elif isinstance(regions, str):
            regions = parse_region_names([regions]).df[["Chromosome", "Start", "End"]]
        else:
            regions = parse_region_names(regions).df[["Chromosome", "Start", "End"]]
        global_coords = get_global_coords(
            chrom_offsets=self.offsets, region_bed_df=regions
        )

        # make sure regions are in the same length
        region_lengths = global_coords[:, 1] - global_coords[:, 0]
        assert (
            region_lengths == region_lengths[0]
        ).all(), "All regions must have the same length."

        region_one_hot = self.get_regions_data(regions)
        return region_one_hot


class GenomeBigWigDataset(GenomeWideDataset):
    """Represents a genomic dataset stored in BigWig format."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Represents a genomic dataset stored in BigWig format.

        Parameters
        ----------
        *args : str
            The paths to the BigWig files. The dataset names will be inferred from the file names.
        **kwargs : str
            The paths to the BigWig files, with the dataset names as the keys.
        """
        super().__init__()
        self.bigwig_path_dict = {}
        self.add_bigwig(*args, **kwargs)

        self._opened_bigwigs = {}

    def __repr__(self):
        repr_str = f"GenomeBigWigDataset ({len(self.bigwig_path_dict)} bigwig)\n"
        for name, path in self.bigwig_path_dict.items():
            repr_str += f"{name}: {path}\n"
        return repr_str

    def add_bigwig(self, *args, **kwargs):
        """
        Add a BigWig file to the dataset.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the BigWig file.
        name : str, optional
            The name of the dataset, by default None.
        """
        for key, value in kwargs.items():
            self.bigwig_path_dict[key] = str(value)
        for arg in args:
            name = pathlib.Path(arg).name
            self.bigwig_path_dict[name] = str(arg)

    def _open(self) -> None:
        """
        Open the BigWig files.
        """
        for name, path in self.bigwig_path_dict.items():
            self._opened_bigwigs[name] = pyBigWig.open(path)

    def _close(self) -> None:
        """
        Close the opened BigWig files.
        """
        for bw in self._opened_bigwigs.values():
            bw.close()
        self._opened_bigwigs = {}

    def __enter__(self) -> "GenomeBigWigDataset":
        """
        Enter the context manager and open the BigWig files.
        """
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the context manager and close the opened BigWig files.
        """
        self._close()

    def get_region_data(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> dict[str, np.ndarray]:
        """
        Get the data for a specific genomic region.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        start : int
            The start position of the region.
        end : int
            The end position of the region.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the region data for each dataset,
            where the keys are the dataset names and the values are the data arrays.
        """
        with self:
            region_data = {}
            for name, bw in self._opened_bigwigs.items():
                region_data[name] = _bw_values(bw, chrom, start, end)
        return region_data

    def get_regions_data(
        self,
        regions: Union[pr.PyRanges, pd.DataFrame],
        chunk_size: Optional[int] = None,
    ) -> dict[str, Union[np.ndarray, list[float]]]:
        """
        Get the data for multiple genomic regions.

        Parameters
        ----------
        regions : pr.PyRanges or pd.DataFrame
            The regions to retrieve data for.
        chunk_size : int, optional
            The number of regions to process in each chunk, by default None.

        Returns
        -------
        Dict[str, Union[np.ndarray, List[float]]]
            A dictionary containing the region data for each dataset,
            where the keys are the dataset names and the values are the data arrays or lists.

        Raises
        ------
        ValueError
            If the regions parameter is not a PyRanges or DataFrame.
        """
        if isinstance(regions, pr.PyRanges):
            regions_df = regions.df
        elif isinstance(regions, pd.DataFrame):
            regions_df = regions
        else:
            raise ValueError("regions must be a PyRanges or DataFrame")

        if chunk_size is None:
            chunk_size = regions_df.shape[0]

        names = []
        tasks = []
        for name, path in self.bigwig_path_dict.items():
            this_tasks = []
            for chunk_start in range(0, regions_df.shape[0], chunk_size):
                chunk_slice = slice(chunk_start, chunk_start + chunk_size)
                regions = regions_df.iloc[chunk_slice, :3].copy()
                this_tasks.append(_remote_bw_values_chunk.remote(path, regions, sparse=False))
            tasks.append(this_tasks)
            names.append(name)

        regions_data = {}
        for name, task in zip(names, tasks):
            regions_data[name] = np.concatenate(ray.get(task))
        return regions_data
