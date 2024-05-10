import gzip
import pathlib
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import ray
import xarray as xr
from bolero_process.atac.sc.zarr_io import CutSitesZarr
from scipy.sparse import csr_matrix, vstack

from bolero.pp.utils import get_global_coords
from bolero.utils import parse_region_name, parse_region_names


def array_to_compressed_bytes(array, level=5):
    """
    Compresses an array to bytes.

    Parameters
    ----------
        array: The array to compress.
        level: The compression level. Default is 5.

    Returns
    -------
        bytes: The compressed array.
    """
    return gzip.compress(array.tobytes(), compresslevel=level)


def csr_matrix_to_compressed_bytes_dict(
    prefix: str, matrix: csr_matrix, level: int = 5
) -> dict[str, bytes]:
    """
    Compresses a CSR matrix to a dictionary of compressed bytes.

    Parameters
    ----------
    prefix : str
        The prefix for the keys in the dictionary.
    matrix : csr_matrix
        The CSR matrix to compress.
    level : int, optional
        The compression level. Default is 5.

    Returns
    -------
    dict[str, bytes]
        The dictionary of compressed bytes.

    """
    data_dict = {
        f"{prefix}:indices+uint32": array_to_compressed_bytes(
            matrix.indices.astype(np.uint32), level=level
        ),
        f"{prefix}:indptr+uint32": array_to_compressed_bytes(
            matrix.indptr.astype(np.uint32), level=level
        ),
        f"{prefix}:data+float32": array_to_compressed_bytes(
            matrix.data.astype(np.float32), level=level
        ),
        f"{prefix}:shape+uint32": array_to_compressed_bytes(
            np.array(matrix.shape).astype(np.uint32), level=level
        ),
    }
    return data_dict


def prepare_meta_region(
    bed: Union[str, pathlib.Path], meta_region_size: int
) -> pd.DataFrame:
    """
    Process the region BED file to add a meta region column.

    Parameters
    ----------
        bed (str or pathlib.Path): The path to the BED file.
        meta_region_size (int): The size of the meta region.

    Returns
    -------
        pd.DataFrame: The processed region BED data.
    """
    if isinstance(bed, (str, pathlib.Path)):
        bed = pr.read_bed(bed, as_df=True)

    start_bin = bed["Start"] // meta_region_size
    end_bin = bed["End"] // meta_region_size
    start_mode = bed["Start"] % meta_region_size
    end_mode = bed["End"] % meta_region_size
    use_bin = np.where(start_mode > end_mode, start_bin, end_bin)
    bed["ChromBin"] = bed["Chromosome"].astype(str) + "-" + use_bin.astype(str)
    return bed


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
                this_tasks.append(
                    _remote_bw_values_chunk.remote(path, regions, sparse=False)
                )
            tasks.append(this_tasks)
            names.append(name)

        regions_data = {}
        for name, task in zip(names, tasks):
            regions_data[name] = np.concatenate(ray.get(task))
        return regions_data

    def get_meta_regions_data(
        self, regions: Union[pr.PyRanges, pd.DataFrame], meta_region_size: int = 100000
    ) -> dict[str, np.ndarray]:
        """
        Get the data for meta regions.

        Parameters
        ----------
        regions : pr.PyRanges or pd.DataFrame
            The regions to retrieve data for.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the meta region data for each dataset,
            where the keys are the dataset names and the values are the data arrays.

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
        bw_order = list(self.bigwig_path_dict.keys())

        # prepare meta region
        regions_df = prepare_meta_region(
            bed=regions_df, meta_region_size=meta_region_size
        )

        @ray.remote
        def get_meta_regions_worker(_regions_df):
            # get all regions first
            regions_data = self.get_regions_data(_regions_df)

            # add a row index
            dict_list = []
            prefix = "bigwig"
            _regions_df["region_idx"] = range(_regions_df.shape[0])
            for chrom_bin, _regions in _regions_df.groupby("ChromBin"):
                use_rows = _regions["region_idx"].values
                _use_regions_data = {
                    name: value[use_rows] for name, value in regions_data.items()
                }
                meta_region_chrom = chrom_bin.split("-")[0]
                meta_region_start = _regions["Start"].min()
                meta_region_end = _regions["End"].max()

                meta_region_data = np.zeros(
                    shape=(len(bw_order), meta_region_end - meta_region_start),
                    dtype=np.float32,
                )
                relative_coords = (
                    _regions[["Start", "End"]] - meta_region_start
                ).values
                for row, bw_name in enumerate(bw_order):
                    bw_data = _use_regions_data[bw_name]
                    for (rstart, rend), _one_region_data in zip(
                        relative_coords, bw_data
                    ):
                        meta_region_data[row, rstart:rend] = _one_region_data

                final_data = csr_matrix_to_compressed_bytes_dict(
                    prefix="bigwig", matrix=csr_matrix(meta_region_data), level=5
                )
                final_data[f"{prefix}:name_order"] = "|".join(bw_order)
                final_data[f"{prefix}:meta_region"] = (
                    f"{meta_region_chrom}:{meta_region_start}-{meta_region_end}"
                )
                final_data[f"{prefix}:relative_coords+uint32"] = (
                    array_to_compressed_bytes(relative_coords.astype(np.uint32))
                )
                dict_list.append(final_data)
            return dict_list

        tasks = []
        chrom_bins_list = regions_df["ChromBin"].unique().tolist()
        chunk_size = 100
        for chunk_start in range(0, len(chrom_bins_list), chunk_size):
            chunk_slice = slice(chunk_start, chunk_start + chunk_size)
            chunk_regions = regions_df[
                regions_df["ChromBin"].isin(chrom_bins_list[chunk_slice])
            ].copy()
            tasks.append(get_meta_regions_worker.remote(chunk_regions))

        total_dict_list = []
        for dl in ray.get(tasks):
            total_dict_list.extend(dl)
        return total_dict_list


@ray.remote
def process_meta_region(smat: csr_matrix, region_df: pd.DataFrame) -> csr_matrix:
    """
    Process the meta region data.

    Parameters
    ----------
    smat : csr_matrix
        The sparse matrix containing the raw data.
    region_df : pd.DataFrame
        The DataFrame containing the region information.

    Returns
    -------
    csr_matrix
        The processed meta region data.

    """
    # meta region
    region_df = region_df.sort_values(["Start"])
    meta_region_start = region_df["global_start"].min()
    meta_region_end = region_df["global_end"].max()
    # meta region raw data
    meta_region_smat = smat[:, meta_region_start:meta_region_end].copy()

    # relatvie coords within meta region
    relative_region_merged = (
        pr.PyRanges(region_df).merge().df[["Start", "End"]].copy()
        - region_df["Start"].min()
    )

    # select only relavent regions to further reduce data size
    rows = []
    cols = []
    for rstart, rend in relative_region_merged.values:
        region_coo = meta_region_smat[:, rstart:rend].tocoo()
        rows.append(region_coo.row)
        cols.append(region_coo.col + rstart)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    # reconstruct meta region with relavent data only
    final_data = csr_matrix(
        (np.ones_like(rows, dtype=meta_region_smat.dtype), (rows, cols)),
        shape=meta_region_smat.shape,
    )
    return final_data


class GenomeSingleCellCutsiteDataset(GenomeWideDataset):
    """
    Dataset class for single-cell cutsite data.

    Parameters
    ----------
        name (str): The name of the dataset.
        zarr_path (str): The path to the Zarr file.
        bed (str or pathlib.Path): The path to the BED file.
        meta_region_size (int, optional): The size of the meta region. Defaults to 100000.
    """

    def __init__(
        self,
        name: str,
        zarr_path: str,
        bed: Union[str, pathlib.Path],
        meta_region_size: int = 100000,
    ):
        super().__init__()
        self.dataset = CutSitesZarr(zarr_path)
        self.region_df = prepare_meta_region(bed=bed, meta_region_size=meta_region_size)
        self.name = name

    @staticmethod
    def _prepare_meta_region_worker(ds, region_df, prefix, genome_total_length):
        """
        Prepare the meta region worker.

        Parameters
        ----------
            ds: The dataset.
            region_df: The region DataFrame contains individual regions, who will be grouped into meta_region based on "ChromBin".
            prefix: The prefix.
            genome_total_length: The total length of the genome.

        Returns
        -------
            List[Dict]: The meta region data.
        """
        barcode_batch_size = 100000
        barcode_batches = ds.barcode_to_idx // barcode_batch_size

        total_csrs = []
        for _, _batch_barcodes in ds.barcode_to_idx.groupby(barcode_batches):
            site_sel = (
                ds["cutsite"].sel(value="barcode").isin(_batch_barcodes.values).values
            )
            sites_data = ds["cutsite"].sel(site=site_sel).to_pandas()
            # csr is very efficient even doing genome position selection
            smat = csr_matrix(
                (
                    np.ones(sites_data.shape[0], dtype=bool),
                    (sites_data["barcode"].values, sites_data["global_pos"].values),
                ),
                shape=(sites_data["barcode"].max() + 1, genome_total_length),
            )
            smat_remote = ray.put(smat)
            batch_csrs = []
            for _, sub_df in region_df.groupby("ChromBin"):
                task = process_meta_region.remote(smat=smat_remote, region_df=sub_df)
                batch_csrs.append(task)
            batch_csrs = ray.get(batch_csrs)
            total_csrs.append(batch_csrs)

        # vstack batches and add meta region info
        meta_region_dicts = []
        for (chrom_bin, sub_df), batch_csrs in zip(
            region_df.groupby("ChromBin"), zip(*total_csrs)
        ):
            chrom = chrom_bin.split("-")[0]
            meta_region_start = sub_df["Start"].min()
            meta_region_end = sub_df["End"].max()
            final_data = vstack(batch_csrs)
            relative_region = sub_df[["Start", "End"]] - meta_region_start
            # return a dict
            # csr_matrix indices, indptr, data are all variable size, ray data has issue support variable size
            # when writing to parquet, we need to convert to bytes and reconstruct csr_matrix during getting data
            # https://github.com/ray-project/ray/issues/41924
            final_data_dict = csr_matrix_to_compressed_bytes_dict(
                prefix=prefix, matrix=final_data, level=5
            )
            final_data_dict.update(
                {
                    f"{prefix}:relative_coords+uint32": array_to_compressed_bytes(
                        relative_region.values.astype(np.uint32)
                    ),
                    f"{prefix}:meta_region": f"{chrom}:{meta_region_start}-{meta_region_end}",
                }
            )

            meta_region_dicts.append(final_data_dict)
        return meta_region_dicts

    def get_meta_region_data(self):
        """
        Get the meta region data for prepare ray data.

        Returns
        -------
            List[Dict]: The meta region data.
        """
        region_df = self.region_df
        ds = self.dataset

        chrom_offset = ds["chrom_offset"].to_pandas()
        global_coords = get_global_coords(
            chrom_offsets=chrom_offset, region_bed_df=region_df
        )
        region_df["global_start"] = global_coords[:, 0]
        region_df["global_end"] = global_coords[:, 1]
        genome_total_length = chrom_offset["global_end"].max()
        meta_region_dicts = self._prepare_meta_region_worker(
            ds=ds,
            region_df=region_df,
            prefix=self.name,
            genome_total_length=genome_total_length,
        )
        return meta_region_dicts
