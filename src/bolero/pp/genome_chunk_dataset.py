import gzip
import pathlib
from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import ray
from bolero_process.atac.sc.zarr_io import CutSitesZarr
from scipy.sparse import csc_matrix, csr_matrix, vstack

from bolero.pp.utils import get_global_coords


class GenericGenomeChunkDataset:
    def __init__(self, **kwargs):
        """
        A generic class for creating genome-chunk list of dicts from single-cell or bulk data.

        The list of dicts is then used to create a ray dataset.
        """
        pass

    def get_regions_data(self, regions_df: pd.DataFrame) -> list[dict[str, bytes]]:
        """
        Take a regions df, return a list of dicts with data for each region.

        Each dict contains components of a row-by-base csr_matrix,
        converted to compressed bytes and stored in a dict,
        the dict key is started by the prefix of the dataset.

        finally, there is a region key for the region coords

        Example Schema:
        [
            {
                "region": str,
                "prefix:indices+uint32": gzip bytes,
                "prefix:indptr+uint32": gzip bytes,
                "prefix:data+float32": gzip bytes,
                "prefix:shape+uint32": gzip bytes,
            }
        ]
        """
        pass

    def get_row_names(self) -> pd.Index:
        """
        Return the row names of the sparse matrix.
        """
        pass


def array_to_compressed_bytes(array, level):
    """
    Compresses an array to bytes.
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


def array_to_compressed_bytes_dict(
    prefix: str, array: np.ndarray, level: int = 5
) -> dict[str, bytes]:
    """
    Compresses an array to a dictionary of compressed bytes.

    Parameters
    ----------
    prefix : str
        The prefix for the keys in the dictionary.
    array : np.ndarray
        The array to compress.
    level : int, optional
        The compression level. Default is 5.

    Returns
    -------
    dict[str, bytes]
        The dictionary of compressed bytes.
    """
    data_dict = {
        f"{prefix}:data+float32": gzip.compress(
            array.astype(np.float32).tobytes(), compresslevel=level
        ),
        f"{prefix}:shape+uint32": gzip.compress(
            np.array(array.shape).astype(np.uint32).tobytes(), compresslevel=level
        ),
    }
    return data_dict


@ray.remote
def select_smat_region(
    smat: csc_matrix,
    prefix: str,
    chrom: str,
    start: int,
    end: int,
    gstart: int,
    gend: int,
) -> csr_matrix:
    """
    Select a region sparse matrix from genome sparse matrix.
    """
    if gstart is not None and gend is not None:
        region_csr_mat = smat[:, gstart:gend].tocsr()
    else:
        region_csr_mat = smat[:, start:end].tocsr()

    data_dict = csr_matrix_to_compressed_bytes_dict(
        prefix=prefix, matrix=region_csr_mat, level=5
    )
    data_dict["region"] = f"{chrom}:{start}-{end}"
    return data_dict


class SingleCellCutsiteDataset:
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
        barcode_whitelist: pd.Index = None,
    ):
        super().__init__()
        self.dataset = CutSitesZarr(zarr_path)
        self.name = name

        # record the barcode whitelist in int index
        if barcode_whitelist is not None:
            self.use_barcode_idx = self.dataset.get_barcodes_idx(barcode_whitelist)
        else:
            self.use_barcode_idx = None
        self.remote_csc_mat = self._put_csc_mat()
        return

    def _put_csc_mat(self):
        """
        Put the sparse matrix into ray object.

        Parameters
        ----------
            barcode_whitelist: The barcode whitelist.
        """
        sites_data = self.dataset["cutsite"].to_pandas()

        # csr is very efficient even doing genome position selection
        smat = csr_matrix(
            (
                np.ones(sites_data.shape[0], dtype=bool),
                (sites_data["barcode"].values, sites_data["global_pos"].values),
            ),
            shape=(sites_data["barcode"].max() + 1, self.dataset.genome_total_length),
        )

        # filter rows
        if self.use_barcode_idx is not None:
            smat = smat[self.use_barcode_idx].copy()
        return ray.put(smat.tocsc())

    def get_regions_data(self, regions_df):
        """
        Get the meta region data for prepare ray data.

        Returns
        -------
            List[Dict]: The meta region data.
        """
        ds = self.dataset

        chrom_offset = ds["chrom_offset"].to_pandas()
        global_coords = get_global_coords(
            chrom_offsets=chrom_offset, region_bed_df=regions_df
        )
        regions_df = regions_df.iloc[:, :3].copy()
        regions_df["global_start"] = global_coords[:, 0]
        regions_df["global_end"] = global_coords[:, 1]

        total_dicts = []
        for _, (chrom, start, end, gstart, gend) in regions_df.iterrows():
            task = select_smat_region.remote(
                smat=self.remote_csc_mat,
                prefix=self.name,
                chrom=chrom,
                start=start,
                end=end,
                gstart=gstart,
                gend=gend,
            )
            total_dicts.append(task)
        total_dicts = ray.get(total_dicts)
        return total_dicts

    def get_row_names(self):
        """
        Get the row names (str) of the sparse matrix.

        Returns
        -------
            pd.Index: The row names.
        """
        barcodes = self.dataset.barcode_to_idx
        if self.use_barcode_idx is not None:
            barcodes = barcodes[barcodes.isin(self.use_barcode_idx)].copy()
        return barcodes.index


@ray.remote
def _bw_values_worker(bw_path, regions):
    regions_data = []
    with pyBigWig.open(bw_path) as bw:
        for _, (chrom, start, end, *_) in regions.iterrows():
            _data = bw.values(chrom, start, end, numpy=True)
            _data = np.nan_to_num(_data).astype("float32")
            regions_data.append(csr_matrix(_data))
    return regions_data


@ray.remote
def _remote_bw_values(bw_path, regions, fetch_chunks=5000) -> list[csr_matrix]:
    regions["chunk_id"] = np.arange(len(regions)) // fetch_chunks

    chunk_col = []
    for _, chunk_df in regions.groupby("chunk_id"):
        chunk_col.append(_bw_values_worker.remote(bw_path, chunk_df))

    regions_data = []
    for chunk_data in ray.get(chunk_col):
        regions_data.extend(chunk_data)
    return regions_data


class GenomeBigWigDataset:
    """Represents a genomic dataset stored in BigWig format."""

    def __init__(
        self,
        *args,
        prefix="bigwig",
        sparse=True,
        compress_level=5,
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
        self.prefix = prefix
        self.sparse = sparse
        self.compress_level = compress_level

        self.bigwig_path_dict = OrderedDict()
        self._add_bigwig(*args, **kwargs)

        self._opened_bigwigs = {}

    def __repr__(self):
        repr_str = f"GenomeBigWigDataset ({len(self.bigwig_path_dict)} bigwig)\n"
        for name, path in self.bigwig_path_dict.items():
            repr_str += f"{name}: {path}\n"
        return repr_str

    def _add_bigwig(self, *args, **kwargs):
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

    def get_regions_data(
        self,
        regions_df: pd.DataFrame,
    ) -> list[dict[str, Union[np.ndarray, list[float]]]]:
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
        names = self.get_row_names()
        tasks = []
        for name in names:
            path = self.bigwig_path_dict[name]
            this_task = _remote_bw_values.remote(path, regions_df)
            tasks.append(this_task)

        for i, task in enumerate(tasks):
            list_of_csr = ray.get(task)
            if i == 0:
                list_of_lists: list[list[csr_matrix]] = [
                    [region_csr] for region_csr in list_of_csr
                ]
            else:
                for idx, region_csr in enumerate(list_of_csr):
                    list_of_lists[idx].append(region_csr)

        # region
        region_names = (
            regions_df["Chromosome"]
            + ":"
            + regions_df["Start"].astype(str)
            + "-"
            + regions_df["End"].astype(str)
        ).tolist()

        @ray.remote
        def _get_data_dict(region_csr_list, region, sparse, prefix, compress_level):
            if sparse:
                data_dict = csr_matrix_to_compressed_bytes_dict(
                    prefix=prefix, matrix=vstack(region_csr_list), level=compress_level
                )
            else:
                data_dict = array_to_compressed_bytes_dict(
                    prefix=prefix,
                    array=np.vstack([x.toarray() for x in region_csr_list]),
                    level=compress_level,
                )
            data_dict["region"] = region
            return data_dict

        list_of_dicts = []
        for region, region_csr_list in zip(region_names, list_of_lists):
            task = _get_data_dict.remote(
                region_csr_list=region_csr_list,
                region=region,
                sparse=self.sparse,
                prefix=self.prefix,
                compress_level=self.compress_level,
            )
            list_of_dicts.append(task)
        list_of_dicts = ray.get(list_of_dicts)
        return list_of_dicts

    def get_row_names(self):
        """
        Get the row names of the csr_matrix.
        """
        return pd.Index(self.bigwig_path_dict.keys())


def query_allc_region(allc_handle, chrom, start, end):
    """Get region data from an ALLC file handle."""
    mc_values = np.zeros(end - start, dtype="float32")
    cov_values = np.zeros(end - start, dtype="float32")
    for row in allc_handle.fetch(chrom, start + 1, end):
        _, pos, *_, mc, cov, _ = row.strip().split("\t")
        rel_pos = int(pos) - 1 - start
        mc_values[rel_pos] = float(mc)
        cov_values[rel_pos] = float(cov)
    return mc_values, cov_values


@ray.remote
def _allc_values_worker(allc_path: str, regions: pd.DataFrame) -> list[csr_matrix]:
    mc_data = []
    cov_data = []
    with pysam.TabixFile(allc_path) as allc:
        for _, (chrom, start, end, *_) in regions.iterrows():
            # query allc file for each region, get two numpy arrays for this region
            mc_region_data, cov_region_data = query_allc_region(allc, chrom, start, end)
            mc_data.append(csr_matrix(mc_region_data))
            cov_data.append(csr_matrix(cov_region_data))

    return mc_data, cov_data


@ray.remote
def _remote_allc_values(allc_path, regions, fetch_chunks=5000) -> list[csr_matrix]:
    regions["chunk_id"] = np.arange(len(regions)) // fetch_chunks

    chunk_col = []
    for _, chunk_df in regions.groupby("chunk_id"):
        chunk_col.append(_allc_values_worker.remote(allc_path, chunk_df))

    # In single ALLC file,
    # list of mc csr_matrix for each region, the shape of csr_matrix is (1, region_length)
    regions_mc_data = []
    # list of cov csr_matrix for each region, the shape of csr_matrix is (1, region_length)
    regions_cov_data = []
    for _mc, _cov in ray.get(chunk_col):
        regions_mc_data.extend(_mc)
        regions_cov_data.extend(_cov)
    return regions_mc_data, regions_cov_data


class GenomeALLCDataset:
    def __init__(
        self,
        *args: str,
        prefix: str = "allc",
        sparse: bool = True,
        compress_level: int = 5,
        **kwargs: str,
    ) -> None:
        """
        Initialize the GenomeALLCDataset.
        This dataset is a IO helper to create genome chunk parquet for ray.data.Dataset

        Parameters
        ----------
        args : str
            The paths to the ALLC files. Name of the ALLC file will be row name.
        prefix : str, optional
            The prefix for the ALLC dataset name in the compressed bytes dictionary. Default is "allc".
        sparse : bool, optional
            Whether to use sparse representation for the sample-by-base matrix. Default is True.
        compress_level : int, optional
            The compression level. Default is 5.
        kwargs : str
            Additional paths to the ALLC files. The keys will be used as the row names.
        """
        self.prefix = prefix
        self.sparse = sparse
        self.compress_level = compress_level

        self.allc_path_dict = OrderedDict()
        self._add_allc(*args, **kwargs)

        self._opened_allcs = {}

    def __repr__(self) -> str:
        """
        Return a string representation of the GenomeALLCDataset.
        """
        repr_str = f"GenomeALLCDataset ({len(self.allc_path_dict)} allc)\n"
        for name, path in self.allc_path_dict.items():
            repr_str += f"{name}: {path}\n"
        return repr_str

    def _add_allc(self, *args: str, **kwargs: str) -> None:
        """
        Add paths to the ALLC files.

        Parameters
        ----------
        args : str
            The paths to the ALLC files. The name of the ALLC file will be the row name.
        kwargs : str
            Additional paths to the ALLC files. The keys will be used as the row names.
        """
        for key, value in kwargs.items():
            self.allc_path_dict[key] = str(value)
        for arg in args:
            name = pathlib.Path(arg).name
            self.allc_path_dict[name] = str(arg)

    def get_regions_data(
        self, regions_df: pd.DataFrame
    ) -> list[dict[str, Union[np.ndarray, list[float]]]]:
        """
        Get the data for the specified regions.

        Parameters
        ----------
        regions_df : pd.DataFrame
            The DataFrame containing the regions.

        Returns
        -------
        list[dict[str, Union[np.ndarray, list[float]]]]
            The list of data dictionaries for each region.
        """
        names = self.get_row_names()
        tasks = []
        for name in names:
            path = self.allc_path_dict[name]
            this_task = _remote_allc_values.remote(path, regions_df)
            tasks.append(this_task)

        for i, task in enumerate(tasks):
            list_of_mc_csr, list_of_cov_csr = ray.get(task)
            if i == 0:
                list_of_mc_lists: list[list[csr_matrix]] = [
                    [region_csr] for region_csr in list_of_mc_csr
                ]
                list_of_cov_lists: list[list[csr_matrix]] = [
                    [region_csr] for region_csr in list_of_cov_csr
                ]
            else:
                for idx, region_csr in enumerate(list_of_mc_csr):
                    list_of_mc_lists[idx].append(region_csr)
                for idx, region_csr in enumerate(list_of_cov_csr):
                    list_of_cov_lists[idx].append(region_csr)

        region_names = (
            regions_df["Chromosome"]
            + ":"
            + regions_df["Start"].astype(str)
            + "-"
            + regions_df["End"].astype(str)
        ).tolist()

        def _rename_k(k, suffix):
            prefix, data_info = k.split(":")
            return f"{prefix}{suffix}:{data_info}"

        @ray.remote
        def _get_data_dict(
            region_mc_csr_list: list[csr_matrix],
            region_cov_csr_list: list[csr_matrix],
            region: str,
            sparse: bool,
            prefix: str,
            compress_level: int,
        ) -> dict[str, Union[bytes, str]]:
            data_dicts = []
            for _suffix, _list_of_csr in zip(
                ["_mc", "_cov"], [region_mc_csr_list, region_cov_csr_list]
            ):
                if sparse:
                    data_dict = csr_matrix_to_compressed_bytes_dict(
                        prefix=prefix, matrix=vstack(_list_of_csr), level=compress_level
                    )
                else:
                    data_dict = array_to_compressed_bytes_dict(
                        prefix=prefix,
                        array=np.vstack([x.toarray() for x in _list_of_csr]),
                        level=compress_level,
                    )
                data_dict = {_rename_k(k, _suffix): v for k, v in data_dict.items()}

                data_dicts.append(data_dict)

            final_data_dict = {}
            for _d in data_dicts:
                final_data_dict.update(_d)
            final_data_dict["region"] = region
            return final_data_dict

        list_of_dicts = []
        for region, _mc, _cov in zip(region_names, list_of_mc_lists, list_of_cov_lists):
            task = _get_data_dict.remote(
                region_mc_csr_list=_mc,
                region_cov_csr_list=_cov,
                region=region,
                sparse=self.sparse,
                prefix=self.prefix,
                compress_level=self.compress_level,
            )
            list_of_dicts.append(task)
        list_of_dicts = ray.get(list_of_dicts)
        return list_of_dicts

    def get_row_names(self) -> pd.Index:
        """
        Get the row names of the GenomeALLCDataset.

        Returns
        -------
        pd.Index
            The row names.
        """
        return pd.Index(self.allc_path_dict.keys())


class SnapAnnDataDataset:
    def __init__(self, name, path, barcode_whitelist=None):
        import snapatac2 as snap

        self.name = name
        self.adata = snap.read(path, backed="r")

        self.use_barcodes = pd.Series(
            {name: idx for idx, name in enumerate(self.adata.obs_names)}
        )
        if barcode_whitelist is not None:
            self.use_barcodes = self.use_barcodes[
                self.use_barcodes.index.isin(barcode_whitelist)
            ]
        self.use_barcodes_idx = self.use_barcodes.values

    def _get_remote_csc_mat(self, chrom):
        adata = self.adata

        insertion_key = f"insertion_{chrom}"
        insertion_csc = adata.obsm[insertion_key].tocsc()
        if self.use_barcodes_idx is not None:
            insertion_csc = insertion_csc[self.use_barcodes_idx, :].copy()
        return ray.put(insertion_csc)

    def get_regions_data(self, regions_df):
        """Get list of dicts for each region's sparse matrix"""
        total_dicts = []
        for chrom, chrom_df in regions_df.groupby("Chromosome"):
            try:
                remote_smat = self._get_remote_csc_mat(chrom)
            except KeyError:
                print(f"No insertion data for {chrom}")
                continue

            for _, (chrom, start, end) in chrom_df.iterrows():
                task = select_smat_region.remote(
                    smat=remote_smat,
                    prefix=self.name,
                    chrom=chrom,
                    start=start,
                    end=end,
                    gstart=None,
                    gend=None,
                )
                total_dicts.append(task)
        total_dicts = ray.get(total_dicts)
        return total_dicts

    def get_row_names(self):
        """Get row names of the sparse matrix."""
        names = pd.Index(self.adata.obs_names)
        if self.use_barcodes_idx is not None:
            names = names[names.isin(self.use_barcodes.index)].copy()
        return names
