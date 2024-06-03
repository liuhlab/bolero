import gzip
import pathlib
from typing import Union

import numpy as np
import pandas as pd
import pyBigWig
import ray
from bolero_process.atac.sc.zarr_io import CutSitesZarr
from scipy.sparse import csr_matrix, vstack

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


@ray.remote
def select_smat_region(
    smat: csr_matrix,
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
    region_smat = smat[:, gstart:gend].copy()
    data_dict = csr_matrix_to_compressed_bytes_dict(
        prefix=prefix, matrix=region_smat, level=5
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
        self.remote_smat = self._put_smat(barcode_whitelist)

    def _put_smat(self, barcode_whitelist):
        """
        Put the sparse matrix into ray object.

        Parameters
        ----------
            barcode_whitelist: The barcode whitelist.
        """
        if barcode_whitelist is None:
            site_sel = (
                self.dataset["cutsite"]
                .sel(value="barcode")
                .isin(barcode_whitelist)
                .values
            )
            sites_data = self.dataset["cutsite"].sel(site=site_sel).to_pandas()
        else:
            sites_data = self.dataset["cutsite"].to_pandas()

        # csr is very efficient even doing genome position selection
        smat = csr_matrix(
            (
                np.ones(sites_data.shape[0], dtype=bool),
                (sites_data["barcode"].values, sites_data["global_pos"].values),
            ),
            shape=(sites_data["barcode"].max() + 1, self.dataset.genome_total_length),
        )
        return ray.put(smat)

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
                smat=self.remote_smat,
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
        Get the row names of the sparse matrix.

        Returns
        -------
            pd.Index: The row names.
        """
        barcodes = self.dataset.barcode_to_idx.index
        barcodes = pd.Index(barcodes)
        if self.barcode_whitelist is not None:
            barcodes = barcodes[barcodes.isin(self.barcode_whitelist)].copy()
        return barcodes


def _bw_values(bw, chrom, start, end):
    # inside bw, always keep numpy true
    _data = np.nan_to_num(bw.values(chrom, start, end, numpy=True)).astype("float32")
    return _data


@ray.remote
def _remote_bw_values(bw_path, regions) -> list[csr_matrix]:
    regions_data = []
    with pyBigWig.open(bw_path) as bw:
        for _, (chrom, start, end, *_) in regions.iterrows():
            values = _bw_values(bw=bw, chrom=chrom, start=start, end=end)
            values = csr_matrix(values)
    return regions_data


class GenomeBigWigDataset:
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
        names = self.get_row_names()
        tasks = []
        for name in names:
            path = self.bigwig_path_dict[name]
            this_tasks = []  # will be a list of csr_matrix for each region
            this_tasks.append(_remote_bw_values.remote(path, regions_df, sparse=True))
            tasks.append(this_tasks)

        for i, task in enumerate(tasks):
            if i == 0:
                list_of_lists: list[list[csr_matrix]] = [
                    [region_csr] for region_csr in ray.get(task)
                ]
            else:
                for idx, region_csr in enumerate(ray.get(task)):
                    list_of_lists[idx].append(region_csr)

        # region
        region_names = (
            regions_df["Chromosome"]
            + ":"
            + regions_df["Start"].astype(str)
            + "-"
            + regions_df["End"].astype(str)
        ).tolist()

        list_of_dicts = []
        for region, region_csr_list in zip(region_names, list_of_lists):
            data_dict = csr_matrix_to_compressed_bytes_dict(vstack(region_csr_list))
            data_dict["region"] = region
            list_of_dicts.append(data_dict)
        return list_of_dicts

    def get_row_names(self):
        """
        Get the row names of the csr_matrix.
        """
        return pd.Index(self.bigwig_path_dict.keys())


class SnapAnnDataDataset:
    def __init__(self, name, path, barcode_whitelist=None):
        import snapatac2 as snap

        self.name = name
        self.adata = snap.read(path)

        self.use_barcodes = self.adata.obs_names
        if barcode_whitelist is not None:
            self.use_barcodes = self.use_barcodes[
                self.use_barcodes.isin(barcode_whitelist)
            ]

    def _put_smat(self):
        # put insertion sites into ray object
        # select rows if needed
        pass

    def get_regions_data(self, regions_df):
        """Get list of dicts for each region's sparse matrix"""
        pass

    def get_row_names(self):
        """Get row names of the sparse matrix."""
        return self.adata.obs.index
