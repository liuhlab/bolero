import pathlib
from typing import Any, Dict, List, Tuple, Union

import cooler
import h5py
import numpy as np
import pyBigWig
import pysam
from cooler.api import matrix
from cooler.core import (
    RangeSelector2D,
    region_to_extent,
)
from cooler.util import parse_region
from skimage.transform import resize

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
        data_prefix: str = "",
        data_suffix: str = "",
        mode: str = "bp",
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
        self.data_prefix = data_prefix
        self.data_suffix = data_suffix
        self.allc_handles = [_open_allc(path) for path in allc_paths]
        self.mode = mode

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
        regions = understand_regions(region_, as_df=True)

        n_regions = len(region_)
        n_allc = len(self.allc_paths)

        if self.mode == "bp":
            assert (regions["End"] - regions["Start"]).unique().shape[
                0
            ] == 1, "Regions must have the same length."
            region_length = regions["End"].iloc[0] - regions["Start"].iloc[0]

            total_mc_values = np.zeros(
                shape=(n_regions, n_allc, region_length), dtype=np.float32
            )
            total_cov_values = np.zeros(
                shape=(n_regions, n_allc, region_length), dtype=np.float32
            )
        elif self.mode == "region":
            total_mc_values = np.zeros(shape=(n_regions, n_allc), dtype=np.float32)
            total_cov_values = np.zeros(shape=(n_regions, n_allc), dtype=np.float32)
        else:
            raise ValueError("mode must be 'bp' or 'region'.")

        for idx, (_, (chrom, start, end, *_)) in enumerate(regions.iterrows()):
            for idy, allc_handle in enumerate(self.allc_handles):
                mc_values, cov_values = query_allc_region(
                    allc_handle, chrom, start, end
                )
                if self.mode == "bp":
                    total_mc_values[idx, idy, :] = mc_values
                    total_cov_values[idx, idy, :] = cov_values
                elif self.mode == "region":
                    total_mc_values[idx, idy] = mc_values.sum()
                    total_cov_values[idx, idy] = cov_values.sum()

        data_dict[f"{self.data_prefix}mc{self.data_suffix}"] = total_mc_values
        data_dict[f"{self.data_prefix}cov{self.data_suffix}"] = total_cov_values
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
        balance: bool = False,
        data_key="values",
        norm_mode="log",
        image_scale=256,
    ) -> None:
        """
        Initialize FetchRegionCools.

        Parameters
        ----------
        - cool_paths: Path(s) to the cool file(s).
        - resolution: Resolution of the cool file.
        - region_key: Key in the data_dict that represents the region.
        - balance: Whether to balance the cool matrix.
        - data_key: Key in the data_dict to store the fetched data.
        - norm_mode: Normalization mode. Default is "log".
        - image_scale: The scale size of the image, data matrix loaded from the cool file will be resized to this scale. Default is 256.

        Returns
        -------
        None
        """
        if isinstance(cool_paths, (str, pathlib.Path)):
            cool_paths = [cool_paths]
        self.cool_paths = cool_paths
        self.region_key = region_key
        self.resolution = resolution
        self.cool_handles = [
            (
                h5py.File(path.split("::")[0])["/" + path.split("::")[1]]
                if "::" in path
                else h5py.File(path)
            )
            for path in cool_paths
        ]
        self.cool_objects = [cooler.Cooler(path) for path in cool_paths]
        self.balance = balance
        self.data_key = data_key
        self.norm_mode = norm_mode
        self.image_scale = image_scale

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

        if self.norm_mode == "log":
            # norm_mode "log" only works on count matrix,
            # if your data is normalized or balanced, should not do this step
            assert np.min(total_values) >= 0, "The matrix contains negative values."
            total_values = np.log1p(total_values + 1)

        total_values = resize(
            total_values,
            (n_regions, n_cool, self.image_scale, self.image_scale),
            anti_aliasing=True,
        )
        data_dict[self.data_key] = total_values
        return data_dict

    def query_cool_region(self, cool_handle, cool_object, chrom, start, end):
        """Get region data from an COOL file handle."""
        # bin_start = start // resolution
        # bin_end = (end-1) // resolution + 1
        try:
            data = (
                self.matrix(h5=cool_handle, cool=cool_object, balance=self.balance)
                .fetch(f"{chrom}:{start}-{end}")
                .astype("float32")
            )
        except ValueError:
            print(
                "Got ValueError when fetching region: {chrom}:{start}-{end}, return 0"
            )
            return 0
        return data

    def matrix(
        self,
        h5: h5py.File,
        cool: cooler.Cooler,
        balance: bool = False,
    ) -> RangeSelector2D:
        """
        Contact matrix selector

        Parameters
        ----------
        h5 : h5py.File
            The h5py file handle.
        cool : cooler.Cooler
            The cooler object.
        balance : bool, optional
            Whether to apply pre-calculated matrix balancing weights to the
            selection. Default is True and uses a column named 'weight'.
            Alternatively, pass the name of the bin table column containing
            the desired balancing weights. Set to False to return untransformed
            counts.
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
        sparse = False
        as_pixels = False
        chunksize = 10000000
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


class FetchRegionBigWigs:
    def __init__(
        self,
        bw_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        region_key: str = "region",
        data_key="bw_values",
        norm_mode="log",
    ):
        """
        Initialize FetchRegionBigWigs.

        Parameters
        ----------
        - bw_paths: Path(s) to the allc file(s).
        - region_key: Key in the data_dict that represents the region.
        - data_key: Key in the data_dict to store the fetched data.
        - norm_mode: Normalization mode. Default is "log".

        Returns
        -------
        None
        """
        if isinstance(bw_paths, (str, pathlib.Path)):
            bw_paths = [bw_paths]
        self.bw_paths = bw_paths
        self.region_key = region_key
        self.bw_handles = [pyBigWig.open(path) for path in bw_paths]
        self.data_key = data_key
        self.norm_mode = norm_mode

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch region BigWigs.
        """
        # region is an array of strings np.array["chr1:100-200", "chr2:300-400"]
        region_ = data_dict[self.region_key]

        if isinstance(region_, str):
            region_ = [region_]
        regions = understand_regions(region_, as_df=True)
        assert (regions["End"] - regions["Start"]).unique().shape[
            0
        ] == 1, "Regions must have the same length."
        # regions is a bed dataframe with columns ["Chromosome", "Start", "End"]

        n_regions = len(region_)
        region_length = regions["End"].iloc[0] - regions["Start"].iloc[0]
        n_bw = len(self.bw_paths)

        total_values = np.zeros(
            shape=(n_regions, n_bw, region_length), dtype=np.float32 #
        )
        for idx, (_, (chrom, start, end, *_)) in enumerate(regions.iterrows()):
            for idy, bw_handle in enumerate(self.bw_handles):
                temp_values = self.query_bw_region(bw_handle, chrom, start, end)
                total_values[idx, idy, :] = temp_values
        if self.norm_mode == "log":
            assert np.min(total_values) >= 0, "The matrix contains negative values."
            total_values = np.log(total_values + 1)
        data_dict[self.data_key] = total_values
        return data_dict

    def query_bw_region(self, bw_handle, chrom, start, end):
        """Get region data from an bigwig file handle."""
        data = bw_handle.values(chrom, start, end, numpy=True)
        # fill the nan value with 0
        data = np.nan_to_num(data)
        return data


class FetchRegionBigWigsReduced(FetchRegionBigWigs):
    def __init__(
        self,
        bw_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        region_key: str = "region",
        data_key: str = "bw_values",
        norm_mode: str = "log",
        resolution: int = 32,
    ):
        """
        Initialize FetchRegionBigWigsReduced.

        Parameters
        ----------
        - bw_paths: Path(s) to the BigWig file(s).
        - region_key: Key in the data_dict that represents the region.
        - data_key: Key in the data_dict to store the reduced fetched data.
        - norm_mode: Normalization mode. Default is "log".
        - resolution: Size of each bin in base pairs. Default is 32.

        Returns
        -------
        None
        """
        super().__init__(bw_paths, region_key, data_key, norm_mode)
        self.resolution = resolution

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch region BigWigs and reduce the data to specified bin size.

        Parameters
        ----------
        - data_dict: Dictionary containing region information.

        Returns
        -------
        - data_dict with reduced total_values.
        """

        # Call the superclass method to fetch the original total_values
        data_dict = super().__call__(data_dict)

        # Retrieve the fetched data
        total_values = data_dict[self.data_key]  # Shape: (n_regions, n_bw, region_length)

        # Get the shape parameters
        n_regions, n_bw, region_length = total_values.shape

        # Define the bin size
        bin_size = self.resolution

        # Calculate the number of bins
        n_bins = region_length // bin_size

        # Check if the region_length is divisible by bin_size
        if region_length % bin_size != 0:
            # If not, trim the excess data to make it divisible
            trimmed_length = n_bins * bin_size
            total_values = total_values[:, :, :trimmed_length]
            print(
                f"Warning: region_length ({region_length}) is not divisible by bin_size ({bin_size}). "
                f"Trimmed to {trimmed_length}."
            )

        # Reshape and aggregate the data to reduce the resolution
        # New shape will be (n_regions, n_bw, n_bins, bin_size)
        reshaped = total_values.reshape(n_regions, n_bw, n_bins, bin_size)

        # Aggregate by taking the mean across the bin_size axis
        # Resulting shape: (n_regions, n_bw, n_bins)
        total_values = reshaped.mean(axis=-1)

        # If normalization is set to "log", apply log transformation
        if self.norm_mode == "log":
            if np.min(total_values) < 0:
                raise ValueError("The reduced matrix contains negative values, cannot apply log normalization.")
            total_values = np.log(total_values + 1)

        # Update the data_dict with the reduced data
        data_dict[self.data_key] = total_values

        return data_dict





class ReverseCompHicData:
    def __init__(
        self,
        data_1d_keys: Tuple[str],
        data_2d_keys: Tuple[str],
        dna_key: str,
        chance: float = 0.5,
    ):
        """
        Initialize ReverseCompHicData.

        Parameters
        ----------
        - data_1d_keys: Keys in the data_dict to store the 1D data.
        - data_2d_keys: Keys in the data_dict to store the 2D data.
        - dna_key: Key in the data_dict to store the DNA sequence.
        - chance: The chance to reverse complement the data.

        Returns
        -------
        None
        """
        self.data_1d_keys = data_1d_keys
        self.data_2d_keys = data_2d_keys
        self.dna_key = dna_key
        self.chance = chance

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reverse complement the DNA sequence and the Hi-C data.

        Parameters
        ----------
        - data_dict: Dictionary containing the data.

        Returns
        -------
        Dictionary containing the updated data.
        """
        _bool = np.random.rand(1)
        if _bool < self.chance:
            if self.data_1d_keys is not None:
                for key in self.data_1d_keys:
                    data_dict[key] = np.flip(
                        data_dict[key], axis=-1
                    )  # -1 flip the sequence
            if self.data_2d_keys is not None:
                for key in self.data_2d_keys:
                    data_dict[key] = np.flip(
                        data_dict[key], axis=[-1, -2]
                    )  # -1 and -2 both filp the sequence, because the data is 2D
            data_dict[self.dna_key] = np.flip(
                data_dict[self.dna_key], axis=[-1, -2]
            )  # -1 flip the sequence, -2 flip the base pair (complement)
        return data_dict


class AddGaussianNoise:
    def __init__(self, data_keys: Tuple[str], std=0.1):
        """
        Initialize GaussianNoise.

        Parameters
        ----------
        - data_keys: Keys in the data_dict to store the data.
        - std: The standard deviation of the Gaussian noise. Default is 0.1.

        Returns
        -------
        None
        """
        self.data_keys = data_keys
        self.std = std

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add Gaussian noise to the data.

        Parameters
        ----------
        - data_dict: Dictionary containing the data.

        Returns
        -------
        Dictionary containing the updated data.
        """
        for key in self.data_keys:
            data_dict[key] += np.random.randn(*data_dict[key].shape) * self.std
        return data_dict



class GetEmbedding:
    def __init__(
        self,
        data_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        region_key: str = "region", #or original_name for bw_values maybe
        data_key="embedding",
        leg_map: Dict[str, int] = None,
    ):
        """
        Initialize GetEmbedding

        Parameters
        ----------
        - data_paths: Path(s) to the cool or bigwig file(s).
        - region_key: Key in the data_dict that represents the region.
        - data_key: Key in the data_dict to store the fetched data.

        Returns
        -------
        None
        """
        if isinstance(data_paths, (str, pathlib.Path)):
            data_paths = [data_paths]
        self.data_paths = data_paths #don't rely on path, try to provide information directly
        self.region_key = region_key
        self.data_key = data_key
        self.leg_map = leg_map #Suggestion: cell type as key and track, when you query path in fetchbigwig operator of prev step, organise matrix here / path, add key to data dict which is name of cell types
        #get cell class directly from data dict to turn to number
        #can save embedding directly, use consistent cell type name
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch region BigWigs.
        """
        # region is an array of strings np.array["chr1:100-200", "chr2:300-400"]
        region_ = data_dict[self.region_key]

        if isinstance(region_, str):
            region_ = [region_]
        regions = understand_regions(region_, as_df=True)
        assert (regions["End"] - regions["Start"]).unique().shape[
            0
        ] == 1, "Regions must have the same length."
        # regions is a bed dataframe with columns ["Chromosome", "Start", "End"]

        n_regions = len(region_)
        n_data = len(self.data_paths)

        total_values = np.zeros(shape=(n_regions, n_data, 1), dtype=np.int64)
        for idx in range(n_regions):
            for idy, data_path in enumerate(self.data_path):
                leg = data_path.split("::")[0].split("/")[-1].split(".")[0]
                total_values[idx, idy, :] = self.leg_map[leg]
        data_dict[self.data_key] = total_values
        return data_dict