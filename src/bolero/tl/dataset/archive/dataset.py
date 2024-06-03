import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyranges as pr
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from bolero.pp import Genome
from bolero.pp.normalize import convolve_data, normalize_atac_batch
from bolero.pp.genome_dataset import GenomePositionZarr, GenomeRegionZarr
from bolero.utils import try_gpu, understand_regions

DEFAULT_DEVICE = try_gpu()


def split_genome_regions(
    bed,
    n_parts=100,
    train_ratio=0.7,
    valid_ratio=0.1,
    test_ratio=0.2,
    random_state=None,
):
    """
    Split the genome regions into train, valid, and test sets with large genome partitioning.

    Parameters
    ----------
    bed : pyranges.PyRanges
        The genome regions to be split.
    n_parts : int, optional
        The number of partitions to split the genome into. Default is 100.
    train_ratio : float, optional
        The ratio of the training set. Default is 0.7.
    valid_ratio : float, optional
        The ratio of the validation set. Default is 0.1.
    test_ratio : float, optional
        The ratio of the test set. Default is 0.2.
    random_state : int, optional
        The random seed for splitting. Default is None.

    Returns
    -------
    train_regions : pyranges.PyRanges
        The training set.
    valid_regions : pyranges.PyRanges
        The validation set.
    test_regions : pyranges.PyRanges
        The test set.
    """
    if isinstance(bed, pd.DataFrame):
        bed = pr.PyRanges(bed)
    if len(bed) <= 3:
        raise ValueError("Too few regions to split")

    n_parts = min(len(bed), n_parts)
    _t = train_ratio + valid_ratio + test_ratio
    n_train_parts = int(np.round(train_ratio / _t * n_parts))
    n_train_parts = max(1, n_train_parts)
    n_valid_parts = int(np.round(valid_ratio / _t * n_parts))
    n_valid_parts = max(1, n_valid_parts)

    partition_order = pd.Series(range(n_parts))
    partition_order = partition_order.sample(
        n_parts, random_state=random_state
    ).tolist()

    bed = bed.sort()
    n_regions_in_chunk = len(bed) // n_parts
    partition_regions = dict(
        bed.df.groupby(pd.Series(range(len(bed))) // n_regions_in_chunk)
    )

    train_regions = pd.concat(
        [partition_regions[p] for p in sorted(partition_order[:n_train_parts])]
    )
    train_regions = pr.PyRanges(train_regions)

    valid_regions = pd.concat(
        [
            partition_regions[p]
            for p in sorted(
                partition_order[n_train_parts : n_train_parts + n_valid_parts]
            )
        ]
    )
    valid_regions = pr.PyRanges(valid_regions)

    test_regions = pd.concat(
        [
            partition_regions[p]
            for p in sorted(partition_order[n_train_parts + n_valid_parts :])
        ]
    )
    test_regions = pr.PyRanges(test_regions)

    return train_regions, valid_regions, test_regions


class GenomeDataset(Dataset):
    """
    Dataset class for working with genomic data.

    Args:
        regions (str or list): The regions of interest.
        genome (Genome or str): The genome object or the path to the genome file.
        save_dir (str, optional): The directory to save the genome file. Defaults to None.

    Attributes
    ----------
        region_bed (BedTool): The regions of interest in BED format.
        region_bed_df (pandas DataFrame): The regions of interest in DataFrame format.
        regions (pandas Index): The regions of interest as an index.
        genome (Genome): The genome object.
        offsets (dict): The chromosome offsets.
        _datasets (OrderedDict): The datasets used in the genome dataset.
        input_datasets (list): The input datasets.
        output_datasets (list): The output datasets.

    Methods
    -------
        __len__(): Returns the number of regions in the dataset.
        _get_idx_data(name, idx): Returns the data for a specific index.
        _get_slice_or_list_data(name, sel): Returns the data for a slice or list of indices.
        __getitem__(idx): Returns the input and output data for a given index or slice.
        __getitems__(idx_list): Returns the input and output data for a list of indices.
        __repr__(): Returns a string representation of the GenomeDataset object.
        add_position_dataset(name, da, datatype, load=False, pos_dim="pos"): Adds a position dataset to the genome dataset.
        add_region_dataset(name, da, datatype, load=False, region_dim="region"): Adds a region dataset to the genome dataset.
        downsample(downsample): Downsamples the dataset.
        get_subset(regions): Returns a subset of the dataset based on the specified regions.
        get_dataloader(train_ratio, valid_ratio, test_ratio, random_state, n_parts, batch_size, shuffle): Returns a dataloader for training, validation, and testing.

    """

    def __init__(self, regions, genome, save_dir=None) -> None:
        super().__init__()
        self.region_bed = understand_regions(regions)
        self.region_bed_df = self.region_bed.df
        self.regions = pd.Index(self.region_bed_df["Name"].values)

        if isinstance(genome, Genome):
            self.genome = genome
        else:
            self.genome = Genome(genome, save_dir=save_dir)
        self.offsets = self.genome.chrom_offsets.copy()

        self._datasets = OrderedDict()
        self.input_datasets = []
        self.output_datasets = []

        # add genome one-hot encoding
        self._datasets["genome_one_hot"] = GenomePositionZarr(
            da=self.genome.genome_one_hot.one_hot, offsets=self.offsets, load=True
        )

    def __len__(self):
        return len(self.regions)

    def _get_idx_data(self, name, idx):
        ds = self._datasets[name]
        if isinstance(ds, GenomePositionZarr):
            chrom, start, end, *_ = self.region_bed_df.iloc[idx]
            _data = ds.get_region_data(chrom, start, end)
        elif isinstance(ds, GenomeRegionZarr):
            _data = ds.get_region_data(self.regions[idx])
        else:
            raise ValueError("Unknown dataset type")
        return _data.copy()

    def _get_slice_or_list_data(self, name, sel):
        ds = self._datasets[name]
        if isinstance(ds, GenomePositionZarr):
            _data = ds.get_regions_data(self.region_bed_df.iloc[sel])
        elif isinstance(ds, GenomeRegionZarr):
            _data = ds.get_regions_data(self.regions[sel])
        else:
            raise ValueError("Unknown dataset type")
        return _data.copy()

    def __getitem__(self, idx):
        if isinstance(idx, (slice, list)):
            _func = self._get_slice_or_list_data
        elif isinstance(idx, int):
            _func = self._get_idx_data
        else:
            raise ValueError(f"Unknown idx type, got {type(idx)} idx {idx}")

        input = []
        output = []
        for name in self.input_datasets:
            input.append(_func(name, idx))
        for name in self.output_datasets:
            output.append(_func(name, idx))
        return input, output

    def __getitems__(self, idx_list):
        # if __getitems__ is defined, pytorch dataloader's fetch function will use
        # this instead of __getitem__ and pass a list of indices at once.
        # See pytorch code here:
        # https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/fetch.py#L51
        return self.__getitem__(idx_list)

    def __repr__(self) -> str:
        class_str = f"{self.__class__.__name__} object with {len(self)} regions"
        genome_str = f"Genome: {self.genome.name}"
        return f"{class_str}\n{genome_str}"

    def add_position_dataset(self, name, da, datatype, load=False, pos_dim="pos"):
        """
        Add a position dataset to the Bolero dataset.

        Parameters
        ----------
        name : str
            The name of the dataset.
        da : xr.DataArray
            The xarray DataArray containing the dataset.
        datatype : str
            The type of the dataset. Must be either 'input' or 'output'.
        load : bool, optional
            Whether to load the dataset immediately. Default is False.
        pos_dim : str, optional
            The name of the dimension representing the positions. Default is 'pos'.

        Raises
        ------
        AssertionError
            If the datatype is not 'input' or 'output'.
            If the dataset with the same name already exists.
            If the input da is not an xarray DataArray.
            If the pos_dim is not found in the da.

        """
        if "position" in da.dims:
            pos_dim = "position"

        assert datatype in (
            "input",
            "output",
        ), "datatype must be either 'input' or 'output'"
        assert name not in self._datasets, f"Dataset {name} already exists"
        assert isinstance(da, xr.DataArray), "da must be an xarray DataArray"
        assert pos_dim in da.dims, f"pos_dim {pos_dim} not found in da"
        self._datasets[name] = GenomePositionZarr(
            da=da, offsets=self.offsets, load=load, pos_dim=pos_dim
        )
        if datatype == "input":
            self.input_datasets.append(name)
        else:
            self.output_datasets.append(name)

    def add_region_dataset(self, name, da, datatype, load=False, region_dim="region"):
        """
        Add a region dataset to the dataset manager.

        Parameters
        ----------
        name : str
            The name of the dataset.
        da : xr.DataArray
            The xarray DataArray containing the dataset.
        datatype : str
            The type of the dataset. Must be either 'input' or 'output'.
        load : bool, optional
            Whether to load the dataset immediately. Default is False.
        region_dim : str, optional
            The name of the region dimension. Default is 'region'.

        Raises
        ------
        AssertionError
            If the datatype is not 'input' or 'output'.
            If the dataset with the given name already exists.
            If the input `da` is not an xarray DataArray.

        """
        assert datatype in (
            "input",
            "output",
        ), "datatype must be either 'input' or 'output'"
        assert name not in self._datasets, f"Dataset {name} already exists"
        assert isinstance(da, xr.DataArray), "da must be an xarray DataArray"
        self._datasets[name] = GenomeRegionZarr(da=da, load=load, region_dim=region_dim)
        if datatype == "input":
            self.input_datasets.append(name)
        else:
            self.output_datasets.append(name)

    def downsample(self, downsample):
        """
        Downsample the dataset.

        Parameters
        ----------
        downsample : int
            The number of regions to downsample to.

        Returns
        -------
        Dataset
            The downsampled dataset.

        Notes
        -----
        If the `downsample` parameter is less than the current number of regions in the dataset,
        a random subset of regions will be selected and returned as a new dataset.
        If the `downsample` parameter is greater than or equal to the current number of regions,
        the original dataset will be returned unchanged.

        """
        if downsample < len(self):
            _regions = self.regions
            # random downsample while keep the order
            sel_regions = np.random.choice(_regions, downsample, replace=False)
            return self.get_subset(sel_regions)
        else:
            return self

    def get_subset(self, regions):
        """
        Subset the dataset to a new set of regions.

        Only regions needs to be subsetted, the genome and other datasets are shared and queried on the fly.

        Args:
            regions (list): The regions to subset the dataset to.

        Returns
        -------
            GenomeDataset: A new GenomeDataset object with the subsetted regions.

        """
        # create a new object with the same genome and subsetted regions, using the same subclasses
        subset_obj = self.__class__(
            regions=regions, genome=self.genome, save_dir=self.genome.save_dir
        )
        subset_obj._datasets = self._datasets
        subset_obj.input_datasets = self.input_datasets
        subset_obj.output_datasets = self.output_datasets
        return subset_obj

    def get_dataloader(
        self,
        train_ratio=0.7,
        valid_ratio=0.1,
        test_ratio=0.2,
        random_state=None,
        n_parts=100,
        batch_size=128,
        shuffle=(True, False, False),
    ):
        """
        Get dataloaders for training, validation, and testing.

        Args:
            train_ratio (float): The ratio of training data. Defaults to 0.7.
            valid_ratio (float): The ratio of validation data. Defaults to 0.1.
            test_ratio (float): The ratio of testing data. Defaults to 0.2.
            random_state (int or None): The random state for shuffling the regions. Defaults to None.
            n_parts (int): The number of parts to split the regions into. Defaults to 100.
            batch_size (int): The batch size. Defaults to 128.
            shuffle (tuple): The shuffle configuration for training, validation, and testing. Defaults to (True, False, False).

        Returns
        -------
            tuple: A tuple of dataloaders for training, validation, and testing.

        """
        train_regions, valid_regions, test_regions = split_genome_regions(
            self.region_bed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            n_parts=n_parts,
        )
        train, valid, test = (
            DataLoader(
                dataset=self.get_subset(region_sel),
                batch_size=batch_size,
                shuffle=sh,
                num_workers=0,  # DO NOT USE MULTIPROCESSING, it has issue with the genome object
                collate_fn=lambda x: x,
            )
            for region_sel, sh in zip(
                [train_regions, valid_regions, test_regions], shuffle
            )
        )
        return train, valid, test


class RegionDataset(GenomeDataset):
    """
    A dataset class for working with region-based data.

    This class extends the GenomeDataset class and provides additional methods for reading and manipulating region data.

    Attributes
    ----------
        genome (str): The genome associated with the dataset.
        regions (list): The list of regions in the dataset.
        save_dir (str): The directory to save the dataset.
    """

    @staticmethod
    def read_region_data(task_data, label_da_name="y"):
        """
        Read region data from a file or DataFrame.

        Args:
            task_data (str or pathlib.Path or pd.DataFrame): The path to the file or DataFrame containing the region data.
            label_da_name (str, optional): The name of the label data array. Defaults to "y".

        Returns
        -------
            xr.Dataset: The region dataset.
        """
        if isinstance(task_data, (str, pathlib.Path)):
            task_path = str(task_data)
            if task_path.endswith(".zarr"):
                _ds = xr.open_zarr(task_path)
            elif task_path.endswith(".feather"):
                _df = pd.read_feather(task_path)
                _df.set_index(_df.columns[0], inplace=True)
                _df.index.name = "region"
                _ds = xr.Dataset({label_da_name: _df})
            else:
                raise ValueError(f"Unknown file format {task_path}")
        else:
            if isinstance(task_data, pd.DataFrame):
                task_data.index.name = "region"
                _ds = xr.Dataset({label_da_name: task_data})
        return _ds

    @classmethod
    def from_labels(cls, labels, genome, save_dir=None, label_name="y", load=True):
        """
        Create a new RegionDataset object from a binary dataframe.

        Args:
            labels (str or pathlib.Path or pd.DataFrame): The path to the file or DataFrame containing the label data.
            genome (str): The genome associated with the dataset.
            save_dir (str, optional): The directory to save the dataset. Defaults to None.
            label_name (str, optional): The name of the label data array. Defaults to "y".
            load (bool, optional): Whether to load the dataset. Defaults to True.

        Returns
        -------
            RegionDataset: The new RegionDataset object.
        """
        _ds = cls.read_region_data(labels, label_da_name=label_name)
        regions = understand_regions(_ds.get_index("region"))

        obj = cls(genome=genome, regions=regions, save_dir=save_dir)
        obj.input_datasets.append("genome_one_hot")
        obj.add_region_dataset(
            name=label_name,
            da=_ds[label_name],
            datatype="output",
            load=load,
            region_dim="region",
        )
        return obj


class ATACTrackDataset(GenomeDataset):
    """
    A dataset class for ATAC-seq track data.

    Parameters
    ----------
    - regions (PyRanges): Genomic regions to load data from.
    - genome (Genome): Genome object representing the reference genome.
    - save_dir (str, optional): Directory to save the genome data. Defaults to None.
    - conv_size (int, optional): Size of the convolution window. Defaults to 50.
    """

    def __init__(self, regions, genome, save_dir=None, conv_size=50) -> None:
        super().__init__(regions, genome, save_dir)
        self.conv_size = conv_size
        self.position_dataset_norm_value = {}

    @classmethod
    def from_regions(
        cls,
        regions,
        genome,
        conv_size=50,
        save_dir=None,
    ):
        """
        Create an ATACTrackDataset object from a set of genomic regions.

        Parameters
        ----------
        - regions (PyRanges): Genomic regions to load data from.
        - genome (Genome): Genome object representing the reference genome.
        - conv_size (int, optional): Size of the convolution window. Defaults to 50.
        - save_dir (str, optional): Directory to save the genome data. Defaults to None.

        Returns
        -------
        - ATACTrackDataset: An instance of the ATACTrackDataset class.
        """
        # load regions and extend by conv_size, the additional bases are loaded to prevent boundary effect during convolution
        regions = understand_regions(regions)
        if not isinstance(genome, Genome):
            genome = Genome(genome, save_dir=save_dir)

        # get region length
        region_length = regions.End - regions.Start
        # make sure region length is all the same
        assert region_length.unique().shape[0] == 1, "Region length is not consistent"
        region_length = region_length[0]
        regions = cls._extend_regions_for_conv(
            regions=regions,
            region_length=region_length,
            conv_size=conv_size,
            chrom_sizes=genome.chrom_sizes,
        )

        obj = cls(
            genome=genome, regions=regions, save_dir=save_dir, conv_size=conv_size
        )
        obj.input_datasets.append("genome_one_hot")
        return obj

    def get_subset(self, regions):
        """
        Get a subset of the dataset based on the given genomic regions.

        Parameters
        ----------
        - regions (PyRanges): Genomic regions to include in the subset.

        Returns
        -------
        - ATACTrackDataset: A new instance of the ATACTrackDataset class representing the subset.
        """
        obj = super().get_subset(regions)
        obj.conv_size = self.conv_size
        obj.position_dataset_norm_value = self.position_dataset_norm_value
        return obj

    def add_position_dataset(self, zarr_path, datatype, load=False, pos_dim="pos"):
        """
        Add a position dataset to the ATACTrackDataset.

        Parameters
        ----------
        - zarr_path (str): Path to the Zarr dataset.
        - datatype (str): Type of the dataset.
        - load (bool, optional): Whether to load the dataset. Defaults to False.
        - pos_dim (str, optional): Name of the position dimension. Defaults to "pos".
        """
        ds = xr.open_zarr(zarr_path)
        da = ds["site_count"]
        name = str(zarr_path)
        super().add_position_dataset(
            name=name, da=da, datatype=datatype, load=load, pos_dim=pos_dim
        )
        try:
            norm_value = ds["normalize"].to_pandas()
            self.position_dataset_norm_value[name] = norm_value
        except KeyError:
            print(
                f"Normalization value not found in {zarr_path}, run calculate_atac_norm_value first"
            )
            return

    @staticmethod
    def _extend_regions_for_conv(regions, region_length, conv_size, chrom_sizes):
        """
        Extend the genomic regions for convolution.

        Parameters
        ----------
        - regions (PyRanges): Genomic regions to extend.
        - region_length (int): Length of the genomic regions.
        - conv_size (int): Size of the convolution window.
        - chrom_sizes (dict): Chromosome sizes.

        Returns
        -------
        - PyRanges: Extended genomic regions.
        """
        # NOTE: This function only changes region coordinates, but not the region names
        # For region dataset that uses region names as index, their data will not be impacted
        # For position dataset, the region will be loaded with a flanking size of conv_size
        regions = regions.extend(conv_size).df
        not_length_judge = (regions["End"] - regions["Start"]) != int(
            region_length + 2 * conv_size
        )
        pass_end_judge = regions["End"] > regions["Chromosome"].map(chrom_sizes).astype(
            int
        )
        regions = regions.loc[~(not_length_judge | pass_end_judge).values]
        return pr.PyRanges(regions)

    def __process_batch__(self, input, output):
        """
        Process a batch of input and output data.

        Parameters
        ----------
        - input (ndarray): Input data.
        - output (ndarray): Output data.

        Returns
        -------
        - tuple: Processed input and output data.
        """

        # process atac data
        def _process_batch(batch, norm_value):
            norm_data = normalize_atac_batch(batch=batch, norm_value=norm_value)
            conv_data = convolve_data(norm_data, conv_size=self.conv_size)
            # remove the additional extended bases after convolution to prevent boundary effect
            conv_data = conv_data[..., self.conv_size : -self.conv_size]
            return conv_data

        # normalize input and output from a position zarr dataset when its norm value is available
        for i, name in enumerate(self.input_datasets):
            if name in self.position_dataset_norm_value:
                norm_value = self.position_dataset_norm_value[name]
                input[i] = _process_batch(input[i], norm_value)
        for i, name in enumerate(self.output_datasets):
            if name in self.position_dataset_norm_value:
                norm_value = self.position_dataset_norm_value[name]
                output[i] = _process_batch(output[i], norm_value)
        return input, output

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Parameters
        ----------
        - idx (int): Index of the item.

        Returns
        -------
        - tuple: Input and output data at the given index.
        """
        input, output = super().__getitem__(idx)
        input, output = self.__process_batch__(input, output)
        return input, output

    def __getitems__(self, idx_list):
        """
        Get the items at the given indices.

        Parameters
        ----------
        - idx_list (list): List of indices.

        Returns
        -------
        - tuple: Input and output data at the given indices.
        """
        input, output = super().__getitems__(idx_list)
        return input, output
