import pandas as pd
import torch
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader, Dataset
from bolero.pp import Genome, GenomePositionZarr, GenomeRegionZarr
from bolero.utils import parse_region_names
import pathlib
import pyranges as pr
from collections import OrderedDict
from bolero.utils import *

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
    partition_regions = {
        p: r
        for p, r in bed.df.groupby(pd.Series(range(len(bed))) // n_regions_in_chunk)
    }
    train_regions = pd.concat(
        [partition_regions[p] for p in sorted(partition_order[:n_train_parts])]
    )
    train_regions = pd.Index(train_regions["Name"])
    valid_regions = pd.concat(
        [
            partition_regions[p]
            for p in sorted(
                partition_order[n_train_parts : n_train_parts + n_valid_parts]
            )
        ]
    )
    valid_regions = pd.Index(valid_regions["Name"])
    test_regions = pd.concat(
        [
            partition_regions[p]
            for p in sorted(partition_order[n_train_parts + n_valid_parts :])
        ]
    )
    test_regions = pd.Index(test_regions["Name"])
    return train_regions, valid_regions, test_regions


class GenomeDataset(Dataset):
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

    def _get_slice_data(self, name, slice_obj):
        ds = self._datasets[name]
        if isinstance(ds, GenomePositionZarr):
            _data = ds.get_regions_data(self.region_bed_df.iloc[slice_obj])
        elif isinstance(ds, GenomeRegionZarr):
            _data = ds.get_regions_data(self.regions[slice_obj])
        else:
            raise ValueError("Unknown dataset type")
        return _data.copy()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            _func = self._get_idx_data
        elif isinstance(idx, slice):
            _func = self._get_slice_data
        else:
            raise ValueError(f"Unknown idx type, got {type(idx)} idx {idx}")

        input = []
        output = []
        for name in self.input_datasets:
            input.append(_func(name, idx))
        for name in self.output_datasets:
            output.append(_func(name, idx))
        return input, output

    def __repr__(self) -> str:
        class_str = f"{self.__class__.__name__} object with {len(self)} regions"
        genome_str = f"Genome: {self.genome.name}"
        return f"{class_str}\n{genome_str}"

    def add_position_dataset(self, name, da, datatype, load=False, pos_dim="pos"):
        if "position" in da.dims:
            pos_dim = "position"

        assert datatype in (
            "input",
            "output",
        ), f"datatype must be either 'input' or 'output'"
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
        assert datatype in (
            "input",
            "output",
        ), f"datatype must be either 'input' or 'output'"
        assert name not in self._datasets, f"Dataset {name} already exists"
        assert isinstance(da, xr.DataArray), "da must be an xarray DataArray"
        self._datasets[name] = GenomeRegionZarr(da=da, load=load, region_dim=region_dim)
        if datatype == "input":
            self.input_datasets.append(name)
        else:
            self.output_datasets.append(name)

    def downsample(self, downsample):
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
            )
            for region_sel, sh in zip(
                [train_regions, valid_regions, test_regions], shuffle
            )
        )
        return train, valid, test


class BinaryDataset(GenomeDataset):
    def __init__(
        self,
        data,
        genome,
        save_dir=None,
        label_da_name="y",
        load=True,
    ):
        self.region_dim = "region"
        self.label_da_name = label_da_name
        self._ds = self.read_binary_data(data, label_da_name)

        super().__init__(
            regions=self._ds.get_index("region"),
            genome=genome,
            save_dir=save_dir,
        )

        # setup input output datasets
        self.input_datasets.append("genome_one_hot")
        self._register_binary_label_dataset(load)

    @staticmethod
    def read_binary_data(task_data, label_da_name="y"):
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
                raise ValueError("Unknown file format {}".format(task_path))
        else:
            if isinstance(task_data, pd.DataFrame):
                task_data.index.name = "region"
                _ds = xr.Dataset({label_da_name: task_data})
        return _ds

    def _register_binary_label_dataset(self, load):
        self.add_region_dataset(
            name=self.label_da_name,
            da=self._ds[self.label_da_name],
            datatype="output",
            load=load,
            region_dim=self.region_dim,
        )
        self.output_datasets.append(self.label_da_name)


class TrackDataset(GenomeDataset):
    def __init__(
        self,
        genome,
        regions,
        save_dir=None,
    ):
        super().__init__(genome=genome, regions=regions, save_dir=save_dir)

    def add_position_dataset(self, *args, **kwargs):
        norm_func = kwargs.pop("norm_func", None)
        
        self.add_position_dataset(*args, **kwargs)