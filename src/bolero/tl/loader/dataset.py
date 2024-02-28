import pandas as pd
import torch
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader, Dataset
from bolero.pp import Genome
from bolero.pp.genome import parse_region_names
import pathlib


def try_gpu():
    """
    Try to use GPU if available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


class BinaryDataset(Dataset):
    def __init__(
        self,
        task_data,
        genome,
        genome_dir=None,
        label_da_name="y",
        downsample=None,
        load=True,
    ):
        if isinstance(genome, Genome):
            self.genome = genome
        else:
            self.genome = Genome(genome, save_dir=genome_dir)
        self.region_dim = "region"
        self.label_da_name = label_da_name

        if isinstance(task_data, (str, pathlib.Path)):
            task_path = str(task_data)
            if task_path.endswith(".zarr"):
                self._ds = xr.open_zarr(task_path)
            elif task_path.endswith(".feather"):
                _df = pd.read_feather(task_path)
                _df.set_index(_df.columns[0], inplace=True)
                _df.index.name = self.region_dim
                self._ds = xr.Dataset({"y": _df})
            else:
                raise ValueError("Unknown file format {}".format(task_path))
        else:
            self._ds = task_data

        if load:
            self._ds = self._ds.load()
        if downsample is not None and (downsample > len(self)):
            _regions = self._ds.get_index(self.region_dim)
            # random downsample while keep the order
            sel_regions = np.random.choice(_regions, downsample, replace=False)
            self._ds = self._ds.sel({self.region_dim: _regions.isin(sel_regions)})

        self.regions = self._ds.get_index(self.region_dim)
        self.region_bed = parse_region_names(self.regions)

    def __len__(self):
        return self._ds.sizes[self.region_dim]

    def _isel_region(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        return self._ds[self.label_da_name].isel({self.region_dim: idx})

    def __getitem__(self, idx):
        # new input
        _data = self._isel_region(idx)
        label = torch.FloatTensor(_data.values).squeeze(0)

        regions = _data.get_index(self.region_dim)
        one_hot = self.genome.get_regions_one_hot(regions).squeeze(0)
        return one_hot, label

    def __repr__(self) -> str:
        class_str = f"{self.__class__.__name__} object with {len(self)} regions"
        genome_str = f"Genome: {self.genome.name}"
        return f"{class_str}\n{genome_str}"

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
        tri_datasets = [
            self.__class__(
                task_data=self._ds.sel(**{self.region_dim: region_sel}),
                genome=self.genome,
                genome_dir=self.genome.save_dir,
                label_da_name=self.label_da_name,
                load=False,
            )
            for region_sel in [train_regions, valid_regions, test_regions]
        ]
        tri_loaders = [
            DataLoader(
                dataset=ds,
                batch_size=batch_size,
                shuffle=sh,
                num_workers=0,  # DO NOT USE MULTIPROCESSING, it has issue with the genome object
            )
            for ds, sh in zip(tri_datasets, shuffle)
        ]
        return tri_loaders

class TrackDataset(Dataset):
    def __init__(
        self,
        task_data,
        genome,
        genome_dir=None,
        label_da_name="y",
        downsample=None,
        load=True,
    ):
        pass