import os
import pathlib
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pyarrow
import ray
from pyarrow.fs import FileSystem
from ray.data.dataset import Dataset

DNA_NAME = "dna_one_hot"
REGION_IDS_NAME = "region_ids"

# set environment variable to ignore unhandled errors
RAY_IGNORE_UNHANDLED_ERRORS = 1
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = str(RAY_IGNORE_UNHANDLED_ERRORS)


class RayGenomeDataset:
    """RayDataset class for working with ray.data.Dataset objects."""

    def __init__(
        self,
        dataset: Union[ray.data.Dataset, str, pathlib.Path, List[str]],
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a RayDataset object.

        Parameters
        ----------
        dataset : ray.data.Dataset or str or pathlib.Path or list
            The input dataset. It can be a ray.data.Dataset object, a string or
            pathlib.Path representing the path to a parquet file, or a list of
            parquet file paths.
        columns : list, optional
            The list of columns to select, if None, all columns are selected (default is None).
        **kwargs
            Additional keyword arguments passed to ray.data.read_parquet.

        Returns
        -------
        None
        """
        if isinstance(dataset, (str, pathlib.Path, list)):
            dataset = ray.data.read_parquet(
                dataset, file_extensions=["parquet"], columns=columns, **kwargs
            )
        self.input_files: List[str] = dataset.input_files()
        self.file_system: FileSystem = self._get_filesystem()
        self.stats_files: List[str] = self._get_stats_files()
        self._summary_stats: Union[None, dict] = None
        self.dataset: Dataset = dataset

        _schema = dataset.schema()
        self.schema: dict = dict(zip(_schema.names, _schema.types))
        self.dna_name: str = DNA_NAME
        self.region_ids_name: str = REGION_IDS_NAME
        self.regions: List[str] = self._parse_regions_and_samples()[0]
        self.samples: List[str] = self._parse_regions_and_samples()[1]
        self.columns: List[str] = list(self.schema.keys())

        # working dataset for producing data loaders
        self._dataset_mode = None
        self._working_dataset = None
        return

    def __repr__(self) -> str:
        return self.dataset.__repr__()

    def train(self) -> None:
        """
        Set the dataset mode to "train".

        Returns
        -------
        None
        """
        self._dataset_mode = "train"
        return

    def eval(self) -> None:
        """
        Set the dataset mode to "eval".

        Returns
        -------
        None
        """
        self._dataset_mode = "eval"
        return

    def __len__(self) -> int:
        return self.dataset.count()

    def _get_filesystem(self) -> FileSystem:
        """
        Get the filesystem associated with the dataset.

        Returns
        -------
        FileSystem
            The filesystem object.
        """
        _path = self.input_files[0]
        try:
            fs, _ = FileSystem.from_uri(_path)
        except pyarrow.ArrowInvalid:
            fs = pyarrow.fs.LocalFileSystem()
        return fs

    def _get_stats_files(self) -> List[str]:
        """
        Get the statistics files associated with the dataset.

        Returns
        -------
        List[str]
            The list of statistics files.
        """
        stats_dirs = set()
        for file in self.input_files:
            stats_dir = "/".join(file.split("/")[:-2]) + "/stats"
            stats_dirs.add(stats_dir)
        stats_files = []
        for stats_dir in stats_dirs:
            stats_files.append(f"{stats_dir}/summary_stats.npz")
        return stats_files

    @property
    def summary_stats(self) -> dict:
        """
        Get the summary statistics for the dataset.

        Returns
        -------
        dict
            The summary statistics.
        """
        if self._summary_stats is None:
            if len(self.stats_files) == 0:
                return None
            elif len(self.stats_files) == 1:
                with self.file_system.open_input_file(self.stats_files[0]) as f:
                    self._summary_stats = dict(np.load(f))
            else:
                summary_stats = defaultdict(list)
                for stats_file in self.stats_files:
                    with self.file_system.open_input_file(stats_file) as f:
                        stats = dict(np.load(f))
                        for key, val in stats.items():
                            summary_stats[key].append(val)
                self._summary_stats = {
                    key: np.concatenate(val) for key, val in summary_stats.items()
                }
        return self._summary_stats

    def _parse_regions_and_samples(self):
        """
        Parse regions and samples from the dataset.
        """
        regions = set()
        samples = set()
        for name in self.schema.keys():
            if name == self.region_ids_name:
                continue
            else:
                try:
                    region, sample = name.split("|")
                except ValueError:
                    continue
                regions.add(region)
                if sample != self.dna_name:
                    samples.add(sample)
        return list(regions), list(samples)
