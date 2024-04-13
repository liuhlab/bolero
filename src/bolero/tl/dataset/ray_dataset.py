import pathlib
import numpy as np
from collections import defaultdict
from typing import List, Union
import pyarrow

import ray
from pyarrow.fs import FileSystem


class RayGenomeDataset:
    """RayDataset class for working with ray.data.Dataset objects."""

    def __init__(self, dataset) -> None:
        """
        Initialize a RayDataset object.

        Parameters
        ----------
            dataset (ray.data.Dataset): The Ray dataset.

        Returns
        -------
            None
        """
        if isinstance(dataset, (str, pathlib.Path, list)):
            dataset = ray.data.read_parquet(dataset, file_extensions=["parquet"])
        self.input_files = dataset.input_files()
        self.file_system = self._get_filesystem()
        self.stats_files = self._get_stats_files()
        self._summary_stats = None
        self.dataset = dataset

    def _get_filesystem(self) -> FileSystem:
        """
        Get the filesystem associated with the dataset.

        Returns
        -------
            FileSystem: The filesystem object.
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
            List[str]: The list of statistics files.
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
    def summary_stats(self):
        """
        Get the summary statistics for the dataset.

        Returns
        -------
            dict: The summary statistics.
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
                    stats = np.load(stats_file)
                    for key, val in stats.items():
                        summary_stats[key].append(val)
                self._summary_stats = {
                    key: np.concatenate(val) for key, val in summary_stats.items()
                }
        return self._summary_stats


class scPrinterDataset(RayGenomeDataset):
    """RayDataset class for working with scPrinter model."""

    def __init__(self, dataset: ray.data.Dataset) -> None:
        """
        Initialize a scPrinterDataset object.

        Parameters
        ----------
            dataset (ray.data.Dataset): The Ray dataset.
            device (str): The device to use for the dataset.

        Returns
        -------
            None
        """
        super().__init__(dataset)
