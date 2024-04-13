import pathlib
import numpy as np
from collections import defaultdict
from typing import List, Union, Optional
import pyarrow
import joblib

import ray
from pyarrow.fs import FileSystem

from .filters import RowSumFilter
from .transforms import BatchCropRegions, BatchReverseComplement, BatchToFloat, RowFootprint


DNA_NAME = "dna_one_hot"
REGION_IDS_NAME = "region_ids"


class RayGenomeDataset:
    """RayDataset class for working with ray.data.Dataset objects."""

    def __init__(self, dataset) -> None:
        """
        Initialize a RayDataset object.

        Parameters
        ----------
        dataset : ray.data.Dataset or str or pathlib.Path or list
            The input dataset. It can be a ray.data.Dataset object, a string or
            pathlib.Path representing the path to a parquet file, or a list of
            parquet file paths.

        Returns
        -------
        None
        """
        if isinstance(dataset, (str, pathlib.Path, list)):
            dataset = ray.data.read_parquet(dataset, file_extensions=["parquet"])
        self.input_files: List[str] = dataset.input_files()
        self.file_system: FileSystem = self._get_filesystem()
        self.stats_files: List[str] = self._get_stats_files()
        self._summary_stats: Union[None, dict] = None
        self.dataset: ray.data.Dataset = dataset

        _schema = dataset.schema()
        self.schema: dict = {k: v for k, v in zip(_schema.names, _schema.types)}
        self.dna_name = DNA_NAME
        self.region_ids_name = REGION_IDS_NAME
        self.regions, self.samples = self._parse_regions_and_samples()
        self.columns = list(self.schema.keys())

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
                    self._summary_stats = joblib.load(f)
            else:
                summary_stats = defaultdict(list)
                for stats_file in self.stats_files:
                    with self.file_system.open_input_file(stats_file) as f:
                        stats = joblib.load(f)
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


class scPrinterDataset(RayGenomeDataset):
    """RayDataset class for working with scPrinter model."""

    def __init__(
        self,
        dataset: ray.data.Dataset,
        bias_name: str = None,
        dna_window: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        min_counts: int = 10,
        max_counts: Optional[int] = None,
        reverse_complement: bool = True,
    ) -> None:
        """
        Initialize a scPrinterDataset object.

        Parameters
        ----------
        dataset : ray.data.Dataset
            The Ray dataset.
        bias_name : str, optional
            The name of the bias.
        dna_window : int, optional
            The size of the DNA window.
        signal_window : int, optional
            The size of the signal window.
        max_jitter : int, optional
            The maximum jitter value.
        min_counts : int, optional
            The minimum counts value.
        max_counts : int, optional
            The maximum counts value.
        reverse_complement : bool, optional
            Whether to use reverse complement.

        Returns
        -------
        None
        """
        super().__init__(dataset)

        if bias_name is None:
            # guess the bias name
            _names = [s for s in self.samples if "bias" in s.lower()]
            if len(_names) == 1:
                self.bias_name = _names[0]
            else:
                raise ValueError(
                    "Bias name not provided and could not be guessed, please provide the bias name."
                )
        self.bias_name = bias_name
        # remove bias name from samples
        self.samples = [s for s in self.samples if s != self.bias_name]

        # region properties
        self.dna_window = dna_window
        self.signal_window = signal_window
        self.max_jitter = max_jitter
        self.min_counts = min_counts
        self.max_counts = 1e16 if max_counts is None else max_counts
        self.reverse_complement = reverse_complement

    def set_min_max_counts_cutoff(self, column: str):
        _stats = self.summary_stats[column]
        min_, max_ = np.quantile(_stats, 0.0001), np.quantile(_stats, 0.9999)
        self.min_counts = max(self.min_counts, min_)
        self.max_counts = min(self.max_counts, max_)
        return

    def filter_by_coverage(self, column):
        self.set_min_max_counts_cutoff(column)
        
