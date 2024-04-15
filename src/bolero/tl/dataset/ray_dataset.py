import pathlib
from collections import defaultdict
from typing import List, Union, Tuple, Optional
import joblib
import numpy as np
import pyarrow
import ray
from pyarrow.fs import FileSystem
from ray.data.dataset import Dataset
import os

from .filters import RowSumFilter
from .transforms import (
    CropRegionsWithJitter,
    ReverseComplement,
    BatchToFloat,
    BatchFootPrint,
)

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


class scPrinterDataset(RayGenomeDataset):
    """
    RayDataset class for working with scPrinter model.

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
    reverse_complement : bool, optional
        Whether to use reverse complement.

    Attributes
    ----------
    _working_dataset : ray.data.Dataset
        The working dataset used for filter and map operations.
    dna_name : str
        The name of the DNA.
    region_ids_name : str
        The name of the region IDs.
    min_counts : int
        The minimum counts value.
    max_counts : int
        The maximum counts value.

    Methods
    -------
    set_min_max_counts_cutoff(column: str, min_q=0.0001, max_q=0.9999) -> None:
        Set the minimum and maximum counts cutoff based on the given column.
    _filter_by_coverage(column) -> None:
        Filter the working dataset based on the coverage of the given column.
    dna_to_float() -> None:
        Convert the DNA data to float.
    crop_regions() -> None:
        Crop the regions in the working dataset.
    reverse_complement() -> None:
        Reverse complement the DNA sequences.

    """

    def __init__(
        self,
        dataset: Dataset,
        columns: Optional[List[str]] = None,
        bias_name: str = None,
        batch_size=64,
        dna_window: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        reverse_complement: bool = True,
        modes=None,
        clip_min=-10,
        clip_max=10,
    ) -> None:
        """
        Initialize a scPrinterDataset object.

        Parameters
        ----------
        dataset : ray.data.Dataset
            The Ray dataset.
        columns : list, optional
            The list of columns to select, if None, all columns are selected (default is None).
        bias_name : str, optional
            The name of the bias.
        dna_window : int, optional
            The size of the DNA window.
        signal_window : int, optional
            The size of the signal window.
        max_jitter : int, optional
            The maximum jitter value.
        reverse_complement : bool, optional
            Whether to use reverse complement.
        modes : np.ndarray, optional
            Modes for the footprint transformation.
        clip_min : float, optional
            Minimum value for clipping footprint zscores (default is -10).
        clip_max : float, optional
            Maximum value for clipping footprint zscores (default is 10).
        purpose : str, optional
            The purpose of the dataset, either "train" or "eval" (default is "train").
            In the eval model, random jitter, reverse complement are disabled.

        Returns
        -------
        None
        """
        super().__init__(dataset, columns=columns)
        # all filter and map operations will be done on this working dataset

        if bias_name is None:
            # guess the bias name
            _names = [s for s in self.samples if "bias" in s.lower()]
            if len(_names) == 1:
                self.bias_name = _names[0]
            else:
                raise ValueError(
                    "Bias name not provided and could not be guessed, please provide the bias name."
                )
        else:
            self.bias_name = bias_name
        # remove bias name from samples
        self.samples = [s for s in self.samples if s != self.bias_name]

        self.batch_size = batch_size

        # region properties
        self.dna_window = dna_window
        self.signal_window = signal_window
        self.max_jitter = max_jitter
        self.min_counts = 10
        self.max_counts = 1e16
        self.reverse_complement = reverse_complement

        # footprint properties
        self.modes = modes if modes is not None else np.arange(2, 101, 1)
        self.clip_min = clip_min
        self.clip_max = clip_max

        self._dataset_mode = None
        self._working_dataset = None
        self.add_footprint = self._get_footprint_func()
        return

    def train(self):
        self._dataset_mode = "train"
        return

    def eval(self):
        self._dataset_mode = "eval"
        return

    def _dataset_preprocess(self, column) -> None:
        """
        Preprocess the dataset.

        Returns
        -------
        None
        """

        # row operations
        self._filter_by_coverage(column)
        if self.reverse_complement and self._dataset_mode == "train":
            self._reverse_complement_region()
        self._crop_regions()
        # batch operations
        self._dna_to_float()
        return

    def get_dataloader(self, sample=None, region=None):
        if self._dataset_mode is None:
            raise ValueError(
                "Set .train() or .eval() first before calling .get_dataloader()"
            )

        if sample is None:
            if len(self.samples) == 1:
                sample = self.samples[0]
            else:
                raise ValueError(
                    "Sample name not provided and could not be guessed, please provide the sample name."
                )

        if region is None:
            if len(self.regions) == 1:
                region = self.regions[0]
            else:
                raise ValueError(
                    "Region name not provided and could not be guessed, please provide the region name."
                )

        filter_column = f"{region}|{sample}"

        self._working_dataset = self.dataset
        self._dataset_preprocess(filter_column)

        for batch in self._working_dataset.iter_torch_batches(
            batch_size=self.batch_size
        ):
            batch = self.add_footprint(batch)
            yield batch

    def set_min_max_counts_cutoff(
        self, column: str, min_q=0.0001, max_q=0.9999
    ) -> None:
        """
        Set the minimum and maximum counts cutoff based on the given column.

        Parameters
        ----------
        column : str
            The column name.
        min_q : float, optional
            The minimum quantile value (default is 0.0001).
        max_q : float, optional
            The maximum quantile value (default is 0.9999).

        Returns
        -------
        None
        """
        _stats = self.summary_stats[column]
        min_, max_ = np.quantile(_stats, min_q), np.quantile(_stats, max_q)
        self.min_counts = max(self.min_counts, min_)
        self.max_counts = min(self.max_counts, max_)
        return

    def get_dna_and_signal_columns(
        self, separate_bias: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Get the DNA and signal columns from the dataset.

        Parameters
        ----------
        separate_bias : bool, optional
            Whether to separate the bias columns (default is False).

        Returns
        -------
        Tuple[List[str], List[str]]
            A tuple containing the DNA columns and signal columns.
        """
        dna_columns = []
        signal_columns = []
        bias_columns = []
        for column in self.columns:
            try:
                _, sample = column.split("|")
            except ValueError:
                continue
            if sample == self.dna_name:
                dna_columns.append(column)
            elif sample == self.bias_name:
                bias_columns.append(column)
            else:
                signal_columns.append(column)
        if separate_bias:
            return dna_columns, bias_columns, signal_columns
        else:
            signal_columns = signal_columns + bias_columns
            return dna_columns, signal_columns

    def _filter_by_coverage(self, column: str) -> None:
        """
        Filter the working dataset based on the coverage of the given column.

        Parameters
        ----------
        column : str
            The column name.

        Returns
        -------
        None
        """
        self.set_min_max_counts_cutoff(column)
        _filter = RowSumFilter(column, self.min_counts, self.max_counts)
        self._working_dataset = self._working_dataset.filter(_filter)
        return

    def _dna_to_float(self) -> None:
        """
        Convert the DNA data to float.

        Returns
        -------
        None
        """
        dna_columns, _ = self.get_dna_and_signal_columns()
        _map = BatchToFloat(dna_columns)
        self._working_dataset = self._working_dataset.map_batches(_map)
        return

    def _reverse_complement_region(self, *args, **kwargs) -> None:
        """
        Reverse complement the DNA sequences by 50% probability.

        Returns
        -------
        None
        """
        dna_columns, signal_columns = self.get_dna_and_signal_columns()
        _rc = ReverseComplement(
            dna_key=dna_columns, signal_key=signal_columns, input_type="row"
        )
        self._working_dataset = self._working_dataset.map(_rc, *args, **kwargs)
        return

    def _crop_regions(self, *args, **kwargs) -> None:
        """
        Crop the regions in the working dataset.

        Returns
        -------
        None
        """
        if self._dataset_mode != "train":
            max_jitter = 0
        else:
            max_jitter = self.max_jitter

        dna_columns, signal_columns = self.get_dna_and_signal_columns()
        key_list = dna_columns + signal_columns
        length_list = [self.dna_window] * len(dna_columns) + [self.signal_window] * len(
            signal_columns
        )

        _cropper = CropRegionsWithJitter(
            key=key_list,
            final_length=length_list,
            max_jitter=max_jitter,
            input_type="row",
        )
        self._working_dataset = self._working_dataset.map(_cropper, *args, **kwargs)
        return

    def _get_footprint_func(self) -> None:
        """
        Compute the footprint.

        Returns
        -------
        None
        """
        _, bias_columns, signal_columns = self.get_dna_and_signal_columns(
            separate_bias=True
        )
        assert len(bias_columns) == 1, f"Bias columns must be one, got {bias_columns}"

        _footprint = BatchFootPrint(
            atac_key=signal_columns,
            bias_key=bias_columns[0],
            modes=self.modes,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            numpy=False,
        )
        return _footprint
