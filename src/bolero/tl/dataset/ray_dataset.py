import os
import pathlib
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pyarrow
import ray
from pyarrow.fs import FileSystem
from ray.data.dataset import Dataset

from bolero.pp.genome import Genome

from .transforms import FetchRegionOneHot

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
            if isinstance(dataset, list):
                _path = dataset[0]
            else:
                _path = dataset
            self.file_system: FileSystem = self._get_filesystem(_path)
            kwargs["filesystem"] = self.file_system

            dataset = ray.data.read_parquet(
                dataset, file_extensions=["parquet"], columns=columns, **kwargs
            )
            self.input_files: List[str] = dataset.input_files()
        else:
            self.input_files: List[str] = dataset.input_files()
            self.file_system: FileSystem = self._get_filesystem(self.input_files[0])

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

    @staticmethod
    def _get_filesystem(path) -> FileSystem:
        """
        Get the filesystem associated with the dataset.

        Returns
        -------
        FileSystem
            The filesystem object.
        """
        try:
            fs, _ = FileSystem.from_uri(path)
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


class RayRegionDataset(RayGenomeDataset):
    """
    A dataset class for working with genomic regions using Ray.

    Args:
        bed (pd.DataFrame or pr.PyRanges or str): The genomic regions in BED format.
        genome (Genome or str): The genome reference or its name.
        standard_length (int): The standard length of the regions.
        **kwargs: Additional keyword arguments for ray.data.from_pandas.

    Attributes
    ----------
        bed (pd.DataFrame): The standardized genomic regions.
        genome (Genome): The genome reference.
        dataset (ray.data.Dataset): The Ray dataset containing the genomic regions.
        _working_dataset (ray.data.Dataset): The working dataset for preprocessing.

    Methods
    -------
        get_dataloader: Get a data loader for iterating over batches of the dataset.

    """

    def __init__(self, bed, genome, standard_length, **kwargs):
        if isinstance(genome, str):
            genome = Genome(genome)

        standard_bed = genome.standard_region_length(
            bed,
            length=standard_length,
            boarder_strategy="drop",
            remove_blacklist=True,
            as_df=True,
            keep_original=True,
        )
        # ray data don't understand categorical dtype in pandas
        standard_bed["Chromosome"] = standard_bed["Chromosome"].astype(str)
        self.bed = standard_bed

        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome
        # trigger loading of genome one hot
        _ = self.genome.genome_one_hot

        self.dataset = ray.data.from_pandas(self.bed, **kwargs)
        self._working_dataset = None

    def _fetch_dna_one_hot(self):
        self._working_dataset = self._working_dataset.map_batches(
            FetchRegionOneHot(self.genome, region_key="Name", output_key="dna_one_hot")
        )
        return

    def _select_one_hot_and_region_name(self):
        keep_cols = ["Name", "Original_Name", "dna_one_hot"]
        self._working_dataset = self._working_dataset.select_columns(keep_cols)
        return

    def _dataset_preprocess(self):
        self._fetch_dna_one_hot()
        self._select_one_hot_and_region_name()
        return

    def get_dataloader(self, batch_size: int = 64, **kwargs):
        """
        Get a data loader for iterating over batches of the dataset.

        Args:
            batch_size (int): The batch size.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            DataLoader: The data loader.

        """
        self._working_dataset = self.dataset
        self._dataset_preprocess()
        loader = self._working_dataset.iter_batches(batch_size=batch_size, **kwargs)
        return loader
