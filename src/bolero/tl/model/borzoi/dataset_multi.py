from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import ray

from bolero import Genome
from bolero.tl.dataset.ray_dataset import _IterableFromIterator
from bolero.tl.dataset.transforms import (
    FetchRegionOneHot,
)
from bolero.tl.generic.dataset import GenericDataset
from bolero.tl.model.borzoi.utils import MultiBorzoiRegions
from bolero.tl.pseudobulk.paired_pseudobulk import MultiPairedPseudobulker

DNA_NAME = "dna_one_hot"


class DatasetRecordManager:
    """
    Manages dataset records for Borzoi multi-dataset.

    This class handles loading dataset records of multiple datasets,
    each record follows the schema:
    {
        "data_path": str,  # Path to the dataset file
        "pseudobulk_path": str,  # Path to the pseudobulk file
        "genome": str,  # Genome name
        "dataset_sample_weights": int | float,  # Optional, Sample weights for pseudobulk
    }
    """

    # data
    dataset_records: dict[str, dict]
    _data_paths: dict[str, str]
    sample_region_fracs: pd.Series
    _barcode_orders: dict[str, dict]
    # genome
    _shared_genome_obj: dict[str, Genome]
    genomes: dict[str, Genome]
    borzoi_regions: MultiBorzoiRegions
    # pseudobulker
    _pseudobulk_paths: dict[str, str]
    pseudobulker: MultiPairedPseudobulker

    def __init__(self, dataset_records: str | dict, pseudobulker_cls: str | type):
        if isinstance(dataset_records, str):
            dataset_records = joblib.load(dataset_records)
        self.dataset_records = dataset_records

        self._init_data()
        self._init_genome()
        self._init_pseudobulker(pseudobulker_cls)

    def _init_data(self):
        self._data_paths = {k: v["data_path"] for k, v in self.dataset_records.items()}
        self.data_keys = list(self._data_paths.keys())
        self.sample_region_fracs = pd.Series(
            {
                k: v.get("dataset_sample_weights", 1)
                for k, v in self.dataset_records.items()
            }
        )
        # sample weights are used to downsample regions with fraction,
        # max sample weight is 1, values are between 0 and 1
        self.sample_region_fracs = (
            self.sample_region_fracs / self.sample_region_fracs.max()
        )

        self._barcode_orders = {
            k: joblib.load(f"{v}/row_names.joblib") for k, v in self._data_paths.items()
        }

    def _init_genome(self):
        self._shared_genome_obj = {}
        self.genomes = {}
        for k, v in self.dataset_records.items():
            _g = v["genome"]
            if _g not in self._shared_genome_obj:
                self._shared_genome_obj[_g] = Genome(_g)
            self.genomes[k] = self._shared_genome_obj[_g]
        self.borzoi_regions = MultiBorzoiRegions(self.genomes)

    def _init_pseudobulker(self, pseudobulker_cls: str | type):
        self._pseudobulk_paths = {
            k: v["pseudobulk_path"] for k, v in self.dataset_records.items()
        }
        self.pseudobulker = MultiPairedPseudobulker(
            pseudobulk_path_dict=self._pseudobulk_paths,
            barcode_order_dict=self._barcode_orders,
            pseudobulker_cls=pseudobulker_cls,
        )

    def _sample_regions(self, regions, seed):
        use_regions = []
        for key, key_regions in regions.groupby("key"):
            frac = self.sample_region_fracs[key]
            use_key_regions = key_regions.sample(
                frac=frac, random_state=seed, replace=False
            )
            use_regions.append(use_key_regions)
        return pd.concat(use_regions, ignore_index=True)

    def get_train_valid_test_regions(
        self, split_id: int, seed: int, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the train, valid, and test folds and regions for the given split_id.
        Then sample the regions for each dataset key based on the sample_region_fracs.
        """
        train_regions, valid_regions, test_regions = (
            self.borzoi_regions.get_train_valid_test_regions(
                split_id=split_id, **kwargs
            )
        )
        rng = np.random.default_rng(seed)
        # each time we get a different set of train regions
        # but the valid and test regions are fixed
        use_train_regions = self._sample_regions(
            train_regions, seed=rng.integers(0, 2**32)
        )
        use_valid_regions = self._sample_regions(valid_regions, seed=seed)
        use_test_regions = self._sample_regions(test_regions, seed=seed)
        return use_train_regions, use_valid_regions, use_test_regions

    def get_fasta_dict(self) -> dict[str, str]:
        """
        Returns a dictionary of genome names and their corresponding FASTA paths.
        """
        return {k: v.fasta_path for k, v in self.genomes.items()}


class BorzoiMultiDataset(GenericDataset):
    """Single cell pseudobulk dataset for Borzoi model."""

    default_config = {
        "dataset_records": "REQUIRED",
        "paired_mode": "ensemble",
        "batch_size": 4,
        "dna_window": 524288,
        "max_jitter": 3,
    }

    def __init__(
        self,
        dataset_records: str | dict,
        paired_mode: str = "ensemble",
        batch_size: int = 4,
        dna_window: int = 524288,
        max_jitter: int = 3,
    ):
        self.dataset_record_manager = DatasetRecordManager(
            dataset_records=dataset_records,
            pseudobulker_cls=paired_mode,
        )
        self.dm = self.dataset_record_manager
        self.borzoi_regions = self.dataset_record_manager.borzoi_regions

        self.paired_mode = paired_mode
        self.batch_size = batch_size
        self.dna_window = dna_window
        self.max_jitter = max_jitter

        self._block_size = 20
        self._max_blocks = 200
        return

    def _get_dna_one_hot(self, dataset, concurrency, batch_size=16):
        fn = FetchRegionOneHot
        fn_constructor_kwargs = {
            "random_shift": self.max_jitter if self.is_train() else 0,
            "fasta_path": self.dm.get_fasta_dict(),
            "dtype": "bool",
        }
        fn_kwargs = {}

        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            fn_kwargs=fn_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        self.dna_column = DNA_NAME
        return dataset

    def _get_pseudobulk(self, dataset, concurrency, batch_size=16):
        """
        Get the pseudobulk data for the dataset.
        """
        # TODO: Try GenomeParquetDB see if parallel actor pool is compatible with dataloaders
        # If not, use a naive version of the GenomeParquetDB
        raise NotImplementedError

    def get_processed_dataset(self, region_bed: pd.DataFrame, concurrency: int):
        """Create a processed dataset from the region BED file."""
        region_bed = region_bed.rename(
            columns={"key": "__dataset_keys__", "Name": "region"}
        )
        region_bed = region_bed[["__dataset_keys__", "region"]].reset_index(drop=True)

        n_blocks = min(len(region_bed) // self._block_size + 1, self._max_blocks)
        dataset = ray.data.from_pandas(region_bed).repartition(n_blocks).materialize()

        # add dna
        dataset = self._get_dna_one_hot(
            dataset=dataset,
            concurrency=1,
        )
        return dataset

    def _get_dataloader_with_wrapper(
        self,
        dataset_kwargs: dict,
        data_iter_kwargs: dict,
        batch_size=8,
        n_batches=None,
        as_torch=False,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader generator.

        The dataset will be init only when entering the __iter__ method.
        """

        # this is adapted from the ray.data.iterator.DataIterator.iter_batches
        # https://github.com/ray-project/ray/blob/master/python/ray/data/iterator.py#L106
        def _create_iterator():
            work_ds = self.get_processed_dataset(**dataset_kwargs)

            if n_batches is not None:
                n_rows = (n_batches + 1) * batch_size
                work_ds = work_ds.limit(n_rows)

            _kwargs = {
                "batch_size": batch_size,
                "prefetch_batches": 3,
                "drop_last": True,  # helps to avoid the last batch with less than batch_size
            }
            _kwargs.update(data_iter_kwargs)
            print("Data loader kwargs", _kwargs)

            if as_torch:
                loader = work_ds.iter_torch_batches(**_kwargs)
            else:
                loader = work_ds.iter_batches(**_kwargs)

            yield from loader

        # the dataset and dataloader are created lazily, until __iter__ is called
        return _IterableFromIterator(_create_iterator)

    def get_dataloader(
        self,
        region_bed: str,
        as_torch: bool = True,
        n_batches: int = None,
        concurrency: int = 8,
        **dataloader_kwargs: Any,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader.

        Parameters
        ----------
        folds : list
            List of folds to include in the dataset.
        region_bed : str
            Path to the BED file containing the regions.
        as_torch : bool, optional
            Whether to return a PyTorch dataloader, by default True.
        n_batches : int, optional
            Number of batches to return, by default None.
        batch_size : int, optional
            Batch size, by default None.
        shuffle_rows : int, optional
            The size of the local shuffle buffer, by default 500.
        concurrency : int, optional
            The number of workers to use for processing the dataset, by default 32.
        **dataloader_kwargs
            Additional keyword arguments pass to ray.data.Dataset.iter_batches.

        Returns
        -------
        DataLoader
            The dataloader.
        """
        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "region_bed": region_bed,
            "concurrency": concurrency,
        }
        data_iter_kwargs = dataloader_kwargs

        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            as_torch=as_torch,
            n_batches=n_batches,
            batch_size=self.batch_size,
        )
        return loader


# TODO:
# 2. load dna of corresponding regions
# 3. load pseudobulk data of corresponding regions in the dataset return a list of dict, see how to parallelize this
# 4. down stream borzoi dataset steps like previous
# Dedup train regions as BorzoiDataset do
