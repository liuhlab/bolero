from collections import OrderedDict
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import ray

from bolero import Genome
from bolero.tl.dataset.parquet_db import GenomeParquetDBNoParallel
from bolero.tl.dataset.ray_dataset import _IterableFromIterator
from bolero.tl.dataset.transforms import FetchRegionOneHot, ReverseComplement
from bolero.tl.generic.dataset import GenericDataset
from bolero.tl.model.borzoi.utils import MultiBorzoiRegions
from bolero.tl.pseudobulk.paired_pseudobulk import MultiPairedPseudobulker
from bolero.utils import understand_regions

DNA_NAME = "dna_one_hot"


class _MaskBlacklistAndClamp:
    def __init__(
        self,
        blacklist_global_coords,
        chrom_offsets,
        region_key,
        data_keys,
        as_nan,
        resolution,
    ):
        self.blacklist_global_coords = blacklist_global_coords
        self.chrom_offsets = chrom_offsets
        self.region_key = region_key
        self.data_keys = data_keys
        self.as_nan = as_nan
        self.resolution = resolution
        self._mask_value = np.nan if as_nan else 0

    def _get_region_chrom(self, region):
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        chrom_start = self.chrom_offsets.loc[chrom, "global_start"]
        global_coords = (chrom_start + start, chrom_start + end)
        return global_coords

    def __call__(self, region, data):
        """
        data.shape = (n_pseudobulks, region_length), it should belong to the same region.
        """
        r_gstart, r_gend = self._get_region_chrom(region)
        bl_coords = self.blacklist_global_coords
        overlap_bl_regions = bl_coords[
            (bl_coords[:, 0] < r_gend) & (bl_coords[:, 1] > r_gstart)
        ].copy()
        overlap_bl_regions -= r_gstart
        region_length = r_gend - r_gstart
        overlap_bl_regions[:, 0] -= 3 * self.resolution
        overlap_bl_regions[:, 1] += 3 * self.resolution
        overlap_bl_regions = np.clip(overlap_bl_regions, 0, region_length)

        # convert base coords to resolution coords
        overlap_bl_regions //= self.resolution

        for bl_start, bl_end in overlap_bl_regions:
            if bl_start >= region_length:
                continue
            nan_slice = slice(bl_start, bl_end)
            data[:, nan_slice] = self._mask_value
        return data


class MultiMaskBlacklistAndClamp:
    def __init__(
        self,
        blacklist_global_coords: dict[str, np.ndarray],
        chrom_offsets: dict[str, pd.Series],
        region_key: str,
        data_keys: list[str],
        as_nan: bool = False,
        resolution: int = 32,
    ):
        genome_actors = {}
        for key, bl_glob_coords in blacklist_global_coords.items():
            this_chrom_offsets = chrom_offsets[key]
            genome_actors[key] = _MaskBlacklistAndClamp(
                blacklist_global_coords=bl_glob_coords,
                chrom_offsets=this_chrom_offsets,
                region_key=region_key,
                data_keys=data_keys,
                as_nan=as_nan,
                resolution=resolution,
            )
        self.genome_actors = genome_actors

    def __call__(self, key, region, data):
        """Mask the blacklist regions in the dataset."""
        if key not in self.genome_actors:
            return data  # no masking needed for this dataset
        masker = self.genome_actors[key]
        return masker(region=region, data=data)


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
        self.dataset_orders = list(self.dataset_records.keys())

        self._init_data()
        self._init_genome()
        self._init_pseudobulker(pseudobulker_cls)

    def _init_data(self):
        self._data_paths = {k: v["data_path"] for k, v in self.dataset_records.items()}
        self.data_keys = list(self._data_paths.keys())
        self.dataset_sample_weights = pd.Series(
            {
                k: v.get("dataset_sample_weights", 1)
                for k, v in self.dataset_records.items()
            }
        )
        # sample weights are used to balance n_pseudobulks across datasets,
        # for each region, n_pseudobulks * weight will be sampled for the corresponding dataset
        # max sample weight is 1, values are between 0 and 1
        self.dataset_sample_weights: dict[str, float] = (
            self.dataset_sample_weights / self.dataset_sample_weights.max()
        ).to_dict()

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
            frac = self.dataset_sample_weights[key]
            use_key_regions = key_regions.sample(
                frac=frac, random_state=seed, replace=False
            )
            use_regions.append(use_key_regions)
        return pd.concat(use_regions, ignore_index=True)

    def _select_full_overlap_regions(self, regions: pd.DataFrame) -> pd.DataFrame:
        # Select regions that fully overlap with the parquet bed files.
        parquet_bed_dict = {}
        for k, p in self._data_paths.items():
            parquet_bed = pd.read_feather(f"{p}/parquet_row_regions.feather")
            parquet_bed = pr.PyRanges(understand_regions(parquet_bed["region"])).merge()
            parquet_bed_dict[k] = parquet_bed

        full_overlap_regions = []
        for key, key_regions in regions.groupby("key"):
            key_regions = pr.PyRanges(key_regions).overlap(
                parquet_bed_dict[key], how="containment"
            )
            full_overlap_regions.append(key_regions.df)
        regions = pd.concat(full_overlap_regions)
        return regions.reset_index(drop=True)

    def get_train_valid_test_regions(
        self, split_id: int, seed: int, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the train, valid, and test folds and regions for the given split_id.
        Then sample the regions for each dataset key based on the sample_region_fracs.
        """
        train_df, valid_df, test_df = self.borzoi_regions.get_train_valid_test_regions(
            split_id=split_id, **kwargs
        )
        train_df = self._select_full_overlap_regions(train_df)
        valid_df = self._select_full_overlap_regions(valid_df)
        test_df = self._select_full_overlap_regions(test_df)
        return train_df, valid_df, test_df

    def get_fasta_dict(self) -> dict[str, str]:
        """
        Returns a dictionary of genome names and their corresponding FASTA paths.
        """
        return {k: v.fasta_path for k, v in self.genomes.items()}

    def get_blacklist_global_coords(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary of genome names and their corresponding blacklist global coordinates.
        """
        return {
            k: g.get_global_coords(g.blacklist_bed)
            for k, g in self.genomes.items()
            if g.blacklist_bed is not None
        }

    def get_chrom_offsets(self) -> dict[str, pd.Series]:
        """
        Returns a dictionary of genome names and their corresponding chromosome offsets.
        """
        return {k: g.chrom_offsets for k, g in self.genomes.items()}


class GenerateMultiGenomeParquetAndPseudobulk:
    """
    This class maintains multiple Genome Parquet DBs for data and a MultiPairedPseudobulker for pseudobulk information.

    Given a batch of input, it uses the dataset key and regions to fetch the corresponding parquet DB.
    And it sample N pseudobulk records from the corresponding dataset's pseudobulk records.
    In the end, it forms a list of dicts, each dict contains one region data for one pseudobulk record.
    """

    def __init__(
        self,
        parquet_paths: dict[str, str],
        pseudobulker: MultiPairedPseudobulker,
        dataset_orders: list[str],
        n_pseudobulks: int = 10,
        region_key="region",
        dataset_key="__dataset_keys__",
        normalize_cov: bool = True,
        output_prefix: str = "pseudobulk",
        blacklist_global_coords: dict[str, np.ndarray] = None,
        chrom_offsets: dict[str, pd.Series] = None,
        dataset_sample_weight: dict[str, float] = None,
    ):
        self.pseudobulker = pseudobulker
        self.n_pseudobulks = n_pseudobulks
        self.dataset_sample_weight = dataset_sample_weight or {}

        self.region_key = region_key
        self.dataset_key = dataset_key
        self.dataset_orders = dataset_orders

        self.parquet_db_dict: dict[str, GenomeParquetDBNoParallel] = {}
        for key, data_path in parquet_paths.items():
            parquet_db = GenomeParquetDBNoParallel(
                dataset_dir=data_path,
                # Do not provide merge plan for this actor
                # as the pseudobulk records are randomly sampled from the pseudobulker
                # merge plan will be dynamically created in the __call__ method
                merge_plan=None,
            )
            self.parquet_db_dict[key] = parquet_db

        self.normalize_cov = normalize_cov
        self.output_prefix = output_prefix

        # mask blacklist regions (optional)
        blacklist_global_coords = blacklist_global_coords or {}
        self.chrom_offsets = chrom_offsets or {}
        self.bl_masker = MultiMaskBlacklistAndClamp(
            blacklist_global_coords=blacklist_global_coords,
            chrom_offsets=self.chrom_offsets,
            region_key=self.region_key,
            data_keys=[
                f"{self.output_prefix}:bulk_data_0",
                f"{self.output_prefix}:bulk_data_1",
            ],
            as_nan=False,
            resolution=32,
        )

    def _sample_pseudobulks_and_get_data(self, key: str, region: str) -> list[dict]:
        """
        1. Sample N pseudobulk records from the pseudobulker.
        2. Create a merge plan for the sampled pseudobulks.
        3. Update the parquet DB with the merge plan.
        4. Return the list of dict contain data and pseudobulk records.
        """
        dataset_prefix = f"{key}.MetaCell"

        # sample pseudobulk, weights are applied to the number of pseudobulks
        ds_weight = self.dataset_sample_weight.get(key, 1)
        _this_n_pseudobulks = max(1, int(self.n_pseudobulks * ds_weight))
        pseudobulk_pairs = self.pseudobulker.take(_this_n_pseudobulks, key=key)

        # make merge plan
        parquet_db: GenomeParquetDBNoParallel = self.parquet_db_dict[key]
        merge_plan = OrderedDict()
        for pidx, (p0, p1) in enumerate(pseudobulk_pairs):
            merge_plan[f"{pidx}_{0}"] = list(p0["cluster_ids"].values())[0]
            merge_plan[f"{pidx}_{1}"] = list(p1["cluster_ids"].values())[0]
        parquet_db.update_row_actor(merge_plan, convert_row_name=False)

        # get region data
        all_pseudobulk_data = parquet_db.query_regions(np.array([region]))[0][
            dataset_prefix
        ].toarray()

        # apply blacklist mask (if applicable)
        all_pseudobulk_data = self.bl_masker(
            key=key, region=region, data=all_pseudobulk_data
        )

        # merge pair pseudobulks into single dict, add corresponding data
        new_batches = self._prepare_paired_pseudobulk_dict(
            pseudobulk_pairs=pseudobulk_pairs,
            all_pseudobulk_data=all_pseudobulk_data,
            dataset_prefix=dataset_prefix,
        )
        return new_batches

    def _prepare_paired_pseudobulk_dict(
        self, pseudobulk_pairs, all_pseudobulk_data, dataset_prefix
    ):
        new_batches = []
        suffix_list = ["_0", "_1"]
        output_prefix = self.output_prefix
        for pair_idx, cond_pair_pseudobulks in enumerate(pseudobulk_pairs):
            # create a new batch for each pseudobulk pair
            this_bulk_dict = {}
            for idx_in_pair, pseudobulk in enumerate(cond_pair_pseudobulks):
                suffix = suffix_list[idx_in_pair]
                # 1. add condition embedding
                if "__conditionemb__" in pseudobulk:
                    this_bulk_dict[f"{output_prefix}:condition_emb{suffix}"] = (
                        pseudobulk["__conditionemb__"]
                    )

                # 2. add pseudobulk embedding
                row_embedding = pseudobulk["__embedding__"]
                this_bulk_dict[f"{output_prefix}:embedding_data{suffix}"] = (
                    row_embedding
                )

                # 3. add trange if available
                _bulk_values = all_pseudobulk_data[
                    None, 2 * pair_idx + idx_in_pair
                ]  # (1, region_length)
                cov_logfc = pseudobulk["__covlogfc__"]
                if self.normalize_cov:
                    prefix_cov_logfc = cov_logfc[dataset_prefix]
                    _bulk_values /= 2**prefix_cov_logfc
                this_bulk_dict[f"{output_prefix}:bulk_data{suffix}"] = _bulk_values

            # TODO: temporary solution to mimic matcher
            this_bulk_dict[f"{output_prefix}:bulk_data_delta"] = (
                this_bulk_dict[f"{output_prefix}:bulk_data_1"]
                - this_bulk_dict[f"{output_prefix}:bulk_data_0"]
            )

            new_batches.append(this_bulk_dict)
        return new_batches

    def _region_to_global_coords(self, key, region):
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        chrom_offsets = self.chrom_offsets[key]
        chrom_start = chrom_offsets.loc[chrom, "global_start"]
        global_coords = np.array([chrom_start + start, chrom_start + end])
        return global_coords

    def _add_region_and_dataset_key_int(self, list_of_batches, key, region):
        key_int = self.dataset_orders.index(key)
        region_coords = self._region_to_global_coords(key, region)
        for batch in list_of_batches:
            batch[self.dataset_key] = key_int
            batch[self.region_key] = region_coords
        return list_of_batches

    def __call__(self, batch: dict) -> list[dict[str, Any]]:
        """
        Generate pseudobulk data for a batch of region.
        """
        region = batch.pop(self.region_key)
        dataset_key = batch.pop(self.dataset_key)
        dna = batch.pop(DNA_NAME)

        list_of_batches = self._sample_pseudobulks_and_get_data(dataset_key, region)
        list_of_batches = self._add_region_and_dataset_key_int(
            list_of_batches, key=dataset_key, region=region
        )
        for batch in list_of_batches:
            batch[DNA_NAME] = dna.copy()
        return list_of_batches


class BorzoiMultiDataset(GenericDataset):
    """Single cell pseudobulk dataset for Borzoi model."""

    default_config = {
        "dataset_records": "REQUIRED",
        "paired_mode": "ensemble",
        "batch_size": 4,
        "dna_window": 524288,
        "max_jitter": 3,
        "reverse_complement": True,
        "n_pseudobulks": 10,
        "output_prefix": "pseudobulk",
        "train_region_step_sample": True,
    }

    def __init__(
        self,
        dataset_records: str | dict,
        paired_mode: str = "ensemble",
        batch_size: int = 4,
        dna_window: int = 524288,
        max_jitter: int = 3,
        reverse_complement: bool = True,
        n_pseudobulks: int = 10,
        output_prefix: str = "pseudobulk",
        train_region_step_sample=True,
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
        self.reverse_complement = reverse_complement
        self.n_pseudobulks = n_pseudobulks
        self.train_region_step_sample = train_region_step_sample

        self._block_size = 20
        self._max_blocks = 200
        self.paired_data = True

        self.output_prefix = output_prefix
        self.dna_column = DNA_NAME
        self.signal_columns = [
            f"{self.output_prefix}:bulk_data_0",
            f"{self.output_prefix}:bulk_data_1",
            f"{self.output_prefix}:bulk_data_delta",
        ]
        return

    def get_train_valid_test(self, fold, seed=0, **kwargs):
        """
        Get the train, valid, and test folds and regions for the given fold.
        Then sample the regions for each dataset key based on the sample_region_fracs.
        """
        fold_split = MultiBorzoiRegions.fold_splits[fold]
        train_folds = fold_split["train"]
        valid_folds = fold_split["valid"]
        test_folds = fold_split["test"]
        train_regions, valid_regions, test_regions = (
            self.dm.get_train_valid_test_regions(split_id=fold, seed=seed)
        )
        return (
            train_folds,
            valid_folds,
            test_folds,
            train_regions,
            valid_regions,
            test_regions,
        )

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
        return dataset

    def _get_pseudobulk_and_data(self, dataset, concurrency):
        """
        Get the pseudobulk data for the dataset.
        """
        fn = GenerateMultiGenomeParquetAndPseudobulk
        fn_constructor_kwargs = {
            "parquet_paths": self.dm._data_paths,
            "pseudobulker": self.dm.pseudobulker,
            "dataset_orders": self.dm.dataset_orders,
            "n_pseudobulks": self.n_pseudobulks,
            "region_key": "region",
            "dataset_key": "__dataset_keys__",
            "normalize_cov": True,
            "output_prefix": "pseudobulk",
            "blacklist_global_coords": self.dm.get_blacklist_global_coords(),
            "chrom_offsets": self.dm.get_chrom_offsets(),
            "dataset_sample_weight": self.dm.dataset_sample_weights,
        }

        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=(1, concurrency),
        )
        return dataset

    def _get_reverse_complement_region(self, dataset, batch_size=16) -> None:
        """
        Reverse complement the DNA sequences by 50% probability.

        Returns
        -------
        None
        """
        _rc = ReverseComplement(
            dna_key=self.dna_column,
            signal_key=self.signal_columns,
        )
        dataset = dataset.map_batches(_rc, batch_size=batch_size)
        return dataset

    def _prepare_region_dataset(self, region_bed: pd.DataFrame):
        region_bed = region_bed.rename(
            columns={"key": "__dataset_keys__", "Name": "region"}
        )
        region_bed = region_bed[["__dataset_keys__", "region"]].reset_index(drop=True)

        if self.is_train():
            # resample regions for training
            region_bed = region_bed.sample(frac=1, replace=False)
        else:
            # shuffle regions for validation and test
            region_bed = region_bed.sample(frac=1, replace=False, random_state=42)

        n_blocks = min(len(region_bed) // self._block_size + 1, self._max_blocks)
        dataset = ray.data.from_pandas(region_bed).repartition(n_blocks).materialize()
        return dataset

    def get_processed_dataset(self, region_bed: pd.DataFrame, concurrency: int):
        """Create a processed dataset from the region BED file."""
        batch_size = self.batch_size * 4
        dataset = self._prepare_region_dataset(region_bed)

        # add dna
        dataset = self._get_dna_one_hot(
            dataset=dataset,
            concurrency=1,
        )

        # add pseudobulk and data
        dataset = self._get_pseudobulk_and_data(
            dataset=dataset,
            concurrency=concurrency,
        )

        if self.reverse_complement and self.is_train():
            dataset = self._get_reverse_complement_region(
                dataset, batch_size=batch_size
            )

        # # remove region column OR turn it into global coordinates (str to numbers)
        # dataset = self._process_region_columns(
        #     dataset=dataset, keep_regions=True, batch_size=batch_size
        # )
        return dataset

    def _get_dataloader_with_wrapper(
        self,
        dataset_kwargs: dict,
        data_iter_kwargs: dict,
        batch_size: int,
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
            _kwargs.pop("folds", None)
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
        shuffle_rows: int = 500,
        concurrency: int = 8,
        **dataloader_kwargs: Any,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader.

        Parameters
        ----------
        region_bed : str
            Path to the BED file containing the regions.
        as_torch : bool, optional
            Whether to return a PyTorch dataloader, by default True.
        n_batches : int, optional
            Number of batches to return, by default None.
        shuffle_rows : int, optional
            The size of the local shuffle buffer, by default 500.
        concurrency : int, optional
            The number of workers to use for processing the dataset, by default 8.
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
        dataloader_kwargs.setdefault("local_shuffle_buffer_size", shuffle_rows)
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
# Dedup train regions as BorzoiDataset do
