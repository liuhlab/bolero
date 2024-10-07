import pathlib
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import ray

from bolero.tl.dataset.ray_dataset import (
    RayGenomeChunkDataset,
)
from bolero.tl.dataset.sc_transforms import FilterRegions
from bolero.tl.dataset.transforms import (
    FetchRegionOneHot,
    ReverseComplement,
)

from .utils import BorzoiRegions

DNA_NAME = "dna_one_hot"


class MaskBlacklist:
    def __init__(self, blacklist_global_coords, chrom_offsets, region_key, data_key):
        self.blacklist_global_coords = blacklist_global_coords
        self.chrom_offsets = chrom_offsets
        self.region_key = region_key
        self.data_key = data_key

    def _get_region_chrom(self, region):
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        chorm_start = self.chrom_offsets.loc[chrom, "global_start"]
        global_coords = (chorm_start + start, chorm_start + end)
        return global_coords

    def __call__(self, batch):
        """Mask the blacklist regions in the dataset."""
        bl_coords = self.blacklist_global_coords

        for region_id, region in enumerate(batch[self.region_key]):
            r_gstart, r_gend = self._get_region_chrom(region)
            overlap_bl_regions = bl_coords[
                (bl_coords[:, 0] < r_gend) & (bl_coords[:, 1] > r_gstart)
            ].copy()
            overlap_bl_regions -= r_gstart
            region_length = r_gend - r_gstart

            for bl_start, bl_end in overlap_bl_regions:
                nan_slice = slice(max(0, bl_start), min(region_length, bl_end))
                batch[self.data_key][region_id, :, nan_slice] = np.NaN
        return batch


class BorzoiDataset(RayGenomeChunkDataset):
    """Singel cell pseudobulk dataset for Borzoi model."""

    default_config = {
        "dataset_path": "REQUIRED",
        "batch_size": 2,
        "dna_window": 524288,
        "pos_resolution": 32,
        "reverse_complement": True,
        "max_jitter": 3,
        "n_pseudobulks": 100,
        "shuffle_files": True,
        "read_parquet_kwargs": None,
        "min_cov": 100,
    }

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 2,
        dna_window: int = 524288,
        pos_resolution: int = 32,
        reverse_complement: bool = True,
        max_jitter: int = 3,
        n_pseudobulks: int = 100,
        cov_filter_name: str = None,
        shuffle_files=False,
        read_parquet_kwargs=None,
        min_cov: int = 100,
    ):
        super().__init__(
            dataset_path=dataset_path,
            shuffle_files=shuffle_files,
            read_parquet_kwargs=read_parquet_kwargs,
        )
        self.batch_size = batch_size

        # region properties
        self.dna_window = dna_window
        self.signal_window = dna_window
        self.pos_resolution = pos_resolution
        self.max_jitter = max_jitter
        self.reverse_complement = reverse_complement
        self.n_pseudobulks = n_pseudobulks
        self.cov_filter_name = cov_filter_name
        self.min_cov = min_cov

        self.name_to_pseudobulker = OrderedDict()

        self.borzoi_regions = BorzoiRegions()
        return

    def get_train_valid_test(
        self,
        fold,
        downsample_train_region=None,
        downsample_valid_region=None,
        downsample_test_region=None,
        seed=0,
    ):
        """Get the train, valid, and test folds and regions for the given fold."""
        fold_split = self.borzoi_regions.fold_splits[fold]
        train_folds = fold_split["train"]
        valid_folds = fold_split["valid"]
        test_folds = fold_split["test"]

        train_regions, valid_regions, test_regions = (
            self.borzoi_regions.get_train_valid_test_regions(
                self.genome.name, split_id=fold, region_length=self.dna_window
            )
        )

        if downsample_train_region and (
            downsample_train_region < train_regions.shape[0]
        ):
            train_regions = train_regions.sample(
                downsample_train_region, random_state=seed
            )
            print(f"Downsampled train regions to {downsample_train_region}")
        if downsample_valid_region and (
            downsample_valid_region < valid_regions.shape[0]
        ):
            valid_regions = valid_regions.sample(
                downsample_valid_region, random_state=seed
            )
            print(f"Downsampled valid regions to {downsample_valid_region}")
        if downsample_test_region and (downsample_test_region < test_regions.shape[0]):
            test_regions = test_regions.sample(
                downsample_test_region, random_state=seed
            )
            print(f"Downsampled test regions to {downsample_test_region}")

        return (
            train_folds,
            valid_folds,
            test_folds,
            train_regions,
            valid_regions,
            test_regions,
        )

    def __repr__(self) -> str:
        _str = (
            f"{self.__name__}\n"
            f"Dataset directory: {self.dataset_path}\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
        )
        return _str

    def _get_dna_one_hot(self, dataset, concurrency, batch_size=16):
        fn = FetchRegionOneHot
        fn_constructor_kwargs = {
            "random_shift": self.max_jitter if self._dataset_mode == "train" else 0,
            "dtype": "bool",
        }
        fn_kwargs = {"remote_genome_one_hot": self.genome.remote_genome_one_hot}

        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            fn_kwargs=fn_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        self.dna_column = DNA_NAME
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
        dataset = dataset.map_batches(_rc, batch_size=16)
        return dataset

    def add_pseudobulker(self, name: str, cls, pseudobulker_kwargs: dict):
        """
        Add a pseudobulker to the dataset.

        Parameters
        ----------
        name : str
            The name of the pseudobulker, will be used as pseudobulk prefix in final dict.
        cls : Pseudobulker class
            The pseudobulker class that can be used to generate pseudobulks.
        pseudobulker_kwargs : dict
            The keyword arguments to pass to the pseudobulker class constructor.
        """
        if "barcode_order" not in pseudobulker_kwargs:
            pseudobulker_kwargs["barcode_order"] = self.barcode_order
        generator = cls.create_from_config(**pseudobulker_kwargs)
        self.name_to_pseudobulker[name] = generator
        return

    def _filter_bed_regions(
        self,
        dataset,
        cov_filter_key,
        min_cov,
        max_cov,
        low_cov_ratio,
        batch_size,
        concurrency,
    ):
        fn = FilterRegions
        fn_constructor_kwargs = {
            "cov_filter_key": cov_filter_key,
            "min_cov": min_cov,
            "max_cov": max_cov,
            "low_cov_ratio": low_cov_ratio,
        }
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _get_folds_dir(self, folds):
        if folds is None:
            fold_dirs = [str(p) for p in pathlib.Path(self.dataset_path).glob("fold*")]
        else:
            if isinstance(folds, str):
                folds = [folds]
            fold_dirs = [f"{self.dataset_path}/fold{fold}" for fold in folds]

            # make sure all fold_dir exists
            fold_dirs = [
                fold_dir for fold_dir in fold_dirs if pathlib.Path(fold_dir).exists()
            ]
            assert (
                len(fold_dirs) > 0
            ), f"None of the fold {folds} exists in {self.dataset_path}"
        return fold_dirs

    def _read_parquet(self, folds):
        _dataset = ray.data.read_parquet(
            self._get_folds_dir(folds),
            file_extensions=["parquet"],
            **self.read_parquet_kwargs,
        )
        return _dataset

    def _filter_min_cov(
        self,
        dataset,
        concurrency=2,
        batch_size=16,
    ):
        def _fn(batch: dict, filter_key, min_cov):
            """Filter regions based on coverage."""
            data = batch[filter_key]

            region_sum = data.sum(axis=(1, 2))

            use_rows = region_sum > min_cov
            if use_rows.sum() == 0:
                # keep at least one region
                use_rows[0] = True

            # apply filter to all keys
            batch = {
                k: v[use_rows, ...].copy()  # if v.ndim > 1 else v[use_rows]
                for k, v in batch.items()
            }
            return batch

        # TODO: prefix pseudobulk may change
        fn_kwargs = {
            "filter_key": "pseudobulk:bulk_data",
            "min_cov": self.min_cov,
        }
        dataset = dataset.map_batches(
            fn=_fn,
            fn_kwargs=fn_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _mask_blacklist(self, dataset, concurrency=(1, 2), batch_size=16):
        """Mask the blacklist regions in the dataset."""
        fn = MaskBlacklist
        bl_coords = self.genome.get_global_coords(self.genome.blacklist_bed)
        fn_constructor_kwargs = {
            "blacklist_global_coords": bl_coords,
            "chrom_offsets": self.genome.chrom_offsets,
            "region_key": "region",
            "data_key": "pseudobulk:bulk_data",
        }
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _get_processed_dataset(
        self,
        folds,
        region_bed,
        name_to_pseudobulker,
        region_action_keys=None,
        concurrency=32,
        **pseudobulk_kwargs,
    ) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        compressed_bytes_to_tensor_concurrency = (1, concurrency // 4)
        generate_pseudobulk_concurrency = (1, concurrency)
        generate_regions_concurrency = (1, concurrency // 2)

        dataset = self._read_parquet(folds=folds)

        # filter meta region length equal to self.window_size
        dataset = self._filter_meta_region_length(dataset=dataset)

        # from compressed bytes to tensor (cell/sample by meta-region matrix) and other information
        dataset = self._compressed_bytes_to_tensor(
            dataset=dataset,
            concurrency=compressed_bytes_to_tensor_concurrency,
        )

        if region_action_keys is None:
            region_action_keys = []
        elif isinstance(region_action_keys, str):
            region_action_keys = [region_action_keys]
        else:
            pass

        # generate pseudobulk
        if len(name_to_pseudobulker) > 0:
            dataset = self._generate_pseudobulk(
                dataset=dataset,
                name_to_pseudobulker=name_to_pseudobulker,
                concurrency=generate_pseudobulk_concurrency,
                **pseudobulk_kwargs,
            )

            # update region_action_keys
            region_action_keys = [
                name for name in region_action_keys if name not in name_to_pseudobulker
            ]
            new_keys = [f"{name}:bulk_data" for name in name_to_pseudobulker.keys()]
            region_action_keys.extend(new_keys)
            region_action_keys = list(set(region_action_keys))
            self.signal_columns = region_action_keys

        if region_bed is not None:
            dataset = self._generate_regions(
                dataset=dataset,
                bed=region_bed,
                action_keys=region_action_keys,
                max_regions=1,
                concurrency=generate_regions_concurrency,
                pos_resolution=self.pos_resolution,
            )
        return dataset

    def get_processed_dataset(
        self,
        folds: list[int],
        region_bed: str,
        return_cells: bool = False,
        return_regions: bool = True,
        concurrency: int = 16,
        batch_size: int = 16,
    ) -> None:
        """
        Process the dataset and return the processed dataset.

        Parameters
        ----------
        - folds (list): List of folds to include in the dataset.
        - region_bed_path (str): Path to the BED file containing the regions.
        - return_cells (bool): Whether to return the cells in the dataset. Default is False.
        - return_regions (bool): Whether to return the regions in the dataset. Default is False.

        Returns
        -------
        - work_ds (Dataset): The processed dataset.

        """
        work_ds = self._get_processed_dataset(
            folds=folds,
            region_bed=region_bed,
            name_to_pseudobulker=self.name_to_pseudobulker,
            n_pseudobulks=self.n_pseudobulks,
            return_rows=return_cells,
            inplace=False,
            concurrency=concurrency,
        )

        # mask blacklist as nan
        # work_ds = self._mask_blacklist(
        #     dataset=work_ds,
        #     concurrency=(1, 2),
        #     batch_size=batch_size,
        # )

        # filter min cov
        work_ds = self._filter_min_cov(
            dataset=work_ds,
            concurrency=2,
            batch_size=batch_size,
        )

        # add dna one hot
        work_ds = self._get_dna_one_hot(
            dataset=work_ds,
            concurrency=1,
            batch_size=batch_size,
        )

        if self.reverse_complement and self._dataset_mode == "train":
            work_ds = self._get_reverse_complement_region(
                work_ds, batch_size=batch_size
            )

        # remove region column OR turn it into global coordinates (str to numbers)
        work_ds = self._process_region_columns(
            dataset=work_ds, keep_regions=return_regions, batch_size=batch_size
        )
        return work_ds

    def get_dataloader(
        self,
        folds,
        region_bed,
        as_torch=True,
        return_regions=True,
        return_cells=False,
        n_batches=None,
        shuffle_rows=500,
        concurrency=20,
        **dataloader_kwargs,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader.

        Parameters
        ----------
        local_shuffle_buffer_size : int, optional
            The size of the local shuffle buffer, by default 10000.
        randomize_block_order : bool, optional
            Whether to randomize the block order, by default False.
        as_torch : bool, optional
            Whether to return a PyTorch dataloader, by default True.
        device : str, optional
            The device to use, by default None.
        return_cells : bool, optional
            Whether to return the cell ids, by default False.
        **dataloader_kwargs
            Additional keyword arguments pass to ray.data.Dataset.iter_batches.

        Returns
        -------
        DataLoader
            The dataloader.
        """
        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "folds": folds,  # for borzoi we don't split train/valid/test via chromosomes, so all chromosomes are included
            "region_bed": region_bed,
            "return_cells": return_cells,
            "return_regions": return_regions,
            "concurrency": concurrency,
        }
        data_iter_kwargs = dataloader_kwargs

        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            as_torch=as_torch,
            shuffle_rows=shuffle_rows,
            n_batches=n_batches,
            batch_size=self.batch_size,
        )
        return loader
