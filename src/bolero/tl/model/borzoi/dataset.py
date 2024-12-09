import pathlib
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import ray
import torch

from bolero.tl.dataset.ray_dataset import (
    RayGenomeChunkDataset,
)
from bolero.tl.dataset.sc_transforms import GeneratePseudobulk
from bolero.tl.dataset.transforms import (
    FetchRegionOneHot,
    ReverseComplement,
)

from .utils import BorzoiRegions, add_position_weights_to_batch

DNA_NAME = "dna_one_hot"


class MaskBlacklistAndClamp:
    def __init__(
        self,
        blacklist_global_coords,
        chrom_offsets,
        region_key,
        data_keys,
        as_nan=False,
        resolution=32,
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
            overlap_bl_regions[:, 0] -= 3 * self.resolution
            overlap_bl_regions[:, 1] += 3 * self.resolution
            overlap_bl_regions = np.clip(overlap_bl_regions, 0, region_length)

            # convert base coords to resolution coords
            overlap_bl_regions //= self.resolution

            for bl_start, bl_end in overlap_bl_regions:
                if bl_start >= region_length:
                    continue
                nan_slice = slice(bl_start, bl_end)
                for data_key in self.data_keys:
                    batch[data_key][region_id, :, nan_slice] = self._mask_value
        return batch


class GenerateBorzoiPseudobulk(GeneratePseudobulk):
    """
    Transform meta region data into bulk region data.
    """

    def __init__(
        self,
        n_pseudobulks=10,
        return_rows=False,
        inplace=False,
        bypass_keys=None,
        paired_data=False,
        normalize_cov=None,
        reduce_resolution=None,
        **name_to_pseudobulker,
    ):
        super().__init__(
            n_pseudobulks=n_pseudobulks,
            return_rows=return_rows,
            inplace=inplace,
            bypass_keys=bypass_keys,
            **name_to_pseudobulker,
        )
        self.paired_data = paired_data
        if self.paired_data:
            # Similar to the Performer paper
            # Data from different pseudobulks will be loaded in pairs
            # Saved into separate keys with suffixes
            # During training, each key will be learned separately,
            # and their difference will be used as the diff loss
            self.suffix = ["_A", "_B"]
        else:
            # Normal data loading
            self.suffix = [""]

        self.normalize_cov = normalize_cov
        self.reduce_resolution = reduce_resolution
        return

    def _reduce_resolution(self, data):
        resolution = self.reduce_resolution
        # from (1, seq_len) to (1, seq_len // resolution) by summing
        data = data.reshape(1, -1, resolution).sum(axis=-1)
        return data

    def __call__(self, data_dict: dict[str, bytes]) -> list[dict[str, np.ndarray]]:
        """Generate pseudobulks for each output prefix."""
        list_of_dicts = []

        assert len(self.name_to_pseudobulker) == 1, "Only one pseudobulker is allowed"
        output_prefix, pseudobulker = list(self.name_to_pseudobulker.items())[0]

        suffix_list = self.suffix
        cycling = len(suffix_list)

        # merge rows (cell or sample) to bulk and also get embedding data
        this_bulk_dict = {}
        for idx, (
            prefix_to_rows,
            row_embedding,
            cov_logfc,
            pseudobulk_id,
        ) in enumerate(pseudobulker.take(self.n_pseudobulks * cycling)):
            suffix = suffix_list[idx % cycling]
            this_bulk_dict[f"{output_prefix}:embedding_data{suffix}"] = row_embedding
            this_bulk_dict[f"{output_prefix}:pseudobulk_ids{suffix}"] = pseudobulk_id

            combined_bulk_data = []
            for prefix_idx, prefix in enumerate(pseudobulker.prefix_order):
                prefix_rows = prefix_to_rows[prefix]
                # row_by_base is a csr_matrix of shape (n_rows, region_length)
                try:
                    row_by_base = data_dict[prefix]
                except KeyError as e:
                    raise KeyError(
                        f"Key {prefix} not found in data_dict, {data_dict.keys()}"
                    ) from e

                _bulk_values = (
                    row_by_base[prefix_rows].sum(axis=0).A1
                )  # (1, region_length)

                if self.normalize_cov:
                    prefix_cov_logfc = cov_logfc[prefix_idx]
                    _bulk_values /= 2**prefix_cov_logfc

                if self.reduce_resolution:
                    _bulk_values = self._reduce_resolution(_bulk_values)

                combined_bulk_data.append(_bulk_values)
            this_bulk_dict[f"{output_prefix}:bulk_data{suffix}"] = np.vstack(
                combined_bulk_data
            )

            # copy shared information to the bulk dict
            for key in self.bypass_keys:
                if key in data_dict:
                    this_bulk_dict[key] = deepcopy(data_dict[key])

            if idx % cycling == cycling - 1:
                list_of_dicts.append(this_bulk_dict)
                this_bulk_dict = {}
        return list_of_dicts


class GenerateBorzoiPseudobulkProfiler:
    """
    Transform meta region data into bulk region data.
    """

    def __init__(
        self,
        n_pseudobulks=1000,
        q=0.995,
        **name_to_pseudobulker,
    ):
        self.n_pseudobulks = n_pseudobulks
        self.normalize_cov = True
        self.q = q
        self.name_to_pseudobulker = name_to_pseudobulker

        assert len(self.name_to_pseudobulker) == 1, "Only one pseudobulker is allowed"
        for name, pseudobulker in self.name_to_pseudobulker.items():
            self.final_prefix = name
            self.pseudobulker = pseudobulker
        return

    def _collect_pseudobulks(self, data_dict):
        pseudobulker = self.pseudobulker

        # merge rows (cell or sample) to bulk and also get embedding data
        prefix_data_dict = defaultdict(list)
        for prefix_to_rows, row_embedding, _ in pseudobulker.take(self.n_pseudobulks):
            cov_logfc = row_embedding[pseudobulker.vq_emb_dims :]
            for prefix_idx, prefix in enumerate(pseudobulker.prefix_order):
                prefix_rows = prefix_to_rows[prefix]
                # row_by_base is a csr_matrix of shape (n_rows, region_length)
                row_by_base = data_dict[prefix]
                _bulk_values = (
                    row_by_base[prefix_rows].sum(axis=0).A1
                )  # (1, region_length)

                if self.normalize_cov:
                    prefix_cov_logfc = cov_logfc[prefix_idx]
                    _bulk_values /= 2**prefix_cov_logfc
                prefix_data_dict[prefix].append(_bulk_values)
        prefix_data_dict = {k: np.vstack(v) for k, v in prefix_data_dict.items()}
        return prefix_data_dict

    def _region_profile_summary(self, prefix_data_dict):
        upper_bound_col = []
        mean_p_col = []
        for prefix in self.pseudobulker.prefix_order:
            prefix_data = prefix_data_dict[prefix]
            upper_bound = np.quantile(prefix_data, self.q, axis=0, keepdims=True) + 1e-8
            data_p = np.clip(prefix_data / upper_bound, a_min=0, a_max=1)
            data_p_mean = data_p.mean(axis=0, keepdims=True)

            upper_bound_col.append(upper_bound)
            mean_p_col.append(data_p_mean)

        summary_dict = {
            f"{self.final_prefix}:upper_bound": np.vstack(upper_bound_col),
            f"{self.final_prefix}:mean_p": np.vstack(mean_p_col),
        }
        return summary_dict

    def __call__(self, data_dict: dict[str, bytes]) -> dict[str, np.ndarray]:
        """Generate pseudobulks for each output prefix."""
        prefix_data_dict = self._collect_pseudobulks(data_dict)

        summary_dict = self._region_profile_summary(prefix_data_dict)

        data_dict.update(summary_dict)
        return data_dict


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
        "min_cov": 0,
        "paired_data": False,
        "normalize_cov": None,
        "use_gene_regions": False,
        "deg_list": None,
        "use_pseudobulk_profile": False,
        "reduce_resolution": False,
        # only applicable when use gene regions
        "gene_weight": 1.0,
        "background_weight": 1e-4,
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
        min_cov: int = 0,
        paired_data=False,
        normalize_cov=None,
        use_gene_regions=False,
        deg_list=None,
        use_pseudobulk_profile=False,
        gene_weight=1.0,
        background_weight=1e-4,
        reduce_resolution=False,
    ):
        super().__init__(
            dataset_path=dataset_path,
            shuffle_files=shuffle_files,
            read_parquet_kwargs=read_parquet_kwargs,
            max_regions_per_genome_chunk=1,
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
        self.paired_data = paired_data
        self.normalize_cov = normalize_cov
        self.use_gene_regions = use_gene_regions
        self.deg_list = deg_list
        self.use_pseudobulk_profile = use_pseudobulk_profile
        self.gene_weight = gene_weight
        self.background_weight = background_weight
        self.reduce_resolution = reduce_resolution

        self.name_to_pseudobulker = OrderedDict()

        self.borzoi_regions = BorzoiRegions()
        return

    def get_train_valid_test(
        self,
        fold,
        downsample_train_region=None,
        downsample_valid_region=None,
        downsample_test_region=None,
        region_length=None,
        seed=0,
    ):
        """Get the train, valid, and test folds and regions for the given fold."""
        fold_split = self.borzoi_regions.fold_splits[fold]
        train_folds = fold_split["train"]
        valid_folds = fold_split["valid"]
        test_folds = fold_split["test"]

        use_gene_regions = getattr(self, "use_gene_regions", False)
        deg_list = getattr(self, "deg_list", None)
        if region_length is None:
            region_length = self.dna_window
        train_regions, valid_regions, test_regions = (
            self.borzoi_regions.get_train_valid_test_regions(
                self.genome.name,
                split_id=fold,
                region_length=region_length,
                use_gene_regions=use_gene_regions,
                deg_list=deg_list,
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
            "random_shift": self.max_jitter if self.is_train() else 0,
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
        dataset = dataset.map_batches(_rc, batch_size=batch_size)
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
        filter_prefix="pseudobulk",
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

        fn_kwargs = {
            "filter_key": f"{filter_prefix}:bulk_data",
            "min_cov": self.min_cov,
        }
        dataset = dataset.map_batches(
            fn=_fn,
            fn_kwargs=fn_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    @property
    def data_keys(self):
        """Return the data keys in batch dict."""
        if self.paired_data:
            suffixes = ["_A", "_B"]
        else:
            suffixes = [""]
        data_keys = [
            f"{name}:bulk_data{suffix}"
            for name in self.name_to_pseudobulker.keys()
            for suffix in suffixes
        ]
        return data_keys

    def _mask_blacklist_and_clamp(self, dataset, concurrency=(1, 2), batch_size=16):
        """Mask the blacklist regions in the dataset."""
        fn = MaskBlacklistAndClamp
        bl_coords = self.genome.get_global_coords(self.genome.blacklist_bed)
        fn_constructor_kwargs = {
            "blacklist_global_coords": bl_coords,
            "chrom_offsets": self.genome.chrom_offsets,
            "region_key": "region",
            "data_keys": self.data_keys,
        }
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _generate_pseudobulk_profile(
        self,
        dataset,
        n_pseudobulks=1000,
        q=0.995,
        concurrency=1,
    ):
        fn = GenerateBorzoiPseudobulkProfiler
        fn_constructor_kwargs = {
            "n_pseudobulks": n_pseudobulks,
            "q": q,
        }
        name_to_pseudobulker = self.name_to_pseudobulker
        fn_constructor_kwargs.update(name_to_pseudobulker)

        dataset = dataset.map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )

        new_suffixes = ["upper_bound", "mean_p"]
        name = list(name_to_pseudobulker.keys())[0]
        self.signal_columns.extend([f"{name}:{suffix}" for suffix in new_suffixes])
        return dataset

    def _generate_pseudobulk(
        self,
        dataset,
        concurrency=1,
        **kwargs,
    ):
        fn = GenerateBorzoiPseudobulk
        fn_constructor_kwargs = {
            "n_pseudobulks": self.n_pseudobulks,
            "paired_data": self.paired_data,
            "normalize_cov": self.normalize_cov,
            "reduce_resolution": self.pos_resolution
            if self.reduce_resolution
            else None,
            **kwargs,
        }
        name_to_pseudobulker = self.name_to_pseudobulker
        fn_constructor_kwargs.update(name_to_pseudobulker)

        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )

        # update signal_columns
        region_action_keys = [
            name for name in self.signal_columns if name not in name_to_pseudobulker
        ]
        new_keys = self.data_keys
        region_action_keys.extend(new_keys)
        self.signal_columns = list(set(region_action_keys))
        return dataset

    def _get_processed_dataset(
        self,
        folds,
        region_bed,
        region_action_keys=None,
        concurrency=32,
        add_original_name=True,
        compressed_bytes_to_tensor_concurrency=None,
        generate_pseudobulk_concurrency=None,
        generate_regions_concurrency=None,
        **pseudobulk_kwargs,
    ) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        if compressed_bytes_to_tensor_concurrency is None:
            compressed_bytes_to_tensor_concurrency = (1, concurrency // 4)
        if generate_pseudobulk_concurrency is None:
            generate_pseudobulk_concurrency = (1, concurrency)
        if generate_regions_concurrency is None:
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
        self.signal_columns = region_action_keys

        # generate pseudobulk
        name_to_pseudobulker = self.name_to_pseudobulker
        if len(name_to_pseudobulker) > 0:
            if getattr(self, "use_pseudobulk_profile", False):
                dataset = self._generate_pseudobulk_profile(
                    dataset=dataset,
                    concurrency=generate_pseudobulk_concurrency,
                )
                pseudobulk_kwargs["bypass_keys"] = self.signal_columns

            dataset = self._generate_pseudobulk(
                dataset=dataset,
                concurrency=generate_pseudobulk_concurrency,
                **pseudobulk_kwargs,
            )

        if region_bed is not None:
            dataset = self._generate_regions(
                dataset=dataset,
                bed=region_bed,
                action_keys=self.signal_columns,
                max_regions=self.max_regions_per_genome_chunk,
                concurrency=generate_regions_concurrency,
                pos_resolution=self.pos_resolution,
                add_original_name=add_original_name,
            )
        return dataset

    def get_processed_dataset(
        self,
        folds: list[int],
        region_bed: str,
        concurrency: int = 32,
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
            inplace=False,
            concurrency=concurrency,
        )

        # mask blacklist as nan
        work_ds = self._mask_blacklist_and_clamp(
            dataset=work_ds,
            concurrency=(1, 2),
            batch_size=batch_size,
        )

        # filter min cov
        if (self.min_cov > 0) and (not self.paired_data):
            work_ds = self._filter_min_cov(
                dataset=work_ds,
                concurrency=2,
                batch_size=batch_size,
                filter_prefix="pseudobulk",
            )

        # add dna one hot
        work_ds = self._get_dna_one_hot(
            dataset=work_ds,
            concurrency=1,
            batch_size=batch_size,
        )

        if self.reverse_complement and self.is_train():
            work_ds = self._get_reverse_complement_region(
                work_ds, batch_size=batch_size
            )

        # remove region column OR turn it into global coordinates (str to numbers)
        work_ds = self._process_region_columns(
            dataset=work_ds, keep_regions=True, batch_size=batch_size
        )
        return work_ds

    def get_dataloader(
        self,
        folds,
        region_bed,
        as_torch=True,
        n_batches=None,
        shuffle_rows=500,
        concurrency=32,
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

    def maybe_preprocess_batch(self, batch):
        """Maybe add region weights to weight on gene regions."""
        if self.use_gene_regions:
            batch = add_position_weights_to_batch(
                batch=batch,
                genome=self.genome,
                region_id_map=self.borzoi_regions.cur_idmap,
                effective_regions=self.borzoi_regions.cur_effective_regions,
                resolution=32,
                gene_value=self.gene_weight,
                other_value=self.background_weight,
            )

        if self.use_pseudobulk_profile:
            batch = self._normalize_by_profile(batch)
        return batch

    def _normalize_by_profile(self, batch, weight_by_mean_p=False):
        prefix = list(self.name_to_pseudobulker.keys())[0]

        data = batch[f"{prefix}:bulk_data"]
        # top quantile value at each position
        upper = batch[f"{prefix}:upper_bound"]
        # mean proportion of each position, indicating cellular diversity
        mean_p = batch[f"{prefix}:mean_p"]

        # calculate the proportion of each position based on the upper bound
        data_p = torch.clamp(data / upper, max=1)

        if weight_by_mean_p:
            # weight the data by the mean proportion
            data_p = data_p * mean_p

        batch[f"{prefix}:bulk_data"] = data_p
        return batch
