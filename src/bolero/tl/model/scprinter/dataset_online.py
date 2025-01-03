import json
from collections import defaultdict
from typing import Any, Iterable

import pyranges as pr

from bolero.tl.dataset.ray_dataset import RayRegionDataset
from bolero.tl.dataset.sc_transforms import FilterRegions
from bolero.tl.dataset.transforms import (
    FetchRegionOneHot,
)
from bolero.tl.model.borzoi.utils import BorzoiRegions
from bolero.tl.model.borzoi_human.dataset import BorzoiDatasetOnline

from .dataset import BatchFootPrint

DNA_NAME = "dna_one_hot"


class scPrinterOnlineDataset(BorzoiDatasetOnline, RayRegionDataset):
    """Singel cell dataset for scPrinter model."""

    default_config = {
        "dataset_path": "REQUIRED",
        "genome": "REQUIRED",
        "region_bed": "REQUIRED",
        "cell_types": None,
        "embeddings_path": "REQUIRED",
        "tn5_bias_path": "REQUIRED",
        "data_key_to_file_type": None,
        "batch_size": 64,
        "dna_window": 1840,
        "signal_window": 1000,
        "max_jitter": 128,
        "clip_min": -10,
        "clip_max": 10,
        "n_pseudobulks": 100,
        "cov_filter_name": "REQUIRED",
        "min_cov": 30,
        "max_cov": 100000,
        "low_cov_ratio": 0.1,
        "reverse_complement": True,
        "max_regions_per_genome_chunk": 100,
        "data_key_to_mc_context": None,
        "merge_cell_type": False,
    }

    def __init__(
        self,
        dataset_path: str,
        region_bed: str,
        genome,
        embeddings_path: str,
        tn5_bias_path: str,
        cell_types: list[str] = None,
        data_key_to_file_type: dict[str, str] = None,
        batch_size: int = 64,
        dna_window: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        clip_min: float = -10,
        clip_max: float = 10,
        n_pseudobulks: int = 10,
        min_cov: int = 30,
        max_cov: int = 100000,
        low_cov_ratio: float = 0.1,
        cov_filter_name: str = None,
        reverse_complement: bool = True,
        max_regions_per_genome_chunk: int = 100,
        data_key_to_mc_context=None,
        merge_cell_type: bool = False,
    ):
        RayRegionDataset.__init__(
            self,
            bed=region_bed,
            genome=genome,
            standard_length=dna_window,
            batch_size=batch_size,
            dna=False,
        )

        with open(dataset_path) as f:
            dataset_path_dict = json.load(f)

        if data_key_to_file_type is None:
            data_key_to_file_type = {}
            cell_type_file_keys = []
            for ct_files in dataset_path_dict.values():
                cell_type_file_keys.extend(list(ct_files.keys()))
            cell_type_file_keys = set(cell_type_file_keys)
            for key in cell_type_file_keys:
                if key.startswith("atac"):
                    data_key_to_file_type[key] = "bw"
                elif key.startswith("allc") or key.startswith("mc"):
                    data_key_to_file_type[key] = "allc"
                else:
                    raise ValueError(f"Unknown file type for key: {key}")
        self.data_key_to_file_type = data_key_to_file_type

        if cell_types is None:
            cell_types = sorted(dataset_path_dict.keys())
        self.cell_types = cell_types
        # dataset_paths is {data_key: [ct_path1, ct_path2, ...]},
        # ct_path order the same as self.cell_types
        self.dataset_paths = defaultdict(list)
        self.dataset_scale_factors = defaultdict(list)
        for cell_type in cell_types:
            cell_type_files = dataset_path_dict[cell_type]
            for data_key in data_key_to_file_type.keys():
                self.dataset_paths[data_key].append(cell_type_files[data_key])

        # Methylation specific
        self.data_key_to_mc_context = (
            {} if data_key_to_mc_context is None else data_key_to_mc_context
        )

        # Embeddings map generation
        self.embeddings_path = embeddings_path
        with open(embeddings_path) as f:
            self.leg_map = json.load(f)

        self.dataset_path = dataset_path
        self.tn5_bias_path = tn5_bias_path
        self.batch_size = batch_size

        # region properties
        self.dna_window = dna_window
        self.signal_window = signal_window
        self.max_jitter = max_jitter
        self.min_counts = min_cov
        self.max_counts = max_cov
        self.reverse_complement = reverse_complement
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.n_pseudobulks = n_pseudobulks
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio
        self.cov_filter_name = cov_filter_name
        self.max_regions_per_genome_chunk = max_regions_per_genome_chunk
        self.borzoi_regions = BorzoiRegions(self.genome)
        self.merge_cell_type = merge_cell_type
        self.unmethylated = False

        # Borzoi dataset requires these attributes but not used in scPrinter
        self.pos_resolution = 1  # scPrinter is single base pair resolution
        self.normalize_cov = False  # footprint needs raw counts
        self.prefix = "pseudobulk"
        return

    def __repr__(self) -> str:
        _str = (
            f"scPrinterDataset\n"
            f"Dataset directory: {self.dataset_path}\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
        )
        return _str

    def get_footprinter(self) -> BatchFootPrint:
        """
        Get the footprint for a specific region and sample.
        """
        atac_keys = [f"{self.prefix}:bulk_data"]

        fn = BatchFootPrint
        fn_constructor_kwargs = {
            "atac_key": atac_keys,
            "bias_key": "tn5_bias",
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "return_pval": False,
            "smooth_radius": None,
            "numpy": False,
            "device": None,
        }

        footprinter = fn(**fn_constructor_kwargs)
        return footprinter

    def _filter_bed_regions(
        self,
        dataset,
        batch_size=512,
        concurrency=1,
    ):
        cov_filter_key = self.cov_filter_name
        fn = FilterRegions
        fn_constructor_kwargs = {
            "cov_filter_key": cov_filter_key,
            "min_cov": self.min_cov,
            "max_cov": self.max_cov,
            "low_cov_ratio": self.low_cov_ratio,
        }
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _get_dna_one_hot(self, dataset, concurrency):
        """
        Get the DNA one hot for the dataset.
        """
        fn = FetchRegionOneHot
        fn_constructor_kwargs = {"dtype": "bool"}
        fn_kwargs = {
            "remote_genome_one_hot": self.genome.remote_genome_one_hot,
        }

        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            fn_kwargs=fn_kwargs,
            batch_size=512,
            concurrency=concurrency,
        )
        self.dna_column = DNA_NAME
        return dataset

    def _merge_cell_type(self, dataset, data_keys, concurrency=6):
        """Sum up data along the cell type axis."""

        def merge_fn(batch):
            for key in data_keys:
                if key.endswith("_mc_frac"):
                    mc_type = key.replace("_mc_frac", "")
                    mc_data = batch[mc_type + "_mc"]
                    cov_data = batch[mc_type + "_cov"]
                    mc_frac = mc_data.sum(axis=1, keepdims=True) / cov_data.sum(
                        axis=1, keepdims=True
                    )
                    batch[key] = mc_frac
                else:
                    # merge along the cell type axis (axis=1)
                    batch[key] = batch[key].sum(axis=1, keepdims=True)

            return batch

        dataset = dataset.map_batches(
            fn=merge_fn,
            concurrency=concurrency,
            batch_size=512,
        )
        return dataset

    def _rename_keys(self, dataset, key_map):
        """Rename the keys in the dataset."""

        def rename_fn(batch):
            for old_key, new_key in key_map.items():
                batch[new_key] = batch.pop(old_key)
            return batch

        dataset = dataset.map_batches(
            fn=rename_fn,
            concurrency=1,
            batch_size=512,
        )
        return dataset

    def _clip_region_size(self, dataset, signal_keys):
        def _clip_fn(batch):
            """Clip last dimension at center"""
            for key in signal_keys:
                signal = batch[key]
                center = signal.shape[-1] // 2
                signal = signal[
                    ...,
                    center - self.signal_window // 2 : center + self.signal_window // 2,
                ]
                batch[key] = signal
            return batch

        dataset = dataset.map_batches(
            fn=_clip_fn,
            concurrency=1,
            batch_size=512,
        )
        return dataset

    def get_processed_dataset(
        self,
        region_bed: str,
        concurrency=16,
        **kwargs,
    ) -> None:
        """
        Process the dataset and return the processed dataset.

        Parameters
        ----------
        - chroms (list): List of chromosomes to include in the dataset.
        - region_bed (str): Path to the BED file containing the regions.
        - return_cells (bool): Whether to return the cells in the dataset. Default is False.
        - return_regions (bool): Whether to return the regions in the dataset. Default is True.

        Returns
        -------
        - work_ds (Dataset): The processed dataset.

        """
        work_ds = RayRegionDataset.get_processed_dataset(
            self,
            bed=region_bed,
            shuffle_bed=True if self.is_train() else False,
            max_jitter=self.max_jitter,
        )

        # get dna one hot
        work_ds = self._get_dna_one_hot(
            dataset=work_ds,
            concurrency=1,
        )

        # load bigwig data
        data_keys = []

        # tn5 bias
        data_keys.append("tn5_bias")
        work_ds = self._get_bigwig_data(
            dataset=work_ds,
            data_key="tn5_bias",
            bigwig_paths=[self.tn5_bias_path],
            concurrency=concurrency,
            norm_mode=None,
            resolution=1,
            scale_factors=None,
        )

        for data_key, file_type in self.data_key_to_file_type.items():
            file_paths = self.dataset_paths[data_key]
            if file_type in ("bw", "bigwig"):
                data_keys.append(data_key)
                work_ds = self._get_bigwig_data(
                    dataset=work_ds,
                    data_key=data_key,
                    bigwig_paths=file_paths,
                    concurrency=concurrency,
                    norm_mode=None,
                    resolution=1,
                    scale_factors=None,
                )
            elif file_type in ("allc",):
                work_ds = self._get_allc_data(
                    dataset=work_ds,
                    allc_paths=file_paths,
                    mc_prefix=data_key,
                    concurrency=concurrency,
                )
                work_ds = self._get_mc_frac(
                    dataset=work_ds,
                    mc_prefix=data_key,
                )
                data_keys.append(f"{data_key}_mc")
                data_keys.append(f"{data_key}_cov")
                data_keys.append(f"{data_key}_mc_frac")
            else:
                raise ValueError(f"Unknown file type: {file_type}")

        # filter coverage
        if self.cov_filter_name is not None:
            work_ds = self._filter_bed_regions(dataset=work_ds)

        if self.reverse_complement and self.is_train():
            work_ds = self._get_reverse_complement_dataset(
                work_ds,
                dna_key="dna_one_hot",
                data_1d_keys=data_keys,
                data_2d_keys=None,
                chance=0.5,
                concurrency=(1, 6),
                batch_size=512,
            )

        # clip signal size
        work_ds = self._clip_region_size(
            dataset=work_ds,
            signal_keys=data_keys,
        )

        # split cell types
        if self.merge_cell_type:
            work_ds = self._merge_cell_type(
                dataset=work_ds,
                data_keys=[k for k in data_keys if k != "tn5_bias"],
                concurrency=6,
            )
        else:
            work_ds = self._convert_to_list_dict(
                work_ds,
                dna_key="dna_one_hot",
                # important: tn5_bias is not cell type specific
                data_1d_keys=[k for k in data_keys if k != "tn5_bias"],
                data_2d_keys=[],
                concurrency=6,
                key_suffix=None,
                keep_channel_dim=True,
            )

        # Turn region into global coordinates (str to numbers)
        work_ds = self._process_region_columns(dataset=work_ds, keep_regions=True)
        work_ds = work_ds.drop_columns(["Original_Name"])

        # rename keys
        work_ds = self._rename_keys(
            dataset=work_ds,
            key_map={
                "atac": "pseudobulk:bulk_data",
            },
        )
        return work_ds

    def get_dataloader(
        self,
        folds,
        region_bed,
        as_torch=True,
        n_batches=None,
        concurrency=8,
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
        _ = folds
        shuffle_rows = int(500 * (self.n_pseudobulks + 1))
        shuffle_rows = max(shuffle_rows, 1000)
        shuffle_rows = min(shuffle_rows, 5000)
        dataloader_kwargs["shuffle_rows"] = shuffle_rows

        loader = super().get_dataloader(
            region_bed=region_bed,
            as_torch=as_torch,
            n_batches=n_batches,
            concurrency=concurrency,
            **dataloader_kwargs,
        )
        return loader

    def get_train_valid_test(self, fold):
        """Get the train, valid, and test regions for the given fold."""
        (
            train_folds,
            valid_folds,
            test_folds,
            borzoi_train_regions,
            borzoi_valid_regions,
            borzoi_test_regions,
        ) = BorzoiDatasetOnline.get_train_valid_test(self, fold)

        # convert regions to peak regions
        # TrainerBorzoiDatasetMixin uses the Borzoi regions as train/valid/test regions
        # Here we need to intersect the Borzoi regions with the peak regions
        def _intersect_region_with_borzoi_regions(region_bed, borzoi_regions):
            borzoi_regions = pr.PyRanges(borzoi_regions)
            region_bed = region_bed.overlap(borzoi_regions).as_df()
            region_bed["Original_Name"] = region_bed["region"]
            return region_bed

        region_bed = pr.PyRanges(self.bed)
        train_regions = _intersect_region_with_borzoi_regions(
            region_bed, borzoi_train_regions
        )
        valid_regions = _intersect_region_with_borzoi_regions(
            region_bed, borzoi_valid_regions
        )
        test_regions = _intersect_region_with_borzoi_regions(
            region_bed, borzoi_test_regions
        )

        return (
            train_folds,
            valid_folds,
            test_folds,
            train_regions,
            valid_regions,
            test_regions,
        )
