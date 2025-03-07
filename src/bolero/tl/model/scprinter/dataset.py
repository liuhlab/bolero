from collections import OrderedDict
from copy import deepcopy
from typing import Any, Iterable, Union

import numpy as np

from bolero.tl.dataset.ray_dataset import (
    RayGenomeChunkDataset,
)
from bolero.tl.dataset.sc_transforms import FilterRegions
from bolero.tl.dataset.transforms import CropLastAxisWithJitter, FetchRegionOneHot
from bolero.tl.footprint import FootPrintModel
from bolero.tl.model.borzoi.dataset import BorzoiDataset
from bolero.tl.model.borzoi.utils import BorzoiRegions

DNA_NAME = "dna_one_hot"


class BatchFootPrint(FootPrintModel):
    """Apply footprint transformation to the given data batch."""

    def __init__(
        self,
        atac_key: Union[str, list[str]],
        bias_key: str,
        modes: np.ndarray = None,
        clip_min: float = -10,
        clip_max: float = 10,
        return_pval: bool = False,
        smooth_radius: int = None,
        numpy=False,
        device=None,
        tfbs_score_all: bool = False,
        tfbs_score_class1: bool = False,
        nucleosome_score: bool = False,
    ):
        """
        Apply footprint transformation to the given data dictionary.

        Args:
            atac_key (Union[str, List[str]]): Key(s) for the ATAC data in the data dictionary.
            bias_key (str): Key for the bias data in the data dictionary.
            modes (np.ndarray): Modes for the footprint transformation.
            clip_min (float, optional): Minimum value for clipping. Defaults to -10.
            clip_max (float, optional): Maximum value for clipping. Defaults to 10.
            return_pval (bool, optional): Whether to return p-values. Defaults to False.
            smooth_radius (int, optional): Radius for smoothing. Defaults to None.
            numpy (bool, optional): Whether to use numpy. Defaults to True.
            device ([type], optional): Device for the model. Defaults to None.
            tfbs_score_all (bool, optional): Whether to use all TFBS scores. Defaults to False.
            tfbs_score_class1 (bool, optional): Whether to use class 1 TFBS scores. Defaults to False.
            nucleosome_score (bool, optional): Whether to use nucleosome scores. Defaults to False.
        """
        if modes is None:
            modes = np.arange(2, 101, 1)
        else:
            modes = np.array(modes)
        super().__init__(bias_bw_path=None, dispmodels=None, modes=modes, device=device)

        # get the device from the parameters
        self.device = next(self.parameters()).device

        if isinstance(atac_key, str):
            atac_key = [atac_key]
        self.atac_key = atac_key
        self.bias_key = bias_key
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.return_pval = return_pval
        self.smooth_radius = smooth_radius
        self.numpy = numpy
        self.tfbs_score_all = tfbs_score_all
        self.tfbs_score_class1 = tfbs_score_class1
        self.nucleosome_score = nucleosome_score

    def __call__(self, data: dict, modes: np.array = None) -> dict:
        """
        Apply the footprint transformation to the given data.

        Args:
            data (dict): Input data dictionary.

        Returns
        -------
            dict: Transformed data dictionary.
        """
        modes = modes if modes is not None else self.modes
        bias_data = data[self.bias_key]
        # if bias_data has 3 dims, drop the second dim (channels)
        if bias_data.ndim == 3:
            bias_data = bias_data.squeeze(1)

        for atac in self.atac_key:
            try:
                atac_data = data[atac]
                # if atac_data has 3 dims, drop the second dim (channels)
                if atac_data.ndim == 3:
                    atac_data = atac_data.squeeze(1)
            except KeyError:
                continue

            result = self.footprint_from_data(
                atac_data=atac_data,
                bias_data=bias_data,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                modes=modes,
                return_pval=self.return_pval,
                smooth_radius=self.smooth_radius,
                numpy=self.numpy,
                tfbs_score_all=self.tfbs_score_all,
                tfbs_score_class1=self.tfbs_score_class1,
                nucleosome_score=self.nucleosome_score,
            )
            if isinstance(result, tuple):
                fp, scores = result
            else:
                fp = result
                scores = {}
            data[f"{atac}_footprint"] = fp
            for key, val in scores.items():
                data[f"{atac}_{key}"] = val
        return data


class scPrinterDataset(BorzoiDataset, RayGenomeChunkDataset):
    """Singel cell dataset for scPrinter model."""

    default_config = {
        "dataset_path": "REQUIRED",
        "genome": "REQUIRED",
        "batch_size": 128,
        "dna_window": 1840,
        "signal_window": 1000,
        "max_jitter": 128,
        "clip_min": -10,
        "clip_max": 10,
        "n_pseudobulks": 10,
        "cov_filter_name": "REQUIRED",
        "min_cov": 40,
        "max_cov": 100000,
        "low_cov_ratio": 0.1,
        "reverse_complement": True,
        "shuffle_files": True,
        "read_parquet_kwargs": None,
        "max_regions_per_genome_chunk": 100,
    }

    def __init__(
        self,
        dataset_path: str,
        genome,
        batch_size: int = 128,
        dna_window: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        clip_min: float = -10,
        clip_max: float = 10,
        n_pseudobulks: int = 10,
        min_cov: int = 40,
        max_cov: int = 100000,
        low_cov_ratio: float = 0.1,
        cov_filter_name: str = None,
        reverse_complement: bool = True,
        shuffle_files=True,
        read_parquet_kwargs=None,
        max_regions_per_genome_chunk: int = 100,
        **kwargs,
    ):
        RayGenomeChunkDataset.__init__(
            self,
            dataset_path=dataset_path,
            genome=genome,
            shuffle_files=shuffle_files,
            read_parquet_kwargs=read_parquet_kwargs,
            max_regions_per_genome_chunk=max_regions_per_genome_chunk,
        )
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

        self.bias_column = "tn5_bias"
        self.name_to_pseudobulker = OrderedDict()

        self.borzoi_regions = BorzoiRegions(self.genome)
        self.use_regions = "borzoi"

        # Borzoi dataset requires these attributes but not used in scPrinter
        self.pos_resolution = 1  # scPrinter is single base pair resolution
        self.reduce_resolution = False
        self.normalize_cov = False  # footprint needs raw counts
        self.paired_data = False

        self.prefix = kwargs.pop("prefix", "pseudobulk")
        return

    def __repr__(self) -> str:
        _str = (
            f"scPrinterDataset\n"
            f"Dataset directory: {self.dataset_path}\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
        )
        return _str

    def _get_region_cropper(self, dataset, batch_size=512) -> None:
        """
        Crop the regions in the working dataset.

        Returns
        -------
        None
        """
        if self.is_eval():
            max_jitter = 0
        else:
            max_jitter = self.max_jitter

        signal_columns = self.signal_columns
        key_list = [self.dna_column] + signal_columns
        length_list = [self.dna_window] + [self.signal_window] * len(signal_columns)

        _cropper = CropLastAxisWithJitter(
            key=key_list,
            final_length=length_list,
            max_jitter=max_jitter,
        )

        def _cropper_squeeze(data):
            data = _cropper(data)
            for sig_col in signal_columns:
                # also reduce single channel signals to 1D
                _data = data[sig_col]
                if _data.ndim == 3 and _data.shape[1] == 1:
                    data[sig_col] = data[sig_col].squeeze(1)
            return data

        dataset = dataset.map_batches(_cropper_squeeze, batch_size=batch_size)
        return dataset

    def get_footprinter(self) -> BatchFootPrint:
        """
        Get the footprint for a specific region and sample.
        """
        atac_keys = [f"{self.prefix}:bulk_data"]

        fn = BatchFootPrint
        fn_constructor_kwargs = {
            "atac_key": atac_keys,
            "bias_key": self.bias_column,
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
        cov_filter_key = f"{self.cov_filter_name}:bulk_data"
        assert (
            cov_filter_key in self.signal_columns
        ), f"cov_filter_key {cov_filter_key} not in {self.signal_columns}"

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

    def _get_dna_one_hot(self, dataset, concurrency, batch_size=1024):
        fn = FetchRegionOneHot
        fn_constructor_kwargs = {
            # TODO HL:
            # the random_shift parameter is not used in scPrinter
            # although we could change scPrinter dataloader to use the same way as Borzoi
            "random_shift": 0,
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
        self.dna_column = "dna_one_hot"
        return dataset

    def get_processed_dataset(
        self,
        folds: list[str],
        region_bed: str,
        return_cells: bool = False,
        concurrency=16,
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
        standard_length = (
            max(self.dna_window, self.signal_window) + self.max_jitter * 2 + 200
        )
        standard_length = int(standard_length)
        region_bed = self.standard_region_length(
            region_bed, standard_length, keep_original=True
        )

        compressed_bytes_to_tensor_concurrency = (1, concurrency // 4)
        generate_pseudobulk_concurrency = (1, concurrency // 4)
        generate_regions_concurrency = (1, concurrency)
        work_ds = self._get_processed_dataset(
            folds=folds,
            region_bed=region_bed,
            bypass_keys=[self.bias_column],
            return_rows=return_cells,
            inplace=False,
            region_action_keys=[self.bias_column],
            concurrency=concurrency,
            add_original_name=False,
            compressed_bytes_to_tensor_concurrency=compressed_bytes_to_tensor_concurrency,
            generate_pseudobulk_concurrency=generate_pseudobulk_concurrency,
            generate_regions_concurrency=generate_regions_concurrency,
        )

        # add dna one hot
        work_ds = self._get_dna_one_hot(
            dataset=work_ds,
            concurrency=1,
        )

        work_ds = self._get_region_cropper(work_ds)

        # filter coverage
        # IMPORTANT: region cov filter must be put after cropping,
        # because region cov changes after cropping
        if self.cov_filter_name is not None:
            work_ds = self._filter_bed_regions(dataset=work_ds)

        if self.reverse_complement and self.is_train():
            work_ds = self._get_reverse_complement_region(work_ds, batch_size=512)

        # remove region column OR turn it into global coordinates (str to numbers)
        work_ds = self._process_region_columns(dataset=work_ds, keep_regions=True)
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
        shuffle_rows = int(500 * (self.n_pseudobulks + 1))
        shuffle_rows = max(shuffle_rows, 1000)
        shuffle_rows = min(shuffle_rows, 5000)
        dataloader_kwargs["shuffle_rows"] = shuffle_rows

        loader = super().get_dataloader(
            folds=folds,
            region_bed=region_bed,
            as_torch=as_torch,
            n_batches=n_batches,
            concurrency=concurrency,
            **dataloader_kwargs,
        )
        return loader

    def get_train_valid_test(
        self,
        fold,
        downsample_train_region=None,
        downsample_valid_region=None,
        downsample_test_region=None,
        seed=0,
    ):
        """Get the train, valid, and test folds and regions."""
        # still use Borzoi region length here to be consistent with Borzi
        # later we will overlap borzoi regions with the peak regions
        (
            train_folds,
            valid_folds,
            test_folds,
            borzoi_train_regions,
            borzoi_valid_regions,
            borzoi_test_regions,
        ) = super().get_train_valid_test(
            fold=fold,
            downsample_train_region=None,
            downsample_valid_region=None,
            downsample_test_region=None,
            region_length=524288,
            seed=seed,
        )

        # convert regions to peak regions
        # TrainerBorzoiDatasetMixin uses the Borzoi regions as train/valid/test regions
        # Here we need to intersect the Borzoi regions with the peak regions
        import pyranges as pr

        def _intersect_region_with_borzoi_regions(region_bed, borzoi_regions):
            borzoi_regions = pr.PyRanges(borzoi_regions)
            region_bed = region_bed.overlap(borzoi_regions).as_df()
            try:
                region_bed["Original_Name"] = region_bed["region"]
            except KeyError:
                region_bed["Original_Name"] = region_bed["Name"]
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


class GenerateBaseModelPseudobulk:
    """
    Simply sum the rows to generate pseudobulks.
    """

    def __init__(self, sample_rows=1000, prefix="pseudobulk", **kwargs):
        self.sample_rows = sample_rows
        self.prefix = prefix

        # fix by set local random state
        self.rand_state = np.random.RandomState(seed=42)
        return

    def __call__(self, data_dict: dict[str, bytes]) -> list[dict[str, np.ndarray]]:
        """Generate pseudobulks for each output prefix."""
        # row_by_base is a csr_matrix of shape (n_rows, region_length)
        row_by_base = data_dict.pop(self.prefix)

        nrows = row_by_base.shape[0]
        if nrows > self.sample_rows:
            use_rows = self.rand_state.choice(nrows, self.sample_rows, replace=False)
        else:
            use_rows = nrows

        _bulk_values = row_by_base[use_rows].sum(axis=0).A1  # (region_length,)
        data_dict[f"{self.prefix}:bulk_data"] = _bulk_values

        # return list of dicts
        return [data_dict]


class scPrinterDatasetBase(scPrinterDataset):
    default_config = deepcopy(scPrinterDataset.default_config)
    default_config.update(
        {
            "prefix": "pseudobulk",
            "sample_rows": 1000,
            "cov_scale": 1,
            "fix_sample_rows": True,
        }
    )

    def __init__(self, *args, **kwargs):
        self.sample_rows = kwargs.pop("sample_rows", 1000)
        print(
            f"Getting pseudobulk with random {self.sample_rows} rows in {self.prefix} data_key."
        )

        super().__init__(*args, **kwargs)
        self.name_to_pseudobulker = {self.prefix: None}
        return

    def _generate_pseudobulk(
        self,
        dataset,
        concurrency=1,
        **kwargs,
    ):
        fn = GenerateBaseModelPseudobulk
        fn_constructor_kwargs = {
            "sample_rows": self.sample_rows,
            "prefix": self.prefix,
            **kwargs,
        }

        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )

        # update region_action_keys
        self.signal_columns.append(f"{self.prefix}:bulk_data")
        return dataset
