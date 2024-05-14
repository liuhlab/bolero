from typing import Any, Iterable, Optional, Union

import numpy as np
import torch
from ray.data.dataset import Dataset

from bolero.tl.dataset.filters import RowSumFilter
from bolero.tl.dataset.ray_dataset import RayGenomeDataset
from bolero.tl.dataset.ray_sc_dataset import RaySingleCellDataset
from bolero.tl.dataset.transforms import (
    BatchRegionEmbedding,
    BatchToFloat,
    CropRegionsWithJitter,
    FetchRegionOneHot,
    ReverseComplement,
)
from bolero.tl.footprint import FootPrintModel
from bolero.utils import try_gpu


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
        bias_data = data[self.bias_key]
        for atac in self.atac_key:
            try:
                atac_data = data[atac]
            except KeyError:
                continue

            result = self.footprint_from_data(
                atac_data=atac_data,
                bias_data=bias_data,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                modes=modes if modes is not None else self.modes,
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
    set_min_max_counts_cutoff(column: str) -> None:
        Set the minimum and maximum counts cutoff based on the given column.
    _filter_by_coverage(column: str) -> None:
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
        columns: Optional[list[str]] = None,
        bias_name: str = None,
        batch_size: int = 64,
        dna_window: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        cov_min_q: float = 0.0001,
        cov_max_q: float = 0.9999,
        clip_min: float = -10,
        clip_max: float = 10,
        reverse_complement: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize a scPrinterDataset object.

        Parameters
        ----------
        dataset : Dataset
            The Ray dataset.
        columns : Optional[List[str]], optional
            The list of columns to select, if None, all columns are selected (default is None).
        bias_name : str, optional
            The name of the bias.
        batch_size : int, optional
            The batch size (default is 64).
        dna_window : int, optional
            The size of the DNA window (default is 1840).
        signal_window : int, optional
            The size of the signal window (default is 1000).
        max_jitter : int, optional
            The maximum jitter value (default is 128).
        cov_min_q : float, optional
            The minimum quantile value for coverage (default is 0.0001).
        cov_max_q : float, optional
            The maximum quantile value for coverage (default is 0.9999).
        clip_min : float, optional
            The minimum clip value (default is -10).
        clip_max : float, optional
            The maximum clip value (default is 10).
        reverse_complement : bool, optional
            Whether to use reverse complement (default is True).
        **kwargs
            Additional keyword arguments passed to the ray.data.read_parquet.

        Returns
        -------
        None
        """
        super().__init__(dataset, columns=columns, **kwargs)
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
        self.cov_min_q = cov_min_q
        self.cov_max_q = cov_max_q
        self.reverse_complement = reverse_complement
        self.clip_min = clip_min
        self.clip_max = clip_max
        return

    def __repr__(self) -> str:
        _super_str = super().__repr__()
        _str = (
            f"scPrinterDataset for {len(self)} regions.\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
            f"DNA name: {self.dna_name}, Bias name: {self.bias_name}\n"
            f"Regions: {self.regions},\nSamples: {self.samples}\n" + _super_str
        )
        return _str

    def _dataset_preprocess(self, column) -> None:
        """
        Preprocess the dataset.

        Returns
        -------
        None
        """
        # row operations
        if column is not None:
            self._filter_by_coverage(column)
        if self.reverse_complement and self._dataset_mode == "train":
            self._reverse_complement_region()
        self._crop_regions()
        # batch operations
        self._dna_to_float()
        return

    def get_footprinter(
        self,
        region: Optional[str] = None,
        sample: Optional[str] = None,
        visualizer=False,
    ) -> BatchFootPrint:
        """
        Get the footprint for a specific region and sample.

        Parameters
        ----------
        region : str, optional
            The region name (default is None).
        sample : str, optional
            The sample name (default is None).
        visualizer : bool, optional
            Whether to return p-values and smooth the data for better visualization (default is False).
            For ML model, we used the raw z-scores to perform training and prediction, so this should be False.

        Returns
        -------
        BatchFootPrint
            The footprint for the specified region and sample.
        """
        if region is None:
            if len(self.regions) == 1:
                region = self.regions[0]
            else:
                raise ValueError(
                    "Region name not provided and could not be guessed, please provide the region name."
                )
        if sample is None:
            samples = self.samples
        else:
            samples = sample
            if isinstance(samples, str):
                samples = [samples]

        if visualizer:
            return_pval = True
            smooth_radius = 5
            numpy = True
        else:
            return_pval = False
            smooth_radius = None
            numpy = False

        atac_keys = [f"{region}|{sample}" for sample in samples]
        footprint = BatchFootPrint(
            atac_key=atac_keys,
            bias_key=f"{region}|{self.bias_name}",
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            return_pval=return_pval,
            smooth_radius=smooth_radius,
            numpy=numpy,
            device=None,
        )
        return footprint

    def get_dataloader(
        self,
        sample: Optional[str] = None,
        region: Optional[str] = None,
        downsample_rate: float = 1,
        local_shuffle_buffer_size: int = 10000,
        randomize_block_order: bool = False,
        as_torch=True,
        batch_collate_fn=None,
        **kwargs,
    ) -> Iterable:
        """
        Get a PyTorch DataLoader for the specified sample and region.

        Parameters
        ----------
        sample : str, optional
            The name of the sample (default is None).
        region : str, optional
            The name of the region (default is None).
        downsample_rate : float, optional
            The downsample rate (default is 1).
        local_shuffle_buffer_size : int, optional
            The size of the local shuffle buffer (default is 10000).
        randomize_block_order : bool, optional
            Whether to randomize the block order (default is False).
        as_torch : bool, optional
            Whether to return a iterator with batches data in torch tensor format (default is True).
        batch_collate_fn : list, optional
            A list of functions to apply to the batch data AFTER all the default preprocessing steps (default is None).
            These functions will be added through the self._working_dataset.map_batches() function.
        **kwargs
            Additional keyword arguments passed to the DataLoader.

        Returns
        -------
        Iterable
            Batch iterator similar to PyTorch DataLoader.
        """
        if self._dataset_mode is None:
            raise ValueError(
                "Set .train() or .eval() first before calling .get_dataloader()"
            )
        self._working_dataset = self.dataset

        if sample is None:
            if len(self.samples) == 1:
                sample = self.samples[0]
        if region is None:
            if len(self.regions) == 1:
                region = self.regions[0]
        if sample is None or region is None:
            filter_column = None
        else:
            filter_column = f"{region}|{sample}"

        if as_torch:
            # the torch iterator can only handle float, int, and bool columns to torch tensors
            use_columns = []
            possible_dtypes = ("float", "int", "bool")
            for column in self.columns:
                column_schema = self.schema[column]
                try:
                    dtype = str(column_schema.scalar_type)
                except AttributeError:
                    dtype = str(column_schema)
                for possible_dtype in possible_dtypes:
                    if possible_dtype in dtype:
                        use_columns.append(column)
                        break
            self._working_dataset = self._working_dataset.select_columns(use_columns)

        if randomize_block_order:
            self._working_dataset = self._working_dataset.randomize_block_order()

        if downsample_rate < 1:
            self._working_dataset = self._working_dataset.random_sample(
                fraction=downsample_rate
            )

        self._dataset_preprocess(filter_column)

        if batch_collate_fn is not None:
            if not isinstance(batch_collate_fn, list):
                batch_collate_fn = [batch_collate_fn]
            for func in batch_collate_fn:
                self._working_dataset = self._working_dataset.map_batches(func)

        if "drop_last" not in kwargs:
            kwargs["drop_last"] = True if self._dataset_mode == "train" else False
        if local_shuffle_buffer_size < self.batch_size:
            local_shuffle_buffer_size = None
        if as_torch:
            loader = self._working_dataset.iter_torch_batches(
                batch_size=self.batch_size,
                local_shuffle_buffer_size=local_shuffle_buffer_size,
                **kwargs,
            )
        else:
            loader = self._working_dataset.iter_batches(
                batch_size=self.batch_size,
                local_shuffle_buffer_size=local_shuffle_buffer_size,
                **kwargs,
            )
        return loader

    def set_min_max_counts_cutoff(self, column: str) -> None:
        """
        Set the minimum and maximum counts cutoff based on the given column.

        Parameters
        ----------
        column : str
            The column name.

        Returns
        -------
        None
        """
        _stats = self.summary_stats[column]

        min_ = np.quantile(_stats, self.cov_min_q)
        self.min_counts = max(self.min_counts, min_)

        max_ = np.quantile(_stats, self.cov_max_q)
        self.max_counts = min(self.max_counts, max_)
        return

    def get_dna_and_signal_columns(
        self, separate_bias: bool = False
    ) -> tuple[list[str], list[str]]:
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


class scPrinterSingleCellDataset(RaySingleCellDataset):
    """Singel cell dataset for scPrinter model."""

    def __init__(
        self,
        dataset: str,
        genome,
        chroms: Optional[list[str]] = None,
        use_prefixs: Optional[list[str]] = None,
        batch_size: int = 64,
        dna_window: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        clip_min: float = -10,
        clip_max: float = 10,
        sample_regions: int = 200,
        n_pseudobulks: int = 10,
        min_cov: int = 10,
        max_cov: int = 100000,
        low_cov_ratio: float = 0.1,
        reverse_complement: bool = True,
        override_num_blocks=None,
    ):
        super().__init__(
            dataset_path=dataset,
            use_prefixs=use_prefixs,
            override_num_blocks=override_num_blocks,
            chroms=chroms,
            genome=genome,
        )
        self.batch_size = batch_size

        # region properties
        self.dna_window = dna_window
        self.signal_window = signal_window
        self.max_jitter = max_jitter
        self.min_counts = 10
        self.max_counts = 1e16
        self.reverse_complement = reverse_complement
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sample_regions = sample_regions
        self.n_pseudobulks = n_pseudobulks
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio

        self.dna_columns = ["dna_one_hot"]
        self.signal_columns = ["bulk_data", "tn5_bias"]

        self.region_embedding = None
        self._working_dataset = None
        return

    def __repr__(self) -> str:
        _super_str = super().__repr__()
        _str = (
            f"scPrinterDataset for {len(self)} regions.\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
            + _super_str
        )
        return _str

    def _get_crop_regions(self, *args, **kwargs) -> None:
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

        dna_columns = self.dna_columns
        signal_columns = self.signal_columns
        key_list = dna_columns + signal_columns
        length_list = [self.dna_window] * len(dna_columns) + [self.signal_window] * len(
            signal_columns
        )

        _cropper = CropRegionsWithJitter(
            key=key_list,
            final_length=length_list,
            max_jitter=max_jitter,
            input_type="batch",
        )
        return _cropper

    def _get_reverse_complement_region(self, *args, **kwargs) -> None:
        """
        Reverse complement the DNA sequences by 50% probability.

        Returns
        -------
        None
        """
        _rc = ReverseComplement(
            dna_key=self.dna_columns, signal_key=self.signal_columns, input_type="row"
        )
        return _rc

    def _get_add_region_embedding(self):
        embedder = BatchRegionEmbedding(
            embedding=self.region_embedding, region_key="region", pop_region_key=True
        )
        return embedder

    def add_region_embedding(self, embedding):
        """Add a predefined region embedding to the dataset."""
        self.region_embedding = embedding
        return

    def _fetch_dna_one_hot(self) -> None:
        """
        Fetch the DNA one hot.

        Returns
        -------
        None
        """
        # add DNA one hot
        one_hot_processor = FetchRegionOneHot(
            genome=self.genome,
            region_key="region",
            output_key="dna_one_hot",
            dtype="float32",
        )
        return one_hot_processor

    def _dataset_preprocess(self) -> None:
        super()._dataset_preprocess(
            sample_regions=self.sample_regions,
            n_pseudobulks=self.n_pseudobulks,
            min_cov=self.min_cov,
            max_cov=self.max_cov,
            low_cov_ratio=self.low_cov_ratio,
        )

        batch_funcs = []

        one_hot_processor = self._fetch_dna_one_hot()
        batch_funcs.append(one_hot_processor)

        cropper = self._get_crop_regions()
        batch_funcs.append(cropper)

        if self.reverse_complement and self._dataset_mode == "train":
            rc = self._get_reverse_complement_region()
            batch_funcs.append(rc)

        if self.region_embedding is not None:
            embedder = self._get_add_region_embedding()
            batch_funcs.append(embedder)
        else:
            # manually drop the region column to keep numbers columns only
            def _func(data):
                data.pop("region", None)
                return data

            batch_funcs.append(_func)

        def _compose_funcs(data):
            for func in batch_funcs:
                data = func(data)
            return data

        # self._working_dataset = self._working_dataset.map(_compose_funcs)
        return _compose_funcs

    def get_footprinter(
        self,
        visualizer=False,
    ) -> BatchFootPrint:
        """
        Get the footprint for a specific region and sample.

        Parameters
        ----------
        region : str, optional
            The region name (default is None).
        sample : str, optional
            The sample name (default is None).
        visualizer : bool, optional
            Whether to return p-values and smooth the data for better visualization (default is False).
            For ML model, we used the raw z-scores to perform training and prediction, so this should be False.

        Returns
        -------
        BatchFootPrint
            The footprint for the specified region and sample.
        """
        if visualizer:
            return_pval = True
            smooth_radius = 5
            numpy = True
        else:
            return_pval = False
            smooth_radius = None
            numpy = False

        atac_keys = ["bulk_data"]
        bias_key = "tn5_bias"
        footprint = BatchFootPrint(
            atac_key=atac_keys,
            bias_key=bias_key,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            return_pval=return_pval,
            smooth_radius=smooth_radius,
            numpy=numpy,
            device=None,
        )
        return footprint

    def get_dataloader(
        self,
        as_torch=True,
        device=None,
        downsample_rate=1,
        **kwargs,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, by default 64.
        sample_regions : int, optional
            The number of sample regions, by default 200.
        n_pseudobulks : int, optional
            The number of pseudobulks, by default 10.
        min_cov : int, optional
            The minimum coverage, by default 10.
        max_cov : int, optional
            The maximum coverage, by default 100000.
        low_cov_ratio : float, optional
            The low coverage ratio, by default 0.1.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DataLoader
            The dataloader.
        """
        self._working_dataset = self._dataset

        if downsample_rate < 1:
            self._working_dataset = self._working_dataset.random_sample(
                fraction=downsample_rate
            )

        additional_funcs = self._dataset_preprocess()

        _default = {
            "prefetch_batches": 5,
            "local_shuffle_buffer_size": 5000,
            "drop_last": True,
            "batch_size": self.batch_size,
        }
        _default.update(kwargs)

        loader = self._working_dataset.iter_batches(**_default)

        for batch in loader:
            batch = additional_funcs(batch)
            if as_torch:
                if device is None:
                    device = try_gpu()
                batch = {k: torch.Tensor(v.copy()).to(device) for k, v in batch.items()}
            batch["cell_embedding"] = batch.pop("bulk_embedding")
            yield batch
