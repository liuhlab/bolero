from typing import Iterable, Optional, Union

import numpy as np

from bolero.tl.dataset.filters import RowSumFilter
from bolero.tl.dataset.ray_dataset import RayGenomeDataset
from bolero.tl.dataset.transforms import (
    AddChannels,
    BatchToFloat,
    CropRegionsWithJitter,
    ReverseComplement,
)
from bolero.tl.model.generic.train_helper import validate_config


class Track1DDataset(RayGenomeDataset):
    """
    RayDataset class for working with bulk 1-D track model.

    Parameters
    ----------
    dataset : ray.data.Dataset
        The Ray dataset.
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

    default_config = {
        "dataset_path": "REQUIRED",
        "dataset_columns": None,
        "batch_size": 64,
        "dna_window": 1840,
        "signal_window": 1000,
        "max_jitter": 128,
        "cov_min_q": 0.0001,
        "cov_max_q": 0.9999,
        "local_shuffle_buffer_size": 5000,
        "reverse_complement": True,
    }

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Get the default configuration.

        Returns
        -------
        dict
            The default configuration.
        """
        return cls.default_config

    @classmethod
    def create_from_config(
        cls,
        config: dict,
    ) -> "Track1DDataset":
        """
        Create a Bulk1DTrackDataset object from the configuration.

        Parameters
        ----------
        config : dict
            The configuration.
        remove_additional_keys : bool, optional
            Whether to remove additional keys in the configuration (default is True).

        Returns
        -------
        Bulk1DTrackDataset
            The Bulk1DTrackDataset object.
        """
        validate_config(config, cls.default_config)
        # remove additional keys in the configuration
        config = {k: v for k, v in config.items() if k in cls.default_config}
        return cls(**config)

    def __init__(
        self,
        dataset_path: Union[str, list[str]],
        dataset_columns: Optional[list[str]] = None,
        batch_size: int = 64,
        dna_window: int = 1840,
        signal_window: int = 1000,
        max_jitter: int = 128,
        cov_min_q: float = 0.0001,
        cov_max_q: float = 0.9999,
        local_shuffle_buffer_size: int = 5000,
        reverse_complement: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize a Bulk1DTrackDataset object.

        Parameters
        ----------
        dataset : Dataset
            The Ray dataset.
        columns : Optional[List[str]], optional
            The list of columns to select, if None, all columns are selected (default is None).
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
        reverse_complement : bool, optional
            Whether to use reverse complement (default is True).
        **kwargs
            Additional keyword arguments passed to the ray.data.read_parquet.

        Returns
        -------
        None
        """
        super().__init__(dataset_path, columns=dataset_columns, **kwargs)

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

        # data loader
        self.local_shuffle_buffer_size = local_shuffle_buffer_size
        return

    def __repr__(self) -> str:
        _super_str = super().__repr__()
        _str = (
            f"BulkGenomeDataset for {len(self)} regions.\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
            f"DNA name: {self.dna_name},\n"
            f"Regions: {self.regions},\nSamples: {self.samples}\n" + _super_str
        )
        return _str

    def _dataset_preprocess(self, dataset, column) -> None:
        """
        Preprocess the dataset.

        Returns
        -------
        None
        """
        # row operations
        if column is not None:
            dataset = self._filter_by_coverage(dataset, column)
        if self.reverse_complement and self._dataset_mode == "train":
            dataset = self._reverse_complement_region(dataset)
        dataset = self._crop_regions(dataset)
        # batch operations
        dataset = self._dna_to_float(dataset)
        dataset = self._add_channels(dataset)
        return dataset

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

    def get_dna_and_signal_columns(self) -> tuple[list[str], list[str]]:
        """
        Get the DNA and signal columns from the dataset.

        Returns
        -------
        Tuple[List[str], List[str]]
            A tuple containing the DNA columns and signal columns.
        """
        dna_columns = []
        signal_columns = []
        for column in self.columns:
            try:
                _, sample = column.split("|")
            except ValueError:
                continue
            if sample == self.dna_name:
                dna_columns.append(column)
            else:
                signal_columns.append(column)
        return dna_columns, signal_columns

    def _filter_by_coverage(self, dataset, column: str) -> None:
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
        dataset = dataset.filter(_filter)
        return dataset

    def _dna_to_float(self, dataset) -> None:
        """
        Convert the DNA data to float.

        Returns
        -------
        None
        """
        dna_columns, _ = self.get_dna_and_signal_columns()
        _map = BatchToFloat(dna_columns)
        dataset = dataset.map_batches(_map)
        return dataset

    def _reverse_complement_region(self, dataset, *args, **kwargs) -> None:
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
        dataset = dataset.map(_rc, *args, **kwargs)
        return dataset

    def _crop_regions(self, dataset, *args, **kwargs) -> None:
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
            crop_axis=0,
        )
        dataset = dataset.map(_cropper, *args, **kwargs)
        return dataset

    def _add_channels(self, dataset, *args, **kwargs) -> None:
        """
        Add channels to the dataset.

        Returns
        -------
        None
        """
        _, signal_columns = self.get_dna_and_signal_columns()
        channel_func = lambda x: np.expand_dims(x, 1)
        _map = AddChannels(signal_columns, channel_func=channel_func)
        dataset = dataset.map_batches(_map, *args, **kwargs)
        return dataset

    def get_dataloader(
        self,
        sample: Optional[str] = None,
        region: Optional[str] = None,
        as_torch=True,
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
        local_shuffle_buffer_size : int, optional
            The size of the local shuffle buffer (default is 5000).
        as_torch : bool, optional
            Whether to return a iterator with batches data in torch tensor format (default is True).
        **kwargs
            Additional keyword arguments passed to the DataLoader.

        Returns
        -------
        Iterable
            Batch iterator similar to PyTorch DataLoader.
        """
        _working_dataset = self.dataset

        if self._dataset_mode is None:
            raise ValueError(
                "Set .train() or .eval() first before calling .get_dataloader()"
            )

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
            _working_dataset = _working_dataset.select_columns(use_columns)

        # preprocess dataset
        _working_dataset = self._dataset_preprocess(_working_dataset, filter_column)

        # get data loader
        default_kwargs = {
            "drop_last": True if self._dataset_mode == "train" else False,
            "local_shuffle_buffer_size": self.local_shuffle_buffer_size,
        }
        kwargs = {**default_kwargs, **kwargs}
        if as_torch:
            loader = _working_dataset.iter_torch_batches(
                batch_size=self.batch_size, **kwargs
            )
        else:
            loader = _working_dataset.iter_batches(batch_size=self.batch_size, **kwargs)
        return loader
