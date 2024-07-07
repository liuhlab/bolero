from typing import Optional, Union

from bolero.tl.dataset.ray_dataset import RayGenomeChunkDataset
from bolero.tl.dataset.sc_transforms import FilterRegions
from bolero.tl.dataset.transforms import CropLastAxisWithJitter, ReverseComplement


class Track1DDataset(RayGenomeChunkDataset):
    """
    Track1DDataset class for working with bulk 1-D track model.

    Parameters
    ----------
    dataset_path : Union[str, list[str]]
        The path(s) to the dataset file(s).
    prefix : str
        The prefix used for the signal column in the dataset. Currently only one signal column is supported.
    genome : Optional[str], default=None
        The genome used for the dataset.
    max_jitter : int, default=128
        The maximum jitter value for cropping the regions.
    dna_length : int, default=1840
        The length of the DNA sequences.
    signal_length : int, default=1000
        The length of the signal sequences.
    min_cov : int, default=50
        The minimum coverage value for filtering regions.
    max_cov : int, default=100000
        The maximum coverage value for filtering regions.
    low_cov_ratio : float, default=0.1
        The low coverage ratio for filtering regions.
        Low coverage regions will be spiked in the dataset with this ratio as negative examples.
    batch_size : int, default=64
        The batch size for the dataloader.
    shuffle_files : bool, default=False
        Whether to shuffle the dataset files.
    read_parquet_kwargs : dict, optional
        Additional keyword arguments for reading the dataset files.

    Attributes
    ----------
    prefix : str
        The prefix used for the signal columns in the dataset.
    batch_size : int
        The batch size for the dataloader.
    max_jitter : int
        The maximum jitter value for cropping the regions.
    dna_length : int
        The length of the DNA sequences.
    signal_length : int
        The length of the signal sequences.
    min_cov : int
        The minimum coverage value for filtering regions.
    max_cov : int
        The maximum coverage value for filtering regions.
    low_cov_ratio : float
        The low coverage ratio for filtering regions.
    signal_columns : list[str]
        The names of the signal columns in the dataset.

    Methods
    -------
    _filter_regions(dataset, concurrency=1)
        Filter the regions in the dataset based on coverage.
    _get_region_cropper(dataset)
        Crop the regions in the dataset.
    _get_reverse_complement_region(dataset)
        Reverse complement the DNA sequences in the dataset.
    get_processed_dataset(chroms, region_bed_path)
        Get the processed dataset with various operators applied.
    get_dataloader(chroms, region_bed_path, n_batches, shuffle_rows=2000, as_torch=True)
        Get the dataloader for the dataset.
    """

    default_config = {
        "dataset_path": "REQUIRED",
        "prefix": "REQUIRED",
        "genome": "REQUIRED",
        "max_jitter": 128,
        "dna_length": 1840,
        "signal_length": 1000,
        "min_cov": 50,
        "max_cov": 100000,
        "low_cov_ratio": 0.1,
        "batch_size": 64,
        "shuffle_files": False,
        "read_parquet_kwargs": None,
    }

    def __init__(
        self,
        dataset_path: Union[str, list[str]],
        prefix: str,
        genome: Optional[str] = None,
        max_jitter: int = 128,
        dna_length: int = 1840,
        signal_length: int = 1000,
        min_cov: int = 50,
        max_cov: int = 100000,
        low_cov_ratio: float = 0.1,
        batch_size: int = 64,
        shuffle_files: bool = False,
        read_parquet_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Initialize a Track1DDataset object.

        Parameters
        ----------
        dataset_path : Union[str, list[str]]
            The path(s) to the dataset file(s).
        prefix : str
            The prefix used for the signal columns in the dataset.
        genome : Optional[str], default=None
            The genome used for the dataset.
        max_jitter : int, default=128
            The maximum jitter value for cropping the regions.
        dna_length : int, default=1840
            The length of the DNA sequences.
        signal_length : int, default=1000
            The length of the signal sequences.
        min_cov : int, default=50
            The minimum coverage value for filtering regions.
        max_cov : int, default=100000
            The maximum coverage value for filtering regions.
        low_cov_ratio : float, default=0.1
            The low coverage ratio for filtering regions.
            Low coverage regions will be spiked in the dataset with this ratio as negative examples.
        batch_size : int, default=64
            The batch size for the dataloader.
        shuffle_files : bool, default=False
            Whether to shuffle the dataset files.
        read_parquet_kwargs : dict, optional
            Additional keyword arguments for reading the dataset files.
        """
        self.prefix = prefix
        self.batch_size = batch_size
        self.max_jitter = max_jitter
        self.dna_length = dna_length
        self.signal_length = signal_length
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio

        super().__init__(
            dataset_path=dataset_path,
            genome=genome,
            shuffle_files=shuffle_files,
            read_parquet_kwargs=read_parquet_kwargs,
        )

        self._cov_filter_key = self.prefix
        self.signal_columns = [self.prefix]
        return

    def _filter_regions(self, dataset, cov_func, concurrency=1):
        """
        Filter the regions in the dataset based on coverage.

        Parameters
        ----------
        dataset : RayDataset
            The input dataset.
        concurrency : int, default=1
            The number of concurrent processes to use.

        Returns
        -------
        RayDataset
            The filtered dataset.
        """
        # filter coverage
        fn = FilterRegions
        fn_constructor_kwargs = {
            "cov_filter_key": self._cov_filter_key,
            "min_cov": self.min_cov,
            "max_cov": self.max_cov,
            "low_cov_ratio": self.low_cov_ratio,
            "cov_func": cov_func,
        }
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=512,
        )
        return dataset

    def _get_region_cropper(self, dataset) -> None:
        """
        Crop the regions in the dataset.

        Parameters
        ----------
        dataset : RayDataset
            The input dataset.

        Returns
        -------
        RayDataset
            The cropped dataset.
        """
        if self.dataset_mode != "train":
            max_jitter = 0
        else:
            max_jitter = self.max_jitter

        key_list = [self.dna_column] + self.signal_columns
        final_length_list = [self.dna_length] + [self.signal_length] * len(
            self.signal_columns
        )

        _cropper = CropLastAxisWithJitter(
            key=key_list,
            final_length=final_length_list,
            max_jitter=max_jitter,
        )

        dataset = dataset.map_batches(_cropper)
        return dataset

    def _get_reverse_complement_region(self, dataset) -> None:
        """
        Reverse complement the DNA sequences in the dataset.

        Parameters
        ----------
        dataset : RayDataset
            The input dataset.

        Returns
        -------
        RayDataset
            The dataset with reverse complemented DNA sequences.
        """
        _rc = ReverseComplement(
            dna_key=self.dna_column,
            signal_key=self.signal_columns,
        )
        dataset = dataset.map_batches(_rc)
        return dataset

    def get_processed_dataset(self, chroms, region_bed_path, cov_func=None) -> None:
        """
        Get the processed dataset with various operators applied.

        Parameters
        ----------
        chroms : list[str]
            The list of chromosomes to include in the dataset.
        region_bed_path : str
            The path to the region BED file.

        Returns
        -------
        RayDataset
            The processed dataset.
        """
        standard_length = max(self.dna_length, self.signal_length) + 500
        region_bed = self.standard_region_length(region_bed_path, standard_length)

        dataset = super()._get_processed_dataset(
            chroms=chroms,
            region_bed=region_bed,
            name_to_pseudobulker={},
            region_action_keys=self.signal_columns,
        )

        # filter regions
        # sum sites within sample, and than take the mean across samples
        if cov_func is None:

            def cov_func(data):
                return data.sum(axis=(-1, -2))

        dataset = self._filter_regions(dataset=dataset, cov_func=cov_func)

        # add dna one hot
        dataset = self._get_dna_one_hot(
            dataset=dataset,
            concurrency=1,
        )

        # crop the regions
        dataset = self._get_region_cropper(dataset)

        if self.dataset_mode == "train":
            # reverse complement the regions
            dataset = self._get_reverse_complement_region(dataset)

        dataset = dataset.drop_columns(["region"])
        return dataset

    def get_dataloader(
        self,
        chroms,
        region_bed_path,
        n_batches,
        shuffle_rows=2000,
        as_torch=True,
    ):
        """
        Get the dataloader for the dataset.

        Parameters
        ----------
        chroms : list[str]
            The list of chromosomes to include in the dataset.
        region_bed_path : str
            The path to the region BED file.
        n_batches : int
            The number of batches to generate.
        shuffle_rows : int, default=2000
            The number of rows to shuffle within each batch.
        as_torch : bool, default=True
            Whether to return the dataloader whoes data will be in torch tensors.

        Returns
        -------
        DataLoader
            The dataloader for the dataset.
        """
        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "chroms": chroms,
            "region_bed_path": region_bed_path,
        }
        data_iter_kwargs = {}

        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            as_torch=as_torch,
            shuffle_rows=shuffle_rows,
            n_batches=n_batches,
            batch_size=self.batch_size,
        )
        return loader
