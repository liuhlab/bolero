from collections import OrderedDict
from typing import Any, Iterable

from bolero.tl.dataset.ray_dataset import (
    RayGenomeChunkDataset,
)
from bolero.tl.dataset.sc_transforms import FilterRegions
from bolero.tl.dataset.transforms import (
    FetchRegionOneHot,
    ReverseComplement,
)
from bolero.utils import get_global_coords, understand_regions

DNA_NAME = "dna_one_hot"


class BorzoiDataset(RayGenomeChunkDataset):
    """Singel cell pseudobulk dataset for Borzoi model."""

    default_config = {
        "dataset_path": "REQUIRED",
        "genome": "REQUIRED",
        "batch_size": 2,
        "dna_window": 524288,
        "reverse_complement": True,
        "max_jitter": 3,
        "n_pseudobulks": 100,
        "shuffle_files": True,
        "read_parquet_kwargs": None,
    }

    def __init__(
        self,
        dataset_path: str,
        genome,
        batch_size: int = 2,
        dna_window: int = 524288,
        reverse_complement: bool = True,
        max_jitter: int = 3,
        n_pseudobulks: int = 100,
        cov_filter_name: str = None,
        min_cov: int = 10,
        max_cov: int = 100000,
        shuffle_files=False,
        read_parquet_kwargs=None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            genome=genome,
            shuffle_files=shuffle_files,
            read_parquet_kwargs=read_parquet_kwargs,
        )
        self.batch_size = batch_size

        # region properties
        self.dna_window = dna_window
        self.signal_window = dna_window
        self.max_jitter = max_jitter
        self.min_counts = min_cov
        self.max_counts = max_cov
        self.reverse_complement = reverse_complement
        self.n_pseudobulks = n_pseudobulks
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.cov_filter_name = cov_filter_name

        self.name_to_pseudobulker = OrderedDict()
        return

    def __repr__(self) -> str:
        _str = (
            f"{self.__name__}\n"
            f"Dataset directory: {self.dataset_path}\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
        )
        return _str

    def _get_dna_one_hot(self, dataset, concurrency):
        fn = FetchRegionOneHot
        fn_constructor_kwargs = {
            "random_shift": self.max_jitter,
            "dtype": "bool",
        }
        fn_kwargs = {"remote_genome_one_hot": self.genome.remote_genome_one_hot}

        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            fn_kwargs=fn_kwargs,
            concurrency=concurrency,
        )
        self.dna_column = DNA_NAME
        return dataset

    def _get_reverse_complement_region(self, dataset) -> None:
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
        dataset = dataset.map_batches(_rc)
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

    def _process_region_columns(self, dataset, keep_regions=False):
        """
        Keep the regions by converting them to global coordinates OR remove the region columns.
        """
        if keep_regions:
            chrom_offsets = self.genome.chrom_offsets.copy()

            def _region_to_global_coords(batch):
                region_df = understand_regions(batch.pop("region"))
                global_coords = get_global_coords(
                    chrom_offsets=chrom_offsets,
                    region_bed_df=region_df,
                )
                batch["region"] = global_coords
                return batch

            dataset = dataset.map_batches(_region_to_global_coords)
        else:
            dataset = dataset.drop_columns(["region"])
        return dataset

    def get_processed_dataset(
        self,
        chroms: list[str],
        region_bed_path: str,
        return_cells: bool = False,
        return_regions: bool = True,
        concurrency: int = 16,
    ) -> None:
        """
        Process the dataset and return the processed dataset.

        Parameters
        ----------
        - chroms (list): List of chromosomes to include in the dataset.
        - region_bed_path (str): Path to the BED file containing the regions.
        - return_cells (bool): Whether to return the cells in the dataset. Default is False.
        - return_regions (bool): Whether to return the regions in the dataset. Default is False.

        Returns
        -------
        - work_ds (Dataset): The processed dataset.

        """
        standard_length = self.dna_window
        region_bed = self.standard_region_length(region_bed_path, standard_length)

        work_ds = super()._get_processed_dataset(
            chroms=chroms,
            region_bed=region_bed,
            name_to_pseudobulker=self.name_to_pseudobulker,
            bypass_keys=[self.bias_column],
            n_pseudobulks=self.n_pseudobulks,
            return_rows=return_cells,
            inplace=False,
            region_action_keys=[self.bias_column],
            concurrency=concurrency,
        )

        # add dna one hot
        work_ds = self._get_dna_one_hot(
            dataset=work_ds,
            concurrency=1,
        )

        if self.reverse_complement and self._dataset_mode == "train":
            work_ds = self._get_reverse_complement_region(work_ds)

        # remove region column OR turn it into global coordinates (str to numbers)
        work_ds = self._process_region_columns(
            dataset=work_ds, keep_regions=return_regions
        )
        return work_ds

    def get_dataloader(
        self,
        chroms,
        region_bed_path,
        as_torch=True,
        return_regions=True,
        return_cells=False,
        n_batches=None,
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
        shuffle_rows = 50

        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "chroms": chroms,
            "region_bed_path": region_bed_path,
            "return_cells": return_cells,
            "return_regions": return_regions,
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
