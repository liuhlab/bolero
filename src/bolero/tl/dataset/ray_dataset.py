import pathlib
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

import joblib
import pandas as pd
import ray

from bolero import Genome
from bolero.tl.dataset.sc_transforms import (
    CompressedBytesToTensor,
    GeneratePseudobulk,
    GenerateRegions,
)
from bolero.tl.dataset.transforms import FetchRegionOneHot
from bolero.tl.generic.dataset import GenericDataset
from bolero.utils import understand_regions

DNA_NAME = "dna_one_hot"

T = TypeVar("T")


class _IterableFromIterator(Iterable[T]):
    def __init__(self, iterator_gen: Callable[[], Iterator[T]]):
        """Constructs an Iterable from an iterator generator.

        Args:
            iterator_gen: A function that returns an iterator each time it
                is called. For example, this can be a generator function.
        """
        self.iterator_gen = iterator_gen

    def __iter__(self):
        return self.iterator_gen()


class RayGenomeChunkDataset(GenericDataset):
    """Single cell dataset for cell-by-meta-region data."""

    default_config = {
        "dataset_path": "REQUIRED",
        "genome": "REQUIRED",
        "shuffle_files": False,
        "read_parquet_kwargs": None,
    }

    def __init__(
        self,
        dataset_path: str,
        genome: Optional[Genome] = None,
        shuffle_files=False,
        read_parquet_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Initialize the RaySingleCellDataset.

        Parameters
        ----------
        dataset_path : str
            The path to the dataset.
        use_prefixs : Optional[List[str]], optional
            The list of prefixes to use, by default None.
        chroms : Optional[Union[str, List[str]]], optional
            The list of chromosomes to use, by default None.
        shuffle_files : bool, optional
            Whether to shuffle the files, by default False.
        genome : str, optional
            The genome, by default None, which will be read from genome.flag.
        read_parquet_kwargs : Optional[dict], optional
            The read_parquet kwargs passed to ray.data.read_parquet, by default None.

        Returns
        -------
        None
        """
        self.dataset_path = dataset_path

        if not shuffle_files:
            print("File shuffle is disabled!!!")
        _kwargs = {
            "shuffle": "files" if shuffle_files else None,
        }
        if read_parquet_kwargs is not None:
            _kwargs.update(read_parquet_kwargs)
        self.read_parquet_kwargs = _kwargs

        # get barcode order
        self.barcode_order: dict[pd.Index] = joblib.load(
            f"{dataset_path}/row_names.joblib"
        )

        # get genome and other metadata
        config = joblib.load(f"{dataset_path}/config.joblib")

        if genome is None:
            genome = config["genome"]
        if isinstance(genome, str):
            self.genome = Genome(genome)
        else:
            self.genome = genome
        # trigger one hot loading
        _ = self.genome.genome_one_hot

        self.window_size = config["window_size"]
        self.step_size = config["step_size"]
        self.num_rows_per_file = config["num_rows_per_file"]

        # slot for later processor
        self.signal_columns = set()
        self.dna_column = DNA_NAME

    def _get_chroms_dir(self, chroms):
        if chroms is None:
            chrom_dirs = [str(p) for p in pathlib.Path(self.dataset_path).glob("chr*")]
        else:
            if isinstance(chroms, str):
                chroms = [chroms]
            chrom_dirs = [f"{self.dataset_path}/{chrom}" for chrom in chroms]

            # make sure all chrom_dir exists
            chrom_dirs = [
                chrom_dir
                for chrom_dir in chrom_dirs
                if pathlib.Path(chrom_dir).exists()
            ]
            assert (
                len(chrom_dirs) > 0
            ), f"None of the chroms {chroms} exists in {self.dataset_path}"
        return chrom_dirs

    def _read_parquet(self, chroms):
        _dataset = ray.data.read_parquet(
            self._get_chroms_dir(chroms),
            file_extensions=["parquet"],
            **self.read_parquet_kwargs,
        )
        return _dataset

    def _filter_meta_region_length(self, dataset):
        standard_region_length = self.window_size

        def region_length_filter(row):
            region = row["region"]
            coords = region.split(":")[1].split("-")
            length = int(coords[1]) - int(coords[0])
            return length == standard_region_length

        dataset = dataset.filter(region_length_filter)
        return dataset

    def _compressed_bytes_to_tensor(self, dataset, concurrency):
        fn = CompressedBytesToTensor
        dataset = dataset.map(fn=fn, concurrency=concurrency)
        # mast use the class, instead of class instance when trying to map an actor to a dataset
        # dataset = dataset.map(fn=CompressedBytesToTensor(), concurrency=concurrency)
        return dataset

    def _generate_pseudobulk(
        self,
        dataset,
        name_to_pseudobulker,
        bypass_keys=None,
        n_pseudobulks=10,
        return_rows=False,
        inplace=False,
        concurrency=1,
    ):
        bypass_keys = [] if bypass_keys is None else bypass_keys

        fn = GeneratePseudobulk
        fn_constructor_kwargs = {
            "n_pseudobulks": n_pseudobulks,
            "return_rows": return_rows,
            "inplace": inplace,
            "bypass_keys": bypass_keys,
        }
        fn_constructor_kwargs.update(name_to_pseudobulker)

        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )
        return dataset

    def _generate_regions(
        self,
        dataset,
        bed,
        action_keys,
        concurrency,
    ):
        # generate region from bed file
        fn = GenerateRegions
        fn_constructor_kwargs = {
            "bed": understand_regions(bed, as_df=True),
            "meta_region_overlap": self.window_size - self.step_size,
            "action_keys": action_keys,
        }
        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )
        return dataset

    def _get_dna_one_hot(self, dataset, concurrency):
        fn = FetchRegionOneHot
        fn_kwargs = {"remote_genome_one_hot": self.genome.remote_genome_one_hot}

        dataset = dataset.map_batches(
            fn=fn, fn_kwargs=fn_kwargs, concurrency=concurrency
        )
        self.dna_column = DNA_NAME
        return dataset

    def _get_processed_dataset(
        self,
        chroms,
        region_bed,
        name_to_pseudobulker,
        region_action_keys=None,
        **pseudobulk_kwargs,
    ) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        compressed_bytes_to_tensor_concurrency = (1, 4)
        generate_pseudobulk_concurrency = (1, 16)
        generate_regions_concurrency = (1, 4)

        dataset = self._read_parquet(chroms=chroms)

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

        dataset = self._generate_regions(
            dataset=dataset,
            bed=region_bed,
            action_keys=region_action_keys,
            concurrency=generate_regions_concurrency,
        )
        return dataset

    def get_processed_dataset(
        self,
        *args,
        **kwargs,
    ):
        """Implement this method to return the processed dataset with more specific operations."""
        # use the default implementation to get basic processed dataset
        # self._get_processed_dataset(*args, **kwargs)

        # then add more specific operations here
        raise NotImplementedError

    def _get_dataloader_with_wrapper(
        self,
        dataset_kwargs: dict,
        data_iter_kwargs: dict,
        as_torch=True,
        shuffle_rows=1000,
        n_batches=None,
        batch_size=64,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader generator.

        The dataset will be init only when entering the __iter__ method.
        """

        # this is adapted from the ray.data.iterator.DataIterator.iter_batches
        # https://github.com/ray-project/ray/blob/master/python/ray/data/iterator.py#L106
        def _create_iterator():
            print(f"Get dataloader with {self.dataset_mode} mode")

            work_ds = self.get_processed_dataset(**dataset_kwargs)

            if n_batches is not None:
                n_rows = (n_batches + 1) * batch_size
                work_ds = work_ds.limit(n_rows)

            _kwargs = {
                "prefetch_batches": 3,
                "local_shuffle_buffer_size": (
                    shuffle_rows if self._dataset_mode == "train" else None
                ),
                "drop_last": True,
                "batch_size": batch_size,
            }
            _kwargs.update(data_iter_kwargs)
            print("Data loader kwargs", _kwargs)

            if as_torch:
                loader = work_ds.iter_torch_batches(**_kwargs)
            else:
                loader = work_ds.iter_batches(**_kwargs)

            yield from loader
            # for batch in loader:
            #     batch['dna'] = self.genome.get_regions_one_hot()
            #     yield batch

        # the dataset and dataloader are created lazily, until __iter__ is called
        return _IterableFromIterator(_create_iterator)

    def standard_region_length(self, bed, standard_length):
        """Standardize the region length."""
        standard_bed = self.genome.standard_region_length(
            bed,
            length=standard_length,
            boarder_strategy="drop",
            remove_blacklist=True,
            as_df=True,
            keep_original=False,
        )
        return standard_bed


class RayRegionDataset(GenericDataset):
    """
    A dataset class for working with genomic regions using Ray.

    Args:
        bed (pd.DataFrame or pr.PyRanges or str): The genomic regions in BED format.
        genome (Genome or str): The genome reference or its name.
        standard_length (int): The standard length of the regions.
        **kwargs: Additional keyword arguments for ray.data.from_pandas.

    Attributes
    ----------
        bed (pd.DataFrame): The standardized genomic regions.
        genome (Genome): The genome reference.
        dataset (ray.data.Dataset): The Ray dataset containing the genomic regions.
        _working_dataset (ray.data.Dataset): The working dataset for preprocessing.

    Methods
    -------
        get_dataloader: Get a data loader for iterating over batches of the dataset.
    """

    default_config = {
        "bed": "REQUIRED",
        "genome": "REQUIRED",
        "standard_length": "REQUIRED",
        "dna": True,
    }

    def __init__(
        self,
        bed,
        genome,
        standard_length,
        dna=True,
        _block_size=20,
        _max_blocks=200,
    ):
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome

        standard_bed = self.genome.standard_region_length(
            bed,
            length=standard_length,
            boarder_strategy="drop",
            remove_blacklist=True,
            as_df=True,
            keep_original=True,
        )
        # ray data don't understand categorical dtype in pandas
        standard_bed["Chromosome"] = standard_bed["Chromosome"].astype(str)
        standard_bed.rename(columns={"Name": "region"}, inplace=True)
        self.bed = standard_bed

        self.dna = dna
        self._block_size = _block_size
        self._max_blocks = _max_blocks
        self.n_blocks = min(len(self.bed) // self._block_size, self._max_blocks)

    def _get_dna_one_hot(self, dataset, concurrency=1):
        fn = FetchRegionOneHot
        fn_kwargs = {"remote_genome_one_hot": self.genome.remote_genome_one_hot}

        dataset = dataset.map_batches(
            fn=fn, fn_kwargs=fn_kwargs, concurrency=concurrency
        )
        self.dna_column = DNA_NAME
        return dataset

    def _select_columns(self, dataset):
        keep_cols = ["region", "Original_Name"]
        if self.dna:
            keep_cols.append(DNA_NAME)
        dataset = dataset.select_columns(keep_cols)
        return dataset

    def get_processed_dataset(self):
        """Get the processed dataset."""
        dataset = (
            ray.data.from_pandas(self.bed).repartition(self.n_blocks).materialize()
        )
        if self.dna:
            dataset = self._get_dna_one_hot(dataset)
        dataset = self._select_columns(dataset)
        return dataset

    def get_dataloader(self, batch_size: int = 64, **kwargs):
        """
        Get a data loader for iterating over batches of the dataset.

        Args:
            batch_size (int): The batch size.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            DataLoader: The data loader.
        """
        dataset = self.get_processed_dataset()
        loader = dataset.iter_batches(batch_size=batch_size, **kwargs)
        return loader
