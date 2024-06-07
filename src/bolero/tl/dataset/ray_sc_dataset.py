# TODO: scRayDataset inherits from RayDataset, and then in scprinter, scPrinterscDataset inherits from scPrinterDataset.
# Once the sc dataset processed pseudobulk and provides region and pseudobulk data dict, the remaining preprocess step should be the same as bulk train model.
import pathlib
from copy import deepcopy
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import ray

from bolero import Genome
from bolero.tl.dataset.sc_transforms import (
    CompressedBytesToTensor,
    FetchRegionOneHot,
    GeneratePseudobulk,
    GenerateRegions,
    scMetaRegionToBulkRegion,
)
from bolero.utils import understand_regions


class RayGenomeChunkDataset:
    """Single cell dataset for cell-by-meta-region data."""

    def __init__(
        self,
        dataset_path: str,
        use_prefixs: Optional[list[str]] = None,
        chroms=None,
        genome: str = None,
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
        if chroms is None:
            chrom_dirs = [str(p) for p in pathlib.Path(dataset_path).glob("chr*")]
        else:
            if isinstance(chroms, str):
                chroms = [chroms]
            chrom_dirs = [f"{dataset_path}/{chrom}" for chrom in chroms]

            # make sure all chrom_dir exists
            chrom_dirs = [
                chrom_dir
                for chrom_dir in chrom_dirs
                if pathlib.Path(chrom_dir).exists()
            ]
            assert (
                len(chrom_dirs) > 0
            ), f"None of the chroms {chroms} exists in {dataset_path}"

        if not shuffle_files:
            print("File shuffle is disabled!!!")

        _kwargs = {
            "shuffle": "files" if shuffle_files else None,
        }
        if read_parquet_kwargs is not None:
            _kwargs.update(read_parquet_kwargs)
        # create ray.data.Dataset
        self._dataset = ray.data.read_parquet(
            chrom_dirs,
            file_extensions=["parquet"],
            **_kwargs,
        )

        _schema = self._dataset.schema()
        self.schema: dict = dict(zip(_schema.names, _schema.types))

        # get prefix
        self.prefixs = list({key.split(":")[0] for key in self.schema.keys()})
        if use_prefixs is not None:
            self.prefixs = [prefix for prefix in self.prefixs if prefix in use_prefixs]

        # get barcode order for each prefix
        self.barcode_order = {
            name: pd.Index(cells)
            for name, cells in np.load(f"{dataset_path}/barcodes.npz").items()
            if name in self.prefixs
        }

        # get genome
        if genome is None:
            with open(f"{dataset_path}/genome.flag") as f:
                genome = f.read().strip()
        if isinstance(genome, str):
            self.genome = Genome(genome)
        else:
            self.genome = genome
        # trigger one hot loading
        _ = self.genome.genome_one_hot

        self._dataset_mode = None

    def __repr__(self) -> str:
        return self._dataset.__repr__()

    def _dataset_preprocess(
        self,
        sample_regions: int,
        n_pseudobulks: int,
        min_cov: int,
        max_cov: int,
        low_cov_ratio: float,
        return_cells: bool = False,
        **kwargs,
    ) -> None:
        """
        Preprocess the dataset.

        Parameters
        ----------
        sample_regions : int
            The number of sample regions.
        n_pseudobulks : int
            The number of pseudobulks.
        min_cov : int
            The minimum coverage.
        max_cov : int
            The maximum coverage.
        low_cov_ratio : float
            The low coverage ratio.
        return_cells : bool, optional
            Whether to return cell ids of each pseudobulk, by default False.

        Returns
        -------
        None
        """
        self._pseudobulk_and_extract_regions(
            sample_regions=sample_regions,
            n_pseudobulks=n_pseudobulks,
            min_cov=min_cov,
            max_cov=max_cov,
            low_cov_ratio=low_cov_ratio,
            return_cells=return_cells,
            **kwargs,
        )
        return

    def _pseudobulk_and_extract_regions(
        self,
        sample_regions: int,
        n_pseudobulks: int,
        min_cov: int,
        max_cov: int,
        low_cov_ratio: float,
        num_cpus: int = 1,
        memory: float = "auto",
        return_cells: bool = False,
        **psuedobulker_kwargs,
    ) -> None:
        """
        Perform pseudobulking and extract regions.

        Parameters
        ----------
        sample_regions : int
            The number of sample regions.
        n_pseudobulks : int
            The number of pseudobulks.
        min_cov : int
            The minimum coverage.
        max_cov : int
            The maximum coverage.
        low_cov_ratio : float
            The low coverage ratio.
        num_cpus : int, optional
            The number of CPUs to use in each ray task, by default 1.
        memory : float, optional
            The memory to use in each ray task, by default "auto".
        return_cells : bool, optional
            Whether to return cell ids of each pseudobulk, by default False.

        Returns
        -------
        None
        """
        # TODO: determine flat_map memory dynamically based on the size of the dataset
        if memory == "auto":
            memory = 3 * 1024**3  # Gb to bytes

        # merge cell into pseudobulk and
        # split large meta region (storage) into smaller final regions (data consumption)
        kwargs = psuedobulker_kwargs
        other_kwargs = {
            "barcode_order": deepcopy(self.barcode_order),
            "prefixs": self.prefixs,
            "sample_regions": sample_regions,
            "min_cov": min_cov,
            "max_cov": max_cov,
            "low_cov_ratio": low_cov_ratio,
            "n_pseudobulks": n_pseudobulks,
            "return_cells": return_cells,
        }
        kwargs.update(other_kwargs)
        self._working_dataset = self._working_dataset.flat_map(
            fn=scMetaRegionToBulkRegion,
            fn_constructor_kwargs=kwargs,
            num_cpus=num_cpus,
            memory=memory,
            concurrency=(2, 15),
        )
        # after flat_map processor, each row in working_dataset is a dict with keys:
        # ["bulk_embedding", "bulk_data", "region"]

    def train(self) -> None:
        """
        Set the dataset mode to "train".

        Returns
        -------
        None
        """
        self._dataset_mode = "train"
        return

    def eval(self) -> None:
        """
        Set the dataset mode to "eval".

        Returns
        -------
        None
        """
        self._dataset_mode = "eval"
        return


class NewRayGenomeChunkDataset:
    """Single cell dataset for cell-by-meta-region data."""

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

        self._dataset_mode = None

        # slot for later processor
        self.signal_columns = set()
        self.dna_column = None

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

    def _compressed_bytes_to_tensor(self, dataset, concurrency):
        fn = CompressedBytesToTensor
        dataset = dataset.map(fn=fn, concurrency=concurrency)
        return dataset

    def _generate_pseudobulk(
        self,
        dataset,
        pseudobulker_and_names,
        bypass_keys,
        n_pseudobulks,
        return_rows,
        inplace,
        concurrency,
    ):
        fn = GeneratePseudobulk
        fn_constructor_kwargs = {
            "pseudobulker_and_names": pseudobulker_and_names,
            "n_pseudobulks": n_pseudobulks,
            "return_rows": return_rows,
            "inplace": inplace,
            "bypass_keys": bypass_keys,
        }

        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )
        return dataset

    def _generate_regions(
        self,
        dataset,
        bed_path,
        action_keys,
        cov_filter_key,
        min_cov,
        max_cov,
        low_cov_ratio,
        concurrency,
    ):
        fn = GenerateRegions
        fn_constructor_kwargs = {
            "bed": understand_regions(bed_path, as_df=True),
            "meta_region_overlap": self.window_size - self.step_size,
            "action_keys": action_keys,
            "cov_filter_key": cov_filter_key,
            "min_cov": min_cov,
            "max_cov": max_cov,
            "low_cov_ratio": low_cov_ratio,
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
        self.dna_column = "dna_one_hot"
        return dataset

    def _get_processed_dataset(
        self,
        chroms,
        region_bed_path,
        pseudobulker_and_names,
        bypass_keys=None,
        n_pseudobulks=10,
        cov_filter_name=None,
        min_cov=10,
        max_cov=100000,
        low_cov_ratio=0.1,
        return_rows=False,
        inplace=False,
        region_action_keys=None,
    ) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        compressed_bytes_to_tensor_concurrency = (2, 5)
        generate_pseudobulk_concurrency = (2, 10)
        generate_regions_concurrency = (2, 10)
        fetch_region_one_hot_concurrency = (2, 5)

        dataset = self._read_parquet(chroms=chroms)

        # from compressed bytes to tensor (cell/sample by meta-region matrix) and other information
        dataset = self._compressed_bytes_to_tensor(
            dataset=dataset,
            concurrency=compressed_bytes_to_tensor_concurrency,
        )

        # generate pseudobulk
        if len(pseudobulker_and_names) > 0:
            dataset = self._generate_pseudobulk(
                dataset=dataset,
                pseudobulker_and_names=pseudobulker_and_names,
                bypass_keys=[] if bypass_keys is None else bypass_keys,
                n_pseudobulks=n_pseudobulks,
                return_rows=return_rows,
                inplace=inplace,
                concurrency=generate_pseudobulk_concurrency,
            )

        # generate regions
        _action_keys = [f"{name}:bulk_data" for (_, name) in pseudobulker_and_names]
        if region_action_keys is not None:
            if isinstance(region_action_keys, str):
                region_action_keys = [region_action_keys]
            _action_keys.extend(region_action_keys)
        _action_keys = list(set(_action_keys))
        if cov_filter_name is not None:
            cov_filter_key = f"{cov_filter_name}:bulk_data"
            assert (
                cov_filter_key in _action_keys
            ), f"cov_filter_key {cov_filter_key} not in {_action_keys}"
        else:
            cov_filter_key = None
        dataset = self._generate_regions(
            dataset=dataset,
            bed_path=region_bed_path,
            action_keys=_action_keys,
            cov_filter_key=cov_filter_key,
            min_cov=min_cov,
            max_cov=max_cov,
            low_cov_ratio=low_cov_ratio,
            concurrency=generate_regions_concurrency,
        )
        self.signal_columns = list(set(_action_keys))

        # add dna one hot
        dataset = self._get_dna_one_hot(
            dataset=dataset,
            concurrency=fetch_region_one_hot_concurrency,
        )
        return dataset

    def train(self) -> None:
        """
        Set the dataset mode to "train".

        Returns
        -------
        None
        """
        self._dataset_mode = "train"
        return

    def eval(self) -> None:
        """
        Set the dataset mode to "eval".

        Returns
        -------
        None
        """
        self._dataset_mode = "eval"
        return
