import pathlib
from typing import Optional

import joblib
import pandas as pd
import ray

from bolero.pp.genome import Genome
from bolero.tl.dataset.sc_transforms import (
    CompressedBytesToTensor,
    FetchRegionOneHot,
    FilterRegions,
    GenerateRegions,
)
from bolero.tl.dataset.transforms import (
    CropLastAxisWithJitter,
    ReverseComplement,
)
from bolero.tl.model.generic.train_helper import validate_config
from bolero.utils import understand_regions


class mCGenomeChunkDataset:
    """Single cell dataset for cell-by-meta-region data."""

    default_config = {
        "dataset_path": "REQUIRED",
        "prefix": "REQUIRED",
        "genome": "REQUIRED",
        "shuffle_files": False,
        "max_jitter": 128,
        "dna_length": 1840,
        "signal_length": 1000,
        "min_cov": 10,
        "max_cov": 100000,
        "low_cov_ratio": 0.1,
        "batch_size": 64,
    }

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Get the default configuration.
        """
        return cls.default_config

    @classmethod
    def create_from_config(cls, config):
        """Create the dataset from a configuration dictionary."""
        config = {k: v for k, v in config.items() if k in cls.default_config}
        validate_config(config, cls.default_config)
        print(f"Create mCGenomeChunkDataset with config: {config}")
        return cls(**config)

    def __init__(
        self,
        dataset_path: str,
        prefix: str,
        genome: Optional[Genome] = None,
        shuffle_files=False,
        read_parquet_kwargs: Optional[dict] = None,
        max_jitter=128,
        dna_length=1840,
        signal_length=1000,
        min_cov=50,
        max_cov=10000,
        low_cov_ratio=0.1,
        batch_size=64,
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
        self.prefix = prefix
        self.batch_size = batch_size
        self.max_jitter = max_jitter
        self.dna_length = dna_length
        self.signal_length = signal_length
        self.min_cov = min_cov
        self.max_cov = max_cov
        self.low_cov_ratio = low_cov_ratio

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

    def _filter_region_length(self, dataset):
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

    def _get_dna_one_hot(self, dataset, concurrency):
        fn = FetchRegionOneHot
        fn_kwargs = {"remote_genome_one_hot": self.genome.remote_genome_one_hot}

        dataset = dataset.map_batches(
            fn=fn, fn_kwargs=fn_kwargs, concurrency=concurrency
        )
        self.dna_column = "dna_one_hot"
        return dataset

    def _generate_regions(
        self,
        dataset,
        region_bed_path,
        action_keys,
        concurrency=1,
    ):
        # generate region from bed file
        fn = GenerateRegions
        fn_constructor_kwargs = {
            "bed": understand_regions(region_bed_path, as_df=True),
            "meta_region_overlap": self.window_size - self.step_size,
            "action_keys": action_keys,
        }

        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )

        # sum sites within sample, and than take the mean across samples
        def _cov_func(data):
            return data.sum(-1).mean(axis=-1)

        # filter coverage
        fn = FilterRegions
        fn_constructor_kwargs = {
            "cov_filter_key": f"{self.prefix}_cov",
            "min_cov": self.min_cov,
            "max_cov": self.max_cov,
            "low_cov_ratio": self.low_cov_ratio,
            "cov_func": _cov_func,
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
        Crop the regions in the working dataset.

        Returns
        -------
        None
        """
        if self._dataset_mode != "train":
            max_jitter = 0
        else:
            max_jitter = self.max_jitter

        key_list = ["dna_one_hot", f"{self.prefix}_mc", f"{self.prefix}_cov"]
        final_length_list = [self.dna_length, self.signal_length, self.signal_length]

        _cropper = CropLastAxisWithJitter(
            key=key_list,
            final_length=final_length_list,
            max_jitter=max_jitter,
        )

        dataset = dataset.map_batches(_cropper)
        return dataset

    def _get_reverse_complement_region(self, dataset) -> None:
        """
        Reverse complement the DNA sequences by 50% probability.

        Returns
        -------
        None
        """
        _rc = ReverseComplement(
            dna_key="dna_one_hot",
            signal_key=[f"{self.prefix}_mc", f"{self.prefix}_cov"],
        )
        dataset = dataset.map_batches(_rc)
        return dataset

    def _get_mc_frac(self, dataset):
        # calculate mC fraction
        def _mc_frac(data_dict):
            mc = data_dict[f"{self.prefix}_mc"]
            cov = data_dict[f"{self.prefix}_cov"]
            data_dict[f"{self.prefix}_mc_frac"] = mc / (cov + 1e-6)
            return data_dict

        dataset = dataset.map_batches(_mc_frac)
        return dataset

    def _get_processed_dataset(self, chroms, region_bed_path) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        compressed_bytes_to_tensor_concurrency = (1, 4)
        fetch_region_one_hot_concurrency = 1

        dataset = self._read_parquet(chroms=chroms)

        # filter meta region length equal to self.window_size
        dataset = self._filter_region_length(dataset=dataset)

        # from compressed bytes to tensor (cell/sample by meta-region matrix) and other information
        dataset = self._compressed_bytes_to_tensor(
            dataset=dataset,
            concurrency=compressed_bytes_to_tensor_concurrency,
        )

        # add dna one hot
        dataset = self._get_dna_one_hot(
            dataset=dataset,
            concurrency=fetch_region_one_hot_concurrency,
        )

        action_keys = ["dna_one_hot", f"{self.prefix}_mc", f"{self.prefix}_cov"]
        # crop specific regions from meta region
        dataset = self._generate_regions(
            dataset=dataset,
            region_bed_path=region_bed_path,
            action_keys=action_keys,
            concurrency=1,
        )

        # crop the regions
        dataset = self._get_region_cropper(dataset)

        if self._dataset_mode == "train":
            # reverse complement the regions
            dataset = self._get_reverse_complement_region(dataset)

        dataset = self._get_mc_frac(dataset)

        dataset = dataset.drop_columns(["region"])
        return dataset

    def get_dataloader(
        self,
        chroms,
        region_bed_path,
        n_batches,
    ):
        """Dataloader for the dataset."""
        work_ds = self._get_processed_dataset(
            chroms,
            region_bed_path,
        )

        if n_batches is not None:
            n_rows = (n_batches + 1) * self.batch_size
            work_ds = work_ds.limit(n_rows)

        _shuffle_rows = 5000
        _kwargs = {
            "prefetch_batches": 3,
            "local_shuffle_buffer_size": (
                _shuffle_rows if self._dataset_mode == "train" else None
            ),
            "drop_last": True,
            "batch_size": self.batch_size,
        }
        print("data loader kwargs", _kwargs)
        return work_ds.iter_torch_batches(**_kwargs)

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
