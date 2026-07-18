import pathlib

import joblib
import pandas as pd
import pyarrow.parquet as pq
import ray

from bolero.pp.genome import Genome
from bolero.pp.genome_chunk_dataset import (
    ChromSparseDataset,
    GenomeALLCDataset,
    GenomeBigWigDataset,
    SnapAnnDataDataset,
)


def _path_exists(path):
    return pathlib.Path(path).exists()


class GenomeChunkDatasetGenerator:
    """
    A generator class for creating genome-chunk ray dataset for single-cell or bulk data.

    Parameters
    ----------
    genome : Union[str, Genome]
        The genome associated with the dataset.
    """

    def __init__(
        self,
        output_dir: str,
        genome: str | Genome,
        window_size: int = 100000,
        step_size: int = 90000,
        num_rows_per_file: int = 50,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir).resolve().absolute()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome

        self.window_size = window_size
        self.step_size = step_size
        assert (
            self.window_size >= self.step_size
        ), "Window size must be greater than step size."
        self.genome_chunk_df = self.genome.make_windows(
            window_size=self.window_size, step=self.step_size, as_df=True
        )
        for chrom in self.genome.chromosomes:
            chrom_dir = self.output_dir / chrom
            chrom_dir.mkdir(exist_ok=True)

        self.num_rows_per_file = num_rows_per_file

        self.uniform_dataset_dict = {
            # prefix: {ds_class, ds_kwargs, remote_kwargs}
        }

    def add_bigwig(self, prefix, name, path, sparse=True, compress_level=5, **kwargs):
        """
        Add BigWig files. BigWig will be aggregated based on the prefix.

        Parameters
        ----------
        kwargs : Dict[str, str]
            The dataset name and the path to the BigWig file.
        """
        assert _path_exists(path)

        bw_class = GenomeBigWigDataset
        if prefix in self.uniform_dataset_dict:
            cur_prefix_dict = self.uniform_dataset_dict[prefix]

            assert (
                cur_prefix_dict["ds_class"] == bw_class
            ), f"Dataset with name {prefix} should be bigwig."
            assert (
                name not in cur_prefix_dict["ds_kwargs"]
            ), f"BigWig with name {name} already exists for dataset {prefix}."
            assert (
                cur_prefix_dict["ds_kwargs"]["sparse"] == sparse
            ), f"Sparse flag must be the same for dataset {prefix}."
            assert (
                cur_prefix_dict["ds_kwargs"]["compress_level"] == compress_level
            ), f"Compress level must be the same for dataset {prefix}."

            self.uniform_dataset_dict[prefix]["ds_kwargs"][name] = str(path)
            self.uniform_dataset_dict[prefix]["remote_kwargs"]["memory"] += (
                0.5 * 1024**3
            )
        else:
            self.uniform_dataset_dict[prefix] = {
                "ds_class": bw_class,
                "ds_kwargs": {
                    name: str(path),
                    "prefix": prefix,
                    "sparse": sparse,
                    "compress_level": compress_level,
                    **kwargs,
                },
                "remote_kwargs": {
                    "memory": 1 * 1024**3,
                    "resources": {"bolero_dataset_gen": 10},
                },
            }
        return

    def add_snap_adata(self, prefix, path, barcode_whitelist=None, **ds_kwargs):
        """
        Add SnapATAC AnnData dataset that contains insersion sites as a sparse matrix.

        Parameters
        ----------
        prefix : str
            The dataset name.
        path : str
            The path to the AnnData file.
        barcode_whitelist : List[str], optional
            The list of barcodes to include in the dataset, by default None.
        """
        assert _path_exists(path)
        if prefix in self.uniform_dataset_dict:
            raise ValueError(f"Dataset with name {prefix} already exists.")
        self.uniform_dataset_dict[prefix] = {
            "ds_class": SnapAnnDataDataset,
            "ds_kwargs": {
                "name": prefix,
                "path": path,
                "barcode_whitelist": barcode_whitelist,
                **ds_kwargs,
            },
            "remote_kwargs": {
                "memory": 15 * 1024**3,
                "resources": {"bolero_dataset_gen": 10},
            },
        }
        return

    def add_chrom_sparse(self, prefix, dataset_dir, **ds_kwargs):
        """
        Add chromatin sparse matrix. Chromatin sparse matrix will be aggregated based on the prefix.

        Parameters
        ----------
        prefix : str
            The dataset name.
        dataset_dir : str
            The path to the chromatin sparse matrix dataset_dir.

        """
        assert _path_exists(dataset_dir)
        if prefix in self.uniform_dataset_dict:
            raise ValueError(f"Dataset with name {prefix} already exists.")
        self.uniform_dataset_dict[prefix] = {
            "ds_class": ChromSparseDataset,
            "ds_kwargs": {
                "name": prefix,
                "dataset_dir": dataset_dir,
                **ds_kwargs,
            },
            "remote_kwargs": {
                "memory": 10 * 1024**3,
                "resources": {"bolero_dataset_gen": 10},
            },
        }
        return

    def add_allc(self, prefix, name, path, sparse=True, compress_level=5, **ds_kwargs):
        """
        Add ALLC files. ALLC will be aggregated based on the prefix.

        Parameters
        ----------
        prefix : str
            The dataset name.
        name : str
            The name of the ALLC file.
        path : str
            The path to the ALLC file.
        sparse : bool, optional
            Whether the sample-by-pos matrix is sparse, by default True.

        """
        assert _path_exists(path)
        ds_class = GenomeALLCDataset
        if prefix in self.uniform_dataset_dict:
            cur_prefix_dict = self.uniform_dataset_dict[prefix]

            assert (
                cur_prefix_dict["ds_class"] == ds_class
            ), f"Dataset with name {prefix} should be bigwig."
            assert (
                name not in cur_prefix_dict["ds_kwargs"]
            ), f"BigWig with name {name} already exists for dataset {prefix}."
            assert (
                cur_prefix_dict["ds_kwargs"]["sparse"] == sparse
            ), f"Sparse flag must be the same for dataset {prefix}."
            assert (
                cur_prefix_dict["ds_kwargs"]["compress_level"] == compress_level
            ), f"Compress level must be the same for dataset {prefix}."

            self.uniform_dataset_dict[prefix]["ds_kwargs"][name] = str(path)
            self.uniform_dataset_dict[prefix]["remote_kwargs"]["memory"] += (
                0.5 * 1024**3
            )
        else:
            self.uniform_dataset_dict[prefix] = {
                "ds_class": ds_class,
                "ds_kwargs": {
                    name: str(path),
                    "prefix": prefix,
                    "sparse": sparse,
                    "compress_level": compress_level,
                    **ds_kwargs,
                },
                "remote_kwargs": {
                    "memory": 1 * 1024**3,
                    "resources": {"bolero_dataset_gen": 10},
                },
            }
        return

    def _process_each_prefix(self):
        for prefix, info_dict in self.uniform_dataset_dict.items():
            # check success flag
            success_flag_path = self.output_dir / f"{prefix}.success.flag"
            if success_flag_path.exists():
                continue
            _ds_class = info_dict["ds_class"]
            _ds_kwargs = info_dict["ds_kwargs"]
            _remote_kwargs = info_dict["remote_kwargs"]

            @ray.remote(**_remote_kwargs)
            def _process_worker(
                prefix, ds_class, ds_kwargs, output_dir, chrom, chrom_regions_df
            ):
                ds = ds_class(**ds_kwargs)
                list_of_dicts = ds.get_regions_data(chrom_regions_df)

                joblib.dump(
                    list_of_dicts,
                    output_dir / chrom / f"{prefix}.list_of_dicts.joblib",
                )
                return

            for chrom, chrom_regions_df in self.genome_chunk_df.groupby("Chromosome"):
                print(f"Processing {prefix} {chrom}")
                prefix_tasks = []
                task = _process_worker.remote(
                    prefix=prefix,
                    ds_class=_ds_class,
                    ds_kwargs=_ds_kwargs,
                    output_dir=self.output_dir,
                    chrom=chrom,
                    chrom_regions_df=chrom_regions_df,
                )
                prefix_tasks.append(task)
                ray.get(prefix_tasks)

            # dump row names
            ds = _ds_class(**_ds_kwargs)
            row_names = ds.get_row_names()
            joblib.dump(row_names, self.output_dir / f"{prefix}.row_names.joblib")

            # create a success flag
            pathlib.Path(success_flag_path).touch()
        return

    def _prepare_single_chrom(self, chrom: str) -> None:
        """
        Prepare the dataset for a single chromosome.

        Parameters
        ----------
        output_dir : str
            The output directory to save the prepared dataset.
        chrom : str
            The chromosome to prepare the dataset for.
        """
        chrom_dir = self.output_dir / chrom
        flag_path = chrom_dir / "success.flag"
        if flag_path.exists():
            return

        print(f"Creating dataset for chromosome {chrom}.")
        for i, prefix in enumerate(self.uniform_dataset_dict.keys()):
            _data = joblib.load(chrom_dir / f"{prefix}.list_of_dicts.joblib")
            if i == 0:
                list_of_dict = _data
            else:
                for idx, d in enumerate(_data):
                    list_of_dict[idx].update(d)

        # create ray dataset
        ray_dataset = ray.data.from_items(list_of_dict)
        ray_dataset.write_parquet(chrom_dir, num_rows_per_file=self.num_rows_per_file)

        # create success flag
        flag_path.touch()

        # clean up
        for prefix in self.uniform_dataset_dict.keys():
            pathlib.Path(f"{chrom_dir}/{prefix}.list_of_dicts.joblib").unlink()
        return

    def _dump_row_names(self):
        row_names_path = self.output_dir / "row_names.joblib"
        if row_names_path.exists():
            return

        row_names = {
            prefix: joblib.load(self.output_dir / f"{prefix}.row_names.joblib")
            for prefix in self.uniform_dataset_dict.keys()
        }
        joblib.dump(row_names, row_names_path)

        # clean up
        for prefix in self.uniform_dataset_dict.keys():
            pathlib.Path(f"{self.output_dir}/{prefix}.row_names.joblib").unlink()
        return

    def _collect_parquet_regions(self):
        total_region_info = []
        dataset_dir = pathlib.Path(self.output_dir).absolute().resolve()
        for parq_file in dataset_dir.glob("*/*.parquet"):
            data = pq.read_table(parq_file, columns=["region"]).to_pydict()
            df = pd.DataFrame({"region": data["region"]})
            df["parquet_file"] = str(parq_file).split(str(dataset_dir))[-1].lstrip("/")
            df["row_id_in_parquet"] = range(df.shape[0])
            total_region_info.append(df)
        total_region_info = pd.concat(total_region_info).reset_index(drop=True)
        total_region_info.to_feather(dataset_dir / "parquet_row_regions.feather")

    def generate(self) -> None:
        """
        Generate the ray dataset.
        """
        # make sure bolero.init is runed and resources are available
        msg = "Please run bolero.init() before create dataset generator."
        try:
            assert "bolero_dataset_gen" in ray.cluster_resources(), msg
        except ray.exceptions.RaySystemError as e:
            raise AssertionError(msg) from e

        output_dir = self.output_dir
        success_flag_path = output_dir / "config.joblib"
        if success_flag_path.exists():
            return

        self._process_each_prefix()

        chroms = self.genome_chunk_df["Chromosome"].unique()
        for chrom in chroms:
            self._prepare_single_chrom(chrom)

        # save row names
        self._dump_row_names()

        # create success flag and record genome name
        config_dict = {
            "genome": self.genome.name,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "num_rows_per_file": self.num_rows_per_file,
            "prefix_metadata": {},
        }
        for prefix, meta in self.uniform_dataset_dict.items():
            config_dict["prefix_metadata"][prefix] = {
                "ds_class": meta["ds_class"].__name__,
                "ds_kwargs": meta["ds_kwargs"],
                "remote_kwargs": meta["remote_kwargs"],
            }
        joblib.dump(config_dict, success_flag_path)

        # cleanup
        for chrom in self.genome_chunk_df["Chromosome"].unique():
            chrom_dir = output_dir / chrom
            pathlib.Path(f"{chrom_dir}/success.flag").unlink()
        for prefix in self.uniform_dataset_dict.keys():
            pathlib.Path(f"{output_dir}/{prefix}.success.flag").unlink()

        self._collect_parquet_regions()
        return
