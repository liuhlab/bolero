import pathlib
from typing import Union

import joblib
import ray

from bolero.pp.genome import Genome
from bolero.pp.genome_chunk_dataset import (
    GenomeALLCDataset,
    GenomeBigWigDataset,
    SingleCellCutsiteDataset,
    SnapAnnDataDataset,
)


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
        genome: Union[str, Genome],
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

    def add_zarr(self, prefix, path, barcode_whitelist=None):
        """
        Add Zarr datasets.

        Parameters
        ----------
        kwargs : Dict[str, str]
            The dataset name and the path to the Zarr file.

        """
        if prefix in self.uniform_dataset_dict:
            raise ValueError(f"Dataset with name {prefix} already exists.")
        self.uniform_dataset_dict[prefix] = {
            "ds_class": SingleCellCutsiteDataset,
            "ds_kwargs": {
                "name": prefix,
                "zarr_path": path,
                "barcode_whitelist": barcode_whitelist,
            },
            "remote_kwargs": {"memory": 15 * 1024**3, "resources": {"bolero": 10}},
        }
        return

    def add_bigwig(self, prefix, name, path, sparse=True, compress_level=5):
        """
        Add BigWig files. BigWig will be aggregated based on the prefix.

        Parameters
        ----------
        kwargs : Dict[str, str]
            The dataset name and the path to the BigWig file.
        """
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
                },
                "remote_kwargs": {"memory": 1 * 1024**3, "resources": {"bolero": 10}},
            }
        return

    def add_snap_adata(self, prefix, path, barcode_whitelist=None):
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
        if prefix in self.uniform_dataset_dict:
            raise ValueError(f"Dataset with name {prefix} already exists.")
        self.uniform_dataset_dict[prefix] = {
            "ds_class": SnapAnnDataDataset,
            "ds_kwargs": {
                "name": prefix,
                "path": path,
                "barcode_whitelist": barcode_whitelist,
            },
            "remote_kwargs": {"memory": 15 * 1024**3, "resources": {"bolero": 10}},
        }
        return

    def add_allc(self, prefix, name, path, sparse=True, compress_level=5):
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
                },
                "remote_kwargs": {"memory": 1 * 1024**3, "resources": {"bolero": 10}},
            }
        return

    def _process_each_prefix(self):
        prefix_tasks = []
        for prefix, info_dict in self.uniform_dataset_dict.items():
            _ds_class = info_dict["ds_class"]
            _ds_kwargs = info_dict["ds_kwargs"]
            _remote_kwargs = info_dict["remote_kwargs"]

            @ray.remote(**_remote_kwargs)
            def _process_worker(prefix, ds_class, ds_kwargs, output_dir, regions_df):
                print("Processing", prefix, ds_class)
                # check success flag
                success_flag_path = output_dir / f"{prefix}.success.flag"
                if success_flag_path.exists():
                    return

                ds = ds_class(**ds_kwargs)
                list_of_dicts = ds.get_regions_data(regions_df)

                chromosomes = regions_df["Chromosome"].unique()
                for chrom in chromosomes:
                    chrom_dir = output_dir / chrom
                    chrom_list_of_dicts = [
                        d for d in list_of_dicts if d["region"].split(":")[0] == chrom
                    ]
                    joblib.dump(
                        chrom_list_of_dicts,
                        chrom_dir / f"{prefix}.list_of_dicts.joblib",
                    )

                # dump row names
                row_names = ds.get_row_names()
                joblib.dump(row_names, output_dir / f"{prefix}.row_names.joblib")

                # create a success flag
                pathlib.Path(success_flag_path).touch()
                return

            task = _process_worker.remote(
                prefix=prefix,
                ds_class=_ds_class,
                ds_kwargs=_ds_kwargs,
                output_dir=self.output_dir,
                regions_df=self.genome_chunk_df,
            )
            prefix_tasks.append(task)
        ray.get(prefix_tasks)
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

    def generate(self) -> None:
        """
        Generate the ray dataset.
        """
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
        }
        joblib.dump(config_dict, success_flag_path)

        # cleanup
        for chrom in self.genome_chunk_df["Chromosome"].unique():
            chrom_dir = output_dir / chrom
            pathlib.Path(f"{chrom_dir}/success.flag").unlink()
        for prefix in self.uniform_dataset_dict.keys():
            pathlib.Path(f"{output_dir}/{prefix}.success.flag").unlink()
        return
