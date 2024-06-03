import pathlib
from typing import Union

import joblib
import ray

from bolero.pp.genome import Genome
from bolero.pp.genome_chunk_dataset import (
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
        num_rows_per_file: int = 100,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir).resolve().absolute()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome
        for chrom in self.genome.chromosomes:
            chrom_dir = self.output_dir / chrom
            chrom_dir.mkdir(exist_ok=True)

        self.window_size = window_size
        self.step_size = step_size
        assert (
            self.window_size >= self.step_size
        ), "Window size must be greater than step size."
        self.genome_chunk_df = self.genome.make_windows(
            window_size=self.window_size, step_size=self.step_size, as_df=True
        )
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
            "remote_kwargs": {"memory": 15 * 1024**3},
        }
        return

    def add_bigwig(self, prefix, name, path):
        """
        Add BigWig files. BigWig will be aggregated based on the prefix.

        Parameters
        ----------
        kwargs : Dict[str, str]
            The dataset name and the path to the BigWig file.
        """
        bw_class = GenomeBigWigDataset
        if prefix in self.uniform_dataset_dict:
            assert (
                self.uniform_dataset_dict[prefix][1] == bw_class
            ), f"Dataset with name {prefix} should be bigwig."
            self.uniform_dataset_dict[prefix]["ds_kwargs"][name] = str(path)
            self.uniform_dataset_dict[prefix]["remote_kwargs"]["memory"] += (
                0.5 * 1024**3
            )
        else:
            self.uniform_dataset_dict[prefix] = {
                "ds_class": bw_class,
                "ds_kwargs": {name: str(path)},
                "remote_kwargs": {"memory": 1 * 1024**3},
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
            "remote_kwargs": {"memory": 15 * 1024**3},
        }
        return

    def _process_each_prefix(self):
        prefix_tasks = []
        for prefix, (
            _ds_class,
            _ds_kwargs,
            _remote_kwargs,
        ) in self.uniform_dataset_dict.items():

            @ray.remote(**_remote_kwargs)
            def _process_worker(prefix, ds_class, ds_kwargs, output_dir, regions_df):
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

    def prepare_ray_dataset(self) -> None:
        """
        Prepare the ray dataset.
        """
        output_dir = self.output_dir
        success_flag_path = output_dir / "genome.flag"
        if success_flag_path.exists():
            return

        self._process_each_prefix()

        for chrom in self.genome.chromosomes:
            self._prepare_single_chrom(output_dir, chrom)

        # save row names
        self._dump_row_names()

        # create success flag and record genome name
        with open(success_flag_path, "w") as f:
            f.write(self.genome.name)

        # cleanup
        for chrom in self.bed["Chromosome"].unique():
            chrom_dir = output_dir / chrom
            pathlib.Path(f"{chrom_dir}/success.flag").unlink()
        for prefix in self.uniform_dataset_dict.keys():
            pathlib.Path(f"{output_dir}/{prefix}.success.flag").unlink()
        return
