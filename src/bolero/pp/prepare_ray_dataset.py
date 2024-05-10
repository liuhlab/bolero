import pathlib
import shutil
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import ray
import xarray as xr
from pyarrow.fs import LocalFileSystem
from tqdm import tqdm

from bolero.pp.genome import Genome
from bolero.pp.genome_dataset import (
    GenomeBigWigDataset,
    GenomePositionZarr,
    GenomeSingleCellCutsiteDataset,
)
from bolero.utils import get_fs_and_path, parse_region_name, understand_regions


class GenomeEnsembleDataset:
    """
    Represents an ensemble dataset for genomic data.

    Parameters
    ----------
        genome (str or Genome): The genome associated with the dataset.

    Attributes
    ----------
        genome (Genome): The genome associated with the dataset.
        datasets (dict): A dictionary of dataset names and corresponding GenomeWideDataset objects.

    """

    def __init__(self, genome: Union[str, Genome], add_genome_one_hot=True):
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome

        self.datasets = {}
        # special slots
        if add_genome_one_hot:
            self.add_genome_one_hot()
        self._bigwig_dataset = None

        self.regions = {}
        self.region_sizes = {}
        self._n_regions = None
        self._region_dataset_query_pairs = set()

    def __repr__(self):
        repr_str = f"GenomeEnsembleDataset(genome={self.genome.name})\n"

        _s = f"\nDatasets ({len(self.datasets)}):\n"
        repr_str += "\n" + "=" * (len(_s) - 1) + _s + "=" * (len(_s) - 1)

        for name, dataset in self.datasets.items():
            # also collect information about which regions will be queried in this dataset
            regions = []
            for region_name, dataset_name in self._region_dataset_query_pairs:
                if dataset_name == name:
                    regions.append(region_name)
            regions_str = "Query by regions: " + ", ".join(regions)
            dataset_str = dataset.__repr__().strip("\n")
            repr_str += (
                f"\n{name}: {type(dataset).__name__}\n{dataset_str}\n{regions_str}\n"
            )

        _s = f"\nRegions ({len(self.regions)}):\n"
        repr_str += "\n" + "=" * (len(_s) - 1) + _s + "=" * (len(_s) - 1) + "\n"

        for name, regions in self.regions.items():
            length = self.region_sizes[name]
            repr_str += f"{name}: {len(regions)} regions, region length {length} bp.\n"
        return repr_str

    def __getitem__(self, key):
        return self.datasets[key]

    def __setitem__(self, key, value):
        if key in self.datasets:
            raise ValueError(f"Dataset {key} already exists.")
        self.datasets[key] = value
        return

    def __delitem__(self, key):
        del self.datasets[key]
        return

    def add_regions(
        self,
        name,
        regions,
        length=2500,
        query_datasets="all",
        remove_blacklist=True,
        boarder_strategy="drop",
    ):
        """
        Adds regions to the ensemble.

        The regions will be used to retrieve region-level data from the genome datasets.

        Parameters
        ----------
            name (str): The name of the regions.
            regions (str, pathlib.Path, PyRanges or pd.DataFrame): The regions bed table to add.
            length (int): The length of the regions to standardize to.
                If None, the regions will not be standardized and
                user must ensure all regions have the same length
                and do not exceed the genome boarder. Default is 2500.
            query_datasets (str or List[str]): The datasets to query when retrieving region data.
                Default is 'all', which queries all datasets in self.datasets.
            check_length (bool): Whether to check if all regions have the same length. Default is False.
            remove_blacklist (bool): Whether to remove regions that overlap with blacklisted regions. Default is True.
            boarder_strategy (str): The stratagy to handle regions that go beyond the genome boarder. Default is 'drop'. See `Genome.standard_region_length` for more details.

        """
        regions = understand_regions(regions, as_df=True)
        if length is not None:
            try:
                regions = self.genome.standard_region_length(
                    regions=regions,
                    length=length,
                    boarder_strategy=boarder_strategy,
                    remove_blacklist=remove_blacklist,
                    as_df=True,
                )
            except pd.errors.EmptyDataError:
                print(f"No regions left after standardizing {name}.")
                return
            if remove_blacklist:
                # region length may change after removing blacklisted regions
                regions = regions[(regions["End"] - regions["Start"]) == length].copy()
                if regions.empty:
                    print(
                        f"No regions left after removing blacklisted regions in {name}."
                    )
                    return
            region_size = length
        else:
            region_size = regions.iloc[0, 2] - regions.iloc[0, 1]
            region_lengths = regions["End"] - regions["Start"]
            assert (
                region_lengths == region_size
            ).all(), "All regions must have the same length."

        self.regions[name] = regions
        self.region_sizes[name] = region_size

        if self._n_regions is None:
            self._n_regions = len(regions)
        else:
            assert (
                len(regions) == self._n_regions
            ), "All region beds must have the same number of regions."

        if query_datasets == "all":
            query_datasets = list(self.datasets.keys())
        elif isinstance(query_datasets, str):
            query_datasets = [query_datasets]
        else:
            query_datasets = list(query_datasets)

        for dataset_name in query_datasets:
            assert dataset_name in self.datasets, f"Dataset {dataset_name} not found."
            self._region_dataset_query_pairs.add((name, dataset_name))
        return

    def add_bigwig(
        self,
        name="bigwig",
        *args,
        **kwargs,
    ):
        """
        Adds a BigWig dataset to the ensemble.

        Parameters
        ----------
            name (str): The name of the dataset.
            bigwig_path (str or List[str] or pathlib.Path): The path(s) to the BigWig file(s).

        """
        bigwig_dict = {}
        for key, value in kwargs.items():
            bigwig_dict[key] = value
        for arg in args:
            _path = pathlib.Path(arg)
            bigwig_dict[_path.name] = str(_path)

        if self._bigwig_dataset is None:
            self._bigwig_dataset = GenomeBigWigDataset(**bigwig_dict)
            self.datasets[name] = self._bigwig_dataset
        else:
            self._bigwig_dataset.add_bigwig(**bigwig_dict)
        return

    def add_position_zarr(
        self,
        zarr_path: Union[str, pathlib.Path, xr.Dataset, xr.DataArray],
        name: str = None,
        da_name: str = None,
        load: bool = False,
    ):
        """
        Adds a position Zarr dataset to the ensemble.

        Parameters
        ----------
            name (str): The name of the dataset.
            zarr_path (str or pathlib.Path): The path to the Zarr dataset.
            load (bool): Whether to load the dataset into memory.

        """
        if isinstance(zarr_path, (str, pathlib.Path)):
            zarr = xr.open_zarr(zarr_path)

        if isinstance(zarr, xr.Dataset):
            if da_name is None:
                _data_vars = list(zarr.data_vars)
                if len(_data_vars) != 1:
                    raise ValueError(
                        "da_name must be specified if there is more than one data variable in the Zarr dataset. "
                        "Available data variables: " + ", ".join(_data_vars)
                    )
                else:
                    da_name = list(zarr.data_vars)[0]
            zarr = zarr[da_name]

        if isinstance(zarr, xr.DataArray):
            self.datasets[name] = GenomePositionZarr(
                da=zarr, offsets=self.genome.chrom_offsets, load=load
            )
        else:
            raise ValueError(
                "zarr must be a path to a Zarr dataset or an xarray Dataset or DataArray."
            )
        return

    def add_genome_one_hot(self, name="dna_one_hot"):
        """
        Adds the genome one-hot encoding to the ensemble.

        Parameters
        ----------
            name (str): The name of the dataset.

        """
        self.datasets[name] = self.genome.genome_one_hot
        return

    def get_region_data(self, *args) -> Dict[str, Any]:
        """
        Retrieves the data for a specific region.

        Parameters
        ----------
            *args: Either a region name or chrom, start, end values.

        Returns
        -------
            region_data (dict): A dictionary containing the data for each dataset.

        Raises
        ------
            ValueError: If args is not a region name or chrom, start, end values.

        """
        if len(args) == 1:
            chrom, start, end = parse_region_name(args[0])
        elif len(args) == 3:
            chrom, start, end = args
        else:
            raise ValueError("args must be a region name or chrom, start, end")

        region_data = {}
        for name, dataset in self.datasets.items():
            _data = dataset.get_region_data(chrom=chrom, start=start, end=end)
            if isinstance(_data, dict):
                region_data.update(_data)
            else:
                region_data[name] = _data
        return region_data

    def get_regions_data(
        self,
        query_chunk_size: int = 5000,
        region_index: pd.Index = None,
        add_region_ids: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieves the data for multiple regions.

        Parameters
        ----------
            chunk_size (int): The size of each chunk of regions during parallel retrieval. Default is 5000.
            region_index (pd.Index): The index of regions to retrieve data for. Default is None.


        Returns
        -------
            regions_data (dict): A dictionary containing the data for each dataset.

        """
        data_collections = {}
        for region_name, dataset_name in self._region_dataset_query_pairs:
            region_name = region_name.replace("|", "_")
            dataset_name = dataset_name.replace("|", "_")

            regions = self.regions[region_name]
            if region_index is not None:
                regions = regions.iloc[region_index, :].copy()

            dataset = self.datasets[dataset_name]
            regions_data = dataset.get_regions_data(
                regions=regions, chunk_size=query_chunk_size
            )
            if isinstance(regions_data, dict):
                for _ds_name, _data in regions_data.items():
                    _final_name = f"{region_name}|{_ds_name}"
                    data_collections[_final_name] = _data
            else:
                _final_name = f"{region_name}|{dataset_name}"
                data_collections[_final_name] = regions_data

            if add_region_ids:
                data_collections["region_ids"] = (
                    regions["Chromosome"].astype(str)
                    + ":"
                    + regions["Start"].astype(str)
                    + "-"
                    + regions["End"].astype(str)
                ).values
        return data_collections

    def _get_ray_dataset(
        self,
        n_regions,
        collate_fn_dict=None,
        region_index=None,
        add_region_ids=True,
    ):
        """
        Internal method to get a Ray dataset.

        Parameters
        ----------
            regions: The regions for which to retrieve the data.
            collate_fn_dict (dict): A dictionary of collate functions for each dataset. The keys can be the dataset name or the region name or their combination.
            Each collate function should take a numpy array as input and return a summary statistic.
            region_index (pd.Index): The index of regions to retrieve data for.

        Returns
        -------
            ds: The Ray dataset.

        """
        data_collections = self.get_regions_data(
            query_chunk_size=5000,
            region_index=region_index,
            add_region_ids=add_region_ids,
        )

        # calculate summary stats
        summary_stats_collections = {}
        if collate_fn_dict:
            for _final_name, _data in data_collections.items():
                try:
                    _region_name, _ds_name = _final_name.split("|")
                except ValueError:
                    _region_name, _ds_name = "", ""
                if _final_name in collate_fn_dict:
                    _funcs = collate_fn_dict[_final_name]
                elif _ds_name in collate_fn_dict:
                    _funcs = collate_fn_dict[_ds_name]
                elif _region_name in collate_fn_dict:
                    _funcs = collate_fn_dict[_region_name]
                else:
                    _funcs = None
                if _funcs:
                    summary_stats_collections[_final_name] = _funcs(_data)

        # reorganize data into items
        item_dicts = []
        for idx in range(n_regions):
            item_dict = {}
            for name, regions_data in data_collections.items():
                item_dict[name] = regions_data[idx]
            item_dicts.append(item_dict)

        ds = ray.data.from_items(item_dicts)
        return ds, summary_stats_collections

    @classmethod
    def subset_regions(cls, dataset, region_sel):
        """
        Subsets the regions in the ensemble.

        Parameters
        ----------
            region_sel: The regions to select.

        Returns
        -------
            ensemble: The ensemble with the selected regions.

        """
        ensemble = cls(dataset.genome)
        ensemble.datasets = dataset.datasets
        ensemble.regions = {
            k: v.iloc[region_sel, :].copy() for k, v in dataset.regions.items()
        }
        ensemble.region_sizes = dataset.region_sizes.copy()
        ensemble._n_regions = len(ensemble.regions)
        ensemble._region_dataset_query_pairs = dataset._region_dataset_query_pairs
        return ensemble

    def prepare_ray_dataset(
        self,
        output_dir: str,
        dataset_size: int = 500000,
        collate_fn_dict: dict = None,
        region_index: pd.Index = None,
        add_region_ids: bool = True,
        num_rows_per_file: int = 2000,
    ) -> None:
        """
        Prepares a Ray dataset for the given regions.

        Parameters
        ----------
            output_dir (str): The directory path to save the dataset.
            dataset_size (int): The maximum size of each dataset.
            collate_fn_dict (dict): A dictionary of collate functions for each dataset.
                The keys can be the dataset name or the region name or their combination.
                Each collate function should take a numpy array as input and return a summary statistic.
                Data will be saved by joblib.dump.
            region_index (pd.Index): The index of regions to retrieve data for.

        """
        if region_index is None:
            n_regions = self._n_regions
        else:
            n_regions = region_index.size
        if n_regions is None or n_regions == 0:
            print("No regions to prepare dataset for.")
            return

        fs, output_dir = get_fs_and_path(output_dir)
        dataset_path = f"{output_dir}/dataset/"
        stats_path = f"{output_dir}/stats"
        success_flag_path = f"{output_dir}/success.flag"

        # save the dataset
        if n_regions <= dataset_size:
            # check if success flag exists
            success = False
            if isinstance(fs, LocalFileSystem):
                if pathlib.Path(success_flag_path).exists():
                    success = True
            else:
                file_type = fs.get_file_info(success_flag_path).type
                if file_type:
                    success = True
            if success:
                print(f"Dataset already exists at {dataset_path}.")
                return

            ds, summary_stats_collections = self._get_ray_dataset(
                n_regions=n_regions,
                collate_fn_dict=collate_fn_dict,
                region_index=region_index,
                add_region_ids=add_region_ids,
            )
            ds.write_parquet(
                dataset_path, filesystem=fs, num_rows_per_file=num_rows_per_file
            )

            # save summary stats
            summary_stats_path = f"{stats_path}/summary_stats.npz"
            # determine if fs is local filesystem
            if isinstance(fs, LocalFileSystem):
                stats_path = pathlib.Path(stats_path)
                stats_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(summary_stats_path, **summary_stats_collections)
            else:
                with fs.open_output_stream(summary_stats_path) as f:
                    np.savez_compressed(f, **summary_stats_collections)

            # create success flag
            if isinstance(fs, LocalFileSystem):
                with open(success_flag_path, "w") as f:
                    f.write("Success")
            else:
                with fs.open_output_stream(success_flag_path) as f:
                    f.write(b"Success")
            return

        # split regions into chunks and save each chunk as a separate dataset
        else:
            starts = list(range(0, n_regions, dataset_size))
            for chunk_start in tqdm(
                starts, desc=f"Preparing dataset chunks of size {dataset_size} "
            ):
                chunk_end = min(chunk_start + dataset_size, n_regions)
                chunk_slice = slice(chunk_start, chunk_end)
                if region_index is None:
                    chunk_region_index = pd.Index(range(n_regions))[chunk_slice].copy()
                else:
                    chunk_region_index = region_index[chunk_slice].copy()

                chunk_output_path = f"{output_dir}/chunk_{chunk_start}_{chunk_end}"
                self.prepare_ray_dataset(
                    output_dir=chunk_output_path,
                    dataset_size=dataset_size,
                    collate_fn_dict=collate_fn_dict,
                    region_index=chunk_region_index,
                    add_region_ids=add_region_ids,
                    num_rows_per_file=num_rows_per_file,
                )
        return


def prepare_chromosome_dataset(
    genome: str,
    output_dir: str,
    regions_config: Union[str, Dict[str, str], List[str]],
    bigwig_config: Optional[Union[str, Dict[str, str], List[str]]] = None,
    zarr_config: Optional[Union[str, Dict[str, str], List[str]]] = None,
    collate_fn_dict: Optional[Dict[str, Callable]] = None,
    dataset_size: int = 100000,
    max_bigwig: int = 50,
    add_genome_one_hot: bool = True,
    num_rows_per_file: int = 2000,
) -> None:
    """
    Prepare chromosome datasets for a given genome.

    Parameters
    ----------
    genome : str
        The genome associated with the dataset.
    output_dir : str
        The directory to save the prepared datasets.
    regions_config : Union[str, Dict[str, str], List[str]]
        The configuration for the regions. It can be a single path string, a dictionary
        mapping region names to paths, or a list of paths.
    bigwig_config : Optional[Union[str, Dict[str, str], List[str]]], optional
        The configuration for the BigWig files, by default None. It can be a single path
        string, a dictionary mapping dataset names to paths, or a list of paths.
    zarr_config : Optional[Union[str, Dict[str, str], List[str]]], optional
        The configuration for the Zarr files, by default None. It can be a single path
        string, a dictionary mapping dataset names to paths, or a list of paths.
    collate_fn_dict : Optional[Dict[str, Callable]], optional
        A dictionary of collate functions for each dataset. The keys can be the dataset name
        or the region name or their combination. Each collate function should take a numpy
        array as input and return a summary statistic. Data will be saved by joblib.dump.
    dataset_size : int, optional
        The maximum size of rows in each dataset, by default 500000.
    max_bigwig : int, optional
        The maximum number of BigWig files to process at once, by default 50.
    add_genome_one_hot : bool, optional
        Whether to add the genome one-hot encoding to the dataset, by default True.
    num_rows_per_file : int, optional
        The number of rows per file in the dataset, by default 2000.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an invalid path type is provided.

    Example
    -------
    >>> genome = "mm10"
    >>> output_dir = "/mnt/datasets/chromosomes"
    >>> regions_config = {
    ...     "enhancers": "/mnt/data/regions/enhancers.bed",
    ...     "promoters": "/mnt/data/regions/promoters.bed",
    ... }
    >>> bigwig_config = {
    ...     "sample1": "/mnt/data/bigwig/sample1.bw",
    ...     "sample2": "/mnt/data/bigwig/sample2.bw",
    ... }
    >>> zarr_config = {
    ...     "methylation": "/mnt/data/zarr/methylation.zarr",
    ...     "histone_modifications": "/mnt/data/zarr/histone_modifications.zarr",
    ... }
    >>> prepare_chromosome_dataset(
    ...     genome, output_dir, regions_config, bigwig_config, zarr_config
    ... )
    """

    def _str_path_to_dict(p: Union[str, Dict[str, str], List[str]]) -> Dict[str, str]:
        if p is None:
            return {}
        if isinstance(p, dict):
            p = {k: str(pathlib.Path(v).absolute().resolve()) for k, v in p.items()}
            return p
        elif isinstance(p, (str, pathlib.Path)):
            ps = [pathlib.Path(p).absolute().resolve()]
        elif isinstance(p, list):
            ps = [pathlib.Path(pp).absolute().resolve() for pp in p]
        else:
            raise ValueError("Invalid path type.")
        p_dict = {pp.name: str(pp) for pp in ps}
        return p_dict

    # standardize paths
    regions_config = _str_path_to_dict(regions_config)
    bigwig_config = _str_path_to_dict(bigwig_config)
    zarr_config = _str_path_to_dict(zarr_config)

    # group each region by chrom and prepare the region configs
    chrom_region_configs = defaultdict(dict)
    for region_name, region_path in regions_config.items():
        region_bed = pr.read_bed(region_path, as_df=True)
        for chrom, chrom_region_bed in region_bed.groupby("Chromosome"):
            chrom_region_configs[chrom][region_name] = chrom_region_bed

    # split bigwig config if there is too many bigwigs
    bigwig_config_list = []
    bigwigs = list(bigwig_config.items())
    for i in range(0, len(bigwig_config), max_bigwig):
        bigwig_config_list.append(dict(bigwigs[i : i + max_bigwig]))

    for part_idx, bigwig_config_part in enumerate(bigwig_config_list):
        if len(bigwig_config_list) > 1:
            print(
                f"Preparing part {part_idx+1}/{len(bigwig_config_list)} of {len(bigwig_config_part)} bigwig files..."
            )
        if part_idx == len(bigwig_config_list) - 1:
            add_region_ids = True
            add_genome_one_hot = add_genome_one_hot
        else:
            add_region_ids = False
            add_genome_one_hot = False

        # prepare the dataset for each chromosome
        bar = tqdm(
            chrom_region_configs.items(),
            desc="Preparing chromosome datasets",
            total=len(chrom_region_configs),
        )
        if isinstance(genome, str):
            genome = Genome(genome)
        for chrom, chrom_region_config in bar:
            ensemble = GenomeEnsembleDataset(
                genome, add_genome_one_hot=add_genome_one_hot
            )

            if bigwig_config_part:
                ensemble.add_bigwig(**bigwig_config_part)

            if zarr_config:
                for name, zarr_path in zarr_config.items():
                    ensemble.add_position_zarr(zarr_path=zarr_path, name=name)

            for n, p in chrom_region_config.items():
                ensemble.add_regions(name=n, regions=p, query_datasets="all")

            output_path = f"{output_dir}/{chrom}/part{part_idx}"
            ensemble.prepare_ray_dataset(
                output_dir=output_path,
                dataset_size=dataset_size,
                collate_fn_dict=collate_fn_dict,
                add_region_ids=add_region_ids,
                num_rows_per_file=num_rows_per_file,
            )
    return


class SingleCellGenomeEnsembleDataset:
    """
    Represents a single-cell genome ensemble dataset.

    Parameters
    ----------
    bed : pd.DataFrame
        The bed file containing the regions.
    genome : Union[str, Genome]
        The genome associated with the dataset.
    zarr_dict : Dict[str, str]
        A dictionary mapping dataset names to Zarr file paths.
    length : int, optional
        The length of the regions, by default 2500.

    Attributes
    ----------
    genome : Genome
        The genome associated with the dataset.
    bed : pd.DataFrame
        The bed file containing the regions.
    zarr_dict : Dict[str, str]
        A dictionary mapping dataset names to Zarr file paths.

    """

    def __init__(
        self,
        bed: pd.DataFrame,
        genome: Union[str, Genome],
        zarr_dict: Dict[str, str],
        bw_dict: Optional[Dict[str, str]] = None,
        length: int = 2500,
        meta_region_size: int = 100000,
    ) -> None:
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome

        bed = genome.standard_region_length(
            bed,
            length=length,
            remove_blacklist=True,
            boarder_strategy="drop",
            as_df=True,
        )
        self.bed = bed

        self.meta_region_size = meta_region_size
        self.zarr_dict = zarr_dict
        self.bw_dict = bw_dict

    def _process_single_zarr(self, output_dir: str) -> None:
        """
        Process each Zarr file and save the data for each chromosome.

        Parameters
        ----------
        output_dir : str
            The output directory to save the processed data.

        Returns
        -------
        None

        """

        @ray.remote(memory=15 * 1024**3)  # request 15 gb mem
        def _process_worker(name, zarr_path, bed, meta_region_size):
            this_output_dir = output_dir / "single_zarr" / name
            this_output_dir.mkdir(exist_ok=True, parents=True)
            flag_path = this_output_dir / "success.flag"
            if flag_path.exists():
                return

            ds = GenomeSingleCellCutsiteDataset(
                name=name,
                bed=bed,
                zarr_path=zarr_path,
                meta_region_size=meta_region_size,
            )
            data_list = ds.get_meta_region_data()
            chrom_data_list = defaultdict(list)
            for data in data_list:
                chrom = data[f"{name}:meta_region"].split(":")[0]
                chrom_data_list[chrom].append(data)
            for chrom, data_list in chrom_data_list.items():
                chrom_path = this_output_dir / chrom
                joblib.dump(data_list, chrom_path)
            flag_path.touch()
            return

        tasks = []
        bed_remote = ray.put(self.bed)
        for name, zarr_path in self.zarr_dict.items():
            task = _process_worker.remote(
                name, zarr_path, bed_remote, self.meta_region_size
            )
            tasks.append(task)
        ray.get(tasks)
        return

    def _process_bigwig(self, output_dir: str) -> None:
        """
        Process BigWig files and save the data for each chromosome.

        Parameters
        ----------
        output_dir : str
            The output directory to save the processed data.

        Returns
        -------
        None
        """
        bw_output_dir = output_dir / "bigwig"
        bw_output_dir.mkdir(exist_ok=True, parents=True)
        flag_path = bw_output_dir / "success.flag"
        if flag_path.exists():
            return

        bigwig_ds = GenomeBigWigDataset(**self.bw_dict)
        data_list = bigwig_ds.get_meta_regions_data(
            regions=self.bed, meta_region_size=self.meta_region_size
        )

        chrom_data_list = defaultdict(list)
        for data in data_list:
            chrom = data["bigwig:meta_region"].split(":")[0]
            chrom_data_list[chrom].append(data)
        for chrom, data_list in chrom_data_list.items():
            chrom_path = output_dir / "bigwig" / chrom
            joblib.dump(data_list, chrom_path)

        flag_path.touch()
        return

    def _prepare_single_chrom(
        self, output_dir: str, chrom: str, num_rows_per_file: int
    ) -> None:
        """
        Prepare the dataset for a single chromosome.

        Parameters
        ----------
        output_dir : str
            The output directory to save the prepared dataset.
        chrom : str
            The chromosome to prepare the dataset for.
        num_rows_per_file : int
            The number of rows per file in the dataset.

        Returns
        -------
        None

        """
        chrom_dir = output_dir / chrom
        chrom_dir.mkdir(exist_ok=True, parents=True)
        flag_path = chrom_dir / "success.flag"
        if flag_path.exists():
            return

        list_of_dict = None
        for idx, name in enumerate(self.zarr_dict.keys()):
            chrom_data = f"{output_dir}/single_zarr/{name}/{chrom}"
            data_list = joblib.load(chrom_data)
            if idx == 0:
                list_of_dict = data_list
            else:
                for idx, d in enumerate(data_list):
                    list_of_dict[idx].update(d)

        if self.bw_dict is not None:
            chrom_data = f"{output_dir}/bigwig/{chrom}"
            data_list = joblib.load(chrom_data)
            if list_of_dict is None:
                list_of_dict = data_list
            else:
                for idx, d in enumerate(data_list):
                    list_of_dict[idx].update(d)

        # create ray dataset
        ray_dataset = ray.data.from_items(list_of_dict)
        ray_dataset.write_parquet(chrom_dir, num_rows_per_file=num_rows_per_file)

        # create success flag
        flag_path.touch()
        return

    def prepare_ray_dataset(
        self, output_dir: str, num_rows_per_file: int = 100
    ) -> None:
        """
        Prepare the ray dataset.

        Parameters
        ----------
        output_dir : str
            The output directory to save the prepared dataset.
        num_rows_per_file : int, optional
            The number of rows per file in the dataset, by default 200.

        Returns
        -------
        None

        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        success_flag_path = output_dir / "genome.flag"
        if success_flag_path.exists():
            return

        self._process_single_zarr(output_dir)
        if self.bw_dict is not None:
            self._process_bigwig(output_dir)

        for chrom in self.bed["Chromosome"].unique():
            self._prepare_single_chrom(output_dir, chrom, num_rows_per_file)

        # save barcodes
        barcodes_dict = {
            name: xr.open_zarr(zarr_path).coords["barcode"].values
            for name, zarr_path in self.zarr_dict.items()
        }
        np.savez_compressed(f"{output_dir}/barcodes.npz", **barcodes_dict)

        # cleanup
        shutil.rmtree(output_dir / "single_zarr")
        if self.bw_dict is not None:
            shutil.rmtree(output_dir / "bigwig")
        for chrom in self.bed["Chromosome"].unique():
            chrom_dir = output_dir / chrom
            pathlib.Path(f"{chrom_dir}/success.flag").unlink()

        # create success flag and record genome name
        with open(success_flag_path, "w") as f:
            f.write(self.genome.name)
        return
