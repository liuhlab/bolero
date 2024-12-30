import json
from collections import OrderedDict, defaultdict
from typing import Any, Iterable

import numpy as np

from bolero.tl.dataset.file_transforms import (
    FetchRegionALLCsReduced,
    FetchRegionBigWigsReduced,
    FetchRegionCools,
    ReverseCompHicData,
)
from bolero.tl.dataset.ray_dataset import RayRegionDataset
from bolero.tl.dataset.transforms import (
    FetchRegionOneHot,
)
from bolero.tl.model.borzoi.utils import BorzoiRegions
from bolero.utils import get_global_coords, understand_regions

DNA_NAME = "dna_one_hot"


class BorzoiDatasetOnline(RayRegionDataset):
    """Singel cell pseudobulk dataset for Borzoi model."""

    default_config = {
        "balance": False,
        "batch_size": 2,
        "bed": None,
        "cell_types": "REQUIRED",
        "dataset_path": "REQUIRED",
        "dna_window": 524288,
        "embeddings_path": "REQUIRED",
        "coverage_path": None,
        "hic_cap_value": None,
        "genome": "hg38",
        "max_jitter": 0,
        "n_pseudobulks": 100,
        "paired_data": False,
        "pos_resolution": "REQUIRED",
        "reverse_complement": True,
        "hic_resolution": "REQUIRED",
        "use_borzoi_regions": True,
        "data_key_to_file_type": None,
        "region2": False,
        "region2_max_dist": 2e6,
        "multihead_output": False,
        # methylation specific
        "unmethylated": False,
        "data_key_to_mc_context": None,
    }

    def __init__(
        self,
        bed: str,
        cell_types: list[str],
        dataset_path: str,
        genome: str,
        hic_resolution: int,
        batch_size: int = 2,
        balance: bool = False,
        coverage_path: str = None,
        hic_cap_value: float = None,
        cool_data_norm_mode: str = None,
        cov_filter_name: str = None,
        dna_window: int = 524288,
        embeddings_path: str = None,
        max_jitter: int = 3,
        n_pseudobulks: int = 100,
        paired_data: bool = False,
        pos_resolution: int = 32,
        reverse_complement: bool = True,
        use_borzoi_regions: bool = True,
        data_key_to_file_type: dict[str, str] = None,
        region2: bool = False,
        region2_max_dist: float = 5e6,
        multihead_output: bool = False,
        # methylation specific
        unmethylated: bool = False,
        data_key_to_mc_context: dict[str, str] = None,
    ):
        super().__init__(
            bed=bed,
            genome=genome,
            standard_length=dna_window,
            batch_size=batch_size,
            window_size=dna_window,
            use_borzoi_regions=use_borzoi_regions,
            dna=False,
            region2=region2,
            region2_max_dist=region2_max_dist,
        )

        with open(dataset_path) as f:
            dataset_path_dict = json.load(f)

        if coverage_path is not None:
            with open(coverage_path) as f:
                # {cell_type: {data_key: scale_factor}}
                # scale_factor is used to scale the coverage data
                # scale_factor = total_reads / reads_target
                coverage_path_dict = json.load(f)
        else:
            coverage_path_dict = {}

        if data_key_to_file_type is None:
            data_key_to_file_type = {}
            cell_type_file_keys = []
            for ct_files in dataset_path_dict.values():
                cell_type_file_keys.extend(list(ct_files.keys()))
            cell_type_file_keys = set(cell_type_file_keys)
            for key in cell_type_file_keys:
                if key.startswith("atac"):
                    data_key_to_file_type[key] = "bw"
                elif key.startswith("hic"):
                    data_key_to_file_type[key] = "cool"
                elif key.startswith("allc") or key.startswith("mc"):
                    data_key_to_file_type[key] = "allc"
                else:
                    raise ValueError(f"Unknown file type for key: {key}")
        self.data_key_to_file_type = data_key_to_file_type

        self.cell_types = cell_types
        # dataset_paths is {data_key: [ct_path1, ct_path2, ...]},
        # ct_path order the same as self.cell_types
        self.dataset_paths = defaultdict(list)
        self.dataset_scale_factors = defaultdict(list)
        for cell_type in cell_types:
            cell_type_files = dataset_path_dict[cell_type]
            cell_type_cov_scale_factors = coverage_path_dict.get(cell_type, {})
            for data_key in data_key_to_file_type.keys():
                # TODO HL: deal with cell type that missing a certain modality
                self.dataset_paths[data_key].append(cell_type_files[data_key])
                self.dataset_scale_factors[data_key].append(
                    cell_type_cov_scale_factors.get(data_key, 1.0)
                )
        # HiC
        self.cool_data_norm_mode = cool_data_norm_mode
        self.hic_cap_value = hic_cap_value
        self.hic_resolution = hic_resolution

        self.balance = balance
        self.multihead_output = multihead_output
        # Methylation specific
        self.unmethylated = unmethylated
        self.data_key_to_mc_context = (
            {} if data_key_to_mc_context is None else data_key_to_mc_context
        )

        # Embeddings map generation
        self.embeddings_path = embeddings_path
        with open(embeddings_path) as f:
            self.leg_map = json.load(f)

        self.batch_size = batch_size

        # Region properties
        self.dna_window = dna_window
        self.signal_window = dna_window
        self.pos_resolution = pos_resolution
        self.max_jitter = max_jitter
        self.reverse_complement = reverse_complement
        self.paired_data = paired_data
        self.n_pseudobulks = n_pseudobulks
        self.cov_filter_name = cov_filter_name
        self.use_borzoi_regions = use_borzoi_regions
        self.borzoi_regions = BorzoiRegions()

        self.name_to_pseudobulker = OrderedDict()
        return

    def get_train_valid_test(self, fold):
        """Get the train, valid, and test folds and regions for the given fold."""
        fold_split = self.borzoi_regions.fold_splits[fold]
        train_folds = fold_split["train"]
        valid_folds = fold_split["valid"]
        test_folds = fold_split["test"]

        train_regions, valid_regions, test_regions = (
            self.borzoi_regions.get_train_valid_test_regions(
                self.genome.name, split_id=fold, region_length=self.dna_window
            )
        )
        return (
            train_folds,
            valid_folds,
            test_folds,
            train_regions,
            valid_regions,
            test_regions,
        )

    def __repr__(self) -> str:
        _str = (
            f"{self.__name__}\n"
            f"Dataset Files: {self.dataset_paths}\n"
            f"Embedding Files: {self.embeddings_path}\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
        )
        return _str

    def _get_cool_data(
        self,
        dataset,
        data_key,
        cool_paths,
        concurrency=(1, 6),
        n_oprators=1,
        batch_size=8,
        norm_mode=None,
        image_scale=64,
        key_suffix=None,
    ):
        """
        Get the cool data for the dataset

        Parameters
        ----------
        dataset : RayRegionDataset
            The dataset to be processed.
        concurrency : tuple
            The concurrency for the dataset, min and max.
        n_oprators : int
            The number of oprators to be used when dataset contains multiple cool paths.
            Each operator will process a chunk of the cool paths and saved in separate data_key.
        batch_size : int
            The batch size for the cool operator.
            Small batch size will increase data fetching batch number and increase the concurrency.
        norm_mode : str
            The normalization mode for the cool data.

        Returns
        -------
        dataset : RayRegionDataset
            The dataset with cool data oprator mapped.
        """
        _chunk_size = max(5, len(cool_paths) // n_oprators)

        for idx, chunk_start in enumerate(range(0, len(cool_paths), _chunk_size)):
            chunk_end = min(len(cool_paths), chunk_start + _chunk_size)
            chunk_paths = cool_paths[chunk_start:chunk_end]

            fn = FetchRegionCools
            fn_constructor_kwargs = {
                "cool_paths": chunk_paths,
                "resolution": self.hic_resolution,
                "balance": self.balance,
                "data_key": f"{data_key}_{idx}",
                "norm_mode": norm_mode,  # Note: if the data is HBA data, no need to log transform, otherwise, log transform the data
                "image_scale": image_scale,
                "cap_value": self.hic_cap_value,
            }
            fn_kwargs = {"key_suffix": key_suffix}
            dataset = dataset.map_batches(
                fn=fn,
                fn_constructor_kwargs=fn_constructor_kwargs,
                fn_kwargs=fn_kwargs,
                concurrency=concurrency,
                batch_size=batch_size,
            )
        total_chunks = idx + 1

        # add a final concat function to merge all the chunks
        def _concat_cool_chunks(data, data_key=data_key, key_suffix=None):
            if key_suffix is None:
                key_suffix = [""]
            try:
                if "_2" in key_suffix:
                    _suffix = key_suffix + ["_1+2"]
                else:
                    _suffix = key_suffix

                for suffix in _suffix:
                    cool_keys = [
                        f"{data_key}_{idx}{suffix}" for idx in range(total_chunks)
                    ]
                    cool_data = [data.pop(key) for key in cool_keys]
                    data[data_key + suffix] = np.concatenate(cool_data, axis=1)

            except KeyError as e:
                print("data keys:", list(data.keys()))
                print("key suffix:", key_suffix)
                print("data key:", data_key)
                raise e
            return data

        dataset = dataset.map_batches(
            fn=_concat_cool_chunks,
            fn_kwargs={"data_key": data_key, "key_suffix": key_suffix},
            batch_size=batch_size,
        )
        return dataset

    def _get_bigwig_data(
        self,
        dataset,
        data_key,
        bigwig_paths,
        scale_factors=None,
        concurrency=(1, 6),
        n_operators=1,
        batch_size=8,
        norm_mode=None,
        key_suffix=None,
    ):
        """
        Get the bigwig data for the dataset, copied from corigami HiCTrackDataset

        Parameters
        ----------
        dataset : RayRegionDataset
            The dataset to be processed.
        data_key : str
            The key to store the bigwig data.
        concurrency : tuple
            The concurrency for the dataset, min and max.
        n_operators : int
            The number of operators to be used when dataset contains multiple cool paths.
            Each operator will process a chunk of the cool paths and saved in separate data_key.
        batch_size : int
            The batch size for the cool operator.
            Small batch size will increase data fetching batch number and increase the concurrency.
        norm_mode : str
            The normalization mode for the bigwig data.

        Returns
        -------
        dataset : RayRegionDataset
            The dataset with bigwig data oprator mapped.
        """
        _chunk_size = max(1, len(bigwig_paths) // n_operators)

        for idx, chunk_start in enumerate(range(0, len(bigwig_paths), _chunk_size)):
            chunk_end = min(len(bigwig_paths), chunk_start + _chunk_size)
            chunk_paths = bigwig_paths[chunk_start:chunk_end]

            # Get the signal into bins of 32 and adds cell type embedding info
            fn = FetchRegionBigWigsReduced
            fn_constructor_kwargs = {
                "bw_paths": chunk_paths,
                "scale_factors": scale_factors,
                "region_key": "region",  # this is what column from the dataframe is acted on by fn
                "data_key": f"{data_key}_{idx}",
                "norm_mode": norm_mode,
                "resolution": self.pos_resolution,
            }
            fn_kwargs = {"key_suffix": key_suffix}
            dataset = dataset.map_batches(
                fn=fn,
                fn_constructor_kwargs=fn_constructor_kwargs,
                fn_kwargs=fn_kwargs,
                concurrency=concurrency,
                batch_size=batch_size,
            )
        total_chunks = idx + 1

        def _concat_bw_chunks(data, data_key, total_chunks, key_suffix=None):
            if key_suffix is None:
                key_suffix = [""]

            for suffix in key_suffix:
                bw_keys = [f"{data_key}_{idx}{suffix}" for idx in range(total_chunks)]
                bw_data = [data.pop(key) for key in bw_keys if key in data]
                if not bw_data:
                    raise ValueError("No bigwig data found to concatenate.")
                data[data_key + suffix] = np.concatenate(bw_data, axis=1)
            return data

        dataset = dataset.map_batches(
            fn=_concat_bw_chunks,
            fn_kwargs={
                "data_key": data_key,
                "total_chunks": total_chunks,
                "key_suffix": key_suffix,
            },
            batch_size=batch_size,
        )
        return dataset

    def _get_allc_data(
        self,
        dataset,
        allc_paths,
        mc_prefix,
        concurrency=(1, 6),
        n_oprators=1,
        batch_size=8,
        key_suffix=None,
    ):
        # raise NotImplementedError
        """
        Get the cool data for the dataset

        Parameters
        ----------
        dataset : RayRegionDataset
            The dataset to be processed.
        concurrency : tuple
            The concurrency for the dataset, min and max.
        n_oprators : int
            The number of oprators to be used when dataset contains multiple data paths.
            Each operator will process a chunk of the data paths and saved in separate data_key.
        batch_size : int
            The batch size for the cool operator.
            Small batch size will increase data fetching batch number and increase the concurrency.

        Returns
        -------
        dataset : RayRegionDataset
            The dataset with cool data oprator mapped.
        """
        _chunk_size = max(1, len(allc_paths) // n_oprators)
        if key_suffix is None:
            key_suffix = [""]

        mc_context = self.data_key_to_mc_context.get(mc_prefix, None)
        for idx, chunk_start in enumerate(range(0, len(allc_paths), _chunk_size)):
            chunk_end = min(len(allc_paths), chunk_start + _chunk_size)
            chunk_paths = allc_paths[chunk_start:chunk_end]

            fn = FetchRegionALLCsReduced
            fn_constructor_kwargs = {
                "allc_paths": chunk_paths,
                "data_prefix": f"{idx}_{mc_prefix}_",
                "data_suffix": key_suffix,
                "region_key": "region",
                "mc_context": mc_context,
            }
            dataset = dataset.map_batches(
                fn=fn,
                fn_constructor_kwargs=fn_constructor_kwargs,
                concurrency=concurrency,
                batch_size=batch_size,
            )
        total_chunks = idx + 1

        # add a final concat function to merge all the chunks
        def _concat_allc_chunks(data, key_suffix):
            for suffix in key_suffix:
                for key in ["mc", "cov"]:
                    allc_keys = [
                        f"{idx}_{mc_prefix}_{key}{suffix}"
                        for idx in range(total_chunks)
                    ]
                    allc_data = [data.pop(key) for key in allc_keys]
                    data[f"{mc_prefix}_{key}{suffix}"] = np.concatenate(
                        allc_data, axis=1
                    )
            return data

        dataset = dataset.map_batches(
            fn=_concat_allc_chunks,
            fn_kwargs={"key_suffix": key_suffix},
            batch_size=batch_size,
        )
        return dataset

    def _get_mc_frac(self, dataset, mc_prefix, key_suffix=None):
        if key_suffix is None:
            key_suffix = [""]

        # calculate mC fraction
        def _mc_frac(data_dict, mc_prefix, key_suffix, unmethylated):
            for suffix in key_suffix:
                mc = data_dict[f"{mc_prefix}_mc{suffix}"]
                cov = data_dict[f"{mc_prefix}_cov{suffix}"]
                if unmethylated:
                    frac = (cov - mc) / (cov + 1e-6)
                else:
                    frac = mc / (cov + 1e-6)
                data_dict[f"{mc_prefix}_mc_frac{suffix}"] = frac
            return data_dict

        dataset = dataset.map_batches(
            _mc_frac,
            fn_kwargs={
                "mc_prefix": mc_prefix,
                "key_suffix": key_suffix,
                "unmethylated": self.unmethylated,
            },
        )
        return dataset

    def _get_dna_one_hot(self, dataset, concurrency, key_suffix=None):
        """
        Get the DNA one hot for the dataset.
        """
        fn = FetchRegionOneHot
        fn_constructor_kwargs = {
            "random_shift": self.max_jitter if self._dataset_mode == "train" else 0,
            "dtype": "bool",
        }
        fn_kwargs = {
            "remote_genome_one_hot": self.genome.remote_genome_one_hot,
            "key_suffix": key_suffix,
        }

        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            fn_kwargs=fn_kwargs,
            concurrency=concurrency,
        )
        self.dna_column = DNA_NAME
        return dataset

    def _get_reverse_complement_dataset(
        self,
        dataset,
        dna_key="dna_one_hot",
        data_1d_keys=("atac",),
        data_2d_keys=("hic",),
        chance=0.5,
        concurrency=(1, 6),
        batch_size=8,
        key_suffix=None,
    ):
        """
        Reverse complement the DNA, ATAC and HiC data.
        """
        fn = ReverseCompHicData
        fn_constructor_kwargs = {
            "dna_key": dna_key,
            "data_1d_keys": data_1d_keys,
            "data_2d_keys": data_2d_keys,
            "chance": chance,
        }
        fn_kwargs = {
            "key_suffix": key_suffix,
        }
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            fn_kwargs=fn_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _process_region_columns(self, dataset, keep_regions=False, key_suffix=None):
        """
        Keep the regions by converting them to global coordinates OR remove the region columns.
        """
        if key_suffix is None:
            key_suffix = [""]

        if keep_regions:
            chrom_offsets = self.genome.chrom_offsets.copy()

            def _region_to_global_coords(batch, key_suffix=None):
                for suffix in key_suffix:
                    region_df = understand_regions(batch.pop("region" + suffix))
                    global_coords = get_global_coords(
                        chrom_offsets=chrom_offsets,
                        region_bed_df=region_df,
                    )
                    batch["region" + suffix] = global_coords
                return batch

            dataset = dataset.map_batches(
                _region_to_global_coords, fn_kwargs={"key_suffix": key_suffix}
            )
        else:
            dataset = dataset.drop_columns(["region" + suffix for suffix in key_suffix])
        return dataset

    def _convert_to_list_dict(
        self,
        dataset,
        dna_key="dna_one_hot",
        data_1d_keys=("atac",),
        data_2d_keys=("hic",),
        concurrency=6,
        key_suffix=None,
    ):
        """
        Convert the data to list of dict.
        """

        def _convert_data(data_dict, key_suffix=None):
            if key_suffix is None:
                key_suffix = [""]

            list_data_dict = []
            for i, cell_type in enumerate(self.cell_types):
                new_data_dict = OrderedDict()

                # TODO: this implementation is also potentially problematic,
                # for read-only this might be OK, but if embedding change,
                # this will silently pruduce unpredictable results
                # add cell type embedding in separate operation
                new_data_dict["cell_type_embedding"] = np.array(
                    self.leg_map[cell_type]
                ).astype(np.float32)  # puts in embeddings
                new_data_dict["cell_type_id"] = i

                cell_type_related_keys = set()
                for suffix in key_suffix:
                    for feature in data_1d_keys:
                        new_data_dict[feature + suffix] = data_dict[feature + suffix][
                            i, :
                        ]  # this only works because bw names are enumerated in order of cell type
                        cell_type_related_keys.add(feature + suffix)
                    for feature in data_2d_keys:
                        new_data_dict[feature + suffix] = data_dict[feature + suffix][
                            i, :, :
                        ]
                        cell_type_related_keys.add(feature + suffix)
                    new_data_dict[dna_key + suffix] = data_dict[dna_key + suffix]
                    cell_type_related_keys.add(dna_key + suffix)

                # deal with the interaction data in 2d data
                if "_2" in key_suffix:
                    for feature in data_2d_keys:
                        new_data_dict[feature + "_1+2"] = data_dict[feature + "_1+2"][
                            i, :, :
                        ]
                        cell_type_related_keys.add(feature + "_1+2")

                # copy any remaining data
                for k in data_dict.keys():
                    if k not in cell_type_related_keys:
                        new_data_dict[k] = data_dict[k]

                list_data_dict.append(new_data_dict)

            return list_data_dict

        dataset = dataset.flat_map(
            fn=_convert_data,
            concurrency=concurrency,
            fn_kwargs={"key_suffix": key_suffix},
        )

        return dataset

    def _combine_channels(self, dataset, key_list, out_key):
        """Concatenate the channels in key_list to form out_key."""

        def _combine_channels_fn(data_dict, key_list, out_key):
            data_to_concat = []
            for key in key_list:
                data = data_dict[key]
                if data.ndim == 2:
                    data = data[:, np.newaxis, :]  # add channel dimension
                data_to_concat.append(data)

            out_data = np.concatenate(data_to_concat, axis=1)
            data_dict[out_key] = out_data
            return data_dict

        dataset = dataset.map_batches(
            _combine_channels_fn,
            fn_kwargs={"key_list": key_list, "out_key": out_key},
            batch_size=8,
        )
        return dataset

    def get_processed_dataset(
        self,
        region_bed: str,
        return_regions: bool = True,
        concurrency: int = 6,
    ) -> None:
        """
        Process the dataset and return the processed dataset.

        Parameters
        ----------
        - folds (list): List of folds to include in the dataset.
        - region_bed_path (str): Path to the BED file containing the regions.
        - return_cells (bool): Whether to return the cells in the dataset. Default is False.
        - return_regions (bool): Whether to return the regions in the dataset. Default is False.

        Returns
        -------
        - work_ds (Dataset): The processed dataset.

        """
        concurrency = (1, concurrency)

        # Get the bed in dataframe with ray (region_bed has been determined using train_regions for example)
        work_ds = super().get_processed_dataset(
            bed=region_bed,
            shuffle_bed=self.is_train(),
        )  # comes directly preprocessed as dataframe for fold split we're using

        if self.region2:
            key_suffix = ["", "_2"]
        else:
            key_suffix = None  # [""]

        data_1d_keys = []
        data_2d_keys = []
        for data_key, file_type in self.data_key_to_file_type.items():
            file_paths = self.dataset_paths[data_key]
            if file_type in ("bw", "bigwig"):
                data_1d_keys.append(data_key)
                work_ds = self._get_bigwig_data(
                    dataset=work_ds,
                    data_key=data_key,
                    bigwig_paths=file_paths,
                    concurrency=concurrency,
                    norm_mode=None,
                    key_suffix=key_suffix,
                    scale_factors=self.dataset_scale_factors[data_key],
                )
            elif file_type in ("allc",):
                work_ds = self._get_allc_data(
                    dataset=work_ds,
                    allc_paths=file_paths,
                    mc_prefix=data_key,
                    concurrency=concurrency,
                    key_suffix=key_suffix,
                )
                work_ds = self._get_mc_frac(
                    dataset=work_ds,
                    mc_prefix=data_key,
                    key_suffix=key_suffix,
                )
                data_1d_keys.append(f"{data_key}_mc")
                data_1d_keys.append(f"{data_key}_cov")
                data_1d_keys.append(f"{data_key}_mc_frac")
            elif file_type in ("cool", "hic"):
                data_2d_keys.append(data_key)
                work_ds = self._get_cool_data(
                    dataset=work_ds,
                    data_key=data_key,
                    cool_paths=file_paths,
                    concurrency=concurrency,
                    norm_mode=self.cool_data_norm_mode,
                    key_suffix=key_suffix,
                )
            else:
                raise ValueError(f"Unknown file type: {file_type}")

        # add dna one hot, add bool datatype
        work_ds = self._get_dna_one_hot(
            work_ds,
            concurrency=concurrency,
            key_suffix=key_suffix,
        )

        if self.reverse_complement and self._dataset_mode == "train":
            work_ds = self._get_reverse_complement_dataset(
                work_ds,
                data_1d_keys=data_1d_keys,
                data_2d_keys=data_2d_keys,
                key_suffix=key_suffix,
            )

        # remove region column OR turn it into global coordinates (str to numbers)
        work_ds = self._process_region_columns(
            dataset=work_ds, keep_regions=return_regions, key_suffix=key_suffix
        )

        # add clamp sqrt
        # work_ds = self._add_clamp_sqrt(work_ds)

        if not self.multihead_output:
            work_ds = self._convert_to_list_dict(
                work_ds,
                data_1d_keys=data_1d_keys,
                data_2d_keys=data_2d_keys,
                key_suffix=key_suffix,
            )

        if (
            self.data_key_to_mc_context is not None
            and len(self.data_key_to_mc_context) > 1
        ):
            # combine the mc_frac channels
            mc_frac_keys = [
                f"{data_key}_mc_frac" for data_key in self.data_key_to_mc_context.keys()
            ]
            work_ds = self._combine_channels(
                dataset=work_ds,
                key_list=mc_frac_keys,
                out_key="mc_frac",
            )
        return work_ds

    def get_dataloader(
        self,
        region_bed,
        as_torch=True,
        return_regions=True,
        n_batches=None,
        shuffle_rows=300,
        concurrency=6,
        **dataloader_kwargs,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader.

        Parameters
        ----------
        region_bed : str
            The path to the region bed file.
        as_torch : bool, optional
            Whether to return a PyTorch dataloader, by default True.
        return_regions : bool, optional
            Whether to return the regions, by default True.
        n_batches : int, optional
            The number of batches to return, by default None.
        concurrency : int, optional
            The number of concurrent processes to use, by default 20.
        **dataloader_kwargs
            Additional keyword arguments pass to ray.data.Dataset.iter_batches.

        Returns
        -------
        DataLoader
            The dataloader.
        """
        # the region bed is already preprocessed for folds but we want to make sure names are corerct
        # for rename
        region_bed["Chromosome"] = region_bed["Chromosome"].astype(str)
        region_bed.rename(columns={"Name": "region"}, inplace=True)
        self.bed = region_bed

        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "region_bed": region_bed,
            "return_regions": return_regions,
            "concurrency": concurrency,
        }
        data_iter_kwargs = {
            "local_shuffle_buffer_size": (shuffle_rows if self.is_train() else None),
        }
        data_iter_kwargs.update(dataloader_kwargs)

        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            as_torch=as_torch,
            n_batches=n_batches,
            batch_size=self.batch_size,
        )
        return loader
