import pathlib
from collections import OrderedDict
from typing import Any, Iterable

import ray
import anndata
import numpy as np
from bolero.tl.dataset.ray_dataset import (
    RayGenomeChunkDataset, RayRegionDataset
)
from bolero.tl.dataset.sc_transforms import FilterRegions
from bolero.tl.dataset.transforms import (
    FetchRegionOneHot,
    ReverseComplement,
)


from bolero.utils import get_global_coords, understand_regions
from bolero.tl.dataset.file_transforms import FetchRegionBigWigs, FetchRegionBigWigsReduced, FetchRegionALLCs, FetchRegionALLCsReduced
from .utils import BorzoiRegions, clamp_sqrt_large_value
import pandas as pd
DNA_NAME = "dna_one_hot"


class BorzoiDataset(RayGenomeChunkDataset):
    """Singel cell pseudobulk dataset for Borzoi model."""

    default_config = {
        "dataset_path": "REQUIRED",
        "batch_size": 2,
        "dna_window": 524288,
        "pos_resolution": 32,
        "reverse_complement": True,
        "max_jitter": 3,
        "n_pseudobulks": 100,
        "clamp_sqrt_threshold": None,
        "shuffle_files": True,
        "read_parquet_kwargs": None,
    }

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 2,
        dna_window: int = 524288,
        pos_resolution: int = 32,
        reverse_complement: bool = True,
        max_jitter: int = 3,
        n_pseudobulks: int = 100,
        cov_filter_name: str = None,
        clamp_sqrt_threshold: int = None,
        shuffle_files=False,
        read_parquet_kwargs=None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            shuffle_files=shuffle_files,
            read_parquet_kwargs=read_parquet_kwargs,
        )
        self.batch_size = batch_size

        # region properties
        self.dna_window = dna_window
        self.signal_window = dna_window
        self.pos_resolution = pos_resolution
        self.max_jitter = max_jitter
        self.reverse_complement = reverse_complement
        self.n_pseudobulks = n_pseudobulks
        self.cov_filter_name = cov_filter_name
        self.clamp_sqrt_threshold = clamp_sqrt_threshold

        self.name_to_pseudobulker = OrderedDict()

        self.borzoi_regions = BorzoiRegions()
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
            f"Dataset directory: {self.dataset_path}\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
        )
        return _str

    def _get_dna_one_hot(self, dataset, concurrency):
        fn = FetchRegionOneHot
        fn_constructor_kwargs = {
            "random_shift": self.max_jitter if self._dataset_mode == "train" else 0,
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

    def _process_region_columns(self, dataset, keep_regions=False): #TODO: might need to undo this later
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

    def _get_folds_dir(self, folds):
        if folds is None:
            fold_dirs = [str(p) for p in pathlib.Path(self.dataset_path).glob("fold*")]
        else:
            if isinstance(folds, str):
                folds = [folds]
            fold_dirs = [f"{self.dataset_path}/fold{fold}" for fold in folds]

            # make sure all fold_dir exists
            fold_dirs = [
                fold_dir for fold_dir in fold_dirs if pathlib.Path(fold_dir).exists()
            ]
            assert (
                len(fold_dirs) > 0
            ), f"None of the fold {folds} exists in {self.dataset_path}"
        return fold_dirs

    def _read_parquet(self, folds):
        _dataset = ray.data.read_parquet(
            self._get_folds_dir(folds),
            file_extensions=["parquet"],
            **self.read_parquet_kwargs,
        )
        return _dataset

    def _add_clamp_sqrt(self, dataset):
        if self.clamp_sqrt_threshold is None:
            return dataset

        signal_columns = self.signal_columns
        threshold = self.clamp_sqrt_threshold

        def _oprator(batch):
            for key in signal_columns:
                batch[key] = clamp_sqrt_large_value(batch[key], threshold=threshold)
            return batch

        dataset = dataset.map_batches(
            fn=_oprator,
            concurrency=(1, 4),
        )
        return dataset
    

    def _get_processed_dataset(
        self,
        folds,
        region_bed,
        name_to_pseudobulker,
        region_action_keys=None,
        concurrency=32,
        **pseudobulk_kwargs,
    ) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        compressed_bytes_to_tensor_concurrency = (1, concurrency // 4)
        generate_pseudobulk_concurrency = (1, concurrency)
        generate_regions_concurrency = (1, concurrency // 2)

        dataset = self._read_parquet(folds=folds)

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

        if region_bed is not None:
            dataset = self._generate_regions(
                dataset=dataset,
                bed=region_bed,
                action_keys=region_action_keys,
                max_regions=1,
                concurrency=generate_regions_concurrency,
                pos_resolution=self.pos_resolution,
            )
        return dataset

    def get_processed_dataset(
        self,
        folds: list[int],
        region_bed: str,
        return_cells: bool = False,
        return_regions: bool = True,
        concurrency: int = 16,
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
        # standard_length = self.dna_window
        # region_bed = self.standard_region_length(region_bed, standard_length)

        work_ds = self._get_processed_dataset(
            folds=folds,
            region_bed=region_bed,
            name_to_pseudobulker=self.name_to_pseudobulker,
            n_pseudobulks=self.n_pseudobulks,
            return_rows=return_cells,
            inplace=False,
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

        # add clamp sqrt
        work_ds = self._add_clamp_sqrt(work_ds)
        return work_ds

    def get_dataloader(
        self,
        folds,
        region_bed,
        as_torch=True,
        return_regions=True,
        return_cells=False,
        n_batches=None,
        shuffle_rows=500,
        concurrency=16,
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
        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "folds": folds,  # for borzoi we don't split train/valid/test via chromosomes, so all chromosomes are included
            "region_bed": region_bed,
            "return_cells": return_cells,
            "return_regions": return_regions,
            "concurrency": concurrency,
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



class BorzoiDatasetOnline(RayRegionDataset):
    """Singel cell pseudobulk dataset for Borzoi model."""

    default_config = {
        "bed": "REQUIRED",
        "embeddings_path": "REQUIRED",
        "lora": "REQUIRED",
        "genome": "hg38",
        "bigwig_paths": None,
        "allc_paths": None,
        "batch_size": 2,
        "dna_window": 524288,
        "pos_resolution": 32,
        "reverse_complement": True,
        # "max_jitter": 3,
        "max_jitter": 0,
        "clamp_sqrt_threshold": None,
        "shuffle_files": True,
        "use_borzoi_regions": True,
    }

    def __init__(
        self,
        bed = str,
        # embedding_paths=list[str],
        embeddings_path=None,
        keys = list[str],
        lora=bool,
        genome=str,
        bigwig_paths = None,
        allc_paths = None,
        batch_size: int = 2,
        dna_window: int = 524288,
        pos_resolution: int = 32,
        reverse_complement: bool = True,
        max_jitter: int = 3,
        cov_filter_name: str = None,
        clamp_sqrt_threshold: int = None,
        shuffle_files=False,
        use_borzoi_regions=True,
    ):
    
        super().__init__(
            bed=bed,
            genome=genome,
            standard_length=dna_window,
            batch_size=batch_size,
            use_borzoi_regions=use_borzoi_regions,
            dna=False,
        )

        # ======================
        #      ATAC data setup
        # ======================
        self.bigwig_paths = bigwig_paths
        if  self.bigwig_paths is not None:
            
            
            #dictionary with cell type name and path to embeddings          
            self.bigwig_names_dict = OrderedDict()
            self.bigwig_id_dict = OrderedDict()
            for i, path in enumerate(bigwig_paths):
                cell_type = pathlib.Path(path).name.rsplit('.',1)[0]
                self.bigwig_names_dict[cell_type] = path
                self.bigwig_id_dict[cell_type] = i
            
            assert len(self.bigwig_names_dict) == len(bigwig_paths), 'bw names dict not same length as bw paths'
        # ======================
        #      ALLC data setup
        # ======================
        self.allc_paths = allc_paths
        
        if self.allc_paths is not None:
            self.signal_columns = []
            self.mc_prefix = "allc"
            self.signal_mode = "bp"
            
            #dictionary with cell type name and path to embeddings          
            self.allc_names_dict = OrderedDict()
            self.allc_id_dict = OrderedDict()
            for i, path in enumerate(allc_paths):
                cell_type = path.split('/')[-1].split('.')[0]
                self.allc_names_dict[cell_type] = path
                self.allc_id_dict[cell_type] = i
            
            assert len(self.allc_names_dict) == len(allc_paths), 'allc names dict not same length as bw paths'
        
        # ============================================
        #       Embeddings Data Setup
        # ============================================
        self.embeddings_path = embeddings_path
        adata = anndata.read_h5ad(embeddings_path)
        scvi_embedding = adata.obsm['X_scVI']
        # Create a DataFrame with embeddings and cell types
        df = pd.DataFrame(scvi_embedding, index=adata.obs.index)
        df['cell_type'] = adata.obs['MajorType']
        
        # Group by cell type
        grouped = df.groupby('cell_type').mean()
        
        # Get embedding
        recalculated_embedding = grouped.to_numpy()

        self.leg_map = {item: recalculated_embedding[index] for index, item in enumerate(grouped.index.to_list())}
        
        self.batch_size = batch_size
        self.keys = keys
        
        self.lora = lora    

        # region properties
        self.dna_window = dna_window
        self.signal_window = dna_window
        self.pos_resolution = pos_resolution
        self.max_jitter = max_jitter
        self.reverse_complement = reverse_complement
        self.cov_filter_name = cov_filter_name
        self.clamp_sqrt_threshold = clamp_sqrt_threshold
        self.use_borzoi_regions = use_borzoi_regions


        #embedding properties
        # self.embeddings_df = embeddings_df

        self.borzoi_regions = BorzoiRegions()

        self.atac_coverage_summary = pd.read_csv('/home/tlgallent/projects/finetune_borzoi/cell_type_coverage.csv', index_col=0)

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
            f"BigWig Files: {self.bigiwig_paths}\n"
            f"Embedding Files: {self.embedding_paths}\n"
            f"DNA window: {self.dna_window}, Signal window: {self.signal_window},\n"
            f"Max jitter: {self.max_jitter}, Batch size: {self.batch_size},\n"
        )
        return _str

    def _get_dna_one_hot(self, dataset, concurrency):


        fn = FetchRegionOneHot
        fn_constructor_kwargs = {
            "random_shift": self.max_jitter if self._dataset_mode == "train" else 0,
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

    def _get_folds_dir(self, folds):
        if folds is None:
            fold_dirs = [str(p) for p in pathlib.Path(self.dataset_path).glob("fold*")]
        else:
            if isinstance(folds, str):
                folds = [folds]
            fold_dirs = [f"{self.dataset_path}/fold{fold}" for fold in folds]

            # make sure all fold_dir exists
            fold_dirs = [
                fold_dir for fold_dir in fold_dirs if pathlib.Path(fold_dir).exists()
            ]
            assert (
                len(fold_dirs) > 0
            ), f"None of the fold {folds} exists in {self.dataset_path}"
        return fold_dirs


    def _add_clamp_sqrt(self, dataset):
        if self.clamp_sqrt_threshold is None:
            return dataset

        signal_columns = self.signal_columns
        threshold = self.clamp_sqrt_threshold

        def _oprator(batch):
            for key in signal_columns:
                batch[key] = clamp_sqrt_large_value(batch[key], threshold=threshold)
            return batch

        dataset = dataset.map_batches(
            fn=_oprator,
            concurrency=(1, 4),
        )
        return dataset

    def _get_bigwig_data(
            self,
            dataset,
            data_key="bw_values",
            concurrency=(1, 6),
            n_operators=5,
            batch_size=8,
            norm_mode=None,
            resolution=32,
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
            _chunk_size = max(1, len(self.bigwig_paths) // n_operators)
            # breakpoint()
            for idx, chunk_start in enumerate(
                range(0, len(self.bigwig_paths), _chunk_size)
            ):
                chunk_end = min(len(self.bigwig_paths), chunk_start + _chunk_size)
                chunk_paths = self.bigwig_paths[chunk_start:chunk_end]


                #gets the signal into bins of 32 and adds cell type embedding info
                fn = FetchRegionBigWigsReduced 
                fn_constructor_kwargs = {
                    "bw_paths": chunk_paths,
                    "region_key": "region", #this is what column from the dataframe is acted on by fn 
                    "data_key": f"{data_key}_{idx}",
                    "norm_mode": norm_mode,
                    "resolution": resolution,
                }
                dataset = dataset.map_batches(
                    fn=fn,
                    fn_constructor_kwargs=fn_constructor_kwargs,
                    concurrency=concurrency,
                    batch_size=batch_size,
                )

            

            total_chunks = idx + 1

            def _concat_bw_chunks(data, data_key=data_key, total_chunks=total_chunks):
                bw_keys = [f"{data_key}_{idx}" for idx in range(total_chunks)]
                bw_data = [data.pop(key) for key in bw_keys if key in data]
                if not bw_data:
                    raise ValueError("No bigwig data found to concatenate.")
                data[data_key] = np.concatenate(bw_data, axis=1)
                return data            

            dataset = dataset.map_batches(
                fn=_concat_bw_chunks,
                batch_size=batch_size,
            )
            return dataset       


    def _get_allc_data(self, dataset, concurrency=(1, 6), n_oprators=5, batch_size=8):
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
        _chunk_size = max(1, len(self.allc_paths) // n_oprators)

        for idx, chunk_start in enumerate(range(0, len(self.allc_paths), _chunk_size)):
            chunk_end = min(len(self.allc_paths), chunk_start + _chunk_size)
            chunk_paths = self.allc_paths[chunk_start:chunk_end]

            fn = FetchRegionALLCsReduced
            fn_constructor_kwargs = {
                "allc_paths": chunk_paths,
                "data_prefix": f"{self.mc_prefix}_",
                "data_suffix": f"_{idx}",
                "region_key": "region",
                "mode": self.signal_mode,
            }
            dataset = dataset.map_batches(
                fn=fn,
                fn_constructor_kwargs=fn_constructor_kwargs,
                concurrency=concurrency,
                batch_size=batch_size,
            )
        total_chunks = idx + 1

        # add a final concat function to merge all the chunks
        def _concat_allc_chunks(data):
            for key in ["mc", "cov"]:
                allc_keys = [
                    f"{self.mc_prefix}_{key}_{idx}" for idx in range(total_chunks)
                ]
                allc_data = [data.pop(key) for key in allc_keys]
                data[f"{self.mc_prefix}_{key}"] = np.concatenate(allc_data, axis=1)
            return data

        dataset = dataset.map_batches(
            fn=_concat_allc_chunks,
            batch_size=batch_size,
        )

        for key in ["mc", "cov"]:
            if f"{self.mc_prefix}_{key}" not in self.signal_columns:
                self.signal_columns.append(f"{self.mc_prefix}_{key}")
        return dataset


    def _get_mc_frac(self, dataset):
        # calculate mC fraction
        def _mc_frac(data_dict):
            mc = data_dict[f"{self.mc_prefix}_mc"]
            cov = data_dict[f"{self.mc_prefix}_cov"]
            data_dict[f"{self.mc_prefix}_mc_frac"] = mc / (cov + 1e-6)
            return data_dict

        dataset = dataset.map_batches(_mc_frac)

        # add the data key to the signal columns so later crop function can work
        # Check if the string is not already in the list
        data_key = f"{self.mc_prefix}_mc_frac"
        if data_key not in self.signal_columns:
            self.signal_columns.append(data_key)
        return dataset

    def _get_processed_dataset(
        self,
        folds,
        region_bed,
        data_key,
        concurrency=32,
    ) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """

        

        #gets the bed in dataframe with ray (region_bed has been determined using train_regions for example)        
        dataset = super().get_processed_dataset(bed=region_bed) #comes directly preprocessed as dataframe for fold split we're using 


        if data_key == 'bw_values':
            bw_concurrency = (1, concurrency // 4)
            #Get the ATAC signals, add them to the dataset under 'bw_values' key.
            dataset = self._get_bigwig_data(dataset, concurrency=bw_concurrency, norm_mode=None, resolution=self.pos_resolution)
        

        elif data_key == 'allc_values':
            concurrency_allc = (1, concurrency // 4)
            dataset = self._get_allc_data(dataset, concurrency=concurrency_allc)
            dataset = self._get_mc_frac(dataset)
        
        else:
            raise NotImplementedError
        
        return dataset
    
    def get_processed_dataset(
        self,
        folds,
        region_bed: str,
        signal_columns: str,
        return_regions: bool = True,
        concurrency: int = 32,
    ) -> None:
        """
        Process the dataset and return the processed dataset.

        Parameters
        ----------
        - region_bed_path (str): Path to the BED file containing the regions.
        - return_cells (bool): Whether to return the cells in the dataset. Default is False.
        - return_regions (bool): Whether to return the regions in the dataset. Default is False.

        Returns
        -------
        - work_ds (Dataset): The processed dataset.

        """
        # standard_length = self.dna_window
        # region_bed = self.standard_region_length(region_bed, standard_length)
        
        #

        if signal_columns == 'allc_values':
            self.signal_columns = []
        
        elif signal_columns == 'bw_values':
            self.signal_columns = signal_columns

        work_ds = self._get_processed_dataset(
            folds=folds,
            region_bed=region_bed,
            concurrency=concurrency, 
            data_key=signal_columns,
        )

        # add dna one hot, add bool datatype
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

        # add clamp sqrt
        # work_ds = self._add_clamp_sqrt(work_ds) 


        #sample size by cell type by position

        work_ds = self._convert_to_list_dict(work_ds, data_key=signal_columns) 

        return work_ds

    def get_dataloader(
        self,
        folds,
        region_bed,
        signal_columns='allc_values',
        as_torch=True,
        return_regions=True,
        n_batches=None,
        concurrency=20,
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

        #the region bed is already preprocessed for folds but we want to make sure names are corerct
        #for rename
        region_bed["Chromosome"] = region_bed["Chromosome"].astype(str)
        region_bed.rename(columns={"Name": "region"}, inplace=True)
        self.bed = region_bed
        
        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "folds": folds,
            "region_bed": region_bed,
            "signal_columns": signal_columns,
            "return_regions": return_regions,
            "concurrency": concurrency,
        }
        data_iter_kwargs = dataloader_kwargs

        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            as_torch=as_torch,
            n_batches=n_batches,
            batch_size=self.batch_size,
        )


        return loader

    def _convert_to_list_dict(
            self,
            dataset,
            dna_key="dna_one_hot",
            data_key="bw_values", # "allc_values"
            concurrency=32,
        ):
            """
            Convert the data to list of dict.
            """

            

            def _convert_data(data_dict):

                list_data_dict = []
                if data_key == 'bw_values':

                    names_dict = self.bigwig_names_dict
                    id_dict = self.bigwig_id_dict
                    
                elif data_key == 'allc_values':
                    names_dict = self.allc_names_dict
                    id_dict = self.allc_id_dict

                for i, cell_type_id in enumerate(names_dict): #enumerate bw list
                    new_data_dict = OrderedDict()
                    new_data_dict['cell_type_embedding'] = self.leg_map[cell_type_id] #puts in embeddings
                    new_data_dict['cell_type_id'] = id_dict[cell_type_id] #puts in corresponding int id
                    

                    if data_key == 'bw_values':
                        # scaling_factor = self.atac_coverage_summary.loc[cell_type_id, 'total_coverage'] / 10**7 #per 10 M reads
                        # new_data_dict[data_key] = data_dict[data_key][i,:] / scaling_factor #this only works because bw names are enumerated in order of cell type
                        new_data_dict[data_key] = data_dict[data_key][i,:]
                        new_data_dict[dna_key] = data_dict[dna_key]


                        for k in data_dict.keys():

                            if k != data_key:
        
                                new_data_dict[k] = data_dict[k]

                        list_data_dict.append(new_data_dict)

                    elif data_key == 'allc_values':
                        # print(f'Shape of signal: {data_dict[f"{self.mc_prefix}_mc_frac"][i,:].shape}')
                        # new_data_dict[f"{self.mc_prefix}_mc"] = data_dict[f"{self.mc_prefix}_mc"][i,:]
                        # new_data_dict[f"{self.mc_prefix}_cov"] = data_dict[f"{self.mc_prefix}_cov"][i,:]
                        new_data_dict[f"{self.mc_prefix}_mc_frac"] = data_dict[f"{self.mc_prefix}_mc_frac"][i,:]
                        
                        new_data_dict[dna_key] = data_dict[dna_key]


                        for k in data_dict.keys():
                            if k not in [f"{self.mc_prefix}_mc", f"{self.mc_prefix}_cov", f"{self.mc_prefix}_mc_frac"]:
                                new_data_dict[k] = data_dict[k]

                        list_data_dict.append(new_data_dict)
                    else:
                        raise  NotImplementedError
                     

                return list_data_dict

            dataset = dataset.flat_map(
                fn=_convert_data,
                concurrency=concurrency,
            )
            
            return dataset
    