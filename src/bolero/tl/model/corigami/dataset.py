import pathlib

import numpy as np

from bolero.tl.dataset.file_transforms import (
    AddGaussianNoise,
    FetchRegionBigWigs,
    FetchRegionCools,
    ReverseCompHicData,
)
from bolero.tl.dataset.ray_dataset import RayRegionDataset


class HiCTrackDataset(RayRegionDataset):
    """Single cell dataset for cell-by-meta-region data."""

    default_config = {
        "cool_paths": "REQUIRED",
        "atac_paths": "REQUIRED",
        "ctcf_paths": "REQUIRED",
        "resolution": "REQUIRED",
        "balance": False,
        "genome": "REQUIRED",
        "window_size": "REQUIRED",
        "step": "REQUIRED",
        "batch_size": "REQUIRED",
        "bed": "REQUIRED",
        "standard_length": "REQUIRED",
        "dna_fifth_channel": False,
        "data_1d_keys": "REQUIRED",
    }

    def __init__(
        self,
        bed,
        standard_length,
        cool_paths,
        resolution,
        genome,
        window_size,
        step,
        batch_size,
        data_1d_keys,
        cool_names=None,
        atac_paths=None,
        atac_names=None,
        ctcf_paths=None,
        ctcf_names=None,
        balance=False,
        dna_fifth_channel=False,
        boarder_strategy="drop",
        remove_blacklist=False,
    ) -> None:
        """
        Initialize the HiCTrackDataset.
        """
        super().__init__(
            # ========================
            # these has no effect for hic dataset
            bed=bed,
            standard_length=standard_length,
            # ========================
            genome=genome,
            batch_size=batch_size,
            window_size=window_size,
            step=step,
            # In HiC dataset, we set dna=False first to prevent early fetch of DNA and copy along the pipeline
            # We will manually add DNA after the other online data loading operators are mapped.
            # See the get_processed_dataset method below.
            dna=False,
            boarder_strategy=boarder_strategy,
            remove_blacklist=remove_blacklist,
        )

        # Cooler files
        self.cool_paths = cool_paths
        if cool_names is None:
            cool_names = [pathlib.Path(path).name for path in cool_paths]
        else:
            self.cool_names = cool_names
        assert len(cool_paths) == len(cool_names)

        # Bigwig Files
        self.atac_paths = atac_paths
        if atac_names is None and atac_paths is not None:
            atac_names = [pathlib.Path(path).name for path in atac_paths]
        else:
            self.atac_names = atac_names

        self.ctcf_paths = ctcf_paths
        if ctcf_names is None and ctcf_paths is not None:
            ctcf_names = [pathlib.Path(path).name for path in ctcf_paths]
        else:
            self.ctcf_names = ctcf_names

        self.resolution = resolution
        self.balance = balance
        self.step = step
        self.dna_fifth_channel = dna_fifth_channel
        self.data_1d_keys = data_1d_keys

    def _get_cool_data(
        self,
        dataset,
        data_key="values",
        concurrency=(1, 6),
        n_oprators=5,
        batch_size=8,
        norm_mode="log",
        image_scale=256,
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
        _chunk_size = max(5, len(self.cool_paths) // n_oprators)

        for idx, chunk_start in enumerate(range(0, len(self.cool_paths), _chunk_size)):
            chunk_end = min(len(self.cool_paths), chunk_start + _chunk_size)
            chunk_paths = self.cool_paths[chunk_start:chunk_end]

            fn = FetchRegionCools
            fn_constructor_kwargs = {
                "cool_paths": chunk_paths,
                "resolution": self.resolution,
                "balance": self.balance,
                "data_key": f"{data_key}_{idx}",
                "norm_mode": norm_mode,  # Note: if the data is HBA data, no need to log transform, otherwise, log transform the data
                "image_scale": image_scale,
            }
            dataset = dataset.map_batches(
                fn=fn,
                fn_constructor_kwargs=fn_constructor_kwargs,
                concurrency=concurrency,
                batch_size=batch_size,
            )
        total_chunks = idx + 1

        # add a final concat function to merge all the chunks
        def _concat_cool_chunks(data):
            cool_keys = [f"{data_key}_{idx}" for idx in range(total_chunks)]
            cool_data = [data.pop(key) for key in cool_keys]
            data[data_key] = np.concatenate(cool_data, axis=1)
            return data

        dataset = dataset.map_batches(
            fn=_concat_cool_chunks,
            batch_size=batch_size,
        )
        return dataset

    def _get_bigwig_data(
        self,
        dataset,
        bigwig_paths,
        data_key,
        concurrency=(1, 6),
        n_oprators=5,
        batch_size=8,
        norm_mode="log",
    ):
        """
        Get the bigwig data for the dataset

        Parameters
        ----------
        dataset : RayRegionDataset
            The dataset to be processed.
        data_key : str
            The key to store the bigwig data.
        concurrency : tuple
            The concurrency for the dataset, min and max.
        n_oprators : int
            The number of oprators to be used when dataset contains multiple cool paths.
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
        _chunk_size = max(5, len(bigwig_paths) // n_oprators)

        for idx, chunk_start in enumerate(range(0, len(bigwig_paths), _chunk_size)):
            chunk_end = min(len(bigwig_paths), chunk_start + _chunk_size)
            chunk_paths = bigwig_paths[chunk_start:chunk_end]

            fn = FetchRegionBigWigs
            fn_constructor_kwargs = {
                "bw_paths": chunk_paths,
                "region_key": "region",
                "data_key": f"{data_key}_{idx}",
                "norm_mode": norm_mode,
            }
            dataset = dataset.map_batches(
                fn=fn,
                fn_constructor_kwargs=fn_constructor_kwargs,
                concurrency=concurrency,
                batch_size=batch_size,
            )
        total_chunks = idx + 1

        # add a final concat function to merge all the chunks
        def _concat_bw_chunks(data):
            bw_keys = [f"{data_key}_{idx}" for idx in range(total_chunks)]
            bw_data = [data.pop(key) for key in bw_keys]
            data[data_key] = np.concatenate(bw_data, axis=1)
            return data

        dataset = dataset.map_batches(
            fn=_concat_bw_chunks,
            batch_size=batch_size,
        )
        return dataset

    def _random_shift_bed(self, bed):
        """
        Randomly shift the bed region.
        """
        max_shift_bins = self.step // self.resolution // 2

        if max_shift_bins > 0:
            shift = np.random.randint(-max_shift_bins, max_shift_bins) * self.resolution
            print(f"Shifting bed by {shift} bp")

            bed["Start"] = bed["Start"] + shift
            bed["End"] = bed["End"] + shift
        return bed

    def _select_valid_regions(self, bed):
        # need to confirm region is still valid:
        # start > 0, end < chromosome length
        print(f"Before valid region selection, bed shape: {bed.shape}")
        start_judge = bed["Start"] > 0
        end_judge = bed["End"] < (
            bed["Chromosome"].map(self.genome.chrom_sizes) - self.resolution * 1.01
        )
        bed = bed.loc[start_judge & end_judge].copy()
        print(f"After valid region selection, bed shape: {bed.shape}")
        return bed

    def _add_gaussian_noise(
        self,
        dataset,
        dna_key="dna_one_hot",
        data_1d_keys=(
            "atac",
            "ctcf",
        ),
        std=0.1,
        concurrency=(1, 6),
        batch_size=8,
    ):
        """
        Add Gaussian noise to the dataset.
        """
        # Need to update the std when providing different dataset
        fn = AddGaussianNoise
        data_keys = (
            [dna_key] + list(data_1d_keys) if data_1d_keys is not None else [dna_key]
        )
        fn_constructor_kwargs = {"data_keys": data_keys, "std": std}
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _add_fifth_dna_channel(
        self, dataset, dna_key="dna_one_hot", concurrency=6, batch_size=8
    ):
        """
        Add a fifth channel representing "N" to the DNA data.
        """

        def _add_fifth_channel(data_dict):
            dna_one_hot = data_dict.pop(dna_key)

            n_bases = np.sum(dna_one_hot, axis=1, keepdims=True) == 0
            n_bases = n_bases.astype(dna_one_hot.dtype)
            data_dict[dna_key] = np.concatenate([dna_one_hot, n_bases], axis=1)
            return data_dict

        dataset = dataset.map_batches(
            fn=_add_fifth_channel,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        return dataset

    def _reverse_comp_hic_data(
        self,
        dataset,
        dna_key="dna_one_hot",
        data_1d_keys=(
            "atac",
            "ctcf",
        ),
        data_2d_keys=("values",),
        chance=0.5,
        concurrency=(1, 6),
        batch_size=8,
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
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        return dataset

    def _add_corigami_dim_shift(
        self,
        dataset,
        data_1d_keys=(
            "atac",
            "ctcf",
        ),
        data_2d_keys=("values",),
        batch_size=8,
        concurrency=6,
    ):
        def _dim_shift(data_dict):
            # DNA data shape: (batch_size, channel, seq_len), already in the correct shape
            for feature in data_2d_keys:
                data_dict[feature] = data_dict[feature][:, 0, :, :]

            if data_1d_keys is not None:
                for feature in data_1d_keys:
                    if feature in data_dict:
                        data_dict[feature] = data_dict[feature][:, 0, :].astype(
                            np.float32
                        )
            return data_dict

        dataset = dataset.map_batches(
            fn=_dim_shift,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        return dataset

    def get_processed_dataset(self, chroms, shuffle_bed, drop_str=True):
        """
        Get the processed dataset with many oprators applied.
        """
        # if multiple oprator is used, decrease the max concurrency to allow them parallel evenly
        max_concurrency = 6

        _bed = self.bed.copy()

        if self.dataset_mode == "train":
            _bed = self._random_shift_bed(_bed)

        _bed = self._select_valid_regions(_bed)

        dataset = super().get_processed_dataset(
            chroms=chroms, shuffle_bed=shuffle_bed, bed=_bed
        )

        dataset = self._get_cool_data(
            dataset, concurrency=(1, int(max_concurrency / 2))
        )

        if self.atac_paths is not None:
            dataset = self._get_bigwig_data(
                dataset,
                bigwig_paths=self.atac_paths,
                data_key="atac",
                concurrency=(1, max_concurrency),
                norm_mode="log",
            )

        if self.ctcf_paths is not None:
            dataset = self._get_bigwig_data(
                dataset,
                bigwig_paths=self.ctcf_paths,
                data_key="ctcf",
                concurrency=(1, max_concurrency),
                norm_mode=None,
            )

        dataset = self._get_dna_one_hot(
            dataset=dataset,
            dtype="float32",
            concurrency=(1, max_concurrency),
            batch_size=8,
        )

        if self.dataset_mode == "train":
            dataset = self._add_gaussian_noise(
                dataset,
                data_1d_keys=self.data_1d_keys,
                concurrency=(1, max_concurrency),
            )
            dataset = self._reverse_comp_hic_data(
                dataset,
                data_1d_keys=self.data_1d_keys,
                concurrency=(1, int(max_concurrency / 2)),
            )

        if self.dna_fifth_channel:
            # add the fifth channel to the DNA data
            # batch[dna_key] shape: (batch_size, 5, seq_len)
            # Must do this AFTER the self._reverse_comp_hic_data step, because ACGTN[::-1] -> NACGT is wrong
            dataset = self._add_fifth_dna_channel(dataset, concurrency=max_concurrency)

        dataset = self._add_corigami_dim_shift(
            dataset, data_1d_keys=self.data_1d_keys, concurrency=max_concurrency
        )

        if drop_str:
            # in order to set as_torch=True, we need to drop the string columns
            dataset = dataset.drop_columns(["region", "Original_Name"])
        return dataset

    def get_dataloader(
        self,
        chroms,
        n_batches=None,
        batch_size=8,
        as_torch=True,
    ):
        """
        Get the dataloader for the dataset.

        Parameters
        ----------
        chroms : list[str]
            The list of chromosomes to include in the dataset.
        n_batches : int
            The number of batches to generate.
        as_torch : bool, default=True
            Whether to return the dataloader whoes data will be in torch tensors.

        Returns
        -------
        DataLoader
            The dataloader for the dataset.
        """
        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "chroms": chroms,
            "shuffle_bed": True if self.dataset_mode == "train" else False,
            "drop_str": as_torch,
        }
        data_iter_kwargs = {}
        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            batch_size=batch_size,
            as_torch=as_torch,
            n_batches=n_batches,
        )
        return loader
