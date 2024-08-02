import pathlib

import numpy as np

from bolero.tl.dataset.file_transforms import FetchRegionBigWigs, FetchRegionCools
from bolero.tl.dataset.ray_dataset import RayRegionDataset


class HiCTrackDataset(RayRegionDataset):
    """Single cell dataset for cell-by-meta-region data."""

    default_config = {
        "cool_paths": "REQUIRED",
        "bigwig_paths": "REQUIRED",
        "resolution": "REQUIRED",
        "balance": False,
        "genome": "REQUIRED",
        "window_size": 5000000,
        "step": 1000000,
        "batch_size": "REQUIRED",
    }

    def __init__(
        self,
        cool_paths,
        resolution,
        genome,
        window_size,
        step,
        batch_size,
        bigwig_paths=None,
        cool_names=None,
        bigwig_names=None,
        balance=False,
        boarder_strategy="drop",
        remove_blacklist=False,
    ) -> None:
        """
        Initialize the HiCTrackDataset.
        """
        super().__init__(
            # ========================
            # these has no effect for hic dataset
            bed=None,
            standard_length=None,
            # ========================
            genome=genome,
            window_size=window_size,
            step=step,
            batch_size=batch_size,
            dna=False,  # do not load dna_one_hot from the RayRegionDataset, we will do it later after loading the cool and bigwig data
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
        self.bigwig_paths = bigwig_paths
        if bigwig_names is None and bigwig_paths is not None:
            bigwig_names = [pathlib.Path(path).name for path in bigwig_paths]
        else:
            self.bigwig_names = bigwig_names

        self.resolution = resolution
        self.balance = balance
        self.step = step

    def _get_cool_data(
        self, dataset, data_key="values", concurrency=(1, 6), n_oprators=5, batch_size=8
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

        Returns
        -------
        dataset : RayRegionDataset
            The dataset with cool data oprator mapped.
        """
        _chunk_size = max(1, len(self.cool_paths) // n_oprators)

        for idx, chunk_start in enumerate(range(0, len(self.cool_paths), _chunk_size)):
            chunk_end = min(len(self.cool_paths), chunk_start + _chunk_size)
            chunk_paths = self.cool_paths[chunk_start:chunk_end]

            fn = FetchRegionCools
            fn_constructor_kwargs = {
                "cool_paths": chunk_paths,
                "resolution": self.resolution,
                "balance": self.balance,
                "data_key": f"{data_key}_{idx}",
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
        data_key="bw_values",
        concurrency=(1, 6),
        n_oprators=5,
        batch_size=8,
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

        Returns
        -------
        dataset : RayRegionDataset
            The dataset with bigwig data oprator mapped.
        """
        _chunk_size = max(1, len(self.bigwig_paths) // n_oprators)

        for idx, chunk_start in enumerate(
            range(0, len(self.bigwig_paths), _chunk_size)
        ):
            chunk_end = min(len(self.bigwig_paths), chunk_start + _chunk_size)
            chunk_paths = self.bigwig_paths[chunk_start:chunk_end]

            fn = FetchRegionBigWigs
            fn_constructor_kwargs = {
                "bw_paths": chunk_paths,
                "region_key": "region",
                "data_key": f"{data_key}_{idx}",
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

        shift = np.random.randint(-max_shift_bins, max_shift_bins) * self.resolution
        bed["Start"] = bed["Start"] + shift
        bed["End"] = bed["End"] + shift

        # need to confirm region is still valid:
        # start > 0, end < chromosome length
        start_judge = bed["Start"] > 0
        end_judge = bed["End"] < bed["Chromosome"].map(self.genome.chrom_sizes)
        bed = bed.loc[start_judge & end_judge].copy()
        return bed

    def get_processed_dataset(self, chroms, shuffle_bed):
        """
        Get the processed dataset with many oprators applied.
        """
        # if multiple oprator is used, decrease the max concurrency to allow them parallel evenly
        concurrency_cool = (1, 6)
        concurrency_bigwig = (1, 6)

        if self.dataset_mode == "train":
            _bed = self._random_shift_bed(self.bed)
        else:
            _bed = self.bed

        dataset = super().get_processed_dataset(
            chroms=chroms, shuffle_bed=shuffle_bed, bed=_bed
        )

        dataset = self._get_cool_data(dataset, concurrency=concurrency_cool)

        dataset = self._get_bigwig_data(dataset, concurrency=concurrency_bigwig)

        return dataset

    def get_dataloader(
        self,
        chroms,
        n_batches=None,
        batch_size=8,
        as_torch=False,
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


def reverse_comp_hic_data_batch(
    data_dict,
    dna_key="dna_one_hot",
    data_1d_keys=("bw_values",),
    data_2d_keys=("values",),
    chance=0.5,
):
    """
    Reverse complement the hic data batch.

    Parameters
    ----------
    data_dict : dict
        The data dictionary to be reversed.
    dna_key : str
        The key for the dna data to be reversed.
    data_1d_keys : list
        The list of 1d data keys to be reversed.
    data_2d_keys : list
        The list of 2d data keys to be reversed.
    chance : float
        The chance to reverse the data.

    Returns
    -------
    data : dict
        The reversed data dictionary.
    """
    _bool = np.random.rand(1)
    if _bool < chance:
        for key in data_1d_keys:
            data_dict[key] = np.flip(data_dict[key], axis=-1)  # -1 flip the sequence
        for key in data_2d_keys:
            data_dict[key] = np.flip(
                data_dict[key], axis=[-1, -2]
            )  # -1 and -2 both filp the sequence, because the data is 2D
        data_dict[dna_key] = np.flip(
            data_dict[dna_key], axis=[-1, -2]
        )  # -1 flip the sequence, -2 flip the base pair (complement)
    return data_dict
