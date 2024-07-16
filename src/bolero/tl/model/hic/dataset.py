import pathlib

import numpy as np

from bolero.tl.dataset.file_transforms import FetchRegionCools
from bolero.tl.dataset.ray_dataset import RayRegionDataset


class HiCTrackDataset(RayRegionDataset):
    """Single cell dataset for cell-by-meta-region data."""

    default_config = {
        "cool_paths": "REQUIRED",
        "resolution": "REQUIRED",
        "balance": False,
    }

    def __init__(
        self,
        cool_paths,
        resolution,
        bed,
        genome,
        standard_length,
        cool_names=None,
        balance=False,
        dna=False,
        boarder_strategy="drop",
        remove_blacklist=False,
    ) -> None:
        """
        Initialize the HiCTrackDataset.
        """
        super().__init__(
            bed=bed,
            genome=genome,
            standard_length=standard_length,
            dna=dna,
            boarder_strategy=boarder_strategy,
            remove_blacklist=remove_blacklist,
        )

        self.cool_paths = cool_paths
        if cool_names is None:
            cool_names = [pathlib.Path(path).name for path in cool_paths]
        else:
            self.cool_names = cool_names
        assert len(cool_paths) == len(cool_names)

        self.resolution = resolution
        self.balance = balance

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

    def get_processed_dataset(self, chroms):
        """
        Get the processed dataset with many oprators applied.
        """
        # if multiple oprator is used, decrease the max concurrency to allow them parallel evenly
        concurrency_cool = (1, 6)

        dataset = super().get_processed_dataset(
            chroms=chroms,
        )

        dataset = self._get_cool_data(dataset, concurrency=concurrency_cool)

        return dataset
