from bolero.tl.dataset.ray_dataset import RayRegionDataset
from bolero.tl.dataset.file_transforms import FetchRegionCools

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
        *args,
        balance=False,
        **kwargs,
    ) -> None:
        """
        Initialize the HiCTrackDataset.
        """
        super().__init__(*args, **kwargs)
        # bed=bins, genome='hg38', standard_length=5000000, dna=True

        self.cool_paths = cool_paths
        self.resolution = resolution
        self.balance = balance

    def _get_cool_data(self, dataset, concurrency=1):

        fn = FetchRegionCools
        fn_constructor_kwargs = {
            "cool_paths": self.cool_paths, 
            "resolution": self.resolution,
            "balance": self.balance,
        }
        dataset = dataset.map_batches(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
            batch_size=64,
        )
        return dataset
        
    def get_processed_dataset(self, chroms):
        """
        Get the processed dataset with many oprators applied.
        """
        concurrency_cool = (1,16)

        # def _cov_func(data):
        #     return data.sum(-1).mean(-1)

        dataset = super().get_processed_dataset(
            chroms=chroms, # region_bed_path=region_bed_path, cov_func=_cov_func
        )

        dataset = self._get_cool_data(dataset, concurrency=concurrency_cool)

        return dataset
