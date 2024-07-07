from bolero.tl.model.track1d.dataset import Track1DDataset


class mCTrackDataset(Track1DDataset):
    """Single cell dataset for cell-by-meta-region data."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the mCTrackDataset.
        """
        super().__init__(*args, **kwargs)
        self._cov_filter_key = f"{self.prefix}_cov"
        self.signal_columns = [f"{self.prefix}_mc", f"{self.prefix}_cov"]

    def _get_mc_frac(self, dataset):
        # calculate mC fraction
        def _mc_frac(data_dict):
            mc = data_dict[f"{self.prefix}_mc"]
            cov = data_dict[f"{self.prefix}_cov"]
            data_dict[f"{self.prefix}_mc_frac"] = mc / (cov + 1e-6)
            return data_dict

        dataset = dataset.map_batches(_mc_frac)
        return dataset

    def get_processed_dataset(self, chroms, region_bed_path) -> None:
        """
        Get the processed dataset with many oprators applied.
        """

        def _cov_func(data):
            return data.sum(-1).mean(-1)

        dataset = super().get_processed_dataset(
            chroms=chroms, region_bed_path=region_bed_path, cov_func=_cov_func
        )

        dataset = self._get_mc_frac(dataset)
        return dataset
