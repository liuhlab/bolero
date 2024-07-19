from collections import defaultdict

import numpy as np

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


class SplitRegionTomCSite:
    def __init__(
        self,
        prefix: str = "allc",
        hypo_frac_cutoff: float = 0.8,
        cov_cutoff: int = 10,
        hypo_ratio: float = 1,
        hyper_ratio: float = 0.2,
        max_site_per_region: int = 3,
        dna_radius: int = 920,
    ) -> None:
        """
        Split region to mC site.

        Parameters
        ----------
        prefix : str, optional
            Prefix for the data columns, by default "allc"
        hypo_frac_cutoff : float, optional
            Hypomethylation fraction cutoff, sites with fraction less than this value will be selected with probability hypo_ratio,
            sites with fraction greater than this value will be selected with probability hyper_ratio,
            by default 0.8
        cov_cutoff : int, optional
            Coverage cutoff, site with coverage less than this value will be filtered out, by default 10
        hypo_ratio : float, optional
            Hypomethylation ratio, probability of selecting a hypomethylated site, by default 1
        hyper_ratio : float, optional
            Hypermethylation ratio, probability of selecting a hypermethylated site, by default 0.2
        max_site_per_region : int, optional
            Maximum number of sites per region, the final number of sites selected
            from each batch will be approximately max_site_per_region * batch_size, by default 3
        dna_radius : int, optional
            DNA radius, by default 920
        """
        self.prefix = prefix
        self.hypo_frac_cutoff = hypo_frac_cutoff
        self.cov_cutoff = cov_cutoff
        self.hypo_ratio = hypo_ratio
        self.hyper_ratio = hyper_ratio
        self.max_site_per_region = max_site_per_region
        self.dna_radius = dna_radius

    def __call__(self, data_dict: dict) -> dict:
        """
        Split region to mC site.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data columns

        Returns
        -------
        dict
            Dictionary containing the split mC site data columns
        """
        prefix = self.prefix
        dna_one_hot = data_dict["dna_one_hot"]
        mc_frac = data_dict[f"{prefix}_mc_frac"]
        cov = data_dict[f"{prefix}_cov"]
        mc = data_dict[f"{prefix}_mc"]

        # self.dna_length           ---------c----------
        # self.signal_length           ------f------
        # dna_pos_offset            --- # half delta length between DNA and mC frac in bp
        dna_pos_offset = (dna_one_hot.shape[-1] - mc_frac.shape[-1]) // 2
        assert dna_pos_offset > 0

        # mc fraction filter & cov filter & random downsample
        hypo_site = (
            (mc_frac < self.hypo_frac_cutoff)
            & (cov > self.cov_cutoff)
            & (np.random.rand(*cov.shape) < self.hypo_ratio)
        )
        hypo_site = hypo_site.any(axis=1)
        hyper_site = (
            (mc_frac > self.hypo_frac_cutoff)
            & (cov > self.cov_cutoff)
            & (np.random.rand(*cov.shape) < self.hyper_ratio)
        )
        hyper_site = hyper_site.any(axis=1)
        # combine hypo and hyer sel
        final_site = hypo_site | hyper_site

        # final downsample, in order to prevent too many sites in one region
        # apply a final downsample to select max_site
        # which approximately select max_site_per_region * n_region final sites
        max_site = final_site.shape[0] * self.max_site_per_region
        final_downsample_ratio = max_site / final_site.sum()
        if final_downsample_ratio < 1:
            downsample_sel = np.random.rand(*final_site.shape) < final_downsample_ratio
        final_site *= downsample_sel

        data_col = defaultdict(list)
        for region_site, region_mc, region_cov, region_frac, region_onehot in zip(
            final_site, mc, cov, mc_frac, dna_one_hot
        ):
            for pos in np.where(region_site)[0]:
                pos_mc = region_mc[:, pos]
                pos_cov = region_cov[:, pos]
                pos_frac = region_frac[:, pos]

                dna_pos = pos + dna_pos_offset
                pos_dna_onehot = region_onehot[
                    :, dna_pos - self.dna_radius : dna_pos + self.dna_radius
                ]

                data_col["allc_mc"].append(pos_mc)
                data_col["allc_cov"].append(pos_cov)
                data_col["allc_frac"].append(pos_frac)
                data_col["dna_one_hot"].append(pos_dna_onehot)
        try:
            data_col = {k: np.array(v) for k, v in data_col.items()}
        except ValueError:
            print("Error in SplitRegionTomCSite")
            print(data_col.keys())
            print("allc_mc", [v.shape for v in data_col["allc_mc"]])
            print("allc_cov", [v.shape for v in data_col["allc_cov"]])
            print("allc_frac", [v.shape for v in data_col["allc_frac"]])
            print("dna_one_hot", [v.shape for v in data_col["dna_one_hot"]])
        return data_col


class mCSiteDataset(Track1DDataset):
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

        self._site_dna_radius = self.dna_length // 2
        self.dna_length = self.dna_length + self.signal_length + 2

    def _get_mc_frac(self, dataset):
        # calculate mC fraction
        def _mc_frac(data_dict):
            mc = data_dict[f"{self.prefix}_mc"]
            cov = data_dict[f"{self.prefix}_cov"]
            data_dict[f"{self.prefix}_mc_frac"] = mc / (cov + 1e-6)
            return data_dict

        dataset = dataset.map_batches(_mc_frac)
        return dataset

    def _split_region_to_site(self, dataset):
        fn = SplitRegionTomCSite
        fn_constructor_kwargs = {
            "prefix": self.prefix,
            "hypo_frac_cutoff": 0.8,
            "cov_cutoff": 10,
            "hypo_ratio": 1,
            "hyper_ratio": 0.2,
            "max_site_per_region": 3,
            "dna_radius": self._site_dna_radius,
        }
        dataset = dataset.map_batches(
            fn=fn, fn_constructor_kwargs=fn_constructor_kwargs, concurrency=(1, 4)
        )
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
        dataset = self._split_region_to_site(dataset)
        return dataset
