import pathlib
from collections import defaultdict

import numpy as np

from bolero.tl.dataset.file_transforms import FetchRegionALLCs
from bolero.tl.dataset.ray_dataset import RayRegionDataset
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

class mCRegionOnlineDataset(RayRegionDataset):
    """Single cell dataset for cell-by-meta-region data."""

    default_config = {
        "allc_paths": "REQUIRED",
        "bed": "REQUIRED",
        "genome": "REQUIRED",
        "dna_length": "REQUIRED",
        "batch_size": 64,
        "allc_names": None,
        "dna": False,
        "boarder_strategy": "drop",
        "remove_blacklist": False,
        "mc_prefix": "allc",
        "signal_mode": "region", ### bp or region
        "signal_length": "given" ### given or int
        # TODO support any signal length
    }

    def __init__(
        self,
        allc_paths,
        bed,
        genome,
        dna_length,
        batch_size,
        allc_names=None,
        dna=False,
        boarder_strategy="drop",
        remove_blacklist=False,
        mc_prefix="allc",
        signal_mode="region",
        signal_length="given",
    ) -> None:
        """
        Initialize the mCTrackOnlineDataset.
        """
        super().__init__(
            bed=bed,
            genome=genome,
            standard_length=dna_length,
            dna=dna,
            batch_size=batch_size,
            boarder_strategy=boarder_strategy,
            remove_blacklist=remove_blacklist,
            signal_length=signal_length,
        )

        self.mc_prefix = mc_prefix
        self.signal_mode = signal_mode
        # self.signal_length = signal_length
        self.allc_paths = allc_paths
        if allc_names is None:
            allc_names = [pathlib.Path(path).name for path in allc_paths]
        else:
            self.allc_names = allc_names
        assert len(allc_paths) == len(allc_names)

    def _get_allc_data(self, dataset, concurrency=(1, 6), n_oprators=5, batch_size=8):
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
        _chunk_size = max(1, len(self.allc_paths) // n_oprators)

        for idx, chunk_start in enumerate(range(0, len(self.allc_paths), _chunk_size)):
            chunk_end = min(len(self.allc_paths), chunk_start + _chunk_size)
            chunk_paths = self.allc_paths[chunk_start:chunk_end]

            fn = FetchRegionALLCs
            fn_constructor_kwargs = {
                "allc_paths": chunk_paths,
                "data_prefix": f"{self.mc_prefix}_",
                "data_suffix": f"_{idx}",
                "region_key": "Original_Name",
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
        return dataset

    def _get_mc_frac(self, dataset):
        # calculate mC fraction
        def _mc_frac(data_dict):
            mc = data_dict[f"{self.mc_prefix}_mc"]
            cov = data_dict[f"{self.mc_prefix}_cov"]
            data_dict[f"{self.mc_prefix}_mc_frac"] = mc / (cov + 1e-6)
            return data_dict

        dataset = dataset.map_batches(_mc_frac)
        return dataset

    def get_processed_dataset(self, chroms, shuffle_bed=False):
        """
        Get the processed dataset with many oprators applied.
        """
        # if multiple oprator is used, decrease the max concurrency to allow them parallel evenly
        concurrency_allc = (1, 6)

        dataset = super().get_processed_dataset(
            chroms=chroms,
            shuffle_bed=shuffle_bed,
        )

        dataset = self._get_allc_data(dataset, concurrency=concurrency_allc)

        dataset = dataset.drop_columns(["Original_Name", "region"])
        return dataset

    def get_dataloader(self, chroms=None, n_batches=None, batch_size=64, as_torch=False, shuffle_bed=False):
        dataset_kwargs = {
            "chroms": chroms,
            "shuffle_bed": shuffle_bed,
        }
        data_iter_kwargs = {}
        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            n_batches=n_batches,
            batch_size=batch_size,
            as_torch=as_torch,
        )
        return loader
    