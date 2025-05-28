import re
from functools import partial

import numpy as np
import pandas as pd
import torch
from einops import einsum, rearrange
from torchmetrics import Metric, PearsonCorrCoef, R2Score
from torchmetrics.functional import pearson_corrcoef, r2_score

from bolero.utils import understand_regions


class MetricCallback:
    def __init__(
        self,
        metric_cls: Metric,
        output_key: str,
        ytrue_key: str = "__ytrue__",
        ypred_key: str = "__ypred__",
        permute: tuple[int] | None = None,
        numpy: bool = True,
        cumulative: bool = False,
        **metric_kwargs,
    ):
        """
        Calculate the metrics along the first dimension.

        Parameters
        ----------
        metric_cls : Metric
            The metric class to be used. It should be a subclass of torchmetrics.Metric.
        output_key : str
            The key for the output in the batch dictionary.
        ytrue_key : str
            The key for the true values in the batch dictionary.
        ypred_key : str
            The key for the predicted values in the batch dictionary.
        permute : tuple[int] | None
            If the dimention of interest is not the first dimension, use this to permute the dimensions.
        numpy : bool
            If True, return the result as a numpy array. Default is False (returns a torch tensor).
        cumulative : bool
            If True, calculate the metric cumulatively. Default is False (calculate the metric for each batch).
        metric_kwargs : dict
            Additional keyword arguments to be passed to the metric class.
        """
        self.output_key: str = output_key
        self.ytrue_key: str = ytrue_key
        self.ypred_key: str = ypred_key
        self.permute: tuple[int] | None = permute
        self.numpy: bool = numpy
        self.cumulative: bool = cumulative

        self.metric: Metric | None = None
        self.metric_cls = metric_cls
        self._metric_kwargs = metric_kwargs
        self._output_shape: tuple[int] | None = None
        self.first_call: bool = True

    @property
    def _functional_metric(self):
        raise NotImplementedError(
            "The functional metric should be defined in the subclass."
        )

    def _create_metric_if_none(self, *args, **kwargs):
        if self.metric is None:
            if self.cumulative:
                # For cumulative metric, use the metric class which saves the state
                self.metric = self.metric_cls(**self._metric_kwargs)
            else:
                # For non-cumulative metric, use the functional metric
                self.metric = self._functional_metric
        return

    def to(self, device) -> "MetricCallback":
        """Move the metric to the specified device."""
        if hasattr(self.metric, "to"):
            self.metric.to(device)
        return

    def _compute(self, ypred=None, ytrue=None) -> torch.Tensor | np.ndarray:
        """Compute the metric."""
        if self.cumulative:
            score: torch.Tensor = self.metric.compute().to(torch.float32)
        else:
            score: torch.Tensor = self.metric(ypred, ytrue).to(torch.float32)

        if self._output_shape is not None:
            score = score.reshape(*self._output_shape)

        if self.numpy:
            score = score.cpu().numpy()
        return score

    def __call__(self, batch: dict) -> dict:
        """Calculate the Pearson correlation coefficient."""
        self._create_metric_if_none(batch)

        ytrue: torch.Tensor = batch[self.ytrue_key]
        ypred: torch.Tensor = batch[self.ypred_key]

        if self.first_call:
            self.to(ytrue.device)
            self.first_call = False

        # use permute to put the dimension of interest at the first dimension
        if self.permute is not None:
            ytrue = ytrue.permute(*self.permute)
            ypred = ypred.permute(*self.permute)

        if ytrue.ndim > 2:
            _, *out_shape = ytrue.shape
            ytrue = rearrange(ytrue, "l ... -> l (...)")
            ypred = rearrange(ypred, "l ... -> l (...)")
            self._output_shape = out_shape
        else:
            out_shape = None

        ypred = ypred.to(torch.float64)
        ytrue = ytrue.to(torch.float64)

        if self.cumulative:
            # for cumulative metric, we need to update the metric instead of calling it
            self.metric.update(ypred, ytrue)
            # and do nothing to the batch
            return batch

        # calculate the metric within batch
        score = self._compute(ypred, ytrue)
        batch[self.output_key] = score
        return batch

    def compute(self) -> dict:
        """Compute the metric."""
        score = self._compute()

        data_dict = {
            self.output_key: score,
        }
        return data_dict


class PearsonCorrcoefCallback(MetricCallback):
    def __init__(self, **kwargs):
        """
        Calculate the Pearson correlation coefficient along the first dimension.
        """
        default_kwargs = {
            "metric_cls": PearsonCorrCoef,
            "output_key": "pearsonr",
        }
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def _create_metric_if_none(self, batch):
        if self.metric is None:
            if self.cumulative:
                # init metric shape with example
                ytrue: torch.Tensor = batch[self.ytrue_key]
                if self.permute is not None:
                    ytrue = ytrue.permute(*self.permute)
                num_outputs = ytrue[0].shape.numel()
                self._metric_kwargs["num_outputs"] = num_outputs
                self.metric = self.metric_cls(**self._metric_kwargs)
            else:
                self.metric = self._functional_metric
        return

    @property
    def _functional_metric(self):
        return pearson_corrcoef


class R2ScoreCallback(MetricCallback):
    def __init__(self, **kwargs):
        """
        Calculate the R2 score along the first dimension.
        """
        default_kwargs = {
            "metric_cls": R2Score,
            "output_key": "r2_score",
            "multioutput": "raw_values",
        }
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    @property
    def _functional_metric(self):
        return partial(
            r2_score,
            multioutput=self._metric_kwargs.get("multioutput", "raw_values"),
            adjusted=self._metric_kwargs.get("adjusted", 0),
        )


class PeakDataSummary:
    def __init__(
        self,
        peak_bed: pd.DataFrame,
        data_keys: list[str] = ("__ytrue__", "__ypred__"),
        region_key="region",
        suffix="peak",
        resolution=32,
        seq_len=16352,
    ):
        self.data_keys = list(data_keys)
        self.region_key = region_key
        self.resolution = resolution
        self.seq_len = seq_len
        self.seq_len_bp = seq_len * resolution

        assert isinstance(
            peak_bed, pd.DataFrame
        ), "peak_bed should be a pandas DataFrame"
        if "Name" not in peak_bed.columns:
            peak_bed["Name"] = (
                peak_bed["Chromosome"]
                + ":"
                + peak_bed["Start"].astype(str)
                + "-"
                + peak_bed["End"].astype(str)
            )
        self.peak_bed: pd.DataFrame = peak_bed
        self.suffix = suffix
        return

    def _get_peak_and_mask(self, chrom, start, end):
        features = self.peak_bed.query(
            f"Chromosome == '{chrom}' and Start >= {start} and End <= {end}"
        ).copy()

        total_bins = self.seq_len_bp // self.resolution
        features["StartBin"] = (features["Start"] - start) // self.resolution
        features["StartBin"] = features["StartBin"].clip(0, total_bins)
        features["EndBin"] = (features["End"] - start) // self.resolution
        features["EndBin"] = features["EndBin"].clip(0, total_bins)

        feature_mask = np.zeros((features.shape[0], total_bins), dtype="bool")
        for i, (start, end) in enumerate(features[["StartBin", "EndBin"]].values):
            feature_mask[i, start:end] = True
        return features, feature_mask

    def get_region_features_and_masks(self, regions: pd.DataFrame):
        """Get features and masks for each region"""
        peaks_col = []
        mask_col = []
        for _, (chrom, start, end, *_) in regions.iterrows():
            peaks, mask = self._get_peak_and_mask(chrom, start, end)
            mask_col.append(mask)
            peaks_col.append(peaks)
        peaks_col = pd.concat(peaks_col)
        return peaks_col, mask_col

    def get_regions(self, data_dict):
        """Get regions from data_dict"""
        region_df = understand_regions(data_dict[self.region_key])
        crop_adjust = ((region_df["End"] - region_df["Start"]) - self.seq_len_bp) // 2
        region_df["Start"] += crop_adjust
        region_df["End"] -= crop_adjust
        return region_df

    def get_feature_level_data(
        self, data: torch.Tensor, feature_masks: torch.Tensor
    ) -> torch.Tensor:
        """Get feature level data from raw data"""
        result = einsum(data, feature_masks, "c s, f s -> c f")
        return result

    def __call__(self, data_dict: dict):
        """Summarize peak data"""
        # prepare region mask
        regions = self.get_regions(data_dict)
        features, feature_masks = self.get_region_features_and_masks(regions)

        _data = data_dict[self.data_keys[0]]
        feature_masks = [
            torch.from_numpy(m).to(device=_data.device, dtype=_data.dtype)
            for m in feature_masks
        ]

        suffix = self.suffix
        feature_data_dict = {suffix: features}
        for key in self.data_keys:
            data = data_dict[key]
            feture_data_col = []
            for _data, mask in zip(data, feature_masks):
                feature_data = self.get_feature_level_data(_data, mask)
                feture_data_col.append(feature_data)
            feture_data_col = torch.concat(feture_data_col, dim=-1)
            feature_data_dict[f"{key}:{suffix}"] = feture_data_col
        data_dict.update(feature_data_dict)
        return data_dict


class ProcessPairedData:
    def __init__(self, data_keys: list[str] = ("__ytrue__", "__ypred__"), split_dim=0):
        """
        This class will understand the matching pseudobulk pairs from two conditions through the pseudobulk_ids;
        and calculate the delta between the each pseudobulk pair for both true and predicted values.

        The paired data is expected to have pseudobulk_ids in the format:
        cond0|cond1:cond-idxN, where cond0 and cond1 are the conditions, cond is the condition of the data,
        and idx is the index of the pairs.

        Parameters
        ----------
        data_keys : list[str]
            The keys for the true and predicted values in the batch dictionary.
            Default is ["__ytrue__", "__ypred__"].
        output_suffix : str
            The suffix to be added to the output keys for the delta values.
            Default is "delta".
        """
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        self.data_keys: list[str] = data_keys
        if isinstance(split_dim, int):
            split_dim = [split_dim for _ in range(len(data_keys))]
        assert (
            len(split_dim) == len(data_keys)
        ), "split_dim should be an int or a list of ints with the same length as data_keys."
        self.split_dims: list[int] = split_dim

        pattern = r"(?P<cond0>[^|]+)\|(?P<cond1>[^:]+):(?P<cond>[^-]+)-(?P<idx>\d+)"
        self.paired_pid_pattern = re.compile(pattern)
        self._pid_table = None

    def _make_or_check_pid_table(self, batch: dict) -> pd.DataFrame:
        """
        Make or check the pid table in the batch.
        """
        if self._pid_table is None:
            # first batch, make the pid table
            all_pids = pd.Index(batch["pseudobulk_ids"])

            pid_table = []
            for pid in all_pids:
                match = self.paired_pid_pattern.fullmatch(pid)
                if match is None:
                    raise ValueError(
                        f"Can not parse pid: {pid} with pattern {self.paired_pid_pattern.pattern}"
                    )
                result = match.groupdict()
                result["pid"] = pid
                pid_table.append(result)
            pid_table = pd.DataFrame(pid_table).set_index("pid")
            pid_table["idx"] = pid_table["idx"].astype(int)
            self._pid_table = pid_table
        else:
            # check the pid table
            all_pids = pd.Index(batch["pseudobulk_ids"])
            if not all_pids.equals(self._pid_table.index):
                raise ValueError(
                    "The pseudobulk_ids in the batch do not match the pid table."
                )
        return self._pid_table

    @staticmethod
    def _bool_sel_at_dim(dim, tensor, bool_sel):
        """Select data with a boolean mask at the specified dimension."""
        if dim == 0:
            return tensor[bool_sel]
        elif dim == -1:
            return tensor[..., bool_sel]
        else:
            sel = [slice(None)] * tensor.ndim
            sel[dim] = bool_sel
            return tensor[tuple(sel)]

    def __call__(self, batch: dict) -> dict:
        """
        Calculate the delta between the true and predicted values for each paired condition.
        """
        pid_table = self._make_or_check_pid_table(batch)

        for data_key, split_dim in zip(self.data_keys, self.split_dims):
            all_pids = pid_table.index

            cond0_col = []
            cond1_col = []
            cond_pairs = []
            for (cond0, cond1), cond_df in pid_table.groupby(["cond0", "cond1"]):
                # select the pids for the two conditions
                cond0_pids = cond_df[cond_df["cond"] == cond0].sort_values("idx").index
                cond1_pids = cond_df[cond_df["cond"] == cond1].sort_values("idx").index
                # save cond data separately
                data_cond0 = self._bool_sel_at_dim(
                    split_dim, batch[data_key], all_pids.isin(cond0_pids)
                )
                data_cond1 = self._bool_sel_at_dim(
                    split_dim, batch[data_key], all_pids.isin(cond1_pids)
                )
                cond0_col.append(data_cond0)
                cond1_col.append(data_cond1)
                # record the condition pairs
                cond_pairs.extend([[cond0, cond1]] * len(data_cond0))

            batch[f"{data_key}:cond0"] = torch.concatenate(cond0_col)
            batch[f"{data_key}:cond1"] = torch.concatenate(cond1_col)
            batch[f"{data_key}:delta"] = (
                batch[f"{data_key}:cond1"] - batch[f"{data_key}:cond0"]
            )
        batch["condition_pairs"] = np.array(cond_pairs)
        return batch


class Rename:
    def __init__(self, name_map: dict[str, str]):
        """
        Rename keys in the batch dictionary.

        Parameters
        ----------
        name_map : dict[str, str]
            A mapping from old keys to new keys.
        """
        self.name_map = name_map

    def __call__(self, batch: dict) -> dict:
        """
        Rename the keys in the batch dictionary according to the name_map.
        """
        for old_key, new_key in self.name_map.items():
            if old_key in batch:
                batch[new_key] = batch.pop(old_key)
        return batch


CALLBACK_NAME_TO_CLASS = {
    "pearsonr": PearsonCorrcoefCallback,
    "r2_score": R2ScoreCallback,
    "extract_peak": PeakDataSummary,
    "process_paired_data": ProcessPairedData,
    "rename": Rename,
}
