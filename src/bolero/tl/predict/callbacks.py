import numpy as np
import pandas as pd
import torch
from einops import einsum, rearrange
from torchmetrics import Metric, PearsonCorrCoef, R2Score
from torchmetrics.functional import pearson_corrcoef

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
        _init_metric_with_example: bool = False,
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
        if not _init_metric_with_example:
            self.metric: Metric = metric_cls(**metric_kwargs)
        self._output_shape: tuple[int] | None = None
        self.first_call: bool = True

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
            # pearsonr class needs an example to set num_outputs
            "_init_metric_with_example": True,
        }
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def __call__(self, batch: dict) -> dict:
        """Init metric with example, then calculate pearsonr."""
        if self.cumulative:
            if self.metric is None:
                # init metric with example
                ytrue: torch.Tensor = batch[self.ytrue_key]
                if self.permute is not None:
                    ytrue = ytrue.permute(*self.permute)
                num_outputs = ytrue[0].shape.numel()
                self._metric_kwargs["num_outputs"] = num_outputs
                self.metric = self.metric_cls(**self._metric_kwargs)
        else:
            #
            self.metric = pearson_corrcoef

        return super().__call__(batch)


class R2ScoreCallback(MetricCallback):
    def __init__(self, **kwargs):
        """
        Calculate the R2 score along the first dimension.
        """
        default_kwargs = {"metric_cls": R2Score, "output_key": "r2_score"}
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)


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


CALLBACK_NAME_TO_CLASS = {
    "pearsonr": PearsonCorrcoefCallback,
    "r2_score": R2ScoreCallback,
    "extract_peak": PeakDataSummary,
}
