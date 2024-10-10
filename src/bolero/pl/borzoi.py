import matplotlib.pyplot as plt
import numpy as np
import torch

from bolero import Genome
from bolero.tl.model.borzoi.utils import clamp_sqrt_large_value

from .utils import figure_to_array


class BorzoiExamplePlotter:
    def __init__(
        self,
        genome: Genome,
        zoomin_radius: int = 1000,
        true_key="true_data",
        pred_key="pred_data",
        id_key="sample_id",
        power=0.75,
        threshold=200,
    ):
        self.genome = genome
        self.zoomin_radius = zoomin_radius
        self.true_key = true_key
        self.pred_key = pred_key
        self.id_key = id_key
        self.power = power
        self.threshold = threshold
        return

    def plot(self, batch, channel=0, nrows=2, return_array=False, soft_clamp=False):
        """Plot the true and predicted data for a batch of examples."""
        y_true = batch[self.true_key]
        y_pred = batch[self.pred_key]

        if soft_clamp:
            y_true = clamp_sqrt_large_value(
                y_true, power=self.power, threshold=self.threshold
            )
            y_pred = clamp_sqrt_large_value(
                y_pred, power=self.power, threshold=self.threshold
            )

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        sample_ids = batch[self.id_key]

        regions = batch["region"]
        if isinstance(regions, torch.Tensor):
            regions = regions.cpu().numpy()
        regions = self.genome.parse_global_coords(regions)

        row_ids = list(range(nrows))
        fig, axes = plt.subplots(
            figsize=(8, 2.25 * len(row_ids)),
            dpi=150,
            nrows=4 * len(row_ids),
            constrained_layout=True,
        )

        for i, row_id in enumerate(row_ids):
            row_axes = axes[i * 4 : (i + 1) * 4]
            true_data = y_true[row_id, channel, :]
            pred_data = y_pred[row_id, channel, :]
            sample_id = sample_ids[row_id]

            chrom, start, end, *_ = regions.iloc[row_id]
            region = f"{chrom}:{start}-{end}"
            self._plot_single_region(row_axes, true_data, pred_data, region, sample_id)

        for ax in axes.flat:
            ax.tick_params(axis="both", labelsize=8)

        if return_array:
            fig_array = figure_to_array(fig)
            plt.close(fig)
            fig = fig_array

        return fig

    def _plot_single_region(self, axes, true_data, pred_data, region, sample_id):
        zoomin_radius = self.zoomin_radius
        resolution = 32

        seq_len = true_data.shape[-1]
        x = np.arange(seq_len)
        zoomin_start = seq_len // 2 - zoomin_radius
        zoomin_end = seq_len // 2 + zoomin_radius
        zoomin_slice = slice(zoomin_start, zoomin_end)

        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))

        # full region
        ax = axes[0]
        ax.fill_between(
            x=x,
            y1=np.convolve(true_data, np.ones(8) / 8, mode="same"),
            linewidth=0,
            color="salmon",
        )
        ax.set(xlim=(0, seq_len), xticks=[])
        ax.axvspan(zoomin_start, zoomin_end, color="grey", alpha=0.2)
        corr = np.corrcoef(true_data, pred_data)[0, 1]
        true_sum = int(true_data.sum())
        pred_sum = int(pred_data.sum())
        ax.set_title(
            f"{chrom}:{start:,}-{end:,}; Corr={corr:.3f}; Sum T/P={true_sum:,}/{pred_sum:,}; VQ-ID={sample_id}",
            fontsize=8,
        )

        ax = axes[1]
        ax.fill_between(
            x=x,
            y1=np.convolve(pred_data, np.ones(8) / 8, mode="same"),
            linewidth=0,
            color="steelblue",
        )
        ax.set(xlim=(0, seq_len), xticks=[])
        ax.axvspan(zoomin_start, zoomin_end, color="grey", alpha=0.2)

        # zoom in
        ax = axes[2]
        zoomin_x = np.arange(zoomin_radius * 2)
        ax.fill_between(
            x=zoomin_x, y1=true_data[zoomin_slice], linewidth=0, color="salmon"
        )
        rstart = (start + end) // 2 - zoomin_radius * resolution
        rend = (start + end) // 2 + zoomin_radius * resolution
        ax.set(xlim=(0, zoomin_radius * 2), xticks=[])
        corr = np.corrcoef(true_data[zoomin_slice], pred_data[zoomin_slice])[0, 1]
        true_sum = int(true_data[zoomin_slice].sum())
        pred_sum = int(pred_data[zoomin_slice].sum())
        ax.set_title(
            f"{chrom}:{rstart:,}-{rend:,}; Corr. {corr:.3f}; Sum T/P {true_sum:,}/{pred_sum:,}; VQ-ID={sample_id}",
            fontsize=8,
        )

        ax = axes[3]
        ax.fill_between(
            x=zoomin_x, y1=pred_data[zoomin_slice], linewidth=0, color="steelblue"
        )
        ax.set(xlim=(0, zoomin_radius * 2), xticks=[])
        return
