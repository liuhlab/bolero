import matplotlib.pyplot as plt
import numpy as np
import torch

from bolero import Genome
from bolero.pl.utils import figure_to_array


class BorzoiExamplePlotter:
    def __init__(
        self,
        genome: Genome | None,
        zoomin_radius: int = 500,
        true_key="true_data",
        pred_key="pred_data",
        id_key="sample_id",
        plot_mode="atac",
    ):
        self.genome = genome
        self.zoomin_radius = zoomin_radius
        self.true_key = true_key
        self.pred_key = pred_key
        self.id_key = id_key
        self.plot_mode = plot_mode
        return

    def parse_region_coords(self, batch):
        """Parse the region coordinates from a batch of examples"""
        if self.genome is None:
            return None
        regions = batch["region"]
        if isinstance(regions, torch.Tensor):
            regions = regions.cpu().numpy()

            # adjust y_true crop
            y_true_seq_len = batch[self.true_key].shape[-1]
            crop_bp = (16384 - y_true_seq_len) * 32 / 2
            crop_bp = int(crop_bp)
            regions[:, 0] += crop_bp
            regions[:, 1] -= crop_bp

        regions = self.genome.parse_global_coords(regions)
        return regions

    def plot(self, batch, channel=0, nrows=2, return_array=False, y_sync=False):
        """Plot the true and predicted data for a batch of examples."""
        y_true = batch[self.true_key]
        y_pred = batch[self.pred_key]

        if "region_names" in batch:
            region_names = batch["region_names"]
            gene_names = batch["gene_names"]
        else:
            region_names = None
            gene_names = None

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.float().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.float().cpu().numpy()
        sample_ids = batch.get(self.id_key, None)
        if sample_ids is None:
            sample_ids = np.arange(len(y_true))

        regions = self.parse_region_coords(batch)

        bs = y_true.shape[0]
        if isinstance(nrows, int):
            nrows = min(bs, nrows)
        else:
            nrows = [min(bs - 1, n) for n in nrows]
        row_ids = list(range(nrows)) if isinstance(nrows, int) else nrows

        fig, axes = plt.subplots(
            figsize=(8, 2.25 * len(row_ids)),
            dpi=150,
            nrows=4 * len(row_ids),
            constrained_layout=True,
        )

        seq_len = y_true.shape[-1]
        zoomin_start = seq_len // 2 - self.zoomin_radius
        zoomin_end = seq_len // 2 + self.zoomin_radius
        zoomin_slice = slice(zoomin_start, zoomin_end)

        if y_sync:
            y_max = max(np.quantile(y_true, 0.999), np.quantile(y_pred, 0.999))
            y_zoomin_max = max(
                np.quantile(y_true[..., zoomin_slice], 0.999),
                np.quantile(y_pred[..., zoomin_slice], 0.999),
            )
            y_max_pair = (y_max, y_zoomin_max)
        else:
            y_max_pair = None

        for i, row_id in enumerate(row_ids):
            row_axes = axes[i * 4 : (i + 1) * 4]
            true_data = y_true[row_id, channel, :]
            pred_data = y_pred[row_id, channel, :]
            sample_id = sample_ids[row_id]

            if regions is not None:
                chrom, start, end, *_ = regions.iloc[row_id]
                region = f"{chrom}:{start}-{end}"
            else:
                region = "chrUnknown:0-524288"

            if region_names is not None:
                gene_name = gene_names[row_id]
                region_name = region_names[row_id]
            else:
                gene_name = ""
                region_name = ""

            self._plot_single_region(
                row_axes,
                true_data,
                pred_data,
                region,
                sample_id,
                channel,
                zoomin_slice=zoomin_slice,
                gene_name=gene_name,
                region_name=region_name,
                y_max_pair=y_max_pair,
            )

        for ax in axes.flat:
            ax.tick_params(axis="both", labelsize=8)

        if return_array:
            fig_array = figure_to_array(fig)
            plt.close(fig)
            fig = fig_array

        return fig

    def _plot_single_region(
        self,
        axes,
        true_data,
        pred_data,
        region,
        sample_id,
        channel,
        zoomin_slice,
        gene_name="",
        region_name="",
        y_max_pair=None,
    ):
        resolution = 32
        zoomin_start, zoomin_end = zoomin_slice.start, zoomin_slice.stop

        seq_len = true_data.shape[-1]
        x = np.arange(seq_len)
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))

        if self.plot_mode == "atac":
            true_data_conv = np.convolve(true_data, np.ones(8) / 8, mode="same")
            pred_data_conv = np.convolve(pred_data, np.ones(8) / 8, mode="same")
        else:
            true_data_conv = true_data
            pred_data_conv = pred_data

        # full region
        ax = axes[0]
        ax.fill_between(
            x=x,
            y1=true_data_conv,
            linewidth=0,
            color="salmon",
        )
        ax.set(xlim=(0, seq_len), xticks=[])
        ax.axvspan(zoomin_start, zoomin_end, color="grey", alpha=0.2)
        if true_data.std() == 0:
            corr = 0
        else:
            corr = np.corrcoef(true_data, pred_data)[0, 1]
        true_sum = int(true_data.sum())
        pred_sum = int(pred_data.sum())
        ax.set_title(
            f"{chrom}:{start:,}-{end:,} channel{channel}; "
            f"Corr={corr:.3f}; Sum T/P={true_sum:,}/{pred_sum:,}; Sample={sample_id}",
            fontsize=8,
        )
        ax.text(0.01, 0.6, region_name, ha="left", fontsize=8, transform=ax.transAxes)

        ax = axes[1]
        ax.fill_between(
            x=x,
            y1=pred_data_conv,
            linewidth=0,
            color="steelblue",
        )
        ax.set(xlim=(0, seq_len), xticks=[])
        ax.axvspan(zoomin_start, zoomin_end, color="grey", alpha=0.2)

        # zoom in
        ax = axes[2]
        y1 = true_data[zoomin_slice]
        zoomin_x = np.arange(y1.size)
        ax.fill_between(x=zoomin_x, y1=y1, linewidth=0, color="salmon")
        rstart = start + zoomin_slice.start * resolution
        rend = start + zoomin_slice.stop * resolution
        ax.set(xlim=(0, zoomin_x.size), xticks=[])
        if true_data[zoomin_slice].std() == 0:
            corr = 0
        else:
            corr = np.corrcoef(true_data[zoomin_slice], pred_data[zoomin_slice])[0, 1]
        true_sum = int(true_data[zoomin_slice].sum())
        pred_sum = int(pred_data[zoomin_slice].sum())
        ax.set_title(
            f"{chrom}:{rstart:,}-{rend:,} channel{channel}; "
            f"Corr. {corr:.3f}; Sum T/P {true_sum:,}/{pred_sum:,}; Sample={sample_id}",
            fontsize=8,
        )
        ax.text(0.01, 0.6, gene_name, ha="left", fontsize=8, transform=ax.transAxes)

        ax = axes[3]
        y1 = pred_data[zoomin_slice]
        zoomin_x = np.arange(y1.size)
        ax.fill_between(x=zoomin_x, y1=y1, linewidth=0, color="steelblue")
        ax.set(xlim=(0, zoomin_x.size), xticks=[])

        if y_max_pair is not None:
            y_max, y_zoomin_max = y_max_pair
            axes[0].set(ylim=(0, y_max))
            axes[1].set(ylim=(0, y_max))
            axes[2].set(ylim=(0, y_zoomin_max))
            axes[3].set(ylim=(0, y_zoomin_max))
        return
