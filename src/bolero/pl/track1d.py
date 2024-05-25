import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns


def per_row_pearsonr(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation coefficient between each row of two arrays.
    """
    print(arr1.shape, arr2.shape, 'per_row_pearsonr')
    pearson_correlations = []
    for row1, row2 in zip(arr1, arr2):
        correlation, _ = pearsonr(row1, row2)
        pearson_correlations.append(correlation)
    return np.array(pearson_correlations)


def per_row_mse(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Calculate mean squared error between each row of two arrays.
    """
    return np.mean((arr1 - arr2) ** 2, axis=1)


class Track1DExamplePlotter:
    def __init__(self, target_key, predict_key):
        self.target_key = target_key
        self.predict_key = predict_key
        pass

    def _select_example_by_corr(self, batch, top_example=1, bottom_example=1, plot_channel=0):
        target_count = batch[self.target_key].cpu().numpy()[:, plot_channel, ...]
        target = np.log1p(target_count)
        predict = batch[self.predict_key].cpu().numpy()[:, plot_channel, ...]
        predict_count = np.expm1(predict)

        corr = per_row_pearsonr(target, predict)
        mse = per_row_mse(target, predict)

        # get the index of top and bottom examples based on their correlation
        index_order = np.argsort(corr)
        top_example_idx = index_order[-top_example:]
        bottom_example_idx = index_order[:bottom_example]
        example_idx = np.concatenate([top_example_idx, bottom_example_idx])

        target_data = target_count[example_idx]
        predict_data = predict_count[example_idx]
        corr_data = corr[example_idx]
        mse_data = mse[example_idx]
        return target_data, predict_data, corr_data, mse_data

    def plot(
        self,
        batch,
        figsize=(6, 2.5),
        dpi=100,
        top_example=1,
        bottom_example=1,
        plot_channel=0,
    ):
        nrows = int((top_example + bottom_example) * 3)
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(ncols=1, nrows=nrows)
        axes = [fig.add_subplot(gs[i]) for i in range(nrows)]

        target_data, predict_data, corr_data, mse_data = self._select_example_by_corr(
            batch, top_example, bottom_example, plot_channel
        )
        y_max = np.quantile(np.concatenate([target_data, predict_data]), 0.95)

        title_fs = 8
        for i, (target, predict, corr, mse) in enumerate(
            zip(target_data, predict_data, corr_data, mse_data)
        ):
            base = int(i * 3)
            ax = axes[base]
            ax.plot(target)
            ax.set_title("Target", fontsize=title_fs)
            ax = axes[base + 1]
            ax.plot(predict)
            ax.set_title("Predict", fontsize=title_fs)
            ax = axes[base + 2]
            ax.plot(target - predict)
            ax.set_title(
                f"Delta (Pearson Corr: {corr:.3f}; MSE: {mse:.3f})", fontsize=title_fs
            )

        for i, ax in enumerate(axes):
            ax.set(xlim=(0, len(target_data[0])), ylim=(0, y_max), yticks=[])
            sns.despine(ax=ax, left=True)
        return fig, axes
