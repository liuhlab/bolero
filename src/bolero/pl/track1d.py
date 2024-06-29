import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr


def per_row_pearsonr(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation coefficient between each row of two arrays.
    """
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
    def __init__(self, target_key: str, predict_key: str, data_mode="count"):
        """
        Initialize the Track1DExamplePlotter class.

        Parameters
        ----------
        - target_key (str): The key for the target data in the batch.
        - predict_key (str): The key for the predicted data in the batch.
        """
        self.target_key = target_key
        self.predict_key = predict_key
        self.data_mode = data_mode

    def _select_example_by_corr(
        self,
        batch: dict,
        top_example: int = 1,
        bottom_example: int = 1,
        plot_channel: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select examples based on correlation between target and predicted data.

        Parameters
        ----------
        - batch (dict): The batch containing target and predicted data.
        - top_example (int): Number of top examples to select.
        - bottom_example (int): Number of bottom examples to select.
        - plot_channel (int): The channel to plot.

        Returns
        -------
        - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the selected target data,
          selected predicted data, correlation values, and mean squared error values.
        """
        if self.data_mode == "count":
            target_data = batch[self.target_key].cpu().numpy()[:, plot_channel, ...]
            target = np.log1p(target_data)
            predict = batch[self.predict_key].cpu().numpy()[:, plot_channel, ...]
            predict_data = np.expm1(predict)
        else:
            target = batch[self.target_key].cpu().numpy()[:, plot_channel, ...]
            target_data = target
            predict = batch[self.predict_key].cpu().numpy()[:, plot_channel, ...]
            predict_data = predict

        corr = per_row_pearsonr(target, predict)
        mse = per_row_mse(target, predict)

        # get the index of top and bottom examples based on their correlation
        index_order = np.argsort(corr)
        top_example_idx = index_order[-top_example:]
        bottom_example_idx = index_order[:bottom_example]
        example_idx = np.concatenate([top_example_idx, bottom_example_idx])

        target_data = target_data[example_idx]
        predict_data = predict_data[example_idx]
        corr_data = corr[example_idx]
        mse_data = mse[example_idx]
        return target_data, predict_data, corr_data, mse_data

    @staticmethod
    def moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate the moving average of a given data array.

        Parameters
        ----------
        - data (np.ndarray): The input data array.
        - window (int): The size of the moving average window.

        Returns
        -------
        - np.ndarray: The moving average of the input data.
        """
        return np.convolve(data, np.ones(window) / window, mode="same")

    def plot(
        self,
        batch: dict,
        figsize: tuple[int, int] = (6, 6),
        dpi: int = 100,
        top_example: int = 2,
        bottom_example: int = 2,
        plot_channel: int = 0,
        moving_ave_window: int = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the target and predicted data.

        Parameters
        ----------
        - batch (dict): The batch containing target and predicted data.
        - figsize (tuple[int, int]): The size of the figure.
        - dpi (int): The resolution of the figure.
        - top_example (int): Number of top examples to plot.
        - bottom_example (int): Number of bottom examples to plot.
        - plot_channel (int): The channel to plot.
        - moving_ave_window (int): The size of the moving average window.

        Returns
        -------
        - tuple[plt.Figure, list[plt.Axes]]: A tuple containing the figure and a list of axes.
        """
        nrows = int((top_example + bottom_example) * 2)
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(ncols=1, nrows=nrows)
        axes = [fig.add_subplot(gs[i]) for i in range(nrows)]

        target_data, predict_data, corr_data, mse_data = self._select_example_by_corr(
            batch, top_example, bottom_example, plot_channel
        )

        title_fs = 8
        for i, (target, predict, corr, mse) in enumerate(
            zip(target_data, predict_data, corr_data, mse_data)
        ):
            base = int(i * 2)
            ax = axes[base]
            if moving_ave_window is None:
                ax.plot(target, color="steelblue")
            else:
                ax.plot(
                    self.moving_average(target, moving_ave_window),
                    color="steelblue",
                )
            ax.set_title("Target", fontsize=title_fs)
            ax = axes[base + 1]
            if moving_ave_window is None:
                ax.plot(predict, color="salmon")
            else:
                ax.plot(
                    self.moving_average(predict, moving_ave_window),
                    color="salmon",
                )
            ax.set_title(
                f"Predict (Pearson Corr: {corr:.3f}; MSE: {mse:.3f})", fontsize=title_fs
            )

        for ax in axes:
            ax.set(xlim=(0, len(target_data[0])))
            sns.despine(ax=ax)
        return fig, axes
