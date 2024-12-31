import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from bolero.pl.track1d import Track1DExamplePlotter, per_row_mse, per_row_pearsonr
from bolero.pl.utils import figure_to_array


class HicExamplePlotter(Track1DExamplePlotter):
    def __init__(self, target_key: str, predict_key: str):
        """
        Initialize the Hic Plotter.

        Args:
            target_key: The key for the target value in the batch.
            predict_key: The key for the predicted value in the batch.
        """
        self.target_key = target_key
        self.predict_key = predict_key

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
        # For Corigami dataset, the target is after log1p transformation
        target = batch[self.target_key].float().cpu().numpy()
        target_data = target
        predict = batch[self.predict_key]
        predict = batch[self.predict_key].float().cpu().numpy()
        predict_data = predict

        target = target.reshape(target.shape[0], -1)
        predict = predict.reshape(predict.shape[0], -1)
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

    def _diagonal_normalization(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the matrix along the diagonal.

        Args:
            matrix: The matrix to normalize.

        Returns
        -------
            np.ndarray: The normalized matrix.
        """
        size = matrix.shape[0]
        normalized_matrix = np.zeros_like(matrix)
        for d in range(size):
            diagonal = np.diag(matrix, k=d)
            if len(diagonal) > 0:
                mean_value = np.mean(diagonal)
                std_value = np.std(diagonal)
                if mean_value != 0:
                    normalized_diagonal = (diagonal - mean_value) / std_value
                    np.fill_diagonal(normalized_matrix[d:], normalized_diagonal)
                    np.fill_diagonal(normalized_matrix[:, d:], normalized_diagonal)
        return normalized_matrix

    def plot(
        self,
        batch,
        figsize: tuple[int, int] = (11, 20),
        dpi: int = 100,
        top_example: int = 2,
        bottom_example: int = 2,
        plot_channel: int = 0,
        vmin: float = -2,
        vmax: float = 2,
        return_array: bool = False,
    ):
        """
        Plot the target and predicted values in the batch.

        Args:
            batch: The batch of data to plot.
            figsize: The size of the figure.
            dpi: The dots per inch of the figure.
            top_example: The number of top examples to plot.
            bottom_example: The number of bottom examples to plot.
            plot_channel: The channel to plot.
        """
        mpl.style.use("default")
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42

        nrows = int(top_example + bottom_example)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize, dpi=dpi)
        target_data, predict_data, corr_data, mse_data = self._select_example_by_corr(
            batch, top_example, bottom_example, plot_channel
        )
        for i, (target, predict, corr, mse) in enumerate(
            zip(target_data, predict_data, corr_data, mse_data)
        ):
            # normalized_target = self._diagonal_normalization(target)
            # normalized_predict = self._diagonal_normalization(predict)
            # color_map = LinearSegmentedColormap.from_list(
            #     "bright_red", [(1, 1, 1), (1, 0, 0)]
            # )
            base = int(i * 2)
            ax = axes.flatten()[base]
            ax.imshow(target, cmap="bwr", vmin=vmin, vmax=vmax)
            # ax.set_title(f"Target - Corr: {corr:.2f}, MSE: {mse:.2f}")
            ax.text(
                0.01,
                0.99,
                f"Min: {np.min(target):.2f}\nMax: {np.max(target):.2f}",
                color="black",
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax_2 = axes.flatten()[base + 1]
            ax_2.imshow(predict, cmap="bwr", vmin=vmin, vmax=vmax)
            # ax_2.set_title(f"Predict - Corr: {corr:.2f}, MSE: {mse:.2f}")
            ax_2.text(
                0.01,
                0.99,
                f"Min: {np.min(predict):.2f}\nMax: {np.max(predict):.2f}\nCorr: {corr:.2f}\nMSE: {mse:.2f}",
                color="black",
                ha="left",
                va="top",
                transform=ax_2.transAxes,
            )
            ax_2.set_xticks([])
            ax_2.set_yticks([])
        plt.tight_layout()

        if return_array:
            fig_array = figure_to_array(fig)
            plt.close(fig)
            fig = fig_array
        return fig
