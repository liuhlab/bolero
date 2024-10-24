import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class mcExamplePlotter:
    def __init__(self, target_key: list[str], predict_key: list[str]):
        """
        Initialize the Track1DExamplePlotter class.

        Parameters
        ----------
        - target_key (str): The key for the target data in the batch.
        - predict_key (str): The key for the predicted data in the batch.
        """
        self.target_key = target_key
        self.predict_key = predict_key

    def _select_example(
        self,
        batch: dict,
        example: int = 2,
        plot_channel: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomlu select examples from the batch.

        Parameters
        ----------
        - batch (dict): The batch containing target and predicted data.
        - example (int): Number of examples to select.
        - plot_channel (int): The channel to plot.

        Returns
        -------
        - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the selected target data,
          selected predicted data, correlation values, and mean squared error values.
        """
        key_list = self.target_key + self.predict_key

        mc_target = batch[key_list[0]].cpu().numpy()[:, plot_channel, ...]
        atac_target = batch[key_list[1]].cpu().numpy()[:, plot_channel, ...]
        mc_predict = batch[key_list[2]].cpu().numpy()[:, plot_channel, ...]
        atac_prediction = batch[key_list[3]].cpu().numpy()[:, plot_channel, ...]
        region = batch["final_region"].cpu().numpy()

        # randomly select example to plot
        idx = np.random.choice(mc_target.shape[0], example, replace=False)
        mc_target = mc_target[idx]
        mc_predict = mc_predict[idx]
        atac_target = np.log1p(atac_target[idx])
        atac_prediction = np.expm1(atac_prediction[idx])
        region = region[idx]

        return mc_target, mc_predict, atac_target, atac_prediction, region

    def plot_alltrack(
        self,
        batch: dict,
        figsize: tuple[int, int] = (12, 12),  # Increased figure size for more subplots
        dpi: int = 100,
        example: int = 2,
        total_channel: int = 2,
        moving_ave_window: int = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the target and predicted data with additional data tracks.

        Parameters
        ----------
        - batch (dict): The batch containing target and predicted data.
        - figsize (tuple[int, int]): The size of the figure.
        - dpi (int): The resolution of the figure.
        - example (int): Number of examples to plot.
        - total_channel (int): The number of channels to plot.
        - moving_ave_window (int): The size of the moving average window.

        Returns
        -------
        - tuple[plt.Figure, list[plt.Axes]]: A tuple containing the figure and a list of axes.
        """
        nrows = example * total_channel * 2
        ncols = 1
        plot_channel = 0

        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
        axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

        # New function returns 8 sample data (2 samples x 4 tracks each)
        mc_target_data, mc_pred_data, atac_target_data, atac_pred_data, regions = (
            self._select_example(batch, example, plot_channel)
        )

        for i, (mc_target, mc_pred, atac_target, atac_pred, re) in enumerate(
            zip(mc_target_data, mc_pred_data, atac_target_data, atac_pred_data, regions)
        ):
            base = int(i * total_channel * 2)
            ax = axes[base + 0]
            if moving_ave_window is None:
                ax.plot(mc_target, color="steelblue")
            else:
                ax.plot(
                    self.moving_average(mc_target, moving_ave_window),
                    color="steelblue",
                )
            ax.set_title("mC Target", fontsize=8)

            ax = axes[base + 1]
            if moving_ave_window is None:
                ax.plot(mc_pred, color="salmon")
            else:
                ax.plot(
                    self.moving_average(mc_pred, moving_ave_window),
                    color="salmon",
                )
            ax.set_title("mC Predict", fontsize=8)

            # Plot additional data tracks
            ax = axes[base + 2]
            if moving_ave_window is None:
                ax.plot(atac_target, color="green")
            else:
                ax.plot(
                    self.moving_average(atac_target, moving_ave_window),
                    color="green",
                )
            ax.set_title(f"ATAC Target with {re}", fontsize=8)

            ax = axes[base + 3]
            if moving_ave_window is None:
                ax.plot(atac_pred, color="orange")
            else:
                ax.plot(
                    self.moving_average(atac_pred, moving_ave_window),
                    color="orange",
                )
            ax.set_title("ATAC Prediction", fontsize=8)

        # Setting xlim and removing spines
        for ax in axes:
            ax.set(xlim=(0, len(mc_target)))  # Adjusted to length of the target
            sns.despine(ax=ax)

        return fig, axes
