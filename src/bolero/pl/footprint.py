from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def figure_to_array(fig):
    """Convert a Matplotlib figure to a numpy array with RGB channels."""
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


class FootPrintExamplePlotter:
    """Plot signal, bias, target and predict footprints for a single example.

    Args:
        signal (np.ndarray): The signal data.
        bias (np.ndarray): The bias data.
        target (np.ndarray): The target footprint data.
        predict (np.ndarray): The predicted footprint data.
        footprinter (FootPrinter): The footprinter object used for postprocessing.

    Attributes
    ----------
        signal (np.ndarray): The signal data.
        bias (np.ndarray): The bias data.
        target (np.ndarray): The target footprint data.
        predict (np.ndarray): The predicted footprint data.
        batch_size (int): The batch size.
        signal_length (int): The length of the signal.

    """

    def __init__(
        self,
        signal: np.ndarray,
        bias: np.ndarray,
        target: np.ndarray,
        predict: np.ndarray,
        footprinter: Callable,
    ):
        self.signal = signal
        self.bias = bias
        self.target = footprinter.postprocess_footprint(target)
        self.predict = footprinter.postprocess_footprint(predict)

        self.batch_size: int = self.signal.shape[0]
        self.signal_length: int = self.signal.shape[1]

    @staticmethod
    def _patch_footprint(footprint: np.ndarray) -> np.ndarray:
        """Patch zeros with shape (modes, 100) to both ends of footprint on the second axis.

        Args:
            footprint (np.ndarray): The footprint data.

        Returns
        -------
            np.ndarray: The patched footprint data.

        """
        modes = footprint.shape[0]
        patch = np.zeros((modes, 100))
        footprint = np.concatenate((patch, footprint, patch), axis=1)
        return footprint

    def _take(
        self, idx: Union[int, str] = "random"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Take a random example from the data.

        Args:
            idx (Union[int, str], optional): The index of the example to take. Defaults to "random".

        Returns
        -------
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The signal, bias, target, and predict data.

        """
        if idx == "random":
            idx = np.random.randint(self.batch_size)
        signal = self.signal[idx]
        bias = self.bias[idx]
        target = self._patch_footprint(self.target[idx])
        predict = self._patch_footprint(self.predict[idx])
        return signal, bias, target, predict

    def plot(
        self,
        idx: Union[int, str] = "random",
        fig: Optional[plt.Figure] = None,
        axes: Optional[list[plt.Axes]] = None,
        figsize: tuple[int, int] = (8, 3),
        dpi: int = 150,
        barplot: bool = False,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot the footprints.

        Args:
            idx (Union[int, str], optional): The index of the example to plot. Defaults to "random".
            fig (Optional[plt.Figure], optional): The figure object to use for plotting. Defaults to None.
            axes (Optional[List[plt.Axes]], optional): The axes objects to use for plotting. Defaults to None.
            figsize (Tuple[int, int], optional): The figure size. Defaults to (8, 3).
            dpi (int, optional): The figure DPI. Defaults to 150.
            barplot (bool, optional): Use barplot instead of fill_between. Defaults to False because fill_between is much faster.

        Returns
        -------
            Tuple[plt.Figure, List[plt.Axes]]: The figure and axes objects.

        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
            gs = fig.add_gridspec(ncols=1, nrows=4, height_ratios=[1, 1, 3, 3])
            axes = [fig.add_subplot(gs[i]) for i in range(4)]

        signal, bias, target, predict = self._take(idx)
        _plot_func = self._plot_bar if barplot else self._plot_fill_between
        _plot_func(ax=axes[0], data=signal, title="Signal")
        _plot_func(ax=axes[1], data=bias, title="Bias")

        # calculate hue norm
        target_hue_norm = (0, 2)
        target_hue_min_quantile = (target < target_hue_norm[0]).sum() / target.size
        target_hue_max_quantile = (target < target_hue_norm[1]).sum() / target.size
        predict_hue_norm = (
            np.quantile(predict, target_hue_min_quantile),
            np.quantile(predict, target_hue_max_quantile),
        )

        self._plot_image(
            ax=axes[2], image=target, title="Target", hue_norm=target_hue_norm
        )
        self._plot_image(
            ax=axes[3], image=predict, title="Predict", hue_norm=predict_hue_norm
        )
        return fig, axes

    def _common_axes_setup(self, ax: plt.Axes) -> None:
        """Common setup for axes.

        Args:
            ax (plt.Axes): The axes object.

        """
        ax.set_xlim(0, self.signal_length)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        sns.despine(ax=ax, left=True)
        ax.tick_params(axis="both", which="major", labelsize=6)

    def _plot_fill_between(
        self, ax: plt.Axes, data: Union[np.ndarray, torch.Tensor], title: str
    ) -> None:
        """Plot fill_between.

        Args:
            ax (plt.Axes): The axes object.
            data (Union[np.ndarray, torch.Tensor]): The data to plot.
            title (str): The title of the plot.

        """
        if hasattr(data, "numpy"):
            data = data.detach().cpu().numpy()
        ax.fill_between(range(len(data)), data, linewidth=0)
        ax.set_title(title, fontsize=8)
        self._common_axes_setup(ax)

    def _plot_bar(
        self, ax: plt.Axes, data: Union[np.ndarray, torch.Tensor], title: str
    ) -> None:
        """Plot bar.

        Barplot is much slower than fill_between.

        Args:
            ax (plt.Axes): The axes object.
            data (Union[np.ndarray, torch.Tensor]): The data to plot.
            title (str): The title of the plot.

        """
        if hasattr(data, "numpy"):
            data = data.detach().cpu().numpy()
        ax.bar(range(len(data)), data, linewidth=0)
        ax.set_title(title, fontsize=8)
        self._common_axes_setup(ax)

    def _plot_image(
        self,
        ax: plt.Axes,
        image: Union[np.ndarray, torch.Tensor],
        title: str,
        hue_norm: tuple[float, float],
    ) -> None:
        """Plot image.

        Args:
            ax (plt.Axes): The axes object.
            image (Union[np.ndarray, torch.Tensor]): The image data to plot.
            title (str): The title of the plot.
            hue_norm (Tuple[float, float]): The hue norm.

        """
        if hasattr(image, "numpy"):
            image = image.detach().cpu().numpy()

        vmin, vmax = hue_norm
        ax.imshow(image[::-1], cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        self._common_axes_setup(ax)
