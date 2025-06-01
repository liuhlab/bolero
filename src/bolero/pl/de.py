import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _volcano_scatter(ax, data, pval_cutoff, max_dots=5000):
    if data.shape[0] > max_dots:
        dot_data = data.sample(max_dots, random_state=42)
    else:
        dot_data = data

    sns.scatterplot(
        data=dot_data,
        x="x",
        y="y",
        hue="hue",
        palette={0: "grey", -1: "steelblue", 1: "salmon"},
        linewidth=0,
        ax=ax,
        s=5,
        legend=None,
        rasterized=True,
    )
    ax.axhline(y=-np.log10(pval_cutoff), color="black", linestyle="--")
    ax.axvline(x=0, color="black", linestyle="--")

    sig_count = data["hue"].value_counts()
    for key, x, color in zip([-1, 1], [0.05, 0.95], ["steelblue", "salmon"]):
        count = sig_count.get(key, 0)
        ax.text(
            x,
            0.85,
            count,
            transform=ax.transAxes,
            fontsize=10,
            color=color,
            weight="bold",
            ha="right" if key == 1 else "left",
        )

    ax.set_xlabel("Log Fold Change")
    ax.set_ylabel("-Log10 Adjusted P-Value")
    return ax


class VolcanoPlot:
    def __init__(
        self,
        data,
        x="log_fc",
        y="adj_p_value",
        tranform_y=True,
        pval_cutoff=0.05,
        log_fc_cutoff=0,
        y_qmax=0.9999,
        x_qmax=0.9999,
    ):
        self.data = data

        self.plot_data = pd.DataFrame(
            {
                "x": data[x],
                "y": (
                    -np.log10(data[y].astype("float64") + 1e-50)
                    if tranform_y
                    else data[y]
                ),
                "pass_filter": (data[y] < pval_cutoff)
                & (data[x].abs() > log_fc_cutoff),
            }
        )
        self.plot_data["y"] = self.plot_data["y"].clip(upper=50)
        # get color, non-sig is grey, sig and neg is blue, sig and pos is red
        self.plot_data["hue"] = (
            np.sign(self.plot_data["x"]) * self.plot_data["pass_filter"]
        ).astype("int8")
        self.pval_cutoff = pval_cutoff
        self.y_cutoff = -np.log10(pval_cutoff)

        # cap y values at y_qmax
        sig_plot_data = self.plot_data[self.plot_data["pass_filter"]]
        if sig_plot_data.empty:
            self.y_max = 5
            self.x_max = 3
        else:
            self.y_max = max(5, sig_plot_data["y"].quantile(y_qmax))
            self.x_max = max(3, sig_plot_data["x"].abs().quantile(x_qmax))
        self.plot_data["y"] = self.plot_data["y"].clip(upper=self.y_max)

    @staticmethod
    def _sel_group_by_sig(plot_data, groupby, min_sig=1, max_axes=16):
        """
        Select groups with at least `min_sig` significant points.
        """
        if isinstance(groupby, str):
            groupby = [groupby]

        data_col = []
        for _, group_df in plot_data.groupby(groupby, observed=True):
            if group_df["pass_filter"].sum() >= min_sig:
                data_col.append(group_df)

        if len(data_col) > max_axes:
            # sel df by data shape, df with more rows are selected first
            data_col = sorted(data_col, key=lambda df: df.shape[0], reverse=True)[
                :max_axes
            ]

        if len(data_col) == 0:
            plot_data = pd.DataFrame(columns=plot_data.columns)
        else:
            plot_data = pd.concat(data_col, ignore_index=True)
        return plot_data

    def plot(
        self,
        groupby=None,
        panel_size=3,
        max_cols=4,
        fig_kwargs=None,
        max_dots=10000,
        group_min_sig=10,
        max_axes=16,
    ):
        """
        Plot the volcano plot with optional grouping.

        Parameters
        ----------
        groupby : str or list of str, optional
            Column(s) to group by. If None, no grouping is applied.
        panel_size : int or tuple, optional
            Size of each panel in inches. If int, both width and height are set to this value.
        max_cols : int, optional
            Maximum number of columns in the plot grid.
        fig_kwargs : dict, optional
            Additional keyword arguments for `plt.subplots`.
        max_dots : int, optional
            Maximum number of points to plot in each panel. If the data has more points, a random sample is taken.
        group_min_sig : int, optional
            Minimum number of significant points required in each group to be included in the plot.
            If a group has fewer significant points, it will be excluded from the plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        axes : numpy.ndarray
            Array of axes objects for each subplot.
        """
        if groupby is None:
            n_groups = 1
        else:
            if isinstance(groupby, str):
                groupby = [groupby]

            plot_data = self.plot_data
            for col in groupby:
                plot_data[col] = self.data[col]
            if group_min_sig > 0:
                plot_data = self._sel_group_by_sig(
                    plot_data, groupby, group_min_sig, max_axes
                )

            n_groups = plot_data[groupby].nunique().prod()

        n_rows = int(np.ceil(n_groups / max_cols))
        if n_rows == 0:
            n_rows = 1
        if isinstance(panel_size, int):
            panel_size = (panel_size, panel_size)

        default_fig_kwargs = {
            "constrained_layout": True,
            "dpi": 200,
            **(fig_kwargs or {}),
        }

        fig, axes = plt.subplots(
            n_rows,
            max_cols,
            figsize=(panel_size[0] * max_cols, panel_size[1] * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
            **default_fig_kwargs,
        )

        if n_groups == 1:
            group_iter = [("All", plot_data)]
        else:
            group_iter = plot_data.groupby(groupby, observed=True)

        for ax, (group, group_data) in zip(axes.flatten(), group_iter):
            ax.set_title(" ".join(map(str, group)))
            group_data = group_data.reset_index(drop=True)
            _volcano_scatter(ax, group_data, self.pval_cutoff, max_dots=max_dots)
            ax.set_ylim(0, self.y_max)
            ax.set_xlim(-self.x_max, self.x_max)

        for ax in axes.flatten()[n_groups:]:
            ax.axis("off")
        return fig, axes
