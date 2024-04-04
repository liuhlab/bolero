import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns


def _insert_to_xranges(row_xranges, motif_range, space=2):
    # Insert at beginning
    if motif_range[1] < row_xranges[0][0] - space:
        row_xranges.insert(0, motif_range)  # Use list.insert method
        return True

    # Insert at end
    if motif_range[0] > row_xranges[-1][1] + space:
        row_xranges.append(motif_range)  # Use list.append method for clarity
        return True

    # Check for insertion between existing tuples using list.insert method
    for i in range(len(row_xranges) - 1):
        if (
            motif_range[0] > row_xranges[i][1] + space
            and motif_range[1] < row_xranges[i + 1][0] - space
        ):
            row_xranges.insert(i + 1, motif_range)  # Insertion at the correct position
            return True
    return False


class MotifTrackPlotter:
    """Plot motif tracks."""

    def __init__(self, motifs_df, name_col="Gene", plot_order=None, plot_genes=None):
        self.motifs_df = motifs_df
        self.name_col = name_col
        self.plot_order = None
        self.plot_genes = None
        self._update_plot_order_and_genes(plot_order=plot_order, plot_genes=plot_genes)

    def _update_plot_order_and_genes(self, plot_order=None, plot_genes=None):
        motifs_df = self.motifs_df
        name_col = self.name_col

        # determine plot order and gene list
        if plot_order is None:
            plot_order = (
                motifs_df.groupby(name_col)["Score"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
        if plot_genes is None:
            plot_genes = plot_order
        else:
            plot_order = [g for g in plot_order if g in plot_genes]

        self.plot_order = plot_order
        self.plot_genes = plot_genes

    def _generate_non_overlap_xrange_rows(self):
        motifs_df = self.motifs_df
        name_col = self.name_col
        plot_order = self.plot_order
        plot_genes = self.plot_genes

        motifs_df = motifs_df[motifs_df[name_col].isin(plot_genes)]
        motifs_df = motifs_df.reindex(
            motifs_df[name_col].map(lambda i: plot_order.index(i)).sort_values().index
        ).reset_index(drop=True)

        rows = []
        for _, (start, end, name) in motifs_df[["Start", "End", name_col]].iterrows():
            for row_xranges in rows:
                inserted = _insert_to_xranges(row_xranges, (start, end, name))
                if inserted:
                    break
            else:
                rows.append([(start, end, name)])
        return rows

    def _color_map(self, cmap):
        plot_order = self.plot_order

        # determine color map
        n_genes = len(plot_order)
        if cmap == "auto":
            cmap = "hls"
        colors = sns.color_palette(cmap, n_genes)
        color_map = {}
        for i, gene in enumerate(
            pd.Series(plot_order).sample(len(plot_order), replace=False, random_state=0)
        ):
            idx = i % n_genes
            color_map[gene] = colors[idx]
        return color_map

    def plot(
        self,
        ax,
        cmap="auto",
        bar_height=0.8,
        top_ticks=True,
        show_legend=True,
        legend_loc=(0.5, -0.1),
        **legend_kwargs,
    ):
        """Plot motif tracks on a given axis."""
        rows = self._generate_non_overlap_xrange_rows()
        color_map = self._color_map(cmap)

        for row, row_xranges in enumerate(rows):
            xrange = [(s, e - s) for s, e, _ in row_xranges]
            color_list = [color_map[name] for *_, name in row_xranges]
            if top_ticks:
                y = len(rows) - row - 1
            else:
                y = row
            yrange = (y, bar_height)
            ax.broken_barh(xrange, yrange, facecolor=color_list)

        ax.set_yticks([])

        if top_ticks:
            sns.despine(left=True, bottom=True, top=False, ax=ax)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
        else:
            sns.despine(left=True, bottom=False, top=True, ax=ax)

        if show_legend:
            self._plot_legend(ax, color_map, legend_loc, **legend_kwargs)

    def _plot_legend(self, ax, color_map, legend_loc, **legend_kwargs):
        patches = [
            mpatches.Patch(color=color_map[label], label=label, linewidth=0)
            for label in self.plot_order
        ]

        fig_size_inches = ax.figure.get_size_inches()
        ax_bbox = ax.get_position()
        ax_size_inches = [
            fig_size_inches[0] * ax_bbox.width,
            fig_size_inches[1] * ax_bbox.height,
        ]
        ncol = int(ax_size_inches[0] / 0.8)

        anchor_loc = "lower center" if legend_loc == "top" else "upper center"

        _legend_kwargs = {
            "handles": patches,
            "loc": anchor_loc,
            "bbox_to_anchor": legend_loc,
            "ncol": ncol,
            "frameon": False,
            "handlelength": 1.5,
            "handleheight": 1,
            "handletextpad": 0.5,
            "fontsize": 10,
        }
        _legend_kwargs.update(legend_kwargs)
        ax.legend(**_legend_kwargs)
        return
