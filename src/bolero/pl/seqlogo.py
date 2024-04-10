"""
This module is a modified version of the motif_plotter module from the motif_plotter package.

https://github.com/const-ae/motif_plotter/tree/master
"""

import matplotlib.patches as patches
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


def approximate_error(motif):
    """Calculate approximate error"""
    pwm = motif.pwm
    bases = list(pwm.keys())
    n = sum(motif.counts[bases[0]])
    approx_error = (len(bases) - 1) / (2 * np.log(2) * n)
    return approx_error


def exact_error(motif):
    """Calculate exact error, using multinomial(na,nc,ng,nt)"""
    ## Super Slow. O(n^3)
    na = sum(motif.counts["A"])
    n = na
    nc = 0
    ng = 0
    nt = 0
    done = False
    exact_error = 0
    while not done:
        print(na, nc, ng, nt)
        exact_error += sum([-p * np.log2(p) for p in [na / n, nc / n, ng / n, nt / n]])
        if nt <= 0:
            ## iterate inner loop
            if ng > 0:
                ## g => t
                ng = ng - 1
                nt = nt + 1
            elif nc > 0:
                ## c -> g
                nc = nc - 1
                ng = ng + 1
            else:
                ## a->c
                na = na - 1
                nc = nc + 1
        else:
            if ng > 0:
                ## g => t
                ng = ng - 1
                nt = nt + 1
            elif nc > 0:
                ## c => g; all t -> g
                nc = nc - 1
                ng = nt + 1
                nt = 0
            elif na > 0:
                ## a => c; all g,t -> c
                nc = nt + 1
                na = na - 1
                nt = 0
            else:
                done = True
    return exact_error


def calc_info_matrix(motif, correction_type="approx"):
    """Calculate information matrix with small sample correction"""
    pwm = motif.pwm
    bases = pwm.keys()
    if correction_type == "approx":
        error = approximate_error(motif)
    else:
        error = exact_error(motif)
    info_matrix = [
        2 - error + sum([pwm[b][l] * np.nan_to_num(np.log2(pwm[b][l])) for b in bases])
        for l in range(0, len(motif))
    ]
    return info_matrix


def calc_relative_information(motif, correction_type="approx"):
    """Calculate relative information matrix"""
    pwm = motif.pwm
    bases = pwm.keys()
    if correction_type == "approx":
        info_matrix = calc_info_matrix(motif)
    else:
        info_matrix = calc_info_matrix(motif, "exact")
    relative_info = {
        base: [prob * info for prob, info in zip(pwm[base], info_matrix)]
        for base in bases
    }
    return relative_info


def make_text_elements(
    text,
    x=0.0,
    y=0.0,
    width=1.0,
    height=1.0,
    color="blue",
    edgecolor="black",
    font=FontProperties(family="monospace"),
):
    """
    Create text elements as patches for visualization.

    Parameters
    ----------
    text : str
        The text to be displayed.
    x : float, optional
        The x-coordinate of the text element's position. Default is 0.0.
    y : float, optional
        The y-coordinate of the text element's position. Default is 0.0.
    width : float, optional
        The width of the text element. Default is 1.0.
    height : float, optional
        The height of the text element. Default is 1.0.
    color : str, optional
        The color of the text element. Default is "blue".
    edgecolor : str, optional
        The edge color of the text element. Default is "black".
    font : FontProperties, optional
        The font properties of the text element. Default is FontProperties(family="monospace").

    Returns
    -------
    patches.PathPatch
        A PathPatch object representing the text element.

    """
    tp = TextPath((0.0, 0.0), text, size=1, prop=font)
    bbox = tp.get_extents()
    bwidth = bbox.x1 - bbox.x0
    bheight = bbox.y1 - bbox.y0
    trafo = Affine2D()
    trafo.translate(-bbox.x0, -bbox.y0)
    trafo.scale(1 / bwidth * width, 1 / bheight * height)
    trafo.translate(x, y)
    tp = tp.transformed(trafo)
    return patches.PathPatch(tp, facecolor=color, edgecolor=edgecolor)


def make_bar_plot(axes, texts, heights, width=0.8, colors=None):
    """
    Makes a bar plot but each bar is not just a rectangle but an element from the texts list.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes that is modified
    texts : list of str
        A list of strings, where each element is plotted as a "bar"
    heights : list of float
        A list of the height of each texts element
    width : float, optional
        The width of the bar. Default: 0.8
    colors : list of str, optional
        A list of colors, a list with a single entry or None. Default: None, which is plotted as blue
    """
    texts = list(texts)
    heights = list(heights)
    n_elem = len(texts)
    if n_elem != len(heights):
        raise ValueError("Texts and heights must be of the same length")
    if colors is None:
        colors = ["blue"] * n_elem
    elif len(colors) == 1:
        colors *= n_elem

    axes.set_ylim(min(0, min(heights)), max(0, max(heights)))
    axes.set_xlim(0, n_elem)
    for idx, (text, height, color) in enumerate(zip(texts, heights, colors)):
        text_shape = make_text_elements(
            text,
            x=idx + (1 - width) / 2,
            y=0,
            width=width,
            height=height,
            color=color,
            edgecolor=color,
        )
        axes.add_patch(text_shape)


def make_single_sequence_spectrum(
    axis, row, row_scores, one_hot_decoding=None, colors=None
):
    """
    Makes a bar plot of a single sequence where only the base with the highest score is plotted.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axes that is modified
    row : numpy.ndarray
        A one-hot encoded sequence
    row_scores : numpy.ndarray
        The scores of each position in the sequence
    one_hot_decoding : list of str, optional
        A list of the one-hot encoding. Default: ["A", "T", "C", "G"]

    """
    if one_hot_decoding is None:
        one_hot_decoding = ["A", "C", "G", "T"]
    if colors is None:
        colors = ["#008000", "#0000cc", "#ffb300", "#cc0000"]
    sequence = [
        np.array(one_hot_decoding)[x] for x in np.apply_along_axis(np.argmax, 1, row)
    ]
    score_sequence = np.apply_along_axis(
        lambda e: np.max(e) if abs(np.min(e)) < np.max(e) else np.min(e), 1, row_scores
    )
    color_sequence = [
        np.array(colors)[x] for x in np.apply_along_axis(np.argmax, 1, row)
    ]
    make_bar_plot(axis, sequence, score_sequence, colors=color_sequence)


def make_stacked_bar_plot(axes, texts, heights, width=0.8, colors=None):
    """
    Makes a stackedbar plot but each bar is not just a rectangle but an element from the texts list.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes that is modified
    texts : list of list of str
        A list of list of strings, where each element is plotted as a "bar"
    heights : list of list of float
        A list of list of the height of each texts element
    width : float, optional
        The width of the bar. Default: 0.8
    colors : list of list of str, optional
        A list of list of colors, a list with a single entry or None. Default: None, which is plotted as blue
    """
    if colors is None:
        colors = [["blue"] * len(text) for text in texts]
    elif len(colors) == 1:
        colors = [colors * len(text) for text in texts]

    if len(texts) != len(heights):
        raise ValueError("Texts and heights must be of the same length")
    for idx, (text, height, color) in enumerate(zip(texts, heights, colors)):
        y_stack_pos = 0
        y_stack_neg = 0
        for _, (t, h, c) in enumerate(zip(text, height, color)):
            if h > 0:
                text_shape = make_text_elements(
                    t,
                    x=idx + (1 - width) / 2,
                    y=y_stack_pos,
                    width=width,
                    height=h,
                    color=c,
                    edgecolor=c,
                )
                y_stack_pos += h
                axes.add_patch(text_shape)
            elif h < 0:
                text_shape = make_text_elements(
                    t,
                    x=idx + (1 - width) / 2,
                    y=y_stack_neg,
                    width=width,
                    height=h,
                    color=c,
                    edgecolor=c,
                )
                y_stack_neg += h
                axes.add_patch(text_shape)

    axes.autoscale()
    axes.set_xlim(0, len(texts))


class ConsensusMotifPlotter:
    """
    A class for plotting consensus motifs.

    Parameters
    ----------
    - elements (list): A list of elements representing the nucleotides.
    - weights (list): A list of weights representing the scores of the elements.
    - colors (list, optional): A list of colors for the elements. Defaults to None.

    Methods
    -------
    - from_scores(scores, base_order="ACGT"): Creates a ConsensusMotifPlotter object from scores.
    - plot(axes): Adds the motif to an axes object.

    """

    def __init__(self, elements, weights, colors=None):
        self.n_elem = len(elements)
        self.colors = colors
        self.elements = elements
        self.weights = weights

    @classmethod
    def from_scores(cls, scores, base_order="ACGT"):
        """
        Creates a ConsensusMotifPlotter object from scores.

        Parameters
        ----------
        - scores (list): A list of scores representing the motif scores.
        - base_order (str, optional): The order of the nucleotide bases. Defaults to "ACGT".

        Returns
        -------
        - ConsensusMotifPlotter: A ConsensusMotifPlotter object.

        """
        nucleotides = [list(base_order)] * len(scores)
        colors = [["#008000", "#0000cc", "#cc0000", "#ffb300"]] * len(scores)
        sorted_nucleotides = np.array(nucleotides)
        sorted_scores = np.array(scores)
        sorted_colors = np.array(colors)
        order = np.absolute(scores).argsort()
        for i, _order in enumerate(order):
            sorted_scores[i, :] = sorted_scores[i, _order]
            sorted_nucleotides[i, :] = sorted_nucleotides[i, _order]
            sorted_colors[i, :] = sorted_colors[i, _order]
        return cls(sorted_nucleotides, sorted_scores, sorted_colors)

    def plot(self, axes):
        """
        Adds the motif to an axes object.

        Parameters
        ----------
        - axes: The axes object to which the motif will be added.

        Returns
        -------
        - None

        """
        make_stacked_bar_plot(
            axes, self.elements, self.weights, width=1, colors=self.colors
        )
