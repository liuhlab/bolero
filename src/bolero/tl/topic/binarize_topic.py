from typing import Optional

import numpy as np
import pandas as pd


def _norm_topics(x):
    """
    Smooth topic-region distributions.
    """
    return x * (np.log(x + 1e-100) - np.sum(np.log(x + 1e-100)) / len(x))


def _smooth_topics_f(topic_region):
    """
    Smooth topic-region distributions.

    Parameters
    ----------
    topic_region: `class::pd.DataFrame`
            A pandas dataframe with topic-region distributions (with topics as columns and regions as rows)

    Return
    ---------
    pd.DataFrame
    """
    topic_region_np = np.apply_along_axis(_norm_topics, 1, topic_region.values)
    topic_region = pd.DataFrame(
        topic_region_np, index=topic_region.index, columns=topic_region.columns
    )
    return topic_region


def _threshold_otsu(array, nbins=100):
    """
    Apply Otsu threshold on topic-region distributions [Otsu, 1979].

    Parameters
    ----------
    array: `class::np.array`
            Array containing the region values for the topic to be binarized.
    nbins: int
            Number of bins to use in the binarization histogram

    Return
    ---------
    float
            Binarization threshold.

    Reference
    ---------
    Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and
    cybernetics, 9(1), pp.62-66.
    """
    hist, bin_centers = _histogram(array, nbins)
    hist = hist.astype(float)
    # Class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def _histogram(array, nbins=100):
    """
    Draw histogram from distribution and identify centers.

    Parameters
    ----------
    array: `class::np.array`
            Scores distribution
    nbins: int
            Number of bins to use in the histogram

    Return
    ---------
    float
            Histogram values and bin centers.
    """
    array = array.ravel().flatten()
    hist, bin_edges = np.histogram(array, bins=nbins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return hist, bin_centers


def binarize_topics(
    topic_dist: pd.DataFrame,
    nbins: Optional[int] = 100,
):
    """
    Binarize topic distributions.

    Parameters
    ----------
    nbins: int, optional
        Number of bins to use in the histogram used for otsu, yen and li thresholding. Default: 100

    Return
    ---------
    dict
        A dictionary containing a pd.DataFrame with the selected regions with region names as indexes and a topic score
        column.

    References
    ----------
    Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and
    cybernetics, 9(1), pp.62-66.
    Yen, J.C., Chang, F.J. and Chang, S., 1995. A new criterion for automatic multilevel thresholding. IEEE Transactions on
    Image Processing, 4(3), pp.370-378.
    Li, C.H. and Lee, C.K., 1993. Minimum cross entropy thresholding. Pattern recognition, 26(4), pp.617-625.
    Van de Sande, B., Flerin, C., Davie, K., De Waegeneer, M., Hulselmans, G., Aibar, S., Seurinck, R., Saelens, W., Cannoodt, R.,
    Rouchon, Q. and Verbeiren, T., 2020. A scalable SCENIC workflow for single-cell gene regulatory network analysis. Nature Protocols,
    15(7), pp.2247-2276.
    """
    # smooth topics:
    topic_dist = _smooth_topics_f(topic_dist)

    binarized_topics = {}

    for i, col in enumerate(topic_dist.columns):
        l = np.asarray(topic_dist.iloc[:, i])
        l_norm = (l - np.min(l)) / np.ptp(l)

        thr = _threshold_otsu(l_norm, nbins=nbins)
        binarized_topics[col] = pd.DataFrame(topic_dist.iloc[l_norm > thr, i])

    # binary empty df
    binarized_df = pd.DataFrame(
        np.zeros(shape=topic_dist.shape, dtype=bool),
        index=topic_dist.index,
        columns=topic_dist.columns,
    )
    for k, v in binarized_topics.items():
        binarized_df.loc[v.index, k] = True
    return binarized_df
