import numpy as np
import scipy
import torch


def zscore2pval_torch(footprint):
    """
    Convert z-scores to p-values using the torch library.

    Parameters
    ----------
    - footprint (torch.Tensor): A tensor containing z-scores.

    Returns
    -------
    - pval_log (torch.Tensor): A tensor containing the corresponding p-values in logarithmic scale.
    """
    # fill nan with 0
    footprint[torch.isnan(footprint)] = 0

    # Calculate the CDF of the normal distribution for the given footprint
    pval = torch.distributions.Normal(0, 1).cdf(footprint)

    # Clamp pval to prevent log(0) which leads to -inf. Use a very small value as the lower bound.
    eps = torch.finfo(pval.dtype).eps
    pval_clamped = torch.clamp(pval, min=eps)

    # Compute the negative log10, using the clamped values to avoid -inf
    pval_log = -torch.log10(pval_clamped)

    # Optionally, to handle values very close to 1 (which would result in a negative pval_log),
    # you can clamp the output to be non-negative. This is a design choice depending on your requirements.
    pval_log = torch.clamp(pval_log, min=0, max=10)

    return pval_log


def zscore2pval(footprint: np.ndarray) -> np.ndarray:
    """
    Convert z-scores to p-values using the scipy library.

    Parameters
    ----------
    - footprint (np.ndarray): An array containing z-scores.

    Returns
    -------
    - pval (np.ndarray): An array containing the corresponding p-values in logarithmic scale.
    """
    pval = scipy.stats.norm.cdf(footprint, 0, 1)
    pval = -np.log10(pval)
    pval[np.isnan(pval)] = 0
    return pval


def rz_conv(a: np.ndarray, n: int = 2) -> np.ndarray:
    """
    Apply convolution to the input array on the last dimension.

    Parameters
    ----------
    - a (np.ndarray): The input array.
    - n (int): The number of elements to convolve on.

    Returns
    -------
    - np.ndarray: The convolved array.
    """
    if n == 0:
        return a
    # a can be shape of (batch, sample,...,  x) and x will be the dim to be conv on
    # pad first:
    shapes = np.array(a.shape)
    shapes[-1] = n
    a = np.concatenate([np.zeros(shapes), a, np.zeros(shapes)], axis=-1)
    ret = np.cumsum(a, axis=-1)
    # ret[..., n * 2:] = ret[..., n * 2:] - ret[..., :-n * 2]
    # ret = ret[..., n * 2:]
    ret = ret[..., n * 2 :] - ret[..., : -n * 2]
    return ret
