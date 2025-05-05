from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import r2_score

__all__ = [
    "compute_scalar_mmd",
    "compute_r_squared",
    "compute_e_distance",
    "maximum_mean_discrepancy",
]


def compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the R squared score between means of the true (x) and predicted (y) distributions.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the R squared score.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))


def compute_e_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the energy distance between x and y as in :cite:`Peidli2024`.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the energy distance value.
    """
    sigma_X = pairwise_squeuclidean(x, x).mean()
    sigma_Y = pairwise_squeuclidean(y, y).mean()
    delta = pairwise_squeuclidean(x, y).mean()
    return 2 * delta - sigma_X - sigma_Y


def pairwise_squeuclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared euclidean distances."""
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def rbf_kernel_fast(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Approximate the RBF kernel."""
    xx = (x**2).sum(1)
    yy = (y**2).sum(1)
    xy = x @ y.T
    sq_distances = xx[:, None] + yy - 2 * xy
    return torch.exp(-gamma * sq_distances)


def maximum_mean_discrepancy(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gamma
            Parameter for the rbf kernel.

    Returns
    -------
        A scalar denoting the squared maximum mean discrepancy loss.
    """
    xx = rbf_kernel_fast(x, x, gamma)
    xy = rbf_kernel_fast(x, y, gamma)
    yy = rbf_kernel_fast(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(
    x: torch.Tensor, y: torch.Tensor, gammas: Sequence[float] | None = None
) -> float:
    """Compute the Mean Maximum Discrepancy (MMD) across different length scales

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gammas
            A sequence of values for the paramater gamma of the rbf kernel.

    Returns
    -------
        A scalar denoting the average MMD over all gammas.
    """
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [maximum_mean_discrepancy(x, y, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))
