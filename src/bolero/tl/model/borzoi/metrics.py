from typing import Optional, Union

import numpy as np
import torch
from torchmetrics import Metric

from .utils import clamp_sqrt_large_value


class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True

    def __init__(self, n_channels: int, dist_sync_on_step=False, reduce_dims=(0, 2)):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims = reduce_dims
        self.add_state(
            "product",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "true",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "true_squared",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_squared",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.zeros(n_channels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Updates the metric with new predictions and target"""
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        """Computes the mean pearson correlation coefficient"""
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (
            self.product
            - true_mean * self.pred
            - pred_mean * self.true
            + self.count * true_mean * pred_mean
        )

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation


def get_position_weights(
    seq_len: int, weight_range: float, weight_exp: int, device: str
):
    """
    Generate smooth position-wise weights that are high in the middle and low at the ends.

    This function creates a set of weights for positions in a sequence, where the weights
    are highest in the middle of the sequence and decrease towards the ends. The weights
    are normalized to have a maximum value of 1. Note that this function is not used in
    the referenced paper.

    Args:
        seq_len (int): The length of the sequence.
        weight_range (float): The range of the weights.
        weight_exp (int): The exponent used to control the smoothness of the weights.
                          This value is adjusted to be an even number.
        device (str): The device to which the tensors are moved (e.g., 'cpu' or 'cuda').

    Returns
    -------
        torch.Tensor: A tensor containing the position-wise weights, with shape (1, seq_len, 1).
    """
    weight_exp = max(2, weight_exp // 2 * 2)  # Ensure weight_exp is even number
    pos_start = -(seq_len / 2 - 0.5)
    pos_end = seq_len / 2 + 0.5
    positions = torch.arange(pos_start, pos_end, dtype=torch.float32).to(device)
    sigma = -pos_start / (np.log(weight_range)) ** (1 / weight_exp)
    position_weights = torch.exp(-((positions / sigma) ** weight_exp))
    position_weights /= torch.max(position_weights)
    position_weights = position_weights.unsqueeze(0).unsqueeze(-1)
    return position_weights


def _log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def poisson_loss(pred, target):
    """Compute Poisson loss."""
    return pred - target * _log(pred)


def poisson_multinomial(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    soft_clamp: Union[int, torch.Tensor, None] = None,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 4,
    epsilon: float = 1e-7,
    rescale: bool = False,
    return_breakdown: bool = False,
):
    """
    Compositional loss containing the overall poisson term and position-wise multinomial term.

    NaN values in y_true are ignored in the loss computation.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (bs, c, seq_len).
        y_pred (torch.Tensor): Predicted tensor of shape (bs, c, seq_len).
        total_weight (float): Weight of the Poisson total term.
        weight_range (float): Range of the position-specific weights.
        weight_exp (int): Exponent of the position-specific weights,
            larger values put more positions to 1 from center to both ends.
        epsilon (float): Small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
    """
    seq_len = y_true.shape[-1]

    if soft_clamp is not None:
        y_true = clamp_sqrt_large_value(y_true, soft_clamp)

    valid_mask = torch.isfinite(y_true)

    # Position-specific weights (similar to TensorFlow code)
    if weight_range < 1:
        raise ValueError("Poisson Multinomial weight_range must be >=1")
    elif weight_range == 1:
        weight_scale = valid_mask.sum(dim=-1).float()
    else:
        position_weights = get_position_weights(
            seq_len, weight_range, weight_exp, y_true.device
        )
        # Apply position weights to true and predicted values
        y_true = y_true * position_weights
        y_pred = y_pred * position_weights

        weight_scale = torch.sum(position_weights)

    # Poisson loss computation (sum across lengths, then compute loss)
    s_true = torch.nansum(y_true, dim=-1)  # (bs, c)
    s_pred = torch.nansum(y_pred, dim=-1)  # (bs, c)

    poisson_term = poisson_loss(s_pred, s_true)  # (bs, c)
    poisson_term /= weight_scale

    # Add epsilon to avoid log(0)
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    # Normalize predictions to sum to one (multinomial probability)
    p_pred = y_pred / s_pred.unsqueeze(-1)  # (bs, c, seq_len)

    # Multinomial loss
    pl_pred = torch.log(p_pred)  # (bs, c, seq_len)
    multinomial_term = torch.nansum(-y_true * pl_pred, dim=-1)  # (bs, c)
    multinomial_term /= weight_scale

    # Combine Poisson and Multinomial terms
    loss_raw = multinomial_term + total_weight * poisson_term  # (bs, c)

    # Rescale if required
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    loss_breakdown = {
        "multinomial": multinomial_term.detach(),
        "poisson": poisson_term.detach(),
    }

    if return_breakdown:
        return loss_rescale, loss_breakdown
    else:
        return loss_rescale
