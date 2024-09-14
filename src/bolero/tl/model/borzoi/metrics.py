from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric


class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True

    def __init__(self, n_channels: int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims = (0, 1)
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


def poisson_multinomial(
    y_true,
    y_pred,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 4,
    epsilon: float = 1e-7,
    rescale: bool = False,
):
    """Poisson decomposition with multinomial specificity term in PyTorch.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape [B, L, T].
        y_pred (torch.Tensor): Predicted tensor of shape [B, L, T].
        total_weight (float): Weight of the Poisson total term.
        weight_range (float): Range of the position-specific weights.
        weight_exp (int): Exponent of the position-specific weights,
            larger values put more positions to 1 from center to both ends.
        epsilon (float): Small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
    """
    seq_len = y_true.shape[1]

    # Position-specific weights (similar to TensorFlow code)
    if weight_range < 1:
        raise ValueError("Poisson Multinomial weight_range must be >=1")
    elif weight_range == 1:
        weigh_by_position = False
        weight_scale = seq_len
    else:
        # this is aim to create an smooth position wise weight
        # that is high in the middle and low in the ends
        # This is not actually used in the paper
        weight_exp = max(2, weight_exp // 2 * 2)  # Ensure weight_exp is even number
        pos_start = -(seq_len / 2 - 0.5)
        pos_end = seq_len / 2 + 0.5
        positions = torch.arange(pos_start, pos_end, dtype=torch.float32).to(
            y_true.device
        )
        sigma = -pos_start / (np.log(weight_range)) ** (1 / weight_exp)
        position_weights = torch.exp(-((positions / sigma) ** weight_exp))
        position_weights /= torch.max(position_weights)
        position_weights = position_weights.unsqueeze(0).unsqueeze(-1)

        weigh_by_position = True
        weight_scale = torch.sum(position_weights)

    if weigh_by_position:
        # Apply position weights to true and predicted values
        y_true = y_true * position_weights
        y_pred = y_pred * position_weights

    # Poisson loss computation (sum across lengths, then compute loss)
    s_true = torch.sum(y_true, dim=-2)  # B x T
    s_pred = torch.sum(y_pred, dim=-2)  # B x T

    poisson_term = F.poisson_nll_loss(
        s_pred, s_true, log_input=False, reduction="none"
    )  # B x T
    poisson_term /= weight_scale

    # Add epsilon to avoid log(0)
    y_true += epsilon
    y_pred += epsilon

    # Normalize predictions to sum to one (multinomial probability)
    p_pred = y_pred / s_pred.unsqueeze(-2)  # B x L x T

    # Multinomial loss
    pl_pred = torch.log(p_pred)  # B x L x T
    multinomial_term = torch.sum(-y_true * pl_pred, dim=-2)  # B x T
    multinomial_term /= weight_scale

    # Combine Poisson and Multinomial terms
    loss_raw = multinomial_term + total_weight * poisson_term  # B x T

    # Rescale if required
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale
