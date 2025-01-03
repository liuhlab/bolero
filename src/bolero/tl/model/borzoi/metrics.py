from typing import Optional

import torch
from einops import rearrange
from torch.nn import functional as F
from torchmetrics import Metric


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
        assert (
            preds.shape == target.shape
        ), f"Shapes do not match: {preds.shape} != {target.shape}"

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        """Computes the mean pearson correlation coefficient"""
        true_mean = self.true / (self.count + 1e-7)
        pred_mean = self.pred / (self.count + 1e-7)

        covariance = (
            self.product
            - true_mean * self.pred
            - pred_mean * self.true
            + self.count * true_mean * pred_mean
        )

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / (tp_var + 1e-7)
        return correlation

    def compute_tensor(self):
        """Computes the mean pearson correlation coefficient and returns it as a tensor"""
        corr = self.compute()
        if isinstance(corr, torch.Tensor):
            corr = corr.cpu().numpy()

        if corr.ndim == 0:
            corr = corr[None]
        return torch.tensor(corr)

    def get_corr_str(self, reduce_cufoff=5):
        """Computes the mean pearson correlation coefficient and returns it as a string"""
        corr = self.compute()
        if isinstance(corr, torch.Tensor):
            corr = corr.cpu().numpy()

        if corr.ndim == 0:
            corr = corr[None]

        if corr.size > reduce_cufoff:
            return f"{corr.mean():.3f}"

        return ", ".join([f"{c:.3f}" for c in corr])


def _log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def poisson_loss(pred, target):
    """Compute Poisson loss."""
    return pred - target * _log(pred)


def poisson_multinomial(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    total_weight: float = 0.16,
    epsilon: float = 1e-7,
    return_breakdown: bool = False,
    loss_chunks: int = 1,
    position_weights: Optional[torch.Tensor] = None,
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
        n_chunks (int): Number of chunks to split the sequence length into.
    """
    seq_len = y_true.shape[-1]
    if loss_chunks > 1:
        if seq_len % loss_chunks != 0:
            raise ValueError("Sequence length must be divisible by n_chunks.")
        y_true = rearrange(y_true, "b c (n s) -> (b n) c s", n=loss_chunks)
        y_pred = rearrange(y_pred, "b c (n s) -> (b n) c s", n=loss_chunks)

    valid_mask = torch.isfinite(y_true)

    weight_scale = valid_mask.sum(dim=-1).float()

    if position_weights is not None:
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
    final_loss = (
        1 - total_weight
    ) * multinomial_term + total_weight * poisson_term  # (bs, c)

    loss_breakdown = {
        "multinomial": multinomial_term.detach(),
        "poisson": poisson_term.detach(),
    }

    if return_breakdown:
        return final_loss, loss_breakdown
    else:
        return final_loss


def bce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, return_breakdown=False):
    """
    Compute binary cross-entropy loss.

    Input shape: (bs, c, seq_len)
    Loss output shape: (bs, c)
    """
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none").mean(-1)

    loss_breakdown = {"bce_loss": loss.detach()}

    if return_breakdown:
        return loss, loss_breakdown
    return loss


def mse_diff_loss(y_pred_a, y_pred_b, y_true_a, y_true_b):
    """
    Compute the MSE loss between paired log fold changes.

    data shape (bs, c, seq_len)
    loss shape (bs, c)
    """
    delta_a = torch.log1p(y_pred_a) - torch.log1p(y_true_a)  # fold change diff
    delta_b = torch.log1p(y_pred_b) - torch.log1p(y_true_b)  # fold change diff
    loss = F.mse_loss(delta_a, delta_b, reduction="none").mean(dim=-1)
    return loss
