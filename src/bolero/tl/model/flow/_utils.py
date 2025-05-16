import math
from typing import Any, Sequence

import anndata
import numpy as np
import torch

Layers_t = Sequence[dict[str, Any]]
Layers_separate_input_t = dict[str, Layers_t]


def write_predictions(
    adata: anndata.AnnData,
    predictions: dict[str, np.ndarray],
    key_added_prefix: str,
) -> None:
    """Write predictions to AnnData object obsm."""
    for pred_key, pred_value in predictions.items():
        if isinstance(pred_value, torch.Tensor):
            pred_value = pred_value.detach().cpu().numpy()

        if pred_value.ndim == 2:
            adata.obsm[f"{key_added_prefix}{pred_key}"] = pred_value
        elif pred_value.ndim == 3:
            for i in range(pred_value.shape[2]):
                adata.obsm[f"{key_added_prefix}{pred_key}_{i}"] = pred_value[..., i]
        else:
            raise ValueError(
                f"Predictions for '{pred_key}' have an invalid shape: {pred_value.shape}"
            )


def cyclical_time_encoder(t: torch.Tensor, n_freqs: int = 128) -> torch.Tensor:
    """
    Encode time t into a cyclical representation using cosine and sine functions.

    Torch implementation of the JAX function here:
    https://github.com/ott-jax/ott/blob/67d5131d7b2d46964acc3f6e39def43ec7248db1/src/ott/neural/networks/layers/time_encoder.py#L19

    Args:
        t: Tensor of shape [n, 1]
        n_freqs: Number of frequency components

    Returns
    -------
        Tensor of shape [n, 2 * n_freqs]
    """
    freq = 2 * math.pi * torch.arange(n_freqs, dtype=t.dtype, device=t.device)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    t = t * freq  # [n, 1] * [n_freqs] -> broadcast to [n, n_freqs]
    return torch.cat([torch.cos(t), torch.sin(t)], dim=-1)
