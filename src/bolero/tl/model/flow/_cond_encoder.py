from dataclasses import field as dc_field
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ._utils import Layers_separate_input_t, Layers_t


class MLPBlock(nn.Module):
    """
    MLP block, translated from Flax linen to PyTorch.

    Parameters
    ----------
    input_dim
        Size of the last dimension of x.
    dims
        Sequence of hidden/output sizes.
    dropout_rate
        Dropout probability.
    act_last_layer
        Whether to apply activation on the final dense layer.
    act_fn
        Elementwise activation function (e.g. torch.nn.functional.silu).
    """

    def __init__(
        self,
        input_dim: int,
        dims: Sequence[int],
        dropout_rate: float = 0.0,
        act_last_layer: bool = True,
        act_type: str = "silu",
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.act_last_layer = act_last_layer

        dims_full = [input_dim] + list(dims)
        self.layers = nn.Sequential()
        self.output_dim = dims[-1]

        match act_type:
            case "silu":
                act_module = nn.SiLU
            case "gelu":
                act_module = nn.GELU
            case "relu":
                act_module = nn.ReLU

        for i in range(len(dims_full) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dims_full[i], dims_full[i + 1]),
                    act_module(),
                    nn.Dropout(dropout_rate),
                )
            )

        if not act_last_layer:
            self.layers[-1] = nn.Sequential(
                nn.Linear(dims_full[-2], dims_full[-1]),
                nn.Dropout(dropout_rate),
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: (batch_size, input_dim)
        training: if False, disables dropout (like Flax's deterministic=not training)
        """
        z = x
        if len(self.layers) == 0:
            return z

        z = self.layers(z)
        return z


class TokenAttentionPooling(nn.Module):
    """
    Implementation of TokenAttentionPooling from JAX in CellFlow
    cellflow.networks._utils.TokenAttentionPooling
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_dim: int = 64,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        assert qkv_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads
        self.dropout_rate = dropout_rate

        # learnable class token
        self.class_token = nn.Embedding(1, input_dim)
        # joint QKV projection
        self.qkv_proj = nn.Linear(input_dim, 3 * qkv_dim)
        # out projection
        self.out_proj = nn.Linear(qkv_dim, input_dim)

    def _build_mask(self, mask, B, S, device):
        """
        Build boolean attn_mask of shape (B, heads, L, L) with True=include
        """
        L = S + 1

        if mask is not None:
            if mask.ndim == 4:
                mask = mask[:, 0, :, :]  # (B, S, S)

            m = torch.zeros((B, L, L), dtype=torch.bool, device=device)
            # paylode mask (B, S, S)
            m[:, 1:, 1:] = mask
            # always allow row/col 0
            m[:, 0, :] = True
            m[:, :, 0] = True
        else:
            m = torch.ones((B, L, L), dtype=torch.bool, device=device)

        return repeat(m, "b l1 l2 -> b h l1 l2", h=self.num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, S, _ = x.shape
        device = x.device

        # prepend class token
        ids = torch.zeros(B, dtype=torch.long, device=device)
        cls = self.class_token(ids).unsqueeze(1)  # (B,1,D)
        z = torch.cat([cls, x], dim=1)  # (B, S+1, D)

        # project to QKV and split
        qkv = self.qkv_proj(z)  # (B, L, 3Q)
        q, k, v = rearrange(
            qkv, "b l (t h d) -> t b h l d", t=3, h=self.num_heads, d=self.head_dim
        )

        # build mask and run flash-attention
        attn_mask = self._build_mask(mask, B, S, device)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if self.training else 0,
            is_causal=False,
        )  # (B, heads, L, head_dim)

        # merge heads & project out
        out = rearrange(out, "b h l d -> b l (h d)")  # (B, L, Q)
        out = self.out_proj(out)  # (B, L, D)

        return out[:, 0, :]  # (B, D)


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: (B, S, D)
        mask: (B, D)
        """
        return torch.mean(x * mask, dim=-2)


class SequentialWithArgs(nn.Sequential):
    output_dim: int

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: (B, S, D)
        args: (B, S, S)
        kwargs: (B, S, S)
        """
        for module in self:
            if isinstance(module, nn.Identity):
                continue
            x = module(x, *args, **kwargs)
        return x


def _get_layers(layers: Layers_t) -> list[nn.Module]:
    """Get modules from layer parameters."""
    modules = []
    total_output_dim = 0
    for layer in layers:
        try:
            layer = dict(layer)
        except ValueError as e:
            raise ValueError(f"Layer {layer} is not a dict.") from e
        layer_type = layer.pop("layer_type", "identity")

        if layer_type == "mlp":
            output_dim = layer["dims"][-1]
            lay = MLPBlock(**layer)
        elif layer_type == "identity":
            output_dim = layer["input_dim"]
            lay = nn.Identity()
        elif layer_type == "self_attention":
            raise NotImplementedError("SeedAttentionPooling not implemented yet.")
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        modules.append(lay)
        total_output_dim += output_dim

    modules = SequentialWithArgs(*modules)
    modules.output_dim = total_output_dim
    return modules


def _get_masks(mask_value, conditions):
    # conditions: dict of torch.Tensor, each (B, S, D)
    padded_flags = [torch.all(c == mask_value, dim=-1) for c in conditions.values()]
    stacked = torch.stack(padded_flags, dim=0)  # (num_conds, B, S)
    all_padded = torch.all(stacked, dim=0)  # (B, S)
    mask = (~all_padded).to(torch.int32)  # (B, S)
    mask = mask.unsqueeze(-1)  # (B, S, 1)
    attn = mask & mask.transpose(1, 2)  # (B, S, S)
    attn = attn.unsqueeze(1)  # (B, 1, S, S)
    return mask, attn


class ConditionEncoder(nn.Module):
    """
    Encoder for conditions represented as sets of perturbations.
    See usage example in test code below.
    Also see cellflow.networks._set_encoders.ConditionEncoder
    """

    def __init__(
        self,
        output_dim: int,
        condition_mode: Literal["deterministic", "stochastic"] = "deterministic",
        regularization: float = 0.0,
        decoder: bool = False,
        pooling: Literal[
            "mean", "attention_token", "attention_seed"
        ] = "attention_token",
        pooling_kwargs: dict[str, Any] = None,
        covariates_not_pooled: Sequence[str] = dc_field(default_factory=list),
        layers_before_pool: Layers_t | Layers_separate_input_t = dc_field(
            default_factory=list
        ),
        layers_after_pool: Layers_t = dc_field(default_factory=list),
        layers_decoder: Layers_t = dc_field(default_factory=list),
        output_dropout: float = 0.0,
        mask_value: float = 0.0,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.condition_mode = condition_mode
        self.regularization = regularization
        self.decoder = decoder
        self.pooling = pooling
        self.pooling_kwargs = pooling_kwargs if pooling_kwargs else {}
        self.covariates_not_pooled = covariates_not_pooled
        self.layers_before_pool = layers_before_pool
        self.layers_after_pool = layers_after_pool
        self.layers_decoder = layers_decoder
        self.output_dropout = output_dropout
        self.mask_value = mask_value

        # modules before pooling
        self.separate_inputs = isinstance(self.layers_before_pool, dict)
        if self.separate_inputs:
            # different layers for different inputs, before_pool_modules is of type Layers_separate_input_t
            self.before_pool_modules: nn.ModuleDict = nn.ModuleDict(
                {
                    key: _get_layers(layers)
                    for key, layers in self.layers_before_pool.items()  # type: ignore[union-attr]
                }
            )
        else:
            raise NotImplementedError(
                "Currently only separate_inputs are supported, "
                "please provide layer specification as a dict."
            )
            # self.before_pool_modules = _get_layers(self.layers_before_pool)  # type: ignore[arg-type]

        # pooling
        if self.pooling == "mean":
            self.pool_module = MeanPooling()
        elif self.pooling == "attention_token":
            total_before_pool_dim = sum(
                layer.output_dim for layer in self.before_pool_modules.values()
            )
            _kwargs = {
                "input_dim": total_before_pool_dim,
                **self.pooling_kwargs,
            }
            self.pool_module = TokenAttentionPooling(**_kwargs)
        elif self.pooling == "attention_seed":
            raise NotImplementedError("attention_seed pooling not implemented yet")
        else:
            raise NotImplementedError(f"Unknown pooling type: {self.pooling}")

        # modules after pooling
        self.after_pool_modules_mean = _get_layers(self.layers_after_pool)
        assert self.output_dim == self.after_pool_modules_mean.output_dim, (
            f"self.output_dim ({self.output_dim}) "
            f"dose not match after_pool_modules.output_dim "
            f"({self.after_pool_modules_mean.output_dim})"
        )

        if self.condition_mode == "stochastic":
            self.after_pool_modules_var = _get_layers(self.layers_after_pool)
        return

    def forward(self, conditions: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Given a dictionary of conditions, apply the encoder.

        Parameters
        ----------
        conditions: dict
            example data structure:
            {
                'drug': (B, S, D1),
                'dose': (B, S, D2),
                ...
            }

        Returns
        -------
        conditions_mean: (B, output_dim)
        conditions_logvar: (B, output_dim)
        """
        mask, attention_mask = _get_masks(self.mask_value, conditions)

        # apply modules before pooling
        if self.separate_inputs:
            processed_inputs_pooling = []
            processed_inputs_other = []
            for pert_cov, conditions_i in conditions.items():
                # apply separate modules for all inputs
                _module = self.before_pool_modules[pert_cov]
                conditions_i = _module(conditions_i, attention_mask)

                if pert_cov in self.covariates_not_pooled:
                    # only keep first set element for covariates that are not pooled
                    processed_inputs_other.append(conditions_i[:, 0, :])
                else:
                    processed_inputs_pooling.append(conditions_i)

            conditions_pooling_arr = torch.concatenate(processed_inputs_pooling, dim=-1)
            conditions_not_pooled = (
                torch.concatenate(processed_inputs_other, dim=-1)
                if self.covariates_not_pooled
                else None
            )
        else:
            assert NotImplementedError("Currently only separate_inputs are supported.")
            # # by default, no modules before pooling for covariates that are not pooled
            # if self.covariates_not_pooled:
            #     # divide conditions into pooled and not pooled
            #     conditions_not_pooled = []
            #     conditions_pooling = []
            #     for pert_cov in conditions:
            #         if pert_cov in self.covariates_not_pooled:
            #             conditions_not_pooled.append(conditions[pert_cov][:, 0, :])
            #         else:
            #             conditions_pooling.append(conditions[pert_cov])
            #     conditions_not_pooled = torch.concatenate(conditions_not_pooled, dim=-1)
            #     conditions_pooling_arr = torch.concatenate(conditions_pooling, dim=-1)

            #     # apply modules to pooled covariates
            #     conditions_pooling_arr = self.before_pool_modules(
            #         conditions_pooling_arr,
            #         attention_mask,
            #     )
            # else:
            #     conditions = torch.concatenate(list(conditions.values()), dim=-1)
            #     conditions_pooling_arr = self.before_pool_modules(
            #         conditions,
            #         attention_mask,
            #     )

        # pooling
        pool_mask = mask if self.pooling == "mean" else attention_mask
        conditions = self.pool_module(conditions_pooling_arr, pool_mask)
        if self.covariates_not_pooled:
            conditions = torch.concatenate([conditions, conditions_not_pooled], dim=-1)

        # apply modules after pooling
        conditions_mean = self.after_pool_modules_mean(conditions, None)

        if self.condition_mode == "stochastic":
            conditions_logvar = self.after_pool_modules_var(conditions, None)
        else:
            conditions_logvar = torch.zeros_like(conditions_mean)
        return conditions_mean, conditions_logvar


if __name__ == "__main__":
    # Simple test
    # test_condition_encoder_torch.py

    # import pytest
    import numpy as np
    import torch

    # Replace this with the actual import path to your PyTorch implementation

    # --- build the same “cond” dictionary, but as NumPy first ---
    cond_np = {
        "pert1": np.ones((1, 3, 3), dtype=np.float32),
        "pert2": np.ones((1, 3, 10), dtype=np.float32),
        "pert3": np.ones((1, 3, 5), dtype=np.float32),
    }
    # zero out the last slot in each
    for k in list(cond_np):
        cond_np[k][0, 2, :] = 0.0
    # add the “skip_pool” covariate
    cond_np["pert4_skip_pool"] = np.ones((1, 3, 5), dtype=np.float32)

    # convert to torch tensors once
    cond_torch = {k: torch.from_numpy(v) for k, v in cond_np.items()}

    # exactly the same layer‐descriptions as in your JAX test
    layers_before_pool_all = [
        {
            "pert1": (
                {"layer_type": "mlp", "dims": (32, 32), "input_dim": 3},
                # {"layer_type": "self_attention", "num_heads": 4, "qkv_dim": 32},
            ),
            "pert2": ({"layer_type": "mlp", "dims": (32, 32), "input_dim": 10},),
            "pert3": ({"input_dim": 5},),
            "pert4_skip_pool": ({"input_dim": 5},),
        },
        (),
    ]

    layers_after_pool_all = [
        ({"layer_type": "mlp", "dims": (32, 32, 5), "input_dim": 74},),
        (),
    ]

    # class TestConditionEncoderTorch:
    # @pytest.mark.parametrize("pooling", ["mean", "attention_token"])
    # @pytest.mark.parametrize("covariates_not_pooled", [[], ["pert4_skip_pool"]])
    # @pytest.mark.parametrize("layers_before_pool", layers_before_pool)
    # @pytest.mark.parametrize("layers_after_pool", layers_after_pool)
    # @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    # @pytest.mark.parametrize("regularization", [0.0, 0.1])
    def test_forward_shapes_and_consistency(
        pooling,
        covariates_not_pooled,
        layers_before_pool,
        layers_after_pool,
        condition_mode,
        regularization,
    ):
        # instantiate and run in eval() so dropout (if any) is disabled
        model = ConditionEncoder(
            output_dim=5,
            condition_mode=condition_mode,
            regularization=regularization,
            pooling=pooling,
            covariates_not_pooled=covariates_not_pooled,
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool,
            output_dropout=0.1,
        )
        model.eval()

        # forward pass
        out1, logvar1 = model(cond_torch)

        # basic shape checks
        assert isinstance(out1, torch.Tensor)
        assert isinstance(logvar1, torch.Tensor)
        assert out1.shape == (1, 5), out1.shape
        assert logvar1.shape == (1, 5)

        # running it again (in eval mode) should give the same result
        out2, logvar2 = model(cond_torch)
        assert torch.allclose(out1, out2, atol=1e-6)
        assert torch.allclose(logvar1, logvar2, atol=1e-6)

        # no NaNs
        assert not torch.isnan(out1).any()
        assert not torch.isnan(logvar1).any()
        return model

    pooling = "attention_token"
    covariates_not_pooled = []
    condition_mode = "stochastic"
    regularization = 0.1
    layers_before_pool = layers_before_pool_all[0]
    layers_after_pool = layers_after_pool_all[0]
    model = test_forward_shapes_and_consistency(
        pooling,
        covariates_not_pooled,
        layers_before_pool,
        layers_after_pool,
        condition_mode,
        regularization,
    )

    pooling = "mean"
    covariates_not_pooled = []
    condition_mode = "deterministic"
    regularization = 0.1
    layers_before_pool = layers_before_pool_all[0]
    layers_after_pool = layers_after_pool_all[0]
    model = test_forward_shapes_and_consistency(
        pooling,
        covariates_not_pooled,
        layers_before_pool,
        layers_after_pool,
        condition_mode,
        regularization,
    )
