import dataclasses
import math
from typing import Any, Literal, Sequence

import torch
from torch import nn

from ._cond_encoder import ConditionEncoder, MLPBlock
from ._utils import Layers_separate_input_t, Layers_t


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
    t = t * freq  # [n, 1] * [n_freqs] -> broadcast to [n, n_freqs]
    return torch.cat([torch.cos(t), torch.sin(t)], dim=-1)


class ConditionalVelocityField(nn.Module):
    def __init__(
        self,
        output_dim: int,
        max_combination_length: int,
        condition_mode: Literal["deterministic", "stochastic"] = "deterministic",
        regularization: float = 1.0,
        encode_conditions: bool = True,
        condition_embedding_dim: int = 32,
        covariates_not_pooled: Sequence[str] = None,
        pooling: Literal[
            "mean", "attention_token", "attention_seed"
        ] = "attention_token",
        pooling_kwargs: dict[str, Any] = None,
        layers_before_pool: Layers_separate_input_t | Layers_t = None,
        layers_after_pool: Layers_t = None,
        cond_output_dropout: float = 0.0,
        mask_value: float = 0.0,
        condition_encoder_kwargs: dict[str, Any] = None,
        act_type: str = "silu",
        time_freqs: int = 1024,
        time_encoder_dims: Sequence[int] = (1024, 1024, 1024),
        time_encoder_dropout: float = 0.0,
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
        hidden_dropout: float = 0.0,
        conditioning: Literal["concatenation"] = "concatenation",
        conditioning_kwargs: dict[str, Any] = None,
        decoder_dims: Sequence[int] = (1024, 1024, 1024),
        decoder_dropout: float = 0.0,
        layer_norm_before_concatenation: bool = False,
        linear_projection_before_concatenation: bool = False,
    ):
        """Initialize the network."""
        super().__init__()

        self.output_dim = output_dim
        self.emb_input_dim = output_dim
        self.max_combination_length = max_combination_length
        self.condition_mode = condition_mode
        self.regularization = regularization
        self.encode_conditions = encode_conditions
        self.condition_embedding_dim = condition_embedding_dim
        self.covariates_not_pooled = (
            covariates_not_pooled if covariates_not_pooled is not None else []
        )
        self.pooling = pooling
        self.pooling_kwargs = pooling_kwargs
        self.layers_before_pool = (
            layers_before_pool if layers_before_pool is not None else {}
        )
        self.layers_after_pool = (
            layers_after_pool if layers_after_pool is not None else {}
        )
        self.cond_output_dropout = cond_output_dropout
        self.mask_value = mask_value
        self.condition_encoder_kwargs = (
            condition_encoder_kwargs if condition_encoder_kwargs is not None else {}
        )
        self.act_type = act_type
        self.time_freqs = time_freqs
        self.time_encoder_dims = time_encoder_dims
        self.time_encoder_dropout = time_encoder_dropout
        self.hidden_dims = hidden_dims
        self.hidden_dropout = hidden_dropout
        self.conditioning = conditioning
        self.conditioning_kwargs = (
            conditioning_kwargs if conditioning_kwargs is not None else {}
        )
        self.decoder_dims = decoder_dims
        self.decoder_dropout = decoder_dropout
        self.layer_norm_before_concatenation = layer_norm_before_concatenation
        self.linear_projection_before_concatenation = (
            linear_projection_before_concatenation
        )

        if isinstance(self.conditioning_kwargs, dataclasses.Field):
            conditioning_kwargs = dict(self.conditioning_kwargs.default_factory())
        else:
            conditioning_kwargs = dict(self.conditioning_kwargs)
        if self.encode_conditions:
            self.condition_encoder = ConditionEncoder(
                condition_mode=self.condition_mode,
                regularization=self.regularization,
                output_dim=self.condition_embedding_dim,
                pooling=self.pooling,
                pooling_kwargs=self.pooling_kwargs,
                layers_before_pool=self.layers_before_pool,
                layers_after_pool=self.layers_after_pool,
                covariates_not_pooled=self.covariates_not_pooled,
                mask_value=self.mask_value,
                **self.condition_encoder_kwargs,
            )

        self.layer_cond_output_dropout = nn.Dropout(self.cond_output_dropout)
        self.layer_norm_condition = (
            nn.LayerNorm(self.condition_encoder.output_dim)
            if self.layer_norm_before_concatenation
            else nn.Identity()
        )

        self.time_encoder = MLPBlock(
            input_dim=self.time_freqs * 2,
            dims=self.time_encoder_dims,
            act_type=self.act_type,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = (
            nn.LayerNorm(self.time_encoder.output_dim)
            if self.layer_norm_before_concatenation
            else nn.Identity()
        )

        self.x_encoder = MLPBlock(
            input_dim=self.emb_input_dim,
            dims=self.hidden_dims,
            act_type=self.act_type,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(
                False if self.linear_projection_before_concatenation else True
            ),
        )
        self.layer_norm_x = (
            nn.LayerNorm(self.x_encoder.output_dim)
            if self.layer_norm_before_concatenation
            else nn.Identity()
        )

        if self.conditioning == "concatenation":
            decoder_input_dim = (
                self.time_encoder.output_dim
                + self.x_encoder.output_dim
                + self.condition_encoder.output_dim
            )
        else:
            raise NotImplementedError("Only concatenation conditioning is implemented.")

        self.decoder = MLPBlock(
            input_dim=decoder_input_dim,
            dims=self.decoder_dims,
            act_type=self.act_type,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(
                False if self.linear_projection_before_concatenation else True
            ),
        )

        self.output_layer = nn.Linear(self.decoder.output_dim, self.output_dim)

        if self.conditioning == "film":
            raise NotImplementedError("Film conditioning is not implemented yet.")
            # self.film_block = FilmBlock(
            #     input_dim=self.hidden_dims[-1],
            #     cond_dim=self.time_encoder_dims[-1] + self.condition_embedding_dim,
            #     **conditioning_kwargs,
            # )
        elif self.conditioning == "resnet":
            raise NotImplementedError("resnet conditioning is not implemented yet.")
            # self.resnet_block = ResNetBlock(
            #     input_dim=self.hidden_dims[-1],
            #     **conditioning_kwargs,
            # )
        elif self.conditioning == "concatenation":
            if len(conditioning_kwargs) > 0:
                raise ValueError(
                    "If `conditioning=='concatenation' mode, no conditioning kwargs can be passed."
                )
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}")

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        cond: dict[str, torch.Tensor],
        encoder_noise: torch.Tensor | None = None,
    ):
        squeeze = x_t.ndim == 1
        if not self.encode_conditions:
            cond_embedding = torch.concatenate(list(cond.values()), dim=-1)
        else:
            cond_mean, cond_logvar = self.condition_encoder(cond)
            if self.condition_mode == "deterministic":
                cond_embedding = cond_mean
            else:
                cond_embedding = cond_mean + encoder_noise * torch.exp(
                    cond_logvar / 2.0
                )
        cond_embedding = self.layer_cond_output_dropout(cond_embedding)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        t_encoded = cyclical_time_encoder(t, n_freqs=self.time_freqs)
        t_encoded = self.time_encoder(t_encoded)
        t_encoded = self.layer_norm_time(t_encoded)

        x_encoded = self.x_encoder(x_t)
        x_encoded = self.layer_norm_x(x_encoded)

        if squeeze:
            cond_embedding = cond_embedding.squeeze()  # , 0)
        elif cond_embedding.shape[0] != x_t.shape[0]:  # type: ignore[attr-defined]
            # Original JAX
            # cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))
            cond_embedding = cond_embedding.repeat(x_t.shape[0], 1)

        if self.conditioning == "concatenation":
            # print(t_encoded.shape, x_encoded.shape, cond_embedding.shape)
            out = torch.concatenate((t_encoded, x_encoded, cond_embedding), dim=-1)
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")

        out = self.decoder(out)
        v_t = self.output_layer(out)

        return v_t, cond_mean, cond_logvar

    def get_condition_embedding(
        self, condition: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embedding of the condition.
        """
        if self.encode_conditions:
            condition_mean, condition_logvar = self.condition_encoder(condition)
        else:
            condition = torch.concatenate(list(condition.values()), dim=-1)
            print(
                "Condition encoder is not defined. Returning concatenated input as the embedding."
            )
        return condition_mean, condition_logvar

    @property
    def output_dims(self):
        """Dimensions of the output layers."""
        return tuple(self.decoder_dims) + (self.output_dim,)

    @property
    def time_encoder(self):
        """The time encoder used."""
        return self._time_encoder

    @time_encoder.setter
    def time_encoder(self, encoder):
        """Set the time encoder."""
        self._time_encoder = encoder

    @property
    def x_encoder(self):
        """The x encoder used."""
        return self._x_encoder

    @x_encoder.setter
    def x_encoder(self, encoder):
        """Set the x encoder."""
        self._x_encoder = encoder

    @property
    def decoder(self):
        """The decoder used."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        """Set the decoder."""
        self._decoder = decoder
