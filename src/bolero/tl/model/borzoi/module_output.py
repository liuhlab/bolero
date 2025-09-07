import torch
import torch.nn as nn
from einops import einsum, rearrange

from .module import SequentialwithArgs


class OutputHead(SequentialwithArgs):
    """A simple output head with out context input."""

    def __init__(
        self, in_channels, out_channels, activation="softplus", avg_pool=False
    ):
        conv_layer = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        modules = [conv_layer]

        if avg_pool:
            # average pool across the sequence length dimension
            # (bs, out, l) -> (bs, out, 1)
            pooling = nn.AdaptiveAvgPool1d(1)
            modules.append(pooling)

        if activation == "softplus":
            activation = nn.Softplus()
        elif activation == "sigmoid":
            activation = nn.Sigmoid()
        elif activation == "gelu":
            activation = nn.GELU()
        elif activation is None:
            activation = nn.Identity()
        else:
            raise ValueError(f"Activation function {activation} not supported")
        modules.append(activation)

        super().__init__(*modules)


class DualOutputHead(SequentialwithArgs):
    # TG TODO: Needed if we have 2 channels? Needless computation for each head. Maybe activate within loss.
    """A dual output head that produces both activated and raw logit outputs."""

    def __init__(self, in_channels, mc_channels, atac_channels):
        super().__init__()
        # Create two separate output heads
        self.atac_head = OutputHead(in_channels, atac_channels, activation="softplus")
        self.mc_head = OutputHead(in_channels, mc_channels, activation=None)

    def forward(self, x, *args, **kwargs) -> dict:
        """Forward pass of the Dual and return outputs as a dictionary."""
        return {
            "atac": self.atac_head(x, *args, **kwargs),
            "mc": self.mc_head(x, *args, **kwargs),
        }


class ScoobyOutputHead(nn.Module):
    """
    This class is function as an Conv1d head where the weights are conditioned on the cell embeddings.

    Comparing to ConditionalConvLoRA, this class don't have the base convolutional layer.
    """

    def __init__(
        self,
        embedding_dim,
        input_dim=1920,
        hidden_dim=4096,
        output_dim=1,
        final_activation="softplus",
    ) -> None:
        """
        Setup the head for the Scooby model.

        Adapted from:
        https://github.com/gagneurlab/scooby/blob/main/scooby/modeling/scooby.py
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(
                1024, (self.input_dim + 1) * self.output_dim
            ),  # bias gets one more, and we predict pos. and neg. strand
        )

        self.pre_embedding_conv = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.input_dim, 1),
            nn.GELU(),
        )

        # follow the same way to init the weights as in the Scooby model
        nn.init.zeros_(self.embedding_mlp[-1].bias)
        nn.init.zeros_(self.pre_embedding_conv[-2].weight)
        nn.init.zeros_(self.pre_embedding_conv[-2].bias)

        if final_activation == "softplus":
            self.final_activation = nn.Softplus()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {final_activation} not supported")

    def forward(self, x, embedding):
        """
        Parameters
        ----------
        x: torch.tensor, shape=(bs, input_dim, seq_len)
        embedding: torch.tensor, shape=(bs, emb_dim)

        Returns
        -------
        torch.tensor, shape=(bs, output_dim, seq_len)
        """
        # (bs, i, l) -> (bs, i, l)
        x = self.pre_embedding_conv(x)

        # (bs, emb) -> (bs, i + 1, o)
        weight_and_bias = self.embedding_mlp(embedding)
        weight_and_bias = rearrange(
            weight_and_bias,
            "b (i_plus_1 o) -> b i_plus_1 o",
            i_plus_1=self.input_dim + 1,
        )

        # (bs, i + 1, o) -> (bs, i, o), (bs, 1, o)
        weight, bias = torch.split(weight_and_bias, [self.input_dim, 1], dim=1)

        # item wise conv1d with kernel size 1, (bs, i, l) -> (bs, o, l)
        bias = rearrange(bias, "b 1 o -> b o 1")
        x = einsum(x, weight, "b i l, b i o -> b o l") + bias

        # activation
        x = self.final_activation(x)
        return x


class GeneCountSoftClip(nn.Module):
    """
    AlphaGenome style squashing and soft clipping on gene counts.
    """

    def __init__(
        self,
        scale=2.0,
        gamma=10.0,
        threshold=10.0,
        apply_squashing=0.75,
        log_input=True,
        enable=True,
    ):
        """
        soft clip equation:
        y = scale * sqrt(x * gamma) - threshold, if x > threshold
        y = x, else

        Parameters
        ----------
        scale: float
            Scaling factor for soft clipping.
        gamma: float
            Gamma factor for soft clipping.
        threshold: float
            Threshold for soft clipping.
        apply_squashing: float
            Exponent for squashing before soft clip.
        log_input: bool
            Whether to apply log1p to input before squashing and soft clip.
            True if the input value is after log1p transform.
        """
        super().__init__()
        self.scale = scale
        self.gamma = gamma
        self.threshold = threshold
        self.apply_squashing = apply_squashing
        self.log_input = log_input
        self.enable = enable

    def inverse(self, clipped):
        """
        Undo soft clip, undo squashing, and log1p transform
        """
        if not self.enable:
            return clipped

        threshold, scale, gamma = self.threshold, self.scale, self.gamma
        x = torch.where(
            clipped > threshold,
            ((clipped + threshold) ** 2) / (gamma * scale**2),
            clipped,
        )
        x = x ** (1 / self.apply_squashing)
        if self.log_input:
            x = torch.log1p(x)
        return x

    def forward(self, x):
        """
        Turn x into count scale, apply squashing, and soft clip.
        """
        if not self.enable:
            return x

        threshold, scale, gamma = self.threshold, self.scale, self.gamma
        if self.log_input:
            x = torch.expm1(x)
        x = x**self.apply_squashing
        return torch.where(x > threshold, scale * torch.sqrt(x * gamma) - threshold, x)
