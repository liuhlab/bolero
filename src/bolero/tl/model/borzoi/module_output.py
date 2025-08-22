import torch
import torch.nn as nn
from einops import einsum, rearrange

try:
    from torchscale.component.dilated_attention import DilatedAttention
except ImportError:
    pass

from .module import (
    FeedForward,
    Residual,
    SequentialwithArgs,
)


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.GroupNorm(1, out_channel),
            nn.GELU(),
        )
        self.res = nn.Sequential(
            nn.Linear(out_channel, out_channel),
            nn.GroupNorm(1, out_channel),
            nn.GELU(),
        )
        self.activate = nn.GELU()

    def forward(self, x):
        """
        It consists of three components
        - Scaling
        - Residual
        - ReLU
        """
        scaled = x
        for module in self.scale:
            scaled = module(scaled)
        identity = scaled
        res_out = scaled
        for module in self.res:
            res_out = module(res_out)
        out = self.activate(res_out + identity)
        return out


class CountDataProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = [1, 8, 64, 512]
        out_channels = [8, 64, 512, 1920]
        conv_blocks = []
        for _in, _out in zip(in_channels, out_channels):
            conv_blocks.append(MLP(_in, _out))
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):
        """
        Forward pass of the CountDataProcessor
        """
        # log1p transform count
        x = torch.log1p(x)
        return self.conv_blocks(x)


class DilatedAttentionArgs:
    def __init__(self, **kwargs):
        default_dilated_ratio = [1, 2, 4, 8, 16, 32]
        default_segment_length = [512, 1024, 2048, 4096, 8192, 16384]

        self.multiway = kwargs.get("multiway", False)
        self.layernorm_eps = kwargs.get("layernorm_eps", 1e-5)
        self.xpos_rel_pos = kwargs.get("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.get("xpos_scale_base", 512)
        self.flash_attention = kwargs.get("flash_attention", True)
        self.dilated_ratio = kwargs.get("dilated_ratio", default_dilated_ratio)
        self.segment_length = kwargs.get("segment_length", default_segment_length)
        self.seq_parallel = kwargs.get("seq_parallel", False)


class DilatedTransformerLayer(nn.Module):
    """
    DilatedTransformerLayer is a neural network module that implements a dilated attention mechanism
    with optional self-attention or cross-attention, followed by a feed-forward network.

    Parameters
    ----------
    embed_dim : int, optional
        The dimensionality of the input embeddings. Default is 1536.
    heads : int, optional
        The number of attention heads. Default is 16.
    attn_dropout : float, optional
        Dropout rate for the attention mechanism. Default is 0.05.
    ff_dropout : float, optional
        Dropout rate for the feed-forward network. Default is 0.1.
    **kwargs : dict
        Additional arguments for DilatedAttentionArgs.

    Attributes
    ----------
    args : DilatedAttentionArgs
        Arguments for the dilated attention mechanism.
    layers : SequentialwithArgs
        A sequential container of residual layers, each containing a dilated attention mechanism
        followed by a feed-forward network.

    Methods
    -------
    forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor
        Performs the forward pass of the DilatedTransformerLayer.

    Parameters
    ----------
        x : torch.Tensor
            The input tensor.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

    Returns
    -------
        torch.Tensor
            The output tensor after applying attention and feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        heads: int = 16,
        attn_dropout: float = 0.05,
        ff_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.args = DilatedAttentionArgs(**kwargs)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-3)
        self.attn = DilatedAttention(
            args=self.args,
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=attn_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=False,
        )
        self.ff = Residual(
            FeedForward(
                input_dim=embed_dim,
                hidden_dim=embed_dim * 2,
                output_dim=embed_dim,
                dropout=ff_dropout,
                activation="gelu",
            )
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """TransformerLayer forward pass."""
        x = x.bfloat16()

        x0 = x

        x = self.norm(x)

        # TODO: currently DilatedAttention not support args and kwargs
        x, _ = self.attn(x, x, x)
        x = x0 + x

        x = self.ff(x)
        return x


class OutputHead(SequentialwithArgs):
    """A simple output head with out context input."""

    def __init__(self, in_channels, out_channels, activation="softplus"):
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

        super().__init__(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            activation,
        )


class GeneCountOutputHead(nn.Module):
    """Version 2: use dialated convolutions and only take center bin to predict gene count"""

    def __init__(self, embedding_dim=1920, n_blocks=8):
        super().__init__()

        self.conv_blocks = nn.ModuleList()

        in_channels = embedding_dim
        dil = 1
        for _ in range(n_blocks):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=dil,
                        dilation=dil,
                    ),
                    nn.GroupNorm(1, in_channels),
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=dil,
                        dilation=dil,
                    ),
                    nn.GroupNorm(1, in_channels),
                )
            )
            dil *= 2
        self.reception_field = int(dil * 2)

        self.final_fc = nn.Linear(in_channels, 1)
        self.activation = nn.Softplus()

    def forward(self, x):
        """Compute the forward pass of the Gene Count Output Head."""
        # x shape: (bs, 1920, 16352)
        seq_len = x.shape[-1]
        if len(self.conv_blocks) > 0:
            # take the middle bins (likely promoter related)
            promoter_slice = slice(
                seq_len // 2 - self.reception_field, seq_len // 2 + self.reception_field
            )
            x = x[:, :, promoter_slice]
            middle = self.reception_field
            for conv in self.conv_blocks:
                x = conv(x)
        else:
            # no conv blocks
            middle = seq_len // 2
        # x shape: (bs, 1920)
        x = self.final_fc(x[:, :, middle])
        # x shape: (bs, 1)
        x = self.activation(x)
        return x


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


class RNAOutputHead(nn.Module):
    def __init__(
        self,
        # Borzoi model
        output_channels=1,
        embed_dim=1920,
        # Epi input Processor
        epi_input=False,
        epi_dropout=0.0,
        epi_channels=1,
        # transformer
        num_layers=8,
        num_heads=32,
        final_conv_dropout=0.0,
        dilated_ratio=(1, 2, 4, 8, 16, 32),
        segment_length=(512, 1024, 2048, 4096, 8192, 16384),
        xpos_rel_pos=False,
    ):
        """DNAATACtoRNA model for predicting RNA raw count track signal from DNA and ATAC data"""
        super().__init__()

        if epi_input:
            self.epi_processor = CountDataProcessor()
        else:
            self.epi_processor = nn.Identity()

        if num_layers > 0:
            kwargs = {
                "dilated_ratio": dilated_ratio,
                "segment_length": segment_length,
                "xpos_rel_pos": xpos_rel_pos,
            }
            transformer = [
                DilatedTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
            self.attn_layers = SequentialwithArgs(*transformer)
        else:
            self.attn_layers = FeedForward(
                input_dim=embed_dim,
                hidden_dim=embed_dim * 2,
                output_dim=embed_dim,
                dropout=final_conv_dropout,
                activation="gelu",
            )

        self.output_head = OutputHead(
            in_channels=embed_dim, out_channels=output_channels
        )

    def forward(self, dna_embedding, epi_input=None, detach_input=False):
        """Forward pass of the DNAATACtoRNA model"""
        if detach_input:
            dna_embedding = dna_embedding.detach()

        if epi_input is None:
            x = dna_embedding
        else:
            x = dna_embedding + self.epi_processor(epi_input)

        x = x.permute(0, 2, 1)
        # Apply Cross and Self Attention Blocks
        x = self.attn_layers(x)
        x = x.permute(0, 2, 1)

        # Apply Output Head to predict RNA raw count track signal
        output = self.output_head(x)
        return output


class GeneCountAttnOutputHead(RNAOutputHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_fc = nn.Linear(kwargs.get("embed_dim", 1920), 1)
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor, epi_input=None):
        """Forward pass of the GeneCountAttnOutputHead"""
        if epi_input is not None:
            x = x + self.epi_processor(epi_input)

        x = x.permute(0, 2, 1)
        x = self.attn_layers(x)
        x = x.permute(0, 2, 1)

        # taka the middle bin for count prediction
        seq_len = x.shape[-1]
        middle = seq_len // 2
        x = self.final_fc(x[:, :, middle])
        x = self.activation(x)
        return x


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
