import torch
import torch.nn as nn
from torchscale.component.dilated_attention import DilatedAttention

from bolero.tl.generic.module_embedding import KVBottleNeckMixin

from .module import (
    ContextCrossAttention,
    FeedForward,
    GEGLUFeedForward,
    Residual,
    SequentialwithArgs,
)


class CountDataProcessor(nn.Module):
    """
    ATACProcessor is a neural network module designed to process ATAC-seq count data.

    # TODO: Change to dilated convolutions and gradually increase channels
    """

    def __init__(
        self,
        in_channels=1,
        dna_emb_dims=1536,
        seq_len=16384,
        kernel_size=4,
        api_dropout=0.1,
    ):
        super().__init__()

        self.layers = SequentialwithArgs(
            nn.Conv1d(
                in_channels, dna_emb_dims, kernel_size=kernel_size, padding="same"
            ),
            nn.LayerNorm(seq_len),
            nn.GELU(approximate="tanh"),
            nn.Dropout(api_dropout),
        )

    def forward(self, atac_counts, *args, **kwargs):
        """
        Process ATAC-seq counts
        """
        atac_log = torch.log1p(atac_counts)

        atac_processed = self.layers(atac_log, *args, **kwargs)
        return atac_processed


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


class RNAOutputHead(nn.Module):
    def __init__(
        self,
        # Borzoi model
        output_channels=1,
        seq_len=16384,
        embed_dim=1920,
        # Epi input Processor
        epi_input=False,
        epi_dropout=0.1,
        epi_channels=1,
        # transformer
        num_layers=0,
        num_heads=16,
        final_conv_dropout=0.05,
        dilated_ratio=(1, 2, 4, 8, 16, 32),
        segment_length=(512, 1024, 2048, 4096, 8192, 16384),
        xpos_rel_pos=False,
    ):
        """DNAATACtoRNA model for predicting RNA raw count track signal from DNA and ATAC data"""
        super().__init__()

        if epi_input:
            self.epi_processor = CountDataProcessor(
                in_channels=epi_channels,
                dna_emb_dims=embed_dim,
                seq_len=seq_len,
                kernel_size=5,
                api_dropout=epi_dropout,
            )
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


class ContextOutputHead(nn.Module, KVBottleNeckMixin):
    """A simple output head with context input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        context_dim,
        cross_attn_heads=8,
        cross_attn_dim=64,
        ff_mult=2,
        dropout=0.05,
        kv_bottleneck=False,
        num_memories=256,
        dim_memory=20,
        num_memory_codebooks=2,
        additional_embs=1,
    ):
        super().__init__()
        cross_attn = Residual(
            # layer norm included in the cross attention
            ContextCrossAttention(
                in_channels=in_channels,
                context_dim=context_dim,
                heads=cross_attn_heads,
                dim_head=cross_attn_dim,
            )
        )
        feed_forward = Residual(
            # layer norm included in the feed forward
            GEGLUFeedForward(dim=in_channels, dropout=dropout, mult=ff_mult),
        )
        self.residual_context = Residual(
            SequentialwithArgs(
                cross_attn,
                feed_forward,
                nn.Dropout(dropout),
                # nn.GELU(approximate="tanh"),  # TODO: should we add GELU here?
            )
        )
        self.final_output_head = OutputHead(in_channels, out_channels)

        # key-value bottleneck for converting indices to embeddings
        if kv_bottleneck:
            self.kv_bottleneck, _ = self.setup_kv_bottleneck(
                num_memory_codebooks=num_memory_codebooks,
                num_memories=num_memories,
                dim_memory=dim_memory,
                additional_embs=additional_embs,
            )
        else:
            self.kv_bottleneck = None

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        ContextOutputHead forward pass.

        Parameters
        ----------
            x: torch.Tensor, shape (batch, dim, seq_len)
            embedding: torch.Tensor, shape (batch, context_dim)

        Returns
        -------
            torch.Tensor, shape (batch, out_channels, seq_len)
        """
        if self.kv_bottleneck is not None:
            embedding = self.vq_ind_to_emb(embedding)

        x = x.permute(0, 2, 1)
        x = self.residual_context(x, embedding=embedding)
        x = x.permute(0, 2, 1)
        x = self.final_output_head(x)
        return x
