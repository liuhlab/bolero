import torch
import torch.nn as nn
from torchscale.component.dilated_attention import DilatedAttention

from .module import ConvBlock, FeedForward, OutputHead, Residual, SequentialwithArgs


class CountDataProcessor(nn.Module):
    """
    ATACProcessor is a neural network module designed to process ATAC-seq count data.

    Methods
    -------
    __init__():
        Initializes the ATACProcessor module with a Conv1D layer, LayerNorm, and Dropout.

    forward(atac_counts: torch.Tensor) -> torch.Tensor:
        Forward pass of the ATACProcessor module.

    Parameters
    ----------
        atac_counts (torch.Tensor): Input tensor containing ATAC-seq counts.

    Returns
    -------
        torch.Tensor: Processed tensor after applying log1p transformation, Conv1D, LayerNorm, and Dropout.
    """

    def __init__(
        self,
        in_channels=1,
        dna_emb_dims=1536,
        seq_len=16384,
        kernel_size=4,
        atac_dropout=0.1,
    ):
        super().__init__()

        self.layers = SequentialwithArgs(
            nn.Conv1d(
                in_channels, dna_emb_dims, kernel_size=kernel_size, padding="same"
            ),
            nn.LayerNorm(seq_len),
            nn.GELU(approximate="tanh"),
            nn.Dropout(atac_dropout),
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
            )
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """TransformerLayer forward pass."""
        x0 = x
        x = self.norm(x)
        # TODO: currently DilatedAttention not support args and kwargs
        x, _ = self.attn(x, x, x)
        x = x0 + x
        x = self.ff(x)
        return x


class DNAATACtoRNA(nn.Module):
    def __init__(
        self,
        borzoi_model,
        output_channels=1,
        seq_len=16384,
        num_heads=16,
        num_layers=0,
        atac_dropout=0.1,
        final_conv_dropout=0.05,
        dilated_ratio=(1, 2, 4, 8, 16, 32),
        segment_length=(512, 1024, 2048, 4096, 8192, 16384),
        xpos_rel_pos=False,
    ):
        """DNAATACtoRNA model for predicting RNA raw count track signal from DNA and ATAC data"""
        embed_dim = 1536
        final_dim = 1920
        epi_channels = 1

        super().__init__()
        self.borzoi_model = borzoi_model

        self.atac_processor = CountDataProcessor(
            in_channels=epi_channels,
            dna_emb_dims=embed_dim,
            seq_len=seq_len,
            kernel_size=5,
            atac_dropout=atac_dropout,
        )

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
            self.attn_layers = nn.Identity()

        self.final_joined_convs = SequentialwithArgs(
            ConvBlock(in_channels=embed_dim, out_channels=final_dim, kernel_size=1),
            nn.Dropout(final_conv_dropout),
            nn.GELU(approximate="tanh"),
        )
        self.output_head = OutputHead(
            in_channels=final_dim, out_channels=output_channels
        )

    def forward(self, dna_embedding, atac_counts, detach_input=False):
        """Forward pass of the DNAATACtoRNA model"""
        if detach_input:
            dna_embedding = dna_embedding.detach()
            atac_counts = atac_counts.detach()

        # Process ATAC counts
        dna_embedding = dna_embedding + self.atac_processor(atac_counts)

        dna_embedding = dna_embedding.permute(0, 2, 1).bfloat16()
        # Apply Cross and Self Attention Blocks
        attn_output = self.attn_layers(dna_embedding)
        attn_output = attn_output.permute(0, 2, 1)

        final_embedding = self.final_joined_convs(attn_output)

        # Apply Output Head to predict RNA raw count track signal
        final_output = self.output_head(final_embedding)
        return final_output
