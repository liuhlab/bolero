import math
from copy import deepcopy

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

from bolero.utils import validate_config


def compute_selfattention(
    transformer_encoder: nn.TransformerEncoder,
    x: torch.Tensor,
    i_layer: int,
    d_model: int,
    num_heads: int,
):
    """
    Compute self-attention probabilities for a given layer of a transformer encoder.
    """
    h = F.linear(
        x,
        transformer_encoder.layers[i_layer].self_attn.in_proj_weight,
        bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias,
    )
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model // num_heads)
    qkv = qkv.permute(0, 2, 1, 3)
    q, k, _ = qkv.chunk(3, dim=-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    return attn_probs


def extract_selfattention_maps(
    transformer_encoder: nn.TransformerEncoder, x: torch.Tensor
):
    """Extract self-attention maps from a transformer encoder."""
    d_model = transformer_encoder.layers[0].self_attn.embed_dim
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    i = 0
    h = x.clone()
    if norm_first:
        h = transformer_encoder.layers[i].norm1(h)
    attn_probs = compute_selfattention(transformer_encoder, h, i, d_model, num_heads)
    x = transformer_encoder.layers[i](x)
    return attn_probs


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        """
        Forward pass AttentionPool to mix pool_size information in dim -1.

        Input shape (b, d, (n * p))
        Output shape (b, d, n)
        """
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model=2048,
        nhead=32,
        dim_ffn=2048 * 4,
        num_layer=20,
        drop=0,
        LNM=1e-05,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ffn,
            batch_first=True,
            dropout=drop,
            layer_norm_eps=LNM,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layer)

    def forward(self, x):
        """Forward pass of the encoder."""
        output = self.encoder(self.norm(x))
        return output


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1 * 150 * 2048, 32)
        self.l2 = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x1):
        """
        Final ANN module

        input shape: (150, 2048)
        output shape: (1,)
        """
        x1 = x1.view(-1, 1 * 150 * 2048)
        x1 = self.act(self.l1(x1))
        x2 = self.l2(x1)
        return x2


class CREFormer(nn.Module):
    default_config = {"d_model": 2048}

    @classmethod
    def get_default_config(cls):
        """Get default config."""
        return deepcopy(cls.default_config)

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        default_config = cls.get_default_config()
        config = {k: v for k, v in config.items() if k in default_config}
        validate_config(config, default_config)
        return cls(**config)

    def __init__(self, d_model=2048, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== Elementary stage ==========
        # Input Embeddings
        self.dna_embed = nn.Embedding(4100, d_model)
        self.atac_embed = nn.Embedding(38, d_model)
        # Positional Embeddings
        self.pos1_embed = nn.Embedding(130, d_model)
        self.pos2_embed = nn.Embedding(129, d_model)
        # Encoder
        # base pair and CLS encoder within 128bp-bin
        self.encoder_1 = Encoder()
        # bin CLS encoder across 8 bins within 1024bp peak region
        self.encoder_2 = Encoder()

        # ========== Regulatory stage ==========
        # Reduce 8 bins into 1 for peak embedding
        self.atten_pool = AttentionPool(2048)
        # peak embedding encoder across 150 peaks
        self.encoder_3 = Encoder()
        # Positional Embeddings
        self.tss_embed = nn.Embedding(320, d_model)
        self.pad_embed = nn.Embedding(4100, d_model)
        # Final count prediction
        self.ann = ANN()

        self.to(self.device)
        return

    def _prepare_pos_input(self, tss_loc: int, direction: str, n_peaks: int):
        # pos1 shape (129,) which is bp position within 128bp bin
        pos1 = torch.arange(1, 130, dtype=int, device=self.device)
        # pos2 shape (8,) which is bin position within 8 128bp-bin
        pos2 = torch.arange(1, 9, dtype=int, device=self.device)

        # pos3 shape (150,) which is peak position
        # relative to TSS peak within 150 peaks of a gene batch
        # 0: TSS or padding peaks,
        # odds positions: upstream peaks, the larger the further,
        # even numbers: downstream peaks, the larger the further,
        pos3 = torch.zeros(150, dtype=int, device=self.device)
        pos3[tss_loc] = 0
        if direction == 1:  # "+" strand
            pos3[tss_loc - 1] = 1  # upstream of TSS is odds number
            pos3[tss_loc + 1] = 2  # downstream of TSS is even number
            for tss_i in range(tss_loc - 1):
                pos3[tss_loc - 1 - tss_i - 1] = pos3[tss_loc - 1 - tss_i] + 2
            for tss_i in range(n_peaks - tss_loc - 2):
                pos3[tss_loc + 1 + tss_i + 1] = pos3[tss_loc + 1 + tss_i] + 2
        elif direction == 0:  # "-" strand
            pos3[tss_loc - 1] = 2  # upstream of TSS is even number
            pos3[tss_loc + 1] = 1  # downstream of TSS is odds number
            for tss_i in range(tss_loc - 1):
                pos3[tss_loc - 1 - tss_i - 1] = pos3[tss_loc - 1 - tss_i] + 2
            for tss_i in range(n_peaks - tss_loc - 2):
                pos3[tss_loc + 1 + tss_i + 1] = pos3[tss_loc + 1 + tss_i] + 2
        else:
            raise ValueError("direction should be 0 or 1")
        return pos1, pos2, pos3

    def attention_pool(self, x_enc_2: torch.Tensor) -> torch.Tensor:
        """
        Attention Pooling

        Input shape (n_peaks, d, 8)
        Output shape (n_peaks, d)
        """
        # x_enc_2.shape [44, 2048, 8] # put n_bins in the end to perform attn pool
        x_enc_2 = rearrange(x_enc_2, "b n d -> b d n")
        # x_enc_2.shape [44, 2048, 1] # bins are merged after attn pool
        x_enc_2 = self.atten_pool(x_enc_2).squeeze(-1)
        return x_enc_2

    def forward_elementary_and_attn_pool(
        self,
        n_peaks: int,
        dna_in: torch.Tensor,
        atac_in: torch.Tensor,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """
        CREFormer elementary stage

        Parameters
        ----------
        n_peaks : int
            Number of peaks
        dna_in : torch.Tensor
            DNA input, shape (n_peaks, 1024)
        atac_in : torch.Tensor
            ATAC input, shape (n_peaks, 1024)
        pos1 : torch.Tensor
            Positional input 1, shape (129,)
        pos2 : torch.Tensor
            Positional input 2, shape (8,)

        Returns
        -------
        torch.Tensor
            Elementary stage output, shape (n_peaks, 2048)
        """
        # for 1024 bp peak region, cut to 8*128 bins
        dna_in = dna_in.int().reshape(n_peaks * 8, 128)
        atac_in = atac_in.int().reshape(n_peaks * 8, 128)
        # x_mul.shape [n_peak * 8, 128, 2048]
        x_mul = self.dna_embed(dna_in) + self.atac_embed(atac_in)
        # x_embed.shape [n_peak * 8, 129, 2048]
        # CLS.shape [n_peak * 8, 1, 2048]
        CLS = self.dna_embed(torch.ones(n_peaks * 8, 1, dtype=int, device=self.device))
        x_embed = torch.cat((CLS, x_mul), dim=1)

        # attention within 128bp bins
        # x_enc_1.shape [44, 8, 2048]
        # take the first token for each region, and reshape to get 8 bp
        # x_POS_1.shape [129, 2048] (1 cls + 128 bp)
        x_POS_1 = self.pos1_embed(pos1)
        x_enc_1 = self.encoder_1(x_embed + x_POS_1)
        x_enc_1 = x_enc_1[:, 0, :].reshape(n_peaks, 8, 2048)

        # attention across the 8 128bp bins
        # x_POS_2.shape [8, 2048]
        x_POS_2 = self.pos2_embed(pos2)
        # x_enc_2.shape [n_peaks, 8, 2048]
        x_enc_2 = self.encoder_2(x_enc_1 + x_POS_2)

        # attention pooling across 8 bins and merge them into one embedding vector
        # x_enc_2.shape [n_peaks, 2048]
        x_enc_2 = self.attention_pool(x_enc_2)
        return x_enc_2

    def forward_regulatory(self, x_enc_2: torch.Tensor, pos3: torch.Tensor):
        """
        CREFormer regulatory stage

        Parameters
        ----------
        x_enc_2 : torch.Tensor
            Elementary stage output, shape (n_peaks, 2048)
        pos3 : torch.Tensor
            Positional input 3, shape (150,)

        Returns
        -------
        torch.Tensor
            Regulatory stage output, shape (1,)
        """
        # x_pad.shape (150 - n_peaks, 2048) # pad n_peaks upto 150
        x_pad = self.pad_embed(
            torch.zeros(150 - x_enc_2.shape[0], dtype=int, device=self.device)
        )
        # x_eb3.shape [150, 2048] # fix size input for encoder 3
        x_eb3 = torch.cat((x_enc_2, x_pad), dim=0)
        # x_POS_3.shape [150, 2048]
        x_POS_3 = self.tss_embed(pos3)
        # x_enc_3.shape [150, 2048]
        x_enc_3 = self.encoder_3(x_eb3 + x_POS_3)
        # result.shape [1]
        result = self.ann(x_enc_3).squeeze(1)
        return result

    def extract_regulatory_attention(self, x_enc_2: torch.Tensor, pos3: torch.Tensor):
        """
        Extract regulatory attention scores from the first layer of the encoder_3.

        Parameters
        ----------
        x_enc_2 : torch.Tensor
            Elementary stage output, shape (n_peaks, 2048)
        pos3 : torch.Tensor
            Positional input 3, shape (150,)

        Returns
        -------
        torch.Tensor
            Regulatory attention scores, shape (n_peaks,)
        """
        n_peaks = x_enc_2.shape[0]
        x_pad = self.pad_embed(
            torch.zeros(150 - n_peaks, dtype=int, device=self.device)
        )
        x_eb3 = torch.cat((x_enc_2, x_pad), dim=0)
        x_POS_3 = self.tss_embed(pos3)
        x_enc_3 = x_eb3 + x_POS_3

        attn_probs = extract_selfattention_maps(
            self.encoder_3.encoder, x_enc_3.unsqueeze(0)
        )
        SM = nn.Softmax(dim=2)
        attention_score = SM(attn_probs[0]).mean(0).sum(0)[:n_peaks]
        return attention_score

    def forward(
        self,
        dna_in: torch.Tensor,
        atac_in: torch.Tensor,
        tss_loc: int,
        direction: int,
        gene_count: bool = True,
        attention_score=False,
    ):
        """
        Forward pass of CREFormer model.

        Parameters
        ----------
        dna_in : torch.Tensor
            DNA input, shape (n_peaks, 1024)
        atac_in : torch.Tensor
            ATAC input, shape (n_peaks, 1024)
        tss_loc : int
            TSS location in n_peaks dim
        direction : int
            Direction of TSS (1 for +, 0 for -)
        """
        n_peaks = dna_in.shape[0]
        pos1, pos2, pos3 = self._prepare_pos_input(tss_loc, direction, n_peaks)

        x = self.forward_elementary_and_attn_pool(n_peaks, dna_in, atac_in, pos1, pos2)

        result = {}
        if gene_count:
            gene_count = self.forward_regulatory(x, pos3)
            result["gene_count"] = gene_count
        if attention_score:
            with torch.no_grad():
                x = x.detach()
                pos3 = pos3.detach()
                attention_score = self.extract_regulatory_attention(x, pos3)
            result["attention_score"] = attention_score
        return result

    def summary(self):
        """Print model summary."""
        t = summary(
            self,
            input_data={
                # simulate 10 peaks for 1 gene
                "dna_in": torch.randint(0, 4100, (10, 1024)).cuda(),
                # simulate 10 peaks for 1 gene
                "atac_in": torch.randint(0, 38, (10, 1024)).cuda(),
                # TSS at peak 5
                "tss_loc": 5,
                # "+" strand
                "direction": 1,
            },
            col_names=["input_size", "output_size", "num_params"],
            row_settings=("var_names",),
        )
        return t

    def __repr__(self):
        return str(self.summary())

    def loss(self, y_pred, y_true):
        """Gene count MSE loss at log1p scale."""
        y_true = torch.log1p(y_true)
        y_pred = torch.log1p(y_pred)
        loss = F.mse_loss(input=y_pred, target=y_true)
        return loss.mean()
