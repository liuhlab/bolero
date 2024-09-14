"""
Adapted from
borzoi pytorch: https://github.com/johahi/borzoi-pytorch/blob/main/LICENSE

enformer pytorch: https://github.com/lucidrains/enformer-pytorch/blob/main/LICENSE
"""

import math

import torch
from einops import rearrange
from torch import einsum, nn

from bolero.tl.generic.module_lora_cond import ConditionalLoRALayer


def relative_shift(x):
    """
    Perform relative shift on the input tensor.

    (bs, head, seq_len, seq_len * 2 - 1) -> (bs, head, seq_len, seq_len)
    """
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


def get_positional_features_central_mask(positions, features, seq_len):
    """
    Get positional features with central mask.

    return shape (seq_len * 2 - 1, features) = (8191, 16) by default
    """
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(
        pow_rate, torch.arange(1, features + 1, device=positions.device)
    ).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_embed(seq_len, feature_size, device):
    """
    Get positional embeddings.

    The positional embeddings contains two components:
    first is the positional_features_central_mask, second is the signed positional_features_central_mask.
    return shape (seq_len * 2 - 1, feature_size) = (8191, 32) by default
    """
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_central_mask,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings


def maybe_pass_additional_params(module, x, *args, **kwargs):
    """Maybe pass additional parameters to the module if it is a subclass of ConditionalLoRALayer."""
    if isinstance(module, ConditionalLoRALayer):
        return module(x, *args, **kwargs)
    else:
        return module(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """Residual forward pass."""
        return self.fn(x, *args, **kwargs) + x


class TargetLengthCrop(nn.Module):
    """
    This class simply crops the input into the target length symmetrically at seq_length dimension.
    """

    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        """
        x: torch.Tensor, shape (batch, seq_len, dim)
        """
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


class SequentialwithArgs(nn.Sequential):
    """Sequential module that can pass additional arguments to the modules."""

    no_args_modules = (nn.MaxPool1d, nn.LayerNorm, nn.Dropout, nn.ReLU, nn.GELU)

    def forward(self, x, *args, **kwargs):
        """SequentialwithArgs forward pass."""
        for module in self:
            if isinstance(module, self.no_args_modules):
                x = module(x)
            else:
                x = module(x, *args, **kwargs)
        return x


class Attention(nn.Module):
    """
    Original TF version:
    https://github.com/calico/baskerville/blob/16815faf23a0790c34c3631e30ee80506d567d77/src/baskerville/blocks.py#L1095
    """

    def __init__(
        self,
        dim=1536,
        num_rel_pos_features=32,
        heads=8,
        dim_key=64,
        attn_dropout=0.05,
        pos_dropout=0.01,
        seq_len=4096,
    ):
        # TODO: check L2 normalization
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        dim_value = dim // heads
        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.register_buffer(
            "positions",
            get_positional_embed(
                seq_len, self.num_rel_pos_features, self.to_v.weight.device
            ),
            persistent=False,
        )
        # positions.shape = (2 * seq_len - 1, num_rel_pos_features) = (8191, 32)

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, *args, **kwargs):
        """Attention forward pass."""
        h = self.heads

        q = maybe_pass_additional_params(self.to_q, x, *args, **kwargs)
        k = maybe_pass_additional_params(self.to_k, x, *args, **kwargs)
        v = maybe_pass_additional_params(self.to_v, x, *args, **kwargs)

        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))
        # q, k, v = (b, 8, 4096, 64)

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )
        # (b, 8, 4096, 64) * (b, 8, 4096, 64) -> (b, 8, 4096, 4096)

        # relative positional encoding
        positions = self.pos_dropout(self.positions)
        # (8191, 32)
        rel_k = maybe_pass_additional_params(self.to_rel_k, positions)
        # (8191, 32) -> (8191, 512)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        # (8191, 512) -> (8, 8191, 64)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        # (b, 8, 4096, 64) * (8, 8191, 64) -> (b, 8, 4096, 8191)
        rel_logits = relative_shift(rel_logits)
        # (b, 8, 4096, 8191) -> (b, 8, 4096, 4096)
        logits = content_logits + rel_logits
        # (b, 8, 4096, 4096)

        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        # (b, 8, 4096, 4096) * (b, 8, 4096, 192) -> (b, 8, 4096, 192)
        out = rearrange(out, "b h n d -> b n (h d)")
        # (b, 8, 4096, 192) -> (b, 4096, 1536)
        out = maybe_pass_additional_params(self.to_out, out, *args, **kwargs)
        # (b, 4096, 1536)
        return out


class ConvDna(nn.Module):
    def __init__(self, in_channels=4, out_channels=512, dna_kernel_size=15):
        super().__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=dna_kernel_size,
            padding="same",
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=0)

    def forward(self, x, *args, **kwargs):
        """ConvDna forward pass."""
        x = maybe_pass_additional_params(self.conv_layer, x, *args, **kwargs)
        return self.max_pool(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        conv_type="standard",
        batchnorm_momentum=0.9,
    ):
        super().__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                padding="same",
                bias=False,
            )
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(
                in_channels, eps=0.001, momentum=batchnorm_momentum
            )
            self.activation = nn.GELU(approximate="tanh")
            self.conv_layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            )

    def forward(self, x, *args, **kwargs):
        """ConvBlock forward pass."""
        x = self.norm(x)
        x = self.activation(x)
        if isinstance(self.conv_layer, nn.Sequential):
            for layer in self.conv_layer:
                x = maybe_pass_additional_params(layer, x, *args, **kwargs)
        else:
            x = maybe_pass_additional_params(self.conv_layer, x, *args, **kwargs)
        return x


class FeedForward(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__(
            nn.LayerNorm(input_dim, eps=0.001),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, *args, **kwargs):
        """FeedForward forward pass."""
        for module in self:
            x = maybe_pass_additional_params(module, x, *args, **kwargs)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        channels=1536,
        heads=8,
        dim_key=64,
        attn_dropout=0.05,
        pos_dropout=0.01,
        dropout=0.2,
        num_rel_pos_features=32,
        seq_len=4096,
    ):
        self.layers = SequentialwithArgs(
            Residual(
                SequentialwithArgs(
                    nn.LayerNorm(channels, eps=0.001),
                    Attention(
                        dim=channels,
                        heads=heads,
                        dim_key=dim_key,
                        attn_dropout=attn_dropout,
                        pos_dropout=pos_dropout,
                        num_rel_pos_features=num_rel_pos_features,
                        seq_len=seq_len,
                    ),
                    nn.Dropout(dropout),
                )
            ),
            Residual(
                FeedForward(
                    input_dim=channels,
                    hidden_dim=channels * 2,
                    output_dim=channels,
                    dropout=dropout,
                ),
            ),
        )

    def forward(self, x, *args, **kwargs):
        """TransformerLayer forward pass."""
        return self.layers(x, *args, **kwargs)
