"""
Adapted from
borzoi pytorch: https://github.com/johahi/borzoi-pytorch/blob/main/LICENSE

enformer pytorch: https://github.com/lucidrains/enformer-pytorch/blob/main/LICENSE
"""

import math

import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn import functional as F

from bolero.tl.generic.module import KVBottleNeckMixin
from bolero.tl.generic.module_lora_cond import ConditionalLoRALayer
from collections import OrderedDict

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
        x: torch.Tensor, shape (batch, dim, seq_len)
        """
        seq_len, target_len = x.shape[-1], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[..., -trim:trim]


class GEGLU(nn.Module):
    def forward(self, x):
        """GEGLU forward pass."""
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class SequentialwithArgs(nn.Sequential):
    """Sequential module that can pass additional arguments to the modules."""

    no_args_modules = (
        nn.MaxPool1d,
        nn.LayerNorm,
        nn.Dropout,
        nn.ReLU,
        nn.GELU,
        nn.Upsample,
        nn.Softplus,
        nn.Conv1d,
        nn.Conv2d,
        nn.Linear,
        GEGLU,
    )

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
        ff_dropout=0.2,
        num_rel_pos_features=32,
        seq_len=4096,
    ):
        super().__init__()
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
                    nn.Dropout(ff_dropout),
                )
            ),
            Residual(
                FeedForward(
                    input_dim=channels,
                    hidden_dim=channels * 2,
                    output_dim=channels,
                    dropout=ff_dropout,
                ),
            ),
        )

    def forward(self, x, *args, **kwargs):
        """TransformerLayer forward pass."""
        return self.layers(x, *args, **kwargs)


class OutputHead(SequentialwithArgs):
    """A simple output head with out context input."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            # nn.Softplus(),
        )


class ContextCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        context_dim,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(in_channels)
        self.key_values_norm = nn.LayerNorm(context_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_queries = nn.Linear(in_channels, inner_dim, bias=False)

        self.null_key = nn.Parameter(torch.randn(inner_dim))
        self.null_value = nn.Parameter(torch.randn(inner_dim))

        self.to_key_values = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, in_channels)

    def forward(self, x, embedding):
        """
        b - batch
        e - borzoi embedding dim, 1920
        c - context dim, 41
        d - inner dimension
        i - sequence length (query embeddings), 16352
        j - sequence length (keys / values contexts)
        h - attention heads

        x: torch.Tensor, shape (batch, seq_len, dim)
        """
        h = self.heads

        # perform cross attention from dna -> context
        if embedding.ndim == 2:
            embedding = rearrange(embedding, "b c -> b 1 c")

        # (b, 16352, 1920) -> (b, 16352, 512)
        q = self.to_queries(self.query_norm(x))
        # (b, 1, 41) -> (b, 1, 512), (b, 1, 512)
        k, v = self.to_key_values(self.key_values_norm(embedding)).chunk(2, dim=-1)
        null_k = repeat(self.null_key, "d -> b 1 d", b=embedding.shape[0])
        null_v = repeat(self.null_value, "d -> b 1 d", b=embedding.shape[0])
        # (b, 1, 512), (b, 1, 512) -> (b, 2, 512)
        k = torch.cat((null_k, k), dim=1)
        v = torch.cat((null_v, v), dim=1)

        # split out head
        # (b, 16352, 512) -> (b, 8, 16352, 64)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        # (b, 2, 512) -> (b, 8, 2, 64)
        k = rearrange(k, "b j (h d) -> b h j d", h=h)
        v = rearrange(v, "b j (h d) -> b h j d", h=h)
        # (b, 8, 16352, 64), (b, 8, 2, 64) -> (b, 8, 16352, 2)
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # attention
        attn = sim.softmax(dim=-1)

        # aggregate
        # (b, 8, 16352, 2), (b, 8, 2, 64) -> (b, 8, 16352, 64)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        # (b, 8, 16352, 64) -> (b, 16352, 512)
        out = rearrange(out, "b h i d -> b i (h d)", h=h)

        # combine heads
        # (b, 16352, 512) -> (b, 16352, 1920)
        out = self.to_out(out)
        return out


class GEGLUFeedForward(nn.Module):
    def __init__(self, dim, dropout=0.05, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            nn.Dropout(dropout),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, *args, **kwargs):
        """GEGLUFeedForward forward pass."""
        # placeholder args and kwargs, but not used
        return self.net(x)


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


class MultiCellTypeOutputHead(nn.Module):
    """Output head with separate Conv + Softplus layers for each cell type."""

    def __init__(self, in_channels, cell_types):
        """
        Initialize the MultiCellTypeOutputHead.

        Args:
            in_channels (int): Number of input channels.
            cell_types (dict): A dictionary where keys are cell type names (str)
                               and values are the number of output channels (int)
                               for that cell type.
                               Example:
                               {
                                   'cell_type_1': 5313,
                                   'cell_type_2': 1643,
                                   ...
                               }
        """
        super(MultiCellTypeOutputHead, self).__init__()

        if not cell_types:
            raise ValueError("At least one cell type must be specified.")

        # Store separate Conv + Softplus layers for each cell type
        self.output_heads = nn.ModuleDict()
        for cell_type, out_channels in cell_types.items():
            self.output_heads[cell_type] = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.Softplus()
            )

    def forward(self, x, cell_type_id):
        """
        Forward pass to generate output for the specified cell type.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, seq_len).
            cell_type_id (str): The cell type identifier (key) for which to generate the output.

        Returns:
            dict: A dictionary with a single key-value pair, where the key is the cell type
                  and the value is the corresponding output tensor.
        """
        if cell_type_id not in self.output_heads:
            raise ValueError(f"Invalid cell type ID: {cell_type_id}")

        # Pass through the specific output head for the given cell type
        output = self.output_heads[cell_type_id](x)
        return {cell_type_id: output}
