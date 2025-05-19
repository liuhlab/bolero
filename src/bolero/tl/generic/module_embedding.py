from functools import partial
from typing import Union

import numpy as np
import torch
from einops import einsum, rearrange, reduce, repeat
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import VectorQuantize

from bolero.tl.generic.module import GroupedLinear, Residual
from bolero.tl.model.flow._utils import cyclical_time_encoder

from .module_lora import set_submodule_by_name


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Linear(dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        init with zeros so the beginning the attention weights are uniform
        (1/N, 1/N, …, 1/N), which means the attention pooling is equivalent to mean pooling
        """
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    @staticmethod
    def _make_all_zero_mask(x: torch.Tensor) -> torch.Tensor:
        """
        mask (B, N) False where all D features are zero
        """
        # A token is valid if any of its features is non-zero
        return torch.any(x != 0, dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B, N, D), return (B, D)
        """
        if x.shape[1] == 1:
            # if the input is a single token, return it
            return x.squeeze(1)

        if mask is None:
            mask = self._make_all_zero_mask(x)

        # x: (B, N, D)
        logits = self.scorer(x).squeeze(-1)  # → (B, N)
        logits = logits.masked_fill(~mask, -float("inf"))

        weights = F.softmax(logits, dim=-1)  # → (B, N)
        weights = weights.unsqueeze(-1)  # → (B, N, 1)
        pooled = (weights * x).sum(dim=1)  # → (B, D)
        return pooled


class ArchLinear(nn.Linear):
    def __init__(self, linear_module: nn.Linear, arch_features: int, bias: bool = True):
        super().__init__(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=bias,
            device=linear_module.weight.device,
            dtype=linear_module.weight.dtype,
        )

        self.arch_features = arch_features
        self.arch_linear = nn.Linear(
            arch_features, linear_module.out_features, bias=bias
        )

        # zero the weights and bias of the arch_linear
        self.arch_linear.weight.data[...] = 0
        if self.arch_linear.bias is not None:
            self.arch_linear.bias.data[...] = 0

        # freeze the original linear layer and only train the arch_linear
        self.weight.data = linear_module.weight.data
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.data = linear_module.bias.data
            self.bias.requires_grad = False

    def forward(self, x, arch):
        """
        Forward pass of the ArchLinear module.

        Args:
            x (torch.Tensor): The input tensor, arch dim is in last arch_features dim.

        Returns
        -------
            torch.Tensor: The output tensor after passing through the ArchLinear layer.
        """
        linear_out = F.linear(x, self.weight, self.bias)
        arch_out = self.arch_linear(arch)
        return linear_out + arch_out


class ArchEmbeddingMixin:
    """
    Mixin class to add Arch Embedding to MLP-like modules.
    """

    def add_arch_embedding(self, input_features: int, bias=True):
        """
        Add an ArchEmbedding to the MLP.

        Parameters
        ----------
        input_features : int
            The number of input features.
        first_only : bool
            If True, only inject the first layer.
        """
        for name, module in self.mlp.named_modules():
            if isinstance(module, nn.Linear):
                arch_module = ArchLinear(
                    linear_module=module,
                    arch_features=input_features,
                    bias=bias,
                )
                set_submodule_by_name(self.mlp, name, arch_module)
        self.arch_features = input_features
        return


# CODE BELOW IS FROM
# https://github.com/lucidrains/discrete-key-value-bottleneck-pytorch/tree/main
# LICENSE: MIT https://github.com/lucidrains/discrete-key-value-bottleneck-pytorch/blob/main/LICENSE
class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        dim=50,
        *,
        dim_embed=None,
        num_memories=256,
        num_memory_codebooks=2,
        dim_memory=256,
        encoder=None,
        average_pool_memories=True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        if dim_embed is None:
            dim_embed = dim
        self.dim_embed = dim_embed

        self.vq = VectorQuantize(
            dim=dim * num_memory_codebooks,
            codebook_size=num_memories,
            heads=num_memory_codebooks,
            separate_codebook_per_head=True,
            **kwargs,
        )

        if dim_memory is None:
            dim_memory = dim
        # self.values.shape (h, n, d), leanable memory vectors for decoder input
        self.values = nn.Parameter(
            torch.randn(num_memory_codebooks, num_memories, dim_memory)
        )

        rand_proj = torch.empty(num_memory_codebooks, dim_embed, dim)
        nn.init.xavier_normal_(rand_proj)

        self.register_buffer("rand_proj", rand_proj)
        self.average_pool_memories = average_pool_memories

    def forward(self, x, return_intermediates=False, **kwargs):
        """Get the memory embeddings from the input embeddings."""
        if self.encoder is not None:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, **kwargs)
                x.detach_()

        # add n dim if not exist, but remember to convert back to 2D after forward
        if x.ndim == 2:
            x = rearrange(x, "b d -> b 1 d")
            has_n_dim = False

        assert (
            x.shape[-1] == self.dim_embed
        ), f"encoding has a dimension of {x.shape[-1]} but dim_embed (defaults to dim) is set to {self.dim_embed} on init"

        x = einsum(x, self.rand_proj, "b n d, c d e -> b n c e")
        x = rearrange(x, "b n c e -> b n (c e)")

        vq_out = self.vq(x)

        _, memory_indices, _ = vq_out

        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, "... -> ... 1")

        memory_indices = rearrange(memory_indices, "b n h -> b h n")

        values = repeat(self.values, "h n d -> b h n d", b=memory_indices.shape[0])
        memory_indices = repeat(memory_indices, "b h n -> b h n d", d=values.shape[-1])

        memories = values.gather(2, memory_indices)

        if self.average_pool_memories:
            memories = reduce(memories, "b h n d -> b n d", "mean")

        if not has_n_dim:
            memories = rearrange(memories, "b 1 d -> b d")

        if return_intermediates:
            return memories, vq_out

        return memories


class DiscreteKeyValueBottleneckNoVQ(nn.Module):
    def __init__(
        self,
        num_memories=256,
        dim_memory=64,
        num_memory_codebooks=2,
        average_pool_memories=False,
    ):
        """
        A simple implementation of a discrete key-value bottleneck without VQ part.

        Adapted from https://github.com/lucidrains/discrete-key-value-bottleneck-pytorch/tree/main

        Parameters
        ----------
        num_memories: int
            number of memories, which is the codebook size in VQ
        dim_memory: int
            dimension of memory vector in each codebook
        num_memory_codebooks: int
            number of codebooks, which is the number of heads in multi-head VQ
        """
        super().__init__()
        self.values = nn.Parameter(
            torch.clamp(
                torch.randn(num_memory_codebooks, num_memories, dim_memory),
                min=-3,
                max=3,
            ),
        )
        # self.values.shape (h, n, d)

        self.num_memory_codebooks = num_memory_codebooks
        self.dim_memory = dim_memory
        self.num_memories = num_memories

        self.average_pool_memories = average_pool_memories

    def forward(
        self,
        vq_indices,
    ):
        """Turn vq indices into memory embeddings."""
        vq_indices = vq_indices.long()
        input_shape = vq_indices.shape
        if vq_indices.ndim == 2:
            vq_indices = rearrange(vq_indices, "bs h -> bs 1 h")
        vq_indices = rearrange(vq_indices, "b n h -> b h n")

        values = repeat(self.values, "h n d -> b h n d", b=input_shape[0])
        vq_indices = repeat(vq_indices, "b h n -> b h n d", d=values.shape[-1])
        memories = values.gather(2, vq_indices)

        if self.average_pool_memories:
            memories = memories.mean(dim=1)
        else:
            memories = rearrange(memories, "b h n d -> b n (h d)")

        if len(input_shape) == 2:
            memories = memories.squeeze(1)
        return memories


class KVBottleNeckMixin:
    kv_bottleneck: Union[DiscreteKeyValueBottleneckNoVQ, DiscreteKeyValueBottleneck]

    def setup_kv_bottleneck(
        self,
        num_memory_codebooks,
        num_memories,
        dim_memory,
        additional_embs,
        emb_input=False,
        emb_input_dims=None,
        average_pool_memories=False,
    ):
        """
        Setup the key-value bottleneck for converting indices to embeddings.

        Parameters
        ----------
        num_memory_codebooks: int
            number of codebooks
        num_memories: int
            number of memories in each codebook
        dim_memory: int
            dimension of memory vector in each codebook
        additional_embs: int
            number of additional embeddings to be concatenated with the memory embeddings

        Returns
        -------
        kv_bottleneck: nn.Module
            key-value bottleneck module
        emb_input_features: int
            number of input features for the embeddings after concatenating the additional embeddings
        """
        # key-value bottleneck for converting indices to embeddings
        self.additional_embs = additional_embs
        self.num_memory_codebooks = num_memory_codebooks
        self.num_memories = num_memories
        self.dim_memory = dim_memory
        self.emb_input = emb_input

        if self.emb_input:
            assert (
                emb_input_dims is not None
            ), "emb_input_dims must be provided if emb_input is True"
            kv_bottleneck = DiscreteKeyValueBottleneck(
                dim=emb_input_dims,
                num_memories=num_memories,
                dim_memory=dim_memory,
                num_memory_codebooks=num_memory_codebooks,
                average_pool_memories=True,
            )
            self.input_dims_for_kv = emb_input_dims
        else:
            kv_bottleneck = DiscreteKeyValueBottleneckNoVQ(
                num_memories=num_memories,
                dim_memory=dim_memory,
                num_memory_codebooks=num_memory_codebooks,
                average_pool_memories=average_pool_memories,
            )
            self.input_dims_for_kv = num_memory_codebooks

        emb_input_features = num_memory_codebooks * dim_memory + additional_embs
        return kv_bottleneck, emb_input_features

    def vq_ind_to_emb(self, emb_data):
        """
        VQ index to embedding.

        (bs, n_cbs + additional_embs) -> (bs, n_cbs * dim_memory + additional_embs).
        """
        kv_input = emb_data[:, : self.input_dims_for_kv]
        other_emb_data = emb_data[:, self.input_dims_for_kv :]
        emb_data = self.kv_bottleneck(kv_input)
        emb_data = torch.cat((emb_data, other_emb_data), dim=-1)
        return emb_data


class EmbeddingMLP(nn.Module, KVBottleNeckMixin, ArchEmbeddingMixin):
    """
    This class turn the input embedding into one of the LoRA low-rank weight matrix (A or B) through a simple MLP.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        output_shape: torch.Size,
        hidden_dim: Union[int, list],
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
        bias=True,
        kv_bottleneck=False,
        num_memory_codebooks=2,
        num_memories=256,
        dim_memory=20,
        additional_embs=1,
        emb_input=False,
        emb_input_dims=None,
        norm_type="batch",
        batchnorm_momentum=0.1,
        dropout=0.0,
        residual=False,
        rescale_factor=1,
        attn_pooling=False,
    ) -> None:
        """
        Initialize the EmbeddingMLP module.

        Args:
            input_features (int): The number of input features, usually the Encoder's embedding dimension.
            output_features (int): The number of output features, usually the number of parameters in the LoRA A or B matrix.
            hidden_dim (int): The number of hidden dimensions in the MLP.
            hidden_layers (int): The number of hidden layers in the MLP. Default is 0.
            output_layer_groups (int): The number of groups in the output layer. Default is 1.
                If set to more than 1, the output layer will be a GroupedLinear layer to reduce the number of parameters.
        """
        super().__init__()

        self.input_features = input_features
        if hidden_dim is None:
            hidden_dim = self.input_features
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * (hidden_layers + 1)
        assert (
            len(hidden_dim) == hidden_layers + 1
        ), "hidden_dim must match hidden_layers"
        self.hidden_dim = hidden_dim

        self.out_feathres = output_features
        self.output_shape = output_shape
        self.norm_type = norm_type
        self.batchnorm_momentum = batchnorm_momentum
        self.bias = bias
        self.residual = residual
        self.dropout = dropout

        # Output layer
        if output_layer_groups > 1:
            if self.out_feathres > 8:
                # Grouped Linear has smaller number of parameters
                output_module = partial(
                    GroupedLinear, groups=output_layer_groups, bias=bias
                )
            else:
                output_module = partial(nn.Linear, bias=bias)
        else:
            output_module = partial(nn.Linear, bias=bias)

        # Key-Value Bottleneck for converting indices to embeddings
        if kv_bottleneck:
            self.kv_bottleneck, _ = self.setup_kv_bottleneck(
                num_memory_codebooks=num_memory_codebooks,
                num_memories=num_memories,
                dim_memory=dim_memory,
                additional_embs=additional_embs,
                emb_input=emb_input,
                emb_input_dims=emb_input_dims,
            )
        else:
            self.kv_bottleneck = None

        # MLP layers
        layers = [self._generate_linear_module(self.input_features, self.hidden_dim[0])]
        for idx in range(hidden_layers):
            in_features = self.hidden_dim[idx]
            out_features = self.hidden_dim[idx + 1]

            layers.append(
                self._generate_linear_module(
                    in_features=in_features, out_features=out_features
                )
            )
        layers.append(
            output_module(
                in_features=self.hidden_dim[-1], out_features=self.out_feathres
            )
        )
        self.mlp = nn.Sequential(*layers)
        self.rescale_factor = nn.Parameter(
            torch.tensor(float(rescale_factor)).float(), requires_grad=False
        )

        self.arch_features = 0

        if attn_pooling:
            self.attn_pooling = AttentionPooling(self.input_features)
        else:
            self.attn_pooling = None
        return

    def _generate_linear_module(self, in_features, out_features):
        if self.norm_type == "batch":
            norm = nn.BatchNorm1d(out_features, momentum=self.batchnorm_momentum)
        elif self.norm_type == "layer":
            norm = nn.LayerNorm(out_features)
        elif self.norm_type == "none":
            norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm type {self.norm_type}")

        layers = nn.Sequential(
            nn.Linear(
                in_features=in_features, out_features=out_features, bias=self.bias
            ),
            norm,
            nn.GELU(),
        )
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        if self.residual:
            # only add residual connection if the input and output features are the same
            if in_features == out_features:
                layers = Residual(layers)
        return layers

    def forward_mlp_with_arch(self, x):
        """Forward pass of the MLP with arch features."""
        # split the embedding into the original embedding and the arch features
        n_emb = x.shape[-1]
        n_arch = self.arch_features
        x, arch = torch.split(x, [n_emb - n_arch, n_arch], dim=-1)

        # forward pass through the MLP layers
        for layer in self.mlp:
            if isinstance(layer, ArchLinear):
                x = layer(x, arch)
            elif isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, ArchLinear):
                        x = sublayer(x, arch)
                    else:
                        x = sublayer(x)
            else:
                x = layer(x)
        return x

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EmbeddingMLP module.

        Args:
            embedding (torch.Tensor): The input embedding tensor.
                If the input tensor is 2D, it is assumed to be (bs, emb_dim).
                If the input tensor is 3D, it is assumed to be (bs, seq_len, emb_dim).

        Returns
        -------
            torch.Tensor: The output tensor after passing through the MLP layers.
        """
        if self.kv_bottleneck is not None:
            embedding = self.vq_ind_to_emb(embedding)

        if embedding.ndim == 3:
            # expect input embedding shape (bs, l, d)
            if self.attn_pooling is not None:
                embedding = self.attn_pooling(embedding)  # (bs, l, d) -> (bs, d)
            else:
                embedding = embedding.mean(dim=1)
        embedding = embedding * self.rescale_factor

        if self.arch_features > 0:
            x = self.forward_mlp_with_arch(embedding)
        else:
            x = self.mlp(embedding)

        a, b = self.output_shape
        x = rearrange(x, "bs (a b) -> bs a b", a=a, b=b)
        return x

    def zero_weights_and_bias(self):
        """
        Zero the weights and bias of the MLP's last layer, use this in B embedding.
        """
        last_layer = self.mlp[-1]

        assert isinstance(
            last_layer, (nn.Linear, GroupedLinear)
        ), f"Last layer is {type(last_layer)}, expected nn.Linear or GroupedLinear"
        if last_layer.bias is not None:
            last_layer.bias.data[...] = 0
        last_layer.weight.data[...] = 0
        return

    def scale_weights(self, example_embedding: np.ndarray):
        """
        Scale the weights of the MLP's first layer based on the example embedding, use this in A embedding.
        """
        with torch.no_grad():
            self.eval()
            try:
                self.cuda()
            except AssertionError:
                pass

            example_embedding = example_embedding.to(self.mlp[0].weight.device)
            example_output = self(example_embedding)
            mean, std = example_output.mean(), example_output.std()
            print(f"Embedding example mean: {mean}, std: {std}")
            rescale_factor = 1 / (std)
            self.rescale_factor = nn.Parameter(
                rescale_factor.clone().detach(), requires_grad=False
            )
            # rescale the embedding matrix
        return

    def fix_parameters(self):
        """
        Fix the parameters of the MLP.
        """
        for param in self.parameters():
            param.requires_grad = False
        return


class _TimeEncoder(nn.Module):
    """
    TimeEncoder is a module that encodes the time information into a vector.
    """

    def __init__(
        self, time_freqs: int, time_encoder_dims: list[int], time_encoder_dropout: float
    ):
        super().__init__()
        self.time_freqs = time_freqs
        self.encoder_dims = time_encoder_dims
        self.encoder_dropout = time_encoder_dropout

        layers = []
        _dims = [self.time_freqs * 2] + self.encoder_dims
        for i in range(len(_dims) - 1):
            layers.append(nn.Linear(_dims[i], _dims[i + 1]))
            layers.append(nn.SiLU())
            if self.encoder_dropout > 0:
                layers.append(nn.Dropout(self.encoder_dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (bs, 1)

        return: (bs, d_out)
        """
        x = cyclical_time_encoder(x, self.time_freqs)
        return self.layers(x)


class _SimpleEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        encoder_dropout: float,
        attn_pooling: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.encoder_dropout = encoder_dropout

        layers = []
        _dims = [self.input_dim] + self.encoder_dims
        for i in range(len(_dims) - 1):
            layers.append(nn.Linear(_dims[i], _dims[i + 1]))
            layers.append(nn.SiLU())
            if self.encoder_dropout > 0:
                layers.append(nn.Dropout(self.encoder_dropout))
        self.layers = nn.Sequential(*layers)
        if attn_pooling:
            self.attn_pooling = AttentionPooling(_dims[-1])
        else:
            self.attn_pooling = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (bs, ..., d_in) or (bs, ..., d_in)

        return: (bs, ..., d_out)
        """
        if x.ndim == 3:
            attn_mask = self.attn_pooling._make_all_zero_mask(x)
        x = self.layers(x)

        if x.ndim == 3 and self.attn_pooling is not None:
            # expect input embedding shape (bs, l, d)
            x = self.attn_pooling(x, attn_mask)
        return x


class CondFlowModule(nn.Module):
    """
    CondFlowModule is a module that process condtional input
    and combine them into single embedding for flow model.
    """

    def __init__(
        self,
        cell_emb_dim: int,
        cond_emb_dim: int | dict[int],
        cell_encoder_dims: list[int] = (256, 256),
        cell_encoder_dropout: float = 0.1,
        cell_attn_pooling: bool = True,
        time_freqs: int = 128,
        time_encoder_dims: list[int] = (128, 128),
        time_encoder_dropout: float = 0.1,
        cond_encoder_dims: list[int] = (128, 128),
        cond_encoder_dropout: float = 0.1,
        cond_attn_pooling: bool = True,
    ):
        super().__init__()

        cell_encoder_dims = list(cell_encoder_dims)
        time_encoder_dims = list(time_encoder_dims)
        cond_encoder_dims = list(cond_encoder_dims)

        # cell embedding encoder
        self.cell_encoder = _SimpleEncoder(
            input_dim=cell_emb_dim,
            encoder_dims=cell_encoder_dims,
            encoder_dropout=cell_encoder_dropout,
            attn_pooling=cell_attn_pooling,
        )

        # time encoder
        self.time_encoder = _TimeEncoder(
            time_freqs=time_freqs,
            time_encoder_dims=time_encoder_dims,
            time_encoder_dropout=time_encoder_dropout,
        )

        # condition encoder
        # TODO: this condition encoder only consider the most simple case,
        # need to change to cellflow's condition encoder for more complex case
        if isinstance(cond_emb_dim, int):
            self.cond_encoder = _SimpleEncoder(
                input_dim=cond_emb_dim,
                encoder_dims=cond_encoder_dims,
                encoder_dropout=cond_encoder_dropout,
                attn_pooling=cond_attn_pooling,
            )
        elif isinstance(cond_emb_dim, dict):
            self.cond_encoder = nn.ModuleDict()
            for k, v in cond_emb_dim.items():
                self.cond_encoder[k] = _SimpleEncoder(
                    input_dim=v,
                    encoder_dims=cond_encoder_dims[k],
                    encoder_dropout=cond_encoder_dropout[k],
                    attn_pooling=cond_attn_pooling,
                )
        else:
            raise ValueError(
                f"cond_emb_dim must be int or dict, but got {type(cond_emb_dim)}"
            )
        self.output_dim = (
            cell_encoder_dims[-1] + time_encoder_dims[-1] + cond_encoder_dims[-1]
        )
        return

    def forward(
        self,
        cell_emb: torch.Tensor,
        time: torch.Tensor,
        cond_emb: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        cell_emb: (bs, d_e) or (bs, n, h_e)
        time_emb: (bs, 1) or (bs, h_t)
        cond_emb: (bs, d_c) or (bs, n, d_c)

        return: (bs, h_e + h_t + h_c)
        """
        # encode cell embedding
        cell_emb = self.cell_encoder(cell_emb)
        if cell_emb.ndim == 3:
            # if not attn_pooling in encoder,
            # here just perform mean pooling on the cell embedding
            cell_emb = cell_emb.mean(dim=1)

        # encode time embedding
        time_emb = self.time_encoder(time)

        # encode condition embedding
        if isinstance(self.cond_encoder, dict):
            assert isinstance(
                cond_emb, dict
            ), f"cond_emb must be dict, but got {type(cond_emb)}"
            cond_emb = [
                encoder[cond_emb[k]] for k, encoder in self.cond_encoder.items()
            ]
            cond_emb = torch.cat(cond_emb, dim=-1)
        else:
            cond_emb = self.cond_encoder(cond_emb)

        if cond_emb.ndim == 3:
            # if not attn_pooling in encoder,
            # here just perform mean pooling on the condition embedding
            cond_emb = cond_emb.mean(dim=1)

        # combine all embeddings
        emb = torch.cat([cell_emb, time_emb, cond_emb], dim=-1)
        return emb
