from functools import partial
from typing import Union

import numpy as np
import torch
from einops import einsum, rearrange, reduce, repeat
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import VectorQuantize

from bolero.tl.generic.module import GroupedLinear, Residual


class ArchEmbedding(nn.Module):
    """
    ArchEmbedding mimicing scArches' dimention injection into MLP layers.
    """

    def __init__(
        self, input_features, output_features_list, first_only=False, bias=True
    ):
        """
        Initialize the ArchEmbedding module.

        Parameters
        ----------
        input_features : int
            The number of input features.
        output_features_list : list
            The list of output features for each MLP layer to be injected.
        first_only : bool
            If True, only inject the first layer.
        """
        super().__init__()
        self.input_features = input_features
        self.output_features_list = output_features_list
        self.first_only = first_only

        self.arch_linears = nn.ModuleList()
        for layer_id, output_features in enumerate(output_features_list):
            if self.first_only and layer_id > 0:
                layer = None
            else:
                layer = nn.Linear(input_features, output_features, bias=bias)
            self.arch_linears.append(layer)

    def forward(self, x):
        """
        ArchEmbedding forward

        Here we calculate the value to be injected into the MLP layers as its independent from the MLP themselves.
        """
        results = []
        for layer in self.arch_linears:
            if layer is None:
                results.append(None)
            else:
                results.append(layer(x))
        return results


class ArchEmbeddingMixin:
    """
    Mixin class to add Arch Embedding to MLP-like modules.
    """

    def add_arch_embedding(
        self, input_features: int, mlp_module: nn.Module, first_only=False, bias=True
    ):
        """
        Add an ArchEmbedding to the MLP.

        Parameters
        ----------
        input_features : int
            The number of input features.
        mlp_module : nn.Module
            The MLP module to be injected.
        first_only : bool
            If True, only inject the first layer.
        """
        output_features = [
            layer.out_features for layer in mlp_module if isinstance(layer, nn.Linear)
        ]

        arch_embedding = ArchEmbedding(
            input_features=input_features,
            output_features_list=output_features,
            first_only=first_only,
            bias=bias,
        )

        self.arch_modules.append(arch_embedding)
        return

    def forward_arch_embedding(
        self, arch_embedding: list[torch.Tensor]
    ) -> list[list[torch.Tensor]]:
        """
        Collect the ArchEmbedding results to be injected to MLP.

        Results are aggregated per layer.
        """
        # forward the arch_embedding
        arch_results = []
        for _emb, arch_module in zip(arch_embedding, self.arch_modules):
            arch_results.append(arch_module(_emb))

        # aggregate the results per layer
        final_results = []
        for layer_results in zip(*arch_results):
            layer_results = [
                layer_result
                for layer_result in layer_results
                if layer_result is not None
            ]
            if len(layer_results) > 1:
                # sum the results
                layer_results = torch.stack(layer_results, dim=0).sum(dim=0)
            elif len(layer_results) == 1:
                layer_results = layer_results[0]
            else:
                layer_results = None
            final_results.append(layer_results)

        return layer_results

    def forward_arch_embedding_and_mlp(
        self,
        mlp_module: nn.Module,
        embedding: torch.Tensor,
        arch_embedding: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of the ArchEmbedding and MLP layers."""
        arch_layer_results = self.forward_arch_embedding(arch_embedding)

        x = embedding

        linear_idx = 0
        for layer in mlp_module:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if arch_layer_results[linear_idx] is not None:
                    x += arch_layer_results[linear_idx]
                linear_idx += 1
            else:
                x = layer(x)
        return x

    def freeze_everything_except_arch(self):
        """
        Freeze all parameters except the ArchEmbedding parameters.
        """
        for name, params in self.named_parameters():
            if "arch_modules" not in name:
                params.requires_grad = False
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

        # ArchEmbedding placeholder
        self.arch_modules = torch.nn.ModuleList()
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

    def forward(
        self,
        embedding: torch.Tensor,
        emb_weights: torch.Tensor = None,
        arch_embedding=None,
    ) -> torch.Tensor:
        """
        Forward pass of the EmbeddingMLP module.

        Args:
            embedding (torch.Tensor): The input embedding tensor.
                If the input tensor is 2D, it is assumed to be (bs, emb_dim).
                If the input tensor is 3D, it is assumed to be (bs, seq_len, emb_dim).
            emb_weights (torch.Tensor): The embedding weights tensor.
                Only applicable for 3D input tensor, where the final weights will be weighted sum across the sequence length dimension.
                it is assumed to be (bs, seq_len).
                If None, the output will be the mean across the sequence length dimension.

        Returns
        -------
            torch.Tensor: The output tensor after passing through the MLP layers.
        """
        if self.kv_bottleneck is not None:
            embedding = self.vq_ind_to_emb(embedding)

        ndim = embedding.ndim
        bs = embedding.shape[0]
        if ndim == 3:
            # expect input shape (bs, seq_len, emb_dim)
            embedding = rearrange(embedding, "bs l d -> (bs l) d")

            if arch_embedding is not None:
                arch_embedding = [
                    rearrange(arch_emb, "bs l d -> (bs l) d")
                    for arch_emb in arch_embedding
                ]

        embedding = embedding * self.rescale_factor
        if arch_embedding is not None:
            # MLP forward with ArchEmbedding injection
            x = self.forward_arch_embedding_and_mlp(
                mlp_module=self.mlp, embedding=embedding, arch_embedding=arch_embedding
            )
        else:
            # normal MLP forward
            x = self.mlp(embedding)

        if ndim == 3:
            x = rearrange(x, "(bs l) d -> bs l d", bs=bs)
            if emb_weights is None:
                x = x.mean(dim=1)
            else:
                emb_weights = F.softmax(emb_weights, dim=1)
                x = einsum(x, emb_weights, "bs l d, bs l -> bs d")

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


class ConditionalConv1dHead(nn.Module):
    """
    This class is function as an Conv1d head where the weights are conditioned on the cell embeddings.

    Comparing to ConditionalConvLoRA, this class don't have the base convolutional layer.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, embedding_dim, preset=None, **kwargs
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        if preset == "scooby":
            self.setup_scooby_head()
        else:
            self.setup_head(**kwargs)

    def setup_head(self, **kwargs):
        """
        Setup the head.
        """
        output_features = (self.input_dim + 1) * self.output_dim
        output_shape = (self.input_dim + 1, self.output_dim)

        self.embedding_mlp = EmbeddingMLP(
            input_features=self.embedding_dim,
            output_features=output_features,
            output_shape=output_shape,
            hidden_dim=self.hidden_dim,
            **kwargs,
        )
        self.pre_embedding_conv = nn.Identity()

        self.batch_conv1d = torch.vmap(nn.functional.conv1d)

    def setup_scooby_head(self):
        """
        Setup the head for the Scooby model.
        """
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
            nn.Conv1d(self.input_dim, 4096, 1),
            nn.GELU(),
            nn.Conv1d(4096, self.input_dim, 1),
            nn.GELU(),
        )

        nn.init.zeros_(self.embedding_mlp[-1].bias)
        nn.init.zeros_(self.pre_embedding_conv[-2].weight)
        nn.init.zeros_(self.pre_embedding_conv[-2].bias)

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
        weight_and_bias = self.embedding_mlp(embedding).view(
            -1, self.input_dim + 1, self.output_dim, 1
        )

        # (bs, i + 1, o, 1) -> (bs, i, o, 1), (bs, 1, o, 1)
        weight, bias = torch.split(weight_and_bias, [self.input_dim, 1], dim=1)
        weight = rearrange(weight, "bs i o 1 -> bs o i 1")
        bias = rearrange(bias, "bs 1 o 1 -> bs o")

        # (bs, c, l) -> (bs, output_dim, l)
        result = self.batch_conv1d(x, weight, bias)
        return result
