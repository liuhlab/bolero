from typing import Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from bolero.tl.generic.module import Residual
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


# TODO: ArchEmbedding maybe redundant to ConditionEmbeddingModule
# Main difference is that ArchEmbedding takes additional dims in embedding input
# and maintain separate encoder and added to each layer of EmbeddingMLP
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


class EmbeddingMLP(nn.Module, ArchEmbeddingMixin):
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
        norm_type="layer",
        batchnorm_momentum=0.1,
        dropout=0.0,
        residual=False,
        attn_pooling=False,
    ) -> None:
        """
        Initialize the EmbeddingMLP module.

        Args:
            input_features (int): The number of input features, usually the Encoder's embedding dimension.
            output_features (int): The number of output features, usually the number of parameters in the LoRA A or B matrix.
            hidden_dim (int): The number of hidden dimensions in the MLP.
            hidden_layers (int): The number of hidden layers in the MLP. Default is 0.
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
        self.residual = residual
        self.dropout = dropout

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
            nn.Linear(in_features=self.hidden_dim[-1], out_features=self.out_feathres)
        )
        self.mlp = nn.Sequential(*layers)

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
            nn.Linear(in_features=in_features, out_features=out_features, bias=True),
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
        if embedding.ndim == 3:
            # expect input embedding shape (bs, l, d)
            if self.attn_pooling is not None:
                embedding = self.attn_pooling(embedding)  # (bs, l, d) -> (bs, d)
            else:
                embedding = embedding.mean(dim=1)

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
            last_layer, nn.Linear
        ), f"Last layer is {type(last_layer)}, expected nn.Linear"
        if last_layer.bias is not None:
            last_layer.bias.data[...] = 0
        last_layer.weight.data[...] = 0
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
        while x.ndim < 2:
            x = x.unsqueeze(0)

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


class _MultiSimpleEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int | list[int],
        n_datasets: int,
        encoder_cls: type,
        **encoder_kwargs,
    ):
        super().__init__()

        modules = []
        if isinstance(input_dim, int):
            input_dim = [input_dim] * n_datasets
        assert (
            len(input_dim) == n_datasets
        ), "input_dim must be a list of length n_datasets"
        for _dim in input_dim:
            modules.append(
                encoder_cls(
                    input_dim=_dim,
                    **encoder_kwargs,
                )
            )
        self.encoders = nn.ModuleList(modules)
        self.output_dim = self.encoders[0].encoder_dims[-1]

    def forward(
        self, embedding: torch.Tensor, dataset_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        embedding: (bs, input_dim)
        dataset_keys: (bs,) int tensor in range [0, num_datasets)
        Returns: (bs, hidden_dim)
        """
        bs = embedding.shape[0]
        outputs = embedding.new_zeros(bs, self.output_dim)

        for i, encoder in enumerate(self.encoders):
            idx = (dataset_keys == i).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                encoded = encoder(embedding[idx])
                outputs[idx] = encoded.to(outputs.dtype)
        return outputs


class ConditionEmbeddingModule(nn.Module):
    """
    ConditionEmbeddingModule is a module that combines multiple embedding sources
    (cell emb, cond emb) into single hidden embedding.
    """

    def __init__(
        self,
        cell_emb_dim: int,
        cond_emb_dim: int | dict[int],
        cell_encoder_dims: list[int] = (256, 256),
        cell_encoder_dropout: float = 0.1,
        cell_attn_pooling: bool = True,
        cond_encoder_dims: list[int] = (256, 256),
        cond_encoder_dropout: float = 0.1,
        cond_attn_pooling: bool = True,
        n_cell_encoder: int = 1,
    ):
        super().__init__()

        cell_encoder_dims = list(cell_encoder_dims)
        cond_encoder_dims = list(cond_encoder_dims)

        # cell embedding encoder
        self.n_cell_encoder = n_cell_encoder
        if n_cell_encoder == 1:
            self.cell_encoder = _SimpleEncoder(
                input_dim=cell_emb_dim,
                encoder_dims=cell_encoder_dims,
                encoder_dropout=cell_encoder_dropout,
                attn_pooling=cell_attn_pooling,
            )
        else:
            self.cell_encoder = _MultiSimpleEncoder(
                input_dim=cell_emb_dim,
                n_datasets=n_cell_encoder,
                encoder_cls=_SimpleEncoder,
                encoder_dims=cell_encoder_dims,
                encoder_dropout=cell_encoder_dropout,
                attn_pooling=cell_attn_pooling,
            )

        # condition encoder
        if isinstance(cond_emb_dim, int):
            self.cond_encoder = _SimpleEncoder(
                input_dim=cond_emb_dim,
                encoder_dims=cond_encoder_dims,
                encoder_dropout=cond_encoder_dropout,
                attn_pooling=cond_attn_pooling,
            )
            cond_enc_output_dim = cond_encoder_dims[-1]
        elif isinstance(cond_emb_dim, dict):
            self.cond_encoder = nn.ModuleDict()
            for k, v in cond_emb_dim.items():
                self.cond_encoder[k] = _SimpleEncoder(
                    input_dim=v,
                    encoder_dims=cond_encoder_dims,
                    encoder_dropout=cond_encoder_dropout,
                    attn_pooling=cond_attn_pooling,
                )
            cond_enc_output_dim = len(self.cond_encoder) * cond_encoder_dims[-1]
        else:
            self.cond_encoder = None
            print(
                f"cond_emb_dim is not int nor dict, no cond_encoder for the model. "
                f"Make sure this looks intended: {cond_emb_dim}."
            )
            cond_enc_output_dim = 0

        self.output_dim = cell_encoder_dims[-1] + cond_enc_output_dim
        return

    def forward(
        self,
        cell_emb: torch.Tensor,
        cond_emb: torch.Tensor | dict[str, torch.Tensor],
        dataset_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        cell_emb: (bs, d_e) or (bs, n, d_e)
        cond_emb: (bs, d_c) or (bs, n, d_c)

        return: (bs, h_e + h_t + h_c)
        """
        to_cat = []

        # encode cell embedding
        if self.n_cell_encoder > 1:
            assert (
                dataset_keys is not None
            ), "dataset_key must be provided for multi-dataset case"
            cell_emb = self.cell_encoder(cell_emb, dataset_keys)
        else:
            cell_emb = self.cell_encoder(cell_emb)
        if cell_emb.ndim == 3:
            # if not attn_pooling in encoder,
            # here just perform mean pooling on the cell embedding
            cell_emb = cell_emb.mean(dim=1)
        to_cat.append(cell_emb)

        # encode condition embedding
        if self.cond_encoder is not None:
            if isinstance(self.cond_encoder, nn.ModuleDict):
                assert isinstance(
                    cond_emb, dict
                ), f"cond_emb must be dict, but got {type(cond_emb)}"
                cond_emb = [
                    encoder(cond_emb[k]) for k, encoder in self.cond_encoder.items()
                ]
                cond_emb = torch.cat(cond_emb, dim=-1)
            else:
                cond_emb = self.cond_encoder(cond_emb)

            if cond_emb.ndim == 3:
                # if not attn_pooling in encoder,
                # here just perform mean pooling on the condition embedding
                cond_emb = cond_emb.mean(dim=1)
            to_cat.append(cond_emb)

        # combine all embeddings
        emb = torch.cat(to_cat, dim=-1)
        return emb


class CondFlowModule(nn.Module):
    """
    CondFlowModule is a module that process condtional input (cell emb, cond emb, time)
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
        n_cell_encoder: int = 1,
    ):
        super().__init__()

        cell_encoder_dims = list(cell_encoder_dims)
        time_encoder_dims = list(time_encoder_dims)
        cond_encoder_dims = list(cond_encoder_dims)

        # cell embedding encoder
        self.n_cell_encoder = n_cell_encoder
        if n_cell_encoder == 1:
            self.cell_encoder = _SimpleEncoder(
                input_dim=cell_emb_dim,
                encoder_dims=cell_encoder_dims,
                encoder_dropout=cell_encoder_dropout,
                attn_pooling=cell_attn_pooling,
            )
        else:
            self.cell_encoder = _MultiSimpleEncoder(
                input_dim=cell_emb_dim,
                n_datasets=n_cell_encoder,
                encoder_cls=_SimpleEncoder,
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
            cond_enc_output_dim = cond_encoder_dims[-1]
        elif isinstance(cond_emb_dim, dict):
            self.cond_encoder = nn.ModuleDict()
            for k, v in cond_emb_dim.items():
                self.cond_encoder[k] = _SimpleEncoder(
                    input_dim=v,
                    encoder_dims=cond_encoder_dims,
                    encoder_dropout=cond_encoder_dropout,
                    attn_pooling=cond_attn_pooling,
                )
            cond_enc_output_dim = len(self.cond_encoder) * cond_encoder_dims[-1]
        else:
            self.cond_encoder = None
            print(
                f"cond_emb_dim is not int nor dict, no cond_encoder for the model. "
                f"Make sure this looks intended: {cond_emb_dim}."
            )
            cond_enc_output_dim = 0

        self.output_dim = (
            cell_encoder_dims[-1] + time_encoder_dims[-1] + cond_enc_output_dim
        )
        return

    def forward(
        self,
        cell_emb: torch.Tensor,
        time: torch.Tensor,
        cond_emb: torch.Tensor | dict[str, torch.Tensor],
        dataset_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        cell_emb: (bs, d_e) or (bs, n, d_e)
        time: (bs, 1) or (1,)
        cond_emb: (bs, d_c) or (bs, n, d_c)

        return: (bs, h_e + h_t + h_c)
        """
        to_cat = []

        # encode cell embedding
        if self.n_cell_encoder > 1:
            assert (
                dataset_keys is not None
            ), "dataset_key must be provided for multi-dataset case"
            cell_emb = self.cell_encoder(cell_emb, dataset_keys)
        else:
            cell_emb = self.cell_encoder(cell_emb)
        if cell_emb.ndim == 3:
            # if not attn_pooling in encoder,
            # here just perform mean pooling on the cell embedding
            cell_emb = cell_emb.mean(dim=1)
        to_cat.append(cell_emb)

        # encode time embedding
        time_emb: torch.Tensor = self.time_encoder(time)
        if time_emb.shape[0] == 1:
            # if the input is a single time step, repeat it to match the batch size
            bs = cell_emb.shape[0]
            time_emb = time_emb.repeat(bs, 1)
        to_cat.append(time_emb)

        # encode condition embedding
        if self.cond_encoder is not None:
            if isinstance(self.cond_encoder, nn.ModuleDict):
                assert isinstance(
                    cond_emb, dict
                ), f"cond_emb must be dict, but got {type(cond_emb)}"
                cond_emb = [
                    encoder(cond_emb[k]) for k, encoder in self.cond_encoder.items()
                ]
                cond_emb = torch.cat(cond_emb, dim=-1)
            else:
                cond_emb = self.cond_encoder(cond_emb)

            if cond_emb.ndim == 3:
                # if not attn_pooling in encoder,
                # here just perform mean pooling on the condition embedding
                cond_emb = cond_emb.mean(dim=1)
            to_cat.append(cond_emb)

        # combine all embeddings
        emb = torch.cat(to_cat, dim=-1)
        return emb


class ConditionEmbeddingModuleNoEffect(ConditionEmbeddingModule):
    """
    Only return unchanged cell embedding, this is for ablation study.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = kwargs["cell_emb_dim"]

    def forward(self, cell_emb: torch.Tensor, *args, **kwargs):
        """Do nothing and return the cell embedding."""
        return cell_emb
