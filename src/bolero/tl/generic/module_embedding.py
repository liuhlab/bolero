from copy import deepcopy
from typing import Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from bolero.tl.generic.module import Residual

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


class _SimpleEncoder(nn.Module):
    """
    _SimpleEncoder class handles encoder network for single datasets.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        encoder_dropout: float,
        attn_pooling: bool = False,
        norm_type: str | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.encoder_dropout = encoder_dropout

        layers = []
        _dims = [self.input_dim] + self.encoder_dims
        for i in range(len(_dims) - 1):
            layers.append(nn.Linear(_dims[i], _dims[i + 1]))
            if norm_type == "layer":
                layers.append(nn.LayerNorm(_dims[i + 1]))
            elif norm_type == "batch":
                layers.append(nn.BatchNorm1d(_dims[i + 1]))
            else:
                assert norm_type in (None, "none"), f"Unknown norm type {norm_type}"
            layers.append(nn.SiLU())
            if self.encoder_dropout > 0:
                layers.append(nn.Dropout(self.encoder_dropout))
        self.layers = nn.Sequential(*layers)
        if attn_pooling:
            self.attn_pooling = AttentionPooling(_dims[-1])
        else:
            self.attn_pooling = None

        self.output_dim = _dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (bs, ..., d_in) or (bs, ..., d_in)

        return: (bs, ..., d_out)
        """
        if x.ndim == 3:
            attn_mask = self.attn_pooling._make_all_zero_mask(x)
        x = self.layers(x)

        if x.ndim == 3:
            if self.attn_pooling is not None:
                # expect input embedding shape (bs, l, d)
                x = self.attn_pooling(x, attn_mask)
            else:
                # if not attn_pooling in encoder,
                # here just perform mean pooling on the condition embedding
                x = x.mean(dim=1)
        return x  # shape (bs, d)


class _SimpleEncoderDict(nn.ModuleDict):
    def __init__(
        self,
        input_dims: dict[str:int],
        encoder_dims: list[int],
        encoder_dropout: float,
        attn_pooling: bool = False,
        norm_type: str | None = None,
        pool_mode="concat",
    ):
        _dict = {
            key: _SimpleEncoder(
                input_dim,
                encoder_dims=encoder_dims,
                encoder_dropout=encoder_dropout,
                attn_pooling=attn_pooling,
                norm_type=norm_type,
            )
            for key, input_dim in input_dims.items()
        }
        self.pool_mode = pool_mode
        super().__init__(_dict)

        if pool_mode == "concat":
            self.output_dim = encoder_dims[-1] * len(_dict)
        else:
            self.output_dim = encoder_dims[-1]

    def forward(self, input_emb: dict[str, torch.Tensor]):
        assert isinstance(
            input_emb, dict
        ), f"input_emb must be dict, but got {type(input_emb)}"
        try:
            out_emb = [_encoder(input_emb[k]) for k, _encoder in self.items()]
        except KeyError:
            raise KeyError(
                f"input_emb must have keys {list(self.keys())}, but got {list(input_emb.keys())}"
            ) from None
        if self.pool_mode == "concat":
            out_emb = torch.cat(out_emb, dim=-1)
        else:
            # sum out emb
            out_emb = torch.stack(out_emb).mean(dim=0)
        return out_emb


def _create_encoder(
    emb_dim: int | dict[int],
    encoder_dims: list[int],
    encoder_dropout: float,
    attn_pooling: bool,
    norm_type: str | None = None,
    pool_mode: str = "concat",
) -> tuple[nn.Module | nn.ModuleDict, int]:
    encoder_kwargs = {
        "encoder_dims": encoder_dims,
        "encoder_dropout": encoder_dropout,
        "attn_pooling": attn_pooling,
        "norm_type": norm_type,
    }

    # condition encoder
    if isinstance(emb_dim, int):
        encoder = _SimpleEncoder(input_dim=emb_dim, **encoder_kwargs)
        enc_output_dim = encoder.output_dim
    elif isinstance(emb_dim, dict):
        encoder = _SimpleEncoderDict(
            input_dims=emb_dim, pool_mode=pool_mode, **encoder_kwargs
        )
        enc_output_dim = encoder.output_dim
    else:
        encoder = None
        enc_output_dim = 0
    return encoder, enc_output_dim


class ConditionEmbeddingModule(nn.Module):
    """
    ConditionEmbeddingModule is a module that combines multiple embedding sources
    (cell emb, cond emb) into single hidden embedding.
    """

    def __init__(
        self,
        cell_emb_dim: int,
        cond_emb_dim: int | dict[int] | None,
        cell_encoder_dims: list[int] = (256, 256),
        cell_encoder_dropout: float = 0.1,
        cell_attn_pooling: bool = True,
        cond_encoder_dims: list[int] = (256, 256),
        cond_encoder_dropout: float = 0.1,
        cond_attn_pooling: bool = True,
    ):
        super().__init__()

        cell_encoder_dims = list(cell_encoder_dims)
        cond_encoder_dims = list(cond_encoder_dims)

        # cell embedding encoder
        self.cell_encoder, cell_out_dim = _create_encoder(
            emb_dim=cell_emb_dim,
            encoder_dims=cell_encoder_dims,
            encoder_dropout=cell_encoder_dropout,
            attn_pooling=cell_attn_pooling,
        )

        self.cond_encoder, cond_out_dim = _create_encoder(
            emb_dim=cond_emb_dim,
            encoder_dims=cond_encoder_dims,
            encoder_dropout=cond_encoder_dropout,
            attn_pooling=cond_attn_pooling,
        )
        if self.cond_encoder is None:
            print(
                f"cond_emb_dim ({cond_emb_dim}) is not int nor dict, no cond_encoder for the model."
            )

        self.output_dim = cell_out_dim + cond_out_dim
        return

    def forward(
        self,
        cell_emb: torch.Tensor,
        cond_emb: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        cell_emb: (bs, d_e) or (bs, n, d_e)
        cond_emb: (bs, d_c) or (bs, n, d_c)

        return: (bs, h_e + h_c)
        """
        # encode cell embedding
        cell_emb = self.cell_encoder(cell_emb)

        if self.cond_encoder is None:
            final_emb = cell_emb
        else:
            cond_emb = self.cond_encoder(cond_emb)
            final_emb = torch.cat([cell_emb, cond_emb], dim=-1)
        return final_emb


class ConditionEmbeddingModuleMulti(nn.Module):
    """
    ConditionEmbeddingModuleMulti is a module that combines multiple embedding sources
    (cell emb, cond emb) into single hidden embedding.

    Comparing from ConditionEmbeddingModule, this module supports dataset specific
    cell and condition encoders and also a shared condition encoder.
    """

    def _validate_dataset_info(self):
        assert (
            len(self.dataset_specific_dims) == self.n_datasets
        ), "dataset_specific_dims must have the same length as n_datasets."

        dataset_dims = {}
        for dataset_name, dims in self.dataset_specific_dims.items():
            dims_dict = {}
            if isinstance(dims, int):
                # if only a number is provided, assume this number is cell emb dim,
                # and this dataset has no cond emb.
                dims_dict["cell"] = dims
                dims_dict["cond"] = None
            else:
                try:
                    dims_dict["cell"] = dims["cell"]
                    # IMPORTANT: remove cell key from dims and handle the rest as cond emb dims
                    dims = {k: v for k, v in dims.items() if k != "cell"}
                except KeyError:
                    raise KeyError(
                        f"dataset {dataset_name} must have 'cell' key, "
                        f"got {list(dims.keys())}"
                    ) from None
                # remaining parts are cond emb
                dims_dict["cond"] = dims if len(dims) > 0 else None
            dataset_dims[dataset_name] = dims_dict
        self.dataset_specific_dims = dataset_dims
        # schema of dataset_specific_dims:
        # {
        #     dataset_name: {
        #         "cell": int,
        #         "cond": dict[int] | None
        #     }
        # }
        return

    def __init__(
        self,
        dataset_order: list[str],
        dataset_specific_dims: dict[str, int | dict[int]],
        dataset_shared_dims: int | dict[int],
        encoder_dims: list[int] = (256, 256),
        encoder_dropout: float = 0.1,
        attn_pooling: bool = True,
        norm_type: str | None = None,
        **kwargs,
    ):
        super().__init__()

        # prepare basic info
        self.n_datasets = len(dataset_order)
        self.dataset_order = dataset_order
        self.dataset_specific_dims = dataset_specific_dims
        self.dataset_shared_dims = dataset_shared_dims
        encoder_dims = list(encoder_dims)
        self.encoder_kwargs = {
            "encoder_dims": encoder_dims,
            "encoder_dropout": encoder_dropout,
            "attn_pooling": attn_pooling,
            "norm_type": norm_type,
        }
        self._validate_dataset_info()

        # dataset specific cell encoder
        self._setup_dataset_specific_encoder()
        self._setup_dataset_shared_encoder()

        self.output_dim = encoder_dims[-1] * 3
        return

    def _setup_dataset_specific_encoder(self):
        self.cell_encoder_dict = nn.ModuleDict()
        self.cond_encoder_dict = nn.ModuleDict()

        for dataset, dims in self.dataset_specific_dims.items():
            cell_dim = dims["cell"]
            cond_dim = dims["cond"]
            encoder_kwargs = deepcopy(self.encoder_kwargs)

            if cond_dim is None:
                # no cond encoder, increase encoder dim to keep final shape the same
                encoder_kwargs["encoder_dims"] = [
                    d * 2 for d in encoder_kwargs["encoder_dims"]
                ]
                cell_encoder, _ = _create_encoder(
                    emb_dim=cell_dim,
                    **encoder_kwargs,
                )
                self.cell_encoder_dict[dataset] = cell_encoder
                self.cond_encoder_dict[dataset] = None
            else:
                cell_encoder, _ = _create_encoder(
                    emb_dim=cell_dim,
                    **encoder_kwargs,
                )
                cond_encoder, _ = _create_encoder(
                    emb_dim=cond_dim,
                    # pool_mode mean makes output dim constant
                    pool_mode="mean",
                    **encoder_kwargs,
                )
                self.cell_encoder_dict[dataset] = cell_encoder
                self.cond_encoder_dict[dataset] = cond_encoder
        return

    def _setup_dataset_shared_encoder(self):
        self.shared_encoder, _ = _create_encoder(
            emb_dim=self.dataset_shared_dims,
            pool_mode="mean",
            **self.encoder_kwargs,
        )
        return

    def forward(
        self,
        cell_emb: list[torch.Tensor],
        cond_emb: list[torch.Tensor | dict[str, torch.Tensor]],
        shared_emb: torch.Tensor,
        dataset_keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        cell_emb: list[(d_e,)]
        cond_emb: list[(d_c,)]

        return: (bs, h_e + h_c + h_shared)
        """
        dataset_keys = dataset_keys.cpu().numpy()

        dataset_specific_emb = []
        for _cell_emb, _cond_emb, dataset_idx in zip(cell_emb, cond_emb, dataset_keys):
            dataset_name = self.dataset_order[dataset_idx]
            cell_encoder = self.cell_encoder_dict[dataset_name]
            cond_encoder = self.cond_encoder_dict[dataset_name]

            _cell_emb = cell_encoder(_cell_emb)
            if _cell_emb.ndim == 2:
                _cell_emb.squeeze_(0)

            if cond_encoder is None:
                combine_emb = _cell_emb
            else:
                _cond_emb = cond_encoder(_cond_emb)
                if _cond_emb.ndim == 2:
                    _cond_emb.squeeze_(0)
                combine_emb = torch.cat([_cell_emb, _cond_emb], dim=-1)
            dataset_specific_emb.append(combine_emb)
        dataset_specific_emb = torch.stack(dataset_specific_emb)

        shared_emb["__shared_data__"] = shared_emb["__shared_data__"].to(torch.float32)
        dataset_shared_emb = self.shared_encoder(shared_emb)

        final_emb = torch.cat([dataset_specific_emb, dataset_shared_emb], dim=-1)
        return final_emb

    def forward_single_dataset(
        self,
        cell_emb: torch.Tensor,
        cond_emb: torch.Tensor | dict[str, torch.Tensor],
        shared_emb: torch.Tensor,
        dataset_key: int,
    ):
        """Forward pass for a single dataset."""
        dataset_name = self.dataset_order[dataset_key]
        cell_encoder = self.cell_encoder_dict[dataset_name]
        cond_encoder = self.cond_encoder_dict[dataset_name]

        # (bs, cell) -> (bs, cell_hidden)
        cell_emb = cell_encoder(cell_emb)

        if cond_encoder is None:
            combine_emb = cell_emb
        else:
            cond_emb = cond_encoder(cond_emb)
            combine_emb = torch.cat([cell_emb, cond_emb], dim=-1)

        shared_emb["__shared_data__"] = shared_emb["__shared_data__"].to(torch.float32)
        dataset_shared_emb = self.shared_encoder(shared_emb)
        final_emb = torch.cat([combine_emb, dataset_shared_emb], dim=-1)
        return final_emb


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
