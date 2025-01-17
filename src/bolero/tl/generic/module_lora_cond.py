"""
Making LoRA fine tuning weights conditionally depending on the embedding input.
"""

from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from torch.nn import functional as F

from .module import Conv1dWrapper
from .module_embedding import EmbeddingMLP
from .module_lora import (
    DoRAMixin,
    LoRAConv,
    LoRAEmbedding,
    LoRALinear,
    mark_only_lora_as_trainable,
    name_in_patterns,
    set_submodule_by_name,
)


class UnconditionalParameters(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.output_shape = shape
        self.values = nn.Parameter(torch.randn(shape), requires_grad=True)

    def forward(self, embedding, *args, **kwargs):
        """Forward"""
        bs = embedding.shape[0]
        values = repeat(self.values, "a b -> bs a b", bs=bs)
        return values

    def zero_weights_and_bias(self):
        """Make the values zero."""
        self.values.data[...] = 0
        return


class ConditionalLoRALayer:
    def __init__(
        self,
        shape_a: torch.Size,
        shape_b: torch.Size,
        lora_alpha: int,
        lora_scale: int,
        lora_dropout: float,
        emb_input_features: int,
        hidden_dim: int,
        hidden_layers: int,
        output_layer_groups: int,
        conditional_b: bool = True,
        kv_bottleneck: bool = False,
        num_memory_codebooks: int = 2,
        num_memories: int = 256,
        dim_memory: int = 20,
        additional_embs: int = 1,
        emb_input=False,
        emb_input_dims=None,
        norm_type: str = "batch",
        batchnorm_momentum: float = 0.1,
        embedding_dropout: float = 0.0,
    ):
        self.base_class = nn.Module
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        r_a, in_features = shape_a
        out_features, r_b = shape_b
        assert (
            r_a == r_b
        ), f"The ranks of A and B should be the same, but got {r_a} for a and {r_b} for b"
        self.r = r_a

        if lora_alpha is None:
            assert (
                lora_scale is not None
            ), "Either lora_alpha or lora_scale should be set"
            lora_alpha = self.r * lora_scale
        self.lora_alpha = lora_alpha

        # Actual trainable parameters
        self.lora_A_module = EmbeddingMLP(
            input_features=emb_input_features,
            output_features=self.r * in_features,
            output_shape=(self.r, in_features),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
            kv_bottleneck=kv_bottleneck,
            num_memory_codebooks=num_memory_codebooks,
            num_memories=num_memories,
            dim_memory=dim_memory,
            additional_embs=additional_embs,
            emb_input=emb_input,
            emb_input_dims=emb_input_dims,
            norm_type=norm_type,
            batchnorm_momentum=batchnorm_momentum,
            dropout=embedding_dropout,
        )
        if conditional_b:
            self.lora_B_module = EmbeddingMLP(
                input_features=emb_input_features,
                output_features=out_features * self.r,
                output_shape=(out_features, self.r),
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_layer_groups=output_layer_groups,
                kv_bottleneck=kv_bottleneck,
                num_memory_codebooks=num_memory_codebooks,
                num_memories=num_memories,
                dim_memory=dim_memory,
                additional_embs=additional_embs,
                emb_input=emb_input,
                emb_input_dims=emb_input_dims,
                norm_type=norm_type,
                batchnorm_momentum=batchnorm_momentum,
                dropout=embedding_dropout,
            )
        else:
            self.lora_B_module = UnconditionalParameters(shape=(out_features, self.r))
        self.scaling = self.lora_alpha / self.r

    def lora_A(self, *args, **kwargs) -> torch.Tensor:
        """Get the A module output."""
        return self.lora_A_module(*args, **kwargs)

    def lora_B(self, *args, **kwargs) -> torch.Tensor:
        """Get the B module output."""
        return self.lora_B_module(*args, **kwargs)

    def _lora_ab(self, *args, **kwargs):
        a = rearrange(self.lora_A(*args, **kwargs), "b r i -> b i r")
        b = rearrange(self.lora_B(*args, **kwargs), "b o r -> b r o")
        ab = einsum(a, b, "b i r, b r o -> b i o") * self.scaling
        return ab

    def reset_lora_parameters(self):
        """Reset the parameters of the LoRA layer."""
        # TODO: Initialize the weights of the A module
        # Original LoRA uses kaiming_uniform initialization
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Our lora_A is an MLP, this line below is used to scale the weights with the example embedding
        # self.lora_A_module.scale_weights(example_embedding)
        # not sure how to initialize the weights here

        # Initialize the weights of the B module to zero
        self.lora_B_module.zero_weights_and_bias()

    def train(self, mode: bool = True):
        """Set the training mode of the LoRA layer."""
        self.base_class.train(self, mode)


class ConditionalLoRALinear(nn.Linear, ConditionalLoRALayer, DoRAMixin):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        emb_input_features: int,
        hidden_dim: int,
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
        lora_rank: int = 1,
        lora_alpha: int = None,
        lora_scale: int = 1,
        lora_dropout: float = 0.0,
        conditional_b=True,
        kv_bottleneck=False,
        num_memory_codebooks=2,
        num_memories=256,
        dim_memory=20,
        additional_embs=1,
        emb_input=False,
        emb_input_dims=None,
        norm_type="batch",
        batchnorm_momentum=0.1,
        embedding_dropout=0.0,
        use_dora=False,
        reset_lora_in_init=True,
        **kwargs,
    ):
        self.base_class = nn.Linear
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ConditionalLoRALayer.__init__(
            self,
            lora_alpha=lora_alpha,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
            emb_input_features=emb_input_features,
            shape_a=torch.Size([lora_rank, in_features]),
            shape_b=torch.Size([out_features, lora_rank]),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
            conditional_b=conditional_b,
            kv_bottleneck=kv_bottleneck,
            num_memory_codebooks=num_memory_codebooks,
            num_memories=num_memories,
            dim_memory=dim_memory,
            additional_embs=additional_embs,
            emb_input=emb_input,
            emb_input_dims=emb_input_dims,
            norm_type=norm_type,
            batchnorm_momentum=batchnorm_momentum,
            embedding_dropout=embedding_dropout,
        )

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Linear.reset_parameters(self)

        if reset_lora_in_init:
            self.reset_lora_parameters()

        self.use_dora = use_dora
        if self.use_dora:
            self._prepare_dora_magnitude()

    def _lora_adaptive_weight(self, *args, **kwargs):
        # (bs, i, o) -> (bs, o, i)
        lora_weight = self._lora_ab(*args, **kwargs).transpose(1, 2)
        # (o, i) -> (1, o, i)
        base_weight = rearrange(self.weight, "o i -> 1 o i")
        weight = lora_weight + base_weight
        return weight  # (bs, o, i)

    def _adaptive_weight(self, *args, **kwargs):
        weight = self._lora_adaptive_weight(*args, **kwargs)
        if self.use_dora:
            weight = self._dora_adaptive_weight(weight, has_bs_dim=True)
        return weight  # (bs, o, i)

    def collapse(self, *args, **kwargs) -> nn.Linear:
        """Collapse the LoRA layer."""
        weight = self._adaptive_weight(*args, **kwargs).squeeze(0)

        # Create a new nn.Linear instance with the LoRA weights
        linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight.data = weight.data.clone()
        if self.bias is not None:
            linear.bias.data = self.bias.data.clone()
        return linear

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the LoRA layer."""
        # weight: (bs, o, i)
        weight = self._adaptive_weight(*args, **kwargs)

        # vectorized linear operation on each item in the batch
        lora_result = einsum(x, weight, "b ... i, b o i -> b ... o")
        if self.bias is not None:
            lora_result += self.bias
        lora_result = self.lora_dropout(lora_result)
        return lora_result

    @classmethod
    def from_nn(
        cls,
        linear_module: nn.Linear,
        **kwargs,
    ) -> "ConditionalLoRALinear":
        """
        Create a LoRALinear instance from an existing nn.Linear module.
        """
        lora_linear = cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=linear_module.bias is not None,
            device=linear_module.weight.device,
            dtype=linear_module.weight.dtype,
            **kwargs,
        )

        # Copy the original weight and bias to the new LoRALinear instance
        lora_linear.weight.data = linear_module.weight.data.clone()
        if linear_module.bias is not None:
            lora_linear.bias.data = linear_module.bias.data.clone()
        return lora_linear


class ConditionalLoRAConv(nn.Module, ConditionalLoRALayer, DoRAMixin):
    def __init__(
        self,
        conv_class,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        emb_input_features: int,
        hidden_dim: int,
        lora_rank=2,
        lora_alpha=None,
        lora_scale=1,
        lora_dropout=0.0,
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
        conditional_b=True,
        kv_bottleneck=False,
        num_memory_codebooks=2,
        num_memories=256,
        dim_memory=20,
        additional_embs=1,
        emb_input=False,
        emb_input_dims=None,
        norm_type="batch",
        batchnorm_momentum=0.1,
        embedding_dropout=0.0,
        use_dora=False,
        reset_lora_in_init=True,
        **kwargs,
    ):
        if issubclass(conv_class, nn.Conv1d):
            conv_type = "1d"
            shape_a, shape_b = self._gen_conv1d_lora_shape(
                in_channels, out_channels, groups, kernel_size, lora_rank
            )
        elif issubclass(conv_class, nn.Conv2d):
            conv_type = "2d"
            shape_a, shape_b = self._gen_conv2d_lora_shape(
                in_channels, out_channels, groups, kernel_size, lora_rank
            )
        elif issubclass(conv_class, nn.Conv3d):
            conv_type = "3d"
            shape_a, shape_b = self._gen_conv3d_lora_shape(
                in_channels, out_channels, groups, kernel_size, lora_rank
            )
        else:
            raise ValueError(f"Unsupported convolution module class {conv_class}")

        self.base_class = conv_class
        self.conv_type = conv_type

        super().__init__()
        self.conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            **kwargs,
        )
        self.weight = self.conv.weight
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        assert isinstance(kernel_size, int), "Only square kernels are supported"

        ConditionalLoRALayer.__init__(
            self,
            shape_a=shape_a,
            shape_b=shape_b,
            lora_alpha=lora_alpha,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
            emb_input_features=emb_input_features,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
            conditional_b=conditional_b,
            kv_bottleneck=kv_bottleneck,
            num_memory_codebooks=num_memory_codebooks,
            num_memories=num_memories,
            dim_memory=dim_memory,
            additional_embs=additional_embs,
            emb_input=emb_input,
            emb_input_dims=emb_input_dims,
            norm_type=norm_type,
            batchnorm_momentum=batchnorm_momentum,
            embedding_dropout=embedding_dropout,
        )

        # Freezing the pre-trained weight matrix
        self.conv.weight.requires_grad = False

        self.conv.reset_parameters()

        if reset_lora_in_init:
            self.reset_lora_parameters()

        self.use_dora = use_dora
        if use_dora:
            self._prepare_dora_magnitude()

    def _gen_conv1d_lora_shape(self, in_channels, out_channels, groups, kernel_size, r):
        shape_a = (r * kernel_size, in_channels // groups * kernel_size)
        shape_b = (out_channels, r * kernel_size)
        return shape_a, shape_b

    def _gen_conv2d_lora_shape(self, in_channels, out_channels, groups, kernel_size, r):
        shape_a = (r * kernel_size, in_channels // groups * kernel_size)
        shape_b = (out_channels * kernel_size, r * kernel_size)
        return shape_a, shape_b

    def _gen_conv3d_lora_shape(self, in_channels, out_channels, groups, kernel_size, r):
        raise NotImplementedError

    def _maybe_to_dora_weights(self, lora_weights):
        if self.use_dora:
            # lora_weights: (b o i k)
            weight = self._dora_adaptive_weight(lora_weights, has_bs_dim=True)
        else:
            weight = lora_weights
        return weight

    def _prepare_conv1d_lora(self, x, *args, **kwargs):
        """
        Get the LoRA added weights for a 1D convolution.

        Returns
        -------
            torch.Tensor: The LoRA added weights with shape (batch_size * out_channels, in_channels, kernel_size).
        """
        lora_weights = self._lora_ab(*args, **kwargs)
        _i = self.in_channels // self.groups
        _k = self.kernel_size
        lora_weights = rearrange(lora_weights, "b (i k) o -> b o i k", i=_i, k=_k)

        base_weights = rearrange(self.conv.weight, "o i k -> 1 o i k")

        weights = base_weights + lora_weights
        weights = self._maybe_to_dora_weights(weights)
        weights = rearrange(weights, "b o i k -> (b o) i k")

        b = x.shape[0]
        x = rearrange(x, "b i l -> 1 (b i) l")

        f_conv = F.conv1d
        revert_x = lambda x: rearrange(x, "1 (b i) l -> b i l", b=b)
        return x, weights, f_conv, revert_x

    def _prepare_conv2d_lora(self, x, *args, **kwargs) -> torch.Tensor:
        """
        Get the LoRA added weights for a 2D convolution.

        Returns
        -------
            torch.Tensor: The LoRA added weights with shape (batch_size * out_channels, in_channels, kernel_size, kernel_size).
        """
        lora_weights = self._lora_ab(*args, **kwargs)
        _i = self.in_channels // self.groups
        _o = self.out_channels
        _k = self.kernel_size
        lora_weights = rearrange(
            lora_weights, "b (i k1) (o k2) -> b o i k1 k2", i=_i, o=_o, k1=_k, k2=_k
        )

        base_weights = rearrange(self.conv.weight, "o i k1 k2 -> 1 o i k1 k2")

        weights = rearrange(base_weights + lora_weights, "b o i k1 k2 -> (b o) i k1 k2")
        weights = self._maybe_to_dora_weights(weights)

        b = x.shape[0]
        x = rearrange(x, "b i h w -> 1 (b i) h w")

        f_conv = F.conv2d
        revert_x = lambda x: rearrange(x, "1 (b i) h w -> b i h w", b=b)
        return x, weights, f_conv, revert_x

    def _prepare_conv3d_lora(self, *args, **kwargs):
        raise NotImplementedError

    def collapse(self, *args, **kwargs) -> nn.modules.conv._ConvNd:
        """Collapse the LoRA layer."""
        new_conv = deepcopy(self.conv)
        new_conv.weight.data = self.conv.weight + self._lora_ab(
            *args, **kwargs
        ).reshape(self.conv.weight.shape)
        return new_conv

    def forward(self, x, *args, **kwargs):
        """Forward pass of the LoRA layer."""
        bs = x.shape[0]

        if self.conv_type == "1d":
            x, lora_weights, f_conv, revert_x = self._prepare_conv1d_lora(
                x, *args, **kwargs
            )
        elif self.conv_type == "2d":
            x, lora_weights, f_conv, revert_x = self._prepare_conv2d_lora(
                x, *args, **kwargs
            )
        elif self.conv_type == "3d":
            x, lora_weights, f_conv, revert_x = self._prepare_conv3d_lora(
                x, *args, **kwargs
            )
        else:
            raise ValueError(f"Unsupported convolution module class {self.conv_type}")

        # X shape (1, batch_size * layer_dim_in, ...)
        # weight shape (batch_size * layer_dim_out, layer_dim_in / groups, kernel_size ...)
        # within each group, the convolution projects from (layer_dim_in, ...) to (layer_dim_out, ...)
        # This way we vectorized the convolution operation when weight contains batch_size dimension
        conv_x = f_conv(
            input=x,
            weight=lora_weights,
            bias=(
                repeat(self.conv.bias, "o -> (bs o)", bs=bs)
                if self.conv.bias is not None
                else None
            ),
            groups=bs * self.groups,  # each batch_size is a group
            dilation=self.conv.dilation,
            padding=self.conv.padding,
            stride=self.conv.stride,
        )
        # return result with shape (batch_size, out_channels, ...)
        conv_x = revert_x(conv_x)

        conv_x = self.lora_dropout(conv_x)
        return conv_x

    @classmethod
    def from_nn(
        cls,
        conv_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        **kwargs,
    ) -> "ConditionalLoRAConv":
        """
        Create a LoRAConvND instance from an existing nn.Conv1d, nn.Conv2d, or nn.Conv3d module.
        """
        assert (
            isinstance(conv_module.kernel_size, int)
            or len(set(conv_module.kernel_size)) == 1
        ), f"Only square kernels are supported, got {conv_module.kernel_size}"
        kernel_size_int = (
            conv_module.kernel_size[0]
            if isinstance(conv_module.kernel_size, tuple)
            else conv_module.kernel_size
        )

        lora_conv = cls(
            conv_class=type(conv_module),
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size_int,
            groups=conv_module.groups,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            bias=conv_module.bias is not None,
            device=conv_module.weight.device,
            dtype=conv_module.weight.dtype,
            **kwargs,
        )

        # Copy the original weight and bias to the new LoRAConv1d instance
        lora_conv.conv.weight.data = conv_module.weight.data.clone()
        if conv_module.bias is not None:
            lora_conv.conv.bias.data = conv_module.bias.data.clone()
        return lora_conv


def convert_to_conditional_lora_model(
    model,
    emb_input_features: int,
    hidden_dim: int = 256,
    hidden_layers: int = 0,
    output_layer_groups: int = 1,
    convert_linear=False,
    convert_conv=False,
    convert_embedding=False,
    lora_rank=1,
    lora_alpha=None,
    lora_scale=1,
    lora_dropout=0.0,
    inplace=False,
    bias_learable=None,
    verbose=False,
    include_name_patterns: list = None,
    exclude_name_patterns: list = None,
    default_conditional: bool = True,
    include_cond_lora_patterns: list = None,
    exclude_cond_lora_patterns: list = None,
    **kwargs,
):
    """
    Replace all pytorch modules (parent class of the lora class)
    in the given model with lora modules.

    Args:
        model (nn.Module): The original PyTorch model.
        emb_input_features (int): The number of input features for the embedding.
        hidden_dim (int): The number of hidden dimensions in the MLP.
        hidden_layers (int): The number of hidden layers in the MLP. Default is 0.
        output_layer_groups (int): The number of groups in the output layer. Default is 1.
        convert_linear (bool): If set to True, nn.Linear modules are replaced.
            Default is False.
        convert_conv (bool): If set to True, nn.Conv1d, nn.Conv2d, and nn.Conv3d
            modules are replaced. Default is False.
        rank (int): The rank for LoRA parameterization.
        alpha (float): The scaling factor for LoRA, default is rank * lora_scale.
        lora_dropout (float): The dropout rate for LoRA input.
        inplace (bool): If set to True, the original model is modified.
            Default is False.
        bias_trainable (str): If set to None, will not make any changes to the bias.
            If set to True, all biases are trainable.
            If set to False, only base model biases are freezed.
            Default is "none".
        verbose (bool): If set to True, print the conversion process.
            Default is False.
        include_name_patterns (list): A list of patterns to include in the LoRA conversion.
            Default is None.
        exclude_name_patterns (list): A list of patterns to exclude from the LoRA conversion.
            Default is None.
        default_conditional (bool): Whether use conditional LoRA conversion for names not matching
            include_cond_lora_patterns OR exclude_cond_lora_patterns.
            Default is True.
        include_cond_lora_patterns (list): A list of patterns to include in the conditional LoRA conversion.
            Default is ('.+',), which means all LoRA layers will be conditional.
        exclude_cond_lora_patterns (list): A list of patterns to exclude from the conditional LoRA conversion.
            Default is None.
        kwargs: Additional keyword arguments to pass to the LoRA modules' constructors.

    Returns
    -------
        nn.Module: The modified model with lora layers.
    """
    if not inplace:
        model = deepcopy(model)

    if exclude_name_patterns is None:
        exclude_name_patterns = []
    if include_name_patterns is None:
        include_name_patterns = []
    if exclude_cond_lora_patterns is None:
        exclude_cond_lora_patterns = []
    if include_cond_lora_patterns is None:
        include_cond_lora_patterns = []

    # Create a list of modules to modify
    modules_to_modify = []
    for name, module in model.named_modules():
        if name_in_patterns(name, exclude_name_patterns):
            continue

        if len(include_name_patterns) > 0:
            if not name_in_patterns(name, include_name_patterns):
                continue

        conditional = default_conditional
        if name_in_patterns(name, include_cond_lora_patterns):
            conditional = True
        if name_in_patterns(name, exclude_cond_lora_patterns):
            conditional = False

        if isinstance(module, nn.Linear) and convert_linear:
            lora_cls = ConditionalLoRALinear if conditional else LoRALinear
            modules_to_modify.append((name, module, lora_cls))
        elif (
            isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, Conv1dWrapper))
            and convert_conv
        ):
            lora_cls = ConditionalLoRAConv if conditional else LoRAConv
            modules_to_modify.append((name, module, lora_cls))
        elif isinstance(module, nn.Embedding) and convert_embedding:
            if conditional:
                raise NotImplementedError(
                    "Conditional LoRA for nn.Embedding is not implemented yet."
                )
            lora_cls = LoRAEmbedding
            modules_to_modify.append((name, module, lora_cls))
        else:
            pass

    if lora_alpha is not None:
        lora_scale = None

    # Update the model with the modified modules
    for name, module, lora_cls in modules_to_modify:
        if issubclass(lora_cls, ConditionalLoRALayer):
            lora_module = lora_cls.from_nn(
                module,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
                emb_input_features=emb_input_features,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_layer_groups=output_layer_groups,
                **kwargs,
            )
        else:
            lora_module = lora_cls.from_nn(
                module,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_scale=lora_scale,
                lora_dropout=lora_dropout,
            )
        if verbose:
            print(
                f"Converting '{name}' <{type(module).__name__}> to <{lora_cls.__name__}> module"
            )
        set_submodule_by_name(model, name, lora_module)

    mark_only_lora_as_trainable(model, bias_learable=bias_learable)
    return model


def collapse_lora_model_(model, *args, **kwargs):
    """
    Collapse all LoRA layers in the given model and return a new model with collapsed non-LoRA layers.

    Args:
        model (nn.Module): The model with LoRA layers.

    Returns
    -------
        nn.Module: The model with collapsed LoRA layers.
    """
    model = deepcopy(model)
    for name, module in model.named_modules():
        if hasattr(module, "collapse"):
            set_submodule_by_name(model, name, module.collapse(*args, **kwargs))
    return model
