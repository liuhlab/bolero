"""
Code adapted from original LoRA repo: https://github.com/microsoft/LoRA/tree/main

LoRA paper: https://arxiv.org/abs/2106.09685
Example LoRA implementation https://github.com/hkproj/pytorch-lora
And Learn this great youtube video: https://www.youtube.com/watch?v=PXWYUTMt-AU&t=1367s
Also video from the LoRA author: https://www.youtube.com/watch?v=DhRoTONcyZE
"""

from copy import deepcopy
from functools import partial
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from torch.nn import functional as F

from bolero.tl.generic.module import Conv1dWrapper, GroupedLinear

from .module_lora import (
    LoRAConv,
    LoRALinear,
    mark_only_lora_as_trainable,
    name_in_patterns,
    set_submodule_by_name,
)


class EmbeddingMLP(nn.Module):
    """
    This class turn the input embedding into one of the LoRA low-rank weight matrix (A or B) through a simple MLP.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        output_shape: torch.Size,
        hidden_dim: int,
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
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
        self.hidden_dim = self.input_features if hidden_dim is None else hidden_dim
        self.out_feathres = output_features
        self.output_shape = output_shape

        if output_layer_groups > 1:
            if self.hidden_dim > 8 and self.out_feathres > 8:
                # Grouped Linear has smaller number of parameters
                output_module = partial(GroupedLinear, groups=output_layer_groups)
            else:
                output_module = nn.Linear
        else:
            output_module = nn.Linear

        def _generate_linear_module(in_features, out_features):
            layers = [
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.BatchNorm1d(out_features),  # TODO: maybe try LayerNorm
                nn.GELU(),
            ]
            return layers

        layers = _generate_linear_module(self.input_features, self.hidden_dim)
        for _ in range(hidden_layers):
            layers += _generate_linear_module(self.hidden_dim, self.hidden_dim)
        layers.append(
            output_module(in_features=self.hidden_dim, out_features=self.out_feathres)
        )

        self.mlp = nn.Sequential(*layers)
        self.rescale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(
        self, embedding: torch.Tensor, emb_weights: torch.Tensor = None
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
        ndim = embedding.ndim
        bs = embedding.shape[0]
        if ndim == 3:
            # expect input shape (bs, seq_len, emb_dim)
            embedding = rearrange(embedding, "bs l d -> (bs l) d")

        x = self.mlp(embedding * self.rescale_factor)

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
        Zero the weights and bias of the MLP's first layer, use this in B embedding.
        """
        for i in range(len(self.mlp)):
            if isinstance(self.mlp[i], (nn.GELU, nn.BatchNorm1d)):
                continue

            if isinstance(self.mlp[i], (nn.Linear, GroupedLinear)):
                self.mlp[i].bias.data[...] = 0
                self.mlp[i].weight.data[...] = 0
            else:
                print("Skip zero weights and bias for layer", i, type(self.mlp[i]))
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


class ConditionalLoRALayer:
    def __init__(
        self,
        shape_a: torch.Size,
        shape_b: torch.Size,
        lora_alpha: int,
        lora_dropout: float,
        emb_input_features: int,
        hidden_dim: int,
        hidden_layers: int,
        output_layer_groups: int,
    ):
        self.base_class = nn.Module
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        r_a, in_features = shape_a
        out_features, r_b = shape_b
        assert (
            r_a == r_b
        ), f"The ranks of A and B should be the same, but got {r_a} for a and {r_b} for b"
        self.r = r_a

        # Actual trainable parameters
        self.lora_A_module = EmbeddingMLP(
            input_features=emb_input_features,
            output_features=self.r * in_features,
            output_shape=(self.r, in_features),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
        )
        self.lora_B_module = EmbeddingMLP(
            input_features=emb_input_features,
            output_features=out_features * self.r,
            output_shape=(out_features, self.r),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
        )
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
        ab = einsum(a, b, "b i r, b r o -> b i o")
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


class ConditionalLoRALinear(nn.Linear, ConditionalLoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        emb_input_features: int,
        hidden_dim: int,
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        self.base_class = nn.Linear
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        ConditionalLoRALayer.__init__(
            self,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            emb_input_features=emb_input_features,
            shape_a=torch.Size([r, in_features]),
            shape_b=torch.Size([out_features, r]),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
        )

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Linear.reset_parameters(self)
        self.reset_lora_parameters()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the LoRA layer."""
        result = F.linear(x, self.weight, bias=self.bias)

        lora_result = einsum(
            self.lora_dropout(x),
            self._lora_ab(*args, **kwargs),
            "b ... i, b i o -> b ... o",
        )
        result += lora_result * self.scaling
        return result

    @classmethod
    def from_nn(
        cls,
        linear_module: nn.Linear,
        emb_input_features: int,
        hidden_dim: int,
        rank: int = 1,
        alpha: float = 1,
        lora_dropout: float = 0.0,
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
        **kwargs,
    ) -> "ConditionalLoRALinear":
        """
        Create a LoRALinear instance from an existing nn.Linear module.
        """
        lora_linear = cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            emb_input_features=emb_input_features,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            output_layer_groups=output_layer_groups,
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


class ConditionalLoRAConv(nn.Module, ConditionalLoRALayer):
    def __init__(
        self,
        conv_class,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        emb_input_features: int,
        hidden_dim: int,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
        **kwargs,
    ):
        if conv_class == nn.Conv1d:
            conv_type = "1d"
            shape_a, shape_b = self._gen_conv1d_lora_shape(
                in_channels, out_channels, groups, kernel_size, r
            )
        elif conv_class == nn.Conv2d:
            conv_type = "2d"
            shape_a, shape_b = self._gen_conv2d_lora_shape(
                in_channels, out_channels, groups, kernel_size, r
            )
        elif conv_class == nn.Conv3d:
            conv_type = "3d"
            shape_a, shape_b = self._gen_conv3d_lora_shape(
                in_channels, out_channels, groups, kernel_size, r
            )
        else:
            raise ValueError(f"Unsupported convolution module class {conv_type}")

        self.base_class = conv_class
        self.conv_type = conv_type

        super().__init__()
        self.conv = conv_class(in_channels, out_channels, kernel_size, **kwargs)
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
            lora_dropout=lora_dropout,
            emb_input_features=emb_input_features,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
        )

        # Freezing the pre-trained weight matrix
        self.conv.weight.requires_grad = False

        self.conv.reset_parameters()
        self.reset_lora_parameters()

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
        lora_weights = (
            rearrange(lora_weights, "b (i k) o -> b o i k", i=_i, k=_k) * self.scaling
        )

        base_weights = rearrange(self.conv.weight, "o i k -> 1 o i k")

        weights = rearrange(base_weights + lora_weights, "b o i k -> (b o) i k")

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
        lora_weights = (
            rearrange(
                lora_weights, "b (i k1) (o k2) -> b o i k1 k2", i=_i, o=_o, k1=_k, k2=_k
            )
            * self.scaling
        )

        base_weights = rearrange(self.conv.weight, "o i k1 k2 -> 1 o i k1 k2")

        weights = rearrange(base_weights + lora_weights, "b o i k1 k2 -> (b o) i k1 k2")

        b = x.shape[0]
        x = rearrange(x, "b i h w -> 1 (b i) h w")

        f_conv = F.conv2d
        revert_x = lambda x: rearrange(x, "1 (b i) h w -> b i h w", b=b)
        return x, weights, f_conv, revert_x

    def _prepare_conv3d_lora(self, *args, **kwargs):
        raise NotImplementedError

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
        return conv_x

    @classmethod
    def from_nn(
        cls,
        conv_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        emb_input_features: int,
        hidden_dim: int,
        rank: int = 1,
        alpha: float = 1,
        lora_dropout: float = 0.0,
        hidden_layers: int = 0,
        output_layer_groups: int = 1,
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
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            emb_input_features=emb_input_features,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            bias=conv_module.bias is not None,
            device=conv_module.weight.device,
            dtype=conv_module.weight.dtype,
        )

        # Copy the original weight and bias to the new LoRAConv1d instance
        lora_conv.conv.weight.data = conv_module.weight.data.clone()
        if conv_module.bias is not None:
            lora_conv.conv.bias.data = conv_module.bias.data.clone()
        return lora_conv


def convert_to_conditional_lora_model(
    model,
    emb_input_features: int,
    hidden_dim: int,
    hidden_layers: int = 0,
    output_layer_groups: int = 1,
    convert_linear=False,
    convert_conv=False,
    rank=1,
    alpha=1,
    lora_dropout=0.0,
    inplace=False,
    bias_trainable="none",
    verbose=False,
    include_name_patterns: list = None,
    exclude_name_patterns: list = None,
    default_conditional: bool = True,
    include_cond_lora_patterns: list = None,
    exclude_cond_lora_patterns: list = None,
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
        alpha (float): The scaling factor for LoRA.
        lora_dropout (float): The dropout rate for LoRA input.
        inplace (bool): If set to True, the original model is modified.
            Default is False.
        bias_trainable (str): If set to "none", will not make any changes to the bias.
            If set to "all", all biases are trainable.
            If set to "lora_only", only LoRA biases are trainable.
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
        else:
            pass

    # Update the model with the modified modules
    for name, module, lora_cls in modules_to_modify:
        if issubclass(lora_cls, ConditionalLoRALayer):
            lora_module = lora_cls.from_nn(
                module,
                rank=rank,
                alpha=alpha,
                lora_dropout=lora_dropout,
                emb_input_features=emb_input_features,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_layer_groups=output_layer_groups,
            )
        else:
            lora_module = lora_cls.from_nn(
                module,
                rank=rank,
                alpha=alpha,
                lora_dropout=lora_dropout,
            )
        if verbose:
            print(
                f"Converting '{name}' <{type(module).__name__}> to <{lora_cls.__name__}> module"
            )
        set_submodule_by_name(model, name, lora_module)

    mark_only_lora_as_trainable(model, bias=bias_trainable)
    return model
