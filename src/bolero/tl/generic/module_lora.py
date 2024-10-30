"""
Code adapted from original LoRA repo: https://github.com/microsoft/LoRA/tree/main

LoRA paper: https://arxiv.org/abs/2106.09685
Example LoRA implementation https://github.com/hkproj/pytorch-lora
And Learn this great youtube video: https://www.youtube.com/watch?v=PXWYUTMt-AU&t=1367s
Also video from the LoRA author: https://www.youtube.com/watch?v=DhRoTONcyZE
"""

import math
import re
from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F


class DoRAMixin:
    """
    Mixin class to add DoRA to LoRA layers.

    Weight-Decomposed Low-Rank Adaptation (DoRA) is an potential improvment to LoRA.
    https://github.com/NVlabs/DoRA

    This class should work for both linear and conv layers.
    It should also work for both conditional and unconditional LoRA layers.
    When weight comes in with bs dimension, set has_bs_dim to True.
    """

    def _prepare_dora_magnitude(self):
        # record the base magnitude of the output dim of the weight matrix
        self.base_m = nn.Parameter(
            self.weight.norm(p=2, dim=0, keepdim=False), requires_grad=False
        )  # (i, ...)
        # learnable adaptive magnitude, initialized to 0
        self.dora_m = nn.Parameter(torch.zeros_like(self.base_m))  # (i, ...)

    def _dora_adaptive_weight(self, lora_adaptive_weight, has_bs_dim=False):
        """
        lora_adaptive_weight: (bs, o, i, ...) or (o, i, ...)
        """
        output_dim = 1 if has_bs_dim else 0
        output_norm = lora_adaptive_weight.norm(
            p=2, dim=output_dim, keepdim=True
        )  # (bs, 1, i, ...) or (1, i, ...)
        normed_weight = (
            lora_adaptive_weight / output_norm
        )  # (bs, o, i, ...) or (o, i, ...)

        dora_weight = normed_weight * (
            self.base_m + self.dora_m
        )  # (bs, o, i, ...) or (o, i, ...)
        return dora_weight


class LoRALayer:
    def __init__(
        self,
        lora_rank: int,
        lora_alpha: int,
        lora_scale: int,
        lora_dropout: float,
    ):
        self.r = lora_rank

        if lora_alpha is None:
            assert lora_scale is not None, "Either lora_alpha or lora_scale must be set"
            lora_alpha = lora_rank * lora_scale
        self.lora_alpha = lora_alpha

        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_A: nn.Parameter
        self.lora_B: nn.Parameter


class LoRAEmbedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        lora_rank: int = 1,
        lora_alpha: int = None,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0,
        )
        # Actual trainable parameters
        if lora_rank > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((lora_rank, num_embeddings))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((embedding_dim, lora_rank))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the LoRA layer."""
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def collapse(self, *args, **kwargs) -> nn.Embedding:
        """Collapse the LoRA layer into a regular nn.Embedding layer."""
        weight = (
            self.weight.data
            + (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
        )
        new_module = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            freeze=False,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        new_module.weight.data = weight
        return new_module

    def forward(self, x: torch.Tensor):
        """Forward pass of the LoRA layer."""
        result = nn.Embedding.forward(self, x)
        after_A = F.embedding(
            x,
            self.lora_A.transpose(0, 1),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling


class LoRALinear(nn.Linear, LoRALayer, DoRAMixin):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: int = None,
        lora_scale: int = 1,
        lora_dropout: float = 0.0,
        use_dora: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
        )

        # Actual trainable parameters
        self.lora_A = nn.Parameter(self.weight.new_zeros((lora_rank, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, lora_rank)))
        self.scaling = self.lora_alpha / self.r

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()

        self.use_dora = use_dora
        if use_dora:
            self._prepare_dora_magnitude()

    def reset_parameters(self):
        """Reset the parameters of the LoRA layer."""
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _lora_ab(self):
        return self.lora_B @ self.lora_A * self.scaling  # (o, i)

    def _adaptive_weight(self):
        lora_weight = self.weight + self._lora_ab()
        if self.use_dora:
            lora_weight = self._dora_adaptive_weight(lora_weight, has_bs_dim=False)
        return lora_weight

    def collapse(self, *args, **kwargs) -> nn.Linear:
        """Collapse the LoRA layer into a regular nn.Linear layer."""
        new_module = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        weight = self._adaptive_weight()
        new_module.weight.data = weight
        if self.bias is not None:
            new_module.bias.data = self.bias.data
        return new_module

    def forward(self, x: torch.Tensor):
        """Forward pass of the LoRA layer."""
        weight = self._adaptive_weight()
        result = F.linear(x, weight, bias=self.bias)
        result = self.lora_dropout(result)
        return result

    @classmethod
    def from_nn(
        cls,
        linear_module: nn.Linear,
        lora_rank: int = 1,
        lora_alpha: float = None,
        lora_scale: int = 1,
        lora_dropout: float = 0.0,
    ) -> "LoRALinear":
        """
        Create a LoRALinear instance from an existing nn.Linear module.
        """
        lora_linear = cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
            bias=linear_module.bias is not None,
            device=linear_module.weight.device,
            dtype=linear_module.weight.dtype,
        )

        # Copy the original weight and bias to the new LoRALinear instance
        lora_linear.weight.data = linear_module.weight.data.clone()
        if linear_module.bias is not None:
            lora_linear.bias.data = linear_module.bias.data.clone()
        return lora_linear


class LoRAConv(nn.Module, LoRALayer, DoRAMixin):
    def __init__(
        self,
        conv_class,
        in_channels,
        out_channels,
        kernel_size,
        lora_rank,
        lora_alpha=None,
        lora_scale=1,
        lora_dropout=0.0,
        use_dora=False,
        **kwargs,
    ):
        if conv_class == nn.Conv1d:
            conv_type = "1d"
        elif conv_class == nn.Conv2d:
            conv_type = "2d"
        elif conv_class == nn.Conv3d:
            conv_type = "3d"
        else:
            raise ValueError(f"Unsupported convolution module class {conv_type}")

        super().__init__()
        self.conv: nn.modules.conv._ConvNd = conv_class(
            in_channels, out_channels, kernel_size, **kwargs
        )
        self.weight = self.conv.weight

        LoRALayer.__init__(
            self,
            lora_rank=lora_rank * kernel_size,
            lora_alpha=lora_alpha,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
        )
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if conv_type == "1d":
            self.lora_A, self.lora_B = self._gen_conv_1d_lora_parameter(
                in_channels, out_channels, self.conv.groups, kernel_size, lora_rank
            )
        elif conv_type == "2d":
            self.lora_A, self.lora_B = self._gen_conv_2d_lora_parameter(
                in_channels, out_channels, self.conv.groups, kernel_size, lora_rank
            )
        elif conv_type == "3d":
            self.lora_A, self.lora_B = self._gen_conv_3d_lora_parameter(
                in_channels, out_channels, self.conv.groups, kernel_size, lora_rank
            )
        else:
            raise ValueError(f"Unsupported convolution type {conv_type}")

        self.scaling = self.lora_alpha / self.r
        # Freezing the pre-trained weight matrix
        self.conv.weight.requires_grad = False
        self.reset_parameters()

        self.use_dora = use_dora
        if use_dora:
            self._prepare_dora_magnitude()

    def _gen_conv_1d_lora_parameter(
        self, in_channels, out_channels, groups, kernel_size, r
    ):
        lora_a = nn.Parameter(
            self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
        )
        lora_b = nn.Parameter(
            self.conv.weight.new_zeros((out_channels // groups, r * kernel_size))
        )
        return lora_a, lora_b

    def _gen_conv_2d_lora_parameter(
        self, in_channels, out_channels, groups, kernel_size, r
    ):
        lora_a = nn.Parameter(
            self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
        )
        lora_b = nn.Parameter(
            self.conv.weight.new_zeros(
                (out_channels // groups * kernel_size, r * kernel_size)
            )
        )
        return lora_a, lora_b

    def _gen_conv_3d_lora_parameter(
        self, in_channels, out_channels, groups, kernel_size, r
    ):
        raise NotImplementedError

    def reset_parameters(self):
        """Reset the parameters of the LoRA layer."""
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _lora_ab(self):
        lora_ab = (self.lora_B @ self.lora_A).view(
            self.conv.weight.shape
        ) * self.scaling
        return lora_ab  # (o, i, k...)

    def _adaptive_weight(self):
        lora_weight = self.conv.weight + self._lora_ab()
        if self.use_dora:
            lora_weight = self._dora_adaptive_weight(lora_weight, has_bs_dim=False)
        return lora_weight

    def collapse(self, *args, **kwargs) -> nn.modules.conv._ConvNd:
        """Collapse the LoRA layer into a regular nn.ConvNd layer."""
        weight = self._adaptive_weight()
        new_module = deepcopy(self.conv)
        new_module.weight.data = weight
        if self.conv.bias is not None:
            new_module.bias.data = self.conv.bias.data
        return new_module

    def forward(self, x):
        """Forward pass of the LoRA layer."""
        result = self.conv._conv_forward(
            x, weight=self._adaptive_weight(), bias=self.conv.bias
        )
        result = self.lora_dropout(result)
        return result

    @classmethod
    def from_nn(
        cls,
        conv_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        lora_rank: int = 1,
        lora_alpha: float = None,
        lora_scale: int = 1,
        lora_dropout: float = 0.0,
    ) -> "LoRAConv":
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
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=conv_module.bias is not None,
            device=conv_module.weight.device,
            dtype=conv_module.weight.dtype,
        )

        # Copy the original weight and bias to the new LoRAConv1d instance
        lora_conv.conv.weight.data = conv_module.weight.data.clone()
        if conv_module.bias is not None:
            lora_conv.conv.bias.data = conv_module.bias.data.clone()
        return lora_conv


def set_submodule_by_name(
    model: nn.Module, module_name: str, new_module: nn.Module
) -> None:
    """
    Function to replace any submodule of a model with a new module.

    Parameters
    ----------
        model (Any): The parent module.
        module_name (str): The full name of the submodule to be replaced.
        new_module (Any): The new submodule to be set.

    Returns
    -------
        None
    """
    # Split the name by dots to handle nested structures
    name_parts = module_name.split(".")

    # Traverse to the parent module
    submodule = model
    for part in name_parts[:-1]:
        submodule = getattr(submodule, part)

    # Set the new module in the parent
    setattr(submodule, name_parts[-1], new_module)
    return


def mark_only_lora_as_trainable(model: nn.Module, bias_learable=None) -> None:
    """Set the trainable status of the LoRA layers in the model."""
    for n, p in model.named_parameters():
        if "lora_" not in n:
            if "bias" not in n:
                p.requires_grad = False
            else:
                if bias_learable is not None:
                    p.requires_grad = bias_learable
                # otherwise, do not make any changes to the bias


def convert_to_lora_model(
    model,
    convert_linear=False,
    convert_conv=False,
    lora_rank=1,
    lora_alpha=None,
    lora_scale=1,
    inplace=False,
    bias_trainable="none",
):
    """
    Replace all pytorch modules (parent class of the lora class)
    in the given model with lora modules.

    Args:
        model (nn.Module): The original PyTorch model.
        convert_linear (bool): If set to True, nn.Linear modules are replaced.
            Default is False.
        convert_conv (bool): If set to True, nn.Conv1d, nn.Conv2d, and nn.Conv3d
            modules are replaced. Default is False.
        rank (int): The rank for LoRA parameterization.
        alpha (float): The scaling factor for LoRA.
        inplace (bool): If set to True, the original model is modified.
            Default is False.
        bias_trainable (str): If set to "none", will not make any changes to the bias.
            If set to "all", all biases are trainable.
            If set to "lora_only", only LoRA biases are trainable.
            Default is "none".

    Returns
    -------
        nn.Module: The modified model with lora layers.
    """
    if not inplace:
        model = deepcopy(model)

    # Create a list of modules to modify
    modules_to_modify = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and convert_linear:
            modules_to_modify.append((name, module, LoRALinear))
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and convert_conv:
            modules_to_modify.append((name, module, LoRAConv))
        else:
            pass

    # Update the model with the modified modules
    for name, module, lora_cls in modules_to_modify:
        lora_module = lora_cls.from_nn(
            module, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_scale=lora_scale
        )
        set_submodule_by_name(model, name, lora_module)

    mark_only_lora_as_trainable(model, bias=bias_trainable)
    return model


def name_in_patterns(name, exclude_patterns):
    """Check if the name matches any of the exclude patterns."""
    for p in exclude_patterns:
        if re.search(p, name):
            return True
    return False
