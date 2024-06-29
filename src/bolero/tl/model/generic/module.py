import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from bolero.tl.model.generic.train_helper import validate_config


class GenericModule(nn.Module):
    """Generic Module for all the models.

    Attributes
    ----------
    default_config : dict
        Default configuration for the module.

    Methods
    -------
    get_default_config() -> dict
        Get the default configuration for the module.
    create_from_config(config: dict) -> GenericModule
        Create an instance of the module from a configuration.
    forward(*args, **kwargs) -> None
        Forward pass of the module.
    """

    default_config: dict = {}

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Get the default configuration for the module.

        Returns
        -------
        dict
            The default configuration for the module.
        """
        return cls.default_config

    @classmethod
    def create_from_config(cls, config: dict) -> "GenericModule":
        """
        Create an instance of the module from a configuration.

        Parameters
        ----------
        config : dict
            The configuration for the module.

        Returns
        -------
        GenericModule
            An instance of the module.
        """
        # remove additional keys in the configuration
        config = {k: v for k, v in config.items() if k in cls.default_config}
        validate_config(config, cls.default_config)
        return cls(**config)

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> None:
        """
        Forward pass of the module.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        None
        """
        raise NotImplementedError


class GroupedLinear(nn.Module):
    """Use nn.Conv1D with kernel_size=1 to simulate nn.Linear,
    and utilize its groups parameter to simulate grouped linear layer.

    The input and output dimensions are the same as nn.Linear.
    """

    def __init__(self, in_features, out_features, groups=1, bias=True):
        super().__init__()

        self.internal_out_features = int(math.ceil(out_features / groups) * groups)
        self.out_feathers = out_features
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=self.internal_out_features,
            kernel_size=1,
            groups=groups,
            bias=bias,
        )
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x):
        """Forward pass of the GroupedLinear module."""
        x.unsqueeze_(2)  # add the length dimension
        x = self.conv(x)
        x.squeeze_(2)  # remove the length dimension
        if self.internal_out_features != self.out_feathers:
            x = x[:, : self.out_feathers]
        return x


class Conv1dWrapper(nn.Module):
    """Conv1d Layer Wrapper that support modes."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, X, *args, modes=None, **kwargs):
        """Forward pass of the Conv1dWrapper module."""
        # The args, and kwargs are just placeholders
        # Use modes to select a subset of the weights
        return (
            self.conv(X)
            if modes is None
            else F.conv1d(
                X,
                self.conv.weight[modes],
                self.conv.bias[modes],
                padding=self.conv.padding[0],
            )
        )


def _get_activation(activation):
    if not isinstance(activation, str):
        return activation

    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


class DNA_CNN(GenericModule):
    """
    This class represents a DNA Convolutional Neural Network (CNN) module.
    It is used to extract DNA sequence features using a single CNN layer.

    Parameters
    ----------
    n_filters : int
        Number of filters in the CNN layer.
    kernel_size : int
        Size of the kernel in the CNN layer.
    activation : nn.Module
        Activation function to be applied after the convolution operation.
    in_channels : int
        Number of input channels.

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    n_filters : int
        Number of filters in the CNN layer.
    kernel_size : int
        Size of the kernel in the CNN layer.
    conv : nn.Conv1d
        Convolutional layer.
    activation : nn.Module
        Activation function.

    Methods
    -------
    forward(X, *args, **kwargs)
        Forward pass of the CNN module.
    """

    default_config = {
        "n_filters": 1024,
        "dna_kernel_size": 21,
        "activation": "gelu",
        "in_channels": 4,
    }

    def __init__(
        self,
        n_filters: int = 1024,
        dna_kernel_size: int = 21,
        activation: nn.Module = nn.GELU(),
        in_channels: int = 4,
    ):
        """
        Initialize the DNA_CNN module.

        Parameters
        ----------
        n_filters : int, optional
            Number of filters in the CNN layer. Default is 1024.
        kernel_size : int, optional
            Size of the kernel in the CNN layer. Default is 21.
        activation : nn.Module, optional
            Activation function to be applied after the convolution operation. Default is nn.GELU().
        in_channels : int, optional
            Number of input channels. Default is 4.

        """
        activation = _get_activation(activation)

        super().__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters
        self.dna_kernel_size = dna_kernel_size

        self.conv = Conv1dWrapper(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=dna_kernel_size,
            padding=dna_kernel_size // 2,
        )
        self.activation = copy.deepcopy(activation)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Perform a forward pass of the DNA_CNN module.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_filters, sequence_length).

        """
        X = self.conv(X, *args, **kwargs)
        X = self.activation(X)
        return X


class Residual(nn.Module):
    """
    This class represents a residual module that adds the input tensor to the output of another module.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Perform a forward pass of the Residual module.
        """
        return x + self.module(x, *args, **kwargs)


def _default_dilation_func(x):
    return 2 ** (x + 1)


class DilatedCNN(GenericModule):
    """
    This part only takes into account the Dilated CNN stack
    """

    default_config = {
        "n_filters": 1024,
        "bottleneck": 1024,
        "n_blocks": 8,
        "dia_kernel_size": 3,
        "groups": 8,
        "activation": "gelu",
        "batch_norm": True,
        "batch_norm_momentum": 0.1,
        "dilation_func": None,
        "bipass_connect": False,
    }

    def __init__(
        self,
        n_filters=1024,
        bottleneck=1024,
        n_blocks=8,
        dia_kernel_size=3,
        groups=8,
        activation=nn.GELU(),
        batch_norm=True,
        batch_norm_momentum=0.1,
        dilation_func=None,
        bipass_connect=False,
    ):
        activation = _get_activation(activation)
        super().__init__()
        if dilation_func is None:
            dilation_func = _default_dilation_func
        self.dilation_func = dilation_func

        self.n_filters = n_filters
        self.botleneck = bottleneck
        self.n_blocks = n_blocks
        self.dia_kernel_size = dia_kernel_size
        self.groups = groups
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.bipass_connect = bipass_connect
        self.layers = nn.ModuleList(
            [
                Residual(
                    ConvBlockModule(
                        n_filters=n_filters,
                        bottleneck=bottleneck,
                        kernel_size=dia_kernel_size,
                        dilation=self.dilation_func(i),
                        activation=activation,
                        batch_norm=batch_norm,
                        batch_norm_momentum=batch_norm_momentum,
                        groups=groups,
                    )
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, X, *args, **kwargs):
        """
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)
        """
        if self.bipass_connect:
            X0 = X.clone()
            for i in range(self.n_blocks):
                X = self.layers[i](X, *args, **kwargs)
            X += X0
        else:
            for i in range(self.n_blocks):
                X = self.layers[i](X, *args, **kwargs)
        return X


class ConvBlockModule(nn.Module):
    def __init__(
        self,
        n_filters=1024,
        bottleneck=1024,
        kernel_size=3,
        dilation=1,
        activation=nn.GELU(),
        batch_norm=True,
        batch_norm_momentum=0.1,
        groups=8,
        inception_version=2,
    ):
        """
        Parameters
        ----------
        n_filters: int
            number of kernels
        bottleneck: int
            number of kernels in the bottleneck layer
        kernel_size: int
            kernel size
        dilation: int
            dilation rate
        activation: nn.Module
            activation function
        batch_norm: bool
            batch normalization in between layers
        batch_norm_momentum: float
            batch normalization momentum
        groups: int
            number of groups in the conv layer
        """
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.batch_norm = batch_norm
        self.inception_version = inception_version
        self.bottleneck = bottleneck

        self.conv1 = Conv1dWrapper(
            in_channels=n_filters,
            out_channels=bottleneck,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size // 2),
            groups=groups,
            bias=False,
        )
        self.block1 = nn.Sequential(
            (
                nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum)
                if batch_norm
                else nn.Identity()
            ),
            copy.deepcopy(activation),
        )
        self.conv2 = Conv1dWrapper(
            in_channels=bottleneck,
            out_channels=n_filters,
            kernel_size=1,
            bias=False,
        )
        self.block2 = nn.Sequential(
            (
                nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum)
                if batch_norm
                else nn.Identity()
            ),
            copy.deepcopy(activation),
        )

        nn.init.kaiming_normal_(self.conv1.weight.data, mode="fan_out")
        nn.init.kaiming_normal_(self.conv2.weight.data, mode="fan_out")

        self.block1[0].weight.data[...] = 1
        self.block2[0].weight.data[...] = 1

        self.block1[0].bias.data[...] = 0
        self.block2[0].bias.data[...] = 0

    def forward(self, X, *args, **kwargs):
        """
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)
        """
        X = self.conv1(X, *args, **kwargs)
        X = self.block1(X)
        X = self.conv2(X, *args, **kwargs)
        X = self.block2(X)
        return X
