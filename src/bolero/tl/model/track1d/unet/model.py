import torch
import torch.nn as nn

from .module import Conv1DBlock, Down, TransformerStack, Up


class UNetTrans(nn.Module):
    """
    UNetTrans is a U-Net model with Transformer layers for 1D sequence data.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        n_filters (int): Number of filters in the first layer.
        num_heads (int): Number of attention heads in the Transformer layers.
        ff_hidden_dim (int): Hidden dimension of the feed-forward network in the Transformer layers.
        num_layers (int): Number of layers in the Transformer stack.
        dropout (float): Dropout rate.
        output_kernel_size (int): Kernel size of the output convolutional layer.
        output_padding (int): Padding size of the output convolutional layer.
        dna_len (int): Length of the input DNA sequence.
        output_len (int): Length of the output sequence.

    Attributes
    ----------
        inc (Conv1DBlock): First convolutional block.
        down1 (Down): Down-sampling block 1.
        down2 (Down): Down-sampling block 2.
        transformer (TransformerStack): Transformer stack.
        up1 (Up): Up-sampling block 1.
        up2 (Up): Up-sampling block 2.
        conv (nn.Conv1d): Convolutional layer.
        out_conv (nn.Conv1d): Output convolutional layer.
        output_slice (slice): Slice object for selecting the output sequence.

    """

    default_config = {
        "n_filters": 256,
        "input_channels": 4,
        "output_channels": 1,
        "num_heads": 4,
        "ff_hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.1,
        "output_kernel_size": 101,
        "output_padding": 50,
        "dna_len": 2500,
        "output_len": 1000,
    }

    @classmethod
    def get_default_config(cls):
        """
        Get the default configuration for the UNetTrans model.

        Returns
        -------
            dict: Default configuration.

        """
        return cls.default_config

    @classmethod
    def create_from_config(cls, config: dict):
        """
        Create a UNetTrans model from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns
        -------
            UNetTrans: Initialized UNetTrans model.

        """
        _config = cls.default_config.copy()
        _config.update(config)
        _config = {k: v for k, v in _config.items() if k in cls.default_config}
        return cls(**_config)

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        n_filters: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_kernel_size: int,
        output_padding: int,
        dna_len: int,
        output_len: int,
    ):
        super().__init__()
        self.in_channels = input_channels
        self.output_channels = output_channels
        self.base_filters = n_filters

        self.dna_len = dna_len
        self.output_len = output_len
        start = (self.dna_len - self.output_len) // 2
        end = start + self.output_len
        self.output_slice = slice(start, end)

        self.inc = Conv1DBlock(self.in_channels, self.base_filters)
        self.down1 = Down(self.base_filters, self.base_filters * 2)
        self.down2 = Down(self.base_filters * 2, self.base_filters * 4)

        self.transformer = TransformerStack(
            embed_size=self.base_filters * 4,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.up1 = Up(self.base_filters * 4, self.base_filters * 2)
        self.up2 = Up(self.base_filters * 2, self.base_filters)
        self.conv = nn.Conv1d(self.base_filters, self.output_channels, kernel_size=1)

        output_kernel_size = output_kernel_size
        output_padding = output_kernel_size // 2
        self.out_conv = nn.Conv1d(
            self.output_channels,
            self.output_channels,
            kernel_size=output_kernel_size,
            padding=output_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNetTrans model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns
        -------
            torch.Tensor: Output tensor of shape (batch_size, output_channels, output_length).

        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x3 = x3.permute(2, 0, 1)
        x3 = self.transformer(x3)
        x3 = x3.permute(1, 2, 0)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.conv(x)
        x = self.out_conv(x)
        x = x[..., self.output_slice]
        return x
