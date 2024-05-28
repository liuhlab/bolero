import torch
import torch.nn.functional as F
from torch import nn


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Convolutional 1D block module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conv1DBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor after applying the Conv1DBlock module.
        """
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Down module of the U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool1d(2), Conv1DBlock(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Down module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor after applying the Down module.
        """
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Up module of the U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = Conv1DBlock(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Up module.

        Args:
            x1 (torch.Tensor): Input tensor from the previous layer.
            x2 (torch.Tensor): Input tensor from the skip connection.

        Returns
        -------
            torch.Tensor: Output tensor after applying the Up module.
        """
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_size: int, num_heads: int, ff_hidden_dim: int, dropout: float
    ):
        """
        Transformer Block module.

        Args:
            embed_size (int): The input and output embedding size.
            num_heads (int): The number of attention heads.
            ff_hidden_dim (int): The hidden dimension of the feed-forward network.
            dropout (float): The dropout probability.

        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer Block.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, embed_size).

        Returns
        -------
            torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_size).

        """
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerStack(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout, num_layers):
        """
        TransformerStack module that applies multiple TransformerBlocks in sequence.

        Args:
            embed_size (int): The input and output embedding size.
            num_heads (int): The number of attention heads in the TransformerBlock.
            ff_hidden_dim (int): The hidden dimension of the feed-forward network in the TransformerBlock.
            dropout (float): The dropout probability.
            num_layers (int): The number of TransformerBlocks in the stack.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
