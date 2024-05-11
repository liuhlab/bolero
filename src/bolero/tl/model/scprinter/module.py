import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from scprinter.seq.Modules import Conv1dWrapper
from torch import nn


class EmbeddingMLP(nn.Module):
    """Turn the embedding into a weight matrix for the convolution layer."""

    def __init__(
        self,
        embedding_dim: int,
        r: int,
        layer_dim_in: int,
        groups: int,
        hidden_dim: int,
        n_layers: int = 0,
    ) -> None:
        """
        Initialize the EmbeddingMLP module.

        Args:
            embedding_dim (int): The dimension of the input embedding.
            r (int): The reduction factor.
            layer_dim_in (int): The input dimension of the layer.
            groups (int): The number of groups for grouped convolution.
            hidden_dim (int): The dimension of the hidden layer.
            n_layers (int, optional): The number of hidden layers. Defaults to 0.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_dim_in = layer_dim_in
        self.groups = groups

        layers = (
            [
                nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.GELU(),
            ]
            + [
                nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.GELU(),
            ]
            * n_layers
            + [
                nn.Linear(
                    in_features=self.hidden_dim,
                    out_features=int(
                        self.layer_dim_in * r / self.groups
                    ),  # lead to a weight matrix of shape (r, layer_dim_in)
                ),
            ]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EmbeddingMLP module.

        Args:
            embedding (torch.Tensor): The input embedding tensor.

        Returns
        -------
            torch.Tensor: The output tensor after passing through the MLP layers.
        """
        return self.mlp(embedding)


class Conv1dLoRA(nn.Module):
    """Conv1d Layer with Low Rank Adaptation Fine-tuning."""

    def __init__(
        self,
        layer: Conv1dWrapper,
        A_embedding_dim: Optional[int] = None,
        B_embedding_dim: Optional[int] = None,
        r: int = 8,
        alpha: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_layers: int = 0,
        mix_strategy: str = "concat",
    ) -> None:
        """
        Initialize the Conv1dLoRA module.

        Args:
            layer (Conv1dWrapper): The Conv1dWrapper layer.
            A_embedding_dim (int, optional): The dimension of the A embedding. Defaults to None.
            B_embedding_dim (int, optional): The dimension of the B embedding. Defaults to None.
            r (int, optional): The reduction factor. Defaults to 8.
            alpha (int, optional): The alpha value. Defaults to None.
            hidden_dim (int, optional): The dimension of the hidden layer. Defaults to None.
            n_layers (int, optional): The number of hidden layers. Defaults to 0.
            mix_strategy (str, optional): The mixing strategy. Defaults to "concat".
        """
        super().__init__()
        assert isinstance(
            layer, Conv1dWrapper
        ), "The layer must be a Conv1dWrapper layer"
        self.layer = layer
        self.pretrain_conv = layer.conv
        self.layer_dim_in = self.pretrain_conv.in_channels
        self.layer_dim_out = self.pretrain_conv.out_channels
        self.kernel_size = self.pretrain_conv.kernel_size[0]
        self.dilation = self.pretrain_conv.dilation[0]
        self.padding = self.pretrain_conv.padding[0]
        self.groups = self.pretrain_conv.groups

        if alpha is None:
            alpha = r

        self.scale = alpha / r
        self.r = r

        if hidden_dim is None:
            self.hidden_dim = A_embedding_dim
        else:
            self.hidden_dim = hidden_dim

        self.A_embedding = EmbeddingMLP(
            embedding_dim=A_embedding_dim,
            r=self.r,
            layer_dim_in=self.layer_dim_in,
            groups=self.groups,
            hidden_dim=self.hidden_dim,
            n_layers=n_layers,
        )

        self.B_embedding = EmbeddingMLP(
            embedding_dim=B_embedding_dim,
            r=self.r,
            layer_dim_in=self.layer_dim_in,
            groups=self.groups,
            hidden_dim=self.hidden_dim,
            n_layers=n_layers,
        )

        # When combined, this will lead to a weight matrix of shape (layer_dim_out, layer_dim_in, kernel_size)

        ## Make sure B starts as all zeros:
        for i in range(len(self.B_embedding)):
            if isinstance(self.B_embedding[i], nn.Linear):
                self.B_embedding[i].bias.data[...] = 0
                self.B_embedding[i].weight.data[...] = 0

        # test A_output distribution
        with torch.no_grad():
            self.A_embedding.eval()
            self.A_embedding.cuda()
            A_output = self.A_embedding(torch.arange(64).long().cuda())
            mean, std = A_output.mean(), A_output.std()
            print(f"A_output mean: {mean}, std: {std}")
            # self.scale *= 1 / (std * r)
            rescale_factor = 1 / (std)
            self.A_embedding[0].weight.data[...] *= (
                rescale_factor  # rescale the embedding matrix
            )

    @torch.no_grad()
    def collapse_layer(self, cell: int) -> Conv1dWrapper:
        """
        Collapse the layer at the given cell and return a constant Conv1dWrapper layer.

        Args:
            cell (int): The cell index.

        Returns
        -------
            Conv1dWrapper: The collapsed Conv1dWrapper layer.
        """
        if not isinstance(cell, int):
            raise ValueError("cell must be an integer")

        A = self.A_embedding(
            torch.tensor([cell]).long().to(self.A_embedding[0].weight.data.device)
        )
        B = self.B_embedding(
            torch.tensor([cell]).long().to(self.A_embedding[0].weight.data.device)
        )
        if self.kernel_size == 1:
            A = A.reshape((self.r, self.layer_dim_in))
            B = B.reshape((self.layer_dim_out, self.r))
            weight = torch.matmul(B, A)[..., None]
        else:
            A = A.reshape((int(self.layer_dim_in / self.groups), self.r))
            B = B.reshape((self.r, self.layer_dim_out * self.kernel_size))
            weight = (
                torch.matmul(A, B)
                .reshape(
                    (
                        int(self.layer_dim_in / self.groups),
                        self.layer_dim_out,
                        self.kernel_size,
                    )
                )
                .contiguous()
                .permute(1, 0, 2)
            )
        weight_scaled = weight * self.scale
        new_layer = copy.deepcopy(self.layer)
        new_layer.conv.weight.data[...] = new_layer.conv.weight.data + weight_scaled
        return new_layer

    def forward(
        self, X: torch.Tensor, cells: torch.Tensor, modes: Optional[Tuple[int]] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Conv1dLoRA module.

        Args:
            X (torch.Tensor): The input tensor.
            cells (torch.Tensor): The cell tensor.
            modes (Tuple[int], optional): The modes. Defaults to None.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        if self.kernel_size == 1:
            # When kernel_size == 1, the convolution is actually a linear layer, take a short path
            A = self.A_embedding(cells).reshape((-1, self.r, self.layer_dim_in))
            B = self.B_embedding(cells).reshape((-1, self.layer_dim_out, self.r))
            # x: (batch_size, layer_dim_in, seq_len)
            lora_x = torch.bmm(A, X)  # (batch_size, r, seq_len)
            if modes is not None:
                B = B[:, modes]
            lora_x = torch.bmm(B, lora_x)  # (batch_size, layer_dim_out, seq_len
            return lora_x * self.scale + (self.layer(X, modes=modes))
        else:
            # When kernel_size > 1, the convolution can be written as groupped convolutioni,
            # take a long path
            bs = X.shape[0]  # batch_size
            A = self.A_embedding(cells).reshape(
                (bs, int(self.layer_dim_in / self.groups), self.r)
            )
            B = self.B_embedding(cells).reshape(
                (bs, self.r, self.layer_dim_out, self.kernel_size)
            )
            if modes is not None:
                B = B[:, modes]
            B = B.reshape((bs, self.r, self.layer_dim_out * self.kernel_size))
            weight = (
                torch.bmm(A, B)
                .reshape(
                    (
                        bs,
                        int(self.layer_dim_in / self.groups),
                        self.layer_dim_out,
                        self.kernel_size,
                    )
                )
                .contiguous()
                .permute(0, 2, 1, 3)
            )
            # size of (batch_size, layer_dim_out, layer_dim_in / groups, kernel_size)

            # route 1
            weight = weight.reshape(
                (-1, int(self.layer_dim_in / self.groups), self.kernel_size)
            )
            # size of (batch_size * layer_dim_out, layer_dim_in / groups, kernel_size)
            # X after reshape (1, batch_size*layer_dim_in, seq_len)
            lora_x = F.conv1d(
                X.reshape((1, -1, X.shape[-1])),
                weight=weight,
                bias=None,
                dilation=self.dilation,
                groups=bs * self.groups,
                padding=self.padding,
            )  # each batch_size is a group
            # within each group, the convolution projects from (layer_dim_in, seq_len) to (layer_dim_out, seq_len)
            # This is equivalent to a for loop over each sample in the batch
            lora_x = lora_x.view(bs, self.layer_dim_out, -1)
            X = lora_x * self.scale + self.layer(X, modes=modes)
            return X
