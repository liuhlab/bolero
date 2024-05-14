import copy
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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


class EmbeddingMLP(nn.Module):
    """
    This class turn the input embedding into one of the LoRA low-rank weight matrix (A or B) through a simple MLP.
    """

    def __init__(
        self,
        embedding_dim: int,
        r: int,
        layer_dim: int,
        groups: int,
        hidden_dim: int,
        n_layers: int = 0,
    ) -> None:
        """
        Initialize the EmbeddingMLP module.

        Args:
            embedding_dim (int): The dimension of the input embedding.
            r (int): The low-rank dimension in LoRA.
            layer_dim (int): The input (A) or output (B) channel dimension of the layer.
            groups (int): The number of groups for grouped convolution.
            hidden_dim (int): The dimension of the hidden layer.
            n_layers (int, optional): The number of additional hidden layers in this MLP. Defaults to 0.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = self.embedding_dim if hidden_dim is None else hidden_dim
        self.layer_dim = layer_dim
        self.groups = groups

        # lead to a weight matrix of shape (r, layer_dim)
        self.out_feathres = int(self.layer_dim * r / self.groups)

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
                    out_features=self.out_feathres,
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

    def zero_weights_and_bias(self):
        """
        Zero the weights and bias of the MLP's first layer, use this in B embedding.
        """
        for i in range(len(self.mlp)):
            if isinstance(self.mlp[i], nn.Linear):
                self.mlp[i].bias.data[...] = 0
                self.mlp[i].weight.data[...] = 0
        return

    def scale_weights(self, example_embedding):
        """
        Scale the weights of the MLP's first layer based on the example embedding, use this in A embedding.
        """
        with torch.no_grad():
            self.eval()
            try:
                self.cuda()
            except AssertionError:
                pass

            example_output = self(example_embedding)
            mean, std = example_output.mean(), example_output.std()
            print(f"Embedding example mean: {mean}, std: {std}")
            rescale_factor = 1 / (std)
            self.mlp[0].weight.data[...] *= (
                rescale_factor  # rescale the embedding matrix
            )
        return

    def fix_parameters(self):
        """
        Fix the parameters of the MLP.
        """
        for param in self.parameters():
            param.requires_grad = False
        return


class Conv1dMultiLoRA(nn.Module):
    """Conv1d Layer with Multiple Low Rank Adaptation Fine-tuning."""

    def __init__(
        self,
        layer: Conv1dWrapper,
        A_embedding_dims: Union[int, list[int]],
        B_embedding_dims: Optional[Union[int, list[int]]] = None,
        r: int = 8,
        alpha: Optional[int] = None,
        hidden_dims: Optional[list[int]] = None,
        n_layers: int = 0,
        example_a_embedding: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the Conv1dLoRA module.

        Args:
            layer (Conv1dWrapper): The pretrained Conv1dWrapper layer.
            A_embedding_dim (int, optional): The input dimension of the A embedding. Defaults to None.
            B_embedding_dim (int, optional): The input dimension of the B embedding. Defaults to None.
            r (int, optional): The low-rank dimension. Defaults to 8.
            alpha (int, optional): The alpha value to calculate loRA scale. Defaults to None, which will be set to r.
            hidden_dims (int, optional): The dimension of the hidden layer in EmbeddingMLP. Defaults to None, which will be set to A_embedding_dims.
            n_layers (int, optional): The number of additional hidden layers in EmbeddingMLP. Defaults to 0.
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

        if B_embedding_dims is None:
            B_embedding_dims = A_embedding_dims
        if isinstance(A_embedding_dims, int):
            A_embedding_dims = [A_embedding_dims]
        self.A_embedding_dims = np.array(A_embedding_dims)
        if isinstance(B_embedding_dims, int):
            B_embedding_dims = [B_embedding_dims]
        self.B_embedding_dims = np.array(B_embedding_dims)

        assert (
            self.A_embedding_dims.size == self.B_embedding_dims.size
        ), f"A_embedding_dims and B_embedding_dims must have the same length, got {self.A_embedding_dims.size} and {self.B_embedding_dims.size}"

        if hidden_dims is None:
            self.hidden_dim = A_embedding_dims
        else:
            assert (
                len(hidden_dims) == len(A_embedding_dims)
            ), f"hidden_dim must have the same length as A_embedding_dims (length of {len(A_embedding_dims)})"
            self.hidden_dim = hidden_dims

        self.A_embedding_list = nn.ModuleList(
            EmbeddingMLP(
                embedding_dim=A_embedding_dim,
                r=self.r,
                layer_dim_in=self.layer_dim_in,
                groups=self.groups,
                hidden_dim=self.hidden_dim,
                n_layers=n_layers,
            )
            for A_embedding_dim in A_embedding_dims
        )

        self.B_embedding_list = nn.ModuleList(
            EmbeddingMLP(
                embedding_dim=B_embedding_dim,
                r=self.r,
                layer_dim_in=self.layer_dim_out,
                groups=self.groups,
                hidden_dim=self.hidden_dim,
                n_layers=n_layers,
            )
            for B_embedding_dim in B_embedding_dims
        )

        # When combined, this will lead to a weight matrix of shape (layer_dim_out, layer_dim_in, kernel_size)
        ## Make sure B weigths and bias start as all zeros:
        self.b_embedding_zero_weights_and_bias()

        # test A_output distribution and rescale the weights of the first layer
        # this step should be called especially when the model is initialized
        if example_a_embedding is not None:
            self.a_embedding_scale_weights(example_a_embedding)

    def b_embedding_zero_weights_and_bias(self):
        """
        Zero the weights and bias of the B embedding.
        """
        for _m in self.B_embedding_list:
            _m.zero_weights_and_bias()
        return

    def a_embedding_scale_weights(self, example_embedding, max_sample=128):
        """
        Scale the weights of the A embedding based on the example embedding.
        """
        if example_embedding.shape[0] > max_sample:
            # random choice max_sample rows of example_embedding
            example_embedding = example_embedding[
                torch.random.torch.randint(0, example_embedding.shape[0], (max_sample,))
            ]
        for _m in self.A_embedding_list:
            _m.scale_weights(example_embedding)
        return

    def _validate_embedding_sizes(self, A_embeddings, B_embeddings):
        """
        Check if the embedding sizes are correct.
        """
        if B_embeddings is None:
            B_embeddings = A_embeddings
        if not isinstance(A_embeddings, list):
            A_embeddings = [A_embeddings]
        if not isinstance(B_embeddings, list):
            B_embeddings = [B_embeddings]

        # check number of embeddings
        if len(A_embeddings) != len(self.A_embedding_list):
            raise ValueError(
                f"Number of A embeddings {len(A_embeddings)} must match the number of A embedding layers {len(self.A_embedding_list)}"
            )
        if len(B_embeddings) != len(self.B_embedding_list):
            raise ValueError(
                f"Number of B embeddings {len(B_embeddings)} must match the number of B embedding layers {len(self.B_embedding_list)}"
            )

        # check each embedding size
        A_sizes = np.array([e.shape[-1] for e in A_embeddings])
        if not np.all(A_sizes == self.A_embedding_dims):
            raise ValueError(
                f"A embedding sizes {A_sizes} do not match the expected sizes of the A embedding layers {self.A_embedding_dims}"
            )
        B_sizes = np.array([e.shape[-1] for e in B_embeddings])
        if not np.all(B_sizes == self.B_embedding_dims):
            raise ValueError(
                f"B embedding sizes {B_sizes} do not match the expected sizes of the B embedding layers {self.B_embedding_dims}"
            )
        return A_embeddings, B_embeddings

    def fix_parameters(self, layers=None):
        """
        Fix the parameters of the EmbeddingMLP at all or specific layers.
        """
        if layers is None:
            for param in self.parameters():
                param.requires_grad = False
            return

        if isinstance(layers, int):
            layers = [layers]
        for idx, _m in enumerate(self.A_embedding_list):
            if layers is None or idx in layers:
                _m.fix_parameters()
        for idx, _m in enumerate(self.B_embedding_list):
            if layers is None or idx in layers:
                _m.fix_parameters()
        return

    def _collapse_single_layer(self, idx, A_embedding, B_embedding):
        A = self.A_embedding_list[idx](A_embedding)
        B = self.B_embedding_list[idx](B_embedding)

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
        return weight_scaled

    @torch.no_grad()
    def collapse_layer(self, A_embeddings, B_embeddings=None) -> Conv1dWrapper:
        """
        Collapse the layer at the given embedding and return a constant Conv1dWrapper layer.

        Args:
            A_embeddings (torch.Tensor): The input A embeddings.
            B_embeddings (torch.Tensor, optional): The input B embeddings. Defaults to None.

        Returns
        -------
            Conv1dWrapper: The collapsed Conv1dWrapper layer.
        """
        # validate the embeddings
        A_embeddings, B_embeddings = self._validate_embedding_sizes(
            A_embeddings, B_embeddings
        )

        lora_weights = torch.zeros_like(self.layer.conv.weight.data)
        for idx, A_input, B_input in enumerate(zip(A_embeddings, B_embeddings)):
            # collapse each individual LoRA layer
            weight_scaled = self._collapse_single_layer(idx, A_input, B_input)
            lora_weights = lora_weights + weight_scaled

        new_layer = copy.deepcopy(self.layer)
        new_layer.conv.weight.data[...] = new_layer.conv.weight.data + lora_weights
        return new_layer

    def _forward_single_layer(self, idx, X, A_input, B_input, modes):
        A_mlp = self.A_embedding_list[idx]
        B_mlp = self.B_embedding_list[idx]

        if self.kernel_size == 1:
            # When kernel_size == 1, the convolution is actually a linear layer, take a short path
            A = A_mlp(A_input).reshape((-1, self.r, self.layer_dim_in))
            B = B_mlp(B_input).reshape((-1, self.layer_dim_out, self.r))
            # x: (batch_size, layer_dim_in, seq_len)
            lora_x = torch.bmm(A, X)  # (batch_size, r, seq_len)
            if modes is not None:
                B = B[:, modes]
            lora_x = torch.bmm(B, lora_x)  # (batch_size, layer_dim_out, seq_len
        else:
            # When kernel_size > 1, the convolution can be written as groupped convolution,
            # take a long path

            # HL's note: unlike normal Conv1D weights, LoRA weights here are calculated from input embedding's
            # therefore the weights contains the additional batch_size dimension
            # the following code is a way to vectorize the calculation of the lora_x for each sample using their corresponding lora weights.

            bs = X.shape[0]  # batch_size
            A = A_mlp(A_input).reshape(
                (bs, int(self.layer_dim_in / self.groups), self.r)
            )
            B = B_mlp(B_input).reshape(
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
        return lora_x

    def forward(
        self,
        X: torch.Tensor,
        A_embeddings,
        B_embeddings=None,
        modes: Optional[Tuple[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Conv1dLoRA module.

        Args:
            X (torch.Tensor): The input tensor.
            A_embeddings (torch.Tensor): The input A embeddings.
            B_embeddings (torch.Tensor, optional): The input B embeddings. Defaults to None.
            modes (Tuple[int], optional): The modes to select. Defaults to None.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        # validate the embeddings
        A_embeddings, B_embeddings = self._validate_embedding_sizes(
            A_embeddings, B_embeddings
        )

        # pretrain layer output
        layer_output = self.layer(X, modes=modes)

        lora_x = torch.zeros_like(layer_output)
        for idx, A_input, B_input in zip(A_embeddings, B_embeddings):
            _x = self._forward_single_layer(idx, X, A_input, B_input, modes)
            lora_x = lora_x + _x
        final_output = layer_output + lora_x
        return final_output
