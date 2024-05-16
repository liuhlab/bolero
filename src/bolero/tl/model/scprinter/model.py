from copy import deepcopy
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from bolero.tl.model.scprinter.module import Conv1dMultiLoRA, Conv1dWrapper


class scFootprintBPNetLoRA(nn.Module):
    """scFootprintBPNetLoRA model."""

    def __init__(
        self,
        dna_cnn_model: nn.Module = None,
        hidden_layer_model: nn.Module = None,
        profile_cnn_model: nn.Module = None,
        dna_len: int = 1840,
        output_len: int = 1000,
        example_cell_embedding: Optional[np.ndarray] = None,
        example_region_embedding: Optional[np.ndarray] = None,
        a_embedding: str = "cell",
        b_embedding: str = "cell",
        lora_dna_cnn: bool = False,
        lora_dilated_cnn: bool = False,
        lora_pff_cnn: bool = False,
        lora_output_cnn: bool = False,
        lora_count_cnn: bool = False,
        rank: int = 8,
        n_lora_layers: int = 0,
        hidden_dim: Optional[int] = None,
    ):
        # ===============
        # Initialize the model
        # ===============

        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.dna_len = dna_len
        self.output_len = output_len

        # ===============
        # Prepare LoRA Layer Wrapper
        # ===============

        # will store the actual embedding dims
        self.A_embedding_dims = None
        self.B_embedding_dims = None
        # will store the input cell embedding and region embedding, output A or B embedding
        self.A_embedding_process = None
        self.B_embedding_process = None

        # determine the embedding dims based on a_embedding and b_embedding type
        use_rows = min(
            256, example_cell_embedding.shape[0], example_region_embedding.shape[0]
        )
        if example_cell_embedding is not None:
            example_cell_embedding = torch.Tensor(
                np.array(example_cell_embedding[:use_rows])
            )
        if example_region_embedding is not None:
            example_region_embedding = torch.Tensor(
                np.array(example_region_embedding[:use_rows])
            )

        self._determine_embedding_dims(
            cell_embedding=example_cell_embedding,
            region_embedding=example_region_embedding,
            a_embedding=a_embedding,
            b_embedding=b_embedding,
        )
        example_a_embedding = self.A_embedding_process(
            example_cell_embedding, example_region_embedding
        )

        conv1d_lora = partial(
            Conv1dMultiLoRA,
            A_embedding_dims=self.A_embedding_dims,
            B_embedding_dims=self.B_embedding_dims,
            r=rank,
            hidden_dims=hidden_dim,
            n_layers=n_lora_layers,
            example_a_embedding=example_a_embedding,
        )

        # ===============
        # Apply LoRA Layers to each sub-model
        # ===============

        # DNA Model
        if lora_dna_cnn:
            self.dna_cnn_model.conv = conv1d_lora(layer=self.dna_cnn_model.conv)

        # Hidden Layer Model
        hidden_layers = self.hidden_layer_model.layers
        for i in range(len(hidden_layers)):
            if lora_dilated_cnn:
                hidden_layers[i].module.conv1 = conv1d_lora(
                    layer=hidden_layers[i].module.conv1,
                )
            if lora_pff_cnn:
                hidden_layers[i].module.conv2 = conv1d_lora(
                    layer=hidden_layers[i].module.conv2,
                )

        # Profile Model
        if lora_output_cnn:
            self.profile_cnn_model.conv_layer = conv1d_lora(
                layer=self.profile_cnn_model.conv_layer,
            )
        if isinstance(self.profile_cnn_model.linear, nn.Linear):
            print("translating linear into conv1d")
            weight = self.profile_cnn_model.linear.weight.data
            print(weight.shape)
            bias = self.profile_cnn_model.linear.bias.data
            self.profile_cnn_model.linear = Conv1dWrapper(
                weight.shape[1], weight.shape[0], 1
            )
            print(self.profile_cnn_model.linear.conv.weight.shape)
            self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            self.profile_cnn_model.linear.conv.bias.data = bias
        if lora_count_cnn:
            self.profile_cnn_model.linear = conv1d_lora(
                layer=self.profile_cnn_model.linear,
            )

    @staticmethod
    def _get_cell_embedding(cell, region):
        return cell

    @staticmethod
    def _get_region_embedding(cell, region):
        return region

    @staticmethod
    def _concat_cell_region_embedding(cell, region):
        return torch.cat((cell, region), dim=-1)

    @staticmethod
    def _get_null(cell, region):
        return None

    def _determine_embedding_dims(
        self, cell_embedding, region_embedding, a_embedding, b_embedding
    ):
        cell_embedding_dims = (
            None if cell_embedding is None else cell_embedding.shape[-1]
        )
        region_embedding_dims = (
            None if region_embedding is None else region_embedding.shape[-1]
        )

        # determine A B embedding dims
        if a_embedding == "concat":
            self.A_embedding_dims = cell_embedding_dims + region_embedding_dims
            self.A_embedding_process = self._concat_cell_region_embedding
        elif a_embedding == "cell":
            self.A_embedding_dims = cell_embedding_dims
            self.A_embedding_process = self._get_cell_embedding
        elif a_embedding == "region":
            self.A_embedding_dims = region_embedding_dims
            self.A_embedding_process = self._get_region_embedding
        elif a_embedding == "none":
            self.A_embedding_dims = 0
            self.A_embedding_process = self._get_null
        else:
            raise ValueError(f"Invalid A embedding type: {a_embedding}")

        if b_embedding == "concat":
            self.B_embedding_dims = cell_embedding_dims + region_embedding_dims
            self.B_embedding_process = self._concat_cell_region_embedding
        elif b_embedding == "cell":
            self.B_embedding_dims = cell_embedding_dims
            self.B_embedding_process = self._get_cell_embedding
        elif b_embedding == "region":
            self.B_embedding_dims = region_embedding_dims
            self.B_embedding_process = self._get_region_embedding
        elif b_embedding == "none":
            self.B_embedding_dims = 0
            self.B_embedding_process = self._get_null
        else:
            raise ValueError(f"Invalid B embedding type: {b_embedding}")
        return

    def return_origin(self):
        """
        Returns a clone of the model with original layers.

        Returns
        -------
            scFootprintBPNet: A clone of the model with original layers.
        """
        self = self.to("cpu")
        if isinstance(self.profile_cnn_model.linear, nn.Linear):
            print("translating linear into conv1d")
            weight = self.profile_cnn_model.linear.weight.data
            print(weight.shape)
            bias = self.profile_cnn_model.linear.bias.data
            self.profile_cnn_model.linear = Conv1dWrapper(
                weight.shape[1], weight.shape[0], 1
            )
            print(self.profile_cnn_model.linear.conv.weight.shape)
            self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            self.profile_cnn_model.linear.conv.bias.data = bias

        model_clone = deepcopy(self)
        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = model_clone.dna_cnn_model.conv.layer
        if not isinstance(
            model_clone.hidden_layer_model.layers[0].module.conv1, Conv1dWrapper
        ):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv1 = layer.module.conv1.layer
        if not isinstance(
            model_clone.hidden_layer_model.layers[0].module.conv2, Conv1dWrapper
        ):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv2 = layer.module.conv2.layer
        if not isinstance(model_clone.profile_cnn_model.conv_layer, Conv1dWrapper):
            model_clone.profile_cnn_model.conv_layer = (
                model_clone.profile_cnn_model.conv_layer.layer
            )
        if not isinstance(model_clone.profile_cnn_model.linear, Conv1dWrapper):
            model_clone.profile_cnn_model.linear = (
                model_clone.profile_cnn_model.linear.layer
            )

        return model_clone

    def collapse(self, cell_embedding=None, region_embedding=None, turn_on_grads=True):
        """
        Returns a clone of the model with collapsed layers.

        Parameters
        ----------
            cell: The cell parameter.
            turn_on_grads (bool): Whether to turn on gradients.

        Returns
        -------
            scFootprintBPNet: A clone of the model with collapsed layers.
        """
        A_embeddings = self.A_embedding_process(cell_embedding, region_embedding)
        B_embeddings = self.B_embedding_process(cell_embedding, region_embedding)

        if isinstance(self.profile_cnn_model.linear, nn.Linear):
            print("translating linear into conv1d")
            weight = self.profile_cnn_model.linear.weight.data
            print(weight.shape)
            bias = self.profile_cnn_model.linear.bias.data
            self.profile_cnn_model.linear = Conv1dWrapper(
                weight.shape[1], weight.shape[0], 1
            )
            print(self.profile_cnn_model.linear.conv.weight.shape)
            self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            self.profile_cnn_model.linear.conv.bias.data = bias

        model_clone = deepcopy(self)
        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = (
                model_clone.dna_cnn_model.conv.collapse_layer(
                    A_embeddings, B_embeddings
                )
            )
        if not isinstance(
            model_clone.hidden_layer_model.layers[0].module.conv1, Conv1dWrapper
        ):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv1 = layer.module.conv1.collapse_layer(
                    A_embeddings, B_embeddings
                )
        if not isinstance(
            model_clone.hidden_layer_model.layers[0].module.conv2, Conv1dWrapper
        ):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv2 = layer.module.conv2.collapse_layer(
                    A_embeddings, B_embeddings
                )
        if not isinstance(model_clone.profile_cnn_model.conv_layer, Conv1dWrapper):
            model_clone.profile_cnn_model.conv_layer = (
                model_clone.profile_cnn_model.conv_layer.collapse_layer(
                    A_embeddings, B_embeddings
                )
            )
        if not isinstance(model_clone.profile_cnn_model.linear, Conv1dWrapper):
            model_clone.profile_cnn_model.linear = (
                model_clone.profile_cnn_model.linear.collapse_layer(
                    A_embeddings, B_embeddings
                )
            )
        if turn_on_grads:
            for p in model_clone.parameters():
                p.requires_grad = True
        return model_clone

    def forward(
        self, X, cell_embedding=None, region_embedding=None, output_len=None, modes=None
    ):
        """
        Forward pass of the model.

        Parameters
        ----------
            X: The input tensor.
            cell_embedding: The cell embedding tensor.
            region_embedding: The region embedding tensor.
            output_len: The length of the output.
            modes: The modes tensor.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        A_embeddings = self.A_embedding_process(cell_embedding, region_embedding)
        B_embeddings = self.B_embedding_process(cell_embedding, region_embedding)

        if output_len is None:
            output_len = self.output_len

        # get the motifs
        X = self.dna_cnn_model(X, A_embeddings=A_embeddings, B_embeddings=B_embeddings)

        # get the hidden layer
        X = self.hidden_layer_model(
            X, A_embeddings=A_embeddings, B_embeddings=B_embeddings
        )

        # get the profile
        score = self.profile_cnn_model(
            X,
            A_embeddings=A_embeddings,
            B_embeddings=B_embeddings,
            output_len=output_len,
            modes=modes,
        )
        return score
