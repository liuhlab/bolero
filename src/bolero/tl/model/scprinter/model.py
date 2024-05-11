from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from scprinter.seq.Baseline_Modules import BiasInjection
from scprinter.seq.Modules import Conv1dLoRA, Conv1dWrapper


class scFootprintBPNet(nn.Module):
    """
    This class defines the scFootprintBPNet model.

    Parameters
    ----------
        dna_cnn_model (nn.Module): The DNA CNN model.
        hidden_layer_model (nn.Module): The hidden layer model.
        profile_cnn_model (nn.Module): The profile CNN model.
        dna_len (int): The length of the DNA sequence.
        output_len (int): The length of the output.
        embeddings (Optional[torch.Tensor]): The embeddings tensor.
        rank (int): The rank parameter.
        hidden_dim (Optional[int]): The hidden dimension.
        lora_dna_cnn (bool): Whether to use LoRA for DNA CNN.
        lora_dilated_cnn (bool): Whether to use LoRA for dilated CNN.
        lora_pff_cnn (bool): Whether to use LoRA for PFF CNN.
        lora_output_cnn (bool): Whether to use LoRA for output CNN.
        lora_count_cnn (bool): Whether to use LoRA for count CNN.
        method (str): The method to use (either "lora" or "bias").
        n_lora_layers (int): The number of LoRA layers.

    Attributes
    ----------
        dna_cnn_model (nn.Module): The DNA CNN model.
        hidden_layer_model (nn.Module): The hidden layer model.
        profile_cnn_model (nn.Module): The profile CNN model.
        dna_len (int): The length of the DNA sequence.
        output_len (int): The length of the output.
        embeddings (Optional[nn.Embedding]): The embeddings tensor.
    """

    def __init__(
        self,
        dna_cnn_model: nn.Module = None,
        hidden_layer_model: nn.Module = None,
        profile_cnn_model: nn.Module = None,
        dna_len: int = 2114,
        output_len: int = 1000,
        embeddings: Optional[torch.Tensor] = None,
        rank: int = 8,
        hidden_dim: Optional[int] = None,
        lora_dna_cnn: bool = False,
        lora_dilated_cnn: bool = False,
        lora_pff_cnn: bool = False,
        lora_output_cnn: bool = False,
        lora_count_cnn: bool = False,
        method: str = "lora",
        n_lora_layers: int = 0,
    ):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.dna_len = dna_len
        self.output_len = output_len

        if embeddings is not None:
            self.embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.embeddings.weight.data = torch.from_numpy(embeddings).float()
            self.embeddings.weight.requires_grad = False

        if method == "lora":
            wrapper = Conv1dLoRA
        else:
            wrapper = BiasInjection

        if lora_dna_cnn:
            self.dna_cnn_model.conv = wrapper(
                self.dna_cnn_model.conv,
                A_embedding=self.embeddings,
                B_embedding=self.embeddings,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )

        hidden_layers = self.hidden_layer_model.layers
        for i in range(len(hidden_layers)):
            if lora_dilated_cnn:
                hidden_layers[i].module.conv1 = wrapper(
                    hidden_layers[i].module.conv1,
                    A_embedding=self.embeddings,
                    B_embedding=self.embeddings,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )
            if lora_pff_cnn:
                hidden_layers[i].module.conv2 = wrapper(
                    hidden_layers[i].module.conv2,
                    A_embedding=self.embeddings,
                    B_embedding=self.embeddings,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )

        if lora_output_cnn:
            self.profile_cnn_model.conv_layer = wrapper(
                self.profile_cnn_model.conv_layer,
                A_embedding=self.embeddings,
                B_embedding=self.embeddings,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
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
            self.profile_cnn_model.linear = wrapper(
                self.profile_cnn_model.linear,
                A_embedding=self.embeddings,
                B_embedding=self.embeddings,
                r=1,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )

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

    def collapse(self, cell, turn_on_grads=True):
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
                model_clone.dna_cnn_model.conv.collapse_layer(cell)
            )
        if not isinstance(
            model_clone.hidden_layer_model.layers[0].module.conv1, Conv1dWrapper
        ):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv1 = layer.module.conv1.collapse_layer(cell)
        if not isinstance(
            model_clone.hidden_layer_model.layers[0].module.conv2, Conv1dWrapper
        ):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv2 = layer.module.conv2.collapse_layer(cell)
        if not isinstance(model_clone.profile_cnn_model.conv_layer, Conv1dWrapper):
            model_clone.profile_cnn_model.conv_layer = (
                model_clone.profile_cnn_model.conv_layer.collapse_layer(cell)
            )
        if not isinstance(model_clone.profile_cnn_model.linear, Conv1dWrapper):
            model_clone.profile_cnn_model.linear = (
                model_clone.profile_cnn_model.linear.collapse_layer(cell)
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
        if output_len is None:
            output_len = self.output_len
        # get the motifs!
        X = self.dna_cnn_model(
            X, cell_embedding=cell_embedding, region_embedding=region_embedding
        )

        # get the hidden layer
        X = self.hidden_layer_model(
            X, cell_embedding=cell_embedding, region_embedding=region_embedding
        )

        # get the profile
        score = self.profile_cnn_model(
            X,
            cell_embedding=cell_embedding,
            region_embedding=region_embedding,
            output_len=output_len,
            modes=modes,
        )
        return score
