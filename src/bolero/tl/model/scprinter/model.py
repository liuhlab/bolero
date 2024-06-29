"""
scFootprintBPNetLoRA

This model composed of three parts
1. DNA CNN Model:
Description: A convolutional neural network model for extracting motif level features from DNA sequences.
Input Shape: (batch_size, dna_channels, dna_len), default (64, 4, 1840)
Output Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)

2. Hidden Layer Model:
Description: A multi-layer residual connect CNN model for encoding sequence information.
Layers: 8 layers of ConvBlocks, each block contains a dialated, grouped CNN layer and a pointwise feedforward CNN layer; connected by residual connections.
Input Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)
Output Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)

3. Profile CNN Model:
Description: A output model for predicting multi-mode footprint (mode as channels) or coverage signals.
Input Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)
Output Shape:
    - Footprint, (batch_size, n_modes, output_len), default (64, 99, 800)
    - Coverage (log10(cov_mean) of the region), (batch_size, 1), default (64, 1)

"""

from copy import deepcopy
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from bolero.tl.model.generic.module import DNA_CNN, Conv1dWrapper, DilatedCNN
from bolero.tl.model.generic.train_helper import validate_config
from bolero.tl.model.scprinter.module import Conv1dMultiLoRA, FootprintsHead


class scFootprintBPNet(nn.Module):
    """scFootprintBPNet bulk model."""

    default_config = {
        "dna_len": "auto",
        "output_len": 800,
    }

    @classmethod
    def get_default_config(cls) -> dict:
        """Get default configuration combined from dataset, model and trainer."""
        dna_cnn_config = deepcopy(DNA_CNN.default_config)
        hidden_layer_config = deepcopy(DilatedCNN.default_config)
        output_config = deepcopy(FootprintsHead.default_config)
        default_config = {
            **cls.default_config,
            **dna_cnn_config,
            **hidden_layer_config,
            **output_config,
        }
        return default_config

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        default_config = cls.get_default_config()
        config = {k: v for k, v in config.items() if k in default_config}
        validate_config(config, default_config)

        activation = config["activation"]
        if isinstance(activation, str):
            if activation.lower() == "gelu":
                activation = nn.GELU()
            elif activation.lower() == "relu":
                activation = nn.ReLU()
            else:
                raise ValueError(f"Invalid activation: {activation}")

        dna_cnn_model = DNA_CNN.create_from_config(config)

        hidden_layer_model = DilatedCNN.create_from_config(config)

        output_model = FootprintsHead.create_from_config(config)

        dna_len = config["dna_len"]
        output_len = config["output_len"]
        dia_kernel_size = hidden_layer_model.dia_kernel_size
        hidden_conv_blocks = hidden_layer_model.n_blocks
        if dna_len == "auto":
            # calculate the dna_len to prevent the padding issue
            dna_len = output_len + dna_cnn_model.conv.weight.shape[2] - 1
            for i in range(hidden_conv_blocks):
                dna_len = dna_len + 2 * (
                    dia_kernel_size // 2
                ) * hidden_layer_model.dilation_func(i)

        return cls(
            dna_cnn_model=dna_cnn_model,
            hidden_layer_model=hidden_layer_model,
            output_model=output_model,
            dna_len=dna_len,
            output_len=output_len,
        )

    def __init__(
        self,
        dna_cnn_model,
        hidden_layer_model,
        output_model,
        dna_len,
        output_len,
    ):
        # ===============
        # Initialize the model
        # ===============
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = output_model
        self.dna_len = dna_len
        self.output_len = output_len

    def forward(self, X, output_len=None, modes=None, **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
            X: The input tensor.
            output_len: The length of the output.
            modes: The modes tensor.
            kwargs: placeholder for additional keyword arguments to allow for compatibility with other models.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        if output_len is None:
            output_len = self.output_len

        # get the motifs
        X = self.dna_cnn_model(X)

        # get the hidden layer
        X = self.hidden_layer_model(X)

        # get the profile
        score = self.profile_cnn_model(X, output_len=output_len, modes=modes)
        return score


class scFootprintBPNetLoRA(nn.Module):
    """scFootprintBPNetLoRA model."""

    default_config = {
        "dna_cnn_model": None,
        "hidden_layer_model": None,
        "profile_cnn_model": None,
        "dna_len": None,
        "output_len": None,
        "example_cell_embedding": None,
        "example_region_embedding": None,
        "a_embedding": "cell",
        "b_embedding": "cell",
        "lora_dna_cnn": True,
        "lora_dilated_cnn": True,
        "lora_pff_cnn": True,
        "lora_output_cnn": True,
        "lora_count_cnn": True,
        "lora_rank": 8,
        "n_lora_layers": 0,
        "lora_hidden_dim": None,
        "lora_output_layer_groups": 1,
        "no_over_rank": False,
    }

    @classmethod
    def get_default_config(cls):
        """Get the default configuration for the model."""
        return cls.default_config

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        validate_config(config, cls.get_default_config())
        config = {k: v for k, v in config.items() if k in cls.default_config}
        return cls(**config)

    def __init__(
        self,
        dna_cnn_model: DNA_CNN,
        hidden_layer_model: DilatedCNN,
        profile_cnn_model: FootprintsHead,
        dna_len: int = 1840,
        output_len: int = 800,
        example_cell_embedding: Optional[np.ndarray] = None,
        example_region_embedding: Optional[np.ndarray] = None,
        a_embedding: str = "cell",
        b_embedding: str = "cell",
        lora_dna_cnn: bool = False,
        lora_dilated_cnn: bool = False,
        lora_pff_cnn: bool = False,
        lora_output_cnn: bool = False,
        lora_count_cnn: bool = False,
        lora_rank: int = 8,
        n_lora_layers: int = 0,
        lora_hidden_dim: Optional[int] = None,
        lora_output_layer_groups: Optional[int] = 1,
        no_over_rank: bool = False,
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
        if example_cell_embedding is not None:
            use_rows = min(256, example_cell_embedding.shape[0])
            example_cell_embedding = torch.Tensor(
                np.array(example_cell_embedding[:use_rows])
            )
        if example_region_embedding is not None:
            use_rows = min(256, example_region_embedding.shape[0])
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
            hidden_dims=lora_hidden_dim,
            n_layers=n_lora_layers,
            example_a_embedding=example_a_embedding,
            output_layer_groups=lora_output_layer_groups,
            no_over_rank=no_over_rank,
        )

        # ===============
        # Apply LoRA Layers to each sub-model
        # ===============

        # DNA Model
        if lora_dna_cnn:
            self.dna_cnn_model.conv = conv1d_lora(
                layer=self.dna_cnn_model.conv, r=lora_rank
            )

        # Hidden Layer Model
        hidden_layers = self.hidden_layer_model.layers
        for i in range(len(hidden_layers)):
            if lora_dilated_cnn:
                hidden_layers[i].module.conv1 = conv1d_lora(
                    layer=hidden_layers[i].module.conv1,
                    r=lora_rank,
                )
            if lora_pff_cnn:
                hidden_layers[i].module.conv2 = conv1d_lora(
                    layer=hidden_layers[i].module.conv2,
                    r=lora_rank,
                )

        # Profile Model
        if lora_output_cnn:
            self.profile_cnn_model.conv_layer = conv1d_lora(
                layer=self.profile_cnn_model.conv_layer,
                r=lora_rank,
            )
        if lora_count_cnn:
            # if isinstance(self.profile_cnn_model.linear, nn.Linear):
            #     # translating linear into conv1d"
            #     weight = self.profile_cnn_model.linear.weight.data
            #     bias = self.profile_cnn_model.linear.bias.data
            #     self.profile_cnn_model.linear = Conv1dWrapper(
            #         weight.shape[1], weight.shape[0], 1
            #     )
            #     self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            #     self.profile_cnn_model.linear.conv.bias.data = bias
            self.profile_cnn_model.linear = conv1d_lora(
                layer=self.profile_cnn_model.linear,
                r=1,
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

    def return_origin(self, requires_grad: bool = True):
        """
        Returns a clone of the model with original layers.

        Parameters
        ----------
            requires_grad: Whether to require gradients.

        Returns
        -------
            scFootprintBPNet: A clone of the model with original layers.
        """
        self = self.to("cpu")
        if isinstance(self.profile_cnn_model.linear, nn.Linear):
            raise NotImplementedError
            # print("translating linear into conv1d")
            # weight = self.profile_cnn_model.linear.weight.data
            # print(weight.shape)
            # bias = self.profile_cnn_model.linear.bias.data
            # self.profile_cnn_model.linear = Conv1dWrapper(
            #     weight.shape[1], weight.shape[0], 1
            # )
            # print(self.profile_cnn_model.linear.conv.weight.shape)
            # self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            # self.profile_cnn_model.linear.conv.bias.data = bias

        model_clone = deepcopy(self)
        # DNA Input CNN
        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = model_clone.dna_cnn_model.conv.layer
        # Hidden Layer Dialated CNN
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
        # Output Head
        if not isinstance(model_clone.profile_cnn_model.conv_layer, Conv1dWrapper):
            model_clone.profile_cnn_model.conv_layer = (
                model_clone.profile_cnn_model.conv_layer.layer
            )
        if not isinstance(model_clone.profile_cnn_model.linear, Conv1dWrapper):
            model_clone.profile_cnn_model.linear = (
                model_clone.profile_cnn_model.linear.layer
            )
        if requires_grad:
            for p in model_clone.parameters():
                p.requires_grad = True
        return model_clone

    def collapse(self, cell_embedding=None, region_embedding=None, requires_grad=True):
        """
        Returns a clone of the model with collapsed layers.

        Parameters
        ----------
            cell_embedding: The cell embedding tensor.
            region_embedding: The region embedding tensor.
            requires_grad: Whether to require gradients.

        Returns
        -------
            scFootprintBPNet: A clone of the model with collapsed layers.
        """
        A_embeddings = self.A_embedding_process(cell_embedding, region_embedding)
        B_embeddings = self.B_embedding_process(cell_embedding, region_embedding)

        if isinstance(self.profile_cnn_model.linear, nn.Linear):
            raise NotImplementedError
            # print("translating linear into conv1d")
            # weight = self.profile_cnn_model.linear.weight.data
            # print(weight.shape)
            # bias = self.profile_cnn_model.linear.bias.data
            # self.profile_cnn_model.linear = Conv1dWrapper(
            #     weight.shape[1], weight.shape[0], 1
            # )
            # print(self.profile_cnn_model.linear.conv.weight.shape)
            # self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            # self.profile_cnn_model.linear.conv.bias.data = bias

        model_clone = deepcopy(self)
        # DNA Input CNN
        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = (
                model_clone.dna_cnn_model.conv.collapse_layer(
                    A_embeddings, B_embeddings
                )
            )
        # Hidden Layer Dialated CNN
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
        # Output Head
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
        if requires_grad:
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
