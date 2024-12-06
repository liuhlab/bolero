"""
seq2PRINT / seq2PRINTLoRA model

This model composed of three parts
1. DNA CNN Model:
Description: A convolutional neural network model for extracting motif level features from DNA sequences.
Input Shape: (batch_size, dna_channels, dna_len), default (64, 4, 1840)
Output Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)

2. Hidden Layer Model:
Description: A multi-layer residual connect dilated CNN model for encoding sequence information, receptive field is 256.
Layers: 8 layers of ConvBlocks, each block contains a dialated, grouped CNN layer and a pointwise feedforward CNN layer; connected by residual connections.
Input Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)
Output Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)

3. Profile CNN Model:
Description: A output model for predicting multi-mode footprint (mode as channels) or coverage signals.
Coverage head is detached from the main chunk and updated separately during training.
Input Shape: (batch_size, n_filters, dna_len), default (64, 1024, 1840)
Output Shape:
    - Footprint, (batch_size, n_modes, output_len), default (64, 99, 800)
    - Coverage (coverage count of the region), (batch_size, 1), default (64, 1)

Losses:
- Footprint: Mean Squared Error Loss (MSE) on footprint unprocessed z-scores.
- Coverage: Poisson Negative Log Likelihood Loss (Poisson NLL) on raw coverage.
"""

from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary

from bolero.tl.generic.module import DNA_CNN, DilatedCNN
from bolero.tl.generic.module_embedding import KVBottleNeckMixin
from bolero.tl.generic.module_lora_cond import (
    collapse_lora_model_,
    convert_to_conditional_lora_model,
)
from bolero.tl.model.scprinter.module import CoverageHead, FootprintsHead
from bolero.utils import validate_config


class seq2PRINT(nn.Module):
    """seq2PRINT base model."""

    default_config = {
        "n_filters": 1024,
        "dna_kernel_size": 21,
        "in_channels": 4,
        "n_blocks": 8,
        "dia_kernel_size": 3,
        "groups": 8,
        "output_kernel_size": 1,
        "output_scales": 99,
        "dna_len": 1840,
        "output_len": 800,
    }

    @classmethod
    def get_default_config(cls) -> dict:
        """Get default configuration combined from dataset, model and trainer."""
        return deepcopy(cls.default_config)

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        default_config = cls.get_default_config()
        config = {k: v for k, v in config.items() if k in default_config}
        validate_config(config, default_config)
        return cls(**config)

    def __init__(
        self,
        n_filters=1024,
        dna_kernel_size=21,
        in_channels=4,
        n_blocks=8,
        dia_kernel_size=3,
        groups=8,
        dna_len=1840,
        output_len=800,
        output_kernel_size=1,
        output_scales=99,
    ):
        # ===============
        # Initialize the model
        # ===============
        super().__init__()

        activation = nn.GELU()

        self.dna_cnn_model = DNA_CNN(
            n_filters=n_filters,
            dna_kernel_size=dna_kernel_size,
            activation=activation,
            in_channels=in_channels,
        )

        self.hidden_layer_model = DilatedCNN(
            n_filters=n_filters,
            bottleneck=n_filters,
            n_blocks=n_blocks,
            dia_kernel_size=dia_kernel_size,
            groups=groups,
            activation=activation,
            batch_norm=True,
            batch_norm_momentum=0.1,
            dilation_func=None,
            bipass_connect=False,
        )
        self.footprint_head = FootprintsHead(
            n_filters=n_filters,
            output_kernel_size=output_kernel_size,
            output_scales=output_scales,
        )
        self.coverage_head = CoverageHead(n_filters=n_filters)

        self.dna_len = dna_len
        self.output_len = output_len
        return

    def check_input_dtype(self, X):
        """Check the input dtype and convert to float32 or float16."""
        # change dtype to half if not already
        if torch.is_autocast_enabled():
            if X.dtype != torch.float16:
                X = X.half()
        else:
            if X.dtype != torch.float32:
                X = X.float()
        return X

    def forward(self, X, *args, output_len=None, **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
            X: The input tensor.
            output_len: The length of the output.
            args, kwargs: placeholder parameters for conditional LoRA layers.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        X = self.check_input_dtype(X)

        if output_len is None:
            output_len = self.output_len

        # get the motifs
        X = self.dna_cnn_model(X, *args, **kwargs)

        # get the hidden layer
        X = self.hidden_layer_model(X, *args, **kwargs)

        # get the profile
        fp_score = self.footprint_head(X, *args, output_len=output_len, **kwargs)
        coverage = self.coverage_head(X, *args, **kwargs)
        return fp_score, coverage

    @staticmethod
    def footprint_loss(y_pred, y_true):
        """Mean squared error loss for footprint."""
        # footprint contains nan values, remove them when calculating loss
        mask = ~torch.isnan(y_true)
        loss = F.mse_loss(y_pred[mask], y_true[mask])
        return loss

    @staticmethod
    def coverage_loss(y_pred, y_true):
        """Poisson loss for coverage."""
        # Poisson loss for coverage
        # full=True has no effect on gradient, but make the loss value positive so print nicer...
        loss = F.poisson_nll_loss(
            y_pred, y_true, log_input=False, reduction="mean", full=True
        )
        return loss

    def loss(self, y_footprint, y_coverage, pred_footprint, pred_coverage):
        """Compute the loss."""
        fp_loss = self.footprint_loss(y_pred=pred_footprint, y_true=y_footprint)
        cov_loss = self.coverage_loss(y_pred=pred_coverage, y_true=y_coverage)
        return fp_loss, cov_loss

    def model_summary(
        self,
        row_settings=("var_names",),
        input_size=None,
        input_data=None,
        depth=4,
        cache_forward_pass=False,
        col_names=("num_params",),
    ):
        """Print model summary."""
        device = next(self.parameters()).device

        if (input_size is None) and (input_data is None):
            input_size = (1, 4, self.dna_len)

        s = summary(
            self,
            depth=depth,
            row_settings=row_settings,
            input_size=input_size,
            input_data=input_data,
            cache_forward_pass=cache_forward_pass,
            col_names=col_names,
            device=device,
        ).__repr__()
        return s

    def __repr__(self):
        return self.model_summary()

    def footprint_parameters(self):
        """Get the parameters for the chunk and footprint head."""
        for name, params in self.named_parameters():
            if "coverage_head" in name:
                continue
            yield params

    def coverage_parameters(self):
        """Get the parameters for the coverage head ONLY."""
        for name, params in self.named_parameters():
            if "coverage_head" in name:
                yield params


def make_lora_config(
    emb_input_features,
    hidden_dim,
    hidden_layers,
    lora_dropout,
):
    """Make LoRA configuration for the Borzoi model."""
    shared_config = {
        "emb_input_features": emb_input_features,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "output_layer_groups": 1,
        "convert_conv": True,
        "lora_dropout": lora_dropout,
    }
    lora_config = {
        "dna_cnn_model": {
            **shared_config,
            "lora_rank": 4,  # total rank 4 * 21
        },
        "hidden_layer_model": {
            **shared_config,
            "lora_rank": 32,  # total rank 32 * 3 conv1 + 32 * 1 conv2
        },
        "footprint_head": {
            **shared_config,
            "lora_rank": 32,  # total rank 32 * 1
        },
        "coverage_head": {
            **shared_config,
            "lora_rank": 1,  # total rank 1 * 1
        },
    }
    return lora_config


class seq2PRINTLoRA(seq2PRINT, KVBottleNeckMixin):
    """scFootprintBPNetLoRA model."""

    default_config = seq2PRINT.get_default_config()

    default_config.update(
        {
            "base_model": "REQUIRED",
            # LoRA configuration
            "emb_input_features": "REQUIRED",
            "n_lora_layers": 1,
            "lora_hidden_dim": 384,
            "lora_dropout": 0,
            # KV Bottleneck
            "kv_bottleneck": None,
            "num_memories": 256,
            "dim_memory": 50,
            "num_memory_codebooks": 2,
            "additional_embs": 1,
            "emb_input": False,
        }
    )

    @classmethod
    def get_default_config(cls):
        """Get the default configuration for the model."""
        return deepcopy(cls.default_config)

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        validate_config(config, cls.get_default_config())
        config = {k: v for k, v in config.items() if k in cls.default_config}
        return cls(**config)

    def __init__(
        self,
        base_model: Union[str, seq2PRINT],
        # LoRA configuration
        emb_input_features: int,
        n_lora_layers: int = 1,
        lora_hidden_dim: Optional[int] = 384,
        lora_dropout: float = 0,
        # KV Bottleneck
        kv_bottleneck: str = "local",
        num_memories: int = 256,
        dim_memory: int = 50,
        num_memory_codebooks: int = 2,
        additional_embs: int = 1,
        emb_input: bool = True,
        **base_kwargs,
    ):
        # ===============
        # Initialize the model
        # ===============
        super().__init__(**base_kwargs)

        # load checkpoint or get state from pre-trained model
        if isinstance(base_model, seq2PRINT):
            self.load_state_dict(base_model.state_dict())
        elif base_model is None:
            raise ValueError("base_model is required.")
        else:
            checkpoint = torch.load(base_model, weights_only=False)
            if isinstance(checkpoint, dict):
                self.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.load_state_dict(checkpoint.state_dict())

        # key-value bottleneck for converting indices to embeddings
        if kv_bottleneck == "local":
            self.kv_bottleneck_mode = "local"
            lora_input_features = emb_input_features
        elif kv_bottleneck == "global":
            self.kv_bottleneck_mode = "global"
        elif kv_bottleneck is None:
            self.kv_bottleneck_mode = None
            lora_input_features = emb_input_features
        else:
            raise ValueError(
                f"kv_bottleneck value: {kv_bottleneck} is invalid, setting to None"
            )
        self.emb_input_features = emb_input_features
        self.num_memories = num_memories
        self.dim_memory = dim_memory
        self.num_memory_codebooks = num_memory_codebooks
        self.additional_embs = additional_embs
        self.emb_input = emb_input
        self.emb_input_dims = emb_input_features
        if self.kv_bottleneck_mode == "global":
            self.kv_bottleneck, lora_input_features = self.setup_kv_bottleneck(
                num_memory_codebooks=num_memory_codebooks,
                num_memories=num_memories,
                dim_memory=dim_memory,
                additional_embs=additional_embs,
                emb_input=emb_input,
                emb_input_dims=emb_input_features,
            )
            print(
                "Using global shared key-value bottleneck for converting indices to embeddings."
            )
        else:
            self.kv_bottleneck = None

        # LoRA configuration
        self.convert_to_lora(
            emb_input_features=lora_input_features,
            lora_hidden_dim=lora_hidden_dim,
            n_lora_layers=n_lora_layers,
            lora_dropout=lora_dropout,
        )
        return

    def convert_to_lora(
        self,
        emb_input_features,
        lora_hidden_dim,
        n_lora_layers,
        lora_dropout,
    ):
        """Convert the model to LoRA."""
        self.lora_config = make_lora_config(
            emb_input_features=emb_input_features,
            hidden_dim=lora_hidden_dim,
            hidden_layers=n_lora_layers,
            lora_dropout=lora_dropout,
        )

        for module_names, config in self.lora_config.items():
            if isinstance(module_names, str):
                module_names = (module_names,)

            if self.kv_bottleneck_mode == "local":
                config["kv_bottleneck"] = True
                config["num_memories"] = self.num_memories
                config["dim_memory"] = self.dim_memory
                config["num_memory_codebooks"] = self.num_memory_codebooks
                config["additional_embs"] = self.additional_embs
                config["emb_input"] = self.emb_input
                config["emb_input_dims"] = self.emb_input_dims
                # use layer norm
                config["norm_type"] = "layer"

            for module_name in module_names:
                module = getattr(self, module_name)
                module = convert_to_conditional_lora_model(module, **config)
                setattr(self, module_name, module)

        # also make sure kv_bottleneck is trainable
        for name, param in self.named_parameters():
            if "kv_bottleneck" in name:
                param.requires_grad = True
        return

    def collapse(self, embedding=None, requires_grad=True):
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
        # process the embeddings if kv_bottleneck is used
        if self.kv_bottleneck is not None:
            embedding = self.vq_ind_to_emb(embedding)

        model_clone = deepcopy(self)
        model_clone = collapse_lora_model_(model_clone, embedding=embedding)

        if requires_grad:
            for p in model_clone.parameters():
                p.requires_grad = True
        return model_clone

    def model_summary(self):
        """Print model summary."""
        if self.emb_input:
            emb_example = torch.ones(1, self.emb_input_dims)
        else:
            emb_example = torch.ones(1, self.emb_input_features)
        input_data = {
            "X": torch.ones(1, 4, self.dna_len),
            "embedding": emb_example,
        }
        return super().model_summary(input_data=input_data)

    def forward(self, X, embedding=None, output_len=None):
        """
        Forward pass of the model.

        Parameters
        ----------
            X: The input tensor.
            embedding: The embedding tensor.
            output_len: The length of the output.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        X = self.check_input_dtype(X)

        if output_len is None:
            output_len = self.output_len

        # process the embedding if kv_bottleneck is used
        if self.kv_bottleneck is not None:
            embedding = self.vq_ind_to_emb(embedding)

        # get the motifs
        X = self.dna_cnn_model(X, embedding=embedding)

        # get the hidden layer
        X = self.hidden_layer_model(X, embedding=embedding)

        # get the profile
        fp_score = self.footprint_head(
            X,
            embedding=embedding,
            output_len=output_len,
        )
        coverage = self.coverage_head(X, embedding=embedding)
        return fp_score, coverage
