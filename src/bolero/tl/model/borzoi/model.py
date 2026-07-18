from copy import deepcopy

import torch
from torch import nn
from torchinfo import summary

from bolero.utils import validate_config

from .metrics import bce_loss, poisson_multinomial
from .module import (
    ConvBlock,
    ConvDna,
    SequentialwithArgs,
    TargetLengthCrop,
    TransformerLayer,
)
from .utils import clamp_sqrt_large_value

BORZOI_INPUT_LEN = 524288
BORZOI_OUTPUT_LEN = 16384


def model_summary(
    model,
    row_settings=("var_names",),
    input_size=None,
    input_data=None,
    depth=3,
    cache_forward_pass=True,
    col_names=("num_params",),
):
    """Print model summary."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        device = next(model.parameters()).device

        s = summary(
            model,
            depth=depth,
            row_settings=row_settings,
            input_size=input_size,
            input_data=input_data,
            cache_forward_pass=cache_forward_pass,
            col_names=col_names,
            device=device,
        ).__repr__()
    return s


class Borzoi(nn.Module):
    default_config = {
        "transformer_attn_dropout": 0.05,
        "transformer_pos_dropout": 0.01,
        "transformer_ff_dropout": 0.2,
        "final_conv_dropout": 0.1,
        "loss_total_weight": 0.16,
        "loss_chunks": 1,
        "power": None,
        "soft_clamp": None,
        "soft_clamp_bool": None,
        # borzoi crop to 6144 for loss
        # But I found 16352 or 6144 doesn't has impact on cell-type-specific model
        "crop_to_length": 16352,
        "flash_attn": True,
    }

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        default_config = cls.get_default_config()
        config = {k: v for k, v in config.items() if k in default_config}
        validate_config(config, default_config)
        return cls(**config)

    def __init__(
        self,
        transformer_attn_dropout=0.05,
        transformer_pos_dropout=0.01,
        transformer_ff_dropout=0.2,
        final_conv_dropout=0.1,
        loss_total_weight=0.16,
        loss_chunks=1,
        power=None,
        soft_clamp=None,
        soft_clamp_bool=None,
        crop_to_length=16352,
        flash_attn=True,
    ):
        """Initialize Borzoi model."""
        super().__init__()
        self.loss_total_weight = loss_total_weight
        self.loss_chunks = loss_chunks
        self.power = power
        self.soft_clamp = soft_clamp
        self.soft_clamp_bool = soft_clamp_bool
        self.crop_to_length = crop_to_length

        # =========
        # Conv DNA
        # =========
        self.conv_dna = ConvDna(
            in_channels=4,
            out_channels=512,
            dna_kernel_size=15,
        )
        self._max_pool = nn.MaxPool1d(kernel_size=2, padding=0)

        # ==============
        # Residual Tower
        # ==============
        self.res_tower = SequentialwithArgs(
            ConvBlock(in_channels=512, out_channels=608, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=608, out_channels=736, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=736, out_channels=896, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=896, out_channels=1056, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=1056, out_channels=1280, kernel_size=5),
        )

        # ==================
        # UNet connections 1
        # ==================
        self.unet1 = SequentialwithArgs(
            nn.MaxPool1d(kernel_size=2, padding=0),
            ConvBlock(in_channels=1280, out_channels=1536, kernel_size=5),
        )

        # ===========
        # Transformer
        # ===========
        transformer = [
            TransformerLayer(
                channels=1536,
                heads=8,
                dim_key=64,
                attn_dropout=transformer_attn_dropout,
                pos_dropout=transformer_pos_dropout,
                ff_dropout=transformer_ff_dropout,
                num_rel_pos_features=32,
                seq_len=4096,
                flash_attn=flash_attn,
            )
            for _ in range(8)
        ]
        self.transformer = SequentialwithArgs(*transformer)

        # ===========================
        # UNet horizontal connections
        # ===========================
        self.horizontal_conv0 = ConvBlock(
            in_channels=1280, out_channels=1536, kernel_size=1
        )
        self.horizontal_conv1 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=1
        )

        # ===================================
        # UNet upsampling and separable convs
        # ===================================
        self.upsample = torch.nn.Upsample(scale_factor=2)

        self.upsampling_unet1 = SequentialwithArgs(
            ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1),
            torch.nn.Upsample(scale_factor=2),
        )
        self.separable1 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=3, conv_type="separable"
        )

        self.upsampling_unet0 = SequentialwithArgs(
            ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1),
            torch.nn.Upsample(scale_factor=2),
        )
        self.separable0 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=3, conv_type="separable"
        )

        # ===================
        # Final Crop and Conv
        # ===================
        self.crop = TargetLengthCrop(self.crop_to_length)
        self.final_joined_convs = SequentialwithArgs(
            ConvBlock(in_channels=1536, out_channels=1920, kernel_size=1),
            nn.Dropout(final_conv_dropout),
            nn.GELU(approximate="tanh"),
        )

    @classmethod
    def get_default_config(cls):
        """Get default config."""
        return deepcopy(cls.default_config)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, weights_only=False):
        """Load model from checkpoint."""
        model = cls()

        model_weights = torch.load(checkpoint_path, weights_only=weights_only)
        model_weights = {
            k: v
            for k, v in model_weights.items()
            if k.split(".")[0] not in {"human_head", "mouse_head"}
        }
        if checkpoint_path is not None:
            model.load_state_dict(model_weights)
        return model

    def _autocast_dtype(self, x):
        if x is None:
            return x

        if torch.is_autocast_enabled():
            if x.dtype != torch.float16:
                x = x.half()
        else:
            if x.dtype != torch.float32:
                x = x.float()
        return x

    def forward(self, x, crop=True, *args, **kwargs):
        """Borzoi forward pass."""
        # change dtype to half if not already
        x = self._autocast_dtype(x)

        # =================
        # DNA Convolution
        # In - x: (bs, 4, 524288)
        # Out - x: (bs, 512, 262144)
        # signal resolution is 1
        # =================
        # pop gene mask from kwargs if exists, as it is only used in conv_dna
        gene_mask = kwargs.pop("gene_mask", None)
        x = self.conv_dna(x, gene_mask=gene_mask, **kwargs)

        # =================
        # Residual Tower (7 blocks)
        # In - x: (bs, 512, 262144)
        # Out - x_unet0: (bs, 1536, 16384)
        #       x_unet1: (bs, 1536, 8192)
        #       x: (bs, 1536, 4096)
        # signal resolution is 32
        # =================
        x_unet0 = self.res_tower(x, *args, **kwargs)
        x_unet1 = self.unet1(x_unet0, *args, **kwargs)
        x = self._max_pool(x_unet1)
        # horizontal convolution before unet connections
        # conv 1x1, 1280 -> 1536 channels
        x_unet0 = self.horizontal_conv0(x_unet0, *args, **kwargs)
        # conv 1x1, 1536 -> 1536 channels
        x_unet1 = self.horizontal_conv1(x_unet1, *args, **kwargs)

        # =================
        # Transformer
        # In - x: (bs, 1536, 4096)
        # Out - x: (bs, 1536, 4096)
        # signal resolution is 128
        # =================
        x = self.transformer(x.permute(0, 2, 1), *args, **kwargs)
        x = x.permute(0, 2, 1)

        # =================
        # UNet upsampling and separable convs 1
        # In - x: (bs, 1536, 4096)
        # Out - x: (bs, 1536, 8192)
        # signal resolution is 64
        # =================
        x = self.upsampling_unet1(x, *args, **kwargs)
        x += x_unet1
        x = self.separable1(x, *args, **kwargs)

        # =================
        # UNet upsampling and separable convs 0
        # In - x: (bs, 1536, 8192)
        # Out - x: (bs, 1536, 16384)
        # signal resolution is 32
        # =================
        x = self.upsampling_unet0(x, *args, **kwargs)
        x += x_unet0
        x = self.separable0(x, *args, **kwargs)

        # =================
        # Final Crop and Conv
        # In - x: (bs, 1536, 16384)
        # Out - x: (bs, 1920, 16352) if crop is True
        # Out - x: (bs, 1920, 16384) if crop is False
        # signal resolution is 32
        # =================
        x = self.final_joined_convs(x, *args, **kwargs)

        if crop:
            x = self.crop(x)
        return x

    def _model_summary(
        self,
        input_size=(1, 4, 524288),
        input_data=None,
        depth=3,
        col_names=("input_size", "output_size", "num_params"),
        cache_forward_pass=False,
        **kwargs,
    ):
        summary_str = model_summary(
            self,
            input_size=input_size,
            input_data=input_data,
            depth=depth,
            col_names=col_names,
            cache_forward_pass=cache_forward_pass,
            **kwargs,
        )
        return summary_str

    def __repr__(self):
        return self._model_summary()

    def _separate_bce_poisson_multinomial_loss(self, y_true: dict, y_pred: dict):
        """Separate loss for ATAC and methylation."""
        atac_y_true = y_true["atac"]
        mc_y_true = y_true["mc"]
        atac_y_pred = y_pred["atac"]
        mc_y_pred = y_pred["mc"]

        with torch.no_grad():
            if (self.soft_clamp is not None) and (self.power is not None):
                atac_y_true = clamp_sqrt_large_value(
                    atac_y_true,
                    power=self.power,
                    threshold=self.soft_clamp,
                    effective_bool=self.soft_clamp_bool,
                )

        loss_breakdown = {}

        # atac loss
        _poisson_multinomial_loss, _poisson_multinomial_loss_breakdown = (
            poisson_multinomial(
                y_true=atac_y_true,
                y_pred=atac_y_pred,
                total_weight=self.loss_total_weight,
                epsilon=1e-7,  # this is smallest for float16
                return_breakdown=True,
                loss_chunks=getattr(self, "loss_chunks", 1),
                position_weights=None,
            )
        )
        loss_breakdown.update(_poisson_multinomial_loss_breakdown)

        # methylation loss
        _bce_loss, _bce_loss_breakdown = bce_loss(
            mc_y_pred, mc_y_true, return_breakdown=True
        )
        loss_breakdown.update(_bce_loss_breakdown)

        # concat loss on channel axis
        _loss = torch.concat([_poisson_multinomial_loss, _bce_loss], dim=1)

        # processed y_true with atac and mC concatenated on channel axis
        processed_y_true = torch.concat([atac_y_true, mc_y_true], dim=1)
        return _loss, loss_breakdown, processed_y_true

    def loss(
        self,
        y_pred,
        y_true,
        reduce=True,
        position_weights=None,
        loss_type="poisson_multinomial",
    ):
        """
        Compute the loss for the Borzoi model.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values, shape (batch_size, out_channels, seq_len).
        y_true : torch.Tensor
            True values, shape (batch_size, out_channels, seq_len).
        reduce : bool
            Whether to reduce the loss to a scalar.
        position_weights : torch.Tensor, optional
            Per-position weights applied to the loss.
        loss_type : str
            Which loss to compute (e.g. ``"poisson_multinomial"``).
        """
        with torch.no_grad():
            if isinstance(y_true, dict):
                y_true = {k: self.crop(v) for k, v in y_true.items()}
            else:
                y_true = self.crop(y_true)

            if loss_type == "poisson_multinomial":
                if (self.soft_clamp is not None) and (self.power is not None):
                    y_true = clamp_sqrt_large_value(
                        y_true,
                        power=self.power,
                        threshold=self.soft_clamp,
                        effective_bool=self.soft_clamp_bool,
                    )
                if position_weights is not None:
                    position_weights = self.crop(position_weights)

        if loss_type == "poisson_multinomial":
            _loss, loss_breakdown = poisson_multinomial(
                y_true=y_true,
                y_pred=y_pred,
                total_weight=self.loss_total_weight,
                epsilon=1e-7,  # this is smallest for float16
                return_breakdown=True,
                loss_chunks=getattr(self, "loss_chunks", 1),
                position_weights=position_weights,
            )
        elif loss_type == "bce":
            _loss, loss_breakdown = bce_loss(y_pred, y_true, return_breakdown=True)
        elif loss_type == "separate_bce_poisson_multinomial":
            _loss, loss_breakdown, y_true = self._separate_bce_poisson_multinomial_loss(
                y_true, y_pred
            )
        else:
            raise ValueError(f"loss_type {loss_type} not recognized")

        # loss is averaged across batch and channels
        if reduce:
            _loss = _loss.mean()
            with torch.no_grad():
                loss_breakdown = {k: v.mean() for k, v in loss_breakdown.items()}

        # y_true here should be processed single tensor
        return _loss, loss_breakdown, y_true


class BorzoiWithOutputHead(Borzoi):
    """Borzoi model with bulk track output heads, from the original model."""

    def __init__(self, human_head=True, mouse_head=False):
        super().__init__()

        if not (human_head or mouse_head):
            raise ValueError("At least one of human_head or mouse_head should be True")

        self.enable_human_head = human_head
        if self.enable_human_head:
            self.human_head = nn.Conv1d(
                in_channels=1920, out_channels=7611, kernel_size=1
            )

        self.enable_mouse_head = mouse_head
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(
                in_channels=1920, out_channels=2608, kernel_size=1
            )
        self.final_softplus = nn.Softplus()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path, weights_only=False, human_head=True, mouse_head=False
    ):
        """Load model from checkpoint."""
        model = cls(human_head=human_head, mouse_head=mouse_head)
        if checkpoint_path is not None:
            model.load_state_dict(
                torch.load(checkpoint_path, weights_only=weights_only)
            )
        return model

    def forward(self, x, *args, **kwargs):
        """Borzoi forward pass to get human and mouse output."""
        x = super().forward(x, *args, **kwargs)

        output = []

        if self.enable_human_head:
            human_out = self.final_softplus(self.human_head(x))
            # human_out: (bs, 7611, 16352)
            # equivalent to 16352*32 = 523264 bp signal
            output.append(human_out)

        if self.enable_mouse_head:
            mouse_out = self.final_softplus(self.mouse_head(x))
            # mouse_out: (bs, 2608, 16352)
            # equivalent to 16352*32 = 523264 bp signal
            output.append(mouse_out)

        return output
