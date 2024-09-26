from copy import deepcopy

import torch
from torch import nn
from torchinfo import summary

from bolero.utils import validate_config

from .metrics import poisson_multinomial
from .module import (
    ConvBlock,
    ConvDna,
    SequentialwithArgs,
    TargetLengthCrop,
    TransformerLayer,
)


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
    ):
        """Initialize Borzoi model."""
        super().__init__()

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
        self.crop = TargetLengthCrop(16384 - 32)
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

    def forward(self, x, crop=True, *args, **kwargs):
        """Borzoi forward pass."""
        # change dtype to half if not already
        if torch.is_autocast_enabled():
            if x.dtype != torch.float16:
                x = x.half()
        else:
            if x.dtype != torch.float32:
                x = x.float()

        # =================
        # DNA Convolution
        # In - x: (bs, 4, 524288)
        # Out - x: (bs, 512, 262144)
        # signal resolution is 1
        # =================
        x = self.conv_dna(x, *args, **kwargs)

        # =================
        # Residual Tower (7 blocks)
        # In - x: (bs, 512, 262144)
        # Out - x_unet0: (bs, 1536, 16384)
        #       x_unet1: (bs, 1536, 8192)
        #       x: (bs, 1536, 4096)
        # signal resolution is 128
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
        # Out - x: (bs, 1920, 16352)
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

    def loss(self, y_pred, y_true):
        """
        Compute the loss for the Borzoi model.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values, shape (batch_size, out_channels, seq_len).
        y_true : torch.Tensor
            True values, shape (batch_size, out_channels, seq_len).
        """
        y_true_crop = self.crop(y_true)

        _loss = poisson_multinomial(
            y_true=y_true_crop,
            y_pred=y_pred,
            total_weight=self.loss_total_weight,
            weight_range=1,  # 1 means not use the position weighted loss
            weight_exp=4,
            epsilon=1e-7,  # this is smallest for float16
            rescale=False,
        )

        # loss is averaged across batch and channels
        _loss = _loss.mean()
        return _loss


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
