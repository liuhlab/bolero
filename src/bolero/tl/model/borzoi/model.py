import torch
from torch import nn

from .module import (
    ConvBlock,
    ConvDna,
    SequentialwithArgs,
    TargetLengthCrop,
    TransformerLayer,
)


class Borzoi(nn.Module):
    default_config = {
        "checkpoint_path": None,
    }

    def __init__(
        self,
        checkpoint_path,
    ):
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
            self._max_pool,
            ConvBlock(in_channels=608, out_channels=736, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=736, out_channels=896, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=896, out_channels=1056, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=1056, out_channels=1280, kernel_size=5),
        )

        # ==================
        # UNet connections 1
        # ==================
        self.unet1 = SequentialwithArgs(
            self._max_pool,
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
                attn_dropout=0.05,
                pos_dropout=0.01,
                dropout=0.2,
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
            self.upsample,
        )
        self.separable1 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=3, conv_type="separable"
        )

        self.upsampling_unet0 = SequentialwithArgs(
            ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1),
            self.upsample,
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
            nn.Dropout(0.1),
            nn.GELU(approximate="tanh"),
        )

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x, *args, **kwargs):
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
        x = self.crop(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.final_joined_convs(x, *args, **kwargs)
        return x


class BorzoiWithOutputHead(Borzoi):
    def __init__(self, checkpoint_path=None, enable_mouse_head=True):
        super().__init__(checkpoint_path=None)

        self.human_head = nn.Conv1d(in_channels=1920, out_channels=7611, kernel_size=1)

        self.enable_mouse_head = enable_mouse_head
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(
                in_channels=1920, out_channels=2608, kernel_size=1
            )
        self.final_softplus = nn.Softplus()

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x, *args, **kwargs):
        """Borzoi forward pass to get human and mouse output."""
        x = super().forward(x, *args, **kwargs)

        human_out = self.final_softplus(self.human_head(x))
        # human_out: (bs, 7611, 16352)
        # equivalent to 16352*32 = 523264 bp signal

        if self.enable_mouse_head:
            mouse_out = self.final_softplus(self.mouse_head(x))
            # mouse_out: (bs, 2608, 16352)
            # equivalent to 16352*32 = 523264 bp signal
            return human_out, mouse_out
        else:
            return human_out
