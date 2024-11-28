import math
from copy import deepcopy

import torch
import torch.nn as nn

from bolero.tl.generic.module_lora_cond import ConditionalLoRALayer
from bolero.tl.model.corigami.model import model_summary
from bolero.tl.model.corigami.module import AttnModule, ConvBlock, Decoder
from bolero.utils import validate_config

# DNA input: (bs, ch, seq_len) (bs, 1920, 16384), pos resolution 32bp
# ATAC input: (bs, ch, seq_len) (bs, 1, 16384), pos resolution 32bp


def exponential_linspace_int(start, end, num, divisible_by=1):
    """Create a linspace in exponential space, rounded to the nearest divisible_by."""

    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


class Encoder(nn.Module):
    def __init__(self, in_channel=1920, num_epi=1, output_channel=256, kernel_size=5):
        """
        Initialize the EncoderSplit
        """
        super().__init__()
        self.kernel_size = kernel_size

        self.conv_start_seq = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 3, 2, 1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(),
        )
        self.conv_start_epi = nn.Sequential(
            nn.Conv1d(num_epi, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        dna_channels = exponential_linspace_int(1920, 128, 8, 8)
        epi_channels = exponential_linspace_int(16, 128, 8, 1)
        self.res_blocks_seq = self.get_res_blocks(dna_channels)
        self.res_blocks_epi = self.get_res_blocks(epi_channels)
        self.conv_end = nn.Conv1d(256, output_channel, 1)
        self.in_channel = in_channel
        self.num_epi = num_epi

    def forward(self, x, *args, **kwargs):
        """
        Concatenate the dna seq with epigenomic features
        """
        seq = x[:, : self.in_channel, :]
        epi = x[:, self.in_channel :, :]

        for module in self.conv_start_seq:
            if isinstance(module, ConditionalLoRALayer):
                seq = module(seq, *args, **kwargs)
            else:
                seq = module(seq)

        for module in self.conv_start_epi:
            if isinstance(module, ConditionalLoRALayer):
                epi = module(epi, *args, **kwargs)
            else:
                epi = module(epi)

        for block in self.res_blocks_seq:
            seq = block(seq, *args, **kwargs)

        for block in self.res_blocks_epi:
            epi = block(epi, *args, **kwargs)

        x = torch.cat([seq, epi], dim=1)

        if isinstance(self.conv_end, ConditionalLoRALayer):
            out = self.conv_end(x, *args, **kwargs)
        else:
            out = self.conv_end(x)
        return out

    def get_res_blocks(self, channels):
        """
        Get the residual blocks
        """
        blocks = []
        for idx in range(len(channels) - 1):
            in_channels = channels[idx]
            out_channels = channels[idx + 1]
            blocks.append(
                ConvBlock(self.kernel_size, hidden_in=in_channels, hidden=out_channels)
            )
        res_blocks = nn.Sequential(*blocks)
        return res_blocks


class Corigami(nn.Module):
    default_config = {
        "in_channel": 1920,
        "seq_len": 16384,
        "output_channel": 256,
        "image_scale": 64,
        "dig_pw_mode": "concat",
    }

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        default_config = cls.get_default_config()
        config = {k: v for k, v in config.items() if k in default_config}
        validate_config(config, default_config)
        return cls(**config)

    @classmethod
    def get_default_config(cls):
        """Get default config."""
        return deepcopy(cls.default_config)

    def __init__(
        self,
        in_channel=1920,
        num_epi=1,
        seq_len=16384,
        output_channel=256,
        encoder_kernel_size=5,
        decoder_kernel_size=3,
        decoder_num_blocks=5,
        attn_layers=8,
        image_scale=64,
        dig_pw_mode="concat",
    ):
        """
        Initialize the Corigami
        """
        super().__init__()
        self.encoder = Encoder(
            in_channel=in_channel,
            num_epi=num_epi,
            output_channel=output_channel,
            kernel_size=encoder_kernel_size,
        )
        self.attn = AttnModule(
            hidden=output_channel,
            layers=attn_layers,
            record_attn=False,
            input_dim=image_scale,
        )
        additional_decoder_input = 1 if dig_pw_mode == "concat" else 0
        self.decoder = Decoder(
            in_channel=output_channel * 2 + additional_decoder_input,
            hidden=output_channel,
            filter_size=decoder_kernel_size,
            num_blocks=decoder_num_blocks,
        )
        self.image_scale = image_scale
        self.seq_len = seq_len
        self.dig_pw_mode = dig_pw_mode

    def move_feature_forward(self, x):
        """
        input dim:
        bs, seq_len, feat
        to:
        bs, feat, seq_len
        """
        return x.transpose(1, 2).contiguous()

    def make_off_diagnal_position_weight(self, d, reverse_comp=False):
        """
        Make the off-diagnal position weight matrix
        The value on N-th diagnal is log1p(N - d), where d is always > 0

        return shape (image_scale, image_scale)
        """
        indices = torch.arange(self.image_scale)
        d = torch.abs(torch.Tensor([d]))
        pw = torch.log1p(torch.abs(indices[:, None] - indices - d))
        if reverse_comp:
            pw = pw.flip(0).flip(1)
        return pw

    def diagonalize(self, x, x2=None, d=0, reverse_comp=None):
        """
        concatenates each position in the N bins to every other position to form a N-by-N interaction map.
        input dim:
        bs, feat, image_scale
        to:
        bs, feat*2, image_scale, image_scale
        """
        x_i = x.unsqueeze(2).repeat(1, 1, self.image_scale, 1)
        # x_i shape: bs, feat, image_scale, image_scale
        if x2 is None:
            x2 = x
        x_j = x2.unsqueeze(3).repeat(1, 1, 1, self.image_scale)
        # x_j shape: bs, feat, image_scale, image_scale

        # position weight, (1, 1, image_scale, image_scale)
        if isinstance(d, int) and d == 0:
            pw = (
                self.make_off_diagnal_position_weight(d=0, reverse_comp=False)
                .view(1, 1, self.image_scale, self.image_scale)
                .to(x.device)
            )
            pw = torch.concat([pw] * x_i.shape[0], dim=0)
        else:
            assert x2 is not None, "x2 must be provided when d != 0"
            assert reverse_comp is not None, "reverse_comp must be provided when d != 0"

            pw_all = []
            for d_i, reverse_comp_i in zip(d, reverse_comp):
                pw = (
                    self.make_off_diagnal_position_weight(
                        d=d_i.item(), reverse_comp=reverse_comp_i.item()
                    )
                    .view(1, 1, self.image_scale, self.image_scale)
                    .to(x.device)
                )
                pw_all.append(pw)
            pw = torch.concat(pw_all, dim=0)

        # Two potential ways to incorporate the position weight
        # 1. concatenate the position weight to the input map
        # 2. add the position weight to the input map
        # here I try the concatenation way by default

        if self.dig_pw_mode == "concat":
            input_map = torch.cat([x_i, x_j, pw], dim=1)
        elif self.dig_pw_mode == "add":
            input_map = torch.cat([x_i, x_j], dim=1) + pw
        elif self.dig_pw_mode is None:
            input_map = torch.cat([x_i, x_j], dim=1)
        else:
            raise ValueError("Invalid pw_mode")

        return input_map

    def _encoder_and_attn(self, x, *args, **kwargs):
        x = self.encoder(x, *args, **kwargs)
        x = self.move_feature_forward(x)
        x = self.attn(x, *args, **kwargs)
        x = self.move_feature_forward(x)
        return x

    def forward(
        self,
        x,
        return_corigami_embedding=False,
        *args,
        **kwargs,
    ):
        """
        Input feature:
        batch_size, feature_dim, length
        """
        x_emb = self._encoder_and_attn(x, *args, **kwargs)

        decoder_input = self.diagonalize(x=x_emb, x2=None, d=0, reverse_comp=False)
        final_output = self.decoder(decoder_input, *args, **kwargs).squeeze(1)

        if return_corigami_embedding:
            # output_shape: (bs, image_scale, image_scale),
            # emb_shape: (bs, feat, image_scale)
            return final_output, x_emb

        # output_shape: (bs, image_scale, image_scale)
        return final_output

    def forward_from_hic_emb(
        self, x_emb, x2_emb=None, d=0, reverse_comp=False, *args, **kwargs
    ):
        """
        Forward from the embedding
        """
        decoder_input = self.diagonalize(
            x=x_emb, x2=x2_emb, d=d, reverse_comp=reverse_comp
        )
        final_output = self.decoder(decoder_input, *args, **kwargs).squeeze(1)
        # output_shape: (bs, image_scale, image_scale)
        return final_output

    def __repr__(self):
        input_channel = self.encoder.in_channel + self.encoder.num_epi
        return model_summary(self, input_size=(2, input_channel, self.seq_len))
