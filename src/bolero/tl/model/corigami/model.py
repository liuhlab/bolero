import torch
import torch.nn as nn

import bolero.tl.model.corigami.module as blocks


class ConvModel(nn.Module):
    """
    This is the same as C.Origami's ConvModel (https://github.com/tanjimin/C.Origami/blob/main/src/corigami/model/corigami_models.py)

    Parameters
    ----------
    encoder_in_channel : int
        The number of input channels for encoder. Default is 5.
    encoder_num_epi : int
        The number of epigenomic features. Default is 2.
    encoder_output_channel : int
        The number of output channels for encoder. Default is 256.
    encoder_filter_size : int
        The size of the filter for encoder. Default is 5.
    encoder_num_blocks : int
        The number of blocks for encoder. Default is 12.
    decoder_mid_hidden : int
        The number of hidden channels for decoder. Default is 256.
    decoder_filter_size : int
        The size of the filter for decoder. Default is 3.
    decoder_num_blocks : int
        The number of blocks for decoder. Default is 5.

    Attributes
    ----------
    encoder : Encoder
        The encoder module.
    decoder : Decoder
        The decoder module.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    """

    default_config = {
        "encoder_in_channel": 5,
        "encoder_num_epi": 2,
        "encoder_output_channel": 256,
        "encoder_filter_size": 5,
        "encoder_num_blocks": 12,
        "decoder_mid_hidden": 256,
        "decoder_filter_size": 3,
        "decoder_num_blocks": 5,
        "image_scale": 256,
    }

    @classmethod
    def get_default_config(cls):
        """Get the default configuration for the model."""
        return cls.default_config

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        encoder = blocks.EncoderSplit(
            in_channel=config["encoder_in_channel"],
            num_epi=config["encoder_num_epi"],
            output_size=config["encoder_output_channel"],
            filter_size=config["encoder_filter_size"],
            num_blocks=config["encoder_num_blocks"],
        )
        decoder = blocks.Decoder(
            in_channel=config["encoder_output_channel"] * 2,
            hidden=config["decoder_mid_hidden"],
            filter_size=config["decoder_filter_size"],
            num_blocks=config["decoder_num_blocks"],
        )
        image_scale = config["image_scale"]
        return cls(encoder=encoder, decoder=decoder, image_scale=image_scale)

    def __init__(
        self,
        encoder: blocks.Encoder = None,
        decoder: blocks.Decoder = None,
        image_scale: int = 256,
    ):
        super().__init__()
        print("Initializing ConvModel")
        self.encoder = encoder
        self.decoder = decoder
        self.image_scale = image_scale

    def forward(self, x):
        """
        Input feature:
        batch_size, length * res, feature_dim
        """
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        """
        input dim:
        bs, seq_len, feat
        to:
        bs, feat, seq_len
        """
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        """
        concatenates each position in the N bins to every other position to form a N-by-N interaction map.
        input dim:
        bs, feat, seq_len
        to:
        bs, feat*2, seq_len, seq_len
        """
        x_i = x.unsqueeze(2).repeat(1, 1, self.image_scale, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, self.image_scale)
        input_map = torch.cat([x_i, x_j], dim=1)
        return input_map


class ConvModelSeqOnly(ConvModel):
    """
    This class adapts from C.Origami's ConvModel (https://github.com/tanjimin/C.Origami/blob/main/src/corigami/model/corigami_models.py)
    The difference is it only takes in DNA sequence data.
    It's used to predict the 3D genome structure from DNA sequence data.
    """

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        encoder = blocks.Encoder(
            in_channel=config["encoder_in_channel"],
            output_size=config["encoder_output_channel"],
            filter_size=config["encoder_filter_size"],
            num_blocks=config["encoder_num_blocks"],
        )
        decoder = blocks.Decoder(
            in_channel=config["encoder_output_channel"] * 2,
            hidden=config["decoder_mid_hidden"],
            filter_size=config["decoder_filter_size"],
            num_blocks=config["decoder_num_blocks"],
        )
        output_length = config["output_length"]
        return cls(encoder=encoder, decoder=decoder, output_length=output_length)


class ConvTransModel(ConvModel):
    """
    This class is the same as the C.Origami's ConvTransModel (https://github.com/tanjimin/C.Origami/blob/main/src/corigami/model/corigami_models.py)

    Parameters
    ----------
    encoder_in_channel : int
        The number of input channels for encoder. Default is 5.
    encoder_num_epi: int
        The number of epigenomic features. Default is 2.
    encoder_output_channel : int
        The number of output channels for decoder. Default is 256.
    encoder_filter_size : int
        The size of the filter for encoder. Default is 5.
    encoder_num_blocks : int
        The number of blocks for encoder. Default is 12.
    attn_layers : int
        The number of layers for attention module. Default is 8.
    record_attn : bool
        Whether to record the attention weights. Default is False.
    decoder_mid_hidden : int
        The number of hidden channels for decoder. Default is 256.
    decoder_filter_size : int
        The size of the filter for decoder. Default is 3.
    decoder_num_blocks : int
        The number of blocks for decoder. Default is 5.
    image_scale: int
        The dimension of the hic-image. Default is 256

    Attributes
    ----------
    encoder : Encoder
        The encoder module.
    attn : AttnModule
        The attention module.
    decoder : Decoder
        The decoder module.
    record_attn : bool
        Whether to record the attention weights.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    """

    default_config = {
        "encoder_in_channel": 5,
        "encoder_num_epi": 2,
        "encoder_output_channel": 256,
        "encoder_filter_size": 5,
        "encoder_num_blocks": 12,
        "attn_layers": 8,
        "record_attn": False,
        "decoder_mid_hidden": 256,
        "decoder_filter_size": 3,
        "decoder_num_blocks": 5,
        "image_scale": 256,
    }

    @classmethod
    def get_default_config(cls):
        """Get the default configuration for the model."""
        return cls.default_config

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        encoder = blocks.EncoderSplit(
            in_channel=config["encoder_in_channel"],
            num_epi=config["encoder_num_epi"],
            output_size=config["encoder_output_channel"],
            filter_size=config["encoder_filter_size"],
            num_blocks=config["encoder_num_blocks"],
        )
        attn = blocks.AttnModule(
            hidden=config["encoder_output_channel"],
            layers=config["attn_layers"],
            record_attn=config["record_attn"],
            input_dim=config["image_scale"],
        )
        decoder = blocks.Decoder(
            in_channel=config["encoder_output_channel"] * 2,
            hidden=config["decoder_mid_hidden"],
            filter_size=config["decoder_filter_size"],
            num_blocks=config["decoder_num_blocks"],
        )
        record_attn = config["record_attn"]
        image_scale = config["image_scale"]
        return cls(
            encoder=encoder,
            attn=attn,
            decoder=decoder,
            record_attn=record_attn,
            image_scale=image_scale,
        )

    def __init__(self, encoder, attn, decoder, record_attn=False, image_scale=256):
        super().__init__()
        print("Initializing ConvTransModel")
        self.encoder = encoder
        self.attn = attn
        self.decoder = decoder
        self.record_attn = record_attn
        self.image_scale = image_scale

    def trim_encoder_output(self, x):
        """
        input dim:
        bs, feature, seq_len
        to:
        bs, feature, image_scale
        """
        radius = (x.shape[-1] - self.image_scale) // 2
        return x[:, :, radius:-radius]

    def forward(self, x, *args, **kwargs):
        """
        Input feature:
        batch_size, feature_dim, length
        """
        x = self.encoder(x, *args, **kwargs)
        if x.shape[-1] > self.image_scale:
            x = self.trim_encoder_output(x)
        x = self.move_feature_forward(x)
        if self.record_attn:
            x, attn_weights = self.attn(x, *args, **kwargs)
        else:
            x = self.attn(x, *args, **kwargs)
        x = self.move_feature_forward(x)
        x = self.diagonalize(x)
        x = self.decoder(x, *args, **kwargs).squeeze(1)
        if self.record_attn:
            return x, attn_weights
        else:
            return x


class ConvTransModelSeqOnly(ConvTransModel):
    """
    This class adapts from C.Origami's ConvTransModel (https://github.com/tanjimin/C.Origami/blob/main/src/corigami/model/corigami_models.py)
    The difference is it only takes in DNA sequence data.
    It's used to predict the 3D genome structure from DNA sequence data.
    """

    @classmethod
    def create_from_config(cls, config: dict):
        """Create the model from a configuration dictionary."""
        encoder = blocks.Encoder(
            in_channel=config["encoder_in_channel"],
            output_size=config["encoder_output_channel"],
            filter_size=config["encoder_filter_size"],
            num_blocks=config["encoder_num_blocks"],
        )
        attn = blocks.AttnModule(
            hidden=config["encoder_output_channel"],
            layers=config["attn_layers"],
            record_attn=config["record_attn"],
        )
        decoder = blocks.Decoder(
            in_channel=config["encoder_output_channel"] * 2,
            hidden=config["decoder_mid_hidden"],
            filter_size=config["decoder_filter_size"],
            num_blocks=config["decoder_num_blocks"],
        )
        record_attn = config["record_attn"]
        image_scale = config["image_scale"]
        return cls(
            encoder=encoder,
            attn=attn,
            decoder=decoder,
            record_attn=record_attn,
            image_scale=image_scale,
        )
