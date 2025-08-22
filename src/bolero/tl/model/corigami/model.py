import torch
import torch.nn as nn
from torchinfo import summary

import bolero.tl.model.corigami.module as blocks
from bolero.tl.generic.module_lora_cond import convert_to_conditional_lora_model
from bolero.tl.model.corigami.model_lora_config import (
    make_all_conditional_lora_config,
    make_classic_lora_config,
    make_output_conditional_lora_config,
    make_partial_conditional_lora_config,
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

    def __init__(
        self,
        encoder,
        attn,
        decoder,
        record_attn=False,
        image_scale=256,
    ):
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


class ConvTransModelLora(ConvTransModel):
    """
    This class adapts from C.Origami's ConvTransModel and added Lora
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
        "recalculated_embedding": None,
        # lora
        "out_channels": 1,
        "hidden_dim": 256,  # larger than cell type embedding
        "hidden_layers": 1,
        "lora_dropout": 0.01,
        "base_checkpoint_path": None,
        "loss_total_weight": 0.2,
        "rank": 8,
        "alpha": 16,
        "preset": "REQUIRED",
    }

    def make_lora_config(
        self,
        emb_input_features,
        preset,
        hidden_dim=256,
        hidden_layers=1,
        lora_dropout=0.01,
        rank=4,
        alpha=1,
    ):
        """Make LoRA configuration for the Corigami model."""
        kwargs = {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "lora_dropout": lora_dropout,
            "rank": rank,
            "alpha": alpha,
        }
        if preset == "all_conditional":
            lora_config = make_all_conditional_lora_config(**kwargs)
        elif preset == "classic":
            lora_config = make_classic_lora_config(**kwargs)
        elif preset == "partial_conditional":
            lora_config = make_partial_conditional_lora_config(**kwargs)
        elif preset == "output_conditional":
            lora_config = make_output_conditional_lora_config(**kwargs)
        else:
            raise ValueError(f"preset {preset} not recognized")
        return lora_config

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
        recalculated_embedding = (
            torch.tensor(config["recalculated_embedding"])
            if config["recalculated_embedding"] is not None
            else None
        )

        return cls(
            encoder=encoder,
            attn=attn,
            decoder=decoder,
            record_attn=record_attn,
            image_scale=image_scale,
            recalculated_embedding=recalculated_embedding,
        )

    def convert_to_lora(self, configs):
        """Convert the model to LoRA."""
        self.lora_config = self.make_lora_config(
            emb_input_features=self.recalculated_embedding.shape[1]
            if self.recalculated_embedding is not None
            else None,
            preset=configs["preset"],
            hidden_dim=configs["hidden_dim"],
            hidden_layers=configs["hidden_layers"],
            lora_dropout=configs["lora_dropout"],
            rank=configs["rank"],
            alpha=configs["alpha"],
        )
        for module_names, config in self.lora_config.items():
            if isinstance(module_names, str):
                module_names = (module_names,)

            for module_name in module_names:
                module = getattr(self, module_name)
                module = convert_to_conditional_lora_model(module, **config)
                setattr(self, module_name, module)
        return

    def init_embedding(self):
        """Initialize the cell type embedding."""
        if self.recalculated_embedding is None:
            raise ValueError("recalculated_embedding is required for conditional LoRA")
        self.cell_type_embedding = nn.Embedding(
            self.recalculated_embedding.shape[0], self.recalculated_embedding.shape[1]
        )  # N cell types, 50 dim embedding vector
        self.cell_type_embedding.weight.data = self.recalculated_embedding
        self.cell_type_embedding.weight.requires_grad = False
        return

    def __init__(
        self,
        encoder,
        attn,
        decoder,
        record_attn=False,
        image_scale=256,
        recalculated_embedding=None,
    ):
        super().__init__(encoder, attn, decoder, record_attn, image_scale)
        print("Initializing ConvTransModelLora")
        self.encoder = encoder
        self.attn = attn
        self.decoder = decoder
        self.record_attn = record_attn
        self.image_scale = image_scale
        self.recalculated_embedding = recalculated_embedding

    def forward(self, x, embedding):
        """
        Input feature:
        batch_size, feature_dim, length
        """
        # 1. if you don't have cell type embedding, you pass in the cell type emb vector which will be in shape of (batch_size, 512)
        # 2. if you have the cell type embedding layer, and you only pass in the cell type id, then you will need to put in (batch_size, 1)
        # I'm implement the second one here

        if (
            embedding is not None
            and embedding.shape[-1] == 1
            and self.recalculated_embedding is not None
        ):
            embedding = self.cell_type_embedding(embedding).view(
                embedding.shape[0], self.recalculated_embedding.shape[1]
            )  # (batch_size, 1) -> (batch_size, 50)

        x = self.encoder(x, embedding=embedding)
        if x.shape[-1] > self.image_scale:
            x = self.trim_encoder_output(x)
        x = self.move_feature_forward(x)
        if self.record_attn:
            x, attn_weights = self.attn(x, embedding=embedding)
        else:
            x = self.attn(x, embedding=embedding)
        x = self.move_feature_forward(x)
        x = self.diagonalize(x)
        x = self.decoder(x, embedding=embedding).squeeze(1)
        if self.record_attn:
            return x, attn_weights
        else:
            return x

    def _model_summary(
        self,
        input_data=None,
        depth=3,
        col_names=("input_size", "output_size", "num_params"),
        cache_forward_pass=False,
    ):
        if self.recalculated_embedding is None:
            emb_example = None
        else:
            emb_example = torch.randint(
                size=(1, 1), low=0, high=self.recalculated_embedding.shape[0]
            )
        if input_data is None:
            input_data = {
                "x": torch.ones(1, 6, 2097152),
                "embedding": emb_example,
            }
        summary_str = model_summary(
            self,
            input_size=None,
            input_data=input_data,
            depth=depth,
            col_names=col_names,
            cache_forward_pass=cache_forward_pass,
        )
        return summary_str
