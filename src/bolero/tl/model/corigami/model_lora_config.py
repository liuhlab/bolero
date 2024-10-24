def make_classic_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.1,
    rank=4,
    alpha=1,
):
    """Make LoRA configuration for the Corigami model."""
    emb_input_features = None
    lora_config = {
        "encoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "verbose": True,
            # "include_cond_lora_patterns": [".+.scale.0"],
            "lora_dropout": lora_dropout,  # check this
            "alpha": alpha,
            "default_conditional": False,
        },
        "attn": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": False,
            # "exclude_cond_lora_patterns": [".+.self_attn"],
        },
        "decoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": False,
            # "include_cond_lora_patterns": [r"res_blocks.\d+.res.0"],
        },
    }
    return lora_config


def make_partial_conditional_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.1,
    rank=4,
    alpha=1,
):
    """Make conditional LoRA configuration for the Corigami model."""
    lora_config = {
        "encoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": 4,
            "verbose": True,
            "include_cond_lora_patterns": [r"res_blocks.+\d+."],
            "lora_dropout": lora_dropout,  # check this
            "alpha": 8,
            "default_conditional": False,
        },
        "attn": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": True,
            # "exclude_cond_lora_patterns": [".+.self_attn"],
        },
        "decoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": False,
            "include_cond_lora_patterns": [r"res_blocks.\d+.res"],
        },
    }
    return lora_config


def make_all_conditional_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.1,
    rank=4,
    alpha=1,
):
    """Make conditional LoRA configuration for the Corigami model."""
    lora_config = {
        "encoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": 4,
            "verbose": True,
            "lora_dropout": lora_dropout,  # check this
            "alpha": 8,
            "default_conditional": True,
        },
        "attn": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": True,
        },
        "decoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": True,
        },
    }
    return lora_config


def make_output_conditional_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.1,
    rank=4,
    alpha=1,
):
    """Make conditional LoRA configuration for the Corigami model."""
    lora_config = {
        "encoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": 4,
            "verbose": True,
            "lora_dropout": lora_dropout,  # check this
            "alpha": 8,
            "default_conditional": False,
        },
        "attn": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": False,
        },
        "decoder": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "convert_linear": True,
            "rank": rank,
            "lora_dropout": lora_dropout,
            "alpha": alpha,
            "default_conditional": True,
        },
    }
    return lora_config
