def make_output_conditional_lora_config(
    emb_input_features,
    hidden_dim=512,
    hidden_layers=1,
    output_layer_groups=1,
    lora_dropout=0.01,
):
    """
    All layers using simple LoRA, except the final output head which uses conditional LoRA.
    """
    lora_config = {
        # Normal LoRA
        "conv_dna": {
            "emb_input_features": emb_input_features,
            "convert_conv": True,
            "rank": 3,  # total rank 3 * 15
            "lora_dropout": lora_dropout,
            "default_conditional": False,
        },
        (
            "res_tower",
            "unet1",
        ): {
            "emb_input_features": emb_input_features,
            "convert_conv": True,
            "rank": 9,  # total rank 9 * 5
            "lora_dropout": lora_dropout,
            "default_conditional": False,
        },
        (
            "transformer",
            "horizontal_conv0",
            "horizontal_conv1",
            "upsampling_unet0",
            "upsampling_unet1",
            "separable0",
            "separable1",
        ): {
            "emb_input_features": emb_input_features,
            "convert_linear": True,
            "convert_conv": True,
            "rank": 15,  # total rank 15 * 1
            "lora_dropout": lora_dropout,
            "default_conditional": False,
        },
        # Conditional LoRA
        "final_joined_convs": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 50,
            "lora_dropout": lora_dropout,
            "default_conditional": True,
        },
        "final_output_head": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 1,
            "lora_dropout": lora_dropout,
            "default_conditional": True,
        },
    }
    return lora_config


def make_classic_lora_config(
    *args,
    lora_dropout=0.01,
    **kwargs,
):
    """
    All layers using simple LoRA, except the final output head which uses conditional LoRA.
    """
    emb_input_features = 10  # not really used
    lora_config = {
        # Normal LoRA
        "conv_dna": {
            "emb_input_features": emb_input_features,
            "convert_conv": True,
            "rank": 3,  # total rank 3 * 15
            "lora_dropout": lora_dropout,
            "default_conditional": False,
        },
        (
            "res_tower",
            "unet1",
        ): {
            "emb_input_features": emb_input_features,
            "convert_conv": True,
            "rank": 9,  # total rank 9 * 5
            "lora_dropout": lora_dropout,
            "default_conditional": False,
        },
        (
            "transformer",
            "horizontal_conv0",
            "horizontal_conv1",
            "upsampling_unet0",
            "upsampling_unet1",
            "separable0",
            "separable1",
            "final_joined_convs",
        ): {
            "emb_input_features": emb_input_features,
            "convert_linear": True,
            "convert_conv": True,
            "rank": 15,  # total rank 15 * 1
            "lora_dropout": lora_dropout,
            "default_conditional": False,
        },
    }
    return lora_config


def make_all_conditional_large_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.01,
):
    """Make LoRA configuration for the Borzoi model."""
    lora_config = {
        "conv_dna": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 5,  # total rank 3 * 15
            "lora_dropout": lora_dropout,
        },
        (
            "res_tower",
            "unet1",
        ): {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 9,  # total rank 9 * 5
            "lora_dropout": lora_dropout,
        },
        "transformer": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_linear": True,
            "rank": 20,
            "lora_dropout": lora_dropout,
            "exclude_cond_lora_patterns": [
                "to_rel_k",
            ],
        },
        (
            "horizontal_conv0",
            "horizontal_conv1",
            "upsampling_unet0",
            "upsampling_unet1",
            "final_joined_convs",
        ): {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 45,  # total rank 15 * 1
            "lora_dropout": lora_dropout,
        },
        (
            "separable0",
            "separable1",
        ): {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 15,  # total rank 15 * 3
            "lora_dropout": lora_dropout,
        },
        "final_output_head": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": 1,
            "convert_conv": True,
            "rank": 1,
            "lora_dropout": lora_dropout,
        },
    }
    return lora_config


def make_all_conditional_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.01,
):
    """Make LoRA configuration for the Borzoi model."""
    lora_config = {
        "conv_dna": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 3,  # total rank 3 * 15
            "lora_dropout": lora_dropout,
        },
        (
            "res_tower",
            "unet1",
        ): {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 9,  # total rank 9 * 5
            "lora_dropout": lora_dropout,
        },
        "transformer": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_linear": True,
            "rank": 15,
            "lora_dropout": lora_dropout,
            "exclude_cond_lora_patterns": [
                "to_q",
                "to_k",
                "to_v",
                "to_out",
                "to_rel_k",
            ],
        },
        (
            "horizontal_conv0",
            "horizontal_conv1",
            "upsampling_unet0",
            "upsampling_unet1",
            "final_joined_convs",
        ): {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 15,  # total rank 15 * 1
            "lora_dropout": lora_dropout,
        },
        (
            "separable0",
            "separable1",
        ): {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "convert_conv": True,
            "rank": 15,  # total rank 15 * 3
            "lora_dropout": lora_dropout,
        },
        "final_output_head": {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": 1,
            "convert_conv": True,
            "rank": 1,
            "lora_dropout": lora_dropout,
        },
    }
    return lora_config
