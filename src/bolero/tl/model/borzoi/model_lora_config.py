def make_output_conditional_lora_config(
    emb_input_features,
    hidden_dim=512,
    hidden_layers=1,
    output_layer_groups=1,
    lora_dropout=0.01,
    lora_scale=1,
):
    """
    All layers using simple LoRA, except the final output head which uses conditional LoRA.
    """
    shared_config = {
        "emb_input_features": emb_input_features,
        "lora_dropout": lora_dropout,
        "lora_scale": lora_scale,
        "convert_conv": True,
        "convert_linear": True,
    }

    lora_config = {
        # Normal LoRA
        "conv_dna": {
            **shared_config,
            "lora_rank": 3,  # total lora_rank 3 * 15
            "default_conditional": False,
        },
        (
            "res_tower",
            "unet1",
        ): {
            **shared_config,
            "lora_rank": 9,  # total lora_rank 9 * 5
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
            **shared_config,
            "lora_rank": 15,  # total lora_rank 15 * 1
            "default_conditional": False,
        },
        # Conditional LoRA
        "final_joined_convs": {
            **shared_config,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "lora_rank": 50,
            "default_conditional": True,
        },
        "final_output_head": {
            **shared_config,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "lora_rank": 1,
            "default_conditional": True,
        },
    }
    return lora_config


def make_classic_lora_config(
    *args,
    lora_dropout=0.01,
    lora_scale=1,
    **kwargs,
):
    """
    All layers using simple LoRA, except the final output head which uses conditional LoRA.
    """
    shared_config = {
        "emb_input_features": 10,  # not really used
        "lora_dropout": lora_dropout,
        "lora_scale": lora_scale,
        "default_conditional": False,
        "convert_conv": True,
        "convert_linear": True,
    }

    lora_config = {
        # Normal LoRA
        "conv_dna": {
            **shared_config,
            "lora_rank": 3,  # total lora_rank 3 * 15
        },
        (
            "res_tower",
            "unet1",
        ): {
            **shared_config,
            "lora_rank": 9,  # total lora_rank 9 * 5
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
            **shared_config,
            "lora_rank": 30,  # total lora_rank 30 * 1
        },
    }
    return lora_config


def make_all_conditional_large_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.01,
    lora_scale=1,
):
    """Make LoRA configuration for the Borzoi model."""
    shared_config = {
        "emb_input_features": emb_input_features,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "output_layer_groups": output_layer_groups,
        "lora_dropout": lora_dropout,
        "lora_scale": lora_scale,
        "convert_conv": True,
        "convert_linear": True,
    }

    lora_config = {
        "conv_dna": {
            **shared_config,
            "lora_rank": 5,  # total lora_rank 3 * 15
        },
        (
            "res_tower",
            "unet1",
        ): {
            **shared_config,
            "lora_rank": 9,  # total lora_rank 9 * 5
        },
        "transformer": {
            **shared_config,
            "lora_rank": 20,
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
            **shared_config,
            "lora_rank": 45,  # total lora_rank 15 * 1
        },
        (
            "separable0",
            "separable1",
        ): {
            **shared_config,
            "lora_rank": 15,  # total lora_rank 15 * 3
        },
        "final_output_head": {
            **shared_config,
            "output_layer_groups": 1,
            "lora_rank": 1,
        },
    }
    return lora_config


def make_all_conditional_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=4,
    lora_dropout=0.01,
    lora_scale=1,
):
    """Make LoRA configuration for the Borzoi model."""
    shared_config = {
        "emb_input_features": emb_input_features,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "output_layer_groups": output_layer_groups,
        "lora_dropout": lora_dropout,
        "lora_scale": lora_scale,
        "convert_conv": True,
        "convert_linear": True,
    }
    lora_config = {
        "conv_dna": {
            **shared_config,
            "lora_rank": 2,  # total lora_lora_rank 2 * 15
        },
        (
            "res_tower",
            "unet1",
        ): {
            **shared_config,
            "lora_rank": 6,  # total lora_rank 6 * 5
        },
        "transformer": {
            **shared_config,
            "lora_rank": 30,
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
        ): {
            **shared_config,
            "lora_rank": 30,  # total lora_rank 30 * 1
        },
        (
            "separable0",
            "separable1",
        ): {
            **shared_config,
            "lora_rank": 10,  # total lora_rank 10 * 3
        },
        ("final_joined_convs",): {
            **shared_config,
            "lora_rank": 60,  # total lora_rank 60 * 1
        },
        "final_output_head": {
            **shared_config,
            "output_layer_groups": 1,
            "lora_rank": 1,
        },
    }
    return lora_config


def make_241029_v1_lora_config(
    emb_input_features,
    hidden_dim=256,
    hidden_layers=1,
    output_layer_groups=1,
    lora_dropout=0.01,
    lora_scale=1,
):
    """Make LoRA configuration for the Borzoi model."""
    shared_config = {
        "emb_input_features": emb_input_features,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "output_layer_groups": output_layer_groups,
        "lora_dropout": lora_dropout,
        "lora_scale": lora_scale,
        "convert_conv": True,
        "convert_linear": True,
    }
    lora_config = {
        "conv_dna": {
            **shared_config,
            "lora_rank": 2,  # total lora_lora_rank 2 * 15
            "default_conditional": False,
        },
        (
            "res_tower",
            "unet1",
        ): {
            **shared_config,
            "lora_rank": 6,  # total lora_rank 6 * 5
            "default_conditional": False,
        },
        "transformer": {
            **shared_config,
            "lora_rank": 30,
            "exclude_cond_lora_patterns": [
                "to_q",
                "to_k",
                "to_v",
                "to_out",
                "to_rel_k",
                "1.fn.1",  # first linear of FeedForward unconditional
                # "1.fn.4",  # second linear of FeedForward conditional
            ],
        },
        (
            "horizontal_conv0",
            "horizontal_conv1",
            "upsampling_unet0",
            "upsampling_unet1",
        ): {
            **shared_config,
            "lora_rank": 30,  # total lora_rank 30 * 1
        },
        (
            "separable0",
            "separable1",
        ): {
            **shared_config,
            "hidden_dim": [256, 512, 1024],
            "output_layer_groups": 1,
            "hidden_layers": 2,
            "lora_rank": 30,  # total lora_rank 10 * 3
        },
        "final_joined_convs": {
            **shared_config,
            "hidden_dim": [256, 512, 1024],
            "output_layer_groups": 1,
            "hidden_layers": 2,
            "lora_rank": 60,  # total lora_rank 60 * 1
        },
        "final_output_head": {
            **shared_config,
            "hidden_dim": [256, 512, 1024],
            "hidden_layers": 2,
            "output_layer_groups": 1,
            "lora_rank": 1,
        },
    }
    return lora_config


LORA_CONFIG_FUNCTIONS = {
    "output_conditional": make_output_conditional_lora_config,
    "classic": make_classic_lora_config,
    "all_conditional_large": make_all_conditional_large_lora_config,
    "all_conditional": make_all_conditional_lora_config,
    "241029_v1": make_241029_v1_lora_config,
}
