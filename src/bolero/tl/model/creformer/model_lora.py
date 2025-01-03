from copy import deepcopy

import torch

from bolero.tl.generic.module_lora_cond import convert_to_conditional_lora_model
from bolero.tl.model.creformer.model import CREFormer


def make_lora_config(lora_rank, lora_alpha):
    """Make LoRA configuration for the Borzoi model."""
    if lora_alpha is None:
        lora_alpha = lora_rank

    shared_config = {
        "emb_input_features": 1,  # placeholder, not used
        "convert_conv": True,
        "convert_linear": True,
        "convert_embedding": True,
        "default_conditional": False,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }

    lora_config = {
        (
            # Finetune parameter before encoder 1 is very memory intensive
            # Because it involves base pair level transformer
            # "dna_embed",
            # "atac_embed",
            # "pos1_embed",
            # "encoder_1",
            # The remaining transformer are short
            "pos2_embed",
            "encoder_2",
            "atten_pool",
            "encoder_3",
            "tss_embed",
            "pad_embed",
            "ann",
        ): {
            **shared_config,
        },
    }
    return lora_config


class CREFormerLoRA(CREFormer):
    default_config = deepcopy(CREFormer.default_config)
    default_config.update(
        {
            "base_model_checkpoint": "REQUIRED",
            "lora_r": 8,
            "lora_alpha": None,
        }
    )

    def load_checkpoint_from_path(self, checkpoint_path, strict=True):
        """Load the pre-trained LoRA model."""
        print("Loading LoRA model weights from:", checkpoint_path)
        _checkpoint = torch.load(checkpoint_path, weights_only=False)
        if isinstance(_checkpoint, dict):
            if "model_state_dict" in _checkpoint:
                self.load_state_dict(_checkpoint["model_state_dict"], strict=strict)
            elif "state_dict" in _checkpoint:
                self.load_state_dict(_checkpoint["state_dict"], strict=strict)
            else:
                self.load_state_dict(_checkpoint, strict=strict)
        else:
            # load the model directly
            self.load_state_dict(_checkpoint.state_dict(), strict=strict)
        del _checkpoint
        return

    def load_base_ckpt(self, path):
        """Load the base model checkpoint and freeze base model."""
        self.load_checkpoint_from_path(path)

        # freeze the base model
        for params in self.parameters():
            params.requires_grad = False
        return

    def __init__(self, base_model_checkpoint, lora_r, lora_alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.load_base_ckpt(base_model_checkpoint)

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.convert_to_lora()
        return

    def convert_to_lora(self):
        """Convert the model to LoRA."""
        self.lora_config = make_lora_config(
            lora_rank=self.lora_r,
            lora_alpha=self.lora_alpha,
        )

        for module_names, config in self.lora_config.items():
            if isinstance(module_names, str):
                module_names = (module_names,)

            for module_name in module_names:
                module = getattr(self, module_name)
                if isinstance(module, torch.nn.Embedding):
                    # directly convert embedding to LoRA and set attr
                    from bolero.tl.generic.module_lora import LoRAEmbedding

                    lora_module = LoRAEmbedding.from_nn(
                        embed_module=module,
                        lora_rank=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        lora_scale=1,
                        lora_dropout=0.0,
                    )
                    setattr(self, module_name, lora_module)
                else:
                    module = convert_to_conditional_lora_model(module, **config)
                    setattr(self, module_name, module)

        return
