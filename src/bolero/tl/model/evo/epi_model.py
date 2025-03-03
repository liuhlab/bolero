import os
import pkgutil
from functools import partial

import huggingface_hub
import torch
import yaml
from evo2.utils import CONFIG_MAP, HF_MODEL_NAME_MAP
from huggingface_hub import constants, hf_hub_download
from torch import nn
from torch.nn import functional as F
from vortex.logging import activations_logger
from vortex.model.layers import RMSNorm
from vortex.model.model import StripedHyena
from vortex.model.utils import dotdict, grab_first_if_tuple, load_checkpoint

from bolero.tl.generic.module_lora import LoRALinear

# config = {"epi_dims": 1, "epi_activation": "softplus", **lora_kwargs}
# config = {"epi_dims": [1, 2], "epi_activation": ["softplus", "identity"], **lora_kwargs}

# TODO
# How to do norm in encoder and decoder?
# How to deal with all the dtypes in the model?
# Write loss function
# How to balance epi loss and LM cross entropy loss?
# Model memory requirements
# Does it make sense to add context embedding n_layer of times to the backbone? Each time use a separate embedder? See P-tune V2 paper
# Is lora optional with the deep context injection? Deep context injection is essentially fine-tuning the weights of GLU last linear layer in every block


class EpiEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        epi_dims = config.epi_dims
        if isinstance(epi_dims, list):
            epi_dims = sum(epi_dims)
        self.epi_dims = epi_dims
        self.fc = nn.Linear(epi_dims, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size)

        # init fc weights as zeros
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        Input: [batch_size, seq_len, epi_dims]
        Output: [batch_size, seq_len, hidden_size]
        """
        x = self.fc(x)
        x = self.norm(x)
        return x


class EpiDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dims = config.epi_dims
        acts = config.epi_activation
        if not isinstance(dims, list):
            dims = [dims]
        if not isinstance(acts, list):
            acts = [acts]

        ml = []
        for dim, act in zip(dims, acts):
            module = nn.Sequential(
                nn.Linear(config.hidden_size, dim),
                nn.Softplus() if act == "softplus" else nn.Identity(),
            )
            ml.append(module)
        self.output_heads = nn.ModuleList(ml)
        self.pre_norm = RMSNorm(config.hidden_size)

    def forward(self, x):
        """
        Input: [batch_size, seq_len, hidden_size]
        Outputs: list of [batch_size, seq_len, epi_dim]
        """
        x = self.pre_norm(x)
        outs = [head(x) for head in self.output_heads]
        return outs


# This should swap all MLPs in both attention and hyena layers
class ParallelGatedMLPLoRA(nn.Module):
    def __init__(
        self,
        config,
        mlp_module,
        layer_idx,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act_type = config.get("mlp_activation", "gelu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        if self.layer_idx > 0 and config.get("evo2_style_activations", False):
            self.act = nn.Identity()

        self.multiple_of = multiple_of * config.model_parallel_size

        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )
        inner_size = config.get("inner_mlp_size", inner_size)

        make_lora = partial(
            LoRALinear,
            lora_rank=config.lora_rank,
            lora_alpha=None,
            lora_scale=config.get("lora_scale", 1.0),
            lora_dropout=config.get("lora_dropout", 0.0),
        )

        # Define LoRA linear layers
        self.l1 = make_lora(mlp_module.l1)
        self.l2 = make_lora(mlp_module.l2)
        self.l3 = make_lora(mlp_module.l3)

    def forward(self, z):
        """GLU forward pass"""
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)


class StripedHyenaEpi(StripedHyena):
    def __init__(self, model_name, **kwargs):
        config, weights_path = self.get_config_and_weights(model_name)

        # init Evo2 and load weights
        super().__init__(config)
        load_checkpoint(self, weights_path)

        config.update(kwargs)
        self.config = dotdict(config)

        # Epi Encoder - if using epigenetic data in forward pass
        if config.get("use_epi_enc", True):
            epi_enc_layers = config.get("epi_enc_layers", None)
            if epi_enc_layers is None:
                epi_enc_layers = [0]
            elif epi_enc_layers == "all":
                epi_enc_layers = list(range(len(self.blocks)))
            else:
                # provide list of layer idx to inject epi embeddings
                assert isinstance(epi_enc_layers, list)
            self.epi_enc_layers = epi_enc_layers

            # one block one epi encoder
            self.epi_enc = {idx: EpiEncoder(config) for idx in epi_enc_layers}
        else:
            # no epi data is used in forward pass
            self.epi_enc = None

        # Epi Decoder - predict epigenetic data from backbone embeddings
        self.epi_dec = EpiDecoder(config)

        self.convert_to_lora()

    def forward(self, dna, inference_params_dict=None, padding_mask=None, epi=None):
        """Forward pass with epi embedding deeply injected in the input of every block"""
        x = dna
        # L = x.shape[1]
        if self.print_activations:
            activations_logger.info(f"pre embedding: {x}, {x.min()}, {x.max()}")

        x = self.embedding_layer(x)
        if self.epi_enc is None:
            epi = None
        else:
            assert epi is not None

        # ================== Evo2 backbone ==================
        if self.print_activations:
            activations_logger.info(f"post embedding: {x}, {x.min()}, {x.max()}")

        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(
                x,
                epi,
                inference_params_dict=inference_params_dict,
            )
        else:
            x, inference_params_dict_out = self.stateless_forward(
                x, epi, padding_mask=padding_mask
            )

        if self.print_activations:
            activations_logger.info(f"pre norm: {x}, {x.min()}, {x.max()}")

        # By convention, we return results on the first device
        x = x.to(self.block_idx_to_device[0])
        x = self.norm(x)

        if self.print_activations:
            activations_logger.info(
                f"post norm: {x}, {x.min()}, {x.max(), {self.norm.scale}}"
            )
        # ================== Evo2 backbone ==================

        # Decode epi embeddings from backbone embeddings
        epi = self.epi_dec(x)
        # And get DNA sequence
        dna = self.unembed(x)
        return dna, epi, inference_params_dict_out

    def stateful_forward(self, x, epi, inference_params_dict=None):
        """Stateful forward pass with epi embedding optionally injected in the input of block"""
        for block_idx, block in enumerate(self.blocks):
            # add epi embeddings to dna embeddings using this block's epi encoder
            if (epi is not None) and (block_idx in self.epi_enc):
                epi = self.epi_enc[block_idx](epi)
                x = x + epi

            inference_params = inference_params_dict[self.block_idx_to_name(block_idx)]

            if self.print_activations:
                activations_logger.info(
                    f"pre block {block_idx}: {x}, {x.min()}, {x.max()} {block.__class__}"
                )
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(
                        f"{self.ground_truth_activations_path}/pre_block_{block_idx}.pt"
                    )
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(
                        f"pre block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                    )

            x = self.cross_device_transfer(x, block_idx)
            x, _ = block(x, inference_params=inference_params)

            if self.print_activations:
                activations_logger.info(
                    f"post block {block_idx}: {x}, {x.min()}, {x.max()}"
                )
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(
                        f"{self.ground_truth_activations_path}/post_block_{block_idx}.pt"
                    )
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(
                        f"post block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                    )

        return x, inference_params_dict

    def stateless_forward(self, x, epi, padding_mask=None):
        """Stateless forward pass with epi embedding optionally injected in the input of block"""
        if type(padding_mask) == torch.Tensor:
            x = x * padding_mask[..., None]

        for block_idx, block in enumerate(self.blocks):
            # add epi embeddings to dna embeddings using this block's epi encoder
            if (epi is not None) and (block_idx in self.epi_enc):
                epi = self.epi_enc[block_idx](epi)
                x = x + epi

            if self.print_activations:
                activations_logger.info(
                    f"pre block {block_idx}: {x}, {x.min()}, {x.max()} {block.__class__}"
                )
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(
                        f"{self.ground_truth_activations_path}/pre_block_{block_idx}.pt"
                    )
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(
                        f"pre block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                    )

            x = self.cross_device_transfer(x, block_idx)
            x, _ = block(x, inference_params=None, padding_mask=padding_mask)

            if self.print_activations:
                activations_logger.info(
                    f"post block {block_idx}: {x}, {x.min()}, {x.max()}"
                )
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(
                        f"{self.ground_truth_activations_path}/post_block_{block_idx}.pt"
                    )
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(
                        f"post block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                    )

        return x, None


def get_evo2_weights(
    model_name: str,
):
    """
    Load HuggingFace checkpoint using StripedHyena 2.
    """
    # use pkg util to get config from evo2 package
    config_path = CONFIG_MAP[model_name]
    config = yaml.safe_load(pkgutil.get_data("evo2", config_path))
    config = dotdict(config, Loader=yaml.FullLoader)

    hf_model_name = HF_MODEL_NAME_MAP[model_name]
    filename = f"{model_name}.pt"

    # First try normal download
    if model_name != "evo2_40b":
        weights_path = hf_hub_download(
            repo_id=hf_model_name,
            filename=filename,
        )
    # If file is split, download and join parts
    else:
        print(f"Loading checkpoint shards for {filename}")
        # If file is split, get the first part's directory to use the same cache location
        weights_path = os.path.join(os.path.dirname(constants.HF_HUB_CACHE), filename)
        if os.path.exists(weights_path):
            print(f"Found {filename}")
        else:
            # Download and join parts
            parts = []
            part_num = 0
            while True:
                try:
                    part_path = hf_hub_download(
                        repo_id=hf_model_name, filename=f"{filename}.part{part_num}"
                    )
                    parts.append(part_path)
                    part_num += 1
                except huggingface_hub.errors.EntryNotFoundError:
                    break

            # Join in the same directory
            with open(weights_path, "wb") as outfile:
                for part in parts:
                    with open(part, "rb") as infile:
                        while True:
                            chunk = infile.read(8192 * 1024)
                            if not chunk:
                                break
                            outfile.write(chunk)

            # Cleaning up the parts
            for part in parts:
                try:
                    os.remove(part)
                except OSError as e:
                    print(f"Error removing {part}: {e}")
                print("Cleaned up shards, final checkpoint saved to", weights_path)
    return config, weights_path
