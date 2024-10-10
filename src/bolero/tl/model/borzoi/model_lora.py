import torch
from torch import nn

from bolero.tl.generic.module_embedding import KVBottleNeckMixin
from bolero.tl.generic.module_lora_cond import convert_to_conditional_lora_model

from .model import Borzoi, model_summary
from .model_lora_config import (
    make_all_conditional_large_lora_config,
    make_all_conditional_lora_config,
    make_classic_lora_config,
    make_output_conditional_lora_config,
)
from .module import ContextOutputHead, ConvBlock, OutputHead, SequentialwithArgs


class BorzoiLoRA(Borzoi, KVBottleNeckMixin):
    default_config = Borzoi.default_config.copy()
    default_config.update(
        {
            # Conditional LoRA
            "emb_input_features": "REQUIRED",
            "base_checkpoint_path": "REQUIRED",
            "lora_preset": "all_conditional",
            "out_channels": 1,
            "hidden_dim": 256,
            "hidden_layers": 1,
            "output_layer_groups": 4,
            "lora_dropout": 0.01,
            "lora_alpha": 1,
            "final_output_dropout": 0.01,
            "loss_total_weight": 0.2,
            "conditional_b": True,
            "lora_norm": "layer",
            # Key-Value Bottleneck
            "kv_bottleneck": "global",
            "num_memories": 256,
            "dim_memory": 20,
            "num_memory_codebooks": 2,
            "additional_embs": 1,
            # base model
            "context_output": True,
            "transformer_attn_dropout": 0.0,
            "transformer_pos_dropout": 0.0,
            "transformer_ff_dropout": 0.0,
            "final_conv_dropout": 0.0,
            "n_cycles": 1,
        }
    )

    def __init__(
        self,
        # lora config
        emb_input_features,
        base_checkpoint_path,
        lora_preset="all_conditional",
        out_channels=1,
        hidden_dim=256,
        hidden_layers=1,
        output_layer_groups=4,
        lora_dropout=0.01,
        lora_alpha=1,
        final_output_dropout=0.01,
        loss_total_weight=0.2,
        conditional_b=True,
        lora_norm="layer",
        # kv bottleneck
        kv_bottleneck="global",
        num_memories=256,
        dim_memory=20,
        num_memory_codebooks=2,
        additional_embs=1,
        # base model
        context_output=True,
        transformer_attn_dropout=0.0,
        transformer_pos_dropout=0.0,
        transformer_ff_dropout=0.0,
        final_conv_dropout=0.01,
        n_cycles=1,
    ):
        """
        Create a Borzoi model with LoRA layers.

        Parameters
        ----------
        emb_input_features : int
            Number of input features.
        hidden_dim : int
            Hidden dimension of the LoRA layers.
        hidden_layers : int
            Number of hidden layers in the LoRA layers.
        output_layer_groups : int
            Number of output layer groups in the LoRA layers.
        lora_dropout : float
            Dropout rate for the LoRA layers.
        base_checkpoint_path : str
            Path to the checkpoint file for the Borzoi base model.
        kv_bottleneck : bool
            Whether to use key-value bottleneck for converting indices to embeddings.
            If True, the model will expect the first "num_memory_codebooks" dimensions of embedding to be the memory indices.
        num_memories : int
            Number of memories, must be equal to codebook size in vector quantization.
        dim_memory : int
            Dimension of each memory
        num_memory_codebooks : int
            Number of codebooks or memories, must be equal to the number of heads in vector quantization.
        additional_embs : int
            Number of additional embeddings to be concatenated with the memory embeddings.
        """
        super().__init__(
            transformer_attn_dropout=transformer_attn_dropout,
            transformer_pos_dropout=transformer_pos_dropout,
            transformer_ff_dropout=transformer_ff_dropout,
            final_conv_dropout=final_conv_dropout,
        )
        self.out_channels = out_channels
        self.loss_total_weight = loss_total_weight

        # update base model pretrained weights
        print("Loading base model weights from:", base_checkpoint_path)
        model_weights = torch.load(base_checkpoint_path, weights_only=True)
        model_weights = {
            k: v
            for k, v in model_weights.items()
            if k.split(".")[0] not in {"human_head", "mouse_head"}
        }
        self.load_state_dict(model_weights)

        # key-value bottleneck for converting indices to embeddings
        if kv_bottleneck == "local":
            self.kv_bottleneck_mode = "local"
        elif kv_bottleneck == "global":
            self.kv_bottleneck_mode = "global"
        elif kv_bottleneck is None:
            self.kv_bottleneck_mode = None
        else:
            raise ValueError(
                f"kv_bottleneck value: {kv_bottleneck} is invalid, setting to None"
            )
        self.num_memories = num_memories
        self.dim_memory = dim_memory
        self.num_memory_codebooks = num_memory_codebooks
        self.additional_embs = additional_embs
        if self.kv_bottleneck_mode == "global":
            self.kv_bottleneck, self.emb_input_features = self.setup_kv_bottleneck(
                num_memory_codebooks=num_memory_codebooks,
                num_memories=num_memories,
                dim_memory=dim_memory,
                additional_embs=additional_embs,
            )
            print(
                f"Using global shared key-value bottleneck for converting indices to embeddings, "
                f"emb_input_features will become {self.emb_input_features}"
            )
        else:
            self.kv_bottleneck = None

        # Output head
        self.context_output = context_output
        if context_output:
            self.final_output_head = ContextOutputHead(
                in_channels=1920,
                out_channels=out_channels,
                context_dim=emb_input_features,
                cross_attn_heads=8,
                cross_attn_dim=64,
                ff_mult=2,
                dropout=final_output_dropout,
                kv_bottleneck=self.kv_bottleneck_mode == "local",
                num_memories=num_memories,
                dim_memory=dim_memory,
                num_memory_codebooks=num_memory_codebooks,
                additional_embs=additional_embs,
            )
        else:
            self.final_output_head = OutputHead(
                in_channels=1920, out_channels=out_channels
            )

        # recycling
        self.n_cycles = n_cycles
        if n_cycles > 1:
            self.recycle_conv = SequentialwithArgs(
                ConvBlock(in_channels=out_channels, out_channels=1280, kernel_size=1),
            )
            self.recycle_conv[0].conv_layer.weight.data.fill_(0.0)

        # convert model to LoRA
        self.lora_config = self.make_lora_config(
            emb_input_features=emb_input_features,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
            preset=lora_preset,
        )
        self.conditional_b = conditional_b
        self.lora_norm = lora_norm
        self.emb_input_features = emb_input_features

        # make sure batchnorm is frozen
        self.freeze_batchnorms()
        return

    def freeze_batchnorms(self):
        """
        Freeze batchnorms in the base model.

        # https://github.com/lucidrains/tf-bind-transformer/blob/main/tf_bind_transformer/tf_bind_transformer.py#L468-L470
        When finetune Enformer or Borzoi, it is recommended to freeze the batchnorms.
        """
        for name, module in self.named_modules():
            # don't freeze lora modules
            if "lora_A_module" in name:
                continue
            if "lora_B_module" in name:
                continue

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
                for param in module.parameters():
                    param.requires_grad = False
        return

    def freeze_all_parameter_except_output_head(self):
        """Freeze all parameters except the final output head."""
        for name, param in self.named_parameters():
            if "final_output_head" in name:
                continue
            if "recycle_conv" in name:
                continue
            param.requires_grad = False
        return

    def make_lora_config(
        self,
        emb_input_features,
        hidden_dim=256,
        hidden_layers=1,
        output_layer_groups=4,
        lora_dropout=0.01,
        lora_alpha=1,
        preset="all_conditional",
    ):
        """Make LoRA configuration for the Borzoi model."""
        kwargs = {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "lora_dropout": lora_dropout,
            "lora_alpha": lora_alpha,
        }
        if preset == "all_conditional":
            lora_config = make_all_conditional_lora_config(**kwargs)
        elif preset == "all_conditional_large":
            lora_config = make_all_conditional_large_lora_config(**kwargs)
        elif preset == "output_conditional":
            lora_config = make_output_conditional_lora_config(**kwargs)
        elif preset == "classic":
            lora_config = make_classic_lora_config(**kwargs)
        else:
            raise ValueError(f"Invalid LoRA preset: {preset}")

        if self.context_output:
            # do not lora convert the context output head
            lora_config.pop("final_output_head", None)
        return lora_config

    def convert_to_lora(self):
        """Convert the model to LoRA."""
        for module_names, config in self.lora_config.items():
            if isinstance(module_names, str):
                module_names = (module_names,)

            config["conditional_b"] = self.conditional_b
            if self.kv_bottleneck_mode == "local":
                config["kv_bottleneck"] = True
                config["num_memories"] = self.num_memories
                config["dim_memory"] = self.dim_memory
                config["num_memory_codebooks"] = self.num_memory_codebooks
                config["additional_embs"] = self.additional_embs
                config["norm_type"] = self.lora_norm
                config["batchnorm_momentum"] = (
                    0.1  # if using batchnorm, set momentum to 0.9
                )

            for module_name in module_names:
                module = getattr(self, module_name)
                module = convert_to_conditional_lora_model(module, **config)
                setattr(self, module_name, module)

        # also make sure kv_bottleneck is trainable
        for name, param in self.named_parameters():
            if "kv_bottleneck" in name:
                param.requires_grad = True
            if ("final_output_head" in name) and (self.context_output):
                param.requires_grad = True
        return

    def _model_summary(
        self,
        input_data=None,
        depth=3,
        col_names=("input_size", "output_size", "num_params"),
        cache_forward_pass=False,
    ):
        if self.kv_bottleneck_mode is None:
            emb_example = torch.randn(1, self.emb_input_features)
        else:
            emb_example = torch.randint(
                0,
                self.num_memories,
                (1, self.num_memory_codebooks + self.additional_embs),
            )

        if input_data is None:
            input_data = {
                "x": torch.ones(1, 4, 524288),
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

    def vq_ind_to_emb(self, emb_data):
        """
        VQ index to embedding.

        (bs, n_cbs + additional_embs) -> (bs, n_cbs * dim_memory + additional_embs).
        """
        n_cbs = self.kv_bottleneck.num_memory_codebooks
        vq_ind = emb_data[:, :n_cbs].type(torch.int64)
        other_emb_data = emb_data[:, n_cbs:]
        emb_data = self.kv_bottleneck(vq_ind)
        emb_data = torch.cat((emb_data, other_emb_data), dim=-1)
        return emb_data

    def output_recycle(self, x: torch.Tensor, output=None):
        """Combine the input tensor and the output tensor."""
        if output is None:
            return x
        else:
            # important: detach the output tensor so previous cycle's gradients are not used
            output.detach_()

            # output supposed to be count
            output = torch.log1p(output)

            # Questions:
            # should we perform layer norm on X?
            # should we init output recycle conv with zeros?
            # should we make output recycle conv more complex?
            # should we take final output instead of 1920 intermediate output?
            x = x + self.recycle_conv(output)
            return x

    def single_forward_pass(
        self, x: torch.Tensor, output: torch.Tensor, *args, **kwargs
    ):
        """Borzoi forward pass."""
        # change dtype to half if not already
        if torch.is_autocast_enabled():
            if x.dtype != torch.float16:
                x = x.half()
        else:
            if x.dtype != torch.float32:
                x = x.float()

        x = self.conv_dna(x, *args, **kwargs)
        x_unet0 = self.res_tower(x, *args, **kwargs)

        # x_unet0 (bs, 1280, 16384)
        x_unet0 = self.output_recycle(x_unet0, output)

        # =================
        # Remaining part is the same as Borzoi forward pass
        # =================
        # UNet connections
        x_unet1 = self.unet1(x_unet0, *args, **kwargs)
        x = self._max_pool(x_unet1)
        x_unet0 = self.horizontal_conv0(x_unet0, *args, **kwargs)
        x_unet1 = self.horizontal_conv1(x_unet1, *args, **kwargs)

        # Transformer
        x = self.transformer(x.permute(0, 2, 1), *args, **kwargs)
        x = x.permute(0, 2, 1)

        # UNet upsampling and separable convs 1
        x = self.upsampling_unet1(x, *args, **kwargs)
        x += x_unet1
        x = self.separable1(x, *args, **kwargs)

        # UNet upsampling and separable convs 0
        x = self.upsampling_unet0(x, *args, **kwargs)
        x += x_unet0
        x = self.separable0(x, *args, **kwargs)

        # Final Conv WITHOUT Crop
        x = self.final_joined_convs(x, *args, **kwargs)

        # context output
        x = self.final_output_head(x, *args, **kwargs)
        return x

    def forward(self, x, embedding):
        """Borzoi forward pass to get final output."""
        if self.kv_bottleneck is not None:
            embedding = self.vq_ind_to_emb(embedding)

        if self.n_cycles == 1:
            # simple forward
            x = super().forward(x, embedding=embedding)
            output = self.final_output_head(x, embedding=embedding)
        else:
            # recycling forward
            output = None
            if self.training:
                # get a random n from 1 to n_cycles
                n = torch.randint(1, self.n_cycles + 1, (1,)).item()
                for _ in range(n):
                    output = self.single_forward_pass(x, output, embedding=embedding)
            else:
                for _ in range(self.n_cycles):
                    output = self.single_forward_pass(x, output, embedding=embedding)

            # crop in the end
            output = self.crop(output)
        return output
