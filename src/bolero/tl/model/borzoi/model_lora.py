import torch
from torch import nn

from bolero.tl.generic.module_embedding import EmbeddingMLP, KVBottleNeckMixin
from bolero.tl.generic.module_lora_cond import (
    collapse_lora_model_,
    convert_to_conditional_lora_model,
)

from .model import Borzoi, model_summary
from .model_lora_config import LORA_CONFIG_FUNCTIONS
from .module_output import (
    ContextOutputHead,
    DualOutputHead,
    GeneCountAttnOutputHead,
    GeneCountOutputHead,
    OutputHead,
    RNAOutputHead,
    ScoobyOutputHead,
)


class BorzoiLoRA(Borzoi, KVBottleNeckMixin):
    default_config = Borzoi.default_config.copy()
    default_config.update(
        {
            # Conditional LoRA
            "emb_input_features": "REQUIRED",
            "base_checkpoint_path": "REQUIRED",
            "lora_preset": "all_conditional",
            "out_channels": 1,
            "channel_loss_weight": 1,
            "learnable_channel_loss_weight": False,
            "hidden_dim": 256,
            "hidden_layers": 1,
            "output_layer_groups": 1,
            "lora_dropout": 0.01,
            "embedding_dropout": 0,
            "lora_scale": 0.02,
            "lora_alpha": None,
            "final_output_dropout": 0.01,
            "conditional_b": True,
            "lora_norm": "layer",
            "emb_attn_pooling": False,
            # Key-Value Bottleneck
            "kv_bottleneck": None,
            "num_memories": 256,
            "dim_memory": 20,
            "num_memory_codebooks": 2,
            "additional_embs": 1,
            "emb_input": False,
            "emb_input_dims": None,
            "output_head_type": "count",
            "output_head_kwargs": None,
            # conditional flow matching parameters
            "flow_model": False,
            "freeze_signal_encoder": False,  # only for ablation studies
            "cond_emb_dim": None,
            "cond_flow_kwargs": None,
            "signal_norm": False,
            "nosignal_prob": 0.0,
            # benchmark parameters
            "_multihead_model": False,
        }
    )

    def __init__(
        self,
        # lora config
        emb_input_features,
        base_checkpoint_path,
        lora_preset="all_conditional",
        out_channels=1,
        channel_loss_weight=1,
        learnable_channel_loss_weight=False,
        hidden_dim=256,
        hidden_layers=1,
        output_layer_groups=1,
        lora_dropout=0.01,
        embedding_dropout=0,
        lora_scale=0.02,
        lora_alpha=None,
        final_output_dropout=0.01,
        conditional_b=True,
        lora_norm="layer",
        emb_attn_pooling=False,
        # kv bottleneck
        kv_bottleneck=None,
        num_memories=256,
        dim_memory=20,
        num_memory_codebooks=2,
        additional_embs=1,
        emb_input=False,
        emb_input_dims=None,
        # output_head
        output_head_type="count",
        output_head_kwargs=None,
        # conditional flow matching parameters
        flow_model=False,
        freeze_signal_encoder=False,  # only for ablation studies
        cond_emb_dim=None,
        cond_flow_kwargs=None,
        signal_norm=False,
        nosignal_prob=0.0,
        # benchmark parameters
        _multihead_model=False,
        # base model
        **base_model_kwargs,
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
        super().__init__(**base_model_kwargs)
        self.lora_preset = lora_preset
        self.out_channels = out_channels
        self.flow_model = flow_model
        self.freeze_signal_encoder = freeze_signal_encoder
        if self.freeze_signal_encoder:
            print("Freezing signal encoder for ablation study")

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
        self.emb_input = emb_input
        self.emb_input_dims = emb_input_dims
        if self.kv_bottleneck_mode == "global":
            self.kv_bottleneck, self.emb_input_features = self.setup_kv_bottleneck(
                num_memory_codebooks=num_memory_codebooks,
                num_memories=num_memories,
                dim_memory=dim_memory,
                additional_embs=additional_embs,
                emb_input=emb_input,
                emb_input_dims=emb_input_dims,
            )
            print(
                f"Using global shared key-value bottleneck for converting indices to embeddings, "
                f"emb_input_features will become {self.emb_input_features}"
            )
        else:
            self.kv_bottleneck = None

        # Setup Output Head
        # place holder for other modalities
        self.no_lora_on_output = False
        self.rna_output_head = None
        self.upper_bound_head = None
        if output_head_kwargs is None:
            output_head_kwargs = {}
        if output_head_type == "count":
            self.setup_output_head(out_channels=out_channels)
            self.loss_type = "poisson_multinomial"
        elif output_head_type == "frac":
            # output logits, loss function will apply sigmoid
            self.setup_output_head(out_channels=out_channels, activation=None)
            self.loss_type = "bce"
        elif output_head_type == "dual_atac_mc":
            # output logits, loss function will be bce for mC and poisson multinomial for ATAC output (after softplus activation for ATAC)
            self.setup_dual_output_head(**output_head_kwargs)
            self.loss_type = "separate_bce_poisson_multinomial"
        elif output_head_type == "rna":
            self.setup_rna_head(rna_channels=out_channels)
            self.loss_type = "poisson_multinomial"
        elif output_head_type == "countext":
            self.no_lora_on_output = True
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
            self.loss_type = "poisson_multinomial"
        elif output_head_type == "scooby":
            self.setup_scooby_head(
                embedding_dim=emb_input_features,
                out_channels=out_channels,
            )
            self.loss_type = "poisson_multinomial"
        elif output_head_type == "upper_bound":
            self.setup_profile_head()
            self.loss_type = "poisson_multinomial"
        elif output_head_type == "gene_count":
            self.setup_gene_count_head(out_channels=out_channels, **output_head_kwargs)
            # this loss type is still for the tracks, gene count loss is separate
            self.loss_type = "poisson_multinomial"
        elif output_head_type == "gene_count_attn":
            self.setup_gene_count_attn_head(
                out_channels=out_channels, **output_head_kwargs
            )
            # this loss type is still for the tracks, gene count loss is separate
            self.loss_type = "poisson_multinomial"
        else:
            raise ValueError(f"output_head_type: {output_head_type} is invalid")

        # setup flow model
        if flow_model:
            cond_flow_kwargs = cond_flow_kwargs or {}
            self.signal_norm = signal_norm
            self.nosignal_prob = nosignal_prob

            self.signal_encoder, self.cond_flow_module = self.setup_flow_model(
                out_channels=out_channels,
                cell_emb_dim=emb_input_features,
                cond_emb_dim=cond_emb_dim,
                **cond_flow_kwargs,
            )
            emb_input_features = self.cond_flow_module.output_dim
        else:
            self.signal_encoder = None
            self.cond_flow_module = None

        # convert model to LoRA
        self.lora_config = self.make_lora_config(
            emb_input_features=emb_input_features,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_layer_groups=output_layer_groups,
            lora_dropout=lora_dropout,
            lora_scale=lora_scale,
            preset=lora_preset,
            embedding_dropout=embedding_dropout,
            emb_attn_pooling=emb_attn_pooling,
        )
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lora_dropout = lora_dropout
        self.embedding_dropout = embedding_dropout
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_scale
        self.conditional_b = conditional_b
        self.lora_norm = lora_norm
        self.emb_input_features = emb_input_features

        # make sure batchnorm is frozen
        self.freeze_batchnorms()

        # setup channel loss weights
        self._setup_channel_loss_weights(
            channel_loss_weight=channel_loss_weight,
            learnable_channel_loss_weight=learnable_channel_loss_weight,
        )

        # for collaps
        self.collapsed = False

        self._multihead_model = _multihead_model
        return

    def setup_rna_head(self, rna_channels, freeze_other_modules=True):
        """Setup the RNA head for the Borzoi model."""
        # simple RNA output head
        self.rna_output_head = RNAOutputHead(
            output_channels=rna_channels,
            seq_len=16384,
            embed_dim=1920,
            epi_input=False,
            num_layers=0,
        )

        # make only the RNA head trainable
        if freeze_other_modules:
            for param in self.parameters():
                param.requires_grad = False

        # use the same lora config as the final output head
        lora_config = self.lora_config["final_output_head"]
        lora_config["lora_rank"] = rna_channels
        self._convert_single_module("rna_output_head", lora_config)
        return

    def setup_output_head(self, out_channels, activation="softplus"):
        """Setup a single output head"""
        if self.flow_model:
            activation = None
        self.final_output_head = OutputHead(
            in_channels=1920,
            out_channels=out_channels,
            activation=activation,
        )

    def setup_dual_output_head(self, mc_channels=1, atac_channels=1):
        """Setup dual output for mc and atac dual modality training"""
        "returns dictionary 'atac' and 'mc' for each specific output"
        self.final_output_head = DualOutputHead(
            in_channels=1920, mc_channels=mc_channels, atac_channels=atac_channels
        )

    def setup_profile_head(self):
        """Setup a single profile head for predicting profile upper bound."""
        # upper bound are not cell-type-specific, don't add cond lora to it
        self.upper_bound_head = OutputHead(in_channels=1920, out_channels=1)

        # predict prob without sigmoid (need to add sigmoid in the loss function)
        self.final_output_head = OutputHead(
            in_channels=1920, out_channels=1, activation=None
        )
        return

    def setup_scooby_head(self, embedding_dim, out_channels):
        """Setup a single Scooby output head."""
        self.final_output_head = ScoobyOutputHead(
            embedding_dim=embedding_dim,
            input_dim=1920,
            hidden_dim=4096,
            output_dim=out_channels,
        )
        return

    def setup_gene_count_head(self, out_channels, activation="softplus", n_blocks=8):
        """Setup a single gene count output head."""
        self.setup_output_head(out_channels=out_channels, activation=activation)

        self.gene_count_head = GeneCountOutputHead(
            embedding_dim=1920, n_blocks=n_blocks
        )
        # input shape: (bs, 1920, 16352)
        # output shape: (bs, 1)
        # activation: softplus
        return

    def setup_gene_count_attn_head(self, out_channels, activation="softplus", **kwargs):
        """Setup a single gene count output head."""
        self.setup_output_head(out_channels=out_channels, activation=activation)

        self.gene_count_head = GeneCountAttnOutputHead(embed_dim=1920, **kwargs)
        # input shape: (bs, 1920, 16352)
        # output shape: (bs, 1)
        # activation: softplus
        return

    def setup_flow_model(
        self,
        out_channels,
        cell_emb_dim,
        cond_emb_dim,
        **cond_flow_kwargs,
    ):
        """
        Setup special modules for a flow model
        """
        # 1. put out_channels also in the DNA conv layer, model takes dna_and_At as input
        if self.signal_norm:
            signal_encoder = nn.Sequential(
                nn.Conv1d(out_channels, 128, kernel_size=3, padding=1),
                nn.GroupNorm(4, 128),
                nn.SiLU(),
                nn.Conv1d(128, 512, kernel_size=3, padding=1),
                nn.GroupNorm(16, 512),
                nn.SiLU(),
                nn.Conv1d(512, 1280, kernel_size=3, padding=1),
                nn.GroupNorm(40, 1280),
            )
        else:
            signal_encoder = nn.Sequential(
                nn.Conv1d(out_channels, 128, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(128, 512, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(512, 1280, kernel_size=3, padding=1),
            )
        # zero init last layer so that it doesn't affect the model initially
        signal_encoder[-1].weight.data.zero_()
        signal_encoder[-1].bias.data.zero_()

        from bolero.tl.generic.module_embedding import CondFlowModule

        # 2. cond_flow_module combines cell, time, and condition emb
        # LoRA embedding module will then take emb as input
        cond_flow_module = CondFlowModule(
            cell_emb_dim=cell_emb_dim,
            cond_emb_dim=cond_emb_dim,
            **cond_flow_kwargs,
        )
        # For example:
        # emb = self.cond_flow_module(
        #     cell_emb, time, cond_emb
        # )
        return signal_encoder, cond_flow_module

    def _setup_channel_loss_weights(
        self, channel_loss_weight, learnable_channel_loss_weight
    ):
        # channel loss weight
        # When enabled, the model will learn the weight for each channel in the loss function.
        # Based on Kendall et al. 2018, https://arxiv.org/abs/1705.07115
        # Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
        # channel weights for poisson multinomial loss
        if isinstance(channel_loss_weight, (int, float)):
            init_weight = torch.tensor([channel_loss_weight] * self.out_channels)
        else:
            init_weight = torch.tensor(channel_loss_weight)
        init_weight = -torch.log(init_weight).float()[None, :]  # (1, out_channels)

        if self.out_channels == 1:
            learnable_channel_loss_weight = False

        self.log_var_weight = nn.Parameter(
            init_weight, requires_grad=learnable_channel_loss_weight
        )  # (out_channels, 1)
        return

    def freeze_batchnorms(self):
        """
        Freeze batchnorms in the base model.

        # https://github.com/lucidrains/tf-bind-transformer/blob/main/tf_bind_transformer/tf_bind_transformer.py#L468-L470
        When finetune Enformer or Borzoi, it is recommended to freeze the batchnorms.
        """
        for _, module in self.named_modules():
            # do try to freeze lora modules as well (call this after first validation epoch)
            # # don't freeze lora modules
            # if "lora_A_module" in name:
            #     continue
            # if "lora_B_module" in name:
            #     continue

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
                for param in module.parameters():
                    param.requires_grad = False
        return

    def freeze_all_parameter_except_output_head(self):
        """Freeze all parameters except the final output head."""
        for name, param in self.named_parameters():
            if "output_head" in name:
                continue
            if "gene_count_head" in name:
                continue
            param.requires_grad = False
        return

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

    def make_lora_config(
        self,
        emb_input_features,
        hidden_dim=256,
        hidden_layers=1,
        output_layer_groups=4,
        lora_dropout=0.01,
        embedding_dropout=0,
        lora_scale=1,
        emb_attn_pooling=False,
        preset="all_conditional",
    ):
        """Make LoRA configuration for the Borzoi model."""
        kwargs = {
            "emb_input_features": emb_input_features,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "output_layer_groups": output_layer_groups,
            "lora_dropout": lora_dropout,
            "lora_scale": lora_scale,
            "embedding_dropout": embedding_dropout,
            "emb_attn_pooling": emb_attn_pooling,
        }

        # in case some benchmark models are not using LoRA
        if preset is None:
            return {}

        config_func = LORA_CONFIG_FUNCTIONS[preset]
        lora_config = config_func(**kwargs)

        if self.no_lora_on_output:
            # do not lora convert the context output head
            lora_config.pop("final_output_head", None)

        return lora_config

    def _convert_single_module(self, name, config):
        """Convert a single module to LoRA."""
        module = getattr(self, name)
        module = convert_to_conditional_lora_model(module, **config)
        setattr(self, name, module)
        return

    def collapse_lora(self, embedding):
        """Collapse the LoRA model into base form given an embedding."""
        model = collapse_lora_model_(self, embedding)
        model.collapsed = True
        model.eval()
        return model

    def convert_to_lora(self):
        """Convert the model to LoRA."""
        for module_names, config in self.lora_config.items():
            if isinstance(module_names, str):
                module_names = (module_names,)

            config["conditional_b"] = self.conditional_b
            config["norm_type"] = self.lora_norm
            config["batchnorm_momentum"] = (
                0.1  # if using batchnorm, set momentum to 0.9
            )
            if self.kv_bottleneck_mode == "local":
                config["kv_bottleneck"] = True
                config["num_memories"] = self.num_memories
                config["dim_memory"] = self.dim_memory
                config["num_memory_codebooks"] = self.num_memory_codebooks
                config["additional_embs"] = self.additional_embs
                config["emb_input"] = self.emb_input
                config["emb_input_dims"] = self.emb_input_dims
                config["norm_type"] = self.lora_norm
                config["batchnorm_momentum"] = (
                    0.1  # if using batchnorm, set momentum to 0.9
                )

            if self.lora_alpha is not None:
                config["lora_alpha"] = self.lora_alpha
                config.pop("lora_scale", None)
                print("Set all lora_alpha to", self.lora_alpha)

            for module_name in module_names:
                if "cond_flow_module" in module_name:
                    # don't convert the cond flow module
                    continue
                self._convert_single_module(module_name, config)

        # also make sure some part of the model is trainable
        for name, param in self.named_parameters():
            if "kv_bottleneck" in name:
                param.requires_grad = True
            if ("final_output_head" in name) and (self.no_lora_on_output):
                param.requires_grad = True
            if "cond_flow_module" in name:
                param.requires_grad = True
            if "signal_encoder" in name:
                if self.freeze_signal_encoder:
                    # ablation study, freeze the signal encoder
                    param.requires_grad = False
                else:
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
            if self.emb_input:
                emb_example = torch.randn(1, self.emb_input_dims)
            else:
                emb_example = torch.randint(
                    0,
                    self.num_memories,
                    (1, self.num_memory_codebooks + self.additional_embs),
                )

        # concat arch embedding to emb_example
        _arch_input_features = getattr(self, "arch_input_features", 0)
        if _arch_input_features > 0:
            arch_example = torch.randint(0, 2, (1, self.arch_input_features))
            emb_example = torch.cat([emb_example, arch_example], dim=-1)

        if input_data is None:
            input_data = {
                "x": torch.ones(1, 4, 524288),
                "embedding": emb_example,
            }
            if self.signal_encoder is not None:
                input_data["signal"] = torch.ones(1, self.out_channels, 16384).float()

        print("Input shape:")
        for k, v in input_data.items():
            print(f"{k}: {v.shape}")

        summary_str = model_summary(
            self,
            input_size=None,
            input_data=input_data,
            depth=depth,
            col_names=col_names,
            cache_forward_pass=cache_forward_pass,
        )
        return summary_str

    def forward(
        self, x, embedding=None, crop=True, return_dna_embedding=False, **kwargs
    ):
        """Borzoi forward pass to get final output."""
        if self.signal_encoder is not None:
            return self.forward_flow_model(
                x=x,
                signal=kwargs.get("signal"),
                embedding=embedding,
                crop=crop,
                return_dna_embedding=return_dna_embedding,
            )

        if (not getattr(self, "collapsed", False)) and (
            not getattr(self, "_multihead_model", False)
        ):
            assert embedding is not None, "embedding is required for LoRA model"
        else:
            if self.kv_bottleneck is not None:
                embedding = self.vq_ind_to_emb(embedding)

        x = super().forward(x, embedding=embedding, crop=crop)
        output = self.final_output_head(x, embedding=embedding)

        if return_dna_embedding:
            return output, x

        return output

    def forward_flow_model(
        self, x, signal, embedding, crop=True, return_dna_embedding=False
    ):
        """Forward pass for the flow model."""
        if torch.is_autocast_enabled():
            if x.dtype != torch.float16:
                x = x.half()
            if signal is not None:
                if signal.dtype != torch.float16:
                    signal = signal.half()
        else:
            if x.dtype != torch.float32:
                x = x.float()
            if signal is not None:
                if signal.dtype != torch.float32:
                    signal = signal.float()

        x = self.conv_dna(x, embedding)

        x_unet0 = self.res_tower(x, embedding)

        # inject signal into the DNA embedding at resolution 32
        if signal is not None:
            use_signal = True
            if self.nosignal_prob > 0.0 and self.training:
                use_signal = torch.rand(1)[0] > self.nosignal_prob
            if use_signal:
                sig_emb = self.signal_encoder(signal)
                x_unet0 += sig_emb

        x_unet1 = self.unet1(x_unet0, embedding)
        x = self._max_pool(x_unet1)
        x_unet0 = self.horizontal_conv0(x_unet0, embedding)
        x_unet1 = self.horizontal_conv1(x_unet1, embedding)
        x = self.transformer(x.permute(0, 2, 1), embedding)
        x = x.permute(0, 2, 1)
        x = self.upsampling_unet1(x, embedding)
        x += x_unet1
        x = self.separable1(x, embedding)
        x = self.upsampling_unet0(x, embedding)
        x += x_unet0
        x = self.separable0(x, embedding)
        x = self.final_joined_convs(x, embedding)

        if crop:
            x = self.crop(x)

        output = self.final_output_head(x, embedding=embedding)

        if return_dna_embedding:
            return output, x
        return output

    def weighted_loss_per_channel(self, loss_tensor):
        """Compute the weighted loss per channel based on Kendall et al. 2018."""
        weighted_loss = (
            torch.exp(-self.log_var_weight) * loss_tensor + self.log_var_weight
        )
        return weighted_loss

    def loss(
        self, y_pred, y_true, reduce=True, weighted_loss=False, position_weights=None
    ):
        """
        Compute the loss for the Borzoi model.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values, shape (batch_size, out_channels, seq_len).
        y_true : torch.Tensor
            True values, shape (batch_size, out_channels, seq_len).
        """
        if weighted_loss:
            _reduce = False
        else:
            _reduce = reduce

        loss, loss_breakdown, y_true = super().loss(
            y_pred,
            y_true,
            reduce=_reduce,
            position_weights=position_weights,
            loss_type=self.loss_type,
        )
        # loss shape (bs, out_channels)

        # weighted loss per channel and mean across channels
        if weighted_loss:
            loss = self.weighted_loss_per_channel(loss)
            if reduce:
                loss = loss.mean()
                with torch.no_grad():
                    loss_breakdown = {k: v.mean() for k, v in loss_breakdown.items()}

        return loss, loss_breakdown, y_true

    def flow_model_loss(self, v_t, u_t, reduce=True):
        """
        Compute the flow model loss.

        Parameters
        ----------
        v_t : torch.Tensor
            Predicted velocity values, shape (batch_size, out_channels, seq_len).
        u_t : torch.Tensor
            True velocity values, shape (batch_size, out_channels, seq_len).
        """
        # compute the flow model loss
        u_t = self.crop(u_t)

        loss = nn.functional.mse_loss(v_t, u_t, reduction="none")
        if reduce:
            loss = loss.mean()
        return loss, u_t

    def gene_count_loss(self, y_pred, y_true, reduce=True):
        """Compute the gene count loss."""
        # mse loss on log transformed values
        # y_pred is from softplus activation, also at count scale
        y_pred = torch.log1p(y_pred)
        y_true = torch.log1p(y_true)
        loss = nn.functional.mse_loss(y_pred, y_true, reduction="none")
        if reduce:
            loss = loss.mean()
        return loss

    def print_loss_weight(self):
        """Print the loss weight for each channel."""
        loss_weight = torch.exp(-self.log_var_weight.detach()).cpu().numpy().ravel()
        weight_str = ", ".join(f"{w:.4f}" for w in loss_weight)
        print(
            f"Loss weight for each channel: {weight_str}, sum: {loss_weight.sum():.4f}"
        )
        return


class BorzoiLoRAwithArches(BorzoiLoRA):
    default_config = BorzoiLoRA.default_config.copy()

    default_config.update(
        {
            "arch_input_features": "REQUIRED",
            "lora_checkpoint_path": "REQUIRED",
        }
    )

    def __init__(self, arch_input_features, lora_checkpoint_path, **kwargs):
        super().__init__(**kwargs)

        self.convert_to_lora()
        # load lora state dict
        self.load_checkpoint_from_path(lora_checkpoint_path)

        # setup arch embedding
        self.arch_input_features = arch_input_features
        for module in self.modules():
            if isinstance(module, EmbeddingMLP):
                module.add_arch_embedding(arch_input_features, bias=True)
        return

    def freeze_everything_except_arch(self):
        """
        Freeze all parameters except the ArchEmbedding parameters.
        """
        for name, params in self.named_parameters():
            if "arch_linear" not in name:
                params.requires_grad = False
        return
