import torch
from torch import nn

from bolero.tl.generic.module_embedding import EmbeddingMLP
from bolero.tl.generic.module_lora_cond import (
    collapse_lora_model_,
    convert_to_conditional_lora_model,
)

from .model import Borzoi, model_summary
from .model_lora_config import LORA_CONFIG_FUNCTIONS
from .module_output import (
    DualOutputHead,
    GeneCountAttnOutputHead,
    GeneCountOutputHead,
    OutputHead,
    RNAOutputHead,
    ScoobyOutputHead,
)


class BorzoiLoRA(Borzoi):
    default_config = Borzoi.default_config.copy()
    default_config.update(
        {
            # Base Model
            "base_checkpoint_path": "REQUIRED",
            "transformer_attn_dropout": 0.0,
            "transformer_pos_dropout": 0.0,
            "transformer_ff_dropout": 0.0,
            "final_conv_dropout": 0.0,
            # Output Head
            "output_head_type": "count",
            "output_head_kwargs": None,
            "out_channels": 1,
            # Conditional Embedding and Signal Input
            "emb_input_features": "REQUIRED",
            "signal_model": True,
            "nosignal_prob": 0.2,
            "cond_emb_dim": None,
            "cond_module_kwargs": None,
            # LoRA parameters
            "lora_preset": "all_conditional",
            "hidden_dim": 256,
            "hidden_layers": 1,
            "lora_dropout": 0.01,
            "embedding_dropout": 0,
            "lora_scale": 0.02,
            "lora_alpha": None,
            "lora_norm": "layer",
            "emb_attn_pooling": False,
            # benchmark and experimental parameters
            "_multihead_model": False,
            "_disable_cond_module": False,
            "_predict_delta": True,
        }
    )

    def __init__(
        self,
        # Base Model and Output Head
        base_checkpoint_path,
        emb_input_features,
        output_head_type="count",
        output_head_kwargs=None,
        out_channels=1,
        # Conditional Embedding and Signal Input
        signal_model=True,
        nosignal_prob=0.2,
        cond_emb_dim=None,
        cond_module_kwargs=None,
        # LoRA parameters
        lora_preset="all_conditional",
        hidden_dim=256,
        hidden_layers=1,
        lora_dropout=0.01,
        embedding_dropout=0,
        lora_scale=0.02,
        lora_alpha=None,
        lora_norm="layer",
        emb_attn_pooling=False,
        # benchmark and experimental parameters
        _multihead_model=False,
        _disable_cond_module=False,
        _predict_delta=True,
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
        lora_dropout : float
            Dropout rate for the LoRA layers.
        base_checkpoint_path : str
            Path to the checkpoint file for the Borzoi base model.
        """
        self._multihead_model = _multihead_model
        self._disable_cond_module = _disable_cond_module
        self._predict_delta = _predict_delta

        super().__init__(**base_model_kwargs)
        self.lora_preset = lora_preset
        self.out_channels = out_channels
        self.signal_model = signal_model

        # update base model pretrained weights
        print("Loading base model weights from:", base_checkpoint_path)
        model_weights = torch.load(base_checkpoint_path, weights_only=True)
        model_weights = {
            k: v
            for k, v in model_weights.items()
            if k.split(".")[0] not in {"human_head", "mouse_head"}
        }
        self.load_state_dict(model_weights)

        # Setup Output Head
        # place holder for other modalities
        self.rna_output_head = None
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
        elif output_head_type == "scooby":
            self.setup_scooby_head(
                embedding_dim=emb_input_features,
                out_channels=out_channels,
            )
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

        # setup signal model
        if signal_model:
            cond_module_kwargs = cond_module_kwargs or {}
            self.nosignal_prob = nosignal_prob
            # this will setup
            # self.signal_encoder
            # self.cond_emb_module
            # and self.delta_output_head (optional)
            self.setup_signal_model(
                out_channels=out_channels,
                cell_emb_dim=emb_input_features,
                cond_emb_dim=cond_emb_dim,
                **cond_module_kwargs,
            )
            emb_input_features = self.cond_emb_module.output_dim
        else:
            self.nosignal_prob = None
            self.signal_encoder = None
            self.cond_emb_module = None
            self.delta_output_head = None

        # make lora config without converting model here
        self.emb_input_features = emb_input_features
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_scale
        self.embedding_dropout = embedding_dropout
        self.lora_norm = lora_norm
        self.emb_attn_pooling = emb_attn_pooling
        self.lora_config = self.make_lora_config()

        # make sure batchnorm is frozen
        self.freeze_batchnorms()

        # for collapse state
        self.collapsed = False
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

        self.gene_count_output_head = GeneCountOutputHead(
            embedding_dim=1920, n_blocks=n_blocks
        )
        # input shape: (bs, 1920, 16352)
        # output shape: (bs, 1)
        # activation: softplus
        return

    def setup_gene_count_attn_head(self, out_channels, activation="softplus", **kwargs):
        """Setup a single gene count output head."""
        self.setup_output_head(out_channels=out_channels, activation=activation)

        self.gene_count_output_head = GeneCountAttnOutputHead(embed_dim=1920, **kwargs)
        # input shape: (bs, 1920, 16352)
        # output shape: (bs, 1)
        # activation: softplus
        return

    def _setup_cond_emb_module(self, cell_emb_dim, cond_emb_dim, **cond_emb_kwargs):
        if self._disable_cond_module or cond_emb_dim is None or len(cond_emb_dim) == 0:
            print("Condition embedding is not used.")
            from bolero.tl.generic.module_embedding import (
                ConditionEmbeddingModuleNoEffect,
            )

            cond_emb_module = ConditionEmbeddingModuleNoEffect(
                cell_emb_dim=cell_emb_dim,
                cond_emb_dim=cond_emb_dim,
                **cond_emb_kwargs,
            )
            # This module will have no effect and just return the cell embedding
        else:
            from bolero.tl.generic.module_embedding import ConditionEmbeddingModule

            # 2. cond_emb_module combines cell and condition emb
            # LoRA embedding module will then take the combined emb as input
            cond_emb_module = ConditionEmbeddingModule(
                cell_emb_dim=cell_emb_dim,
                cond_emb_dim=cond_emb_dim,
                **cond_emb_kwargs,
            )
        return cond_emb_module

    def setup_signal_model(
        self,
        out_channels,
        cell_emb_dim,
        cond_emb_dim,
        **cond_emb_kwargs,
    ):
        """
        Setup special modules for a signal model
        """
        # 1. put out_channels also in the DNA conv layer, model takes dna_and_At as input
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

        # zero init last Conv1d layer so that it doesn't affect the model initially
        # IMPORTANT: Conv1d is the -2 layer in the signal_encoder
        nn.init.constant_(signal_encoder[-2].weight, 0.0)
        nn.init.constant_(signal_encoder[-2].bias, 0.0)
        # and GroupNorm bias is also init as no effect
        nn.init.constant_(signal_encoder[-1].weight, 1.0)
        nn.init.constant_(signal_encoder[-1].bias, 0.0)

        self.signal_encoder = signal_encoder

        self.cond_emb_module = self._setup_cond_emb_module(
            cell_emb_dim=cell_emb_dim, cond_emb_dim=cond_emb_dim, **cond_emb_kwargs
        )

        if self._predict_delta:
            # optional output head predicting delta signal between x0 and x1
            self.delta_output_head = OutputHead(
                in_channels=1920,
                out_channels=out_channels,
                activation=None,
            )
        else:
            self.delta_output_head = None
        return

    def freeze_batchnorms(self):
        """
        Freeze batchnorms in the base model.

        # https://github.com/lucidrains/tf-bind-transformer/blob/main/tf_bind_transformer/tf_bind_transformer.py#L468-L470
        When finetune Enformer or Borzoi, we have to freeze batch norm due to small batch size used.
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
            if "cond_emb_module" in name:
                continue
            if "signal_encoder" in name:
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

    def make_lora_config(self):
        """Make LoRA configuration for the Borzoi model."""
        kwargs = {
            "emb_input_features": self.emb_input_features,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "lora_dropout": self.lora_dropout,
            "lora_scale": self.lora_scale,
            "embedding_dropout": self.embedding_dropout,
            "emb_attn_pooling": self.emb_attn_pooling,
        }

        # in case some benchmark models are not using LoRA
        if self.lora_preset is None:
            return {}

        config_func = LORA_CONFIG_FUNCTIONS[self.lora_preset]
        lora_config = config_func(**kwargs)
        return lora_config

    def _convert_single_module(self, name, config):
        """Convert a single module to LoRA."""
        try:
            module = getattr(self, name)
        except AttributeError:
            print(f"Model does not have {name}, skip")
            return
        if module is None:
            return
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
        _skip_lora_convert = {
            "cond_emb_module",
            "scooby",
            "delta_output_head",
            "signal_encoder",
        }
        for module_names, config in self.lora_config.items():
            if isinstance(module_names, str):
                module_names = (module_names,)

            config["norm_type"] = self.lora_norm
            if self.lora_alpha is not None:
                config["lora_alpha"] = self.lora_alpha
                config.pop("lora_scale", None)
                print("Set all lora_alpha to", self.lora_alpha)

            for module_name in module_names:
                skip = False
                for name_skip in _skip_lora_convert:
                    if name_skip in module_name:
                        # don't convert the cond emb module
                        skip = True
                if skip:
                    continue
                try:
                    self._convert_single_module(module_name, config)
                except Exception as e:
                    print(
                        f"convert_to_lora failed at module {module_name} with config {config}"
                    )
                    raise e

        # also make sure some part of the model is trainable
        for name, param in self.named_parameters():
            if "cond_emb_module" in name:
                param.requires_grad = True
        return

    def _model_summary(
        self,
        input_data=None,
        depth=3,
        col_names=("input_size", "output_size", "num_params"),
        cache_forward_pass=False,
    ):
        emb_example = torch.randn(1, self.emb_input_features)

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
        if self.signal_model is not None:
            return self.forward_signal_model(
                x=x,
                signal=kwargs.get("signal"),
                embedding=embedding,
                crop=crop,
                return_dna_embedding=return_dna_embedding,
            )
        else:
            return self.forward_dna_only_model(
                x=x,
                embedding=embedding,
                crop=crop,
                return_dna_embedding=return_dna_embedding,
            )

    def forward_dna_only_model(
        self, x, embedding=None, crop=True, return_dna_embedding=False
    ):
        """Forward pass with DNA input only"""
        if (not getattr(self, "collapsed", False)) and (
            not getattr(self, "_multihead_model", False)
        ):
            assert embedding is not None, "embedding is required for LoRA model"

        x = super().forward(x, embedding=embedding, crop=crop)
        output = self.final_output_head(x, embedding=embedding)

        if return_dna_embedding:
            return output, x

        return output

    def forward_signal_model(
        self, x, signal, embedding, crop=True, return_dna_embedding=False
    ):
        """Forward pass with DNA and signal input."""
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

    def loss(self, y_pred, y_true, reduce=True, position_weights=None):
        """
        Compute the loss for the Borzoi model.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values, shape (batch_size, out_channels, seq_len).
        y_true : torch.Tensor
            True values, shape (batch_size, out_channels, seq_len).
        reduce : bool
            Whether to reduce the loss on seq_len dim
        position_weights : torch.Tensor, optional
            Whether to weight loss on each position along seq_len
        """
        loss, loss_breakdown, y_true = super().loss(
            y_pred,
            y_true,
            reduce=reduce,
            position_weights=position_weights,
            loss_type=self.loss_type,
        )
        return loss, loss_breakdown, y_true

    def delta_mse_loss(self, y_pred, y_true, reduce=True):
        """
        Compute the delta MSE loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted delta values, shape (batch_size, out_channels, seq_len).
        y_true : torch.Tensor
            True delta values, shape (batch_size, out_channels, seq_len).
        """
        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = self.crop(y_true)

        loss = nn.functional.mse_loss(y_pred, y_true, reduction="none")
        if reduce:
            loss = loss.mean()
        return loss, y_true

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


class BorzoiLoRAMulti(BorzoiLoRA):
    default_config = BorzoiLoRA.default_config.copy()
    default_config.update(
        {
            # BorzoiLoRAMulti does not requires emb_input_features and cond_emb_dim
            "emb_input_features": None,
            "cond_emb_dim": None,
            # it will be set during _setup_cond_emb
        }
    )

    def _setup_cond_emb_module(
        self,
        dataset_order: list[str],
        dataset_specific_dims: dict[str, int | dict[int]],
        dataset_shared_dims: int | dict[int],
        encoder_dims: list[int] = (256, 256),
        encoder_dropout: float = 0.1,
        attn_pooling: bool = True,
        norm_type: str | None = None,
        **kwargs,
    ):
        from bolero.tl.generic.module_embedding import ConditionEmbeddingModuleMulti

        cond_emb_module = ConditionEmbeddingModuleMulti(
            dataset_order=dataset_order,
            dataset_specific_dims=dataset_specific_dims,
            dataset_shared_dims=dataset_shared_dims,
            encoder_dims=encoder_dims,
            encoder_dropout=encoder_dropout,
            attn_pooling=attn_pooling,
            norm_type=norm_type,
        )
        return cond_emb_module


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
