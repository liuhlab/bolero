import gc
import json
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from scprinter.seq.ema import EMA
from scprinter.seq.Models import scFootprintBPNet
from scprinter.seq.Modules import DNA_CNN, DilatedCNN, Footprints_head
from tqdm import tqdm, trange

from bolero.pl.footprint import FootPrintExamplePlotter, figure_to_array
from bolero.pp.genome import Genome
from bolero.tl.model.scprinter.dataset import (
    scPrinterDataset,
    scPrinterSingleCellDataset,
)
from bolero.tl.model.scprinter.model import scFootprintBPNetLoRA
from bolero.tl.model.scprinter.utils import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
    check_wandb_success,
    compare_configs,
    safe_save,
)
from bolero.utils import get_fs_and_path, try_gpu


class scFootprintTrainer:
    """scFootprintBPNet model for training on pseudobulk ATAC data."""

    default_config = {
        "mode": "init",
        "output_dir": "./scprinter",
        "savename": "model",
        "n_layers": 8,
        "n_filters": 1024,
        "kernel_size": 3,
        "head_kernel_size": 1,
        "activation": "gelu",
        "batch_norm": True,
        "batch_norm_momentum": 0.1,
        "groups": 8,
        "dilation_base": 1,
        "bottleneck_factor": 1,
        "rezero": False,
        "no_inception": False,
        "n_inception_layers": 8,
        "inception_layers_after": True,
        "inception_version": 2,
        "max_epochs": 100,
        "patience": 5,
        "use_amp": True,
        "lr": 0.003,
        "weight_decay": 0.001,
        "scheduler": False,
        "use_ema": True,
        "chrom_split": "REQUIRED",
        "dataset_path": "REQUIRED",
        "dataset_columns": "REQUIRED",
        "read_parquet_kwargs": {},
        "batch_size": 64,
        "bias_name": "tn5_bias",
        "max_jitter": 128,
        "cov_min_q": 0.0001,
        "cov_max_q": 0.9999,
        "clip_min": -10,
        "clip_max": 10,
        "reverse_complement": True,
        "local_shuffle_buffer_size": 10000,
        "randomize_block_order": False,
        "train_downsample": 1,
        "valid_downsample": 0.5,
        "plot_example_per_epoch": 3,
        "wandb_project": "scprinter",
        "wandb_job_type": "train",
        "wandb_group": None,
        "sample": "REQUIRED",
        "region": "REQUIRED",
        "accumulate_grad": 1,
        "val_frequency": 2500,
        "val_batches": 500,
        "output_len": 800,
        "loss_tolerance": 0.003,
    }

    lora_config = default_config.copy()
    lora_config.update(
        {
            "mode": "lora",
            "pretrained_model": "REQUIRED",
            "use_prefix": None,
            "sample_regions": 200,
            "n_pseudobulk": 10,
            "min_cov": 10,
            "max_cov": 100,
            "low_cov_ratio": 0.1,
            "cell_embedding": "REQUIRED",
            "region_embedding": "REQUIRED",
            "a_embedding": "REQUIRED",
            "b_embedding": "REQUIRED",
            "pseudobulk_path": None,
            "lora_dna_cnn": True,
            "lora_dilated_cnn": True,
            "lora_pff_cnn": True,
            "lora_output_cnn": True,
            "lora_count_cnn": True,
            "lora_rank": 32,
            "n_lora_layers": 0,
            "lora_hidden_dim": 256,
            "train_downsample": 1,
            "valid_downsample": 0.1,
            "acumulate_grad": 8,
            "lr": 3e-4,
        }
    )
    # some parameters not used in LoRA mode
    lora_config.pop("dataset_columns")
    lora_config.pop("sample")
    lora_config.pop("region")

    @classmethod
    def example_config(cls, mode="init") -> dict:
        """
        Returns an example configuration dictionary.

        Returns
        -------
            dict: Example configuration dictionary.
        """
        if mode == "lora":
            return cls.lora_config
        else:
            return cls.default_config

    @classmethod
    def make_config(cls, mode="init", **kwargs) -> dict:
        """
        Make a configuration dictionary.

        Args:
            **kwargs: Additional keyword arguments to update the configuration.

        Returns
        -------
            dict: Configuration dictionary.
        """
        if mode == "lora":
            config = cls.lora_config.copy()
        else:
            config = cls.default_config.copy()

        config.update(kwargs)
        # check if all required fields are present
        missing_keys = []
        for key, value in config.items():
            if value == "REQUIRED":
                missing_keys.append(key)
        if missing_keys:
            raise ValueError(f"Missing required fields: {missing_keys}")
        return config

    def __init__(self, config: dict):
        """
        Initialize the scFootprintTrainer class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.config = config.copy()

        # mode controls global trainer behavior in initial training or LoRA fine tuning
        self.mode = config.pop("mode").lower()

    def _setup_wandb(self):
        """
        Set up Weights and Biases for logging.

        Args:
            config (dict): Configuration dictionary.

        Returns
        -------
            Weights and Biases run context.
        """
        self._setup_config()
        config = self.config

        # setup directory
        self.output_dir = config["output_dir"]
        self.output_dir = pathlib.Path(self.output_dir).absolute().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        savename = config["savename"]
        self.savename = str(self.output_dir / savename)
        wandb_run_info_path = self.output_dir / f"{self.savename}.wandb_run_info.json"

        # load wandb run info file if exists
        if wandb_run_info_path.exists():
            with open(wandb_run_info_path) as f:
                wandb_run_info = json.load(f)

            # check if the previous run has finished successfully on W & B API
            success = check_wandb_success(wandb_run_info["path"])
            same_config = compare_configs(wandb_run_info["config"], config)
            if same_config:
                if success:
                    print(
                        f"W & B run {wandb_run_info['name']} {wandb_run_info['id']} was successful. Skipping."
                    )
                    return None
                else:
                    print(
                        f"Resuming W & B run with name: {wandb_run_info['name']} and id: {wandb_run_info['id']}."
                    )
                    wandb_run = wandb.init(
                        id=wandb_run_info["id"],
                        project=config["wandb_project"],
                        job_type=config["wandb_job_type"],
                        entity=wandb_run_info["entity"],
                        name=wandb_run_info["name"],
                        group=wandb_run_info["group"],
                        resume="allow",
                    )
            else:
                print("W & B run exists with different config. Starting a new run.")
                wandb_run = wandb.init(
                    config=config,
                    project=config["wandb_project"],
                    job_type=config["wandb_job_type"],
                    group=config["wandb_group"],
                    save_code=True,
                )
        else:
            wandb_run = wandb.init(
                config=config,
                project=config["wandb_project"],
                job_type=config["wandb_job_type"],
                group=config["wandb_group"],
                save_code=True,
            )

        # save wandb
        wandb_run_info = {
            "id": wandb_run.id,
            "name": wandb_run.name,
            "project": wandb_run.project,
            "entity": wandb_run.entity,
            "job_type": wandb_run.job_type,
            "url": wandb_run.url,
            "path": wandb_run.path,
            "group": wandb_run.group,
            "config": dict(wandb_run.config),
        }
        with open(wandb_run_info_path, "w") as f:
            json.dump(wandb_run_info, f, indent=4)

        self.run_name = wandb.run.name
        self.config = wandb.run.config

        return wandb_run

    def _setup_config(self):
        # validate and split config for later steps
        config = self.config.copy()

        if self.mode == "lora":
            _default_config_dict = self.lora_config
        else:
            _default_config_dict = self.default_config

        # required fields
        required_fields = [
            key for key, value in _default_config_dict.items() if value == "REQUIRED"
        ]
        for field in required_fields:
            assert field in config, f"Required field {field} not found in config."

        # update config with default values
        for key, value in _default_config_dict.items():
            if key not in config:
                config[key] = value

        self.config = config
        return

    def _find_last_checkpoint(self):
        if pathlib.Path(f"{self.savename}.{self.mode}.best_checkpoint.pt").exists():
            return True
        return False

    def _setup_env(self):
        # setup torch environment
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        self.device = try_gpu()

        # save config to output_dir
        with open(f"{self.savename}.config.json", "w") as f:
            json.dump(dict(self.config), f, indent=4)

        self.checkpoint = self._find_last_checkpoint()
        return

    def _setup_model_from_config(self):
        # initialize model with config
        config = self.config
        self.modes = np.arange(2, 101, 1)
        n_layers = config["n_layers"]
        n_filters = config["n_filters"]
        kernel_size = config["kernel_size"]
        head_kernel_size = config["head_kernel_size"]

        activation = config["activation"]
        if activation == "relu":
            activation = torch.nn.ReLU()
        elif activation == "gelu":
            activation = torch.nn.GELU()

        batch_norm = config["batch_norm"]
        batch_norm_momentum = config["batch_norm_momentum"]
        groups = config["groups"]
        dilation_base = config["dilation_base"]
        bottleneck_factor = config["bottleneck_factor"]
        bottleneck = int(n_filters * bottleneck_factor)
        rezero = config["rezero"]

        # CNN block architecture versions
        no_inception = config["no_inception"]
        n_inception_layers = config["n_inception_layers"]
        inception_layers_after = config["inception_layers_after"]
        if no_inception:
            n_inception_layers = 0
        inception_version = config["inception_version"]
        if inception_layers_after:
            inception_bool = [False] * (n_layers - n_inception_layers) + [True] * (
                n_inception_layers
            )
        else:
            inception_bool = [True] * n_inception_layers + [False] * (
                n_layers - n_inception_layers
            )

        acc_dna_cnn = DNA_CNN(
            n_filters=n_filters,
        )
        dilation_func = lambda x: 2 ** (x + dilation_base)
        acc_hidden = DilatedCNN(
            n_filters=n_filters,
            bottleneck=bottleneck,
            n_layers=n_layers,
            kernel_size=kernel_size,
            groups=groups,
            activation=activation,
            batch_norm=batch_norm,
            residual=True,
            rezero=rezero,
            dilation_func=dilation_func,
            batch_norm_momentum=batch_norm_momentum,
            inception=inception_bool,
            inception_version=inception_version,
        )

        acc_head = Footprints_head(
            n_filters, kernel_size=head_kernel_size, n_scales=99, per_peak_feats=1
        )
        output_len = config["output_len"]
        dna_len = output_len + acc_dna_cnn.conv.weight.shape[2] - 1
        for i in range(n_layers):
            dna_len = dna_len + 2 * (kernel_size // 2) * dilation_func(i)
        acc_model = scFootprintBPNet(
            dna_cnn_model=acc_dna_cnn,
            hidden_layer_model=acc_hidden,
            profile_cnn_model=acc_head,
            dna_len=dna_len,
            output_len=output_len,
        )
        acc_model.to(self.device)
        return acc_model

    def _setup_pretrain_model_for_adjust_output(self):
        pretrain_model_path = self.config["pretrained_model"]
        model = torch.load(pretrain_model_path)

        # set all parameters to fixed, except the profile cnn's w&b
        for p in model.parameters():
            p.requires_grad = False
        model.profile_cnn_model.conv_layer.weight.requires_grad = True
        model.profile_cnn_model.conv_layer.bias.requires_grad = True
        model.profile_cnn_model.linear.weight.requires_grad = True
        model.profile_cnn_model.linear.bias.requires_grad = True

        model.to(self.device)
        return model

    def _setup_pretrain_model_for_lora(self):
        config = self.config

        cell_emb = config["cell_embedding"]
        cell_emb = pd.read_feather(cell_emb)
        cell_emb = cell_emb.set_index(cell_emb.columns[0])
        region_emb = config["region_embedding"]
        if region_emb is not None:
            region_emb = pd.read_feather(region_emb)
            region_emb = region_emb.set_index(region_emb.columns[0])

        adj_output_model_path = f"{self.savename}.adj_output.best_model.pt"
        acc_model = torch.load(adj_output_model_path)

        # fix all parameters in the pretrained model
        for p in acc_model.parameters():
            p.requires_grad = False

        acc_model = acc_model.cpu()
        acc_model = scFootprintBPNetLoRA(
            dna_cnn_model=acc_model.dna_cnn_model,
            hidden_layer_model=acc_model.hidden_layer_model,
            profile_cnn_model=acc_model.profile_cnn_model,
            dna_len=acc_model.dna_len,
            output_len=acc_model.output_len,
            example_cell_embedding=cell_emb,
            example_region_embedding=region_emb,
            a_embedding=config["a_embedding"],
            b_embedding=config["b_embedding"],
            lora_dna_cnn=config["lora_dna_cnn"],
            lora_dilated_cnn=config["lora_dilated_cnn"],
            lora_pff_cnn=config["lora_pff_cnn"],
            lora_output_cnn=config["lora_output_cnn"],
            lora_count_cnn=config["lora_count_cnn"],
            rank=config["lora_rank"],
            n_lora_layers=config["n_lora_layers"],
            hidden_dim=config["lora_hidden_dim"],
        )
        acc_model.cuda()
        return acc_model

    def _update_state_dict(self):
        self._cleanup_env()
        cpt_path = f"{self.savename}.{self.mode}.best_checkpoint.pt"
        print(f"Load and update state dict from checkpoint file: {cpt_path}")
        checkpoint = torch.load(cpt_path)
        epoch_info = torch.load(f"{self.savename}.{self.mode}.epoch_info.pt")
        checkpoint.update(epoch_info)

        # adjust epochs
        epoch = checkpoint["epoch"]
        self.max_epochs = max(0, self.config["max_epochs"] - epoch - 1)
        self.early_stopping_counter = checkpoint["early_stopping_counter"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(
            f"Resuming from epoch {epoch+1} with {self.max_epochs} epochs left. "
            f"Best val loss: {self.best_val_loss:.5f}, "
            f"early stopping counter: {self.early_stopping_counter}."
        )

        # load state dict
        self.scp_model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.use_ema:
            self.ema.load_state_dict(checkpoint["ema"])

        del checkpoint
        torch.cuda.empty_cache()
        return

    def _set_total_params(self):
        total_params = 0
        trainable_params = 0
        for p in self.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        self.total_params = total_params
        self.trainable_params = trainable_params
        print(
            f"Total model parameters {total_params}, trainable parameters {trainable_params}"
        )
        return

    def _setup_model(self):
        mode = self.mode

        if mode == "init":
            self.scp_model = self._setup_model_from_config()
        elif mode == "adj_output":
            self.scp_model = self._setup_pretrain_model_for_adjust_output()
        elif mode == "lora":
            self.scp_model = self._setup_pretrain_model_for_lora()
        else:
            raise ValueError(f"Incorrect mode: {mode}.")

        # collect some shortcuts post model setup
        self.parameters = self.scp_model.parameters
        self.forward = self.scp_model.forward
        self.dna_len = self.scp_model.dna_len
        self.output_len = self.scp_model.output_len

        self._set_total_params()
        return

    # dataset paths
    def _get_dataset_paths(self, _chroms):
        # check if the file exists in gcs bucket
        dataset_paths = []
        for chrom in _chroms:
            path = f"{self.dataset_dir}/{chrom}"
            if self.fs.get_file_info(path).type:
                # type is True only if the file exists
                dataset_paths.append(path)
        return dataset_paths

    def _setup_dataset(self):
        config = self.config

        # parameter from setup_model
        output_len = config["output_len"]
        # The footprint function will trim 100bp from both ends of the signal to account for border effects
        # Therefore the signal window should be 200bp longer than the output length of the model
        self.signal_window = output_len + 200

        # train, valid, test split by chromosome
        chrom_split = config["chrom_split"]
        self.train_chroms = chrom_split["train"]
        self.valid_chroms = chrom_split["valid"]
        self.test_chroms = chrom_split["test"]

        # dataset location and schema
        self.fs, self.dataset_dir = get_fs_and_path(config["dataset_path"].rstrip("/"))
        self.read_parquet_kwargs = config["read_parquet_kwargs"]

        # preprocessing parameters
        self.batch_size = config["batch_size"]
        self.bias_name = config["bias_name"]
        self.max_jitter = config["max_jitter"]
        self.cov_min_q = config["cov_min_q"]
        self.cov_max_q = config["cov_max_q"]
        self.clip_min = config["clip_min"]
        self.clip_max = config["clip_max"]
        self.reverse_complement = config["reverse_complement"]

        # dataloader
        self.local_shuffle_buffer_size = config["local_shuffle_buffer_size"]
        self.randomize_block_order = config["randomize_block_order"]
        self.train_downsample = config["train_downsample"]
        self.valid_downsample = config["valid_downsample"]

        # single-cell dataset specfic parameters
        # for LoRA fine tuning
        if self.mode == "lora":
            # genome
            self.genome = Genome(config["genome"])
            # trigger one hot loading
            _ = self.genome.genome_one_hot
            self.use_prefix = config["use_prefix"]
            self.sample_regions = config["sample_regions"]
            self.n_pseudobulk = config["n_pseudobulk"]
            self.min_cov = config["min_cov"]
            self.max_cov = config["max_cov"]
            self.low_cov_ratio = config["low_cov_ratio"]
        else:
            self.columns = config["dataset_columns"]

        # create dataset slots
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    def _get_dataset(self, chroms):
        mode = self.mode
        if mode == "init":
            dataset = scPrinterDataset(
                dataset=self._get_dataset_paths(chroms),
                columns=self.columns,
                bias_name=self.bias_name,
                batch_size=self.batch_size,
                dna_window=self.dna_len,
                signal_window=self.signal_window,
                max_jitter=self.max_jitter,
                cov_min_q=self.cov_min_q,
                cov_max_q=self.cov_max_q,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                reverse_complement=self.reverse_complement,
                **self.read_parquet_kwargs,
            )
        elif mode in ["adj_output", "lora"]:
            dataset = scPrinterSingleCellDataset(
                dataset=self.dataset_dir,
                chroms=chroms,
                use_prefixs=self.use_prefix,
                batch_size=self.batch_size,
                dna_window=self.dna_len,
                signal_window=self.signal_window,
                max_jitter=self.max_jitter,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                sample_regions=self.sample_regions,
                n_pseudobulks=self.n_pseudobulk,
                min_cov=self.min_cov,
                max_cov=self.max_cov,
                low_cov_ratio=self.low_cov_ratio,
                reverse_complement=self.reverse_complement,
                override_num_blocks=None,
                genome=self.genome,
            )
            # setup pseudobulker for sc dataset
            pseudobulk_path = self.config["pseudobulk_path"]
            if pseudobulk_path is not None:
                pseudobulks = joblib.load(pseudobulk_path)
            else:
                pseudobulks = None

            cell_embedding_path = self.config["cell_embedding"]
            region_embedding_path = self.config["region_embedding"]
            dataset.prepare_pseudobulker(
                cell_embedding=cell_embedding_path,
                predefined_pseudobulk=pseudobulks,
            )
            dataset.add_region_embedding(region_embedding_path)
        else:
            raise ValueError(f"Incorrect mode: {mode}.")
        return dataset

    @property
    def train_dataset(self):
        """Training dataset."""
        mode = self.mode
        if mode != "lora":
            # init dataset only once
            if self._train_dataset is None:
                dataset = self._get_dataset(self.train_chroms)
                dataset.train()
                self._train_dataset = dataset
            return self._train_dataset
        else:
            # init dataset everytime and file order will be shuffled
            dataset = self._get_dataset(self.train_chroms)
            dataset.train()
            return dataset

    @property
    def valid_dataset(self):
        """Validation dataset."""
        if self._valid_dataset is None:
            dataset = self._get_dataset(self.valid_chroms)
            dataset.eval()
            self._valid_dataset = dataset
        return self._valid_dataset

    @property
    def test_dataset(self):
        """Test dataset."""
        if self._test_dataset is None:
            dataset = self._get_dataset(self.test_chroms)
            dataset.eval()
            self._test_dataset = dataset
        return self._test_dataset

    def _get_ema(self):
        update_after_step = 100
        ema = EMA(
            self.scp_model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=update_after_step,  # only after this number of .update() calls will it start updating
            update_every=10,
        )  # how often to actually update, to save on compute (updates every 10th .update() call)
        return ema

    def _get_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        return scaler

    def _get_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def _get_scheduler(self, optimizer):
        import transformers

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=3000, num_training_steps=100000
        )
        return scheduler

    def _setup_fit(self):
        config = self.config

        # epochs
        self.max_epochs = config["max_epochs"]
        self.patience = config["patience"]
        self.loss_tolerance = config["loss_tolerance"]
        self.val_frequency = config["val_frequency"]
        self.val_batches = config["val_batches"]
        self.early_stopping_counter = 0
        self.early_stoped = False
        self.best_val_loss = float("inf")
        self.accumulate_grad = config["accumulate_grad"]
        self.cur_epoch = 0

        # scaler
        self.use_amp = config["use_amp"]
        self.scaler = self._get_scaler()

        # optimizer
        self.learning_rate = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.optimizer = self._get_optimizer(self.learning_rate, self.weight_decay)

        # scheduler
        if config.get("scheduler", False):
            self.scheduler = self._get_scheduler(self.optimizer)
        else:
            self.scheduler = None

        # EMA model
        self.use_ema = config["use_ema"]
        if self.use_ema:
            self.ema = self._get_ema()
        else:
            self.ema = None

        # footprints
        self.modes = np.arange(2, 101, 1)
        self.modes_index = list(self.modes)
        self.select_n_modes = 30
        self.plot_example_per_epoch = config["plot_example_per_epoch"]
        if not self.plot_example_per_epoch:
            self.plot_example_per_epoch = 0

        # update state dict if checkpoint exists
        if self.checkpoint:
            self._update_state_dict()
        return

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        val_dataset,
        val_downsample,
        sample=None,
        region=None,
        val_batches=None,
    ):
        if val_batches is None:
            val_batches = self.val_batches
        # if val batches is None, use all batches in the dataset
        mode = self.mode

        if mode == "init":
            val_data_loader = val_dataset.get_dataloader(
                sample=sample,
                region=region,
                local_shuffle_buffer_size=0,
                randomize_block_order=False,
                downsample_rate=val_downsample,
            )
            atac_key = f"{region}|{sample}"
            dna_key = f"{region}|{val_dataset.dna_name}"
            bias_key = f"{region}|{val_dataset.bias_name}"
            cell_embedding_key = "NaNNaN"
            region_embedding_key = "NaNNaN"
            footprint_key = f"{region}|{sample}_footprint"
            footprinter = val_dataset.get_footprinter()
        elif mode in ["adj_output", "lora"]:
            val_data_loader = val_dataset.get_dataloader(
                device=self.device,
                downsample_rate=val_downsample,
            )
            atac_key = "bulk_data"
            dna_key = "dna_one_hot"
            bias_key = "tn5_bias"
            cell_embedding_key = "cell_embedding"
            region_embedding_key = "region_embedding"
            footprint_key = "bulk_data_footprint"
            footprinter = val_dataset.get_footprinter()
        else:
            raise ValueError(f"Incorrect mode: {mode}.")

        size = 1e-9
        val_loss = [0]
        profile_pearson_counter = CumulativeCounter()
        across_batch_pearson_fp = CumulativePearson()
        across_batch_pearson_cov = CumulativePearson()

        example_batches = []  # collect example batches for making images
        for batch_id, batch in tqdm(
            enumerate(val_data_loader),
            desc=" - (Validation)",
            dynamic_ncols=True,
            total=val_batches,
        ):
            # ==========
            # X
            # ==========
            X = batch[dna_key]
            if mode == "lora":
                cell_embedding = batch[cell_embedding_key]
                region_embedding = batch[region_embedding_key]
            else:
                cell_embedding = None
                region_embedding = None

            # ==========
            # y_footprint, y_coverage
            # ==========
            batch = footprinter(data=batch)
            y_footprint = batch[footprint_key]
            mask = ~torch.isnan(
                y_footprint
            )  # footprint contains nan values, remove them when calculating loss

            y_coverage = batch[atac_key].sum(dim=-1)
            y_coverage = torch.log1p(y_coverage)

            # ==========
            # Forward and Loss
            # ==========
            if mode == "lora":
                pred_score, pred_coverage = model(
                    X, cell_embedding=cell_embedding, region_embedding=region_embedding
                )
            else:
                pred_score, pred_coverage = model(X)
            pred_score_img = pred_score.clone().detach().cpu().numpy()
            y_footprint = torch.nan_to_num(y_footprint, nan=0)
            # as is in scPrinter
            # validation loss only has pred_score MSE, no coverage
            loss_ = F.mse_loss(pred_score[mask], y_footprint[mask])
            pred_score = pred_score.reshape((len(pred_score), -1))
            y_footprint = y_footprint.reshape((len(y_footprint), -1))
            val_loss[0] += loss_.item()

            # ==========
            # Within batch pearson and save for across batch pearson
            # ==========
            # within batch pearson
            corr = (
                batch_pearson_correlation(pred_score, y_footprint)
                .detach()
                .cpu()[:, None]
            )
            profile_pearson_counter.update(corr)
            # save for across batch pearson
            across_batch_pearson_fp.update(pred_score, y_footprint)
            across_batch_pearson_cov.update(pred_coverage, y_coverage)

            size += 1
            if batch_id < self.plot_example_per_epoch:
                batch["pred_score"] = pred_score_img
                example_batches.append(batch)

            if batch_id > val_batches:
                break

        del val_data_loader
        self._cleanup_env()
        wandb_images = self._plot_example_footprints(
            example_batches, footprinter, atac_key, bias_key, footprint_key
        )

        # ==========
        # Loss
        # ==========
        val_loss = [l / size for l in val_loss]
        val_loss = np.sum(val_loss)

        # ==========
        # Within batch pearson
        # ==========
        profile_pearson = np.array([profile_pearson_counter.mean()])

        # ==========
        # Across batch pearson
        # ==========
        across_corr = [
            across_batch_pearson_fp.corr(),
            across_batch_pearson_cov.corr(),
        ]
        return val_loss, profile_pearson, across_corr, wandb_images

    def _plot_example_footprints(
        self, example_batches, footprinter, atac_key, bias_key, footprint_key
    ):
        epoch = self.cur_epoch + 1
        wandb_images = []
        for idx, batch in enumerate(example_batches):
            plotter = FootPrintExamplePlotter(
                signal=batch[atac_key],
                bias=batch[bias_key],
                target=batch[footprint_key],
                predict=batch["pred_score"],
                footprinter=footprinter,
            )
            fig, _ = plotter.plot()
            fig_array = figure_to_array(fig)
            plt.close(fig)

            wandb_images.append(
                wandb.Image(
                    fig_array,
                    mode="RGB",
                    caption=f"Epoch {epoch} Example {idx}",
                    grouping=epoch,
                    file_type="jpg",  # reduce file size
                )
            )
        return wandb_images

    def _validation_step(self, sample, region, testing=False, val_batches=None):
        if testing:
            _dataset = self.test_dataset
            _downsample = self.valid_downsample
        else:
            _dataset = self.valid_dataset
            _downsample = self.valid_downsample

        if self.use_ema:
            self.ema.eval()
            self.ema.ema_model.eval()
            val_loss, profile_pearson, across_pearson, wandb_images = (
                self._model_validation_step(
                    model=self.ema.ema_model,
                    val_dataset=_dataset,
                    val_downsample=_downsample,
                    sample=sample,
                    region=region,
                    val_batches=val_batches,
                )
            )
            self.ema.train()
            self.ema.ema_model.train()
        else:
            self.scp_model.eval()
            val_loss, profile_pearson, across_pearson, wandb_images = (
                self._model_validation_step(
                    model=self,
                    val_dataset=_dataset,
                    val_downsample=_downsample,
                    sample=sample,
                    region=region,
                    val_batches=val_batches,
                )
            )
            self.scp_model.train()
        return val_loss, profile_pearson, across_pearson, wandb_images

    def _save_checkpint(self, update_best):
        epoch_info = {
            "epoch": self.cur_epoch,
            "early_stopping_counter": self.early_stopping_counter,
        }
        safe_save(epoch_info, f"{self.savename}.{self.mode}.epoch_info.pt")
        if update_best:
            checkpoint = {
                "best_val_loss": self.best_val_loss,
                "state_dict": self.scp_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "ema": self.ema.state_dict() if self.use_ema else None,
            }
            safe_save(checkpoint, f"{self.savename}.{self.mode}.best_checkpoint.pt")
            if self.use_ema:
                safe_save(
                    self.ema.ema_model, f"{self.savename}.{self.mode}.best_model.pt"
                )
            else:
                safe_save(self.scp_model, f"{self.savename}.{self.mode}.best_model.pt")
        return

    def _log_save_and_check_stop(self, example_images):
        epoch = self.cur_epoch
        train_loss = self.train_loss
        learning_rate = self.cur_lr
        val_loss = self.val_loss
        profile_pearson = self.val_profile_pearson
        across_pearson = self.val_across_pearson

        print(
            f" - (Training) {epoch} Loss: {train_loss:.5f}; Learning rate {learning_rate}."
        )
        print(f" - (Validation) {epoch} Loss: {val_loss:.5f}")
        print("Profile pearson", profile_pearson)
        print("Across peak pearson", across_pearson)

        # only clear the early stopping counter if the loss improvement is better than tolerance
        previous_best = self.best_val_loss
        if val_loss < self.best_val_loss - self.loss_tolerance:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        print(
            f"Previous best loss: {previous_best:.4f}, "
            f"Loss at epoch {epoch}: {val_loss:.4f}; "
            f"Early stopping counter: {self.early_stopping_counter}"
        )
        # save checkpoint if the loss is better
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpint(update_best=True)
        else:
            self._save_checkpint(update_best=False)

        wandb.log(
            {
                "train/train_loss": train_loss,
                "val/val_loss": val_loss,
                "val/best_val_loss": self.best_val_loss,
                "val/early_stopping_counter": self.early_stopping_counter,
                "val/profile_pearson": profile_pearson[0],
                "val/across_pearson_footprint": across_pearson[0],
                "val/across_pearson_coverage": across_pearson[1],
                "val_example/example_footprints": example_images,
            }
        )

        flag = self.early_stopping_counter >= self.patience
        return flag

    def _fit(self, sample, region, max_epochs=None):
        if max_epochs is None:
            max_epochs = self.max_epochs

        mode = self.mode

        # dataset related
        training_dataset = self.train_dataset
        train_downsample = self.train_downsample

        if mode in {"lora", "adj_output"}:
            atac_key = "bulk_data"
            dna_key = "dna_one_hot"
            footprint_key = "bulk_data_footprint"
            cell_embedding_key = "cell_embedding"
            region_embedding_key = "region_embedding"
        else:
            atac_key = f"{region}|{sample}"
            dna_key = f"{region}|{training_dataset.dna_name}"
            footprint_key = f"{region}|{sample}_footprint"
            cell_embedding_key = "NaNNaN"
            region_embedding_key = "NaNNaN"

        # shuffle across epochs
        local_shuffle_buffer_size = self.local_shuffle_buffer_size
        randomize_block_order = self.randomize_block_order

        # backpropagation related
        if mode in {"lora", "adj_output"}:
            footprinter = training_dataset.get_footprinter()
        else:
            footprinter = training_dataset.get_footprinter(region=region)

        scaler = self.scaler
        optimizer = self.optimizer
        scheduler = self.scheduler
        ema = self.ema
        self.val_loss = None

        stop_flag = False
        while self.cur_epoch <= max_epochs:
            if self.early_stopping_counter >= self.patience:
                # early stopping counter could be loaded from the checkpoint
                # check before starting the for loop
                print(f"Early stopping at epoch {self.cur_epoch}")
                self.early_stoped = True
                break

            if mode in {"lora", "adj_output"}:
                train_data_loader = training_dataset.get_dataloader(
                    device=self.device,
                    downsample_rate=train_downsample,
                    local_shuffle_buffer_size=local_shuffle_buffer_size,
                )
            else:
                train_data_loader = training_dataset.get_dataloader(
                    sample=sample,
                    region=region,
                    local_shuffle_buffer_size=local_shuffle_buffer_size,
                    randomize_block_order=randomize_block_order,
                    downsample_rate=train_downsample,
                )

            loader_ended = False
            iter_loader = iter(train_data_loader)
            while not loader_ended:
                moving_avg_loss = 0
                iteration = 0
                nan_loss = False

                bar = trange(
                    self.val_frequency,
                    desc=f" - (Training) {self.cur_epoch}",
                    dynamic_ncols=True,
                )
                for iteration in bar:
                    try:
                        batch = next(iter_loader)
                    except StopIteration:
                        loader_ended = True
                        break

                    try:
                        auto_cast_context = torch.autocast(
                            device_type=str(self.device),
                            dtype=torch.bfloat16,
                            enabled=self.use_amp,
                        )
                    except RuntimeError:
                        # some GPU, such as T4 does not support bfloat16
                        auto_cast_context = torch.autocast(
                            device_type=str(self.device),
                            dtype=torch.float16,
                            enabled=self.use_amp,
                        )
                    with auto_cast_context:
                        # ==========
                        # X
                        # ==========
                        X = batch[dna_key]
                        # LoRA embedding
                        if mode == "lora":
                            cell_embedding = batch[cell_embedding_key]
                            region_embedding = batch[region_embedding_key]
                        else:
                            cell_embedding = None
                            region_embedding = None

                        # ==========
                        # y_footprint, y_coverage
                        # ==========
                        random_modes = np.random.permutation(self.modes)[
                            : self.select_n_modes
                        ]
                        select_index = torch.as_tensor(
                            [self.modes_index.index(mode) for mode in random_modes]
                        )
                        batch = footprinter(data=batch, modes=random_modes)
                        y_footprint = batch[footprint_key]
                        mask = ~torch.isnan(
                            y_footprint
                        )  # footprint contains nan values, remove them when calculating loss

                        y_coverage = batch[atac_key].sum(dim=-1)
                        y_coverage = torch.log1p(y_coverage)

                        # ==========
                        # Forward and Loss
                        # ==========
                        if mode == "lora":
                            pred_score, pred_coverage = self.forward(
                                X,
                                cell_embedding=cell_embedding,
                                region_embedding=region_embedding,
                                modes=select_index,
                            )
                        else:
                            pred_score, pred_coverage = self.forward(
                                X,
                                modes=select_index,
                            )
                        loss_footprint = F.mse_loss(pred_score[mask], y_footprint[mask])
                        loss_coverage = F.mse_loss(y_coverage, pred_coverage)
                        loss = (loss_footprint + loss_coverage) / self.accumulate_grad

                        if np.isnan(loss.item()):
                            nan_loss = True
                            print("Training loss has NaN, skipping epoch.")
                            self._update_state_dict()
                            break

                    # ==========
                    # Backward
                    # ==========
                    scaler.scale(loss).backward()
                    moving_avg_loss += loss_footprint.item()
                    # only update optimizer every accumulate_grad steps
                    # this is equivalent to updating every step but with larger batch size (batch_size * accumulate_grad)
                    # however, with larger batch size, the GPU memory usage will be higher
                    if (iteration + 1) % self.accumulate_grad == 0:
                        scaler.unscale_(
                            optimizer
                        )  # Unscale gradients for clipping without inf/nan gradients affecting the model

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        if ema:
                            ema.update()

                        if scheduler is not None:
                            scheduler.step()

                    desc_str = (
                        f" - (Training) {self.cur_epoch} "
                        f"Footprint Loss: {loss_footprint.item():.2f} "
                        f"Coverage Loss: {loss_coverage.item():.2f}"
                    )
                    bar.set_description(desc_str)
                    bar.update(1)

                if nan_loss:
                    # epoch break due to nan loss, skip validation
                    continue

                self._cleanup_env()

                self.train_loss = moving_avg_loss / iteration
                self.cur_lr = optimizer.param_groups[0]["lr"]

                (
                    self.val_loss,
                    self.val_profile_pearson,
                    self.val_across_pearson,
                    wandb_images,
                ) = self._validation_step(sample=sample, region=region)

                if np.isnan(self.val_loss):
                    print("Validation loss is NaN, skipping epoch.")
                    self._update_state_dict()
                    continue

                self.cur_epoch += 1
                stop_flag = self._log_save_and_check_stop(example_images=wandb_images)
                if stop_flag:
                    print(f"Early stopping at epoch {self.cur_epoch}")
                    self.early_stoped = True
                    break

            del train_data_loader
            self._cleanup_env()
            if stop_flag:
                break
        return

    def _save_stage_flag(self, flag_name):
        with open(f"{self.savename}.{flag_name}.flag", "w") as f:
            f.write("True")
        return

    def _check_stage_flag(self, flag_name):
        return pathlib.Path(f"{self.savename}.{flag_name}.flag").exists()

    def _test(self, sample, region):
        if self.val_loss is None:
            (
                self.val_loss,
                self.val_profile_pearson,
                self.val_across_pearson,
                _,
            ) = self._validation_step(sample=sample, region=region, val_batches=None)
        valid_across_pearson_footprint, valid_across_pearson_coverage = (
            self.val_across_pearson
        )

        (
            self.test_loss,
            self.test_profile_pearson,
            self.test_across_pearson,
            wandb_images,
        ) = self._validation_step(
            sample=sample, region=region, testing=True, val_batches=None
        )
        test_across_pearson_footprint, test_across_pearson_coverage = (
            self.test_across_pearson
        )

        wandb.summary["final_valid_loss"] = self.val_loss
        wandb.summary["final_valid_within"] = self.val_profile_pearson[0]
        wandb.summary["final_valid_across"] = valid_across_pearson_footprint
        wandb.summary["final_valid_cov"] = valid_across_pearson_coverage
        wandb.summary["final_test_loss"] = self.test_loss
        wandb.summary["final_test_within"] = self.test_profile_pearson[0]
        wandb.summary["final_test_across"] = test_across_pearson_footprint
        wandb.summary["final_test_cov"] = test_across_pearson_coverage
        wandb.summary["final_image"] = wandb_images

        # final wandb flag to indicate the run is successfully finished
        wandb.summary["success"] = True
        return

    def _cleanup_env(self):
        gc.collect()
        torch.cuda.empty_cache()
        return

    def train(self) -> None:
        """
        Train the scFootprintTrainer model on a specific sample and region.

        Parameters
        ----------
            sample (str): The name of the sample.
            region (str): The name of the region.

        Returns
        -------
            None
        """
        if self.mode == "lora":
            return self.train_lora()

        sample = self.config["sample"]
        region = self.config["region"]

        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            self._setup_env()
            self._setup_model()
            self._setup_dataset()
            self._setup_fit()
            self._fit(sample=sample, region=region)
            self._test(sample=sample, region=region)
            self._cleanup_env()
            wandb.finish()
        return

    def train_lora(self) -> None:
        """Train the scFootprintTrainer model on LoRA mode."""
        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            self._setup_env()
            self._setup_dataset()

            # Fit the pretrained model on the profile CNN only with pseudobulk data
            if self._check_stage_flag("adj_output"):
                print("Pretrain output exists, skipping pretrain.")
            else:
                self.mode = "adj_output"
                self.checkpoint = self._find_last_checkpoint()
                self._setup_model()
                self._setup_fit()

                # only train for 3 epochs to adjust the output layer
                self._fit(sample=None, region=None, max_epochs=3)
                self._save_stage_flag("adj_output")
                self._cleanup_env()
                self.mode = "lora"

            # Fit LoRA
            self.checkpoint = self._find_last_checkpoint()
            self._setup_model()
            self._setup_fit()
            self._fit(sample=None, region=None)
            self._test(sample=None, region=None)
            self._cleanup_env()
            wandb.finish()
        return
