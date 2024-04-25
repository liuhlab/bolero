import gc
import json
import math
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scprinter.seq.ema import EMA
from scprinter.seq.Models import scFootprintBPNet
from scprinter.seq.Modules import DNA_CNN, DilatedCNN, Footprints_head
from tqdm import trange

from bolero.pl.footprint import FootPrintExamplePlotter, figure_to_array
from bolero.tl.model.scprinter.dataset import scPrinterDataset
from bolero.utils import check_wandb_success, compare_configs, get_fs_and_path, try_gpu


class CumulativeCounter:
    """Cumulative counter for calculating mean and sum of values."""

    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update the cumulative counter with a new value.

        Parameters
        ----------
            value (np.ndarray or torch.Tensor): The value to be added to the counter.
        """
        try:
            self.total += float(np.nansum(value))
        except TypeError:
            # torch
            self.total += float(torch.nansum(value).detach().cpu().item())
        # both numpy and torch will work
        self.count += np.prod(value.shape)

    def mean(self) -> float:
        """
        Calculate the mean of the values in the counter.

        Returns
        -------
            float: The mean value.
        """
        if self.count == 0:
            return 0
        return self.total / self.count

    def sum(self) -> float:
        """
        Calculate the sum of the values in the counter.

        Returns
        -------
            float: The sum value.
        """
        return self.total


class CumulativePearson:
    """Cumulative pearson counter for calculating the pearson correlation coefficient."""

    def __init__(self):
        self.count = 0
        self.x_counter = CumulativeCounter()
        self.y_counter = CumulativeCounter()
        self.xy_counter = CumulativeCounter()
        self.x2_counter = CumulativeCounter()
        self.y2_counter = CumulativeCounter()

    def update(
        self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Update the cumulative pearson counter with new values.

        Parameters
        ----------
            x (np.ndarray or torch.Tensor): The x values to be added to the counter.
            y (np.ndarray or torch.Tensor): The y values to be added to the counter.
        """
        self.x_counter.update(x)
        self.y_counter.update(y)
        self.xy_counter.update(x * y)
        self.x2_counter.update(x**2)
        self.y2_counter.update(y**2)

    def corr(self) -> float:
        """
        Calculate the pearson correlation coefficient.

        Returns
        -------
            float: The pearson correlation coefficient.
        """
        nx = self.x_counter.count
        ny = self.y_counter.count
        assert nx == ny, "Length mismatch between x and y"
        count = nx

        if nx == 0:
            return 0

        sum_x = self.x_counter.sum()
        mean_x = self.x_counter.mean()
        sum_y = self.y_counter.sum()
        mean_y = self.y_counter.mean()
        sum_xy = self.xy_counter.sum()
        sum_x2 = self.x2_counter.sum()
        sum_y2 = self.y2_counter.sum()

        covariance = sum_xy - mean_x * sum_y - mean_y * sum_x + count * mean_x * mean_y
        variance_x = sum_x2 - 2 * mean_x * sum_x + count * mean_x**2
        variance_y = sum_y2 - 2 * mean_y * sum_y + count * mean_y**2

        # Pearson correlation
        correlation = covariance / (
            math.sqrt(variance_x * variance_y) + 1e-8
        )  # Adding small value for numerical stability
        return correlation


def _safe_save(obj, path):
    temp_path = f"{path}.temp"
    torch.save(obj, temp_path)
    pathlib.Path(temp_path).rename(path)
    return


def _batch_pearson_correlation(x, y):
    # Compute means along the batch dimension
    mean_x = torch.mean(x, dim=-1, keepdim=True)
    mean_y = torch.mean(y, dim=-1, keepdim=True)

    diff_x = x - mean_x
    diff_y = y - mean_y

    # Compute covariance and variance
    covariance = torch.sum(diff_x * diff_y, dim=-1)
    variance_x = torch.sum((diff_x) ** 2, dim=-1)
    variance_y = torch.sum((diff_y) ** 2, dim=-1)

    # Pearson correlation
    correlation = covariance / (
        torch.sqrt(variance_x * variance_y) + 1e-8
    )  # Adding small value for numerical stability
    return correlation


class scFootprintTrainer:
    """scFootprintBPNet model for training on pseudobulk ATAC data."""

    default_config = {
        "output_dir": "./scprinter",
        "savename": "WANDB_RUN_NAME",
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
    }

    @classmethod
    def example_config(cls) -> dict:
        """
        Returns an example configuration dictionary.

        Returns
        -------
            dict: Example configuration dictionary.
        """
        return cls.default_config

    @classmethod
    def make_config(cls, **kwargs) -> dict:
        """
        Make a configuration dictionary.

        Args:
            **kwargs: Additional keyword arguments to update the configuration.

        Returns
        -------
            dict: Configuration dictionary.
        """
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
        self.config = config

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
        if savename == "WANDB_RUN_NAME":
            savename = wandb.run.name
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

        # required fields
        required_fields = [
            key for key, value in self.default_config.items() if value == "REQUIRED"
        ]
        for field in required_fields:
            assert field in config, f"Required field {field} not found in config."

        # update config with default values
        for key, value in self.default_config.items():
            if key not in config:
                config[key] = value

        self.config = config
        return

    def _find_last_checkpoint(self):
        if pathlib.Path(f"{self.savename}.checkpoint.pt").exists():
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
        output_len = 800
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
        return acc_model, dna_len, output_len

    def _setup_model_from_checkpoint(self):
        # load model if not exists
        model = torch.load(self.savename + ".model.pt")
        return model, model.dna_len, model.output_len

    def _update_state_dict(self):
        self._cleanup_env()
        checkpoint = torch.load(self.savename + ".checkpoint.pt")

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
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total model parameters", self.total_params)
        return

    def _setup_model(self):
        self.scp_model = None
        if self.checkpoint:
            model, dna_len, output_len = self._setup_model_from_checkpoint()
        else:
            model, dna_len, output_len = self._setup_model_from_config()
        self.scp_model = model.to(self.device)

        # collect some shortcuts post model setup
        self.parameters = self.scp_model.parameters
        self.forward = self.scp_model.forward
        self.dna_len = dna_len
        self.output_len = output_len

        self._set_total_params()
        return

    def _setup_dataset(self):
        config = self.config

        # parameter from setup_model
        dna_len = self.dna_len
        output_len = self.output_len

        # train, valid, test split by chromosome
        chrom_split = config["chrom_split"]
        train_chroms = chrom_split["train"]
        valid_chroms = chrom_split["valid"]
        test_chroms = chrom_split["test"]

        # dataset location and schema
        fs, path = get_fs_and_path(config["dataset_path"].rstrip("/"))
        dataset_dir = path
        columns = config["dataset_columns"]
        read_parquet_kwargs = config["read_parquet_kwargs"]

        # preprocessing parameters
        batch_size = config["batch_size"]
        bias_name = config["bias_name"]
        max_jitter = config["max_jitter"]
        cov_min_q = config["cov_min_q"]
        cov_max_q = config["cov_max_q"]
        clip_min = config["clip_min"]
        clip_max = config["clip_max"]
        reverse_complement = config["reverse_complement"]

        # dataloader
        self.local_shuffle_buffer_size = config["local_shuffle_buffer_size"]
        self.randomize_block_order = config["randomize_block_order"]
        self.train_downsample = config["train_downsample"]
        self.valid_downsample = config["valid_downsample"]

        # The footprint function will trim 100bp from both ends of the signal to account for border effects
        # Therefore the signal window should be 200bp longer than the output length of the model
        signal_window = output_len + 200

        # dataset paths
        def __get_dataset_paths(_chroms):
            # check if the file exists in gcs bucket
            dataset_paths = []
            for chrom in _chroms:
                path = f"{dataset_dir}/{chrom}"
                if fs.get_file_info(path).type:
                    # type is True only if the file exists
                    dataset_paths.append(path)
            return dataset_paths

        # setup dataset
        datasets = (
            scPrinterDataset(
                dataset=__get_dataset_paths(_chroms),
                columns=columns,
                bias_name=bias_name,
                batch_size=batch_size,
                dna_window=dna_len,
                signal_window=signal_window,
                max_jitter=max_jitter,
                cov_min_q=cov_min_q,
                cov_max_q=cov_max_q,
                clip_min=clip_min,
                clip_max=clip_max,
                reverse_complement=reverse_complement,
                **read_parquet_kwargs,
            )
            for _chroms in [train_chroms, valid_chroms, test_chroms]
        )
        self.train_dataset, self.valid_dataset, self.test_dataset = datasets
        self.train_dataset.train()
        self.valid_dataset.eval()
        self.test_dataset.eval()
        return

    def _get_ema(self):
        if self.checkpoint:
            ema = torch.load(self.savename + ".ema.pt")
        else:
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
        self.early_stopping_counter = 0
        self.early_stoped = False
        self.best_val_loss = float("inf")
        self.accumulate_grad = 1
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
        else:
            # save the very first checkpoint to allow init even first epoch fails
            self._save_checkpint(update_best=True)
        return

    @torch.no_grad()
    def _model_validation_step(
        self, model, val_dataset, val_downsample, sample, region
    ):
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
        footprint_key = f"{region}|{sample}_footprint"
        footprinter = val_dataset.get_footprinter()

        validation_size = (
            int(len(val_dataset) * val_downsample) // val_dataset.batch_size
        )
        bar = trange(
            validation_size, desc=" - (Validation)", leave=False, dynamic_ncols=True
        )

        size = 0
        val_loss = [0]
        profile_pearson_counter = CumulativeCounter()
        across_batch_pearson_fp = CumulativePearson()
        across_batch_pearson_cov = CumulativePearson()

        example_batch_ids = {
            np.random.randint(validation_size)
            for _ in range(self.plot_example_per_epoch)
        }
        example_batches = []  # collect example batches for making images
        for batch_id, batch in enumerate(val_data_loader):
            # ==========
            # X
            # ==========
            X = batch[dna_key]
            # TODO: LoRA embedding
            cell = None

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
            pred_score, pred_coverage = model(X, cells=cell)
            pred_score_img = pred_score.clone().detach().cpu().numpy()
            y_footprint = torch.nan_to_num(y_footprint, nan=0)
            loss_ = F.mse_loss(pred_score[mask], y_footprint[mask])
            pred_score = pred_score.reshape((len(pred_score), -1))
            y_footprint = y_footprint.reshape((len(y_footprint), -1))
            val_loss[0] += loss_.item()

            # ==========
            # Within batch pearson and save for across batch pearson
            # ==========
            # within batch pearson
            corr = (
                _batch_pearson_correlation(pred_score, y_footprint)
                .detach()
                .cpu()[:, None]
            )
            profile_pearson_counter.update(corr)
            # save for across batch pearson
            across_batch_pearson_fp.update(pred_score, y_footprint)
            across_batch_pearson_cov.update(pred_coverage, y_coverage)

            size += 1
            bar.update(1)

            if batch_id in example_batch_ids:
                batch["pred_score"] = pred_score_img
                example_batches.append(batch)

        bar.close()
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

    def _validation_step(self, sample, region, testing=False):
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
                )
            )
            self.ema.train()
            self.ema.ema_model.train()
        else:
            self.eval()
            val_loss, profile_pearson, across_pearson, wandb_images = (
                self._model_validation_step(
                    model=self,
                    val_dataset=_dataset,
                    val_downsample=_downsample,
                    sample=sample,
                    region=region,
                )
            )
            self.train()
        return val_loss, profile_pearson, across_pearson, wandb_images

    def _save_checkpint(self, update_best):
        checkpoint = {
            "epoch": self.cur_epoch,
            "early_stopping_counter": self.early_stopping_counter,
            "best_val_loss": self.best_val_loss,
            "state_dict": self.scp_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "ema": self.ema.state_dict() if self.use_ema else None,
        }
        _safe_save(checkpoint, self.savename + ".checkpoint.pt")
        _safe_save(self.scp_model, self.savename + ".model.pt")
        if self.use_ema:
            _safe_save(self.ema, self.savename + ".ema.pt")

        if update_best:
            _safe_save(checkpoint, f"{self.savename}.best_checkpoint.pt")
            if self.use_ema:
                _safe_save(self.ema.model, f"{self.savename}.best_model.pt")
            else:
                _safe_save(self.scp_model, f"{self.savename}.best_model.pt")
        return

    def _log_save_and_check_stop(self, epoch, example_images):
        train_loss = self.train_loss
        learning_rate = self.cur_lr
        val_loss = self.val_loss
        profile_pearson = self.val_profile_pearson
        across_pearson = self.val_across_pearson

        print(
            f" - (Training) {epoch+1} Loss: {train_loss:.5f}; Learning rate {learning_rate}."
        )
        print(f" - (Validation) {epoch+1} Loss: {val_loss:.5f}")
        print("Profile pearson", profile_pearson)
        print("Across peak pearson", across_pearson)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(
                f"Best loss at epoch {epoch+1}: {self.best_val_loss:.5f}, Saving model."
            )
            self.early_stopping_counter = 0
            self._save_checkpint(update_best=True)
        else:
            self.early_stopping_counter += 1
            print(
                f"Best loss at epoch {epoch+1}: {self.best_val_loss:.5f}, "
                f"which is better than current loss {val_loss:.5f}; "
                f"Early stopping counter: {self.early_stopping_counter}"
            )
            self._save_checkpint(update_best=False)

        wandb.log(
            {
                "train/train_loss": train_loss,
                "val/val_loss": val_loss,
                "val/best_val_loss": self.best_val_loss,
                "val/profile_pearson": profile_pearson[0],
                "val/across_pearson_footprint": across_pearson[0],
                "val/across_pearson_coverage": across_pearson[1],
                "val_example/example_footprints": example_images,
            }
        )

        flag = self.early_stopping_counter >= self.patience
        return flag

    def _fit(self, sample, region):
        # dataset related
        training_dataset = self.train_dataset
        train_downsample = self.train_downsample
        atac_key = f"{region}|{sample}"
        dna_key = f"{region}|{training_dataset.dna_name}"
        footprint_key = f"{region}|{sample}_footprint"

        # shuffle across epochs
        local_shuffle_buffer_size = self.local_shuffle_buffer_size
        randomize_block_order = self.randomize_block_order

        # backpropagation related
        footprinter = training_dataset.get_footprinter(region=region)
        scaler = self.scaler
        optimizer = self.optimizer
        scheduler = self.scheduler
        ema = self.ema
        self.val_loss = None

        for epoch in range(self.max_epochs):
            if self.early_stopping_counter >= self.patience:
                # early stopping counter could be loaded from the checkpoint
                # check before starting the for loop
                print(f"Early stopping at epoch {epoch+1}")
                self.early_stoped = True
                break
            self.cur_epoch = epoch
            bar = trange(
                int(len(training_dataset) * train_downsample)
                // training_dataset.batch_size,
                desc=f" - (Training) {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )

            train_data_loader = training_dataset.get_dataloader(
                sample=sample,
                region=region,
                local_shuffle_buffer_size=local_shuffle_buffer_size,
                randomize_block_order=randomize_block_order,
                downsample_rate=train_downsample,
            )

            moving_avg_loss = 0
            iteration = 0
            nan_loss = False
            for batch in train_data_loader:
                try:
                    auto_cast_context = torch.autocast(
                        device_type=str(self.device),
                        dtype=torch.bfloat16,
                        enabled=self.use_amp,
                    )
                except RuntimeError:
                    # some GPU, such as L4 does not support bfloat16
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
                    # TODO: LoRA embedding
                    cell = None

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
                    pred_score, pred_coverage = self.forward(
                        X, cell, modes=select_index
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
                    f" - (Training) {epoch+1} "
                    f"Footprint Loss: {loss_footprint.item():.2f} "
                    f"Coverage Loss: {loss_coverage.item():.2f}"
                )
                bar.set_description(desc_str)
                bar.update(1)
                iteration += 1
            if nan_loss:
                # epoch break due to nan loss, skip validation
                continue
            bar.close()
            del train_data_loader
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

            stop_flag = self._log_save_and_check_stop(
                epoch=epoch, example_images=wandb_images
            )
            if stop_flag:
                print(f"Early stopping at epoch {epoch+1}")
                self.early_stoped = True
                break
        return

    def _test(self, sample, region):
        if self.val_loss is None:
            (
                self.val_loss,
                self.val_profile_pearson,
                self.val_across_pearson,
                _,
            ) = self._validation_step(sample=sample, region=region)
        valid_across_pearson_footprint, valid_across_pearson_coverage = (
            self.val_across_pearson
        )

        (
            self.test_loss,
            self.test_profile_pearson,
            self.test_across_pearson,
            wandb_images,
        ) = self._validation_step(sample=sample, region=region, testing=True)
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
