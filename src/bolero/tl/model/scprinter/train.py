import gc
import math
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scprinter.seq.ema import EMA
from scprinter.seq.Models import scFootprintBPNet
from scprinter.seq.Modules import DNA_CNN, DilatedCNN, Footprints_head
from tqdm import trange

from bolero.tl.model.scprinter.dataset import scPrinterDataset
from bolero.utils import try_gpu


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


def _pearson_correlation(x, y, mean_x, mean_y, bs=1e6):
    bs = int(bs)
    covariance, variance_x, variance_y = 0, 0, 0
    for i in range(0, x.shape[0], bs):
        diff_x, diff_y = (
            x[i : i + bs].to(mean_x.device) - mean_x,
            y[i : i + bs].to(mean_x.device) - mean_y,
        )
        # Compute covariance and variance
        covariance += torch.sum(diff_x * diff_y).detach().cpu().item()
        variance_x += torch.sum((diff_x) ** 2).detach().cpu().item()
        variance_y += torch.sum((diff_y) ** 2).detach().cpu().item()

    # Pearson correlation
    correlation = covariance / (
        math.sqrt(variance_x * variance_y) + 1e-8
    )  # Adding small value for numerical stability
    return correlation


class scFootprintTrainer:
    """scFootprintBPNet model for training on pseudobulk ATAC data."""

    defalut_config = {
        "data_dir": "data",
        "model_dir": "model",
        "temp_dir": "temp",
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
        "clip_min": 10,
        "clip_max": 1e9,
        "reverse_complement": True,
        "local_shuffle_buffer_size": 10000,
        "randomize_block_order": False,
        "train_downsample": 1,
        "valid_downsample": 0.5,
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

    def __init__(self, config: dict):
        """
        Initialize the scFootprintTrainer class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.config = config

    def _setup_wandb(self, config: dict):
        """
        Set up Weights and Biases for logging.

        Args:
            config (dict): Configuration dictionary.

        Returns
        -------
            Weights and Biases run context.
        """
        wandb_context = wandb.init(config=config)
        self.run_name = wandb.run.name
        self.config = wandb.config

        print("Run name:", self.run_name)
        return wandb_context

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
        if pathlib.Path(self.savename).exists():
            return True
        return False

    def _setup_env(self, config):
        config = self.config

        # setup torch environment
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        self.device = try_gpu()

        # setup directory
        self.data_dir = config["data_dir"]
        self.model_dir = config["model_dir"]
        self.temp_dir = config["temp_dir"]
        # save model
        self.savename = config["savename"]

        # TODO check if checkpoint exists
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
        self._cleanup_env()

        checkpoint = torch.load(self.savename)

        # adjust epochs
        epoch = checkpoint["epoch"]
        self.max_epochs = max(0, self.max_epochs - epoch)
        self.early_stopping_counter = checkpoint["early_stopping_counter"]

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

    def _setup_model(self):
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
        dataset_dir = config["dataset_path"]
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

        # setup dataset
        datasets = (
            scPrinterDataset(
                dataset=[f"{dataset_dir}/{chrom}" for chrom in _chroms],
                columns=columns,
                bias_name=bias_name,
                batch_size=batch_size,
                dna_window=dna_len,
                signal_window=output_len,
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
        return

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
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.accumulate_grad = 1

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
        self.use_ema = config.get("use_ema", False)
        if self.use_ema:
            self.ema = self._get_ema()
        else:
            self.ema = None

        # footprints
        self.modes = np.arange(2, 101, 1)
        self.modes_index = list(self.modes)
        self.select_n_modes = 30

    @torch.no_grad()
    def _model_validation_step(
        self, model, val_dataset, val_downsample, sample, region
    ):
        validation_size = len(val_dataset // val_dataset.batch_size)
        val_data_loader = val_dataset.get_dataloader(
            sample=sample,
            region=region,
            local_shuffle_buffer_size=0,
            randomize_block_order=False,
            downsample=val_downsample,
        )
        atac_key = f"{region}|{sample}"
        dna_key = f"{region}|{val_dataset.dna_name}"
        footprint_key = f"{region}|{sample}_footprint"
        footprinter = val_dataset.get_footprinter()

        bar = trange(
            validation_size, desc=" - (Validation)", leave=False, dynamic_ncols=True
        )
        total_len = 0
        size = 0
        profile_pearson = []
        across_batch_pearson = [[], []]
        across_batch_pearson_coverage = [[], []]
        val_loss = [0]
        mean_pred_score, mean_y, mean_pred_coverage, mean_coverage = 0, 0, 0, 0
        for batch in val_data_loader:
            # ==========
            # X
            # ==========
            X = batch[dna_key]
            # TODO: LoRA embedding
            cell = None

            # ==========
            # y_footprint, y_coverage
            # ==========
            random_modes = np.random.permutation(self.modes)[: self.select_n_modes]
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
            pred_score, pred_coverage = model(X, cells=cell)
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
            profile_pearson.append(corr)
            # save for across batch pearson
            mean_pred_score += pred_score.mean() * len(pred_score)
            mean_y += y_footprint.mean() * len(y_footprint)
            mean_pred_coverage += pred_coverage.mean() * len(pred_coverage)
            mean_coverage += y_coverage.mean() * len(y_coverage)
            total_len += len(pred_score)
            across_batch_pearson[0].append(pred_score.detach().cpu().reshape(-1))
            across_batch_pearson[1].append(y_footprint.detach().cpu().reshape(-1))
            across_batch_pearson_coverage[0].append(
                y_coverage.detach().cpu().reshape(-1)
            )
            across_batch_pearson_coverage[1].append(
                pred_coverage.detach().cpu().reshape(-1)
            )
            size += 1

            bar.update(1)
        bar.close()

        # ==========
        # Loss
        # ==========
        val_loss = [l / size for l in val_loss]
        val_loss = np.sum(val_loss)

        # ==========
        # Within batch pearson
        # ==========
        if len(profile_pearson) > 0:
            profile_pearson = (
                torch.cat(profile_pearson, dim=0).mean(dim=0).detach().cpu().numpy()
            )
        else:
            profile_pearson = np.array([0])

        # ==========
        # Across batch pearson
        # ==========
        pred_score, y_footprint = (
            torch.cat(across_batch_pearson[0], dim=0),
            torch.cat(across_batch_pearson[1], dim=0),
        )
        pred_coverage, y_coverage = (
            torch.cat(across_batch_pearson_coverage[1], dim=0),
            torch.cat(across_batch_pearson_coverage[0], dim=0),
        )
        mean_pred_score /= total_len
        mean_y /= total_len
        mean_pred_coverage /= total_len
        mean_coverage /= total_len
        across_corr = [
            _pearson_correlation(pred_score, y_footprint, mean_pred_score, mean_y),
            _pearson_correlation(
                pred_coverage, y_coverage, mean_pred_coverage, mean_coverage
            ),
        ]
        return val_loss, profile_pearson, across_corr

    def _validation_step(self, sample, region, testing=False):
        if testing:
            _dataset = self.test_dataset
            _downsample = 1
        else:
            _dataset = self.valid_dataset
            _downsample = self.valid_downsample

        if self.use_ema:
            self.ema.eval()
            self.ema.ema_model.eval()
            val_loss, profile_pearson, across_pearson = self._model_validation_step(
                model=self.ema.ema_model,
                val_dataset=_dataset,
                val_downsample=_downsample,
                sample=sample,
                region=region,
            )
            self.ema.train()
            self.ema.ema_model.train()
        else:
            self.eval()
            val_loss, profile_pearson, across_pearson = self._model_validation_step(
                model=self,
                val_dataset=_dataset,
                val_downsample=_downsample,
                sample=sample,
                region=region,
            )
            self.train()
        return val_loss, profile_pearson, across_pearson

    def _save_checkpint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "early_stopping_counter": self.early_stopping_counter,
            "state_dict": self.scp_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "ema": self.ema.state_dict() if self.use_ema else None,
        }
        torch.save(checkpoint, self.savename)
        torch.save(self.scp_model, self.savename + ".model.pt")
        if self.use_ema:
            torch.save(self.ema, self.savename + ".ema.pt")
            torch.save(self.ema.ema_model, self.savename + ".ema_model.pt")
        return

    def _log_save_and_check_stop(self, epoch):
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
            self._save_checkpint(epoch)
        else:
            self.early_stopping_counter += 1
            print(
                f"Best loss at epoch {epoch+1}: {self.best_val_loss:.5f}, Early stopping counter: {self.early_stopping_counter}"
            )

        wandb.log(
            {
                "train/train_loss": train_loss,
                "val/val_loss": val_loss,
                "val/best_val_loss": self.best_val_loss,
                "val/profile_pearson": profile_pearson[0],
                "val/across_pearson_footprint": across_pearson[0],
                "val/across_pearson_coverage": across_pearson[1],
                "epoch": epoch,
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

        for epoch in range(self.max_epochs):
            bar = trange(
                len(training_dataset) // training_dataset.batch_size,
                desc=f" - (Training) {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )

            train_data_loader = training_dataset.get_dataloader(
                sample=sample,
                region=region,
                local_shuffle_buffer_size=local_shuffle_buffer_size,
                randomize_block_order=randomize_block_order,
                downsample=train_downsample,
            )

            moving_avg_loss = 0
            iteration = 0
            nan_loss = False
            self.val_loss = None
            for batch in train_data_loader:
                with torch.autocast(
                    device_type=self.device, dtype=torch.bfloat16, enabled=self.use_amp
                ):
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
                        raise NotImplementedError
                        ema, optimizer, scaler = self.load_train_state_dict(
                            ema, optimizer, scaler, self.savename
                        )
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

            self.train_loss = moving_avg_loss / iteration
            self.cur_lr = optimizer.param_groups[0]["lr"]

            self.val_loss, self.val_profile_pearson, self.val_across_pearson = (
                self._validation_step(sample=sample, region=region)
            )

            if np.isnan(self.val_loss):
                raise NotImplementedError
                ema, optimizer, scaler = self.load_train_state_dict(
                    ema, optimizer, scaler, self.savename
                )
                continue

            stop_flag = self._log_save_and_check_stop(epoch=epoch)
            if stop_flag:
                print(f"Early stopping at epoch {epoch+1}")
                break
        return

    def _test(self, sample, region):
        if self.val_loss is None:
            self.val_loss, self.val_profile_pearson, self.val_across_pearson = (
                self._validation_step(sample=sample, region=region)
            )
        valid_across_pearson_footprint, valid_across_pearson_coverage = (
            self.val_across_pearson
        )

        self.test_loos, self.test_profile_pearson, self.test_across_pearson = (
            self._validation_step(sample=sample, region=region, testing=True)
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
        return

    def _cleanup_env(self):
        gc.collect()
        torch.cuda.empty_cache()
        return

    def train(self, sample: str, region: str) -> None:
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
        with self._setup_wandb():
            self.setup_config()

            self._setup_env()

            self._setup_model()

            self._setup_dataset()

            self._setup_fit()

            self._fit(sample=sample, region=region)

            self._test(sample=sample, region=region)

            self._cleanup_env()

            wandb.finish()
        return
