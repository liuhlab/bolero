import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pyranges as pr
import torch
import wandb

from bolero.pl.footprint import FootPrintExamplePlotter
from bolero.pl.utils import figure_to_array
from bolero.tl.generic.train import GenericTrainer
from bolero.tl.generic.train_helper import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
)
from bolero.tl.model.borzoi.train import TrainerBorzoiDatasetMixin
from bolero.tl.model.scprinter.dataset import scPrinterDatasetBase
from bolero.tl.model.scprinter.dataset_online import scPrinterOnlineDataset
from bolero.tl.model.scprinter.model import seq2PRINT


class scFootprintTrainerMixin(TrainerBorzoiDatasetMixin, GenericTrainer):
    trainer_config = {
        "mode": "REQUIRED",
        "fold_split_id": "REQUIRED",
        "region_bed_path": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_group": None,
        "wandb_name": None,
        "max_epochs": 30,
        "use_amp": True,
        "use_ema": True,
        "scheduler": True,
        "lr": "REQUIRED",
        "large_lr_scale": 2,
        "optimizer": "adamw",
        "global_clipnorm": 0.2,
        "train_batches": 10000,
        "val_batches": 250,
        "warmup_steps": 1000,
        "weight_decay": 1e-4,
        "plot_example_per_epoch": 9,
        "accumulate_grad": 1,
        "dataloader_concurrency": 16,
        "downsample_train_region": None,
        "downsample_valid_region": None,
        "downsample_test_region": None,
        "grad_norm_collector": False,
        "save_state_every_n_epoch": None,
    }

    def __init__(self, config):
        super().__init__(config)

        # the prefix of pseudobulk data in the batch dict
        # this is the pseudobulker name passed to dataset
        self.prefix = config["prefix"]

        self.model: torch.nn.Module = None
        self._setup_env()
        self._setup_dataset()

        # placeholders
        self.cur_lr = 0
        self.train_loss = None
        self.val_loss = None
        self.val_images = None
        self.profile_pearson = None
        self.across_pearson = None
        return

    def _setup_dataset(self):
        """
        Set up the dataset by splitting it into train, valid, and test sets.
        """
        # create dataset
        self.dataset = self._get_dataset()
        if not hasattr(self.dataset, "bed"):
            self.dataset.bed = pr.read_bed(self.config["region_bed_path"], as_df=True)
        # train, valid, test split by fold
        (
            self.train_folds,
            self.valid_folds,
            self.test_folds,
            self.train_regions,
            self.valid_regions,
            self.test_regions,
        ) = self.dataset.get_train_valid_test(
            self.config["fold_split_id"],
            downsample_train_region=self.config["downsample_train_region"],
            downsample_valid_region=self.config["downsample_valid_region"],
            downsample_test_region=self.config["downsample_test_region"],
        )

        # add footprinter
        self.footprinter = self.dataset.get_footprinter()
        return

    # =============================
    # Model training and validation
    # =============================
    def _setup_model(self):
        raise NotImplementedError

    def _get_scheduler(self, optimizer):
        self.config["scheduler_type"] = "borzoi"
        warmup_steps = self.config.get("warmup_steps", 5000)
        accumulate_grad = self.config.get("accumulate_grad", 1)
        warmup_steps = (
            warmup_steps // accumulate_grad + 1
        )  # because we update every accumulate_grad steps

        total_steps = self.max_epochs * self.train_batches
        total_steps = total_steps // accumulate_grad + 1

        scheduler = GenericTrainer._get_scheduler(
            self, optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
        return scheduler

    def _get_optimizer(self):
        optimizer_type = self.config.get("optimizer", "adamw").lower()
        base_lr = self.config["lr"]
        large_lr = base_lr * self.config["large_lr_scale"]
        weight_decay = self.config["weight_decay"]

        lr_groups = [
            {
                "params": self.model.footprint_parameters(),
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": self.model.coverage_parameters(),
                "lr": large_lr,
                "weight_decay": weight_decay,
            },
        ]
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                lr_groups, lr=base_lr, weight_decay=weight_decay
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                lr_groups, lr=base_lr, weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                lr_groups, lr=base_lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(
                f"Optimizer {optimizer_type} not supported, should be one of ['adamw', 'adam', 'sgd']."
            )
        return optimizer

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        collect_data=False,
        collect_fn=None,
        save_keys=None,
        **kwargs,
    ):
        if save_keys is not None:
            save_keys = set(save_keys)

        loss_logger = {k: CumulativeCounter() for k in model.output_keys}
        profile_pearson_counter = CumulativeCounter()
        across_batch_pearson_logger = {
            k: CumulativePearson() for k in model.output_keys
        }

        example_batches = []  # collect example batches for making images
        data_collector = []  # collect data for further analysis
        for batch_id, batch in enumerate(dataloader):
            with self.get_autocast():
                batch, _ = self._model_forward_pass(model, batch)
            # as is in scPrinter
            batch["true_footprint"] = torch.nan_to_num(batch["true_footprint"], nan=0.0)

            # ==========
            # Loss and Pearson correlation
            # ==========
            for k in model.output_keys:
                loss_logger[k].update(batch[f"loss_{k}"])
                across_batch_pearson_logger[k].update(
                    batch[f"pred_{k}"], batch[f"true_{k}"]
                )

            corr = (
                batch_pearson_correlation(
                    batch["pred_footprint"], batch["true_footprint"]
                )
                .detach()
                .cpu()[:, None]
            )
            profile_pearson_counter.update(corr)

            if batch_id < self.plot_example_per_epoch:
                example_batches.append(batch)

            # Collect batch data for validation
            if collect_data:
                # add addtional data into batch dict
                new_batch = {}
                for k, v in batch.items():
                    if (save_keys is not None) and (k not in save_keys):
                        continue
                    if isinstance(v, torch.Tensor):
                        if k == "region":
                            v = v.float().cpu().numpy()
                        else:
                            v = v.half().cpu().numpy()
                    new_batch[k] = v
                if collect_fn is not None:
                    new_batch = collect_fn(new_batch)
                data_collector.append(new_batch)

        del dataloader
        self._cleanup_env()

        # ==========
        # Save val results
        # ==========
        self.val_images = self._plot_example_footprints(example_batches)
        self.val_loss = {k: loss_logger[k].mean() for k in model.output_keys}
        # check nan
        for k, v in self.val_loss.items():
            if np.isnan(v):
                raise ValueError(f"Validation loss has NaN for {k}.")

        self.profile_pearson = profile_pearson_counter.mean()
        self.across_pearson = {
            k: across_batch_pearson_logger[k].corr() for k in model.output_keys
        }
        if collect_data:
            return data_collector

    def _model_forward_pass(self, model, batch):
        raise NotImplementedError

    def _plot_example_footprints(self, example_batches):
        epoch = self.cur_epoch + 1
        wandb_images = []
        for idx, batch in enumerate(example_batches):
            plotter = FootPrintExamplePlotter(
                signal=batch["true_atac"],
                bias=batch["tn5_bias"],
                target=batch["true_footprint"],
                predict=batch["pred_footprint"],
                footprinter=self.footprinter,
            )
            fig, _ = plotter.plot(figsize=(6, 2.5), dpi=100)
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

            if "true_mc" in batch:
                # TODO: add mC plot if available, append to wandb_images
                # mc plotter
                # shape (bs, mc_channel, seq_len)
                _ = batch["true_mc"], batch["pred_mc"]
                pass

        return wandb_images

    def _log_save(self):
        epoch = self.cur_epoch

        loss_str = ";\n".join([f"{k}: {v:.3f}" for k, v in self.train_loss.items()])
        print(f" - (Training) {epoch}\n{loss_str};\nLearning rate {self.cur_lr}.")
        loss_str = ";\n".join([f"{k}: {v:.3f}" for k, v in self.val_loss.items()])
        print(f" - (Validation) {epoch}\n{loss_str};")
        print(f"Profile pearson {self.profile_pearson:.3f}")
        for k, p in self.across_pearson.items():
            print(f"Across peak pearson {k} {p:.3f}")

        # determine early stop based on footprint loss
        val_fp_loss = self.val_loss["footprint"]
        print(f"Loss at epoch {epoch}: {val_fp_loss:.3f}.")

        # save epoch model state for comparing model over epochs
        save_every_n_epoch = self.config.get("save_state_every_n_epoch", None)
        if save_every_n_epoch is not None and epoch % save_every_n_epoch == 0:
            self._save_epoch_model_state()

        # save checkpoint if the loss is better
        self._save_checkpoint(update_best=True)

        wandb.log(
            {
                **{f"val/val_loss_{k}": v for k, v in self.val_loss.items()},
                "val/profile_pearson": self.profile_pearson,
                **{
                    f"val/across_pearson_{k}": v for k, v in self.across_pearson.items()
                },
                "val_example/example_footprints": self.val_images,
            }
        )
        return

    def _global_clipnorm(self):
        self.scaler.unscale_(self.optimizer)
        if self.config["global_clipnorm"] is not None:
            # clip footprint parameters
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.footprint_parameters(),
                max_norm=self.config["global_clipnorm"],
            )
            total_norm = total_norm.item()

            # clip coverage parameters
            torch.nn.utils.clip_grad_norm_(
                self.model.coverage_parameters(),
                max_norm=self.config["global_clipnorm"] * 3,
            )

        # collect grad norm if needed
        self._collect_grad_norm(self.model)
        return total_norm

    def _fit(self, max_epochs=None):
        if max_epochs is None:
            max_epochs = self.max_epochs

        # dataset related
        scaler = self.scaler
        optimizer = self.optimizer
        scheduler = self.scheduler
        ema = self.ema
        self.val_loss = None

        if self.cur_epoch > 0:
            print(
                f"Resuming training from epoch {self.cur_epoch+1}, with {max_epochs+1} epochs in total."
            )

        total_norm = 0
        loss_logger = {k: CumulativeCounter() for k in self.model.output_keys}
        while self.cur_epoch <= max_epochs:
            print(f"Current epoch: {self.cur_epoch}, max epochs: {max_epochs}.")

            # get train data loader
            dataloader = self.get_train_dataloader(batches=self.train_batches)

            # start train epochs
            print_steps = max(5, self.train_batches // 500)
            example_step = max(
                5, self.train_batches // (self.plot_example_per_epoch + 1)
            )
            for batch_id, batch in enumerate(dataloader):
                with self.get_autocast():
                    batch, loss = self._model_forward_pass(self.model, batch)

                    # take final loss and scale by accumulate_grad
                    loss = loss / self.accumulate_grad

                    if np.isnan(loss.item()):
                        raise ValueError("Training loss has NaN.")

                # ==========
                # Backward
                # ==========
                scaler.scale(loss).backward()
                # only update optimizer every accumulate_grad steps
                # this is equivalent to updating every step but with larger batch size (batch_size * accumulate_grad)
                # however, with larger batch size, the GPU memory usage will be higher
                if (batch_id + 1) % self.accumulate_grad == 0:
                    total_norm = self._global_clipnorm()

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if ema:
                        ema.update()

                    if scheduler is not None:
                        scheduler.step()
                    self.cur_lr = optimizer.param_groups[0]["lr"]

                with torch.no_grad():
                    for key in self.model.output_keys:
                        loss_logger[key].update(batch[f"loss_{key}"])

                    if ((batch_id + 1) % print_steps == 0) or (
                        (batch_id + 1) % example_step == 0
                    ):
                        log_dict = {
                            f"train/loss_{key}": batch[f"loss_{key}"]
                            for key in self.model.output_keys
                        }
                        log_dict["train/total_grad_norm"] = total_norm
                        log_dict["train/learning_rate"] = self.cur_lr

                        if (batch_id + 1) % example_step == 0:
                            # plot example footprints
                            example_images = self._plot_example_footprints([batch])
                            log_dict["train_example/example_footprints"] = (
                                example_images
                            )
                        wandb.log(log_dict)

            print(f"{batch_id+1} batches finished.")
            del dataloader
            self._cleanup_env()

            self.train_loss = {k: loss_logger[k].mean() for k in self.model.output_keys}

            # validation
            self._validation_step()
            self.cur_epoch += 1
            self._log_save()

        self._cleanup_env()
        return

    def _test(self):
        # load final best checkpoint for testing
        self._update_state_dict()

        # validation
        if self.val_loss is None:
            self._validation_step(val_batches=1500)
        for key in self.model.output_keys:
            wandb.summary[f"final_valid_loss_{key}"] = self.val_loss[key]
            wandb.summary[f"final_valid_across_pearson_{key}"] = self.across_pearson[
                key
            ]
        wandb.summary["final_valid_profile_pearson"] = self.profile_pearson

        # test
        self._validation_step(testing=True, val_batches=1500)
        for key in self.model.output_keys:
            wandb.summary[f"final_test_loss_{key}"] = self.val_loss[key]
            wandb.summary[f"final_test_across_pearson_{key}"] = self.across_pearson[key]
        wandb.summary["final_test_profile_pearson"] = self.profile_pearson
        wandb.summary["final_test_image"] = self.val_images

        # final wandb flag to indicate the run is successfully finished
        wandb.summary["success"] = True
        return

    def train(self) -> None:
        """Train the scFootprintTrainer model on LoRA mode."""
        flag = pathlib.Path(f"{self.savename}.{self.mode}.success.flag")

        if flag.exists():
            print(f"Training already finished, found flag file: {flag}.")
            return

        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            # Fit LoRA
            self.checkpoint = self._has_last_checkpoint()
            self._setup_model()
            self._setup_fit()
            self._fit()
            self._test()
            self._cleanup_env()
            wandb.finish()
        flag.touch()
        return


class scFootprintBaseTrainer(scFootprintTrainerMixin):
    """Train scFootprintBPNet base model on pseudobulk single-cell ATAC data."""

    trainer_config = scFootprintTrainerMixin.trainer_config.copy()
    trainer_config.update(
        {
            "mode": "base",
            "lr": 0.003,
            # dataset related files
            "pretrained_model": None,
            "prefix": "pseudobulk",
        }
    )

    dataset_class = scPrinterDatasetBase
    model_class = seq2PRINT

    @classmethod
    def make_config(cls, **config):
        """Make config for the trainer."""
        config["n_pseudobulks"] = 1
        config = super().make_config(**config)
        return config

    def _setup_model_from_config(self):
        print("Setting up model from config")
        model = seq2PRINT.create_from_config(self.config)
        model.to(self.device)
        return model

    def _setup_model_from_pretrain(self):
        # load model from path, set parameter to requires_grad, and model to train
        model_path = self.config["pretrained_model"]
        if model_path is None:
            raise ValueError("Pretrained model path is required.")
        print(f"Setting up model from pretrain model at {model_path}")

        model = torch.load(model_path, weights_only=False)
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return model

    def _setup_model(self):
        mode = self.mode

        if mode == "finetune":
            self.model = self._setup_model_from_pretrain()
        elif mode == "base":
            self.model = self._setup_model_from_config()
        else:
            raise ValueError(
                f"Incorrect mode: {mode}, should be one of ['base', 'finetune']."
            )

        self._set_total_params()
        return

    def _model_forward_pass(self, model: torch.nn.Module, batch: dict):
        prefix = self.prefix
        atac_key = f"{prefix}:bulk_data"
        batch["true_atac"] = batch[atac_key]
        dna_key = "dna_one_hot"
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter
        if "mc_frac" in batch:
            batch["true_mc"] = batch["mc_frac"]

        # ==========
        # X
        # ==========
        X = batch[dna_key]

        # ==========
        # y_footprint, y_coverage
        # ==========
        batch = footprinter(data=batch)
        batch["true_footprint"] = batch[footprint_key]
        atac_region_sum = batch[atac_key].sum(dim=-1)
        if atac_region_sum.ndim == 2:
            # remove the channel dim
            atac_region_sum = atac_region_sum.squeeze(1)
        batch["true_coverage"] = atac_region_sum

        # ==========
        # Forward and Loss
        # ==========
        result = model(X)
        batch.update(result)

        # clip pred_mc to the same size of true_mc
        if "mc_frac" in batch:
            clip_size = (self.dataset.dna_window - self.dataset.signal_window) // 2
            batch["pred_mc"] = batch["pred_mc"][..., clip_size:-clip_size]

        loss_dict = model.loss(batch)
        batch.update(loss_dict)
        return batch, loss_dict["loss_total"]


class scFootprintBaseTrainerOnline(scFootprintBaseTrainer):
    dataset_class = scPrinterOnlineDataset
