import pathlib

import joblib
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
        "max_epochs": 80,
        "patience": 5,
        "start_early_stop_after_epoch": 30,
        "use_amp": True,
        "use_ema": True,
        "scheduler": True,
        "lr": "REQUIRED",
        "large_lr_scale": 1,
        "optimizer": "adamw",
        "global_clipnorm": 0.2,
        "train_batches": 5000,
        "val_batches": 1000,
        "warmup_steps": 5000,
        "weight_decay": 1e-4,
        "loss_tolerance": 0.0,
        "plot_example_per_epoch": 9,
        "accumulate_grad": 8,
        "dataloader_concurrency": 16,
        "downsample_train_region": None,
        "downsample_valid_region": None,
        "downsample_test_region": None,
        "grad_norm_collector": False,
        "save_state_every_n_epoch": None,
    }

    def __init__(self, config):
        super().__init__(config)
        self.start_early_stop_after_epoch: bool = config["start_early_stop_after_epoch"]

        # the prefix of pseudobulk data in the batch dict
        # this is the pseudobulker name passed to dataset
        self.prefix = config["prefix"]

        self.model: torch.nn.Module = None
        self._setup_env()
        self._setup_dataset()
        return

    def _setup_dataset(self):
        super()._setup_dataset()

        # add footprinter
        self.footprinter = self.dataset.get_footprinter(prefix=self.prefix)

        # convert regions to peak regions
        # TrainerBorzoiDatasetMixin uses the Borzoi regions as train/valid/test regions
        # Here we need to intersect the Borzoi regions with the peak regions
        def _intersect_region_with_borzoi_regions(region_bed, borzoi_regions):
            borzoi_regions = pr.PyRanges(borzoi_regions)
            region_bed = region_bed.overlap(borzoi_regions).as_df()
            region_bed["Original_Name"] = region_bed["Name"]
            return region_bed

        region_bed = pr.read_bed(self.config["region_bed_path"])
        self.train_regions = _intersect_region_with_borzoi_regions(
            region_bed, self.train_regions
        )
        self.valid_regions = _intersect_region_with_borzoi_regions(
            region_bed, self.valid_regions
        )
        self.test_regions = _intersect_region_with_borzoi_regions(
            region_bed, self.test_regions
        )
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

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
        collect_data=False,
    ):
        print_step = max(5, val_batches // 20)
        # if val batches is None, use all batches in the dataset

        prefix = self.prefix
        atac_key = f"{prefix}:bulk_data"
        bias_key = "tn5_bias"
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter

        size = 0
        val_loss = [0, 0]
        profile_pearson_counter = CumulativeCounter()
        across_batch_pearson_fp = CumulativePearson()
        across_batch_pearson_cov = CumulativePearson()

        example_batches = []  # collect example batches for making images
        data_collector = []  # collect data for further analysis
        for batch_id, batch in enumerate(dataloader):
            (
                y_footprint,
                y_coverage,
                pred_footprint,
                pred_coverage,
                loss_footprint,
                loss_coverage,
            ) = self._model_forward_pass(model, batch)

            pred_score_img = pred_footprint.clone().detach().cpu().numpy()
            y_footprint = torch.nan_to_num(y_footprint, nan=0)
            # as is in scPrinter
            pred_footprint = pred_footprint.reshape((len(pred_footprint), -1))
            y_footprint = y_footprint.reshape((len(y_footprint), -1))

            val_loss[0] += loss_footprint.item()
            val_loss[1] += loss_coverage.item()

            # ==========
            # Within batch pearson and save for across batch pearson
            # ==========
            # within batch pearson
            corr = (
                batch_pearson_correlation(pred_footprint, y_footprint)
                .detach()
                .cpu()[:, None]
            )
            profile_pearson_counter.update(corr)
            # save for across batch pearson
            across_batch_pearson_fp.update(pred_footprint, y_footprint)
            across_batch_pearson_cov.update(pred_coverage, y_coverage)

            size += 1
            if batch_id < self.plot_example_per_epoch:
                batch["pred_score"] = pred_score_img
                example_batches.append(batch)

            if ((batch_id + 1) % print_step) == 0:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} [{batch_id}/{val_batches}] "
                    f"FP Loss: {val_loss[0]/size:.3f}; "
                    f"Cov Loss: {val_loss[1]/size:.3f}; "
                    f"Profile Pearson: {profile_pearson_counter.mean():.3f}; "
                    f"Across batch Pearson: FP {across_batch_pearson_fp.corr():.3f}; "
                    f"Cov {across_batch_pearson_cov.corr():.3f}"
                )
                print(desc_str)

            # Collect batch data for validation
            if collect_data:
                # add addtional data into batch dict
                batch_data = {
                    "pred_footprint": pred_footprint,
                    "pred_coverage": pred_coverage,
                    "loss_footprint": loss_footprint,
                    "loss_coverage": loss_coverage,
                    "true_footprint": y_footprint,
                    "true_coverage": y_coverage,
                    f"{prefix}:pseudobulk_ids": batch[f"{prefix}:pseudobulk_ids"],
                    "region": batch["region"],
                }
                data_collector.append(
                    {
                        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in batch_data.items()
                    }
                )

        del dataloader
        self._cleanup_env()

        wandb_images = self._plot_example_footprints(
            example_batches, footprinter, atac_key, bias_key, footprint_key
        )

        # ==========
        # Loss
        # ==========
        val_loss = [l / size for l in val_loss]

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
        if collect_data:
            return val_loss, profile_pearson, across_corr, wandb_images, data_collector
        else:
            return val_loss, profile_pearson, across_corr, wandb_images

    def _model_forward_pass(self, model, batch):
        raise NotImplementedError

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
        return wandb_images

    def _log_save_and_check_stop(self, example_images):
        epoch = self.cur_epoch
        train_fp_loss = self.train_fp_loss
        train_cov_loss = self.train_cov_loss
        learning_rate = self.cur_lr
        val_fp_loss, val_cov_loss = self.val_loss
        profile_pearson = self.val_profile_pearson
        across_pearson = self.val_across_pearson

        print(
            f" - (Training) {epoch} FP Loss: {train_fp_loss:.3f}; "
            f"Cov Loss: {train_cov_loss:.3f}; Learning rate {learning_rate}."
        )
        print(
            f" - (Validation) {epoch} FP Loss: {val_fp_loss:.3f}; Cov Loss: {val_cov_loss:.3f}"
        )
        print(f"Profile pearson {profile_pearson[0]:.3f}")
        print(f"Across peak pearson footprint {across_pearson[0]:.3f}")
        print(f"Across peak pearson coverage {across_pearson[1]:.3f}")

        # only clear the early stopping counter if the loss improvement is better than tolerance
        previous_best = self.best_val_loss
        if val_fp_loss < self.best_val_loss - self.loss_tolerance:
            self.early_stopping_counter = 0
        else:
            if epoch >= self.start_early_stop_after_epoch:
                self.early_stopping_counter += 1
        print(
            f"Previous best loss: {previous_best:.3f}, "
            f"Loss at epoch {epoch}: {val_fp_loss:.3f}; "
            f"Early stopping counter: {self.early_stopping_counter}"
        )
        # save checkpoint if the loss is better
        if epoch < self.start_early_stop_after_epoch:
            self.best_val_loss = val_fp_loss
            self._save_checkpint(update_best=True)
        else:
            if val_fp_loss < self.best_val_loss:
                self.best_val_loss = val_fp_loss
                self._save_checkpint(update_best=True)
            else:
                self._save_checkpint(update_best=False)

        wandb.log(
            {
                "train/train_fp_loss": train_fp_loss,
                "train/train_cov_loss": train_cov_loss,
                "val/val_fp_loss": val_fp_loss,
                "val/val_cov_loss": val_cov_loss,
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

    def _fit(self, max_epochs=None):
        atac_key = f"{self.prefix}:bulk_data"
        bias_key = "tn5_bias"
        footprint_key = f"{self.prefix}:bulk_data_footprint"

        if max_epochs is None:
            max_epochs = self.max_epochs

        # dataset related
        scaler = self.scaler
        optimizer = self.optimizer
        scheduler = self.scheduler
        ema = self.ema
        self.val_loss = None

        stop_flag = self.early_stopping_counter >= self.patience
        if self.cur_epoch > 0:
            print(
                f"Resuming training from epoch {self.cur_epoch+1}, with {max_epochs+1} epochs in total."
            )
        while self.cur_epoch <= max_epochs and not stop_flag:
            # one can manually create a stop flag file to stop the training
            # path: f"{self.savename}.stop.flag"
            if self._check_stage_flag("stop"):
                print(
                    f"Early stopping flag file found, stopping training at {self.cur_epoch}."
                )
                self.early_stoped = True
                break

            print(
                f"Current epoch: {self.cur_epoch}, max epochs: {max_epochs}, stop flag: {stop_flag}."
            )
            # check early stop
            if self.early_stopping_counter >= self.patience:
                # early stopping counter could be loaded from the checkpoint
                # check before starting the for loop
                print(f"Early stopping at epoch {self.cur_epoch}")
                self.early_stoped = True
                break

            # get train data loader
            dataloader = self.get_train_dataloader(batches=self.train_batches)

            # start train epochs
            moving_avg_fp_loss = 0
            moving_avg_cov_loss = 0
            cur_cov_loss = 1e10
            cur_fp_loss = 1e10
            nan_loss = False

            print_steps = max(5, self.train_batches // 50)
            example_step = max(
                5, self.train_batches // (self.plot_example_per_epoch + 1)
            )
            for batch_id, batch in enumerate(dataloader):
                try:
                    auto_cast_context = torch.autocast(
                        device_type=str(self.device).split(":")[0],
                        dtype=torch.bfloat16,
                        enabled=self.use_amp,
                    )
                except RuntimeError:
                    # some GPU, such as T4 does not support bfloat16
                    auto_cast_context = torch.autocast(
                        device_type=str(self.device).split(":")[0],
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    )

                with auto_cast_context:
                    *_, pred_fp, _, loss_footprint, loss_coverage = (
                        self._model_forward_pass(self.model, batch)
                    )
                    batch["pred_score"] = pred_fp.detach().float().cpu().numpy()

                    # because coverage_head is detached from other part of the model
                    # here we just add the loss without lambda weight
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
                moving_avg_fp_loss += loss_footprint.item()
                moving_avg_cov_loss += loss_coverage.item()
                # only update optimizer every accumulate_grad steps
                # this is equivalent to updating every step but with larger batch size (batch_size * accumulate_grad)
                # however, with larger batch size, the GPU memory usage will be higher
                if (batch_id + 1) % self.accumulate_grad == 0:
                    scaler.unscale_(optimizer)
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

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if ema:
                        ema.update()

                    if scheduler is not None:
                        scheduler.step()

                with torch.no_grad():
                    if (batch_id + 1) % print_steps == 0:
                        log_dict = {
                            "train/fp_loss": loss_footprint.item(),
                            "train/cov_loss": loss_coverage.item(),
                            "train/total_grad_norm": total_norm,
                        }
                        wandb.log(log_dict)

                        _fp_loss = moving_avg_fp_loss / (batch_id + 1)
                        _cov_loss = moving_avg_cov_loss / (batch_id + 1)
                        desc_str = (
                            f" - (Training) {self.cur_epoch} [{batch_id}/{self.train_batches}] "
                            f"Ave FP Loss: {_fp_loss:.3f} "
                            f"Ave Cov Loss: {_cov_loss:.3f} "
                            f"Last grad norm: {total_norm:.4f} "
                            f"Last FP Loss: {loss_footprint.item():.3f} "
                            f"Last Cov Loss: {loss_coverage.item():.3f}"
                        )

                        if _fp_loss > (cur_fp_loss + 0.5):
                            batch["cur_fp_loss"] = _fp_loss
                            batch["last_fp_loss"] = cur_fp_loss
                            batch["cur_cov_loss"] = _cov_loss
                            batch["last_cov_loss"] = cur_cov_loss
                            print(f"Batch {batch_id} loss increased.")
                            joblib.dump(
                                batch,
                                f"{self.savename}.epoch{self.cur_epoch}.batch{batch_id}.joblib",
                            )

                        cur_fp_loss = _fp_loss
                        cur_cov_loss = _cov_loss
                        print(desc_str)

                    if (batch_id + 1) % example_step == 0:
                        # plot example footprints
                        example_images = self._plot_example_footprints(
                            [batch],
                            self.footprinter,
                            atac_key=atac_key,
                            bias_key=bias_key,
                            footprint_key=footprint_key,
                        )
                        wandb.log({"train_example/example_footprints": example_images})

            del dataloader
            self._cleanup_env()
            if nan_loss:
                # epoch break due to nan loss, skip validation
                continue

            self.train_fp_loss = moving_avg_fp_loss / (batch_id + 1)
            self.train_cov_loss = moving_avg_cov_loss / (batch_id + 1)
            self.cur_lr = optimizer.param_groups[0]["lr"]
            (
                self.val_loss,
                self.val_profile_pearson,
                self.val_across_pearson,
                wandb_images,
            ) = self._validation_step()

            if np.isnan(self.val_loss[0]):
                print("Validation loss is NaN, skipping epoch.")
                self._update_state_dict()
                continue

            self.cur_epoch += 1
            stop_flag = self._log_save_and_check_stop(example_images=wandb_images)
            if stop_flag:
                print(f"Early stopping at epoch {self.cur_epoch}")
                self.early_stoped = True
                break

        self._cleanup_env()
        return

    def _test(self):
        # load final best checkpoint for testing
        self._update_state_dict()

        (
            self.val_loss,
            self.val_profile_pearson,
            self.val_across_pearson,
            _,
        ) = self._validation_step(val_batches=1500)
        valid_across_pearson_footprint, valid_across_pearson_coverage = (
            self.val_across_pearson
        )
        (
            self.test_loss,
            self.test_profile_pearson,
            self.test_across_pearson,
            wandb_images,
        ) = self._validation_step(testing=True, val_batches=1500)
        test_across_pearson_footprint, test_across_pearson_coverage = (
            self.test_across_pearson
        )

        wandb.summary["final_valid_fp_loss"] = self.val_loss[0]
        wandb.summary["final_valid_cov_loss"] = self.val_loss[1]
        wandb.summary["final_valid_within"] = self.val_profile_pearson[0]
        wandb.summary["final_valid_across"] = valid_across_pearson_footprint
        wandb.summary["final_valid_cov"] = valid_across_pearson_coverage
        wandb.summary["final_test_fp_loss"] = self.test_loss[0]
        wandb.summary["final_test_cov_loss"] = self.test_loss[1]
        wandb.summary["final_test_within"] = self.test_profile_pearson[0]
        wandb.summary["final_test_across"] = test_across_pearson_footprint
        wandb.summary["final_test_cov"] = test_across_pearson_coverage
        wandb.summary["final_image"] = wandb_images

        # final wandb flag to indicate the run is successfully finished
        wandb.summary["success"] = True
        return

    def train(self):
        """Train function should be implemented in the subclass."""
        raise NotImplementedError


class scFootprintBaseTrainer(scFootprintTrainerMixin):
    """Train scFootprintBPNet base model on pseudobulk single-cell ATAC data."""

    trainer_config = scFootprintTrainerMixin.trainer_config.copy()
    trainer_config.update(
        {
            "mode": "base",
            "lr": 0.003,  # use 0.003 for base init, 0.0003 for fine-tune
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
        config["cov_filter_name"] = cls.trainer_config["prefix"]
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
        dna_key = "dna_one_hot"
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter

        # ==========
        # X
        # ==========
        X = batch[dna_key]

        # ==========
        # y_footprint, y_coverage
        # ==========
        batch = footprinter(data=batch)
        y_footprint = batch[footprint_key]
        y_coverage = batch[atac_key].sum(dim=-1)

        # ==========
        # Forward and Loss
        # ==========
        pred_footprint, pred_coverage = model(X)
        fp_loss, cov_loss = model.loss(
            y_footprint=y_footprint,
            y_coverage=y_coverage,
            pred_footprint=pred_footprint,
            pred_coverage=pred_coverage,
        )
        return y_footprint, y_coverage, pred_footprint, pred_coverage, fp_loss, cov_loss

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
