import pathlib

import numpy as np
import ray
import torch
import wandb

from bolero.tl.generic.train import GenericTrainer
from bolero.tl.generic.train_helper import CumulativeCounter
from bolero.tl.model.borzoi.metrics import (
    MeanPearsonCorrCoefPerChannel,
)
from bolero.tl.model.borzoi.train import BorzoiTrainerMixin
from bolero.tl.model.borzoi.utils import MovingMetric
from bolero.tl.model.creformer.dataset import CREFormerDataset
from bolero.tl.model.creformer.model_lora import CREFormerLoRA
from bolero.tl.pseudobulk.rna_atac_pseudobulk import RNAVQPseudobulker


class TrainerCREFormerDatasetMixin:
    """
    Mixin class for managing datasets used in the trainer.

    Attributes
    ----------
    dataset_class : type
        The class of the generic dataset.
    train_dataset : BorzoiDataset
        The training dataset.
    valid_dataset : BorzoiDataset
        The validation dataset.
    test_dataset : BorzoiDataset
        The test dataset.
    config : dict
        The configuration dictionary.

    Methods
    -------
    _setup_dataset()
        Set up the dataset by splitting it into train, valid, and test sets.
    _get_dataset_paths(_folds)
        Get the paths of the dataset files for the given folds.
    train_dataset
        Property that returns the training dataset.
    valid_dataset
        Property that returns the validation dataset.
    test_dataset
        Property that returns the test dataset.
    _get_dataset()
        Get the dataset object for the given folds.
    get_train_dataloader(batches)
        Get the training dataloader.
    get_valid_dataloader(batches)
        Get the validation dataloader.
    get_test_dataloader(batches)
        Get the test dataloader.
    """

    dataset_class = CREFormerDataset

    def _setup_dataset(self):
        """
        Set up the dataset by splitting it into train, valid, and test sets.
        """
        # create dataset
        self.dataset: CREFormerDataset = self._get_dataset()

        # train, valid, test split by fold
        (self.train_folds, self.valid_folds, self.test_folds) = (
            self.dataset.get_train_valid_test(self.config["fold_split_id"])
        )
        return

    def _get_dataset(self) -> CREFormerDataset:
        """
        Get the dataset object for the given folds.

        Returns
        -------
        BorzoiDataset
            The dataset object.
        """
        dataset = self.dataset_class.create_from_config(self.config)
        return dataset

    def get_train_dataloader(self, batches: int):
        """Get the training dataloader.

        Parameters
        ----------
        batches : int
            Number of batches to load.

        Returns
        -------
        torch.utils.data.DataLoader
            The training dataloader.
        """
        self.dataset.train()
        dataloader = self.dataset.get_dataloader(
            folds=self.train_folds,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config.get("shuffle_rows", None),
        )
        return dataloader

    def get_valid_dataloader(self, batches: int):
        """Get the validation dataloader.

        Parameters
        ----------
        batches : int
            Number of batches to load.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            folds=self.valid_folds,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config.get("shuffle_rows", None),
        )
        return dataloader

    def get_test_dataloader(self, batches: int):
        """Get the test dataloader.

        Parameters
        ----------
        batches : int
            Number of batches to load.

        Returns
        -------
        torch.utils.data.DataLoader
            The test dataloader.
        """
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            folds=self.test_folds,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config.get("shuffle_rows", None),
        )
        return dataloader


class CREFormerTrainerMixin(
    TrainerCREFormerDatasetMixin, BorzoiTrainerMixin, GenericTrainer
):
    trainer_config = {
        "mode": "REQUIRED",
        "fold_split_id": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_group": None,
        "wandb_name": None,
        "max_epochs": 100,
        "patience": 5,
        "start_early_stop_after_epoch": 20,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": "REQUIRED",
        "optimizer": "adamw",
        "weight_decay": 1e-7,
        "global_clipnorm": 1,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "loss_tolerance": 0.0,
        "plot_example_per_epoch": 9,
        "accumulate_grad": 4,
        "shuffle_rows": 300,
        "dataloader_concurrency": 24,
        "downsample_train_region": None,
        "downsample_valid_region": None,
        "downsample_test_region": None,
        "grad_norm_collector": False,
        "save_state_every_n_epoch": None,
        "vq_records": "REQUIRED",
        "downsample_vq": None,
    }

    def _get_dataset(self):
        dataset = super()._get_dataset()

        # setup pseudobulker params for sc dataset
        pseudobulker_params = {
            "vq_records": self.config["vq_records"],
            "use_vq_emb": False,
            "prefix_name": "pseudobulk",
            "downsample_vq": self.config["downsample_vq"],
        }

        self.prefix = "pseudobulk"
        dataset.add_pseudobulker(
            name="pseudobulk",
            cls=RNAVQPseudobulker,
            pseudobulker_kwargs=pseudobulker_params,
        )
        return dataset

    # =============================
    # Model training and validation
    # =============================
    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
        collect_data=False,
    ):
        if val_batches is None:
            print_step = 100
        else:
            print_step = max(5, val_batches // 10)

        # if val batches is None, use all batches in the dataset
        size = 0
        mean_val_loss = CumulativeCounter()

        mean_loss_breakdown = {}
        mean_val_corr = MeanPearsonCorrCoefPerChannel(n_channels=1).to(self.device)

        data_collector = []  # collect data for further analysis
        for batch_id, batch in enumerate(dataloader):
            with self._autocast_context():
                y_true, y_pred, loss = self._model_forward_pass(model, batch)

            mean_val_corr.update(target=y_true, preds=y_pred)
            mean_val_loss.update(loss)

            # add additional data into batch dict
            batch["true_data"] = y_true
            batch["pred_data"] = y_pred

            size += 1
            if ((batch_id + 1) % print_step) == 0:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} [{batch_id}/{val_batches}] "
                    f"Mean Loss: {mean_val_loss.mean():.3f}; "
                    f"Mean Track Corr: {mean_val_corr.get_corr_str()}; "
                )
                print(desc_str)

            if collect_data:
                data_collector.append(
                    {
                        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                )

        del dataloader
        self._cleanup_env()

        # ==========
        # Final metrics
        # ==========
        val_loss = mean_val_loss.mean()
        val_loss_breakdown = {k: v.mean() for k, v in mean_loss_breakdown.items()}
        val_corr = mean_val_corr

        if collect_data:
            return val_loss, val_loss_breakdown, val_corr, data_collector
        else:
            return val_loss, val_loss_breakdown, val_corr

    def _validation_step(self, testing=False, val_batches=None):
        """Generic validation step."""
        val_batches = val_batches or self.val_batches
        if testing:
            dataloader = self.get_test_dataloader(batches=val_batches * 3)
        else:
            dataloader = self.get_valid_dataloader(batches=val_batches)

        with torch.inference_mode():
            if self.use_ema:
                self.ema.eval()
                self.ema.ema_model.eval()
                val_loss, val_loss_breakdown, val_corr = self._model_validation_step(
                    model=self.ema.ema_model,
                    dataloader=dataloader,
                    val_batches=val_batches,
                )
            else:
                self.model.eval()
                val_loss, val_loss_breakdown, val_corr = self._model_validation_step(
                    model=self.model,
                    dataloader=dataloader,
                    val_batches=val_batches,
                )
                self.model.train()
        return val_loss, val_loss_breakdown, val_corr

    def _get_optimizer(self):
        optimizer_type = self.config["optimizer"]
        parameters = self.model.parameters()
        lr = self.config["lr"]
        weight_decay = self.config["weight_decay"]

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params=parameters, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params=parameters, lr=lr, weight_decay=weight_decay
            )
        return optimizer

    def _log_save_and_check_stop(self):
        epoch = self.cur_epoch
        train_loss = self.train_loss
        train_corr = self.train_corr
        learning_rate = self.cur_lr
        val_loss = self.val_loss
        val_corr = self.val_corr

        print(
            f" - (Training)   {epoch} Loss: {train_loss:.3f}; Mean Corr. : {train_corr.get_corr_str()}; "
            f"Learning rate {learning_rate}."
        )
        print(
            f" - (Validation) {epoch} Loss: {val_loss:.3f}; Mean Corr. : {val_corr.get_corr_str()}."
        )

        larger_is_better = True  # use corr as the metric, larger is better
        metric_to_use = val_corr.compute_tensor().mean().item()
        previous_best = self.best_val_metric
        if larger_is_better:
            improved = metric_to_use > self.best_val_metric
        else:
            improved = metric_to_use < self.best_val_metric

        # only clear the early stopping counter if the loss improvement is better than tolerance
        if improved:
            self.early_stopping_counter = 0
        else:
            if epoch >= self.start_early_stop_after_epoch:
                self.early_stopping_counter += 1
            else:
                print(
                    f"Early stopping counter is not updated before "
                    f"start_early_stop_after_epoch {self.start_early_stop_after_epoch}."
                )
                self.early_stopping_counter = 0
        print(
            f"Previous best metric: {previous_best:.3f}, "
            f"Metric at epoch {epoch}: {metric_to_use:.3f}; "
            f"Early stopping counter: {self.early_stopping_counter}"
        )
        # save checkpoint if the loss is better
        if epoch < self.start_early_stop_after_epoch:
            self.best_val_metric = metric_to_use
            self._save_checkpoint(update_best=True)
        else:
            if improved:
                self.best_val_metric = metric_to_use
                self._save_checkpoint(update_best=True)
            else:
                self._save_checkpoint(update_best=False)

        # save epoch model state for comparing model over epochs
        save_every_n_epoch = self.config.get("save_state_every_n_epoch", None)
        if save_every_n_epoch is not None and epoch % save_every_n_epoch == 0:
            self._save_epoch_model_state()

        # construct wandb log dict
        log_dict = {
            "val/val_loss": val_loss,
            "val/best_val_corr": self.best_val_metric,
            "val/early_stopping_counter": self.early_stopping_counter,
        }

        for channel, corr in enumerate(val_corr.compute_tensor()):
            log_dict[f"val/val_corr_{channel}"] = corr
        wandb.log(log_dict)

        flag = self.early_stopping_counter >= self.patience
        return flag

    def _fit(self, max_epochs=None):
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
                f"Resuming training from epoch {self.cur_epoch+1}, with {max_epochs} epochs in total."
            )

        window_size = 6400 // self.accumulate_grad
        moving_norm = MovingMetric(window_size=window_size)
        total_norm = 999
        while self.cur_epoch < max_epochs and not stop_flag:
            # one can manually create a stop flag file to stop the training
            # path: f"{self.savename}.stop.flag"
            if self._check_stage_flag("stop"):
                print(
                    f"Early stopping flag file found, stopping training at {self.cur_epoch}."
                )
                self.early_stoped = True
                break

            print(
                f"Current epoch: {self.cur_epoch + 1}, max epochs: {max_epochs}, stop flag: {stop_flag}."
            )
            # check early stop
            if self.early_stopping_counter >= self.patience:
                # early stopping counter could be loaded from the checkpoint
                # check before starting the for loop
                print(f"Early stopping at epoch {self.cur_epoch}")
                self.early_stoped = True
                break

            # get train data loader
            dataloader = self.get_train_dataloader(batches=self.train_batches + 1)

            # start train epochs
            moving_ave_loss = CumulativeCounter()
            moving_ave_corr = MeanPearsonCorrCoefPerChannel(n_channels=1).to(
                self.device
            )
            nan_loss = False
            print_steps = max(5, self.train_batches // 50)
            for batch_id, batch in enumerate(dataloader):
                with self._autocast_context():
                    y_true, y_pred, loss = self._model_forward_pass(self.model, batch)

                    if np.isnan(loss.item()):
                        nan_loss = True
                        print("Training loss has NaN, skipping epoch.")
                        self._update_state_dict()
                        break

                # ==========
                # Backward
                # ==========
                # for backpropagation, we scale the loss with the accumulate_grad
                scale_loss = loss / self.accumulate_grad
                scaler.scale(scale_loss).backward()

                moving_ave_loss.update(loss.item())
                moving_ave_corr.update(preds=y_pred, target=y_true)

                # only update optimizer every accumulate_grad steps
                # this is equivalent to updating every step but with larger batch size (batch_size * accumulate_grad)
                # however, with larger batch size, the GPU memory usage will be higher
                if (batch_id + 1) % self.accumulate_grad == 0:
                    if self.config["global_clipnorm"] is not None:
                        scaler.unscale_(optimizer)
                        # use the first group (major group) as the total norm
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config["global_clipnorm"],
                        )
                        total_norm = total_norm.item()

                        # check moving norm and skip step if the norm is too large (e.g. > 99% moving quantile)
                        # this is to prevent outlier gradients from messing up the training
                        if moving_norm.full:
                            threshold = moving_norm.quantile(0.99).item() * 2
                            if total_norm > threshold:
                                print(
                                    f"Gradient norm is too large: {total_norm:.4f}, "
                                    f"threshold: {threshold:.4f}, prevent update."
                                )
                                optimizer.zero_grad()
                            moving_norm.update(total_norm)
                        else:
                            moving_norm.update(total_norm)
                    self._collect_grad_norm(self.model)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # update scheduler per optimizer step
                    if scheduler is not None:
                        scheduler.step()

                    if ema:
                        ema.update()

                with torch.no_grad():
                    if (batch_id + 1) % print_steps == 0:
                        _corr = moving_ave_corr.get_corr_str()
                        _loss = moving_ave_loss.mean()

                        desc_str = (
                            f" - (Training) {self.cur_epoch} [{batch_id}/{self.train_batches}] "
                            f"Ave Loss: {_loss:.3f}; "
                            f"Ave Corr: {_corr} "
                            f"Last grad norm: {total_norm:.4f} "
                            f"Last batch Loss: {loss.item():.3f} "
                        )
                        print(desc_str)

                        log_dict = {
                            "train/train_loss": loss.item(),
                            "train/train_total_grad_norm": total_norm,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                        }
                        for channel, corr in enumerate(
                            moving_ave_corr.compute_tensor()
                        ):
                            log_dict[f"train/train_corr_{channel}"] = corr
                        wandb.log(log_dict)

            # end of epoch clear any remaining gradients
            optimizer.zero_grad()

            del dataloader
            self._cleanup_env()
            if nan_loss:
                # epoch break due to nan loss, skip validation
                continue

            self.train_loss = moving_ave_loss.mean()
            self.train_corr = moving_ave_corr
            self.cur_lr = optimizer.param_groups[0]["lr"]
            (
                self.val_loss,
                self.val_loss_breakdown,
                self.val_corr,
            ) = self._validation_step()

            if np.isnan(self.val_loss):
                print("Validation loss is NaN, skipping epoch.")
                self._update_state_dict()
                continue

            self.cur_epoch += 1
            stop_flag = self._log_save_and_check_stop()
            if stop_flag:
                print(f"Early stopping at epoch {self.cur_epoch}")
                self.early_stoped = True
                break

        self._cleanup_env()
        return


class CREFormerLoRATrainer(CREFormerTrainerMixin):
    """Train LoRA model on pseudobulk single-cell ATAC data."""

    trainer_config = CREFormerTrainerMixin.trainer_config.copy()
    trainer_config.update(
        {
            "mode": "lora",
            "lr": 5e-5,
            "warmup_steps": 1000,
            "scheduler": True,
        }
    )

    dataset_class = CREFormerDataset
    model_class = CREFormerLoRA

    def _setup_model(self, print_model=True):
        print("Setting up model from config")
        self.model = self.model_class.create_from_config(self.config)

        self.model.to(self.device)
        if print_model:
            print(self.model)
        self._set_total_params()
        return

    @staticmethod
    def _post_process_batch(batch):
        batch = {k: v[0] for k, v in batch.items()}
        batch["tss_pos"] = batch["tss_pos"][0]
        batch["gene_strand"] = batch["gene_strand"][0]
        n_peaks = batch["n_peaks"][0]

        atac = batch["atac_in"][0, :n_peaks, :]  # shape (n_peaks, 1024)
        batch["atac_in"] = torch.from_numpy(atac).to(torch.int32).cuda()

        dna = batch["dna_in"][0, :n_peaks, :]  # shape (n_peaks, 1024)
        batch["dna_in"] = torch.from_numpy(dna).to(torch.int32).cuda()

        batch["gene_cpm_data"] = (
            torch.from_numpy(batch["gene_cpm_data"]).to(torch.float32).cuda()
        )
        return batch

    def _model_forward_pass(self, model: CREFormerLoRA, batch: dict):
        # ==========
        # Get batch data
        # ==========
        batch = self._post_process_batch(batch)
        atac = batch["atac_in"]
        dna = batch["dna_in"]
        y_true = batch["gene_cpm_data"]
        tss_loc = batch["tss_pos"]
        direction = batch["gene_strand"]

        # ==========
        # Forward and Loss
        # ==========
        result = model(
            dna_in=dna,
            atac_in=atac,
            tss_loc=tss_loc,
            direction=direction,
        )
        y_pred = result["gene_count"]
        loss = model.loss(y_pred=y_pred, y_true=y_true)

        y_pred = y_pred.detach()

        # add batch and channel dimension for logging
        y_true = torch.log1p(y_true[None, None, :])
        y_pred = torch.log1p(y_pred[None, None, :])
        return y_true, y_pred, loss

    def _test(self):
        # load final best checkpoint for testing
        self._update_state_dict()

        if self.val_loss is None:
            self.val_loss, self.val_loss_breakdown, self.val_corr = (
                self._validation_step()
            )

        self.test_loss, self.test_loss_breakdown, self.test_corr = (
            self._validation_step(testing=True)
        )

        wandb.summary["final_valid_loss"] = self.val_loss
        for k, v in self.val_loss_breakdown.items():
            wandb.summary[f"final_valid_loss_{k}"] = v
        wandb.summary["final_valid_corr"] = self.val_corr.compute().mean()
        wandb.summary["final_test_loss"] = self.test_loss
        for k, v in self.test_loss_breakdown.items():
            wandb.summary[f"final_test_loss_{k}"] = v
        wandb.summary["final_test_corr"] = self.test_corr.compute().mean()
        # final wandb flag to indicate the run is successfully finished
        wandb.summary["success"] = True
        return

    def train(self) -> None:
        """Train the Borzoi LoRA model."""
        flag = pathlib.Path(f"{self.savename}.{self.mode}.success.flag")

        if flag.exists():
            print(f"Training already finished, found flag file: {flag}.")
            return

        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            self.checkpoint = self._has_last_checkpoint()
            self._setup_model()
            self._setup_fit()
            self._fit()
            self._test()
            self._cleanup_env()
            wandb.finish()
        flag.touch()
        return


class CREFormerTesterMixin:
    def _setup_model(self):
        super()._setup_model(print_model=False)
        self.model.load_checkpoint_from_path(self.config["checkpoint_path"])
        self.model.eval()
        self.model.to(self.device)
        return

    @staticmethod
    def save_batches(data_batches, saveas, num_rows_per_file=100):
        """Save the data batches to parquet."""
        dataset = ray.data.from_items(data_batches)
        dataset.write_parquet(saveas, num_rows_per_file=num_rows_per_file)
        return

    @torch.inference_mode()
    def test(self, saveas=None, batches=None):
        """Test the Borzoi LoRA model."""
        self._setup_model()

        dataloader = self.get_test_dataloader(batches=batches)
        *_, data_batches = self._model_validation_step(
            model=self.model,
            dataloader=dataloader,
            val_batches=None,
            collect_data=True,
        )
        self._cleanup_env()

        if saveas is None:
            return data_batches
        else:
            self.save_batches(data_batches, saveas)
        return


class CREFormerLoRATester(CREFormerTesterMixin, CREFormerLoRATrainer):
    trainer_config = CREFormerLoRATrainer.trainer_config.copy()
    trainer_config["checkpoint_path"] = "REQUIRED"
