import pathlib

import numpy as np
import ray
import torch
import wandb

from bolero.pl.borzoi import BorzoiExamplePlotter
from bolero.tl.generic.train import GenericTrainer
from bolero.tl.generic.train_helper import CumulativeCounter
from bolero.tl.model.borzoi.dataset import BorzoiDataset
from bolero.tl.model.borzoi.metrics import (
    MeanPearsonCorrCoefPerChannel,
)
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.pseudobulk.rna_atac_pseudobulk import RNAVQPseudobulker

from .utils import MovingMetric, reverse_clamp_sqrt


class TrainerBorzoiDatasetMixin:
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
    train_folds : List[str]
        The list of folds used for training.
    valid_folds : List[str]
        The list of folds used for validation.
    test_folds : List[str]
        The list of folds used for testing.
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

    dataset_class = BorzoiDataset

    def _setup_dataset(self):
        """
        Set up the dataset by splitting it into train, valid, and test sets.
        """
        # create dataset
        self.dataset: BorzoiDataset = self._get_dataset()

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

    def _get_dataset_paths(self, _folds: list) -> list:
        """
        Get the paths of the dataset files for the given folds.

        Parameters
        ----------
        _folds : list
            List of folds to get the dataset paths for.

        Returns
        -------
        list
            List of dataset paths.
        """
        # check if the file exists in gcs bucket
        dataset_paths = []
        for fold in _folds:
            dataset_path = self.config["dataset_path"]
            path = f"{dataset_path}/{fold}"
            if self.fs.get_file_info(path).type:
                # type is True only if the file exists
                dataset_paths.append(path)
        return dataset_paths

    def _get_dataset(self) -> BorzoiDataset:
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
            region_bed=self.train_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config["shuffle_rows"],
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
            region_bed=self.valid_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config["shuffle_rows"],
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
            region_bed=self.test_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config["shuffle_rows"],
        )
        return dataloader


class BorzoiTrainerMixin(TrainerBorzoiDatasetMixin, GenericTrainer):
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
        "patience": 10,
        "start_early_stop_after_epoch": 30,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": "REQUIRED",
        "optimizer": "adamw",
        "weight_decay": 1e-7,
        "global_clipnorm": 0.1,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "loss_tolerance": 0.0,
        "plot_example_per_epoch": 6,
        "accumulate_grad": 32,
        "shuffle_rows": 300,
        "dataloader_concurrency": 24,
        "downsample_train_region": None,
        "downsample_valid_region": None,
        "downsample_test_region": None,
        "grad_norm_collector": False,
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

        self.best_val_metric = -1  # corr
        return

    # =============================
    # Model training and validation
    # =============================
    def _setup_model(self):
        raise NotImplementedError

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
            example_step = 100
        else:
            print_step = max(5, val_batches // 10)
            example_step = max(5, val_batches // (self.plot_example_per_epoch + 1))
        # if val batches is None, use all batches in the dataset
        size = 0
        mean_val_loss = CumulativeCounter()
        mean_loss_breakdown = {
            "multinomial": CumulativeCounter(),
            "poisson": CumulativeCounter(),
        }
        mean_val_corr = MeanPearsonCorrCoefPerChannel(n_channels=model.out_channels).to(
            self.device
        )

        pseudobulker = self.dataset.name_to_pseudobulker[self.prefix]

        example_batches = []  # collect example batches for making images
        data_collector = []  # collect data for further analysis

        for batch_id, batch in enumerate(dataloader):
            y_true, y_pred, loss, loss_breakdown = self._model_forward_pass(
                model, batch
            )

            mean_val_corr.update(target=y_true, preds=y_pred)
            for k, v in loss_breakdown.items():
                mean_loss_breakdown[k].update(v)
            mean_val_loss.update(loss)

            # add additional data into batch dict
            batch["true_data"] = y_true
            batch["pred_data"] = y_pred
            id_array = batch[f"{self.prefix}:pseudobulk_ids"].cpu().numpy()
            batch["sample_id"] = pseudobulker.pseudobulk_ids[id_array]
            batch.update(loss_breakdown)
            if collect_data:
                data_collector.append(
                    {
                        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                )

            size += 1
            if batch_id % example_step == 0:
                example_batches.append(batch)

            if ((batch_id + 1) % print_step) == 0:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} [{batch_id}/{val_batches}] "
                    f"Mean Loss: {mean_val_loss.mean():.3f}; "
                    f"Mean Track Corr: {mean_val_corr.compute().mean():.3f}; "
                )
                print(desc_str)

        del dataloader
        self._cleanup_env()

        wandb_images = self._plot_example(example_batches)

        # ==========
        # Final metrics
        # ==========
        val_loss = mean_val_loss.mean()
        val_loss_breakdown = {k: v.mean() for k, v in mean_loss_breakdown.items()}
        val_corr = mean_val_corr.compute().cpu().numpy()

        if collect_data:
            return val_loss, val_loss_breakdown, val_corr, wandb_images, data_collector
        else:
            return val_loss, val_loss_breakdown, val_corr, wandb_images

    def _model_forward_pass(self, model, batch):
        raise NotImplementedError

    def _plot_example(self, example_batches):
        epoch = self.cur_epoch + 1
        wandb_images = []
        for idx, batch in enumerate(example_batches):
            if idx >= self.plot_example_per_epoch:
                break

            power = self.config.get("cov_power", None)
            threshold = self.config.get("cov_soft_clamp", None)
            if (power is not None) and (threshold is not None):
                soft_clamp = True

            plotter = BorzoiExamplePlotter(
                genome=self.dataset.genome,
                zoomin_radius=1000,
                true_key="true_data",
                pred_key="pred_data",
                id_key="sample_id",
                power=power,
                threshold=threshold,
            )
            fig = plotter.plot(
                batch, channel=0, nrows=2, return_array=True, soft_clamp=soft_clamp
            )

            wandb_images.append(
                wandb.Image(
                    fig,
                    mode="RGB",
                    caption=f"Epoch {epoch} Example {idx}",
                    grouping=epoch,
                    file_type="jpg",  # reduce file size
                )
            )
        return wandb_images

    def _log_save_and_check_stop(self):
        train_imgs = self.train_wandb_images
        val_imgs = self.val_wandb_images
        epoch = self.cur_epoch
        train_loss = self.train_loss
        train_corr = self.train_corr.mean()
        learning_rate = self.cur_lr
        val_loss = self.val_loss
        val_loss_breakdown = self.val_loss_breakdown
        val_corr = self.val_corr.mean()

        print(
            f" - (Training)   {epoch} Loss: {train_loss:.3f}; Mean Corr. : {train_corr:.3f}; "
            f"Learning rate {learning_rate}."
        )
        print(
            f" - (Validation) {epoch} Loss: {val_loss:.3f}; Mean Corr. : {val_corr:.3f}."
        )

        larger_is_better = True  # use corr as the metric, larger is better
        metric_to_use = val_corr
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
            self._save_checkpint(update_best=True)
        else:
            if improved:
                self.best_val_metric = metric_to_use
                self._save_checkpint(update_best=True)
            else:
                self._save_checkpint(update_best=False)

        wandb.log(
            {
                "train/train_loss": train_loss,
                "train/train_corr": train_corr,
                "val/val_loss": val_loss,
                "val/val_loss_poisson": val_loss_breakdown["poisson"],
                "val/val_loss_multinomial": val_loss_breakdown["multinomial"],
                "val/best_val_corr": self.best_val_metric,
                "val/early_stopping_counter": self.early_stopping_counter,
                "val/val_corr": val_corr,
                "train_example/example_tracks": train_imgs,
                "val_example/example_tracks": val_imgs,
            }
        )

        flag = self.early_stopping_counter >= self.patience
        return flag

    def _get_scheduler(self, optimizer):
        self.config["scheduler_type"] = "borzoi"
        warmup_steps = self.config.get("warmup_steps", 10000)
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

    def _fine_grained_lr_groups(self, small_scale=4):
        """
        Make the cell-type conditional part of the network learn faster.

        Shared part of the network learns at 1/4 of the learning rate.
        """
        standard_lr = self.config["lr"]
        small_lr = self.config["lr"] / small_scale
        wd = self.config["weight_decay"]

        standard_lr_group = []
        small_lr_group = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A_module" in name:
                standard_lr_group.append(param)
            elif "lora_B_module" in name:
                standard_lr_group.append(param)
            elif "kv_bottleneck" in name:
                standard_lr_group.append(param)
            elif "recycle_conv" in name:
                standard_lr_group.append(param)
            else:
                small_lr_group.append(param)

        parameters = [
            {"params": standard_lr_group, "weight_decay": wd, "lr": standard_lr},
            {"params": small_lr_group, "weight_decay": wd, "lr": small_lr},
        ]
        return parameters

    def _get_optimizer(self):
        optimizer_type = self.config["optimizer"]
        parameter_groups = self._fine_grained_lr_groups()

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(parameter_groups)
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(parameter_groups)
        return optimizer

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

        moving_norm = MovingMetric(window_size=100)
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
            dataloader = self.get_train_dataloader(batches=self.train_batches)

            # start train epochs
            moving_ave_loss = CumulativeCounter()
            moving_ave_corr = MeanPearsonCorrCoefPerChannel(
                n_channels=self.model.out_channels
            ).to(self.device)
            nan_loss = False
            print_steps = max(5, self.train_batches // 20)
            example_step = max(
                5, self.train_batches // (self.plot_example_per_epoch + 1)
            )
            train_example_batches = []
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
                    y_true, y_pred, loss, _ = self._model_forward_pass(
                        self.model, batch
                    )
                    if np.isnan(loss.item()):
                        nan_loss = True
                        print("Training loss has NaN, skipping epoch.")
                        self._update_state_dict()
                        break

                # ==========
                # Backward
                # ==========
                scale_loss = loss / self.accumulate_grad
                scaler.scale(
                    scale_loss
                ).backward()  # for backpropagation, we scale the loss with the accumulate_grad

                moving_ave_loss.update(loss.item())
                moving_ave_corr.update(y_true, y_pred)

                # only update optimizer every accumulate_grad steps
                # this is equivalent to updating every step but with larger batch size (batch_size * accumulate_grad)
                # however, with larger batch size, the GPU memory usage will be higher
                if (batch_id + 1) % self.accumulate_grad == 0:
                    if self.config["global_clipnorm"] is not None:
                        scaler.unscale_(optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config["global_clipnorm"],
                        )
                        total_norm = total_norm.item()

                        # check moving norm and skip step if the norm is too large (e.g. > 99% moving quantile)
                        # this is to prevent outlier gradients from messing up the training
                        moving_norm.update(total_norm)
                        threshold = moving_norm.quantile(0.99).item()
                        if (total_norm > threshold) and moving_norm.full:
                            print(
                                f"Gradient norm is too large: {total_norm:.4f}, "
                                f"threshold: {threshold:.4f}, prevent update."
                            )
                            optimizer.zero_grad()

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
                        _corr = moving_ave_corr.compute().mean()
                        _loss = moving_ave_loss.mean()

                        desc_str = (
                            f" - (Training) {self.cur_epoch} [{batch_id}/{self.train_batches}] "
                            f"Ave Loss: {_loss:.3f}; "
                            f"Ave Corr: {_corr:.3f} "
                            f"Last grad norm: {total_norm:.4f} "
                            f"Last batch Loss: {loss.item():.3f} "
                        )
                        print(desc_str)

                    if batch_id % example_step == 0:
                        batch["pred_data"] = y_pred
                        batch["true_data"] = y_true

                        pseudobulker = self.dataset.name_to_pseudobulker[self.prefix]
                        id_array = batch[f"{self.prefix}:pseudobulk_ids"].cpu().numpy()
                        batch["sample_id"] = pseudobulker.pseudobulk_ids[id_array]

                        train_example_batches.append(batch)

            del dataloader
            self._cleanup_env()
            if nan_loss:
                # epoch break due to nan loss, skip validation
                continue

            self.train_loss = moving_ave_loss.mean()
            self.train_corr = moving_ave_corr.compute()
            self.cur_lr = optimizer.param_groups[0]["lr"]
            (
                self.val_loss,
                self.val_loss_breakdown,
                self.val_corr,
                self.val_wandb_images,
            ) = self._validation_step()

            self.train_wandb_images = self._plot_example(train_example_batches)

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
                val_loss, val_loss_breakdown, val_corr, wandb_images = (
                    self._model_validation_step(
                        model=self.ema.ema_model,
                        dataloader=dataloader,
                        val_batches=val_batches,
                    )
                )
            else:
                self.model.eval()
                val_loss, val_loss_breakdown, val_corr, wandb_images = (
                    self._model_validation_step(
                        model=self.model,
                        dataloader=dataloader,
                        val_batches=val_batches,
                    )
                )
                self.model.train()

        self.model.freeze_batchnorms()
        return val_loss, val_loss_breakdown, val_corr, wandb_images

    def _test(self):
        # load final best checkpoint for testing
        self._update_state_dict()

        if self.val_loss is None:
            self.val_loss, self.val_loss_breakdown, self.val_corr, _ = (
                self._validation_step()
            )

        self.test_loss, self.test_loss_breakdown, self.test_corr, wandb_images = (
            self._validation_step(testing=True)
        )

        wandb.summary["final_valid_loss"] = self.val_loss
        wandb.summary["final_valid_loss_poisson"] = self.val_loss_breakdown["poisson"]
        wandb.summary["final_valid_loss_multinomial"] = self.val_loss_breakdown[
            "multinomial"
        ]
        wandb.summary["final_valid_corr"] = self.val_corr.mean()

        wandb.summary["final_test_loss"] = self.test_loss
        wandb.summary["final_test_loss_poisson"] = self.test_loss_breakdown["poisson"]
        wandb.summary["final_test_loss_multinomial"] = self.test_loss_breakdown[
            "multinomial"
        ]
        wandb.summary["final_test_corr"] = self.test_corr.mean()
        wandb.summary["final_image"] = wandb_images

        # final wandb flag to indicate the run is successfully finished
        wandb.summary["success"] = True
        return

    def train(self):
        """Train function should be implemented in the subclass."""
        raise NotImplementedError


class BorzoiLoRATrainer(BorzoiTrainerMixin):
    """Train LoRA model on pseudobulk single-cell ATAC data."""

    trainer_config = BorzoiTrainerMixin.trainer_config.copy()
    trainer_config.update(
        {
            "mode": "lora",
            "lr": 5e-5,
            "warmup_steps": 10000,
            "scheduler": True,
            # pseudobulk related
            "vq_records": "REQUIRED",
            "target_cov": "REQUIRED",
            "use_vq_emb": "REQUIRED",
            "prefix": "pseudobulk",
            "downsample_vq": None,
            "cov_power": None,
            "cov_soft_clamp": None,
        }
    )

    dataset_class = BorzoiDataset
    model_class = BorzoiLoRA

    def _setup_model(self):
        print("Setting up model from config")
        model = self.model_class.create_from_config(self.config)
        model.to(self.device)

        model.freeze_all_parameter_except_output_head()

        self.model = model
        self._set_total_params()
        return

    def _get_dataset(self):
        dataset = super()._get_dataset()

        # setup pseudobulker params for sc dataset
        pseudobulker_params = {
            "vq_records": self.config["vq_records"],
            "target_cov": self.config["target_cov"],
            "use_vq_emb": self.config["use_vq_emb"],
            "prefix_name": self.config["prefix"],
            "downsample_vq": self.config["downsample_vq"],
        }

        use_vq_emb = self.config["use_vq_emb"]
        kv_bottleneck = self.config["kv_bottleneck"]
        if use_vq_emb:
            assert (
                not kv_bottleneck
            ), "Cannot set both kv_bottleneck and use_vq_emb to True."
        elif kv_bottleneck:
            assert (
                not use_vq_emb
            ), "Cannot set both kv_bottleneck and use_vq_emb to True."

        dataset.add_pseudobulker(
            name=self.prefix,
            cls=RNAVQPseudobulker,
            pseudobulker_kwargs=pseudobulker_params,
        )
        return dataset

    def _model_forward_pass(self, model: BorzoiLoRA, batch: dict):
        data_key = f"{self.prefix}:bulk_data"
        dna_key = "dna_one_hot"
        embedding_key = f"{self.prefix}:embedding_data"
        power = self.config["cov_power"]
        soft_clamp = self.config["cov_soft_clamp"]

        # ==========
        # Get batch data
        # ==========
        X = batch.pop(dna_key)
        embedding = batch.get(embedding_key, None)
        y_true = batch.pop(data_key)

        # ==========
        # Forward and Loss
        # ==========
        y_pred = model(X, embedding=embedding)

        loss, loss_breakdown = model.loss(
            y_true=y_true, y_pred=y_pred, power=power, soft_clamp=soft_clamp
        )

        with torch.no_grad():
            y_true_crop = model.crop(y_true).detach()
            y_pred = y_pred.detach()
            if (soft_clamp is not None) and (power is not None):
                # reverse clamp of pred
                y_pred = reverse_clamp_sqrt(y_pred, power=power, threshold=soft_clamp)
        return y_true_crop, y_pred, loss, loss_breakdown

    def _print_banner(self, text):
        print("=" * len(text) + "\n" + text + "\n" + "=" * len(text))
        return

    def train_output_layer(
        self, batches=1000, epochs=3, output_lr=None, skip_output_adjust=False
    ):
        """Train the output layer only."""
        self._print_banner("Training output layer only")

        # setup output layer only training parameters
        if output_lr is None:
            output_lr = 0.01
        lr = self.config["lr"]
        self.config["lr"] = output_lr
        n_pseudobulk = self.dataset.n_pseudobulks
        self.dataset.n_pseudobulks = 1
        mode = self.config["mode"]
        self.config["mode"] = "output_layer"
        self.mode = "output_layer"
        train_batches = self.train_batches
        self.config["train_batches"] = batches
        self.train_batches = batches

        # train output layer only
        self.checkpoint = self._has_last_checkpoint()
        self._setup_model()
        self._setup_fit()
        if (self.cur_epoch <= epochs) and (not skip_output_adjust):
            print(self.model)
            self._fit(max_epochs=epochs)

        # change things back
        self.config["lr"] = lr
        self.dataset.n_pseudobulks = n_pseudobulk
        self.config["mode"] = mode
        self.mode = mode
        self.train_batches = train_batches
        self.config["train_batches"] = train_batches
        return

    def train_lora(self):
        """Train the LoRA model."""
        self._print_banner("Training LoRA model")
        self.model.convert_to_lora()
        self.model.to(self.device)
        print(self.model)

        # save initial kv before training
        if self.model.kv_bottleneck:
            torch.save(
                self.model.kv_bottleneck.values.detach(), f"{self.savename}.kv_init.pt"
            )

        self.checkpoint = self._has_last_checkpoint()
        self._set_total_params()
        self._setup_fit()
        self._fit()
        return

    def train(
        self,
        output_epochs=3,
        output_batches=1000,
        output_lr=None,
        output_only=False,
        skip_output_adjust=True,
    ) -> None:
        """Train the Borzoi LoRA model."""
        flag = pathlib.Path(f"{self.savename}.{self.mode}.success.flag")

        if flag.exists():
            print(f"Training already finished, found flag file: {flag}.")
            return

        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            # train only output layer
            self.train_output_layer(
                epochs=output_epochs,
                batches=output_batches,
                output_lr=output_lr,
                skip_output_adjust=skip_output_adjust,
            )

            if not output_only:
                # train the lora model
                self.train_lora()
                self._test()
            self._cleanup_env()
            wandb.finish()
        flag.touch()
        return


class BorzoiLoRATester(BorzoiLoRATrainer):
    trainer_config = BorzoiLoRATrainer.trainer_config.copy()
    trainer_config["checkpoint_path"] = "REQUIRED"

    def _setup_model(self):
        checkpoint = torch.load(self.config["checkpoint_path"], weights_only=False)
        if isinstance(checkpoint, dict):
            super()._setup_model()
            self.model.convert_to_lora()
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model = checkpoint

        self.model.eval()
        return

    @staticmethod
    def save_batches(data_batches, saveas, num_rows_per_file=100):
        """Save the data batches to parquet."""
        dataset = ray.data.from_items(data_batches)
        dataset.write_parquet(saveas, num_rows_per_file=num_rows_per_file)
        return

    @torch.inference_mode()
    def test(self, saveas=None, device="cuda"):
        """Test the Borzoi LoRA model."""
        self._setup_model()
        model = self.model.to(device)

        dataloader = self.get_test_dataloader(batches=None)
        *_, data_batches = self._model_validation_step(
            model=model,
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
