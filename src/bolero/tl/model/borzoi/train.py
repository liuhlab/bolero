import pathlib
from collections import defaultdict

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
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA, BorzoiLoRAwithArches
from bolero.tl.model.borzoi.module_output import DualOutputHead

from .utils import MovingMetric


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

        # get the channel order
        try:
            # this part is for mouse pseudobulk dataset
            self.channel_order = self.dataset.name_to_pseudobulker[
                self.prefix
            ].prefix_order
        except AttributeError:
            self.channel_order = ["data"]
        return

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

    @staticmethod
    def _step_sample_regions(bed, offset=None):
        sample_fold = 7
        if offset is None:
            offset = np.random.randint(sample_fold)
        bed = (
            bed.sort_values(["Chromosome", "Start"])
            .iloc[offset::sample_fold]
            .sample(frac=1, replace=False)
        )
        return bed

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
        if getattr(self, "train_region_step_sample", False):
            # downsample the training regions
            _bed = self._step_sample_regions(self.train_regions)
        else:
            _bed = self.train_regions
        dataloader = self.dataset.get_dataloader(
            folds=self.train_folds,
            region_bed=_bed,
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
        batch_size_fold = self.config.get("validation_batch_fold", 3)
        batch_size = int(self.dataset.batch_size * batch_size_fold)
        self.dataset.eval()

        dataloader = self.dataset.get_dataloader(
            folds=self.valid_folds,
            region_bed=self.valid_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config.get("shuffle_rows", None),
            batch_size=batch_size,  # this will overwrite the batch size in training
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
        batch_size_fold = self.config.get("validation_batch_fold", 3)
        batch_size = int(self.dataset.batch_size * batch_size_fold)
        self.dataset.eval()

        dataloader = self.dataset.get_dataloader(
            folds=self.test_folds,
            region_bed=self.test_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            shuffle_rows=self.config.get("shuffle_rows", None),
            batch_size=batch_size,  # this will overwrite the batch size in training
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
        "patience": 5,
        "start_early_stop_after_epoch": 20,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": "REQUIRED",
        "large_lr_scale": 1,
        "optimizer": "adamw",
        "weight_decay": 1e-7,
        "global_clipnorm": 0.1,
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
        "validation_batch_fold": 3,
    }

    def __init__(self, config):
        super().__init__(config)
        self.start_early_stop_after_epoch: bool = config["start_early_stop_after_epoch"]

        # the prefix of pseudobulk data in the batch dict
        # this is the pseudobulker name passed to dataset
        self.prefix = config.get("prefix", "pseudobulk")

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

    def _autocast_context(self):
        try:
            auto_cast_context = torch.autocast(
                device_type=str(self.device).split(":")[0],
                dtype=torch.bfloat16,
                enabled=self.config.get("use_amp", True),
            )
        except RuntimeError:
            # some GPU, such as T4 does not support bfloat16
            auto_cast_context = torch.autocast(
                device_type=str(self.device).split(":")[0],
                dtype=torch.float16,
                enabled=self.config.get("use_amp", True),
            )
        return auto_cast_context

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

        mean_loss_breakdown = {}

        if isinstance(model.final_output_head, DualOutputHead):
            num_channels = len(self.config["data_key"])
            mean_val_corr = MeanPearsonCorrCoefPerChannel(n_channels=num_channels).to(
                self.device
            )
        else:
            mean_val_corr = MeanPearsonCorrCoefPerChannel(
                n_channels=model.out_channels
            ).to(self.device)

        if self.prefix in self.dataset.name_to_pseudobulker:
            # this part is for mouse pseudobulk dataset
            pseudobulker = self.dataset.name_to_pseudobulker[self.prefix]
        else:
            pseudobulker = None

        example_batches = []  # collect example batches for making images
        data_collector = []  # collect data for further analysis

        for batch_id, batch in enumerate(dataloader):
            with self._autocast_context():
                y_true, y_pred, loss, loss_breakdown, *additional_results = (
                    self._model_forward_pass(model, batch)
                )

            if len(additional_results) > 0:
                additional_results = additional_results[0]
            else:
                additional_results = {}

            mean_val_corr.update(target=y_true, preds=y_pred)
            for k, v in loss_breakdown.items():
                try:
                    mean_loss_breakdown[k].update(v)
                except KeyError:
                    mean_loss_breakdown[k] = CumulativeCounter()
                    mean_loss_breakdown[k].update(v)
            mean_val_loss.update(loss)

            # add additional data into batch dict
            batch["true_data"] = y_true
            batch["pred_data"] = y_pred
            if pseudobulker is not None:
                id_array = batch[f"{self.prefix}:pseudobulk_ids"].cpu().numpy()
                batch["sample_id"] = pseudobulker.pseudobulk_ids[id_array]
            else:
                batch["sample_id"] = batch.get("cell_type_id", None)
            # region to region name
            idmap = self.dataset.borzoi_regions.cur_idmap
            region_name = np.array(
                [idmap[i] for i in batch["Original_Name"].cpu().numpy()]
            )
            batch["region_name"] = region_name

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
                    f"Mean Track Corr: {mean_val_corr.get_corr_str()}; "
                )
                print(desc_str)

        del dataloader
        self._cleanup_env()

        wandb_images = self._plot_example(example_batches, y_sync=False)
        # channel to name
        wandb_images = {self.channel_order[k]: v for k, v in wandb_images.items()}

        # conditional plotting for upper bound and p
        for plot_name in ("upper_bound", "p"):
            if f"true_{plot_name}" in example_batches[0]:
                this_imgs = self._plot_example(
                    example_batches,
                    true_key=f"true_{plot_name}",
                    pred_key=f"pred_{plot_name}",
                    y_sync=False,
                )
                for channel, channel_imgs in this_imgs.items():
                    name = self.channel_order[channel]
                    wandb_images[f"{name}_{plot_name}"] = channel_imgs

        # ==========
        # Final metrics
        # ==========
        val_loss = mean_val_loss.mean()
        val_loss_breakdown = {k: v.mean() for k, v in mean_loss_breakdown.items()}
        val_corr = mean_val_corr

        if collect_data:
            return val_loss, val_loss_breakdown, val_corr, wandb_images, data_collector
        else:
            return val_loss, val_loss_breakdown, val_corr, wandb_images

    def _model_forward_pass(self, model, batch):
        raise NotImplementedError

    def _plot_example(
        self, example_batches, true_key="true_data", pred_key="pred_data", y_sync=False
    ):
        epoch = self.cur_epoch + 1
        wandb_images = defaultdict(list)

        for idx, batch in enumerate(example_batches):
            if idx >= self.plot_example_per_epoch:
                break

            try:
                plotter = BorzoiExamplePlotter(
                    genome=self.dataset.genome,
                    true_key=true_key,
                    pred_key=pred_key,
                    id_key="sample_id",
                    plot_mode=(
                        "atac"
                        if self.model.loss_type == "poisson_multinomial"
                        else "mc"
                    ),
                )
                fig = plotter.plot(batch, channel=0, nrows=2, return_array=True)

                if self.dataset.paired_data:
                    nrows = [0, 2]
                else:
                    nrows = 2

                for channel in range(self.model.out_channels):
                    fig = plotter.plot(
                        batch,
                        channel=channel,
                        nrows=nrows,
                        return_array=True,
                        y_sync=y_sync,
                    )

                    wandb_images[channel].append(
                        wandb.Image(
                            fig,
                            mode="RGB",
                            caption=f"Epoch {epoch} Example {idx}",
                            grouping=epoch,
                            file_type="jpg",  # reduce file size
                        )
                    )
            except ValueError as e:
                print(f"Error in plotting example: {e}")
                print(batch)
                for k, v in batch.items():
                    print(k, v.shape)
                continue
        return wandb_images

    def _log_save_and_check_stop(self):
        val_imgs = self.val_wandb_images
        epoch = self.cur_epoch
        train_loss = self.train_loss
        train_corr = self.train_corr
        learning_rate = self.cur_lr
        val_loss = self.val_loss
        val_loss_breakdown = self.val_loss_breakdown
        val_corr = self.val_corr

        print(
            f" - (Training)   {epoch} Loss: {train_loss:.3f}; Mean Corr. : {train_corr.get_corr_str()}; "
            f"Learning rate {learning_rate}."
        )
        print(
            f" - (Validation) {epoch} Loss: {val_loss:.3f}; Mean Corr. : {val_corr.get_corr_str()}."
        )
        self.model.print_loss_weight()

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
        for k, v in val_loss_breakdown.items():
            log_dict[f"val/val_loss_{k}"] = v
        channel_order = self.channel_order
        for channel, corr in enumerate(val_corr.compute_tensor()):
            name = channel_order[channel]
            log_dict[f"val/val_corr_{name}"] = corr
        for channel_name, channel_imgs in val_imgs.items():
            log_dict[f"val_example/example_tracks_{channel_name}"] = channel_imgs
        wandb.log(log_dict)

        flag = self.early_stopping_counter >= self.patience
        return flag

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

    def _fine_grained_lr_groups_original(self, small_scale=4):
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

    def _fine_grained_lr_groups(self):
        """
        Set up fine-grained learning rate groups for the model.
        """
        lora_preset = self.model.lora_preset
        if lora_preset == "original":
            return self._fine_grained_lr_groups_original()

        standard_lr = self.config["lr"]
        large_lr = self.config["lr"] * self.config["large_lr_scale"]
        wd = self.config["weight_decay"]
        loss_weight_lr = 0.001

        self.large_lr_params = []
        self.standard_lr_params = []
        standard_lr_group = []
        large_lr_group = []
        loss_weight_lr_group = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # The last layer of lora_B_module is the critical gate for the cell-type conditional part
            # Mimic the LoRA+ paper, https://arxiv.org/abs/2402.12354
            if "lora_B" in name:
                large_lr_group.append(param)
                self.large_lr_params.append(name)
            elif "log_var_weight" in name:
                loss_weight_lr_group.append(param)
            else:
                standard_lr_group.append(param)
                self.standard_lr_params.append(name)

        parameters = [
            {"params": standard_lr_group, "weight_decay": wd, "lr": standard_lr},
            {"params": large_lr_group, "weight_decay": wd, "lr": large_lr},
        ]
        if len(loss_weight_lr_group) > 0:
            parameters.append(
                {
                    "params": loss_weight_lr_group,
                    "weight_decay": 1e-3,
                    "lr": loss_weight_lr,
                }
            )

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

            if isinstance(self.model.final_output_head, DualOutputHead):
                num_channels = len(self.config["data_key"])
                moving_ave_corr = MeanPearsonCorrCoefPerChannel(
                    n_channels=num_channels
                ).to(self.device)
            else:
                moving_ave_corr = MeanPearsonCorrCoefPerChannel(
                    n_channels=self.model.out_channels
                ).to(self.device)
            nan_loss = False
            print_steps = max(5, self.train_batches // 20)
            example_step = max(
                5, self.train_batches // (self.plot_example_per_epoch + 1)
            )
            for batch_id, batch in enumerate(dataloader):
                with self._autocast_context():
                    y_true, y_pred, loss, loss_breakdown, *additional_results = (
                        self._model_forward_pass(self.model, batch)
                    )

                    if len(additional_results) > 0:
                        additional_results = additional_results[0]
                    else:
                        additional_results = {}

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
                        param_groups = self.model.get_global_clipnorm_params()
                        for i, group in enumerate(param_groups):
                            if len(group) == 0:
                                continue
                            if i == 0:
                                # use the first group (major group) as the total norm
                                total_norm = torch.nn.utils.clip_grad_norm_(
                                    group,
                                    max_norm=self.config["global_clipnorm"],
                                )
                                total_norm = total_norm.item()
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    group,
                                    max_norm=self.config["global_clipnorm"],
                                )

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
                        for loss_name, loss_value in loss_breakdown.items():
                            log_dict[f"train/train_loss_{loss_name}"] = loss_value
                        for channel, corr in enumerate(
                            moving_ave_corr.compute_tensor()
                        ):
                            name = self.channel_order[channel]
                            log_dict[f"train/train_corr_{name}"] = corr
                        wandb.log(log_dict)

                    if batch_id % example_step == 0:
                        batch["pred_data"] = y_pred
                        batch["true_data"] = y_true
                        batch.update(additional_results)

                        if self.prefix in self.dataset.name_to_pseudobulker:
                            # this part is for mouse pseudobulk dataset
                            pseudobulker = self.dataset.name_to_pseudobulker[
                                self.prefix
                            ]
                            id_array = (
                                batch[f"{self.prefix}:pseudobulk_ids"].cpu().numpy()
                            )
                            batch["sample_id"] = pseudobulker.pseudobulk_ids[id_array]
                        else:
                            batch["sample_id"] = batch.get("cell_type_id", None)
                        log_dict = {}
                        train_wandb_images = self._plot_example([batch])
                        for channel, channel_imgs in train_wandb_images.items():
                            name = self.channel_order[channel]
                            log_dict[f"train_example/example_tracks_{name}"] = (
                                channel_imgs
                            )

                        for plot_name in ("upper_bound", "p"):
                            if f"true_{plot_name}" in batch:
                                this_imgs = self._plot_example(
                                    [batch],
                                    true_key=f"true_{plot_name}",
                                    pred_key=f"pred_{plot_name}",
                                )
                                for channel, channel_imgs in this_imgs.items():
                                    name = self.channel_order[channel]
                                    log_dict[
                                        f"train_example/example_tracks_{name}_{plot_name}"
                                    ] = channel_imgs

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
                self.val_wandb_images,
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
        for k, v in self.val_loss_breakdown.items():
            wandb.summary[f"final_valid_loss_{k}"] = v
        wandb.summary["final_valid_corr"] = self.val_corr.compute().mean()
        wandb.summary["final_test_loss"] = self.test_loss
        for k, v in self.test_loss_breakdown.items():
            wandb.summary[f"final_test_loss_{k}"] = v
        wandb.summary["final_test_corr"] = self.test_corr.compute().mean()
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
            "use_vq_emb": "REQUIRED",
            "prefix": "pseudobulk",
            "downsample_vq": None,
            "emb_key": "embedding",
        }
    )

    dataset_class = BorzoiDataset
    model_class = BorzoiLoRA

    def _setup_model(self, print_model=True):
        print("Setting up model from config")
        model = self.model_class.create_from_config(self.config)
        model.freeze_all_parameter_except_output_head()

        self.model = model

        # add the upper bound head and change final output head activation to None
        if self.dataset.use_pseudobulk_profile:
            self.model.setup_profile_head()

        self.model.to(self.device)
        self.model.convert_to_lora()
        if print_model:
            print(self.model)
        self._set_total_params()
        return

    def _get_dataset(self):
        dataset = super()._get_dataset()

        # setup pseudobulker params for sc dataset
        if dataset.paired_data:
            pseudobulker_params = {
                "pseudobulk_and_ot_info": self.config["vq_records"],
                "emb_key": self.config["emb_key"],
                "downsample_pseudobulk": self.config["downsample_vq"],
            }
        else:
            pseudobulker_params = {
                "vq_records": self.config["vq_records"],
                "use_vq_emb": self.config["use_vq_emb"],
                "prefix_name": self.config["prefix"],
                "downsample_vq": self.config["downsample_vq"],
                "emb_key": self.config["emb_key"],
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
            pseudobulker_kwargs=pseudobulker_params,
        )
        return dataset

    def _model_forward_pass_profile(self, model: BorzoiLoRA, batch: dict):
        data_key = f"{self.prefix}:bulk_data"
        dna_key = "dna_one_hot"
        embedding_key = f"{self.prefix}:embedding_data"
        upper_bound_key = f"{self.prefix}:upper_bound"
        position_weights = batch.get("position_weights", None)

        # ==========
        # Get batch data
        # ==========
        X = batch.pop(dna_key)
        embedding = batch.get(embedding_key, None)
        true_p = batch.pop(data_key)
        true_upper_bound = batch.pop(upper_bound_key)

        # ==========
        # Forward and Loss
        # ==========
        pred_logit, dna_embedding = model(
            X, embedding=embedding, return_dna_embedding=True
        )

        # detach the dna_embedding to prevent gradient flow
        pred_upper_bound = model.upper_bound_head(dna_embedding)

        loss, loss_breakdown, _pred, _true = model.loss_profile_and_upper_bound(
            y_pred=(pred_upper_bound, pred_logit),
            y_true=(true_upper_bound, true_p),
            position_weights=position_weights,
        )
        pred_count, pred_p, pred_upper_bound = _pred
        true_count, true_p, true_upper_bound = _true

        # combine upper bound and data_p
        y_pred = pred_count
        y_true = true_count
        additional_results = {
            "true_p": true_p,
            "pred_p": pred_p,
            "pred_upper_bound": pred_upper_bound,
            "true_upper_bound": true_upper_bound,
        }
        return y_true, y_pred, loss, loss_breakdown, additional_results

    def _model_forward_pass_single(self, model: BorzoiLoRA, batch: dict):
        data_key = f"{self.prefix}:bulk_data"
        dna_key = "dna_one_hot"
        embedding_key = f"{self.prefix}:embedding_data"
        position_weights = batch.get("position_weights", None)

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

        loss, loss_breakdown, y_true = model.loss(
            y_true=y_true, y_pred=y_pred, position_weights=position_weights
        )

        y_pred = y_pred.detach()
        return y_true, y_pred, loss, loss_breakdown

    def _model_forward_pass_paired(self, model: BorzoiLoRA, batch: dict):
        suffix_list = ["_A", "_B"]
        y_true_list = []
        y_pred_list = []
        embedding_list = []
        pseudobulk_id_list = []
        for suffix in suffix_list:
            dna_key = "dna_one_hot"
            data_key = f"{self.prefix}:bulk_data{suffix}"
            embedding_key = f"{self.prefix}:embedding_data{suffix}"
            pseudobulk_ids_key = f"{self.prefix}:pseudobulk_ids{suffix}"

            # ==========
            # Get batch data
            # ==========
            X = batch[dna_key]
            embedding = batch.pop(embedding_key)
            y_true = batch.pop(data_key)

            # ==========
            # Forward and Loss
            # ==========
            y_pred = model(X, embedding=embedding)

            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            embedding_list.append(embedding)
            pseudobulk_id_list.append(batch.pop(pseudobulk_ids_key))

        batch.pop(dna_key)
        position_weights = batch.get("position_weights", None)
        loss, loss_breakdown, *y_true_list = model.paired_loss(
            *y_pred_list, *y_true_list, position_weights=position_weights
        )

        with torch.no_grad():
            _y_pred_list = []
            for y_pred in y_pred_list:
                y_pred = y_pred.detach()
                _y_pred_list.append(y_pred)

        # concatenate A and B on the batch dimension
        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(_y_pred_list, dim=0)
        batch[f"{self.prefix}:embedding_data"] = torch.cat(embedding_list, dim=0)
        batch[f"{self.prefix}:pseudobulk_ids"] = torch.cat(pseudobulk_id_list, dim=0)
        batch["region"] = torch.cat(
            [batch["region"], batch["region"]], dim=0
        )  # duplicate region
        return y_true, y_pred, loss, loss_breakdown

    def _model_forward_pass(self, model: BorzoiLoRA, batch: dict):
        batch = self.dataset.maybe_preprocess_batch(batch)

        if self.dataset.paired_data:
            return self._model_forward_pass_paired(model, batch)
        elif self.dataset.use_pseudobulk_profile:
            return self._model_forward_pass_profile(model, batch)
        else:
            return self._model_forward_pass_single(model, batch)

    def _print_banner(self, text):
        print("=" * len(text) + "\n" + text + "\n" + "=" * len(text))
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


class BorzoiArchTrainer(BorzoiLoRATrainer):
    model_class = BorzoiLoRAwithArches

    def _setup_model(self, print_model=True):
        print("Setting up model from config")
        model = self.model_class.create_from_config(self.config)
        self.model = model

        self.model.to(self.device)
        if print_model:
            print(self.model)
        self._set_total_params()
        return


class BorzoiLoRATrainerRNA(BorzoiLoRATrainer):
    trainer_config = BorzoiLoRATrainer.trainer_config.copy()
    trainer_config["lora_checkpoint_path"] = "REQUIRED"

    def _setup_model(self):
        super()._setup_model(print_model=False)

        # load the pre-trained LoRA model
        self.model.load_checkpoint_from_path(self.config["lora_checkpoint_path"])

        print(self.model)
        return

    def _model_forward_pass(self, model: BorzoiLoRA, batch: dict):
        data_key = f"{self.prefix}:bulk_data"
        dna_key = "dna_one_hot"
        embedding_key = f"{self.prefix}:embedding_data"

        # ==========
        # Get batch data
        # ==========
        X = batch.pop(dna_key)
        embedding = batch.get(embedding_key, None)
        y_true = batch.pop(data_key)

        # ==========
        # Forward and Loss
        # ==========
        _, dna_embedding = model(X, embedding=embedding, return_dna_embedding=True)
        y_pred = model.rna_output_head(dna_embedding, embedding=embedding)

        loss, loss_breakdown, y_true = model.loss(y_true=y_true, y_pred=y_pred)

        y_pred = y_pred.detach()
        return y_true, y_pred, loss, loss_breakdown


class BorzoiLoRATrainerWithGeneCount(BorzoiLoRATrainer):
    trainer_config = BorzoiLoRATrainer.trainer_config.copy()
    trainer_config.update(
        {"gene_loss_weight": 1, "freeze_borzoi": False, "lora_checkpoint_path": None}
    )

    def _setup_model(self, print_model=True):
        super()._setup_model(print_model=False)

        if self.config.get("freeze_borzoi", False):
            print("Freeze the Borzoi model except the gene count head.")
            for name, params in self.model.named_parameters():
                # freeze everything except the gene count head
                if not name.startswith("gene_count_head"):
                    params.requires_grad = False
            assert (
                self.config["lora_checkpoint_path"] is not None
            ), "LoRA checkpoint path is required for freezing the model."

        if self.config.get("lora_checkpoint_path", None) is not None:
            # load the pre-trained LoRA model
            self.model.load_checkpoint_from_path(
                self.config["lora_checkpoint_path"], strict=False
            )

        if print_model:
            print(self.model)
        self._set_total_params()
        return

    def _model_forward_pass(self, model: BorzoiLoRA, batch: dict):
        data_key = f"{self.prefix}:bulk_data"
        dna_key = "dna_one_hot"
        embedding_key = f"{self.prefix}:embedding_data"
        position_weights = batch.get("position_weights", None)

        # ==========
        # Get batch data
        # ==========
        X = batch.pop(dna_key)
        embedding = batch.get(embedding_key, None)
        y_true = batch.pop(data_key)

        # ==========
        # Forward and Loss
        # ==========
        y_pred, dna_emb = model(X, embedding=embedding, return_dna_embedding=True)

        loss, loss_breakdown, y_true = model.loss(
            y_true=y_true, y_pred=y_pred, position_weights=position_weights
        )

        y_pred = y_pred.detach()

        # ==========
        # Gene count forward and loss
        # ==========
        gene_true = batch["gene_count"]  # (bs, 1)
        gene_pred = model.gene_count_head(dna_emb)
        gene_loss = model.gene_count_loss(y_pred=gene_pred, y_true=gene_true)
        loss_breakdown["gene_count_mse"] = gene_loss.item()
        loss += gene_loss * self.config["gene_loss_weight"]

        batch["gene_pred"] = gene_pred
        return y_true, y_pred, loss, loss_breakdown


class BorzoiTesterMixin:
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


class BorzoiLoRATester(BorzoiTesterMixin, BorzoiLoRATrainer):
    trainer_config = BorzoiLoRATrainer.trainer_config.copy()
    trainer_config["checkpoint_path"] = "REQUIRED"


class BorzoiArchTester(BorzoiTesterMixin, BorzoiArchTrainer):
    trainer_config = BorzoiArchTrainer.trainer_config.copy()
    trainer_config["checkpoint_path"] = "REQUIRED"


class BorzoiGeneCountTester(BorzoiTesterMixin, BorzoiLoRATrainerWithGeneCount):
    trainer_config = BorzoiLoRATrainerWithGeneCount.trainer_config.copy()
    trainer_config["checkpoint_path"] = "REQUIRED"
