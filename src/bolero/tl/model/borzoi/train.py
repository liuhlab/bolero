import pathlib
import re
from collections import defaultdict

import numpy as np
import torch
import wandb
from einops import reduce

from bolero.pl.borzoi import BorzoiExamplePlotter
from bolero.tl.generic.train import GenericTrainer
from bolero.tl.generic.train_helper import CumulativeCounter
from bolero.tl.model.borzoi.dataset import BorzoiDataset
from bolero.tl.model.borzoi.dataset_multi import BorzoiMultiDataset
from bolero.tl.model.borzoi.metrics import (
    MeanPearsonCorrCoefPerChannel,
)
from bolero.tl.model.borzoi.model_lora import (
    BorzoiLoRA,
    BorzoiLoRAMulti,
    BorzoiLoRAwithArches,
)
from bolero.tl.model.borzoi.module_output import DualOutputHead

from .utils import MovingMetric, gene_mask_coords_to_mask, mutation_info_to_mask


def _to_1d(x):
    """
    Collapse any number of extra singleton dims to shape (bs,).
    Expects x to have shape (bs, 1, 1, ..., 1) or already (bs,).
    """
    # fast path: already 1D
    if x.ndim == 1:
        return x
    # sanity check (optional but helpful)
    if any(d != 1 for d in x.shape[1:]):
        raise ValueError(f"extra dims must be 1, got {tuple(x.shape)}")
    # reduce over the (singleton) ellipsis — max/sum/mean are all equivalent here
    return reduce(x, "b ... -> b", "max")


def _validate_gene_count_model_config(config):
    if config["use_regions"] != "borzoi_gene":
        print(
            "Warning: use_regions must be 'borzoi_gene' when gene count model is used, modifying config..."
        )
        config["use_regions"] = "borzoi_gene"

    if "dataset_records" not in config:
        assert (
            config["deg_list"] is not None
        ), "deg_list is required for gene count model"
        assert (
            config["gene_data_path"] is not None
        ), "gene_data_path is required for gene count model"

    if config["output_head_type"] != "gene_count":
        print(
            "Warning: output_head_type must be 'gene_count' when gene count model is used, modifying config..."
        )
        config["output_head_type"] = "gene_count"

    assert (
        config["lora_checkpoint_path"] is not None
    ), "lora_checkpoint_path from pretrained ATAC model is required for gene count model"
    return


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
            if getattr(self.dataset, "_multihead", False):
                precs = self.dataset.name_to_pseudobulker[
                    "pseudobulk"
                ].pseudobulk_records
                self.channel_order = list(range(len(precs)))
            else:
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
        "max_epochs": "REQUIRED",
        "use_amp": True,
        "scheduler": True,
        "lr": 5e-5,
        "lr_total_steps": 250000,
        "optimizer": "adamw",
        "weight_decay": 1e-8,
        "global_clipnorm": 0.5,
        "train_batches": 5000,
        "val_batches": 300,
        "plot_example_per_epoch": 9,
        "accumulate_grad": 2,
        "shuffle_rows": 300,
        "dataloader_concurrency": 8,
        "downsample_train_region": None,
        "downsample_valid_region": None,
        "downsample_test_region": None,
        "grad_norm_collector": False,
        "save_state_every_n_epoch": None,
        "validation_batch_fold": 3,
    }

    def __init__(self, config):
        if config["output_head_type"] == "gene_count":
            _validate_gene_count_model_config(config)

        super().__init__(config)

        # the prefix of pseudobulk data in the batch dict
        # this is the pseudobulker name passed to dataset
        self.prefix = config.get("prefix", "pseudobulk")

        self.model: torch.nn.Module = None
        self._setup_env()
        self._setup_dataset()
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

    def _add_sample_and_region_info_to_batch(self, batch):
        if self.prefix in self.dataset.name_to_pseudobulker:
            # this part is for mouse pseudobulk dataset
            pseudobulker = self.dataset.name_to_pseudobulker[self.prefix]
        else:
            pseudobulker = None

        if pseudobulker is not None:
            try:
                id_array = batch[f"{self.prefix}:pseudobulk_ids"].cpu().numpy()
                batch["sample_id"] = pseudobulker.pseudobulk_ids[id_array]
            except KeyError:
                batch["sample_id"] = None
        else:
            batch["sample_id"] = batch.get("cell_type_id", None)
        # region to region name
        idmap = self.dataset.borzoi_regions.cur_idmap
        region_name = np.array([idmap[i] for i in batch["Original_Name"].cpu().numpy()])
        batch["region_name"] = region_name
        return batch

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
        collect_data=False,
        _add_sample_and_region_info=True,
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

            if _add_sample_and_region_info:
                batch = self._add_sample_and_region_info_to_batch(batch)

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
        self,
        example_batches,
        true_key="true_data",
        pred_key="pred_data",
        y_sync=False,
        _no_genome=False,
    ):
        epoch = self.cur_epoch + 1
        wandb_images = defaultdict(list)

        for idx, batch in enumerate(example_batches):
            if idx >= self.plot_example_per_epoch:
                break

            try:
                plotter = BorzoiExamplePlotter(
                    genome=None if _no_genome else self.dataset.genome,
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

                for channel in range(min(batch[true_key].shape[1], 3)):
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

    def _log_save(self):
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

        self._save_checkpoint(update_best=True)

        # save epoch model state for comparing model over epochs
        save_every_n_epoch = self.config.get("save_state_every_n_epoch", None)
        if save_every_n_epoch is not None and epoch % save_every_n_epoch == 0:
            self._save_epoch_model_state()

        # construct wandb log dict
        log_dict = {
            "val/val_loss": val_loss,
        }
        for k, v in val_loss_breakdown.items():
            log_dict[f"val/val_loss_{k}"] = v
        channel_order = self.channel_order
        for channel, corr in enumerate(val_corr.compute_tensor()):
            if getattr(self.model, "_multihead", False):
                if channel > 5:
                    break
            name = channel_order[channel]
            log_dict[f"val/val_corr_{name}"] = corr
        for channel_name, channel_imgs in val_imgs.items():
            log_dict[f"val_example/example_tracks_{channel_name}"] = channel_imgs
        wandb.log(log_dict)

        return

    def _get_scheduler(self, optimizer):
        self.config["scheduler_type"] = "borzoi"
        warmup_steps = self.config.get("warmup_steps", 5000)
        accumulate_grad = self.config.get("accumulate_grad", 1)
        warmup_steps = (
            warmup_steps // accumulate_grad + 1
        )  # because we update every accumulate_grad steps

        # was self.max_epochs * self.train_batches, fixed here so changing max_epochs and train_batches don't impact LR
        total_steps = self.config.get("lr_total_steps", 250000)  # was 50 * 5000
        total_steps = total_steps // accumulate_grad + 1

        scheduler = GenericTrainer._get_scheduler(
            self, optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
        return scheduler

    def _fine_grained_lr_groups(self):
        """
        Set up fine-grained learning rate groups for the model.

        We can potentially use different lr for different part of network
        But right now all parameters uses the same lr
        """
        standard_lr = self.config["lr"]
        wd = self.config["weight_decay"]
        standard_lr_group = []
        for _, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            standard_lr_group.append(param)

        parameters = [
            {"params": standard_lr_group, "weight_decay": wd, "lr": standard_lr},
        ]
        return parameters

    def _get_optimizer(self):
        optimizer_type = self.config["optimizer"]
        parameter_groups = self._fine_grained_lr_groups()

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(parameter_groups)
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(parameter_groups)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        return optimizer

    def _example_step(self, batch):
        if self.prefix in self.dataset.name_to_pseudobulker:
            # this part is for mouse pseudobulk dataset
            try:
                id_array = batch[f"{self.prefix}:pseudobulk_ids"].cpu().numpy()
                pseudobulker = self.dataset.name_to_pseudobulker[self.prefix]
                batch["sample_id"] = pseudobulker.pseudobulk_ids[id_array]
            except KeyError:
                batch["sample_id"] = None
        else:
            batch["sample_id"] = batch.get("cell_type_id", None)
        log_dict = {}
        train_wandb_images = self._plot_example([batch])
        for channel, channel_imgs in train_wandb_images.items():
            name = self.channel_order[channel]
            log_dict[f"train_example/example_tracks_{name}"] = channel_imgs
        wandb.log(log_dict)
        return

    def _fit(self, max_epochs=None):
        if max_epochs is None:
            max_epochs = self.max_epochs

        # dataset related
        scaler = self.scaler
        optimizer = self.optimizer
        scheduler = self.scheduler
        self.val_loss = None

        if self.cur_epoch > 0:
            print(
                f"Resuming training from epoch {self.cur_epoch+1}, with {max_epochs} epochs in total."
            )

        window_size = 6400 // self.accumulate_grad
        moving_norm = MovingMetric(window_size=window_size)
        total_norm = 999
        while self.cur_epoch < max_epochs:
            print(f"Current epoch: {self.cur_epoch + 1}, max epochs: {max_epochs}.")

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
            print_steps = max(5, self.train_batches // 50)
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
                        for k, v in loss_breakdown.items():
                            print(f"{k}: {v}")
                        for k, v in batch.items():
                            if hasattr(v, "shape"):
                                print(f"{k}: {v.shape}")
                            else:
                                print(f"{k}: {v}")
                        raise ValueError(
                            f"Training loss has NaN. batch_id: {batch_id}, Loss {loss.item()}"
                        )

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
                            if getattr(self.model, "_multihead", False):
                                if channel > 5:
                                    break
                            name = self.channel_order[channel]
                            log_dict[f"train/train_corr_{name}"] = corr
                        wandb.log(log_dict)

                    if batch_id % example_step == 0:
                        batch["pred_data"] = y_pred
                        batch["true_data"] = y_true
                        batch.update(additional_results)
                        self._example_step(batch)

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
            self._log_save()

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
            "warmup_steps": 500,
            # pseudobulk related
            "pseudobulk_records": "REQUIRED",
            "prefix": "pseudobulk",
            "downsample_pseudobulks": None,
            "emb_key": "embedding",
            # Fine-tuning related
            "lora_checkpoint_path": None,
            "_expm1_gene_count_y_true": False,
        }
    )

    dataset_class = BorzoiDataset
    model_class = BorzoiLoRA

    def _setup_model(self, print_model=True):
        print("Setting up model from config")
        model = self.model_class.create_from_config(self.config)
        model.freeze_all_parameter_except_output_head()

        self.model = model
        self.model.to(self.device)
        self.model.convert_to_lora()

        if print_model:
            print(self.model)
        self._set_total_params()

        if self.config.get("lora_checkpoint_path", None) is not None:
            # load the pre-trained LoRA model
            print(
                "Config contains lora_checkpoint_path, will load weights and perform fine-tuning."
            )
            ckpt_path = self.config["lora_checkpoint_path"]
            if isinstance(ckpt_path, dict):
                ckpt_path = ckpt_path["ckpt_path"]
            self.model.load_checkpoint_from_path(ckpt_path, strict=False)

        # wrap with DataParallel after checkpoint loading to avoid module prefix issues
        self._wrap_model_with_dataparallel()

        return

    def _get_dataset(self):
        dataset = super()._get_dataset()

        # setup pseudobulker params for sc dataset
        if dataset.paired_data:
            pseudobulker_params = {
                "pseudobulk_and_ot_info": self.config["pseudobulk_records"],
                "emb_key": self.config["emb_key"],
                "downsample_pseudobulk": self.config["downsample_pseudobulks"],
            }
        else:
            pseudobulker_params = {
                "pseudobulk_records": self.config["pseudobulk_records"],
                "prefix_name": self.config["prefix"],
                "downsample_pseudobulk": self.config["downsample_pseudobulks"],
                "emb_key": self.config["emb_key"],
            }

        dataset.add_pseudobulker(
            name=self.prefix,
            pseudobulker_kwargs=pseudobulker_params,
        )
        return dataset

    def _add_qtl_info_to_batch(self, batch):
        borzoi_regions = self.dataset.borzoi_regions.borzoi_regions
        batch_region_idx = batch["Original_Name"].cpu().numpy()

        batch_mutation_info = borzoi_regions.loc[
            batch_region_idx, ["Ref", "Alt", "pos2start"]
        ]
        batch["__eqtl__:mutation_info"] = batch_mutation_info

        qtl_sig_pval_and_pip = borzoi_regions.loc[
            batch_region_idx, ["is_sig", "pip"]
        ].values.astype(np.float32)
        _dna = batch["dna_one_hot"]
        batch["__eqtl__:sig_pval_and_pip"] = torch.from_numpy(qtl_sig_pval_and_pip).to(
            _dna.device
        )
        qtl_slope = borzoi_regions.loc[batch_region_idx, ["slope"]].values.astype(
            np.float32
        )
        batch["__eqtl__:slope"] = torch.from_numpy(qtl_slope).to(_dna.device)
        return batch

    def _maybe_generate_gene_mask(self, batch) -> None | torch.Tensor:
        """
        If "MaskCoords" is in the batch, generate a gene mask tensor.

        For gene count model, the gene mask tensor has shape (bs, 1, seq_len) where the gene region is 1, else 0.
        For QTL model, the gene mask tensor has shape (bs, 5, seq_len) where
        the first channel is gene mask, and the rest are mutation mask.

        If "MaskCoords" is not in the batch, return None.
        """
        if self.dataset.qtl_data_path is not None:
            batch = self._add_qtl_info_to_batch(batch)

        gene_mask = batch.get("MaskCoords", None)
        if gene_mask is not None:
            dna: torch.Tensor = batch["dna_one_hot"]
            gene_mask_tensor = gene_mask_coords_to_mask(gene_mask, dna)

            if self.dataset.qtl_data_path is not None:
                mutation_info = batch["__eqtl__:mutation_info"]
                mutation_mask = mutation_info_to_mask(mutation_info, dna)
                gene_mask_tensor = torch.cat([gene_mask_tensor, mutation_mask], dim=1)
            batch["__gene_mask__"] = gene_mask_tensor
            return gene_mask_tensor
        else:
            batch["__gene_mask__"] = None
            return None

    def _model_forward_pass_gene_count(
        self,
        model: BorzoiLoRA,
        batch: dict,
        dna_embedding: torch.Tensor,
        embedding: torch.Tensor,
    ):
        y_true = batch["__gene_value__"]

        if self.config["_expm1_gene_count_y_true"]:
            # y_true is log1p transformed, so we need to expm1 to get the original value
            y_true = torch.expm1(y_true)
        y_true = _to_1d(y_true)
        y_pred = model.gene_count_output_head(dna_embedding, embedding=embedding)
        y_pred = _to_1d(y_pred)

        # select only non-nan values
        # because some dataset may have some pseudobulks with no gene count data
        is_valid = ~torch.isnan(y_true)
        valid_ratio = is_valid.sum() / len(y_true)
        y_true = y_true[is_valid]
        y_pred = y_pred[is_valid]
        if len(y_true) == 0:
            return torch.tensor(0.0).to(y_true.device)

        gene_count_loss = model.gene_count_loss(
            y_pred=y_pred, y_true=y_true, reduce=True
        )
        gene_count_loss = gene_count_loss * valid_ratio
        return gene_count_loss

    def _model_forward_pass_eqtl(
        self,
        model: BorzoiLoRA,
        batch: dict,
        dna_embedding: torch.Tensor,
    ):
        y_true_pval_and_pip = batch["__eqtl__:sig_pval_and_pip"]
        y_true_slope = batch["__eqtl__:slope"]
        gene_mask = batch["__gene_mask__"]

        y_pred_pval_and_pip, y_pred_slope = model.forward_qtl_with_dna_emb_and_mask(
            dna_embedding, gene_mask
        )

        eqtl_loss = model.qtl_loss(
            y_pred_pval_and_pip=y_pred_pval_and_pip,
            y_pred_slope=y_pred_slope,
            y_true_pval_and_pip=y_true_pval_and_pip,
            y_true_slope=y_true_slope,
        )
        return eqtl_loss

    def _maybe_add_gene_or_qtl_related_loss(
        self,
        model: BorzoiLoRA,
        batch: dict,
        dna_embedding: torch.Tensor,
        embedding: torch.Tensor,
        loss: torch.Tensor,
        loss_breakdown: dict,
    ):
        if hasattr(model, "gene_count_output_head"):
            # additional gene count loss
            gene_count_loss = self._model_forward_pass_gene_count(
                model, batch, dna_embedding=dna_embedding, embedding=embedding
            )
            loss = loss + gene_count_loss
            loss_breakdown["gene_count_loss"] = gene_count_loss.clone().detach()
        if hasattr(model, "qtl_slope_output_head"):
            # additional eqtl loss
            eqtl_loss = self._model_forward_pass_eqtl(
                model, batch, dna_embedding=dna_embedding
            )
            loss = loss + eqtl_loss
            loss_breakdown["eqtl_loss"] = eqtl_loss.clone().detach()
        return loss, loss_breakdown

    def _model_forward_pass_single(self, model: BorzoiLoRA, batch: dict):
        data_key = f"{self.prefix}:bulk_data"
        dna_key = "dna_one_hot"
        embedding_key = f"{self.prefix}:embedding_data"
        position_weights = batch.get("position_weights", None)

        # ==========
        # Get batch data
        # ==========
        X = batch[dna_key]
        embedding = batch.get(embedding_key, None)
        y_true = batch[data_key]

        # ==========
        # Forward and Loss
        # ==========
        gene_mask = self._maybe_generate_gene_mask(batch)
        y_pred, dna_embedding = model(
            X, embedding=embedding, gene_mask=gene_mask, return_dna_embedding=True
        )

        loss, loss_breakdown, y_true = model.loss(
            y_true=y_true, y_pred=y_pred, position_weights=position_weights
        )

        y_pred = y_pred.detach()

        # add on gene count or eqtl related loss, if any
        loss, loss_breakdown = self._maybe_add_gene_or_qtl_related_loss(
            model=model,
            batch=batch,
            dna_embedding=dna_embedding,
            embedding=embedding,
            loss=loss,
            loss_breakdown=loss_breakdown,
        )
        return y_true, y_pred, loss, loss_breakdown

    def _split_cond_emb_to_terms(self, batch):
        # split the cond_emb into dict of terms using cond_encoder in pseudobulker
        pseduobulker = self.dataset.name_to_pseudobulker[self.prefix]
        condition_encoder = getattr(pseduobulker, "condition_encoder", None)
        if condition_encoder is not None:
            cond_emb = batch[f"{self.prefix}:condition_emb_1"]
            cond_emb_terms = condition_encoder.split_cond_emb(cond_emb)
            batch[f"{self.prefix}:condition_emb_1"] = cond_emb_terms
        return batch

    def _cond_emb_module_forward_pass(self, module, batch: dict):
        """
        Forward pass for the conditional embedding module.
        """
        cell_emb_0 = batch[f"{self.prefix}:embedding_data_0"]
        cond_emb = batch.get(f"{self.prefix}:condition_emb_1", None)
        if module is None:
            return cell_emb_0
        # 3. aggregate all conditional input
        cond_ensemble = module(cell_emb=cell_emb_0, cond_emb=cond_emb)
        return cond_ensemble

    def _model_forward_pass_paired(self, model: BorzoiLoRA, batch: dict):
        batch = self._split_cond_emb_to_terms(batch)

        # 1. sequence input
        dna_one_hot = batch["dna_one_hot"]  # (bs, 4, seq_len)

        signal = batch[f"{self.prefix}:bulk_data_0"]
        signal = torch.log1p(signal)

        y_true_count = batch[f"{self.prefix}:bulk_data_1"]
        y_true_delta = batch[f"{self.prefix}:bulk_data_delta"]

        # 2. aggregate all conditional input
        cond_ensemble = self._cond_emb_module_forward_pass(model.cond_emb_module, batch)

        # 3. predict count and loss
        gene_mask = self._maybe_generate_gene_mask(batch)
        y_pred_count, dna_emb = model(
            dna_one_hot,
            embedding=cond_ensemble,
            signal=signal,
            return_dna_embedding=True,
            gene_mask=gene_mask,
        )
        batch["__ypred__:count"] = y_pred_count
        loss, loss_breakdown, y_true_count = model.loss(
            y_pred=y_pred_count, y_true=y_true_count, reduce=True, position_weights=None
        )

        # 4. predict delta and loss
        if model.delta_output_head is not None:
            y_pred_delta = model.delta_output_head(dna_emb)
            y_true_delta = y_true_delta
            delta_loss, y_true_delta = model.delta_mse_loss(
                y_pred=y_pred_delta, y_true=y_true_delta, reduce=True
            )
            loss = loss + delta_loss
            loss_breakdown["delta_loss"] = delta_loss

        # add on gene count or eqtl related loss, if any
        loss, loss_breakdown = self._maybe_add_gene_or_qtl_related_loss(
            model=model,
            batch=batch,
            dna_embedding=dna_emb,
            embedding=cond_ensemble,
            loss=loss,
            loss_breakdown=loss_breakdown,
        )
        return y_true_count, y_pred_count, loss, loss_breakdown

    def _model_forward_pass(self, model: BorzoiLoRA, batch: dict):
        if self.dataset.paired_data:
            return self._model_forward_pass_paired(model, batch)
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


class MultiBorzoiLoRATrainer(BorzoiLoRATrainer):
    """Borzoi trainer for training on multiple datasets"""

    dataset_class = BorzoiMultiDataset
    model_class = BorzoiLoRAMulti

    trainer_config = BorzoiTrainerMixin.trainer_config.copy()
    trainer_config.update(
        {
            "mode": "lora",
            "lr_total_steps": 1000000,
            "warmup_steps": 500,
            # after first round training shared parameter,
            # further fine tune dataset specific parameters
            "train_dataset_specific_only": False,
            # when fine tuning, use this to provide trained lora weights
            "lora_checkpoint_path": None,
        }
    )

    def _get_dataset(self):
        dataset = self.dataset_class.create_from_config(self.config)

        # update cond_module_kwargs with dataset specific keys
        cond_module_kwargs = dataset.dm.make_cond_module_kwargs()
        cur_kwargs = self.config["cond_module_kwargs"] or {}
        cur_kwargs.update(cond_module_kwargs)
        self.config["cond_module_kwargs"] = cur_kwargs
        print("Updated cond_module_kwargs in config with dataset specific information.")
        return dataset

    def _train_dataset_specific_only(self):
        """
        Freeze everything except the dataset specific part of cond_emb_module.
        """
        name_pattern = re.compile("cond_emb_module.(cell|cond)_encoder_dict")
        for name, params in self.model.named_parameters():
            if not name_pattern.match(name):
                params.requires_grad = False
        print(
            "Training dataset specific part of cond_emb_module only. "
            "Remaining model parameters are frozen."
        )
        return

    def _setup_model(self, print_model=True):
        super()._setup_model(print_model=False)

        if self.config["train_dataset_specific_only"]:
            self._train_dataset_specific_only()

        if print_model:
            print(self.model)
        return

    def _cond_emb_module_forward_pass(self, module, batch: dict):
        """
        Forward pass for the conditional embedding module.
        """
        cell_emb_0 = batch[f"{self.prefix}:embedding_data_0"]
        cond_emb = batch[f"{self.prefix}:condition_emb_1"]

        shared_emb = {"__genome__": batch["__genome__"].float()}
        if "__shared_data__" in batch:
            shared_emb["__shared_data__"] = batch["__shared_data__"]
        dataset_keys = batch["__dataset_keys__"]

        # 3. aggregate all conditional input
        cond_ensemble = module(
            cell_emb=cell_emb_0,
            cond_emb=cond_emb,
            shared_emb=shared_emb,
            dataset_keys=dataset_keys,
        )
        return cond_ensemble

    def _split_cond_emb_to_terms(self, batch):
        # split the cond_emb into dict of terms using cond_encoder in pseudobulker
        multi_pm = self.dataset.dm.pseudobulker
        dataset_keys = batch["__dataset_keys__"]
        condition_emb_data = batch[f"{self.prefix}:condition_emb_1"]
        split_condition_emb_data: list[dict[str, torch.Tensor]] = []

        for dataset_idx, cond_emb in zip(dataset_keys, condition_emb_data):
            dataset_pm = multi_pm.pseudobulker_dict[multi_pm.keys[dataset_idx]]
            condition_encoder = dataset_pm.condition_encoder
            if condition_encoder is not None:
                cond_emb_terms = condition_encoder.split_cond_emb(cond_emb.unsqueeze(0))
                split_condition_emb_data.append(cond_emb_terms)
            else:
                split_condition_emb_data.append(None)
        batch[f"{self.prefix}:condition_emb_1"] = split_condition_emb_data
        return batch

    def _plot_example(
        self, example_batches, true_key="true_data", pred_key="pred_data", y_sync=False
    ):
        wandb_images = super()._plot_example(
            example_batches,
            true_key=true_key,
            pred_key=pred_key,
            y_sync=y_sync,
            # due to potentially multi genome training,
            # here we do not display genome info in example plot
            _no_genome=True,
        )
        return wandb_images

    def _example_step(self, batch):
        batch["sample_id"] = None
        log_dict = {}
        train_wandb_images = self._plot_example([batch])
        for channel, channel_imgs in train_wandb_images.items():
            log_dict[f"train_example/example_tracks_{channel}"] = channel_imgs
        wandb.log(log_dict)
        return

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
        collect_data=False,
    ):
        results = super()._model_validation_step(
            model=model,
            dataloader=dataloader,
            val_batches=val_batches,
            collect_data=collect_data,
            # main difference is we do not add sample and region info when plotting
            # because its complecated to gather correct info in multiple datasets
            _add_sample_and_region_info=False,
        )
        return results


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
