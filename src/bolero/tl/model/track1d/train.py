from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from bolero.pl.track1d import Track1DExamplePlotter
from bolero.pl.utils import figure_to_array
from bolero.tl.model.generic.train import GenericTrainer, TrainerDatasetMixin
from bolero.tl.model.generic.train_helper import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
)
from bolero.tl.model.track1d.dataset import Track1DDataset
from bolero.tl.model.track1d.model import DialatedCNNTrack1DModel

from .unet.model import UNetTrans


class Track1DModelTrainer(GenericTrainer, TrainerDatasetMixin):
    """Train model for predicting 1-D genome tracks."""

    trainer_config = {
        "mode": "init",
        "chrom_split": "REQUIRED",
        "sample": "REQUIRED",
        "region": "REQUIRED",
        "output_dir": "track_1d_model",
        "savename": "model",
        "wandb_project": "track_1d",
        "wandb_job_type": "train",
        "wandb_group": None,
        "max_epochs": 100,
        "patience": 5,
        "use_amp": True,
        "use_ema": True,
        "scheduler": False,
        "lr": 0.003,
        "weight_decay": 0.001,
        "accumulate_grad": 1,
        "train_batches": 5000,
        "val_batches": 500,
        "loss_tolerance": 0.003,
        "plot_example_per_epoch": 3,
    }
    dataset_class = Track1DDataset
    model_class = DialatedCNNTrack1DModel

    def __init__(self, config):
        super().__init__(config)

        self.model: torch.nn.Module = None

        self._setup_env()
        self._setup_model()
        self._setup_dataset()
        self._setup_fit()
        return

    def _setup_model(self):
        mode = self.mode

        if mode == "init":
            self.model = self._setup_model_from_config()
        else:
            raise ValueError(f"Incorrect mode: {mode}.")

        self._set_total_params()
        return

    def _validation_step(self, testing=False, val_batches=None, **kwargs):
        if testing:
            _dataset = self.test_dataset
        else:
            _dataset = self.valid_dataset

        if self.config["use_ema"]:
            self.ema.eval()
            self.ema.ema_model.eval()
            val_results = self.model_validation_step(
                model=self.ema.ema_model,
                dataset=_dataset,
                val_batches=val_batches,
                **kwargs,
            )
            self.ema.train()
            self.ema.ema_model.train()
        else:
            self.model.eval()
            val_results = self.model_validation_step(
                model=self.model,
                dataset=_dataset,
                val_batches=val_batches,
                **kwargs,
            )
            self.model.train()
        return val_results

    @torch.no_grad()
    def model_validation_step(
        self,
        model: torch.nn.Module,
        dataset: Track1DDataset,
        val_batches: Optional[int] = None,
    ) -> Tuple[float, float, float, list[Any]]:
        """
        Perform model validation step.

        Parameters
        ----------
        model : torch.nn.Module
            The model to validate.
        dataset : Track1DDataset
            The dataset to use for validation.
        val_batches : int, optional
            The number of validation batches to use, by default None.

        Returns
        -------
        Tuple[float, float, float, List[Any]]
            A tuple containing the validation loss, single batch Pearson correlation,
            across batch Pearson correlation, and a list of example images.
        """
        if val_batches is None:
            val_batches = self.val_batches

        sample = self.config["sample"]
        region = self.config["region"]
        val_data_loader = dataset.get_dataloader(
            sample=sample,
            region=region,
            local_shuffle_buffer_size=0,
        )
        data_key = f"{region}|{sample}"
        dna_key = f"{region}|{dataset.dna_name}"

        size = 0
        val_loss = 0
        single_batch_pearson_counter = CumulativeCounter()
        across_batch_pearson_counter = CumulativePearson()

        example_batches = []  # collect example batches for making images
        bar = tqdm(
            enumerate(val_data_loader),
            desc=" - (Validation)",
            dynamic_ncols=True,
            total=val_batches,
        )
        for batch_id, batch in bar:
            # ==========
            # X
            # ==========
            X = batch[dna_key]

            # ==========
            # y
            # ==========
            y = batch[data_key]
            y = torch.log1p(y)

            # ==========
            # Forward and Loss
            # ==========
            pred = model(X)
            loss_ = F.mse_loss(pred, y)
            val_loss += loss_.item()

            # ==========
            # Within batch pearson and save for across batch pearson
            # ==========
            # within batch pearson
            corr = batch_pearson_correlation(pred, y).detach().cpu()[:, None]
            single_batch_pearson_counter.update(corr)
            # save for across batch pearson
            across_batch_pearson_counter.update(pred, y)

            size += 1
            if batch_id < self.plot_example_per_epoch:
                batch["pred_"] = pred.detach()
                example_batches.append(batch)

            if size > 5:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} "
                    f"MSE Loss: {val_loss/size:.4f} "
                )
                bar.set_description(desc_str)
            if batch_id >= val_batches:
                break
        bar.close()
        del val_data_loader

        self._cleanup_env()
        wandb_images = self._plot_example_images(example_batches, target_key=data_key)

        # ==========
        # Loss
        # ==========
        val_loss = val_loss / size

        # ==========
        # Within batch pearson
        # ==========
        single_batch_pearson = single_batch_pearson_counter.mean()

        # ==========
        # Across batch pearson
        # ==========
        across_batch_pearson = across_batch_pearson_counter.corr()
        return val_loss, single_batch_pearson, across_batch_pearson, wandb_images

    def _plot_example_images(self, example_batches, target_key, predict_key="pred_"):
        epoch = self.cur_epoch + 1
        wandb_images = []
        for idx, batch in enumerate(example_batches):
            plotter = Track1DExamplePlotter(
                target_key=target_key, predict_key=predict_key
            )
            fig, _ = plotter.plot(
                batch,
                figsize=(6, 6),
                dpi=100,
                top_example=1,
                bottom_example=1,
                plot_channel=0,
            )
            fig_array = figure_to_array(fig)
            fig.savefig(f"{self.savename}.example_{epoch}_{idx}.jpg")
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

    def _log_save_and_check_stop(self):
        epoch = self.cur_epoch
        train_loss = self.train_loss
        learning_rate = self.cur_lr
        val_loss = self.val_loss
        single_batch_pearson = self.val_single_batch_pearson
        across_batch_pearson = self.val_across_batch_pearson
        example_images = self.example_wandb_images

        print(
            f" - (Training) {epoch}; Coverage Loss: {train_loss:.3f}; Learning rate {learning_rate}."
        )
        print(f" - (Validation) {epoch} Loss: {val_loss:.3f}")
        print(f"Single Batch Pearson Corr.: {single_batch_pearson:.3f}")
        print(f"Across Batch Pearson Corr.: {across_batch_pearson:.3f}")

        # only clear the early stopping counter if the loss improvement is better than tolerance
        previous_best = self.best_val_loss
        if val_loss < self.best_val_loss - self.loss_tolerance:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        print(
            f"Previous best loss: {previous_best:.3f}, "
            f"Loss at epoch {epoch}: {val_loss:.3f}; "
            f"Early stopping counter: {self.early_stopping_counter}"
        )
        # save checkpoint if the loss is better
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpint(update_best=True)
        else:
            self._save_checkpint(update_best=False)
        if self.wandb_active:
            wandb.log(
                {
                    "train/train_loss": train_loss,
                    "val/val_loss": val_loss,
                    "val/best_val_loss": self.best_val_loss,
                    "val/early_stopping_counter": self.early_stopping_counter,
                    "val/single_batch_pearson": single_batch_pearson,
                    "val/across_batch_pearson": across_batch_pearson,
                    "val_example/example_predictions": example_images,
                }
            )
        flag = self.early_stopping_counter >= self.patience
        return flag

    def _fit(self, max_epochs=None, valid_first=False):
        if max_epochs is None:
            max_epochs = self.max_epochs

        # dataset related
        training_dataset = self.train_dataset

        sample = self.config["sample"]
        region = self.config["region"]
        data_key = f"{region}|{sample}"
        dna_key = f"{region}|{training_dataset.dna_name}"

        # backpropagation related
        scaler = self.scaler
        optimizer = self.optimizer
        scheduler = self.scheduler
        ema = self.ema
        self.val_loss = None

        if valid_first:
            print("Perform validation before training.")
            (
                self.val_loss,
                self.val_single_batch_pearson,
                self.val_across_batch_pearson,
                wandb_images,
            ) = self._validation_step()
            print(f"Validation loss before training: {self.val_loss:.3f}")
            print(
                f"Validation single-batch pearson: {self.val_single_batch_pearson:.3f}"
            )
            print(
                f"Validation across batch pearson: {self.val_across_batch_pearson:.3f}."
            )
            if self.wandb_active:
                wandb.log(
                    {
                        "val/val_loss": self.val_loss,
                        "val/val_single_batch_pearson": self.val_single_batch_pearson,
                        "val/val_across_batch_pearson": self.val_across_batch_pearson,
                        "val_example/example_predictions": wandb_images,
                    }
                )

        stop_flag = False
        if self.cur_epoch > 0:
            print(
                f"Resuming training from epoch {self.cur_epoch+1}, with {max_epochs+1} epochs in total."
            )
        while self.cur_epoch < max_epochs and not stop_flag:
            # check early stop
            if self.early_stopping_counter >= self.patience:
                # early stopping counter could be loaded from the checkpoint
                # check before starting the for loop
                print(f"Early stopping at epoch {self.cur_epoch}")
                self.early_stoped = True
                break

            # get train data loader
            train_data_loader = training_dataset.get_dataloader(
                sample=sample,
                region=region,
            )

            # start train epochs
            moving_avg_loss = 0
            nan_loss = False

            bar = tqdm(
                enumerate(train_data_loader),
                desc=f" - (Training) {self.cur_epoch}",
                dynamic_ncols=True,
                total=self.train_batches,
            )
            for batch_id, batch in bar:
                try:
                    auto_cast_context = torch.autocast(
                        device_type=str(self.device),
                        dtype=torch.bfloat16,
                        enabled=self.use_amp,
                    )
                except RuntimeError:
                    # some GPU, such as T4 does not support bfloat16
                    print("bfloat16 autocast failed, using float16 instead.")
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

                    # ==========
                    # y
                    # ==========
                    y = batch[data_key]
                    y = torch.log1p(y)

                    # ==========
                    # Forward and Loss
                    # ==========
                    pred = self.model.forward(X)
                    loss_ = F.mse_loss(y, pred)
                    loss = loss_ / self.accumulate_grad

                    if np.isnan(loss.item()):
                        nan_loss = True
                        print("Training loss has NaN, skipping epoch.")
                        self._update_state_dict()
                        break

                # ==========
                # Backward
                # ==========
                scaler.scale(loss).backward()
                moving_avg_loss += loss.item()
                # only update optimizer every accumulate_grad steps
                # this is equivalent to updating every step but with larger batch size (batch_size * accumulate_grad)
                # however, with larger batch size, the GPU memory usage will be higher
                if (batch_id + 1) % self.accumulate_grad == 0:
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

                if (batch_id + 1) % 10 == 0:
                    desc_str = (
                        f" - (Training) {self.cur_epoch} "
                        f"MSE Loss: {moving_avg_loss / (batch_id + 1):.4f}"
                    )
                    bar.set_description(desc_str)
                bar.update(1)

                # early break batch loop
                if batch_id >= self.train_batches:
                    break

            del train_data_loader
            self._cleanup_env()
            if nan_loss:
                # epoch break due to nan loss, skip validation
                continue

            self.train_loss = moving_avg_loss / (batch_id + 1)
            self.cur_lr = optimizer.param_groups[0]["lr"]
            (
                self.val_loss,
                self.val_single_batch_pearson,
                self.val_across_batch_pearson,
                self.example_wandb_images,
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

    def _test(self):
        if self.val_loss is None:
            (
                self.val_loss,
                self.val_single_batch_pearson,
                self.val_across_batch_pearson,
                _,
            ) = self._validation_step(val_batches=None)

        (
            self.test_loss,
            self.test_single_batch_pearson,
            self.test_across_batch_pearson,
            wandb_images,
        ) = self._validation_step(testing=True, val_batches=None)

        if self.wandb_active:
            wandb.summary["final_valid_loss"] = self.val_loss
            wandb.summary["final_valid_single_batch_pearson"] = (
                self.val_single_batch_pearson
            )
            wandb.summary["final_valid_across_batch_pearson"] = (
                self.val_across_batch_pearson
            )
            wandb.summary["final_test_loss"] = self.test_loss
            wandb.summary["final_test_single_batch_pearson"] = (
                self.test_single_batch_pearson
            )
            wandb.summary["final_test_across_batch_pearson"] = (
                self.test_across_batch_pearson
            )
            wandb.summary["final_test_image"] = wandb_images

            # final wandb flag to indicate the run is successfully finished
            wandb.summary["success"] = True

        self._cleanup_env()
        return

    def train(self, use_wandb=True) -> None:
        """
        Train the model.

        Returns
        -------
            None
        """
        wandb_run = self._setup_wandb(use_wandb)
        if wandb_run is None:
            return

        with wandb_run:
            self._fit()
            self._test()

        wandb.finish()
        return


class Track1DUnetTransTrainer(Track1DModelTrainer):
    model_class = UNetTrans

    def __init__(self, config):
        super().__init__(config)
        return
