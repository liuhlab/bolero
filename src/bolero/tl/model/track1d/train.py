import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from bolero.tl.model.track1d.model import DialatedCNNTrack1DModel
from bolero.tl.model.track1d.dataset import Track1DDataset
from bolero.tl.model.generic.train_helper import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
)
from bolero.pl.track1d import Track1DExamplePlotter
from bolero.pl.utils import figure_to_array
from bolero.tl.model.generic.train import GenericTrainer


class Track1DModelTrainer(GenericTrainer):
    """Train model for predicting 1-D genome tracks."""

    trainer_config = {
        "mode": "init",
        "chrom_split": "REQUIRED",
        "sample": "REQUIRED",
        "region": "REQUIRED",
        "output_dir": "./track_1d",
        "savename": "model",
        "wandb_project": "scprinter",
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
    }
    dataset_class = Track1DDataset
    model_class = DialatedCNNTrack1DModel

    def __init__(self, config):
        super().__init__(config)

        self.model: DialatedCNNTrack1DModel = None

    def _setup_model(self):
        mode = self.mode

        if mode == "init":
            self.model = self._setup_model_from_config()
        else:
            raise ValueError(f"Incorrect mode: {mode}.")

        # collect some shortcuts post model setup
        self.dna_len = self.model.dna_len
        self.output_len = self.model.output_len

        self._set_total_params()
        return

    def _setup_fit(self):
        config = self.config

        # epochs
        self.max_epochs = config["max_epochs"]
        self.patience = config["patience"]
        self.loss_tolerance = config["loss_tolerance"]
        self.train_batches = config["train_batches"]
        self.val_batches = config["val_batches"]
        self.early_stopping_counter = 0
        self.early_stoped = False
        self.best_val_loss = float("inf")
        self.accumulate_grad = config["accumulate_grad"]
        self.cur_epoch = 0

        # scaler
        if config["use_amp"]:
            self.scaler = self._get_scaler()

        # optimizer
        self.learning_rate = config["lr"]
        self.optimizer = self._get_optimizer(self.learning_rate, weight_decay=config["weight_decay"])

        # scheduler
        if config["scheduler"]:
            self.scheduler = self._get_scheduler(self.optimizer)
        else:
            self.scheduler = None

        # EMA model
        self.use_ema = config["use_ema"]
        if self.use_ema:
            self.ema = self._get_ema()
        else:
            self.ema = None

        # plot
        self.plot_example_per_epoch = config["plot_example_per_epoch"]
        if not self.plot_example_per_epoch:
            self.plot_example_per_epoch = 0

        # update state dict if checkpoint exists
        if self.checkpoint:
            self._update_state_dict()
        return

    @torch.no_grad()
    def model_validation_step(
        self,
        model,
        val_dataset,
        sample=None,
        region=None,
        val_batches=None,
    ):
        if val_batches is None:
            val_batches = self.val_batches
        # if val batches is None, use all batches in the dataset

        val_data_loader = val_dataset.get_dataloader(
            sample=sample,
            region=region,
            local_shuffle_buffer_size=0,
            randomize_block_order=False,
        )
        data_key = f"{region}|{sample}"
        dna_key = f"{region}|{val_dataset.dna_name}"

        size = 0
        val_loss = [0]
        profile_pearson_counter = CumulativeCounter()
        across_batch_pearson_cov = CumulativePearson()

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
            # y_footprint, y_coverage
            # ==========
            y_coverage = batch[data_key]
            y_coverage = torch.log1p(y_coverage)

            # ==========
            # Forward and Loss
            # ==========
            pred_coverage = model(X)
            loss_ = F.mse_loss(pred_coverage, y_coverage)
            pred_score = pred_score.reshape((len(pred_score), -1))
            val_loss[0] += loss_.item()

            # ==========
            # Within batch pearson and save for across batch pearson
            # ==========
            # within batch pearson
            corr = (
                batch_pearson_correlation(pred_coverage, y_coverage)
                .detach()
                .cpu()[:, None]
            )
            profile_pearson_counter.update(corr)
            # save for across batch pearson
            across_batch_pearson_cov.update(pred_coverage, y_coverage)

            size += 1
            if batch_id < self.plot_example_per_epoch:
                batch["pred_score"] = pred_coverage.detach().cpu().numpy()
                example_batches.append(batch)

            if size > 5:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} "
                    f"Footprint Loss: {val_loss[0]/size:.4f} "
                )
                bar.set_description(desc_str)
            if batch_id >= val_batches:
                break
        bar.close()
        del val_data_loader

        self._cleanup_env()
        wandb_images = self._plot_example_footprints(
            example_batches,
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
            across_batch_pearson_cov.corr(),
        ]
        return val_loss, profile_pearson, across_corr, wandb_images

    def _plot_example_footprints(
        self, example_batches, footprinter, atac_key, bias_key, footprint_key
    ):
        epoch = self.cur_epoch + 1
        wandb_images = []
        for idx, batch in enumerate(example_batches):
            plotter = Track1DExamplePlotter(
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
        val_loss = self.val_loss
        profile_pearson = self.val_profile_pearson
        across_pearson = self.val_across_pearson

        print(
            f" - (Training) {epoch} Footprint Loss: {train_fp_loss:.5f}; Coverage Loss: {train_cov_loss:.5f}; Learning rate {learning_rate}."
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
                "train/train_fp_loss": train_fp_loss,
                "train/train_cov_loss": train_cov_loss,
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

    def _fit(self, sample, region, max_epochs=None, valid_first=False):
        if max_epochs is None:
            max_epochs = self.max_epochs

        mode = self.mode

        # dataset related
        training_dataset = self.train_dataset

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
                self.val_profile_pearson,
                self.val_across_pearson,
                wandb_images,
            ) = self._validation_step(sample=sample, region=region)
            print(f"Validation loss before training: {self.val_loss:.4f}")
            print(f"Validation Profile pearson: {self.val_profile_pearson[0]:.3f}")
            print(
                f"Validation Across peak footprint pearson: {self.val_across_pearson[0]:.3f}."
            )
            print(
                f"Validation Across peak coverage pearson: {self.val_across_pearson[1]:.3f}."
            )
            wandb.log(
                {
                    "val/val_loss": self.val_loss,
                    "val/profile_pearson": self.val_profile_pearson[0],
                    "val/across_pearson_footprint": self.val_across_pearson[0],
                    "val/across_pearson_coverage": self.val_across_pearson[1],
                    "val_example/example_footprints": wandb_images,
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
            moving_avg_cov_loss = 0
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
                    # y_footprint, y_coverage
                    # ==========
                    random_modes = np.random.permutation(self.modes)[
                        : self.select_n_modes
                    ]
                    select_index = torch.as_tensor(
                        [self.modes_index.index(mode) for mode in random_modes]
                    )
                    y_coverage = batch[data_key].sum(dim=-1)
                    y_coverage = torch.log1p(y_coverage)

                    # ==========
                    # Forward and Loss
                    # ==========
                    pred_coverage = self.model.forward(
                        X,
                        modes=select_index,
                    )
                    loss_coverage = F.mse_loss(y_coverage, pred_coverage)
                    loss = loss_coverage / self.accumulate_grad

                    if np.isnan(loss.item()):
                        nan_loss = True
                        print("Training loss has NaN, skipping epoch.")
                        self._update_state_dict()
                        break

                # ==========
                # Backward
                # ==========
                scaler.scale(loss).backward()
                moving_avg_cov_loss += loss_coverage.item()
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

                if (batch_id + 1) % 5 == 0:
                    desc_str = (
                        f" - (Training) {self.cur_epoch} "
                        f"Coverage Loss: {moving_avg_cov_loss / (batch_id + 1):.4f}"
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

            self.train_cov_loss = moving_avg_cov_loss / (batch_id + 1)
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
        self._cleanup_env()
        return

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
