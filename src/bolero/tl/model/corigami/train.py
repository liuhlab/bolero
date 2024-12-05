import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from bolero.pl.hic import HicExamplePlotter
from bolero.pl.utils import figure_to_array
from bolero.tl.generic.train import GenericTrainer
from bolero.tl.generic.train_helper import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
    insulation_pearson,
    safe_save,
)
from bolero.tl.model.corigami.dataset import HiCTrackDataset
from bolero.tl.model.corigami.model import (
    ConvTransModel,
    ConvTransModelLora,
    ConvTransModelSeqOnly,
)


class CorigamiSeqOnlyTrainer(GenericTrainer):
    trainer_config = {
        "mode": "REQUIRED",
        "chrom_split": None,
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_name": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 80,
        "patience": 80,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": 0.002,
        "weight_decay": 0,
        "accumulate_grad": 1,
        "grad_norm_collector": False,
        "std": 0.1,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "batch_size": "REQUIRED",
        "loss_tolerance": 0.0,
        "pretrained_model": "REQUIRED",
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 1,
        # loss cov cutoff
        "loss_cov_cutoff": 10,
        "plot_example_per_epoch": 9,
    }
    dataset_class = HiCTrackDataset
    model_class = ConvTransModelSeqOnly

    def __init__(self, config):
        # modify model encoder_in_channel based on dna_fifth_channel
        if "dna_fifth_channel" in config and config["dna_fifth_channel"]:
            config["encoder_in_channel"] = 5

        super().__init__(config)
        self.image_scale = config["image_scale"]
        self.std = config["std"]
        self.val_batches = config["val_batches"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.batch_size = config["batch_size"]
        self.best_correlation = -np.Inf

        self._setup_env()
        self._setup_dataset()
        return

    def _load_model_weights(self):
        model_path = self.config["pretrained_model"]
        if model_path is None:
            raise ValueError("Pretrained model path is required.")
        print(f"Setting up model from pretrain model at {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model_weights = checkpoint["state_dict"]
        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)
        return model_weights

    def _setup_model_from_config(self):
        print("Setting up model from config")
        model = self.model_class.create_from_config(self.config)
        model = model.to(self.device)
        return model

    def _setup_model_from_checkpoint(self):
        # load model from path, set parameter to requires_grad, and model to train
        model_weights = self._load_model_weights()
        model = self._setup_model_from_config()
        model.load_state_dict(model_weights)
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return model

    def _setup_model_for_lora(self):
        # load model from path, set parameter to requires_grad, and model to train
        model_weights = self._load_model_weights()
        model = self._setup_model_from_config()
        model.load_state_dict(model_weights)
        model.convert_to_lora(self.config)
        model.train()
        return model

    def _setup_model_for_conditional_lora(self):
        # load model from path, set parameter to requires_grad, and model to train
        model_weights = self._load_model_weights()
        model = self._setup_model_from_config()
        model.load_state_dict(model_weights)
        model.convert_to_lora(self.config)
        model.init_embedding()
        model.to(self.device)
        model.train()
        return model

    def _get_optimizer(self):
        lr = self.config["lr"]
        weight_decay = self.config["weight_decay"]
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def _get_scheduler(self, optimizer):
        try:
            import pl_bolts

            scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=10, max_epochs=40
            )
        except ImportError:
            self.config["scheduler_type"] = "borzoi"
            warmup_steps = self.config.get("warmup_steps", 1000)
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
        return scheduler

    def _setup_model(self):
        mode = self.mode

        if mode == "base":
            self.model = self._setup_model_from_config()
        elif mode == "lora_finetune":
            self.model = self._setup_model_for_lora()
        elif mode == "conditional_lora_finetune":
            self.model = self._setup_model_for_conditional_lora()
        elif mode == "finetune":
            self.model = self._setup_model_from_checkpoint()
        else:
            raise ValueError(
                f"Incorrect mode: {mode}, should be one of ['base', 'finetune']."
            )
        self._set_total_params()
        return

    def _save_checkpoint(self, update_best: bool):
        epoch_info = {
            "epoch": self.cur_epoch,
            "early_stopping_counter": self.early_stopping_counter,
        }
        safe_save(epoch_info, self.epoch_info_path)
        if update_best:
            print("Saving best checkpoint...")
            # check point includes model and other training states
            checkpoint = {
                "best_val_loss": self.best_val_loss,
                "best_correlation": self.best_correlation,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "ema": self.ema.state_dict() if self.ema is not None else None,
            }
            safe_save(checkpoint, self.best_checkpoint_path)

            # save best model in a separate file
            if self.config["use_ema"]:
                safe_save(self.ema.ema_model, self.best_model_path)
            else:
                safe_save(self.model, self.best_model_path)

        if self.grad_norm_collector:
            self.grad_norm_collector.save(
                self.model_log_dir / f"epoch_{self.cur_epoch}_grad_norms.json"
            )
            self.grad_norm_collector.reset()
        return

    def _update_state_dict(self):
        self._cleanup_env()

        print(
            f"Load and update state dict from checkpoint file: {self.best_checkpoint_path}"
        )
        checkpoint: dict = torch.load(self.best_checkpoint_path, weights_only=False)
        try:
            epoch_info = torch.load(self.epoch_info_path, weights_only=False)
            checkpoint.update(epoch_info)
        except FileNotFoundError:
            print("Epoch info not found, skipping.")

        # adjust epochs
        self.cur_epoch = checkpoint.get("epoch", 0)
        self.early_stopping_counter = checkpoint.get("early_stopping_counter", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", np.Inf)
        self.best_correlation = checkpoint.get("best_correlation", -np.Inf)
        print(
            f"Best val loss: {self.best_val_loss:.5f}, "
            f"Best correlation: {self.best_correlation:.5f}, "
            f"early stopping counter: {self.early_stopping_counter}."
        )

        # load state dict
        self.model.load_state_dict(checkpoint["state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])

        del checkpoint
        self._cleanup_env()
        return

    def _model_forward_pass(self, model: torch.nn.Module, batch: dict):
        # ==========
        # X
        # ==========
        X = batch["dna_one_hot"]

        # ==========
        # y_hic
        # ==========
        y = batch["values"]

        # ==========
        # Forward
        # ==========
        pred_y = model(X)
        return y, pred_y

    def _plot_example_images(
        self, example_batches, target_key="values", predict_key="pred_"
    ):
        epoch = self.cur_epoch + 1
        wandb_images = []
        for idx, batch in enumerate(example_batches):
            plotter = HicExamplePlotter(target_key, predict_key)
            fig, _ = plotter.plot(
                batch,
                figsize=(40, 20),
                dpi=100,
                top_example=2,
                bottom_example=2,
                plot_channel=0,
                vmin=self.config["plot_vmin"],
                vmax=self.config["plot_vmax"],
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

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
    ):
        if val_batches is None:
            print_step = 10
        else:
            print_step = max(5, val_batches // 20)
        # if val batches is None, use all batches in the dataset

        size = 0
        val_loss = 0
        single_batch_pearson_counter = CumulativeCounter()
        across_batch_pearson_counter = CumulativePearson()

        region2 = getattr(self.dataset, "region2", None)

        example_batches = []  # collect example batches for making images
        for batch_id, batch in enumerate(dataloader):
            if region2 is None:
                # original corigami model forward pass
                y, pred_y = self._model_forward_pass(model, batch)
                loss_ = F.mse_loss(pred_y, y)
            else:
                # borzoi corigami model forward pass
                if region2:
                    (*_, y), (*_, pred_y), loss_ = self._model_forward_pass(
                        model, batch
                    )
                else:
                    y, pred_y, loss_ = self._model_forward_pass(model, batch)

            # mask is element wise mask based on coverage > cutoff
            val_loss += loss_.item()

            # ==========
            # Within batch pearson and save for across batch pearson
            # ==========
            # within batch pearson
            corr = batch_pearson_correlation(pred_y, y).detach().cpu()[:, None]
            single_batch_pearson_counter.update(corr)
            # save for across batch pearson
            across_batch_pearson_counter.update(pred_y, y)
            # insulation score
            insulation_score = np.mean(insulation_pearson(pred_y, y))

            size += 1
            if batch_id < self.plot_example_per_epoch:
                batch["values"] = y.detach()
                batch["pred_"] = pred_y.detach()
                example_batches.append(batch)

            if ((batch_id + 1) % print_step) == 0:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} [{batch_id}/{val_batches}] "
                    f"Loss: {val_loss/size:.3f}; "
                    f"Within batch Pearson: {single_batch_pearson_counter.mean():.3f}; "
                    f"Across batch Pearson: {across_batch_pearson_counter.corr():.3f}; "
                    f"Insulation Pearson: {insulation_score:.3f}"
                )
                print(desc_str)

        del dataloader
        self._cleanup_env()

        wandb_images = self._plot_example_images(
            example_batches, target_key="values", predict_key="pred_"
        )

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

        return (
            val_loss,
            single_batch_pearson,
            across_batch_pearson,
            insulation_score,
            wandb_images,
        )

    def _log_save_and_check_stop(self):
        epoch = self.cur_epoch
        train_loss = self.train_loss
        learning_rate = self.cur_lr
        val_loss = self.val_loss
        single_batch_pearson = self.single_batch_pearson
        across_batch_pearson = self.across_batch_pearson
        insulation_score = self.insulation_score
        example_images = self.example_wandb_images

        print(
            f" - (Training) {epoch}; Loss: {train_loss:.3f}; Learning rate {learning_rate}."
        )
        print(f" - (Validation) {epoch} Loss: {val_loss:.3f}")
        print(f"Single Batch Pearson Corr.: {single_batch_pearson:.3f}")
        print(f"Across Batch Pearson Corr.: {across_batch_pearson:.3f}")
        print(f"Insulation Pearson Corr.: {insulation_score:.3f}")

        # only clear the early stopping counter if the pearson correlation is better than tolerance
        previous_best = self.best_correlation
        if single_batch_pearson > self.best_correlation:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        print(
            f"Previous best correlation: {previous_best:.3f}, "
            f"Single Pearson Correlation at epoch {epoch}: {single_batch_pearson:.3f}; "
            f"Early stopping counter: {self.early_stopping_counter}"
        )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if single_batch_pearson > self.best_correlation:
            self.best_correlation = single_batch_pearson
            self._save_checkpoint(update_best=True)
        else:
            self._save_checkpoint(update_best=False)
        if self.wandb_active:
            wandb.log(
                {
                    # "train/train_loss": train_loss,
                    # "train/learning_rate": learning_rate,
                    "val/val_loss": val_loss,
                    "val/best_val_loss": self.best_val_loss,
                    "val/best_correlation": self.best_correlation,
                    "val/early_stopping_counter": self.early_stopping_counter,
                    "val/single_batch_pearson": single_batch_pearson,
                    "val/across_batch_pearson": across_batch_pearson,
                    "val/insulation_score": insulation_score,
                    "val_example/example_predictions": example_images,
                }
            )
        flag = self.early_stopping_counter >= self.patience
        return flag

    def _validation_step(self, testing=False, val_batches=None):
        """Generic validation step."""
        val_batches = val_batches or self.val_batches
        if testing:
            dataloader = self.get_test_dataloader(batches=val_batches)
        else:
            dataloader = self.get_valid_dataloader(batches=val_batches)

        with torch.inference_mode():
            if self.use_ema:
                self.ema.eval()
                self.ema.ema_model.eval()
                (
                    val_loss,
                    single_batch_pearson,
                    across_batch_pearson,
                    insulation_score,
                    wandb_images,
                ) = self._model_validation_step(
                    model=self.ema.ema_model,
                    dataloader=dataloader,
                    val_batches=val_batches,
                )
                self.ema.train()
                self.ema.ema_model.train()
            else:
                self.model.eval()
                (
                    val_loss,
                    single_batch_pearson,
                    across_batch_pearson,
                    insulation_score,
                    wandb_images,
                ) = self._model_validation_step(
                    model=self.model,
                    dataloader=dataloader,
                    val_batches=val_batches,
                )
                self.model.train()
        return (
            val_loss,
            single_batch_pearson,
            across_batch_pearson,
            insulation_score,
            wandb_images,
        )

    def _fit(self, max_epochs=None, valid_first=False):
        if max_epochs is None:
            max_epochs = self.max_epochs

        # dataset related
        scaler = self.scaler
        optimizer = self.optimizer
        scheduler = self.scheduler
        ema = self.ema
        self.val_loss = None

        if valid_first:
            print("Perform validation before training.")
            (
                self.val_loss,
                self.single_batch_pearson,
                self.across_batch_pearson,
                self.insulation_score,
                wandb_images,
            ) = self._validation_step()
            print(f"Validation loss before training: {self.val_loss:.4f}")
            print(f"Validation Singe Batch pearson: {self.single_batch_pearson:.3f}")
            print(f"Validation Across Batch pearson: {self.across_batch_pearson:.3f}.")
            print(f"Validation insulation pearson: {self.insulation_score:.3f}")
            wandb.log(
                {
                    "val/val_loss": self.val_loss,
                    "val/single_batch_pearson": self.single_batch_pearson,
                    "val/across_batch_pearson": self.across_batch_pearson,
                    "val/insulation_score": self.insulation_score,
                    "val_example/example_images": wandb_images,
                }
            )

        stop_flag = self.early_stopping_counter >= self.patience
        if self.cur_epoch > 0:
            print(
                f"Resuming training from epoch {self.cur_epoch}, with {max_epochs} epochs in total."
            )
        # moving_norm = MovingMetric(window_size=100)
        example_step = max(5, self.train_batches // (self.plot_example_per_epoch + 1))
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
            moving_avg_loss = 0
            cur_loss = 1e10
            nan_loss = False
            total_norm = 999

            if self.train_batches is None:
                print_steps = 10
            else:
                print_steps = max(5, self.train_batches // 50)
            region2 = getattr(self.dataset, "region2", None)
            for batch_id, batch in enumerate(dataloader):
                try:
                    auto_cast_context = torch.autocast(
                        device_type=str(self.device).split(":")[0],
                        dtype=torch.bfloat16,
                        enabled=self.use_amp,
                    )
                except RuntimeError:
                    # some GPU, such as T4 does not support bfloat16
                    print("bfloat16 autocast failed, using float16 instead.")
                    auto_cast_context = torch.autocast(
                        device_type=str(self.device).split(":")[0],
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    )
                with auto_cast_context:
                    if region2 is None:
                        # original corigami model forward pass
                        y, pred_y = self._model_forward_pass(self.model, batch)
                        loss = F.mse_loss(pred_y, y)
                    else:
                        # borzoi corigami model forward pass
                        y, pred_y, loss = self._model_forward_pass(self.model, batch)
                        # region_1_y_true, region_2_y_true, region_12_y_true = y
                        # region_1_y_pred, region_2_y_pred, region_12_y_pred = pred_y

                    loss = loss / self.accumulate_grad

                    if np.isnan(loss.item()):
                        nan_loss = True
                        print("Training loss has NaN, skipping epoch.")
                        if self.checkpoint:
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

                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_norm
                    )
                    total_norm = total_norm.item()

                    # check moving norm and skip step if the norm is too large (e.g. > 99% moving quantile)
                    # this is to prevent outlier gradients from messing up the training
                    # moving_norm.update(total_norm)
                    # threshold = moving_norm.quantile(0.99).item()
                    # if (total_norm > threshold) and moving_norm.full:
                    #     print(
                    #         f"Gradient norm is too large: {total_norm:.4f}, "
                    #         f"threshold: {threshold:.4f}, prevent update."
                    #     )
                    #     optimizer.zero_grad()

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if self.config.get("scheduler_type", None) == "borzoi":
                        scheduler.step()

                    if ema:
                        ema.update()

                    log_dict = {
                        "train/train_loss": loss.item(),
                        "train/train_total_grad_norm": total_norm,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    wandb.log(log_dict)

                if (batch_id + 1) % print_steps == 0:
                    _loss = moving_avg_loss / (batch_id + 1)
                    _cur_lr = optimizer.param_groups[0]["lr"]
                    desc_str = (
                        f" - (Training) {self.cur_epoch} {batch_id} "
                        f"Loss: {_loss:.4f} "
                        f"Learning rate: {_cur_lr:.4f} "
                        f"Total norm: {total_norm:.4f} "
                    )

                    if _loss > (cur_loss + 0.5):
                        batch["cur_loss"] = _loss
                        batch["last_loss"] = cur_loss
                        print(f"Batch {batch_id} loss increased.")
                        joblib.dump(
                            batch,
                            f"{self.savename}.epoch{self.cur_epoch}.batch{batch_id}.joblib",
                        )

                    cur_loss = _loss
                    print(desc_str)

                if batch_id % example_step == 0:
                    log_dict = {}
                    if region2:
                        for idx, prefix in enumerate(
                            ["region_1", "region_2", "region_1_2"]
                        ):
                            batch["values"] = y[idx].detach()
                            batch["pred_"] = pred_y[idx].detach()
                            wandb_images = self._plot_example_images(
                                [batch], target_key="values", predict_key="pred_"
                            )
                            log_dict[f"train_example/{prefix}_example_heatmap"] = (
                                wandb_images
                            )
                    else:
                        batch["values"] = y.detach()
                        batch["pred_"] = pred_y.detach()
                        wandb_images = self._plot_example_images(
                            [batch], target_key="values", predict_key="pred_"
                        )
                        log_dict["train_example/example_heatmap"] = wandb_images
                    wandb.log(log_dict)

            del dataloader
            self._cleanup_env()
            if nan_loss:
                # epoch break due to nan loss, skip validation
                continue

            self.train_loss = moving_avg_loss / (batch_id + 1)
            self.cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f" - (Training) {self.cur_epoch} Learning rate from optimizer: {self.cur_lr:.3f}"
            )
            try:
                print(
                    f" - (Training) {self.cur_epoch} Learning rate from scheduler: {self.scheduler.get_lr()[0]:.3f}"
                )
            except NotImplementedError:
                print(
                    f" - (Training) {self.cur_epoch} Learning rate from scheduler: {self.scheduler.get_last_lr()[0]:.3f}"
                )

            if self.config.get("scheduler_type", None) != "borzoi":
                if scheduler is not None:
                    scheduler.step()

            (
                self.val_loss,
                self.single_batch_pearson,
                self.across_batch_pearson,
                self.insulation_score,
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
                self.single_batch_pearson,
                self.across_batch_pearson,
                self.insulation_score,
                _,
            ) = self._validation_step()

        (
            self.test_loss,
            self.test_single_batch_pearson,
            self.test_across_batch_pearson,
            self.insulation_score,
            wandb_images,
        ) = self._validation_step(testing=True)

        wandb.summary["final_valid_loss"] = self.val_loss
        wandb.summary["final_valid_within"] = self.single_batch_pearson
        wandb.summary["final_valid_across"] = self.across_batch_pearson
        wandb.summary["final_test_loss"] = self.test_loss
        wandb.summary["final_test_within"] = self.test_single_batch_pearson
        wandb.summary["final_test_across"] = self.test_across_batch_pearson
        wandb.summary["final_insulation_score"] = self.insulation_score
        wandb.summary["final_image"] = wandb_images

        # final wandb flag to indicate the run is successfully finished
        wandb.summary["success"] = True
        return

    def train(self, valid_first=None) -> None:
        """
        Train the model.

        Returns
        -------
        None
        """
        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        if valid_first is None:
            if self.mode == "finetune":
                valid_first = True

        with wandb_run:
            self.checkpoint = self._has_last_checkpoint()
            self._setup_model()
            self._setup_fit()
            self._fit(valid_first=valid_first)
            self._test()
            self._cleanup_env()
            wandb.finish()
        return


class CorigamiTrainer(CorigamiSeqOnlyTrainer):
    trainer_config = {
        "mode": "REQUIRED",
        "chrom_split": None,
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_name": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 80,
        "patience": 80,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": 0.002,
        "weight_decay": 0,
        "accumulate_grad": 1,
        "grad_norm_collector": True,
        "std": "REQUIRED",
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "batch_size": "REQUIRED",
        "loss_tolerance": 0.0,
        "pretrained_model": None,
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 1,
        # loss cov cutoff
        "loss_cov_cutoff": 10,
        "plot_example_per_epoch": 9,
    }
    dataset_class = HiCTrackDataset
    model_class = ConvTransModel

    def __init__(self, config):
        super().__init__(config)

    def _gaussian_noise(self, inputs, std=0.1):
        """Add Gaussian noise to the input tensor."""
        return inputs + torch.randn_like(inputs) * std

    def _model_forward_pass(self, model: torch.nn.Module, batch: dict):
        # ==========
        # X
        # ==========
        start = time.time()
        dna_seq = batch["dna_one_hot"]
        if torch.is_autocast_enabled():
            if dna_seq.dtype != torch.float16:
                dna_seq = dna_seq.half()
        else:
            if dna_seq.dtype != torch.float32:
                dna_seq = dna_seq.float()
        if self.dataset._dataset_mode == "train":
            dna_seq = self._gaussian_noise(dna_seq, self.std)
            feature_list = [
                self._gaussian_noise(batch[feat], self.std)
                for feat in self.config["data_1d_keys"]
            ]
        else:
            feature_list = [batch[feat] for feat in self.config["data_1d_keys"]]
        features = torch.cat([feature.unsqueeze(1) for feature in feature_list], dim=1)
        X = torch.cat([dna_seq, features], dim=1)
        # ==========
        # y_hic
        # ==========
        y = batch["values"]

        end = time.time()
        print(f"Data prep time: {end-start:.3f}")
        # ==========
        # Forward
        # ==========
        pred_y = model(X)
        return y, pred_y


class CorigamiLoraTrainer(CorigamiTrainer):
    trainer_config = {
        "mode": "REQUIRED",
        "chrom_split": None,
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_name": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 80,
        "patience": 80,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": 0.0001,
        "weight_decay": 0,
        "accumulate_grad": 1,
        "grad_norm_collector": True,
        "std": "REQUIRED",
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "batch_size": "REQUIRED",
        "loss_tolerance": 0.0,
        "pretrained_model": None,
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 1,
        # loss cov cutoff
        "loss_cov_cutoff": 10,
        "plot_example_per_epoch": 9,
    }
    dataset_class = HiCTrackDataset
    model_class = ConvTransModelLora

    def __init__(self, config):
        super().__init__(config)

    def _model_forward_pass(self, model: torch.nn.Module, batch: dict):
        # ==========
        # X
        # ==========
        start = time.time()
        dna_seq = batch["dna_one_hot"]
        if torch.is_autocast_enabled():
            if dna_seq.dtype != torch.float16:
                dna_seq = dna_seq.half()
        else:
            if dna_seq.dtype != torch.float32:
                dna_seq = dna_seq.float()
        if self.dataset._dataset_mode == "train":
            dna_seq = self._gaussian_noise(dna_seq, self.std)
            feature_list = [
                self._gaussian_noise(batch[feat], self.std)
                for feat in self.config["data_1d_keys"]
            ]
        else:
            feature_list = [batch[feat] for feat in self.config["data_1d_keys"]]
        features = torch.cat([feature.unsqueeze(1) for feature in feature_list], dim=1)
        X = torch.cat([dna_seq, features], dim=1)
        # ==========
        # y_hic
        # ==========
        y = batch["values"]
        embedding = batch.get("embedding", None)

        end = time.time()
        print(f"Data prep time: {end-start:.3f}")
        # ==========
        # Forward
        # ==========
        pred_y = model(X, embedding=embedding)
        return y, pred_y
