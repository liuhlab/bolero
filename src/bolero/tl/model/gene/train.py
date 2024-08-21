import joblib
import numpy as np
import torch
import wandb

from bolero.tl.generic.train_gene import GenericGeneModelTrainer

from .dataset import GeneDataset
from .model_simple_vae import BaseVQVAE


class GeneModelTrainer(GenericGeneModelTrainer):
    prefix: str
    model_class = BaseVQVAE
    dataset_class = GeneDataset

    def __init__(self, config):
        super().__init__(config)

        self.model: BaseVQVAE = None

        self._setup_env()

        self.dataset: GeneDataset
        self._setup_dataset()
        self._setup_scaler_with_train_folds()
        return

    def _setup_scaler_with_train_folds(self):
        """
        Get the gene scaler using the training folds, apply it to all data loaders.
        """
        self.gene_scaler = self.dataset.get_gene_scaler(self.train_folds)

    def get_train_dataloader(self, batches):
        """Get train data loader with gene scaler."""
        return super().get_train_dataloader(batches, scaler=self.gene_scaler)

    def get_valid_dataloader(self, batches):
        """Get valid data loader with gene scaler."""
        return super().get_valid_dataloader(batches, scaler=self.gene_scaler)

    def get_test_dataloader(self, batches):
        """Get test data loader with gene scaler."""
        return super().get_test_dataloader(batches, scaler=self.gene_scaler)

    def _setup_model(self):
        gene_order = self.dataset.pre_estimate_genes(
            qc_genes=self.config["qc_genes"],
            sel_genes=self.config["sel_genes"],
        )
        if self.dataset.pca:
            input_dim = self.dataset.pca_n_components
        else:
            input_dim = len(gene_order)
        self.config["input_dim"] = input_dim

        self.model = self._setup_model_from_config()

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
    ):
        print_step = max(5, val_batches // 20)
        val_loss = 0

        for batch_id, batch in enumerate(dataloader):
            loss_, loss_breakdown = self._model_forward_pass(model, batch)
            val_loss += loss_.item()
            loss_str = "; ".join(f"{value:.2f}" for value in loss_breakdown)
            if ((batch_id + 1) % print_step) == 0:
                print(
                    f" - (Validation) {self.cur_epoch} [{batch_id}/{val_batches}] "
                    f"Loss: {val_loss/(batch_id + 1):.3f}; {loss_str}"
                )
        val_loss = val_loss / (batch_id + 1)

        del dataloader
        self._cleanup_env()
        return val_loss

    def _validation_step(self, testing=False, val_batches=None):
        """Overwriting the validation step to return pearson correlation per cell"""
        val_batches = val_batches or self.val_batches
        if testing:
            dataloader = self.get_test_dataloader(batches=val_batches)
        else:
            dataloader = self.get_valid_dataloader(batches=val_batches)

        with torch.inference_mode():
            if self.use_ema:
                self.ema.eval()
                self.ema.ema_model.eval()
                val_loss = self._model_validation_step(
                    model=self.ema.ema_model,
                    dataloader=dataloader,
                    val_batches=val_batches,
                )
                self.ema.train()
                self.ema.ema_model.train()
            else:
                self.model.eval()
                val_loss = self._model_validation_step(
                    model=self.model,
                    dataloader=dataloader,
                    val_batches=val_batches,
                )
                self.model.train()
        return val_loss

    def _log_save_and_check_stop(self):
        epoch = self.cur_epoch
        train_loss = self.train_loss
        learning_rate = self.cur_lr
        val_loss = self.val_loss

        print(
            f" - (Training) {epoch}; Loss: {train_loss:.3f}; Learning rate {learning_rate}."
        )
        print(f" - (Validation) {epoch} Loss: {val_loss:.3f}")

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
                }
            )
        flag = self.early_stopping_counter >= self.patience
        return flag

    def _test(self):
        if self.val_loss is None:
            self.val_loss = self._validation_step(val_batches=1500)
        self.test_loss = self._validation_step(testing=True, val_batches=1500)

        wandb.summary["final_valid_loss"] = self.val_loss
        wandb.summary["final_test_loss"] = self.test_loss
        # final wandb flag to indicate the run is successfully finished
        wandb.summary["success"] = True
        return

    def _model_forward_pass(self, model, batch):
        """Model specific validation step."""
        X = batch["gene_exp"]

        _, _, _, loss, loss_breakdown = model(X)
        return loss, loss_breakdown

    def fit(self, max_epochs=None):
        """
        Model specific training loop.
        """
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

            print_steps = max(5, self.train_batches // 50)
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
                    loss, loss_breakdown = self._model_forward_pass(self.model, batch)
                    loss = loss / self.accumulate_grad

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

                if (batch_id + 1) % print_steps == 0:
                    _loss = moving_avg_loss / (batch_id + 1)
                    loss_str = "; ".join(f"{value:.2f}" for value in loss_breakdown)
                    desc_str = (
                        f" - (Training) {self.cur_epoch} {batch_id} "
                        f"Loss: {_loss:.4f}; {loss_str}"
                    )
                    print(desc_str)

                    if _loss > (cur_loss + 0.5):
                        batch["cur_loss"] = _loss
                        batch["last_loss"] = cur_loss
                        print(f"Batch {batch_id} loss increased.")
                        joblib.dump(
                            batch,
                            f"{self.savename}.epoch{self.cur_epoch}.batch{batch_id}.joblib",
                        )

                    cur_loss = _loss

            del dataloader
            self._cleanup_env()
            if nan_loss:
                # epoch break due to nan loss, skip validation
                continue

            self.train_loss = moving_avg_loss / (batch_id + 1)
            self.cur_lr = optimizer.param_groups[0]["lr"]
            self.val_loss = self._validation_step()

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

    def train(self) -> None:
        """Train the model."""
        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            self.checkpoint = self._has_last_checkpoint()
            self._setup_model()
            self._setup_fit()
            self.fit()
            self._test()
            self._cleanup_env()
            wandb.finish()
        return
