import joblib
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from skimage.transform import resize

from bolero.tl.generic.train import GenericTrainer
from bolero.tl.model.corigami.model import ConvTransModel, ConvTransModelSeqOnly
from bolero.tl.model.hic.dataset import HiCTrackDataset, reverse_comp_hic_data_batch


class CorigamiSeqOnlyTrainer(GenericTrainer):
    trainer_config = {
        "mode": "REQUIRED",
        "chrom_split": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 80,
        "patience": 80,
        "use_amp": True,
        "use_ema": True,
        "scheduler": True,
        "lr": 0.0002,
        "weight_decay": 0,
        "accumulate_grad": 1,
        "expected_dna_length": 500 * (2**13),
        "std": 0.1,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "loss_tolerance": 0.0,
        "pretrained_model": None,
        # loss cov cutoff
        "loss_cov_cutoff": 10,
        "plot_example_per_epoch": None,
    }
    dataset_class = HiCTrackDataset
    model_class = ConvTransModelSeqOnly

    def __init__(self, config):
        super().__init__(config)
        self.expected_dna_length = config["expected_dna_length"]
        self.image_scale = config["image_scale"]
        self.std = config["std"]
        self._setup_env()
        self._setup_dataset()
        return

    def _setup_model_from_config(self):
        print("Setting up model from config")
        model = self.model_class.create_from_config(self.config)
        model.to(self.device)
        return model

    def _setup_model_from_pretrain(self):
        # load model from path, set parameter to requires_grad, and model to train
        model_path = self.config["pretrained_model"]
        if model_path is None:
            raise ValueError("Pretrained model path is required.")
        print(f"Setting up model from pretrain model at {model_path}")

        model = torch.load(model_path)
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return model

    def _get_optimizer(self):
        lr = self.config["lr"]
        weight_decay = self.config["weight_decay"]
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def _get_scheduler(self, optimizer):
        import pl_bolts

        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=self.config["max_epochs"]
        )
        return scheduler

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

    def gaussian_noise(self, inputs, std=1):
        """
        Add Gaussian noise to the input.
        """
        noise = np.random.randn(*inputs.shape) * std
        outputs = inputs + noise
        return outputs

    def _model_forward_pass(self, model: torch.nn.Module, batch: dict):
        # ==========
        # X
        # ==========
        genome = self.dataset.genome
        # TODO can test float 16 if not impact performance
        dna_one_hot = genome.get_regions_one_hot(batch["region"])
        curr_dna_length = dna_one_hot.shape[1]
        if self.expected_dna_length and self.expected_dna_length > curr_dna_length:
            raise ValueError(
                f"Expected DNA length {self.expected_dna_length} is longer than current DNA length {curr_dna_length}."
            )
        else:
            radius = (curr_dna_length - self.expected_dna_length) // 2
            dna_one_hot = dna_one_hot[:, radius:-radius, :].astype(np.float32)
        batch["dna_one_hot"] = self.guassian_noise(dna_one_hot, self.std)
        batch["value"] = batch["value"][:, 0, :, :]
        batch["value"] = resize(
            batch["value"],
            (batch["value"].shape[0], self.image_scale, self.image_scale),
            anti_aliasing=True,
        )
        batch["value"] = np.log(batch["value"] + 1)
        batch = reverse_comp_hic_data_batch(batch, data_1d_keys=None)
        X = torch.from_numpy(batch["dna_one_hot"]).to(self.device)

        # ==========
        # y_hic
        # ==========
        y = torch.from_numpy(batch["value"]).float().to(self.device)

        # ==========
        # Forward
        # ==========
        pred_y = model(X)
        return y, pred_y

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
    ):
        print_step = max(5, val_batches // 20)
        # if val batches is None, use all batches in the dataset

        size = 0
        val_loss = 0

        for batch_id, batch in enumerate(dataloader):
            y, pred_y = self._model_forward_pass(model, batch)

            # mask is element wise mask based on coverage > cutoff
            loss_ = F.mse_loss(pred_y, y)
            val_loss += loss_.item()

            size += 1

            # TODO metrics
            # insulation score
            # Pearson correlation

            if ((batch_id + 1) % print_step) == 0:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} [{batch_id}/{val_batches}] "
                    f"Loss: {val_loss/size:.3f}; "
                )
                print(desc_str)

        del dataloader
        self._cleanup_env()

        # ==========
        # Loss
        # ==========
        val_loss = val_loss / size

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
            (self.val_loss,) = self._validation_step()
            print(f"Validation loss before training: {self.val_loss:.4f}")
            wandb.log(
                {
                    "val/val_loss": self.val_loss,
                }
            )

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
                    y, pred_y = self._model_forward_pass(self.model, batch)
                    loss = F.mse_loss(pred_y, y)
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

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if ema:
                        ema.update()

                    if scheduler is not None:
                        scheduler.step()

                if (batch_id + 1) % print_steps == 0:
                    _loss = moving_avg_loss / (batch_id + 1)
                    desc_str = (
                        f" - (Training) {self.cur_epoch} {batch_id} "
                        f"Loss: {_loss:.4f} "
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

    def _test(self):
        if self.val_loss is None:
            self.val_loss = self._validation_step(val_batches=None)
        self.test_loss = self._validation_step(testing=True, val_batches=None)

        wandb.summary["final_valid_loss"] = self.val_loss
        wandb.summary["final_test_loss"] = self.test_loss

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
        "chrom_split": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 80,
        "patience": 80,
        "use_amp": True,
        "use_ema": True,
        "scheduler": True,
        "lr": 0.0002,
        "weight_decay": 0,
        "accumulate_grad": 1,
        "expected_dna_length": 500 * (2**13),
        "std": 0.1,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "loss_tolerance": 0.0,
        "pretrained_model": None,
        # loss cov cutoff
        "loss_cov_cutoff": 10,
        "plot_example_per_epoch": None,
    }
    dataset_class = HiCTrackDataset
    model_class = ConvTransModel

    def _model_forward_pass(self, model: torch.nn.Module, batch: dict):
        # ==========
        # X
        # ==========
        batch["bw_values"] = batch["bw_values"][:, 0, :]
        genome = self.dataset.genome
        dna_one_hot = genome.get_regions_one_hot(batch["region"])
        curr_dna_length = dna_one_hot.shape[1]
        if self.expected_dna_length:
            if self.expected_dna_length >= curr_dna_length:
                raise ValueError(
                    f"Expected DNA length {self.expected_dna_length} is longer than current DNA length {curr_dna_length}."
                )
            else:
                radius = (curr_dna_length - self.expected_dna_length) // 2
                dna_one_hot = dna_one_hot[:, radius:-radius, :].astype(np.float32)
                batch["bw_values"] = batch["bw_values"][:, radius:-radius].astype(
                    np.float32
                )

        batch["values"] = batch["values"][:, 0, :, :]
        batch["values"] = resize(
            batch["values"],
            (batch["values"].shape[0], self.image_scale, self.image_scale),
            anti_aliasing=True,
        )
        # batch["values"] = np.log(batch["values"] + 1)

        batch["dna_one_hot"] = self.gaussian_noise(dna_one_hot, self.std)
        batch["bw_values"] = self.gaussian_noise(batch["bw_values"], self.std)
        batch = reverse_comp_hic_data_batch(batch)

        dna_seq = torch.from_numpy(batch["dna_one_hot"].copy())
        feature = torch.from_numpy(batch["bw_values"].copy())
        X = torch.cat([dna_seq, feature.unsqueeze(2)], dim=2).to(self.device)

        # ==========
        # y_hic
        # ==========
        y = torch.from_numpy(batch["values"].copy()).float().to(self.device)

        # ==========
        # Forward
        # ==========
        pred_y = model(X)
        return y, pred_y
