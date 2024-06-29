import pathlib
from copy import deepcopy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb

from bolero.pl.footprint import FootPrintExamplePlotter
from bolero.pl.utils import figure_to_array
from bolero.tl.model.generic.train import GenericTrainer
from bolero.tl.model.generic.train_helper import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
)
from bolero.tl.model.scprinter.dataset import scPrinterDataset
from bolero.tl.model.scprinter.model import scFootprintBPNet, scFootprintBPNetLoRA
from bolero.tl.pseudobulk.generator import PredefinedPseudobulkGenerator
from bolero.utils import get_fs_and_path


class scFootprintLoRATrainer(GenericTrainer):
    """Train scFootprintBPNet model on pseudobulk single-cell ATAC data."""

    trainer_config = {
        "mode": "lora",
        "chrom_split": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 100,
        "patience": 10,
        "use_amp": True,
        "use_ema": True,
        "scheduler": False,
        "lr": 0.0003,
        "weight_decay": 0.001,
        "accumulate_grad": 8,
        "train_batches": 2000,
        "val_batches": 300,
        "loss_tolerance": 0.0,
        "plot_example_per_epoch": 9,
        # Lora related files
        "pretrained_model": "REQUIRED",
        "output_adjusted_model": None,
        "adjust_output_model": None,
        "cell_embedding": "REQUIRED",
        "region_embedding": None,
        "cell_coverage": "REQUIRED",
        "pseudobulk_path": "REQUIRED",
        "prefix": "REQUIRED",
        "standard_cov": 1e7,
        "standard_cell": None,
        # region file
        "region_bed_path": "REQUIRED",
    }

    dataset_class = scPrinterDataset
    model_class = scFootprintBPNetLoRA

    def __init__(self, config):
        super().__init__(config)

        self.model: torch.nn.Module = None

        self._setup_env()
        self._setup_dataset()
        return

    def _setup_pretrain_model_for_adjust_output(self):
        pretrain_model_path = self.config["pretrained_model"]
        acc_model: scFootprintBPNet = torch.load(pretrain_model_path)

        # set all parameters to fixed, except the profile cnn's w&b
        acc_model.to(self.device)
        for p in acc_model.parameters():
            p.requires_grad = False
        acc_model.profile_cnn_model.conv_layer.weight.requires_grad = True
        acc_model.profile_cnn_model.conv_layer.bias.requires_grad = True
        acc_model.profile_cnn_model.linear.weight.requires_grad = True
        acc_model.profile_cnn_model.linear.bias.requires_grad = True
        return acc_model

    def _setup_pretrain_model_for_lora(self):
        config_for_lora = deepcopy(self.config)

        # get example cell embedding from pseduobulk scaler
        # this file should be created during dataset setup
        scaler = joblib.load(f"{self.savename}.cell_embedding_scaler.joblib")
        example_embedding = np.array(scaler.example_embedding)
        config_for_lora["example_cell_embedding"] = example_embedding
        if self.config["example_region_embedding"] is not None:
            region_emb = pd.read_feather(self.config["example_region_embedding"])
            region_emb = region_emb.set_index(region_emb.columns[0])
            config_for_lora["example_region_embedding"] = region_emb

        adj_output_model_path = self.config["output_adjusted_model"]
        if adj_output_model_path is None:
            # if not provided, use the best model from the adj_output stage
            adj_output_model_path = f"{self.savename}.adj_output.best_model.pt"
        # load output adjusted model and fix all parameters
        acc_model: scFootprintBPNet = torch.load(adj_output_model_path)
        for p in acc_model.parameters():
            p.requires_grad = False
        acc_model = acc_model.cpu()
        _kwargs = {
            "dna_cnn_model": acc_model.dna_cnn_model,
            "hidden_layer_model": acc_model.hidden_layer_model,
            "profile_cnn_model": acc_model.profile_cnn_model,
            "dna_len": acc_model.dna_len,
            "output_len": acc_model.output_len,
        }
        config_for_lora.update(_kwargs)

        acc_model = scFootprintBPNetLoRA.create_from_config(config_for_lora)
        acc_model.cuda()
        return acc_model

    def _setup_model(self):
        mode = self.mode

        if mode == "adj_output":
            self.model = self._setup_pretrain_model_for_adjust_output()
        elif mode == "lora":
            self.model = self._setup_pretrain_model_for_lora()
        else:
            raise ValueError(
                f"Incorrect mode: {mode}, should be 'adj_output' or 'lora'."
            )

        self._set_total_params()
        return

    def _setup_dataset(self):
        config = self.config

        # train, valid, test split by chromosome
        chrom_split = config["chrom_split"]
        self.train_chroms = chrom_split["train"]
        self.valid_chroms = chrom_split["valid"]
        self.test_chroms = chrom_split["test"]

        # dataset location and schema
        self.fs, self.dataset_dir = get_fs_and_path(config["dataset_path"].rstrip("/"))
        self.config["dataset"] = self.dataset_dir
        prefix = self.config["prefix"]

        # create dataset
        self.dataset: scPrinterDataset = self._get_dataset()
        self.footprinter = self.dataset.get_footprinter(prefix=prefix)

    def _get_dataset(self):
        dataset = scPrinterDataset.create_from_config(self.config)

        # setup pseudobulker params for sc dataset
        pseudobulker_params = {
            "cell_embedding": self.config["cell_embedding"],
            "cell_coverage": self.config["cell_coverage"],
            "predefined_pseudobulk_path": self.config["pseudobulk_path"],
            "standard_cov": self.config["standard_cov"],
            "standard_cell": self.config["standard_cell"],
        }
        dataset.add_pseudobulker(
            name=self.config["prefix"],
            cls=PredefinedPseudobulkGenerator,
            pseudobulker_kwargs=pseudobulker_params,
        )
        # save pseudobulker scaler and example pseudobulk embedding
        dataset.name_to_pseudobulker[self.config["prefix"]].save_scaler(
            f"{self.savename}.cell_embedding_scaler.joblib"
        )
        # save pseudobulker
        # dataset.name_to_pseudobulker[self.config["prefix"]].save(
        #     f"{self.savename}.pseudobulker.joblib"
        # )

        region_embedding_path = self.config["region_embedding"]
        if region_embedding_path is not None:
            dataset.add_region_embedding(region_embedding_path)
        return dataset

    def get_train_dataloader(self, batches):
        """Training dataloader."""
        # choose random chromosomes for training
        n_chroms = min(4, len(self.train_chroms))
        use_chrom = np.random.choice(self.train_chroms, n_chroms, replace=False)
        print(f"Using chrom {use_chrom} for training.")

        self.dataset.train()
        dataloader = self.dataset.get_dataloader(
            chroms=use_chrom,
            region_bed_path=self.config["region_bed_path"],
            n_batches=batches,
        )
        return dataloader

    def get_valid_dataloader(self, batches):
        """Validation dataset."""
        # choose random chromosomes for validation
        n_chroms = min(3, len(self.valid_chroms))
        use_chrom = np.random.choice(self.valid_chroms, n_chroms, replace=False)
        print(f"Using chrom {use_chrom} for validation.")

        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            chroms=use_chrom,
            region_bed_path=self.config["region_bed_path"],
            n_batches=batches,
        )
        return dataloader

    def get_test_dataloader(self, batches):
        """Test dataset."""
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            chroms=self.test_chroms,
            region_bed_path=self.config["region_bed_path"],
            n_batches=batches,
        )
        return dataloader

    def _setup_fit(self):
        super()._setup_fit()

        # footprints specific setup
        self.modes = np.arange(2, 101, 1)
        self.modes_index = list(self.modes)
        self.select_n_modes = 30
        return

    @torch.no_grad()
    def _model_validation_step(
        self,
        model,
        dataloader,
        val_batches,
    ):
        print_step = max(5, val_batches // 20)
        # if val batches is None, use all batches in the dataset
        mode = self.mode

        prefix = self.config["prefix"]
        atac_key = f"{prefix}:bulk_data"
        dna_key = "dna_one_hot"
        bias_key = "tn5_bias"
        cell_embedding_key = f"{prefix}:embedding_data"
        region_embedding_key = (
            "region_embedding" if self.config["region_embedding"] is not None else None
        )
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter

        size = 0
        val_loss = [0]
        profile_pearson_counter = CumulativeCounter()
        across_batch_pearson_fp = CumulativePearson()
        across_batch_pearson_cov = CumulativePearson()

        example_batches = []  # collect example batches for making images
        for batch_id, batch in enumerate(dataloader):
            # ==========
            # X
            # ==========
            X = batch[dna_key]
            if mode == "lora":
                cell_embedding = batch[cell_embedding_key]
                if region_embedding_key is None:
                    region_embedding = None
                else:
                    region_embedding = batch[region_embedding_key]
            else:
                cell_embedding = None
                region_embedding = None

            # ==========
            # y_footprint, y_coverage
            # ==========
            batch = footprinter(data=batch)
            y_footprint = batch[footprint_key]
            mask = ~torch.isnan(
                y_footprint
            )  # footprint contains nan values, remove them when calculating loss

            y_coverage = batch[atac_key].sum(dim=-1)
            y_coverage = torch.log1p(y_coverage)

            # ==========
            # Forward and Loss
            # ==========
            if mode == "lora":
                pred_score, pred_coverage = model(
                    X, cell_embedding=cell_embedding, region_embedding=region_embedding
                )
            else:
                pred_score, pred_coverage = model(X)
            pred_score_img = pred_score.clone().detach().cpu().numpy()
            y_footprint = torch.nan_to_num(y_footprint, nan=0)
            # as is in scPrinter
            # validation loss only has pred_score MSE, no coverage
            loss_ = F.mse_loss(pred_score[mask], y_footprint[mask])
            pred_score = pred_score.reshape((len(pred_score), -1))
            y_footprint = y_footprint.reshape((len(y_footprint), -1))
            val_loss[0] += loss_.item()

            # ==========
            # Within batch pearson and save for across batch pearson
            # ==========
            # within batch pearson
            corr = (
                batch_pearson_correlation(pred_score, y_footprint)
                .detach()
                .cpu()[:, None]
            )
            profile_pearson_counter.update(corr)
            # save for across batch pearson
            across_batch_pearson_fp.update(pred_score, y_footprint)
            across_batch_pearson_cov.update(pred_coverage, y_coverage)

            size += 1
            if batch_id < self.plot_example_per_epoch:
                batch["pred_score"] = pred_score_img
                example_batches.append(batch)

            if ((batch_id + 1) % print_step) == 0:
                desc_str = (
                    f" - (Validation) {self.cur_epoch} [{batch_id}/{val_batches}] "
                    f"Footprint Loss: {val_loss[0]/size:.3f}; "
                    f"Profile Pearson: {profile_pearson_counter.mean():.3f}; "
                    f"Across batch Pearson: FP {across_batch_pearson_fp.corr():.3f}; "
                    f"Cov {across_batch_pearson_cov.corr():.3f}"
                )
                print(desc_str)

        del dataloader
        self._cleanup_env()

        wandb_images = self._plot_example_footprints(
            example_batches, footprinter, atac_key, bias_key, footprint_key
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
            across_batch_pearson_fp.corr(),
            across_batch_pearson_cov.corr(),
        ]
        return val_loss, profile_pearson, across_corr, wandb_images

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

    def _validation_step(self, testing=False, val_batches=None):
        val_batches = val_batches or self.val_batches
        if testing:
            dataloader = self.get_test_dataloader(batches=val_batches)
        else:
            dataloader = self.get_valid_dataloader(batches=val_batches)

        with torch.inference_mode():
            if self.use_ema:
                self.ema.eval()
                self.ema.ema_model.eval()
                val_loss, profile_pearson, across_pearson, wandb_images = (
                    self._model_validation_step(
                        model=self.ema.ema_model,
                        dataloader=dataloader,
                        val_batches=val_batches,
                    )
                )
                self.ema.train()
                self.ema.ema_model.train()
            else:
                self.model.eval()
                val_loss, profile_pearson, across_pearson, wandb_images = (
                    self._model_validation_step(
                        model=self.model,
                        dataloader=dataloader,
                        val_batches=val_batches,
                    )
                )
                self.model.train()
        return val_loss, profile_pearson, across_pearson, wandb_images

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

    def _fit(self, max_epochs=None, valid_first=False):
        if max_epochs is None:
            max_epochs = self.max_epochs

        mode = self.mode
        # dataset related
        prefix = self.config["prefix"]
        atac_key = f"{prefix}:bulk_data"
        dna_key = "dna_one_hot"
        footprint_key = f"{prefix}:bulk_data_footprint"
        cell_embedding_key = f"{prefix}:embedding_data"
        region_embedding_key = (
            "region_embedding" if self.config["region_embedding"] is not None else None
        )

        # backpropagation related
        footprinter = self.footprinter

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
            ) = self._validation_step()
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

        stop_flag = self.early_stopping_counter >= self.patience
        if self.cur_epoch > 0:
            print(
                f"Resuming training from epoch {self.cur_epoch+1}, with {max_epochs+1} epochs in total."
            )
        while self.cur_epoch < max_epochs and not stop_flag:
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
            print("Get data loader")
            dataloader = self.get_train_dataloader(batches=self.train_batches)

            # start train epochs
            moving_avg_fp_loss = 0
            moving_avg_cov_loss = 0
            cur_cov_loss = 1e10
            cur_fp_loss = 1e10
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
                    # ==========
                    # X
                    # ==========
                    X = batch[dna_key]
                    # LoRA embedding
                    if mode == "lora":
                        cell_embedding = batch[cell_embedding_key]
                        if region_embedding_key is None:
                            region_embedding = None
                        else:
                            region_embedding = batch[region_embedding_key]
                    else:
                        cell_embedding = None
                        region_embedding = None

                    # ==========
                    # y_footprint, y_coverage
                    # ==========
                    random_modes = np.random.permutation(self.modes)[
                        : self.select_n_modes
                    ]
                    select_index = torch.as_tensor(
                        [self.modes_index.index(mode) for mode in random_modes]
                    )
                    batch = footprinter(data=batch, modes=random_modes)
                    y_footprint = batch[footprint_key]
                    mask = ~torch.isnan(
                        y_footprint
                    )  # footprint contains nan values, remove them when calculating loss

                    y_coverage = batch[atac_key].sum(dim=-1)
                    y_coverage = torch.log1p(y_coverage)

                    # ==========
                    # Forward and Loss
                    # ==========
                    if mode == "lora":
                        pred_score, pred_coverage = self.model.forward(
                            X,
                            cell_embedding=cell_embedding,
                            region_embedding=region_embedding,
                            modes=select_index,
                        )
                    else:
                        pred_score, pred_coverage = self.model.forward(
                            X,
                            modes=select_index,
                        )
                    loss_footprint = F.mse_loss(pred_score[mask], y_footprint[mask])
                    loss_coverage = F.mse_loss(y_coverage, pred_coverage)
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
                    _fp_loss = moving_avg_fp_loss / (batch_id + 1)
                    _cov_loss = moving_avg_cov_loss / (batch_id + 1)
                    desc_str = (
                        f" - (Training) {self.cur_epoch} {batch_id} "
                        f"Footprint Loss: {_fp_loss:.4f} "
                        f"Coverage Loss: {_cov_loss:.4f}"
                    )

                    if (_fp_loss > (cur_fp_loss + 0.5)) or (
                        _cov_loss > (cur_cov_loss + 0.5)
                    ):
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

    def _test(self):
        if self.val_loss is None:
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

    def _check_output_adjust_model(self):
        output_adj_model_path = self.config["output_adjusted_model"]
        if output_adj_model_path is None:
            return False
        elif pathlib.Path(output_adj_model_path).exists():
            return True
        else:
            print(f"Output adjusted model path {output_adj_model_path} does not exist.")
            return False

    def train(self, adj_output_only=False, valid_first=False) -> None:
        """Train the scFootprintTrainer model on LoRA mode."""
        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            # Fit the pretrained model on the profile CNN only with pseudobulk data
            if self._check_output_adjust_model():
                print(
                    f'Using pretrain output adjusted model at {self.config["output_adjusted_model"]}.'
                )
            else:
                if self._check_stage_flag("adj_output"):
                    print("Pretrain output exists, skipping pretrain.")
                else:
                    self.mode = "adj_output"
                    self.checkpoint = self._has_last_checkpoint()
                    self._setup_model()
                    self._setup_fit()

                    # only train for 10000 batches to adjust the output layer
                    max_epochs = int(np.ceil(10000 / self.train_batches))
                    self._fit(max_epochs=max_epochs, valid_first=valid_first)
                    self._save_stage_flag("adj_output")
                    self._cleanup_env()
                    self.config["output_adjusted_model"] = (
                        f"{self.savename}.adj_output.best_model.pt"
                    )

            self.mode = "lora"
            if not adj_output_only:
                # Fit LoRA
                self.checkpoint = self._has_last_checkpoint()
                self._setup_model()
                self._setup_fit()
                self._fit(valid_first=valid_first)
                self._test()
            self._cleanup_env()
            wandb.finish()
        return
