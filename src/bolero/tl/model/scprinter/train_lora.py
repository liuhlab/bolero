import pathlib
from copy import deepcopy

import numpy as np
import torch
import wandb

from bolero.tl.model.scprinter.dataset import scPrinterDataset
from bolero.tl.model.scprinter.model import seq2PRINT, seq2PRINTLoRA
from bolero.tl.model.scprinter.train_base import scFootprintTrainerMixin
from bolero.tl.pseudobulk.generator import PredefinedPseudobulkGenerator
from bolero.tl.pseudobulk.rna_atac_pseudobulk import RNAVQPseudobulker


class scFootprintLoRATrainer(scFootprintTrainerMixin):
    """Train scFootprintBPNet model on pseudobulk single-cell ATAC data."""

    trainer_config = scFootprintTrainerMixin.trainer_config.copy()

    trainer_config.update(
        {
            "mode": "lora",
            "lr": 0.0001,
            # Lora related files
            "accumulate_grad": 8,
            "pretrained_model": "REQUIRED",
            "output_adjusted_model": None,
            "cell_embedding": "REQUIRED",
            "cell_coverage": "REQUIRED",
            "pseudobulk_path": "REQUIRED",
            "prefix": "REQUIRED",
            "standard_cov": 8e6,
            "standard_cell": None,
        }
    )

    dataset_class = scPrinterDataset
    model_class = seq2PRINTLoRA
    pseudobulk_class = PredefinedPseudobulkGenerator

    def _setup_pretrain_model_for_adjust_output(self):
        pretrain_model_path = self.config["pretrained_model"]
        checkpoint = torch.load(pretrain_model_path, weights_only=False)
        print(type(checkpoint))
        if isinstance(checkpoint, seq2PRINT):
            acc_model = checkpoint
        else:
            acc_model = seq2PRINT.create_from_config(self.config)
            acc_model.load_state_dict(checkpoint)

        # set all parameters to fixed, except the profile cnn's w&b
        acc_model.to(self.device)
        for p in acc_model.parameters():
            p.requires_grad = False
        for p in acc_model.footprint_head.parameters():
            p.requires_grad = True
        for p in acc_model.coverage_head.parameters():
            p.requires_grad = True
        return acc_model

    def _setup_pretrain_model_for_lora(self):
        config_for_lora = deepcopy(self.config)

        # this file should be created during dataset setup
        adj_output_model_path = self.config["output_adjusted_model"]

        if adj_output_model_path is None:
            # if not provided, use the best model from the adj_output stage
            adj_output_model_path = f"{self.savename}.adj_output.best_model.pt"

        # load output adjusted model and fix all parameters
        try:
            acc_model: seq2PRINT = torch.load(adj_output_model_path, weights_only=False)
        except FileNotFoundError:
            acc_model = self._setup_pretrain_model_for_adjust_output()

        for p in acc_model.parameters():
            p.requires_grad = False
        acc_model = acc_model.cpu()
        _kwargs = {
            "base_model": acc_model,
        }
        config_for_lora.update(_kwargs)

        acc_model = seq2PRINTLoRA.create_from_config(config_for_lora)
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

    def _get_dataset(self):
        dataset = super()._get_dataset()

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
            cls=self.pseudobulk_class,
            pseudobulker_kwargs=pseudobulker_params,
        )
        # save pseudobulker scaler and example pseudobulk embedding
        dataset.name_to_pseudobulker[self.config["prefix"]].save_scaler(
            f"{self.savename}.cell_embedding_scaler.joblib"
        )
        return dataset

    def _model_forward_pass(self, model, batch):
        prefix = self.config["prefix"]
        atac_key = f"{prefix}:bulk_data"
        dna_key = "dna_one_hot"
        cell_embedding_key = f"{prefix}:embedding_data"
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter

        # ==========
        # X
        # ==========
        X = batch[dna_key]
        embedding = batch[cell_embedding_key]

        # ==========
        # y_footprint, y_coverage
        # ==========
        batch = footprinter(data=batch)
        y_footprint = batch[footprint_key]

        y_coverage = batch[atac_key].sum(dim=-1)

        # ==========
        # Forward and Loss
        # ==========
        pred_footprint, pred_coverage = model(X, embedding=embedding)
        fp_loss, cov_loss = model.loss(
            y_footprint=y_footprint,
            y_coverage=y_coverage,
            pred_footprint=pred_footprint,
            pred_coverage=pred_coverage,
        )

        return y_footprint, y_coverage, pred_footprint, pred_coverage, fp_loss, cov_loss

    def _check_output_adjust_model(self):
        output_adj_model_path = self.config["output_adjusted_model"]
        if output_adj_model_path is None:
            return False
        elif pathlib.Path(output_adj_model_path).exists():
            return True
        else:
            print(f"Output adjusted model path {output_adj_model_path} does not exist.")
            return False

    def train(self, skip_adj_output=False) -> None:
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

                    # only train some batches to adjust the output layer
                    max_epochs = int(np.ceil(9000 / self.train_batches))
                    max_epochs = min(max_epochs, self.config["max_epochs"])
                    if not skip_adj_output:
                        self._fit(max_epochs=max_epochs)
                    self._save_stage_flag("adj_output")
                    self._cleanup_env()
                    self.config["output_adjusted_model"] = (
                        f"{self.savename}.adj_output.best_model.pt"
                    )

            self.mode = "lora"

            flag = pathlib.Path(f"{self.savename}.{self.mode}.success.flag")
            if flag.exists():
                print(f"Training already finished, found flag file: {flag}.")
                return
            # Fit LoRA
            self.checkpoint = self._has_last_checkpoint()
            self._setup_model()
            self._setup_fit()
            self._fit()
            self._test()
            self._cleanup_env()
            wandb.finish()
            flag.touch()
        return


class scFootprintLoRATrainerRNA(scFootprintLoRATrainer):
    """Train scFootprintBPNet model on pseudobulk single-cell ATAC data."""

    trainer_config = scFootprintTrainerMixin.trainer_config.copy()

    trainer_config.update(
        {
            "mode": "lora",
            "lr": 0.0003,
            # Lora related files
            "accumulate_grad": 8,
            "pretrained_model": "REQUIRED",
            "output_adjusted_model": None,
            "vq_records_path": "REQUIRED",
            "use_vq_emb": True,
            "prefix": "REQUIRED",
            "standard_cov": 8e6,
        }
    )

    dataset_class = scPrinterDataset
    model_class = seq2PRINTLoRA
    pseudobulk_class = RNAVQPseudobulker

    def __init__(self, config: dict):
        super().__init__(config)
        if self.config["kv_bottleneck"] is not None:
            assert not self.config[
                "use_vq_emb"
            ], "Cannot use both kv_bottleneck and use_vq_emb."

    def _get_dataset(self):
        dataset = scFootprintTrainerMixin._get_dataset(self)

        # setup pseudobulker params for sc dataset
        pseudobulker_params = {
            "vq_records": self.config["vq_records_path"],
            "use_vq_emb": self.config["use_vq_emb"],
            "target_cov": self.config["standard_cov"],
            "prefix_name": "pseudobulk",
        }
        dataset.add_pseudobulker(
            name=self.config["prefix"],
            cls=self.pseudobulk_class,
            pseudobulker_kwargs=pseudobulker_params,
        )
        # save pseudobulker scaler and example pseudobulk embedding
        dataset.name_to_pseudobulker[self.config["prefix"]].save_scaler(
            f"{self.savename}.cell_embedding_scaler.joblib"
        )
        return dataset
