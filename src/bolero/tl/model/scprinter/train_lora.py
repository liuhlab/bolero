import pathlib
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import torch
import wandb

from bolero.tl.model.scprinter.dataset import scPrinterDataset
from bolero.tl.model.scprinter.model import scFootprintBPNet, scFootprintBPNetLoRA
from bolero.tl.model.scprinter.train_base import scFootprintTrainerMixin
from bolero.tl.pseudobulk.generator import PredefinedPseudobulkGenerator


class scFootprintLoRATrainer(scFootprintTrainerMixin):
    """Train scFootprintBPNet model on pseudobulk single-cell ATAC data."""

    trainer_config = scFootprintTrainerMixin.trainer_config.copy()

    trainer_config.update(
        {
            "mode": "lora",
            "lr": 0.0003,
            # Lora related files
            "accumulate_grad": 1,
            "pretrained_model": "REQUIRED",
            "output_adjusted_model": None,
            "cell_embedding": "REQUIRED",
            "region_embedding": None,
            "cell_coverage": "REQUIRED",
            "pseudobulk_path": "REQUIRED",
            "prefix": "REQUIRED",
            "standard_cov": 1e7,
            "standard_cell": None,
        }
    )

    dataset_class = scPrinterDataset
    model_class = scFootprintBPNetLoRA

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

    def _model_forward_pass(self, model, batch):
        prefix = self.config["prefix"]
        atac_key = f"{prefix}:bulk_data"
        dna_key = "dna_one_hot"
        cell_embedding_key = f"{prefix}:embedding_data"
        region_embedding_key = (
            "region_embedding" if self.config["region_embedding"] is not None else None
        )
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter

        # ==========
        # X
        # ==========
        X = batch[dna_key]
        cell_embedding = batch[cell_embedding_key]
        if region_embedding_key is None:
            region_embedding = None
        else:
            region_embedding = batch[region_embedding_key]

        # ==========
        # y_footprint, y_coverage
        # ==========
        if model.training:
            random_modes = np.random.permutation(self.modes)[: self.select_n_modes]
            select_index = torch.as_tensor(
                [self.modes_index.index(mode) for mode in random_modes]
            )
        else:
            random_modes = None
            select_index = None

        batch = footprinter(data=batch, modes=random_modes)
        y_footprint = batch[footprint_key]

        y_coverage = batch[atac_key].sum(dim=-1)
        y_coverage = torch.log1p(y_coverage)

        # ==========
        # Forward and Loss
        # ==========
        pred_footprint, pred_coverage = model(
            X,
            cell_embedding=cell_embedding,
            region_embedding=region_embedding,
            modes=select_index,
        )
        return y_footprint, y_coverage, pred_footprint, pred_coverage

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
                    max_epochs = min(max_epochs, self.config["max_epochs"])
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
