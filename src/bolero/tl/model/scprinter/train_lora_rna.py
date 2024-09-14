from copy import deepcopy

import joblib
import numpy as np
import torch

from bolero.tl.model.scprinter.dataset import scPrinterDataset
from bolero.tl.model.scprinter.model import scFootprintBPNetLoRA
from bolero.tl.model.scprinter.train_base import scFootprintTrainerMixin
from bolero.tl.pseudobulk.rna_atac_pseudobulk import RNAVQPseudobulker

from .model import scFootprintBPNet
from .train_lora import scFootprintLoRATrainer


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
    model_class = scFootprintBPNetLoRA
    pseudobulk_class = RNAVQPseudobulker

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
        # save pseudobulker
        # dataset.name_to_pseudobulker[self.config["prefix"]].save(
        #     f"{self.savename}.pseudobulker.joblib"
        # )
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
        cell_embedding = batch[cell_embedding_key]

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
            modes=select_index,
        )
        return y_footprint, y_coverage, pred_footprint, pred_coverage

    def _setup_pretrain_model_for_lora(self):
        config_for_lora = deepcopy(self.config)

        # get example cell embedding from pseduobulk scaler
        # this file should be created during dataset setup
        scaler = joblib.load(f"{self.savename}.cell_embedding_scaler.joblib")
        example_embedding = np.array(scaler.example_embedding)
        config_for_lora["example_cell_embedding"] = example_embedding

        adj_output_model_path = self.config["output_adjusted_model"]
        if adj_output_model_path is None:
            # if not provided, use the best model from the adj_output stage
            adj_output_model_path = f"{self.savename}.adj_output.best_model.pt"
        # load output adjusted model and fix all parameters
        acc_model: scFootprintBPNet = torch.load(adj_output_model_path)
        for p in acc_model.parameters():
            p.requires_grad = False
        acc_model = acc_model.cpu()

        if not self.config["use_vq_emb"]:
            assert self.config[
                "kv_bottleneck"
            ], "kv_bottleneck must be true when not using vq emb"
        if self.config["kv_bottleneck"]:
            assert not self.config[
                "use_vq_emb"
            ], "use_vq_emb must be false when using kv bottleneck"

        _kwargs = {
            "dna_cnn_model": acc_model.dna_cnn_model,
            "hidden_layer_model": acc_model.hidden_layer_model,
            "profile_cnn_model": acc_model.profile_cnn_model,
            "dna_len": acc_model.dna_len,
            "output_len": acc_model.output_len,
            "kv_bottleneck": self.config["kv_bottleneck"],
            "num_memories": self.config["num_memories"],
            "dim_memory": self.config["dim_memory"],
            "num_memory_codebooks": self.config["num_memory_codebooks"],
        }
        config_for_lora.update(_kwargs)

        acc_model = scFootprintBPNetLoRA.create_from_config(config_for_lora)
        acc_model.cuda()
        return acc_model
