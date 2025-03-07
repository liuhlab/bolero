from copy import deepcopy

import ray
import torch

from bolero.tl.model.scprinter.dataset import scPrinterDataset
from bolero.tl.model.scprinter.dataset_online import scPrinterOnlineDataset
from bolero.tl.model.scprinter.model import seq2PRINTLoRA
from bolero.tl.model.scprinter.train_base import scFootprintTrainerMixin
from bolero.tl.pseudobulk.rna_atac_pseudobulk import RNAVQPseudobulker


class scFootprintLoRATrainer(scFootprintTrainerMixin):
    """Train scFootprintBPNet model on pseudobulk single-cell ATAC data."""

    trainer_config = scFootprintTrainerMixin.trainer_config.copy()

    trainer_config.update(
        {
            "mode": "lora",
            "lr": 0.0005,
            # Lora related files
            "accumulate_grad": 4,
            "vq_records_path": "REQUIRED",
            "use_vq_emb": False,
            "prefix": "pseudobulk",
            "downsample_vq": None,
        }
    )

    dataset_class = scPrinterDataset
    model_class = seq2PRINTLoRA
    pseudobulk_class = RNAVQPseudobulker

    def _setup_model(self):
        config_for_lora = {
            k: v for k, v in self.config.items() if k in self.model_class.default_config
        }
        self.model = self.model_class.create_from_config(config_for_lora)
        self.model.cuda()
        self._set_total_params()
        return

    def _get_dataset(self):
        dataset = super()._get_dataset()
        # setup pseudobulker params for sc dataset
        pseudobulker_params = {
            "vq_records": self.config["vq_records_path"],
            "use_vq_emb": self.config["use_vq_emb"],
            "downsample_vq": self.config["downsample_vq"],
            "prefix_name": self.config["prefix"],
            "add_cov_to_emb": False,
        }
        dataset.add_pseudobulker(
            name=self.config["prefix"],
            cls=self.pseudobulk_class,
            pseudobulker_kwargs=pseudobulker_params,
        )
        return dataset

    def _model_forward_pass(self, model, batch):
        prefix = self.config["prefix"]
        atac_key = f"{prefix}:bulk_data"
        batch["true_atac"] = batch[atac_key]
        dna_key = "dna_one_hot"
        cell_embedding_key = f"{prefix}:embedding_data"
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter
        if "mc_frac" in batch:
            batch["true_mc"] = batch["mc_frac"].to(self.device)

        # ==========
        # X
        # ==========
        X = batch[dna_key]
        embedding = batch[cell_embedding_key]

        # ==========
        # y_footprint, y_coverage
        # ==========
        batch = footprinter(data=batch)
        batch["true_footprint"] = batch[footprint_key]
        atac_region_sum = batch[atac_key].sum(dim=-1)
        if atac_region_sum.ndim == 2:
            # remove the channel dim
            atac_region_sum = atac_region_sum.squeeze(1)
        batch["true_coverage"] = atac_region_sum

        # ==========
        # Forward and Loss
        # ==========
        result = model(X, embedding=embedding)
        batch.update(result)

        # clip pred_mc to the same size of true_mc
        if "mc_frac" in batch:
            clip_size = (self.dataset.dna_window - self.dataset.signal_window) // 2
            batch["pred_mc"] = batch["pred_mc"][..., clip_size:-clip_size]

        loss_dict = model.loss(batch)
        batch.update(loss_dict)
        return batch, loss_dict["loss_total"]

    def train(self) -> None:
        """Train the scFootprintTrainer model on LoRA mode."""
        self.mode = "lora"
        super().train()


class scFootprintLoRATester(scFootprintLoRATrainer):
    trainer_config = scFootprintLoRATrainer.trainer_config.copy()
    trainer_config["checkpoint_path"] = "REQUIRED"

    def _setup_model(self):
        checkpoint = torch.load(self.config["checkpoint_path"], weights_only=False)
        if isinstance(checkpoint, dict):
            super()._setup_model()
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model = checkpoint

        self.model.eval()
        return

    @staticmethod
    def save_batches(data_batches, saveas, num_rows_per_file=10, save_keys=None):
        """Save the data batches to parquet."""
        if save_keys is not None:
            save_keys = set(save_keys)
            data_batches = [
                {k: v for k, v in batch.items() if k in save_keys}
                for batch in data_batches
            ]
        dataset = ray.data.from_items(data_batches)
        dataset.write_parquet(saveas, num_rows_per_file=num_rows_per_file)
        return

    @torch.inference_mode()
    def test(self, saveas=None, device="cuda", save_keys=None):
        """Test the Borzoi LoRA model."""
        self._setup_model()
        model = self.model.to(device)

        dataloader = self.get_test_dataloader(batches=None)
        data_batches = self._model_validation_step(
            model=model,
            dataloader=dataloader,
            val_batches=None,
            collect_data=True,
        )
        self._cleanup_env()

        if saveas is None:
            return data_batches
        else:
            self.save_batches(
                data_batches, saveas, num_rows_per_file=10, save_keys=save_keys
            )
        return


class scFootprintLoraTrainerOnline(scFootprintLoRATrainer):
    trainer_config = scFootprintTrainerMixin.trainer_config.copy()

    trainer_config.update(
        {
            "mode": "lora",
            "lr": 0.0005,
            "accumulate_grad": 4,
            "prefix": "pseudobulk",
        }
    )

    dataset_class = scPrinterOnlineDataset

    def _setup_model(self):
        config_for_lora = deepcopy(self.config)
        self.model = self.model_class.create_from_config(config_for_lora)
        self.model.cuda()
        self._set_total_params()
        return

    def _get_dataset(self):
        dataset = scFootprintTrainerMixin._get_dataset(self)
        return dataset

    def _model_forward_pass(self, model, batch):
        prefix = self.prefix
        atac_key = f"{prefix}:bulk_data"
        batch["true_atac"] = batch[atac_key]
        dna_key = "dna_one_hot"
        footprint_key = f"{prefix}:bulk_data_footprint"
        footprinter = self.footprinter
        embedding = batch["cell_type_embedding"]
        if "mc_frac" in batch:
            batch["true_mc"] = batch["mc_frac"]

        # ==========
        # X
        # ==========
        X = batch[dna_key]

        # ==========
        # y_footprint, y_coverage
        # ==========
        batch = footprinter(data=batch)
        batch["true_footprint"] = batch[footprint_key]
        batch["true_coverage"] = (
            batch[atac_key].sum(dim=-1).squeeze(1)
        )  # remove the channel dim

        # ==========
        # Forward and Loss
        # ==========
        result = model(X, embedding=embedding)
        batch.update(result)

        # clip pred_mc to the same size of true_mc
        if "mc_frac" in batch:
            clip_size = (self.dataset.dna_window - self.dataset.signal_window) // 2
            batch["pred_mc"] = batch["pred_mc"][..., clip_size:-clip_size]

        loss_dict = model.loss(batch)
        batch.update(loss_dict)
        return batch, loss_dict["loss_total"]
