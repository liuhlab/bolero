import os

import numpy as np
import torch

from bolero.tl.generic.train_helper import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
)
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.model.borzoi_human.dataset import BorzoiDatasetOnline
from bolero.tl.model.borzoi_human.module_hic import Corigami
from bolero.tl.model.borzoi_human.train import BorzoiCorigamiHumanLoRATrainer


class BorzoiCorigamiInferencer(BorzoiCorigamiHumanLoRATrainer):
    inference_config = {
        "mode": "base",
        "fold_split_id": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_name": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 50,
        "patience": 5,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": 0.0002,
        "std": 0.1,
        "weight_decay": 0,
        "accumulate_grad": 1,
        "grad_norm_collector": True,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "pretrained_model": None,
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 1,
        "loss_cov_cutoff": 10,
        "plot_example_per_epoch": 9,
        "use_predicted_atac": True,
        "use_dna_embedding": True,
        "borzoi_checkpoint_path": "REQUIRED",
        "dataloader_concurrency": 4,
    }
    dataset_class = BorzoiDatasetOnline
    borzoi_model_class = BorzoiLoRA
    corigami_model_class = Corigami

    def _setup_corigami_model(self):
        corigami_model = self.corigami_model_class()
        corigami_model.to(self.device)
        model_weights = self._load_model_weights()
        corigami_model.load_state_dict(model_weights)
        corigami_model.eval()
        self.model = corigami_model
        print(corigami_model)
        return

    def _setup_borzoi_model(self):
        borzoi_model = self.borzoi_model_class.create_from_config(self.config)
        self.borzoi_model = borzoi_model
        self.borzoi_model.convert_to_lora()

        checkpoint = torch.load(
            self.config["borzoi_checkpoint_path"], weights_only=False
        )
        model_weights = checkpoint["state_dict"]
        self.borzoi_model.load_state_dict(model_weights)
        for _, param in self.borzoi_model.named_parameters():
            param.requires_grad = False
        self.borzoi_model.to(self.device)
        print(self.borzoi_model)
        return

    def _setup_model_for_inference(self):
        self._setup_corigami_model()
        if self.config["use_predicted_atac"]:
            self._setup_borzoi_model()
        self._set_total_params()
        return

    @torch.no_grad()
    def infer(self, cell_type):
        """Inference for a specific cell type."""
        self._setup_model_for_inference()
        dataloader = self.get_test_dataloader(
            batches=self.val_batches, as_torch=False, return_regions=False
        )

        with torch.inference_mode():
            os.makedirs(self.output_dir, exist_ok=True)
            for _, batch in enumerate(dataloader):
                region = batch.pop("region")
                batch = {k: torch.tensor(v).to("cuda") for k, v in batch.items()}
                _, pred_y, _ = self._model_forward_pass(self.model, batch)
                pred_y = pred_y.cpu().detach().numpy()
                for i in range(pred_y.shape[0]):
                    region[i] = region[i].replace(":", "-")
                    filename = f"{cell_type}-{region[i]}.npy"
                    np.save(f"{self.output_dir}/{filename}", pred_y[i])

        del dataloader
        self._cleanup_env()
        return

    @torch.no_grad()
    def infer_visualize(self):
        """Generic validation step."""
        self._setup_model_for_inference()
        dataloader = self.get_test_dataloader(batches=self.val_batches, as_torch=False)

        single_batch_pearson_counter = CumulativeCounter()
        across_batch_pearson_counter = CumulativePearson()
        example_batches = []
        with torch.inference_mode():
            os.makedirs(self.output_dir, exist_ok=True)
            for _, batch in enumerate(dataloader):
                batch.pop("region")
                batch.pop("Original_Name")
                batch = {k: torch.tensor(v).to("cuda") for k, v in batch.items()}
                y, pred_y, _ = self._model_forward_pass(self.model, batch)

                # ==========
                # within batch pearson
                corr = batch_pearson_correlation(pred_y, y).detach().cpu()[:, None]
                single_batch_pearson_counter.update(corr)
                # across batch pearson
                across_batch_pearson_counter.update(pred_y, y)

                batch["values"] = y.detach()
                batch["pred_"] = pred_y.detach()
                example_batches.append(batch)

            del dataloader
            self._cleanup_env()

            self._plot_example_images(
                example_batches, target_key="values", predict_key="pred_"
            )

            single_batch_pearson = single_batch_pearson_counter.mean()
            across_batch_pearson = across_batch_pearson_counter.corr()
            score_dict = {
                "single_batch_pearson": single_batch_pearson,
                "across_batch_pearson": across_batch_pearson,
            }

            with open(f"{self.output_dir}/score.txt", "w") as f:
                for key, value in score_dict.items():
                    f.write(f"{key}: {value}\n")
        return
