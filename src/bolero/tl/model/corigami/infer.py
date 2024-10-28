import os

import numpy as np
import torch

from bolero.tl.generic.train_helper import (
    CumulativeCounter,
    CumulativePearson,
    batch_pearson_correlation,
)
from bolero.tl.model.corigami.dataset import HiCTrackDataset
from bolero.tl.model.corigami.model import ConvTransModelLora
from bolero.tl.model.corigami.train import CorigamiLoraTrainer


class CorigamiInferencer(CorigamiLoraTrainer):
    inference_config = {
        "mode": "REQUIRED",
        "chrom_split": "REQUIRED",
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
        "lr": 0.0002,
        "weight_decay": 0,
        "accumulate_grad": 1,
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
    model_class = ConvTransModelLora

    def _setup_base_model_for_inference(self):
        model_weights = self._load_model_weights()
        model = self._setup_model_from_config()
        model.load_state_dict(model_weights)
        model.eval()
        return model

    def _setup_lora_model_for_inference(self):
        model_weights = self._load_model_weights()
        model = self._setup_model_from_config()
        model.convert_to_lora(self.config)
        model.load_state_dict(model_weights)
        model.eval()
        return model

    def _setup_model_for_inference(self):
        mode = self.mode
        if mode == "base":
            self.model = self._setup_base_model_for_inference()
        elif mode == "lora_finetune":
            self.model = self._setup_lora_model_for_inference()
        else:
            raise ValueError(f"Invalid mode {mode}")
        self._set_total_params()
        return

    @torch.no_grad()
    def infer(self, cell_type):
        """Inference for a specific cell type."""
        self._setup_model_for_inference()
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            chroms=self.test_chroms,
            n_batches=self.val_batches,
            batch_size=self.batch_size,
            as_torch=False,
        )

        with torch.inference_mode():
            os.makedirs(self.output_dir, exist_ok=True)
            for _, batch in enumerate(dataloader):
                region = batch.pop("region")
                batch.pop("Original_Name")
                batch = {k: torch.tensor(v).to("cuda") for k, v in batch.items()}
                _, pred_y = self._model_forward_pass(self.model, batch)
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
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            chroms=["chr12", "chr5"],
            n_batches=self.val_batches,
            batch_size=self.batch_size,
            as_torch=False,
        )

        single_batch_pearson_counter = CumulativeCounter()
        across_batch_pearson_counter = CumulativePearson()
        example_batches = []
        with torch.inference_mode():
            os.makedirs(self.output_dir, exist_ok=True)
            for _, batch in enumerate(dataloader):
                batch.pop("region")
                batch.pop("Original_Name")
                batch = {k: torch.tensor(v).to("cuda") for k, v in batch.items()}
                y, pred_y = self._model_forward_pass(self.model, batch)

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
