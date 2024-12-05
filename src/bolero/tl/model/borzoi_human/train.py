import pathlib
from copy import deepcopy

import torch
import wandb
from torch.nn import functional as F

from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.model.borzoi.train import BorzoiTrainerMixin
from bolero.tl.model.borzoi_human.dataset import BorzoiDatasetOnline
from bolero.tl.model.borzoi_human.module_hic import Corigami
from bolero.tl.model.corigami.train import CorigamiTrainer


class TrainerBorzoiHumanDatasetMixin:
    """
    Mixin class for managing datasets used in the trainer.

    Attributes
    ----------
    dataset_class : type
        The class of the generic dataset.
    train_dataset : BorzoiDataset
        The training dataset.
    valid_dataset : BorzoiDataset
        The validation dataset.
    test_dataset : BorzoiDataset
        The test dataset.
    train_folds : List[str]
        The list of folds used for training.
    valid_folds : List[str]
        The list of folds used for validation.
    test_folds : List[str]
        The list of folds used for testing.
    config : dict
        The configuration dictionary.

    Methods
    -------
    _setup_dataset()
        Set up the dataset by splitting it into train, valid, and test sets.
    _get_dataset_paths(_folds)
        Get the paths of the dataset files for the given folds.
    train_dataset
        Property that returns the training dataset.
    valid_dataset
        Property that returns the validation dataset.
    test_dataset
        Property that returns the test dataset.
    _get_dataset()
        Get the dataset object for the given folds.
    get_train_dataloader(batches)
        Get the training dataloader.
    get_valid_dataloader(batches)
        Get the validation dataloader.
    get_test_dataloader(batches)
        Get the test dataloader.
    """

    dataset_class = BorzoiDatasetOnline

    def _setup_dataset(self):
        """
        Set up the dataset by splitting it into train, valid, and test sets.
        """
        # create dataset
        self.dataset: BorzoiDatasetOnline = self._get_dataset()

        # train, valid, test split by fold
        (
            self.train_folds,
            self.valid_folds,
            self.test_folds,
            self.train_regions,
            self.valid_regions,
            self.test_regions,
        ) = self.dataset.get_train_valid_test(
            self.config["fold_split_id"],
        )

        # channel_order = self.config.get("channel_order", None)
        # if channel_order is None:
        #     channel_order = [self.data_key] * self.config["out_channels"]
        # self.channel_order = channel_order
        return

    def _get_dataset(self) -> BorzoiDatasetOnline:
        """
        Get the dataset object for the given folds.

        Returns
        -------
        BorzoiDataset
            The dataset object.
        """
        dataset = self.dataset_class.create_from_config(self.config)
        return dataset

    def get_train_dataloader(self, batches: int, return_regions=True):
        """Get the training dataloader.

        Parameters
        ----------
        batches : int
            Number of batches to load.

        Returns
        -------
        torch.utils.data.DataLoader
            The training dataloader.
        """
        self.dataset.train()
        dataloader = self.dataset.get_dataloader(
            region_bed=self.train_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            return_regions=return_regions,
        )
        return dataloader

    def get_valid_dataloader(self, batches: int, return_regions=True):
        """Get the validation dataloader.

        Parameters
        ----------
        batches : int
            Number of batches to load.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            region_bed=self.valid_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            return_regions=return_regions,
        )
        return dataloader

    def get_test_dataloader(
        self, batches: int, as_torch: bool = True, return_regions=True
    ):
        """Get the test dataloader.

        Parameters
        ----------
        batches : int
            Number of batches to load.

        Returns
        -------
        torch.utils.data.DataLoader
            The test dataloader.
        """
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            region_bed=self.test_regions,
            n_batches=batches,
            concurrency=self.config["dataloader_concurrency"],
            return_regions=return_regions,
            as_torch=as_torch,
        )
        return dataloader


class BorzoiHumanTrainerMixin(TrainerBorzoiHumanDatasetMixin, BorzoiTrainerMixin):
    trainer_config = {
        "mode": "REQUIRED",
        "fold_split_id": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_group": None,
        "wandb_name": None,
        "max_epochs": 100,
        "patience": 5,
        "start_early_stop_after_epoch": 20,
        "use_amp": True,
        "use_ema": False,
        "scheduler": True,
        "lr": "REQUIRED",
        "large_lr_scale": 1,
        "optimizer": "adamw",
        "weight_decay": 1e-7,
        "global_clipnorm": 0.1,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "loss_tolerance": 0.0,
        "plot_example_per_epoch": 9,
        "accumulate_grad": 32,
        "dataloader_concurrency": 4,
        "grad_norm_collector": False,
        "save_state_every_n_epoch": None,
    }


class BorzoiHumanLoRATrainer(BorzoiHumanTrainerMixin):
    """Train LoRA model on pseudobulk single-cell ATAC or mC data."""

    trainer_config = BorzoiHumanTrainerMixin.trainer_config.copy()
    trainer_config.update(
        {
            "mode": "lora",
            "lr": 5e-5,
            "warmup_steps": 5000,
            "scheduler": True,
            "data_key": "atac",
            "channel_order": None,
        }
    )

    dataset_class = BorzoiDatasetOnline
    model_class = BorzoiLoRA

    def __init__(self, config: dict):
        self.data_key = config["data_key"]

        super().__init__(config)
        return

    def _setup_model(self):
        print("Setting up model from config")
        model = self.model_class.create_from_config(self.config)
        model.freeze_all_parameter_except_output_head()

        self.model = model
        self.model.to(self.device)
        self.model.convert_to_lora()
        print(self.model)
        self._set_total_params()
        return

    def _model_forward_pass(self, model: BorzoiLoRA, batch: dict):
        data_key = self.data_key
        dna_key = "dna_one_hot"
        embedding_key = "cell_type_embedding"

        # ==========
        # Get batch data
        # ==========
        X = batch.pop(dna_key)
        embedding = batch.get(embedding_key, None)

        y_true = batch.pop(data_key)
        if y_true.ndim == 2:
            # add the channel dimension to y_true
            y_true = y_true.unsqueeze(1)

        # ==========
        # Forward and Loss
        # ==========
        y_pred = model(X, embedding=embedding)
        # assert y_true.shape == y_pred.shape, f"Shapes aren't the same. Preds shape: {y_pred.shape}\n Targets shape: {y_true.shape}"
        loss, loss_breakdown, y_true = model.loss(y_true=y_true, y_pred=y_pred)

        y_pred = y_pred.detach()

        with torch.no_grad():
            if self.model.loss_type == "bce":
                y_pred = torch.sigmoid(y_pred)

        return y_true, y_pred, loss, loss_breakdown

    def _print_banner(self, text):
        print("=" * len(text) + "\n" + text + "\n" + "=" * len(text))
        return

    def _train_lora(self):
        self._print_banner("Training LoRA model")

        self.checkpoint = self._has_last_checkpoint()
        self._set_total_params()
        self._setup_fit()
        self._fit()
        return

    def train(self) -> None:
        """Train the Borzoi LoRA model."""
        flag = pathlib.Path(f"{self.savename}.{self.mode}.success.flag")

        if flag.exists():
            print(f"Training already finished, found flag file: {flag}.")
            return

        wandb_run = self._setup_wandb()
        if wandb_run is None:
            return

        with wandb_run:
            self.checkpoint = self._has_last_checkpoint()
            self._setup_model()
            self._setup_fit()
            self._train_lora()
            self._test()
            self._cleanup_env()
            wandb.finish()
        flag.touch()
        return


class BorzoiCorigamiHumanLoRATrainer(TrainerBorzoiHumanDatasetMixin, CorigamiTrainer):
    """Train LoRA model on pseudobulk single-cell ATAC data."""

    trainer_config = {
        "mode": "base",
        "fold_split_id": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_name": "REQUIRED",
        "wandb_group": None,
        "max_epochs": 40,
        "patience": 20,
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
        "loss_tolerance": 0.0,
        "pretrained_model": None,
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 1,
        "loss_cov_cutoff": 10,
        "plot_example_per_epoch": 9,
        "use_predicted_atac": False,
        "use_dna_embedding": True,
        "borzoi_checkpoint_path": "REQUIRED",
        "dataloader_concurrency": 4,
    }

    dataset_class = BorzoiDatasetOnline
    borzoi_model_class = BorzoiLoRA
    corigami_model_class = Corigami

    @classmethod
    def get_default_config(cls) -> dict:
        """Get default configuration combined from dataset, model and trainer."""
        dataset_config = cls.dataset_class.get_default_config()
        borzoi_model_config = cls.borzoi_model_class.get_default_config()
        corigami_model_config = cls.corigami_model_class.get_default_config()

        default_config = deepcopy(cls.trainer_config)
        for k, v in dataset_config.items():
            if k in default_config:
                print(
                    f"Warning: Overwriting key {k} value "
                    f"{default_config[k]} with dataset default value {v}."
                )
            default_config[k] = v

        for k, v in borzoi_model_config.items():
            if k in default_config:
                print(
                    f"Warning: Overwriting key {k} value "
                    f"{default_config[k]} with borzoi model default value {v}."
                )
            default_config[k] = v

        for k, v in corigami_model_config.items():
            if k in default_config:
                print(
                    f"Warning: Overwriting key {k} value "
                    f"{default_config[k]} with corigami model default value {v}."
                )
            default_config[k] = v

        return default_config

    def __init__(self, config: dict):
        super().__init__(config)
        # TrainerBorzoiHumanDatasetMixin._setup_dataset(self)

        # guess the hic data key
        cool_keys = [
            k for k, v in self.dataset.data_key_to_file_type.items() if v == "cool"
        ]
        assert len(cool_keys) == 1, f"Expected one cool key, got {cool_keys}"
        self.hic_data_key = cool_keys[0]
        self.atac_data_key = "atac"

        # is the model training in region2 mode
        self.region2 = getattr(self.dataset, "region2", None)
        return

    def _setup_model(self):
        print("Setting up model from config")
        if self.config["use_dna_embedding"]:
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

        corigami_model = self.corigami_model_class()
        corigami_model.to(self.device)
        print(corigami_model)
        self.model = corigami_model

        self._set_total_params()
        return

    def _model_forward_pass_single_region(
        self,
        model: Corigami,
        batch: dict,
        suffix: str = "",
        return_corigami_embedding: bool = False,
    ):
        data_key = self.hic_data_key + suffix
        dna_key = "dna_one_hot" + suffix
        embedding_key = "cell_type_embedding"

        # ==========
        # Get batch data
        # ==========
        X = batch.pop(dna_key)
        embedding = batch.get(embedding_key, None)
        if self.config["use_dna_embedding"]:
            atac_count, dna_embedding = self.borzoi_model.forward(
                x=X, embedding=embedding, return_dna_embedding=True, crop=False
            )
        else:
            dna_embedding = X
        if not self.config["use_predicted_atac"]:
            atac_count = batch[self.atac_data_key].unsqueeze(1)
        atac_log = torch.log(atac_count + 1)
        if torch.is_autocast_enabled():
            if dna_embedding.dtype != torch.float16:
                dna_embedding = dna_embedding.half()
            if atac_log.dtype != torch.float16:
                atac_log = atac_log.half()
        else:
            if dna_embedding.dtype != torch.float32:
                dna_embedding = dna_embedding.float()
            if atac_log.dtype != torch.float32:
                atac_log = atac_log.float()
        corigami_input = torch.cat([dna_embedding, atac_log], dim=1)
        y_true = batch.pop(data_key)

        # ==========
        # Forward and Loss
        # ==========
        if return_corigami_embedding:
            y_pred, hic_emb = model(
                x=corigami_input, return_corigami_embedding=return_corigami_embedding
            )
            assert (
                y_true.shape == y_pred.shape
            ), f"Shapes aren't the same. Preds shape: {y_pred.shape}\n Targets shape: {y_true.shape}"

            return y_true, y_pred, hic_emb
        else:
            y_pred = model(
                x=corigami_input, return_corigami_embedding=return_corigami_embedding
            )
            return y_true, y_pred

    def _calculate_region_d(self, batch):
        region = batch["region"]
        region_2 = batch["region_2"]

        d = (region_2 - region)[:, 0]  # take start - start
        # devide by hic bin resolution
        d = d // self.dataset.hic_resolution
        return d  # (bs,)

    def _model_forward_pass_paired_region(
        self, model: Corigami, batch: dict, *args, **kwargs
    ):
        region_1_y_true, region_1_y_pred, x_emb = (
            self._model_forward_pass_single_region(
                *args,
                model=model,
                batch=batch,
                suffix="",
                return_corigami_embedding=True,
                **kwargs,
            )
        )
        region_2_y_true, region_2_y_pred, x2_emb = (
            self._model_forward_pass_single_region(
                *args,
                model=model,
                batch=batch,
                suffix="_2",
                return_corigami_embedding=True,
                **kwargs,
            )
        )

        d = self._calculate_region_d(batch)
        bs = d.shape[0]
        reverse_comp = batch.get("is_reverse_comp", torch.zeros(bs, dtype=torch.bool))
        region_12_y_true = batch[self.hic_data_key + "_1+2"]

        region_12_y_pred = model.forward_from_hic_emb(
            *args, x_emb=x_emb, x2_emb=x2_emb, d=d, reverse_comp=reverse_comp, **kwargs
        )
        # each tensor is in shape (bs, image_scale, image_scale)
        return (
            region_1_y_true,
            region_2_y_true,
            region_12_y_true,
        ), (
            region_1_y_pred,
            region_2_y_pred,
            region_12_y_pred,
        )

    def _model_forward_pass(self, model: Corigami, batch: dict):
        if self.region2:
            y_true, y_pred = self._model_forward_pass_paired_region(
                model=model, batch=batch
            )
            # Compute loss
            loss = 0
            for true, pred in zip(y_true, y_pred):
                loss += F.mse_loss(pred, true)
        else:
            y_true, y_pred = self._model_forward_pass_single_region(
                model=model, batch=batch
            )
            loss = F.mse_loss(y_pred, y_true)
        return y_true, y_pred, loss
