import numpy as np

from bolero.tl.dataset.ray_gene_dataset import RayGeneDataset
from bolero.tl.generic.train import GenericTrainer


def auto_splits(n_folds=10, k=5):
    """Generate train, valid, test splits for k-fold cross validation."""
    assert n_folds // k // 2 * k * 2 == n_folds, "n_folds must be dividable by (k * 2)"
    folds = np.arange(10)

    k_folds = np.array_split(folds, k)  # evenly split folds into k arrays

    fold_splits = []
    for eval_idx in range(k):
        train = np.concatenate(
            [fp for idx, fp in enumerate(k_folds) if idx != eval_idx]
        )
        valid, test = np.array_split(
            k_folds[eval_idx], 2
        )  # evenly split folds into 2 arrays
        fold_splits.append({"train": train, "valid": valid, "test": test})
    return fold_splits


class TrainerGeneDatasetMixin:
    """
    Mixin class for managing datasets used in the trainer.

    Methods
    -------
        _setup_dataset(): Set up the dataset by splitting it into train, valid, and test sets.
        _get_dataset_paths(_chroms): Get the paths of the dataset files for the given chromosomes.
        train_dataset: Property that returns the training dataset.
        valid_dataset: Property that returns the validation dataset.
        test_dataset: Property that returns the test dataset.
        _get_dataset(chroms): Get the dataset object for the given chromosomes.
    """

    dataset_class: RayGeneDataset

    def _setup_dataset(self):
        """
        Set up the dataset by splitting it into train, valid, and test sets.
        """
        config = self.config

        # create dataset
        self.dataset: RayGeneDataset = self._get_dataset()

        # train, valid, test split by chromosome
        split_id = config["split_id"]
        fold_splits = auto_splits(
            n_folds=self.dataset.config["n_folds"], k=config["k_folds"]
        )
        fold_split = fold_splits[split_id]
        self.train_folds = fold_split["train"]
        self.valid_folds = fold_split["valid"]
        self.test_folds = fold_split["test"]

        # dataset location and schema
        # create dataset slots
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    def _get_dataset(self):
        """
        Get the dataset object for the given configuration.
        """
        dataset = self.dataset_class.create_from_config(self.config)
        return dataset

    def get_train_dataloader(self, batches, **kwargs):
        """Training dataloader."""
        self.dataset.train()
        dataloader = self.dataset.get_dataloader(
            folds=self.train_folds,
            n_batches=batches,
            **kwargs,
        )
        return dataloader

    def get_valid_dataloader(self, batches, **kwargs):
        """Validation dataset."""
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            folds=self.valid_folds,
            n_batches=batches,
            **kwargs,
        )
        return dataloader

    def get_test_dataloader(self, batches, **kwargs):
        """Test dataset."""
        self.dataset.eval()
        dataloader = self.dataset.get_dataloader(
            folds=self.test_folds,
            n_batches=batches,
            **kwargs,
        )
        return dataloader


class GenericGeneModelTrainer(TrainerGeneDatasetMixin, GenericTrainer):
    """Generic Trainer for training models."""

    trainer_config = {
        "mode": "REQUIRED",
        "k_folds": "REQUIRED",
        "split_id": "REQUIRED",
        "output_dir": "REQUIRED",
        "savename": "REQUIRED",
        "wandb_project": "REQUIRED",
        "wandb_job_type": "REQUIRED",
        "wandb_group": None,
        "max_epochs": "REQUIRED",
        "patience": "REQUIRED",
        "use_amp": True,
        "use_ema": True,
        "scheduler": False,
        "lr": "REQUIRED",
        "weight_decay": 0.001,
        "train_batches": "REQUIRED",
        "val_batches": "REQUIRED",
        "loss_tolerance": 0.0,
        "accumulate_grad": 1,
    }

    def _model_validation_step(self, *args, **kwargs):
        """Model specific validation step."""
        print("Implement model specific validation step here.")
        raise NotImplementedError

    def fit(self):
        """
        Model specific training loop.
        """
        print("Implement model specific training loop here.")
        raise NotImplementedError

    def train(self):
        """
        Model specific overall training steps.
        """
        raise NotImplementedError
