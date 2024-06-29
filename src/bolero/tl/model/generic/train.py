import gc
import json
import pathlib
import time
from copy import deepcopy

import numpy as np
import torch
import wandb
from torch import nn

from bolero.tl.model.generic.ema import EMA
from bolero.tl.model.generic.train_helper import (
    FakeWandb,
    check_wandb_success,
    compare_configs,
    safe_save,
    validate_config,
)
from bolero.utils import get_fs_and_path, try_gpu


class GenericDataset:
    """
    Generic dataset class.
    """

    default_config: dict = {}

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Get the default configuration for the dataset.

        Returns
        -------
            dict: The default configuration.
        """
        return deepcopy(cls.default_config)

    @classmethod
    def create_from_config(cls, config: dict):
        """
        Create a dataset instance from a configuration.

        Args:
            config (dict): The configuration.

        Returns
        -------
            GenericDataset: The dataset instance.
        """
        return cls(**config)


class GenericModel(nn.Module):
    """
    Generic model class.
    """

    default_config: dict = {}

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Get the default configuration for the model.

        Returns
        -------
            dict: The default configuration.
        """
        return deepcopy(cls.default_config)

    @classmethod
    def create_from_config(cls, config: dict):
        """
        Create a model instance from a configuration.

        Args:
            config (dict): The configuration.

        Returns
        -------
            GenericModel: The model instance.
        """
        return cls(**config)


class TrainerAttributesMixin:
    """
    Mixin class for managing trainer attributes.

    Attributes
    ----------
        savename (str): The name of the save file.
        mode (str): The mode of the trainer.

    Methods
    -------
        wandb_run_info_path(): Property that returns the path to the wandb run info file.
        best_checkpoint_path(): Property that returns the path to the best checkpoint file.
        epoch_info_path(): Property that returns the path to the epoch info file.
        best_model_path(): Property that returns the path to the best model file.

    """

    savename: str
    mode: str

    @property
    def wandb_run_info_path(self) -> pathlib.Path:
        """
        Get the path to the wandb run info file.

        Returns
        -------
            pathlib.Path: The path to the wandb run info file.

        """
        return pathlib.Path(f"{self.savename}.wandb_run_info.json")

    @property
    def best_checkpoint_path(self) -> pathlib.Path:
        """
        Get the path to the best checkpoint file.

        Returns
        -------
            pathlib.Path: The path to the best checkpoint file.

        """
        return pathlib.Path(f"{self.savename}.{self.mode}.best_checkpoint.pt")

    @property
    def epoch_info_path(self) -> pathlib.Path:
        """
        Get the path to the epoch info file.

        Returns
        -------
            pathlib.Path: The path to the epoch info file.

        """
        return pathlib.Path(f"{self.savename}.{self.mode}.epoch_info.pt")

    @property
    def best_model_path(self) -> pathlib.Path:
        """
        Get the path to the best model file.

        Returns
        -------
            pathlib.Path: The path to the best model file.

        """
        return pathlib.Path(f"{self.savename}.{self.mode}.best_model.pt")


class TrainerDatasetMixin:
    """
    Mixin class for managing datasets used in the trainer.

    Attributes
    ----------
        dataset_class (type): The class of the generic dataset.
        train_dataset (GenericDataset): The training dataset.
        valid_dataset (GenericDataset): The validation dataset.
        test_dataset (GenericDataset): The test dataset.
        train_chroms (List[str]): The list of chromosomes used for training.
        valid_chroms (List[str]): The list of chromosomes used for validation.
        test_chroms (List[str]): The list of chromosomes used for testing.
        config (dict): The configuration dictionary.

    Methods
    -------
        _setup_dataset(): Set up the dataset by splitting it into train, valid, and test sets.
        _get_dataset_paths(_chroms): Get the paths of the dataset files for the given chromosomes.
        train_dataset: Property that returns the training dataset.
        valid_dataset: Property that returns the validation dataset.
        test_dataset: Property that returns the test dataset.
        _get_dataset(chroms): Get the dataset object for the given chromosomes.

    """

    dataset_class: type
    train_dataset: GenericDataset
    valid_dataset: GenericDataset
    test_dataset: GenericDataset
    train_chroms: list[str]
    valid_chroms: list[str]
    test_chroms: list[str]
    config: dict

    def _setup_dataset(self):
        """
        Set up the dataset by splitting it into train, valid, and test sets.
        """
        config = self.config

        # train, valid, test split by chromosome
        chrom_split = config["chrom_split"]
        self.train_chroms = chrom_split["train"]
        self.valid_chroms = chrom_split["valid"]
        self.test_chroms = chrom_split["test"]

        # dataset location and schema
        self.fs, self.dataset_dir = get_fs_and_path(config["dataset_path"].rstrip("/"))
        # create dataset slots
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    def _get_dataset_paths(self, _chroms):
        """
        Get the paths of the dataset files for the given chromosomes.

        Args:
            _chroms (List[str]): The list of chromosomes.

        Returns
        -------
            List[str]: The list of dataset file paths.

        """
        # check if the file exists in gcs bucket
        dataset_paths = []
        for chrom in _chroms:
            dataset_path = self.config["dataset_path"]
            path = f"{dataset_path}/{chrom}"
            if self.fs.get_file_info(path).type:
                # type is True only if the file exists
                dataset_paths.append(path)
        return dataset_paths

    @property
    def train_dataset(self):
        """
        Training dataset.

        Returns
        -------
            GenericDataset: The training dataset.

        """
        if self._train_dataset is None:
            dataset = self._get_dataset(self.train_chroms)
            dataset.train()
            self._train_dataset = dataset
        return self._train_dataset

    @property
    def valid_dataset(self):
        """
        Validation dataset.

        Returns
        -------
            GenericDataset: The validation dataset.

        """
        if self._valid_dataset is None:
            self._valid_dataset = self._get_dataset(self.valid_chroms)
            self._valid_dataset.eval()
        return self._valid_dataset

    @property
    def test_dataset(self):
        """
        Test dataset.

        Returns
        -------
            GenericDataset: The test dataset.

        """
        if self._test_dataset is None:
            self._test_dataset = self._get_dataset(self.test_chroms)
            self._test_dataset.eval()
        return self._test_dataset

    def _get_dataset(self, chroms):
        """
        Get the dataset object for the given chromosomes.

        Args:
            chroms (List[str]): The list of chromosomes.

        Returns
        -------
            GenericDataset: The dataset object.

        """
        dataset_paths = self._get_dataset_paths(chroms)
        config = self.config.copy()
        config["dataset_path"] = dataset_paths
        validate_config(
            config, self.dataset_class.default_config, allow_extra_keys=True
        )
        dataset = self.dataset_class.create_from_config(config)
        return dataset


class GenericTrainer(TrainerAttributesMixin):
    """Generic Trainer for training models."""

    trainer_config = {
        "mode": "REQUIRED",
        "chrom_split": "REQUIRED",
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
        "plot_example_per_epoch": 9,
        "accumulate_grad": 1,
    }
    dataset_class = GenericDataset
    model_class = GenericModel

    def __init__(self, config):
        validate_config(config, self.get_default_config(), allow_extra_keys=False)
        self.config: dict = config.copy()

        # mode controls global trainer behavior in initial training or LoRA fine tuning
        self.mode: str = config["mode"]
        self.mode = self.mode.lower()

        # dataset objects
        # see TrainerDatasetMixin for more details

        # model and helper objects
        self.model: GenericModel = None
        self.total_params: int = None
        self.trainable_params: int = None
        self.device: torch.device = try_gpu()
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.scaler: torch.cuda.amp.GradScaler = None
        self.ema: EMA = None

        # epoch info
        self.checkpoint: bool = False
        self.cur_epoch: int = 0
        self.early_stopping_counter: int = 0
        self.best_val_loss: float = np.Inf

        # path and file names
        self.output_dir = pathlib.Path(config["output_dir"]).absolute().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.savename = str(self.output_dir / config["savename"])
        self.wandb_run_name: str = None

    @property
    def wandb_active(self) -> bool:
        """
        Check if Weights and Biases is active.

        Returns
        -------
            bool: True if Weights and Biases is active, False otherwise.
        """
        return wandb.run is not None

    @classmethod
    def get_default_config(cls) -> dict:
        """Get default configuration combined from dataset, model and trainer."""
        dataset_config = cls.dataset_class.get_default_config()
        model_config = cls.model_class.get_default_config()

        default_config = deepcopy(cls.trainer_config)
        for k, v in dataset_config.items():
            if k in default_config:
                print(
                    f"Warning: Overwriting key {k} value "
                    f"{default_config[k]} with dataset default value {v}."
                )
            default_config[k] = v

        for k, v in model_config.items():
            if k in default_config:
                print(
                    f"Warning: Overwriting key {k} value "
                    f"{default_config[k]} with model default value {v}."
                )
            default_config[k] = v
        return default_config

    @classmethod
    def make_config(cls, **kwargs) -> dict:
        """
        Make a configuration dictionary.

        Args:
            **kwargs: Additional keyword arguments to update the configuration.

        Returns
        -------
            dict: Configuration dictionary.
        """
        config = cls.get_default_config()
        config.update(kwargs)
        validate_config(config, cls.get_default_config(), allow_extra_keys=False)
        return config

    def _setup_wandb(self, use_wandb: bool = True):
        """
        Set up Weights and Biases for logging.

        Args:
            config (dict): Configuration dictionary.

        Returns
        -------
            Weights and Biases run context.
        """
        if not use_wandb:
            wandb_run = FakeWandb()
            return wandb_run

        config = self.config
        wandb_run_info_path = self.wandb_run_info_path

        from wandb.errors import CommError

        # load wandb run info file if exists
        if wandb_run_info_path.exists():
            with open(wandb_run_info_path) as f:
                wandb_run_info = json.load(f)

            # check if the previous run has finished successfully on W & B API
            try:
                success = check_wandb_success(wandb_run_info["path"])
            except CommError:
                success = False
            same_config = compare_configs(wandb_run_info["config"], config)
            if same_config:
                if success:
                    print(
                        f"W & B run {wandb_run_info['name']} {wandb_run_info['id']} was successful. Skipping."
                    )
                    return None
                else:
                    print(
                        f"Resuming W & B run with name: {wandb_run_info['name']} and id: {wandb_run_info['id']}."
                    )
                    try:
                        wandb_run = wandb.init(
                            id=wandb_run_info["id"],
                            project=config["wandb_project"],
                            job_type=config["wandb_job_type"],
                            entity=wandb_run_info["entity"],
                            name=wandb_run_info["name"],
                            group=wandb_run_info["group"],
                            resume="allow",
                        )
                    except CommError:
                        print(
                            "W & B run exists but cannot be resumed. Starting a new run."
                        )
                        wandb_run = wandb.init(
                            config=config,
                            project=config["wandb_project"],
                            job_type=config["wandb_job_type"],
                            group=config["wandb_group"],
                            save_code=True,
                        )
            else:
                print("W & B run exists with different config. Starting a new run.")
                wandb_run = wandb.init(
                    config=config,
                    project=config["wandb_project"],
                    job_type=config["wandb_job_type"],
                    group=config["wandb_group"],
                    save_code=True,
                )
        else:
            wandb_run = wandb.init(
                config=config,
                project=config["wandb_project"],
                job_type=config["wandb_job_type"],
                group=config["wandb_group"],
                save_code=True,
            )

        # save wandb
        wandb_run_info = {
            "id": wandb_run.id,
            "name": wandb_run.name,
            "project": wandb_run.project,
            "entity": wandb_run.entity,
            "job_type": wandb_run.job_type,
            "url": wandb_run.url,
            "path": wandb_run.path,
            "group": wandb_run.group,
            "config": dict(wandb_run.config),
        }
        with open(wandb_run_info_path, "w") as f:
            json.dump(wandb_run_info, f, indent=4)

        self.wandb_run_name = wandb.run.name
        return wandb_run

    def _has_last_checkpoint(self):
        if pathlib.Path(self.best_checkpoint_path).exists():
            return True
        return False

    def _setup_env(self):
        # setup torch environment
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        self.device = try_gpu()

        # save config to output_dir
        with open(f"{self.savename}.config.json", "w") as f:
            json.dump(dict(self.config), f, indent=4)

        self.checkpoint = self._has_last_checkpoint()
        return

    def _cleanup_env(self):
        time.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        return

    def _setup_model_from_config(self):
        # initialize model with config
        config = self.config
        validate_config(config, self.model_class.default_config, allow_extra_keys=True)
        model = self.model_class.create_from_config(config)
        model.to(self.device)
        return model

    def _set_total_params(self):
        total_params = 0
        trainable_params = 0
        for p in self.model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        self.total_params = total_params
        self.trainable_params = trainable_params
        print(
            f"Total model parameters {total_params}, trainable parameters {trainable_params}"
        )
        return

    def _update_state_dict(self):
        self._cleanup_env()

        print(
            f"Load and update state dict from checkpoint file: {self.best_checkpoint_path}"
        )
        checkpoint: dict = torch.load(self.best_checkpoint_path)
        try:
            epoch_info = torch.load(self.epoch_info_path)
            checkpoint.update(epoch_info)
        except FileNotFoundError:
            print("Epoch info not found, skipping.")

        # adjust epochs
        self.cur_epoch = checkpoint.get("epoch", 0)
        self.early_stopping_counter = checkpoint.get("early_stopping_counter", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", np.Inf)
        print(
            f"Best val loss: {self.best_val_loss:.5f}, "
            f"early stopping counter: {self.early_stopping_counter}."
        )

        # load state dict
        self.model.load_state_dict(checkpoint["state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])

        del checkpoint
        self._cleanup_env()
        return

    def _get_ema(self):
        ema = EMA(
            self.model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,
        )  # how often to actually update, to save on compute (updates every 10th .update() call)
        return ema

    def _get_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        return scaler

    def _get_optimizer(self):
        lr = self.config["lr"]
        weight_decay = self.config["weight_decay"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def _get_scheduler(self, optimizer):
        import transformers

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=3000, num_training_steps=100000
        )
        return scheduler

    def _setup_fit(self):
        config = self.config

        # epochs
        self.max_epochs = config["max_epochs"]
        self.patience = config["patience"]
        self.loss_tolerance = config["loss_tolerance"]
        self.train_batches = config["train_batches"]
        self.val_batches = config["val_batches"]
        self.early_stopping_counter = 0
        self.early_stoped = False
        self.best_val_loss = float("inf")
        self.accumulate_grad = config["accumulate_grad"]
        self.cur_epoch = 0

        # scaler
        if self.device == torch.device("cpu"):
            self.use_amp = False
        else:
            self.use_amp = self.config["use_amp"]
        self.scaler = self._get_scaler()

        # optimizer
        self.optimizer = self._get_optimizer()

        # scheduler
        if config["scheduler"]:
            self.scheduler = self._get_scheduler(self.optimizer)
        else:
            self.scheduler = None

        # EMA model
        self.use_ema = config["use_ema"]
        if self.use_ema:
            self.ema = self._get_ema()
        else:
            self.ema = None

        # plot
        self.plot_example_per_epoch = config["plot_example_per_epoch"]
        if not self.plot_example_per_epoch:
            self.plot_example_per_epoch = 0

        # update state dict if checkpoint exists
        if self.checkpoint:
            self._update_state_dict()
        return

    def _validation_step(self, *args, **kwargs):
        """Model specific validation step."""
        print("Implement model specific validation step here.")
        raise NotImplementedError

    def _save_checkpint(self, update_best: bool):
        epoch_info = {
            "epoch": self.cur_epoch,
            "early_stopping_counter": self.early_stopping_counter,
        }
        safe_save(epoch_info, self.epoch_info_path)
        if update_best:
            # check point includes model and other training states
            checkpoint = {
                "best_val_loss": self.best_val_loss,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "ema": self.ema.state_dict() if self.ema is not None else None,
            }
            safe_save(checkpoint, self.best_checkpoint_path)

            # save best model in a separate file
            if self.config["use_ema"]:
                safe_save(self.ema.ema_model, self.best_model_path)
            else:
                safe_save(self.model, self.best_model_path)
        return

    def _save_stage_flag(self, flag_name):
        import datetime

        with open(f"{self.savename}.{flag_name}.flag", "w") as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        return

    def _check_stage_flag(self, flag_name) -> bool:
        return pathlib.Path(f"{self.savename}.{flag_name}.flag").exists()

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
