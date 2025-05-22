from copy import deepcopy

import pandas as pd
import pyranges as pr
import torch

from bolero import Genome
from bolero.tl.model.borzoi.model import Borzoi
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.model.borzoi.utils import BorzoiRegions
from bolero.tl.model.scprinter.model import seq2PRINT, seq2PRINTLoRA
from bolero.utils import understand_regions

from .datamanager import GenericGenomeDataManager
from .utils import get_device, load_config, validate_region

_model_cls = Borzoi | BorzoiLoRA | seq2PRINT | seq2PRINTLoRA


def _autocast_context(device, use_amp=True):
    try:
        auto_cast_context = torch.autocast(
            device_type=str(device).split(":")[0],
            dtype=torch.bfloat16,
            enabled=use_amp,
        )
    except RuntimeError:
        # some GPU, such as T4 does not support bfloat16
        auto_cast_context = torch.autocast(
            device_type=str(device).split(":")[0],
            dtype=torch.float16,
            enabled=use_amp,
        )
    return auto_cast_context


class GenericPredictor:
    def __init__(self, config, model_class):
        self._config = load_config(config)
        self._train_config = load_config(self._config["train_config"])
        self.config = {
            **self._train_config,
            **self._config,
        }

        self.genome = Genome(self.config["genome"])
        self.device = get_device()

        self.model_class: _model_cls = model_class
        self._model = None

        self._dm = None

        # Both seq2PRINT and Borzoi use the same region partition for training
        self.borzoi_regions = BorzoiRegions(self.genome)

    def _create_model(self) -> _model_cls:
        model_config = deepcopy(self._train_config)
        default_cfg = self.model_class.get_default_config()
        model_config = {k: v for k, v in model_config.items() if k in default_cfg}
        model_config = {**default_cfg, **model_config}
        model = self.model_class.create_from_config(model_config)
        return model

    def _load_ckeckpoint(self, model: _model_cls) -> _model_cls:
        checkpoint_path = self.config["checkpoint_path"]
        state = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        else:
            pass

        model.load_state_dict(state, strict=True)

        model.to(self.device)
        model.eval()
        return model

    @property
    def model(self) -> _model_cls:
        """
        Get the model, loading it if it hasn't been loaded yet.
        """
        if self._model is None:
            model = self._create_model()
            model = self._load_ckeckpoint(model)
            self._model = model
        return self._model

    def _autocast_context(self):
        """
        Create an autocast context for the model.
        """
        return _autocast_context(self.device, self.config.get("use_amp", True))

    def add_datamanager(self, datamanager: GenericGenomeDataManager):
        """
        Add a datamanager to the model.
        """
        self._dm = datamanager

    @property
    def datamanager(self) -> GenericGenomeDataManager:
        """
        Get the datamanager.
        """
        if self._dm is None:
            raise ValueError(
                "Datamanager not set. Please call add_datamanager() first."
            )
        return self._dm

    @property
    def pseudobulk_manager(self):
        """
        Get the pseudobulk manager inside the datamanager.
        """
        if self._dm is None:
            raise ValueError(
                "Datamanager not set. Please call add_datamanager() first."
            )
        return self._dm.pseudobulk_manager

    def get_fold_regions(self, test_only=True) -> pd.DataFrame | tuple[pd.DataFrame]:
        """
        Get the regions for the fold during training.

        Parameters
        ----------
        test_only : bool
            If True, return only the test regions. If False, return train and valid regions as well.
        """
        fold = self.config["fold_split_id"]
        train_regions, test_regions, valid_regions = (
            self.borzoi_regions.get_train_valid_test_regions(fold)
        )
        if test_only:
            return test_regions
        else:
            return train_regions, valid_regions, test_regions

    def _valid_and_sort_regions(self, regions, return_list=True):
        if isinstance(regions, pr.PyRanges):
            regions = regions.sort()
        else:
            regions = pr.PyRanges(understand_regions(regions)).sort()

        validate_region(
            regions,
            self.genome.chrom_sizes.to_dict(),
        )
        if return_list:
            regions = regions.df["Name"].tolist()
        return regions


# Task: accomplish certain type of prediction
# Metric: for each batch in task, add in certain values OR across batches, add in certain values
