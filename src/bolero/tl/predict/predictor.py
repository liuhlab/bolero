import pathlib
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import torch

from bolero import Genome
from bolero.tl.model.borzoi.model import Borzoi
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.model.borzoi.utils import (
    BorzoiGeneQTLRegions,
    BorzoiGeneRegions,
    BorzoiRegions,
)

# from bolero.tl.model.scprinter.model import seq2PRINT, seq2PRINTLoRA
from bolero.utils import minimize_overlap_regions, understand_regions

from .callbacks import CALLBACK_NAME_TO_CLASS, MetricCallback
from .datamanager import GenericGenomeDataManager
from .dna_gen import DNASynthesisFactory
from .task_aggregate import AggregateMixin
from .utils import get_device, load_config, validate_region

_model_cls = Borzoi | BorzoiLoRA  # | seq2PRINT | seq2PRINTLoRA


def _get_finished_region_names(batch_dir) -> set:
    batch_dir = pathlib.Path(batch_dir)
    finished_region_names = []
    for path in batch_dir.glob("*.joblib.gz"):
        batch = joblib.load(path)
        region_names = batch.get("region_name", [])
        if hasattr(region_names, "tolist"):
            region_names = region_names.tolist()
        finished_region_names.extend(region_names)
    finished_region_names = set(finished_region_names)
    return finished_region_names


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
    task_aggregater = AggregateMixin()
    _embedding_only_mode: bool = False

    def __init__(self, config, model_class):
        self._embedding_only_mode = config.get("embedding_only_mode", False)

        self._config = load_config(config)
        self._train_config = load_config(self._config["train_config"])
        self.config = {
            **self._train_config,
            **self._config,
        }

        _genome = self.config["genome"]
        if isinstance(_genome, dict):
            self.genome = DNASynthesisFactory(
                genome_fastas=_genome, **self.config.get("genome_kwargs", {})
            )
        else:
            self.genome = Genome(_genome)

        self.device = get_device()

        self.model_class: _model_cls = model_class
        self._model = None

        self._dm = None

        use_regions = self._train_config["use_regions"]
        if isinstance(self.genome, DNASynthesisFactory):
            # When using DNASynthesisFactory, we expect the factor class to handle borzoi regions and gene regions
            self.borzoi_regions = None
            # However, we still need borzoi_gene_regions for gene count prediction,
            # because this class handles how to get gene mask and strand information from region_name
            self.borzoi_gene_regions = None
        else:
            self.borzoi_regions = BorzoiRegions(self.genome)
            if use_regions == "borzoi_gene":
                if self.config.get("qtl_data_path", None) is not None:
                    self.borzoi_gene_regions = BorzoiGeneQTLRegions(
                        self.genome, self.config["qtl_data_path"]
                    )
                else:
                    self.borzoi_gene_regions = BorzoiGeneRegions(self.genome)
            else:
                self.borzoi_gene_regions = None

    def _create_model(self) -> _model_cls:
        model_config = deepcopy(self._train_config)
        default_cfg = self.model_class.get_default_config()
        model_config = {k: v for k, v in model_config.items() if k in default_cfg}
        model_config = {**default_cfg, **model_config}
        model = self.model_class.create_from_config(model_config)
        return model

    def _load_ckeckpoint(self, model: _model_cls) -> _model_cls:
        checkpoint_path = self.config["checkpoint_path"]
        print("Loading checkpoint from", checkpoint_path)
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

    def get_fold_regions(
        self, test_only=True, mode="prediction"
    ) -> pd.DataFrame | tuple[pd.DataFrame]:
        """
        Get the regions for the fold during training.

        Parameters
        ----------
        test_only : bool
            If True, return only the test regions. If False, return train and valid regions as well.
        """
        if mode == "prediction":
            region_manager = self.borzoi_regions
            # minimize the overlap between regions to reduce the number of regions.
            # borzoi region are highly overlapped, and no need to run all of them in evaluation tasks
            minimize_overlap = True
        elif mode == "gene_count_prediction":
            region_manager = self.borzoi_gene_regions
            # each gene regions are corresponding to one gene, no need to minimize overlap
            minimize_overlap = False
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Supported modes are 'prediction' and 'gene_count_prediction'."
            )

        fold = self.config["fold_split_id"]
        train_regions, valid_regions, test_regions = (
            region_manager.get_train_valid_test_regions(fold)
        )
        if minimize_overlap:
            # minimize region overlap to reduce number of regions
            test_regions = minimize_overlap_regions(test_regions)
            if not test_only:
                train_regions = minimize_overlap_regions(train_regions)
                valid_regions = minimize_overlap_regions(valid_regions)
        if "Original_Name" in test_regions.columns:
            del test_regions["Original_Name"]
        if "Original_Name" in train_regions.columns:
            del train_regions["Original_Name"]
        if "Original_Name" in valid_regions.columns:
            del valid_regions["Original_Name"]
        if test_only:
            return test_regions
        else:
            return train_regions, valid_regions, test_regions

    def _valid_and_sort_regions(self, regions, standard_size=None, batch_dir=None):
        """
        Validate and sort borzoi regions used in the inference task

        Parameters
        ----------
        regions
        standard_size
        batch_dir
            If provided, will search for existing batch files and exclude region_name that already been saved.
        """
        if standard_size is not None and not isinstance(
            self.genome, DNASynthesisFactory
        ):
            # we skip standardization for DNASynthesisFactory, as it is not a single genome object
            regions = self.genome.standard_region_length(
                regions,
                length=standard_size,
                boarder_strategy="drop",
                keep_original=True,
            )
        if "Original_Name" in regions.columns:
            regions["Name"] = regions["Original_Name"]

        # keep only necessary columns, otherwise pr.PyRanges will have bugs during sorting
        try:
            regions = regions[["Chromosome", "Start", "End", "Name"]].copy()
        except KeyError as e:
            raise KeyError(
                f"Regions must have columns: Chromosome, Start, End, Name, got {regions.columns.tolist()}"
            ) from e

        if isinstance(regions, pr.PyRanges):
            regions = regions.sort()
        else:
            regions = pr.PyRanges(understand_regions(regions)).sort()

        if not isinstance(self.genome, DNASynthesisFactory):
            # we skip validation for DNASynthesisFactory, as it is not a single genome object
            validate_region(
                regions,
                self.genome.chrom_sizes.to_dict(),
            )

        if batch_dir is not None:
            finished_regions = _get_finished_region_names(batch_dir)
            print(len(finished_regions), "regions has finished in", batch_dir)
            regions = regions.df
            regions = regions[~regions.iloc[:, 3].isin(finished_regions)]
            print(regions.shape[0], "regions to compute")
            if regions.shape[0] == 0:
                return [], []
            regions = pr.PyRanges(regions)

        region_names = regions.df.iloc[:, 3].astype(str).tolist()
        regions_list = (
            regions.df["Chromosome"].astype(str)
            + ":"
            + regions.df["Start"].astype(str)
            + "-"
            + regions.df["End"].astype(str)
        ).tolist()
        return regions_list, region_names

    def _prepare_callbacks(self, callbacks: str | list[str] | list[tuple[str, dict]]):
        """
        Prepare the post inference callbacks from its name and arguments.

        callbacks: list[tuple[str, dict]]
            A list of tuples, where each tuple contains the name of the callback and its arguments.
            The callback name should be one of the keys in CALLBACK_NAME_TO_CLASS.
        """
        if isinstance(callbacks, str):
            callbacks = [callbacks]

        callback_list = []
        for name_and_kwargs in callbacks:
            if isinstance(name_and_kwargs, str):
                name_and_kwargs = (name_and_kwargs, {})
            name, kwargs = name_and_kwargs
            callback = CALLBACK_NAME_TO_CLASS[name](**kwargs)
            callback_list.append(callback)
        return callback_list

    def _get_post_callbacks(self, *args, **kwargs) -> list:
        """
        Get the post prediction callbacks to apply after inference.

        This method needs to be overridden by subclasses to provide specific callbacks.
        """
        return self._prepare_callbacks([])

    def _get_pre_callbacks(self, *args, **kwargs) -> list:
        """
        Get the pre prediction callbacks to apply before inference.

        This method needs to be overridden by subclasses to provide specific callbacks.
        """
        return self._prepare_callbacks([])

    def apply_callbacks(self, batch: dict, stage: str) -> dict:
        """
        Apply the callbacks to the batch.
        """
        if stage == "pre":
            callbacks = self._pre_callbacks
        elif stage == "post":
            callbacks = self._post_callbacks
        else:
            raise ValueError(f"Unknown stage: {stage}. Use 'pre' or 'post'.")

        try:
            idx = 0
            for callback in callbacks:
                if isinstance(callback, MetricCallback) and self._embedding_only_mode:
                    # skip all metric callbacks in embedding only mode
                    continue

                batch = callback(batch)
                idx += 1
        except Exception as e:
            print(f"Error in Callback {idx}", callback)
            self._print_batch(batch)
            raise e
        return batch

    def compute_cumulative_callbacks(self):
        """
        Compute the cumulative callbacks.
        """
        total_data = {}
        with torch.inference_mode():
            for callback in self._post_callbacks:
                if getattr(callback, "cumulative", False):
                    d = callback.compute()
                    total_data.update(d)
        return total_data

    @staticmethod
    def _print_batch(batch, prefix=""):
        """
        Print the batch.
        """
        keys = sorted(batch.keys())

        print(f"==========\n{prefix} Batch Schema:")
        for key in keys:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                print(
                    f"- {key}: {type(value)} {value.shape} {value.dtype} {value.device}"
                )
            elif isinstance(value, np.ndarray):
                print(f"- {key}: {type(value)} {value.shape} {value.dtype}")
            elif hasattr(value, "shape"):
                print(f"- {key}: {type(value)} {value.shape}")
            elif isinstance(value, list):
                print(f"- {key}: {type(value)} {len(value)}")
            else:
                print(f"- {key}: {type(value)} {value}")
        print("==========\n")
        return

    def _save_task_configs(self, task_config, output_path):
        """
        Save the task configs.
        """
        ensemble_dict = {
            "config": self.config,
            "pseudobulk_records": self.pseudobulk_manager.original_records,
            "task_config": task_config,
        }
        joblib.dump(
            ensemble_dict,
            output_path,
        )

    def _save_batch(self, batch, batch_dir, idx, save_keys):
        # save the batch to a file
        save_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if k in save_keys:
                save_batch[k] = v
        # data to save for each batch
        save_path = batch_dir / f"batch_{idx}.joblib.gz"
        if save_path.exists():
            raise FileExistsError(f"Batch file {save_path} already exists.")
        joblib.dump(save_batch, save_path)
        return save_batch
