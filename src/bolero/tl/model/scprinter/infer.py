import gc
import pathlib
import shutil
import time
from collections import defaultdict
from copy import deepcopy
import warnings

from bolero.tl.model.borzoi.infer import BorzoiInferenceDataset, BorzoiSNPInferencer
import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import xarray as xr
import torch

from bolero.pp.genome import Genome
from bolero.tl.dataset.ray_dataset import RayRegionDataset
from bolero.tl.footprint.tfbs import FootPrintScoreModel
from bolero.tl.model.scprinter.attribution import BatchAttribution
from bolero.utils import try_gpu, understand_regions, validate_config
from tqdm import tqdm


class BatchInference:
    """
    Perform batch inference using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model used for inference.
    postprocess : bool, optional
        Flag indicating whether to apply post-processing to the output. Default is True.

    Returns
    -------
    dict
        A dictionary containing the input data along with the inferred results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tfbs: bool = True,
        modes=range(2, 101, 1),
    ):
        self.model = model
        self.device = next(model.parameters()).device
        if tfbs:
            self.tfbs_model = FootPrintScoreModel(
                modes=modes, device=self.device, load=True
            )
        else:
            self.tfbs_model = None

    def _get_tfbs_from_footprint(self, footprint) -> dict:
        score_dict = self.tfbs_model.get_all_scores(footprint)
        # convert to numpy
        score_dict = {k: v.cpu().numpy() for k, v in score_dict.items()}
        return score_dict

    def __call__(self, data: dict) -> dict:
        """
        Perform batch inference on the given data.

        Parameters
        ----------
        data : dict
            A dictionary containing the input data.

        Returns
        -------
        dict
            A dictionary containing the input data along with the inferred results.
        """
        one_hot = data["dna_one_hot"]
        one_hot = torch.from_numpy(one_hot).float().to(self.device)
        with torch.inference_mode():
            footprint, coverage = self.model(one_hot)
        if self.tfbs_model is not None:
            tfbs_scores = self._get_tfbs_from_footprint(footprint)
            tfbs_scores = {f"pred_footprint:{k}": v for k, v in tfbs_scores.items()}
            data.update(tfbs_scores)
        data["pred_footprint"] = footprint.cpu().numpy()
        data["pred_coverage"] = coverage.cpu().numpy()
        return data


class _BatchSlice:
    def __init__(self, output_len, keys):
        self.output_len = output_len
        self.keys = keys

    def __call__(self, data: dict):
        """
        Slice the DNA matrix to the output length.
        """
        for key in self.keys:
            try:
                mat = data.pop(key)
            except KeyError:
                continue
            radius = (mat.shape[-1] - self.output_len) // 2
            data[key] = mat[..., radius:-radius]
        return data


class TrainedLoraModel:
    """
    TrainedLoraModel class represents a trained Lora model.

    Parameters
    ----------
        model (Union[str, pathlib.Path, torch.nn.Module]): The trained Lora model.
        embedding (Union[str, pathlib.Path, pd.DataFrame]): The cell embedding data.
        pseudobulks (Union[str, pathlib.Path, Dict[str, List[str]]]): The pseudobulk data.
        embedding_scaler (Union[str, pathlib.Path, Any]): The embedding scaler.

    Attributes
    ----------
        model (torch.nn.Module): The trained Lora model.
        _cell_embedding (pd.DataFrame): The cell embedding data.
        _pseudobulk_to_cells (Dict[str, List[str]]): The pseudobulk data.
        _embedding_scaler (Any): The embedding scaler.
        pseudobulk_embedding (pd.DataFrame): The scaled pseudobulk embedding.
        pseudobulk_names (List[str]): The names of the pseudobulks.

    Methods
    -------
        _get_pseudobulk_embedding(): Get the scaled pseudobulk embedding.
        get_collapsed_model(key: str) -> torch.nn.Module: Get the collapsed model for a given key.

    """

    default_config = {
        "model": "REQUIRED",
        "embedding": "REQUIRED",
        "pseudobulks": "REQUIRED",
        "embedding_scaler": "REQUIRED",
    }

    def __init__(self, model, embedding, pseudobulks, embedding_scaler):
        if isinstance(model, (str, pathlib.Path)):
            model = torch.load(model, map_location="cpu").eval()
        print(type(model), "model loaded")
        self.model = model

        if isinstance(embedding, (str, pathlib.Path)):
            embedding = pd.read_feather(embedding)
            embedding = embedding.set_index(embedding.columns[0])
        self._cell_embedding = embedding

        if isinstance(pseudobulks, (str, pathlib.Path)):
            pseudobulks = joblib.load(pseudobulks)
        self._pseudobulk_to_cells = pseudobulks

        if isinstance(embedding_scaler, (str, pathlib.Path)):
            embedding_scaler = joblib.load(embedding_scaler)
        self._embedding_scaler = embedding_scaler

        self.pseudobulk_embedding = self._get_pseudobulk_embedding()
        self.pseudobulk_names = list(self.pseudobulk_embedding.index)

    def _get_pseudobulk_embedding(self):
        """
        Get the scaled pseudobulk embedding.

        Returns
        -------
            pd.DataFrame: The scaled pseudobulk embedding.

        """
        bulk_emb = {}
        for key, cells in self._pseudobulk_to_cells.items():
            if isinstance(cells, set):
                cells = pd.Index(cells)
            _emb = self._cell_embedding.loc[cells].mean(axis=0)
            bulk_emb[key] = _emb
        bulk_emb = pd.DataFrame(bulk_emb).T

        scale_bulk_emb = pd.DataFrame(
            self._embedding_scaler.transform(bulk_emb),
            index=bulk_emb.index,
            columns=bulk_emb.columns,
        )
        return scale_bulk_emb

    @torch.no_grad()
    def get_collapsed_model(self, key: str) -> torch.nn.Module:
        """
        Get the collapsed model for a given key.

        Parameters
        ----------
            key (str): The key for the pseudobulk.

        Returns
        -------
            torch.nn.Module: The collapsed model.

        Raises
        ------
            ValueError: If the key is not found in the pseudobulk embedding.

        """
        if key not in self.pseudobulk_embedding.index:
            raise ValueError(f"Key {key} not found in pseudobulk embedding")

        emb = torch.Tensor(self.pseudobulk_embedding.loc[key].values)
        _model = self.model.collapse(
            cell_embedding=emb, region_embedding=None, requires_grad=False
        )
        return _model


class BaseFootprintInferencer:
    """Class for getting the inference or attribution dataset for scPrinter model."""

    default_config = {
        "model": "REQUIRED",
        "genome": "REQUIRED",
        "batch_size": 64,
    }

    def __init__(
        self,
        model: object,
        genome: object,
        batch_size: int = 64,
    ) -> None:
        """
        Initialize the scPrinterInferencer.

        Parameters
        ----------
        model : object or str or pathlib.Path
            The model used for inference.
        genome : object or str
            The genome file.

        Returns
        -------
        None
        """
        if isinstance(model, (str, pathlib.Path)):
            model = torch.load(model)
        self.device = try_gpu()
        model.to(self.device)
        self.model = model
        self.dna_len = model.dna_len
        self.output_len = model.output_len

        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome
        self.batch_size = batch_size

        self.fp_attr_norm = None
        self.cov_attr_norm = None

        self._cleanup_env()

    def add_inferencer(self, dataset, tfbs=True) -> BatchInference:
        """
        Get the inferencer for the model.

        Parameters
        ----------
        postprocess : bool, optional
            Flag indicating whether to apply post-processing to the output. Default is True.

        Returns
        -------
        BatchInference
            The inferencer for the model.
        """
        fn = BatchInference
        fn_constructor_kwargs = {
            "model": self.model,
            "tfbs": tfbs,
        }
        dataset = dataset.map_batches(
            fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            num_gpus=0.2,
            batch_size=self.batch_size,
            concurrency=1,
        )
        return dataset

    def add_attributor(
        self,
        dataset,
        prefix,
        wrapper: str = "just_sum",
        num_gpus: float = 0.2,
        concurrency=1,
        score_norm=None,
        tfbs_model_type=None,
    ):
        """
        Get the attributor for analyzing the footprint.

        Parameters
        ----------
        dataset : RayRegionDataset
            The dataset to be used for attribution.
        prefix : str
            The prefix to be used for the attribution input.
        wrapper : str, optional
            The wrapper type (default is "just_sum").
        num_gpus : float, optional
            The number of GPUs to be used.
        concurrency : int, optional
            The number of concurrent processes to be used.

        Returns
        -------
        BatchAttribution
            The attributions dataset.
        """
        fn = BatchAttribution
        kwargs = {
            "model": self.model,
            "wrapper": wrapper,
            "method": "shap_hypo",
            "prefix": prefix,
            "modes": range(0, 30),
            "decay": 0.85,
            "score_norm": score_norm,
            "tfbs_model": tfbs_model_type,
        }

        dataset = dataset.map_batches(
            fn,
            fn_constructor_kwargs=kwargs,
            num_gpus=num_gpus,
            concurrency=concurrency,
            batch_size=self.batch_size,
        )
        return dataset

    def add_slice(self, dataset, keys):
        """
        Slice the dataset to the output length.

        Parameters
        ----------
        dataset : RayRegionDataset
            The dataset to be sliced.

        Returns
        -------
        RayRegionDataset
            The sliced dataset.
        """
        fn = _BatchSlice
        kwargs = {"output_len": self.output_len, "keys": keys}
        dataset = dataset.map_batches(
            fn=fn, fn_constructor_kwargs=kwargs, concurrency=1
        )
        return dataset

    def get_dataset(self, bed: str):
        """
        Get the dataset from the bed file.

        Parameters
        ----------
        bed : str
            The bed file.

        Returns
        -------
        RayRegionDataset
            The dataset.
        """
        ray_ds = RayRegionDataset(
            bed=bed,
            genome=self.genome,
            standard_length=self.dna_len,
            dna=True,
            batch_size=self.batch_size,
        )
        dataset = ray_ds.get_processed_dataset()
        return dataset

    def transform(
        self,
        bed: str,
        output_path: str = None,
        footprint_tfbs: bool = True,
        footprint_attr: bool = True,
        coverage_attr: bool = True,
        attr_tfbs: bool = True,
        _pre_run: bool = False,
        _save_columns=None,
        save_format="parquet",
        **write_parquet_kwargs,
    ):
        """
        Transform the dataset.

        Parameters
        ----------
        bed : str
            The bed file.
        inference : bool, optional
            Flag indicating whether to perform inference. Default is True.
        infer_postprocess : bool, optional
            Flag indicating whether to apply post-processing to the inference output. Default is True.
        footprint_attr : bool, optional
            Flag indicating whether to compute footprint attributions. Default is True.
        fp_attr_method : str, optional
            The attribution method for footprint. Default is "shap_hypo".
        fp_attr_modes : range, optional
            The range of modes for footprint. Default is range(0, 30).
        fp_attr_decay : float, optional
            The decay value for footprint. Default is 0.85.
        coverage_attr : bool, optional
            Flag indicating whether to compute coverage attributions. Default is True.
        cov_attr_method : str, optional
            The attribution method for coverage. Default is "shap_hypo".
        batch_size : int, optional
            The batch size. Default is 64.
        _pre_run : bool, optional
            Flag indicating whether to perform a pre-run to estimate the attribution normalization. Default is False.
        _save_columns : list, optional
            The columns to be saved in the transformed dataset. Default is None, all columns are saved.
        save_format : str, optional
            The format for saving the dataset, choose from "parquet" or "npz". Default is "parquet".
        write_parquet_kwargs : dict, optional
            Additional keyword arguments for writing the dataset to parquet.

        Returns
        -------
        xr.Dataset
            The transformed dataset.
        """
        if output_path is not None:
            output_path = pathlib.Path(output_path).absolute().resolve()

            success_flag = output_path / ".success"
            if output_path.exists():
                if success_flag.exists():
                    print(
                        f"Output path has success flag {output_path}/.success. Skipping."
                    )
                    return
                else:
                    # delete the output_path in case its incomplete
                    shutil.rmtree(output_path)

        dataset = self.get_dataset(bed)
        key_to_slice = ["dna_one_hot"]

        dataset = self.add_inferencer(dataset, tfbs=footprint_tfbs)

        if footprint_attr:
            tfbs_model_type = "attr_fp" if attr_tfbs else None
            tfbs_model_type = None if _pre_run else tfbs_model_type
            dataset = self.add_attributor(
                dataset=dataset,
                prefix="pred_footprint",
                wrapper="just_sum",
                num_gpus=0.2,
                score_norm=self.fp_attr_norm,
                tfbs_model_type=tfbs_model_type,
            )
            key_to_slice += [
                "pred_footprint:attributions",
                "pred_footprint:attributions_1d",
                "pred_footprint:attributions_1d:tfbs",
            ]

        if coverage_attr:
            tfbs_model_type = "attr_cov" if attr_tfbs else None
            tfbs_model_type = None if _pre_run else tfbs_model_type
            dataset = self.add_attributor(
                dataset=dataset,
                prefix="pred_coverage",
                wrapper="count",
                num_gpus=0.2,
                score_norm=self.cov_attr_norm,
                tfbs_model_type=tfbs_model_type,
            )
            key_to_slice += [
                "pred_coverage:attributions",
                "pred_coverage:attributions_1d",
                "pred_coverage:attributions_1d:tfbs",
            ]

        if not _pre_run:
            dataset = self.add_slice(dataset, keys=key_to_slice)
        else:
            fp_norms, cov_norms = _estimate_attr_norm(
                dataset, fp=footprint_attr, cov=coverage_attr, clip=520, batch_size=512
            )
            self.fp_attr_norm = fp_norms
            self.cov_attr_norm = cov_norms

        if _save_columns is not None:
            dataset = dataset.select_columns(_save_columns)

        if output_path is not None:
            if save_format == "parquet":
                dataset.write_parquet(output_path, **write_parquet_kwargs)
            elif save_format == "npz":
                pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

                total_data = defaultdict(list)
                for batch in dataset.iter_batches(batch_size=500):
                    for k, v in batch.items():
                        total_data[k].append(v)
                total_data = {k: np.concatenate(v) for k, v in total_data.items()}
                np.savez_compressed(output_path / "infer", **total_data)
            else:
                raise ValueError(f"Unknown save format {save_format}")
            success_flag.touch()
            return
        else:
            return dataset

    def _cleanup_env(self):
        time.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        return


def _estimate_attr_norm(dataset, fp=True, cov=True, clip=520, batch_size=512):
    fp_attrs = []
    cov_attrs = []
    for batch in dataset.iter_batches(batch_size=batch_size):
        if fp:
            _data = batch["pred_footprint:attributions_1d"][..., clip:-clip]
            fp_attrs.append(_data.ravel())
        if cov:
            _data = batch["pred_coverage:attributions_1d"][..., clip:-clip]
            cov_attrs.append(_data.ravel())

    if len(fp_attrs) > 0:
        fp_norms = np.quantile(np.concatenate(fp_attrs), (0.05, 0.5, 0.95))
    else:
        fp_norms = None

    if len(cov_attrs) > 0:
        cov_norms = np.quantile(np.concatenate(cov_attrs), (0.05, 0.5, 0.95))
    else:
        cov_norms = None
    return fp_norms, cov_norms


class scPrinterPseudobulkInferencer:
    """
    Class for performing pseudobulk inference using scPrinter model.

    Attributes
    ----------
        model_class (type): The class representing the trained Lora model.
        infer_class (type): The class representing the base footprint inferencer.
        default_config (dict): The default configuration for the inferencer.

    Methods
    -------
        get_default_config(): Get the default configuration for the inferencer.
        create_from_config(config: dict): Create an instance of the inferencer from a given configuration.
        make_config(**kwargs): Create a configuration for the inferencer with the given keyword arguments.
        __init__(config: dict): Initialize the inferencer with the given configuration.
        transform(collapse_key: str, bed: str, output_path: str = None, **kwargs): Perform pseudobulk inference and transform the dataset.

    """

    model_class: type = TrainedLoraModel
    infer_class: type = BaseFootprintInferencer
    default_config: dict = {**infer_class.default_config, **model_class.default_config}

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Get the default configuration for the inferencer.

        Returns
        -------
            dict: The default configuration for the inferencer.
        """
        return deepcopy(cls.default_config)

    @classmethod
    def create_from_config(cls, config: dict) -> "scPrinterPseudobulkInferencer":
        """
        Create an instance of the inferencer from a given configuration.

        Parameters
        ----------
            config (dict): The configuration for the inferencer.

        Returns
        -------
            scPrinterPseudobulkInferencer: An instance of the inferencer.
        """
        validate_config(config, cls.default_config)
        return cls(config)

    @classmethod
    def make_config(cls, **kwargs) -> dict:
        """
        Create a configuration for the inferencer with the given keyword arguments.

        Parameters
        ----------
            **kwargs: The keyword arguments for the inferencer configuration.

        Returns
        -------
            dict: The configuration for the inferencer.
        """
        config = deepcopy(cls.default_config)
        config.update(kwargs)
        return config

    def __init__(self, config: dict) -> None:
        """
        Initialize the inferencer with the given configuration.

        Parameters
        ----------
            config (dict): The configuration for the inferencer.
        """
        model_config = {
            k: v for k, v in config.items() if k in self.model_class.default_config
        }
        self.model = self.model_class(**model_config)
        config["model"] = self.model

        infer_config = {
            k: v for k, v in config.items() if k in self.infer_class.default_config
        }
        self.infer_config = infer_config
        return

    def transform(
        self,
        collapse_key: str,
        bed_path: str,
        output_path: str,
        footprint_tfbs: bool = True,
        footprint_attr: bool = True,
        coverage_attr: bool = True,
        attr_tfbs: bool = True,
        _chunk_size=5000,
        _save_columns=None,
        save_format="parquet",
        **write_parquet_kwargs,
    ):
        """
        Perform pseudobulk inference and transform the dataset.

        Parameters
        ----------
            collapse_key (str): The key for the collapsed model.
            bed (str): The bed file.
            output_path (str): The output path for the transformed dataset.
            footprint_tfbs (bool, optional): Flag indicating whether to compute footprint TFBS. Default is True.
            footprint_attr (bool, optional): Flag indicating whether to compute footprint attributions. Default is True.
            coverage_attr (bool, optional): Flag indicating whether to compute coverage attributions. Default is True.
            attr_tfbs (bool, optional): Flag indicating whether to compute attribution TFBS. Default is True.
            _chunk_size (int, optional): The chunk size for processing the bed file. Default is 5000.
            _save_columns (list, optional): The columns to be saved in the transformed dataset.
                Default is None, all columns are saved.
            **kwargs: Additional keyword arguments for the transformation.

        Returns
        -------
            ray.data.Dataset: The transformed dataset.
        """
        if output_path is not None:
            output_path = pathlib.Path(output_path).absolute().resolve()
            output_path.mkdir(parents=True, exist_ok=True)

            success_flag = output_path / ".success"
            if success_flag.exists():
                print(f"Output path has success flag {output_path}/.success. Skipping.")
                return

        _config = self.infer_config.copy()
        _config["model"] = self.model.get_collapsed_model(collapse_key)
        inferencer = self.infer_class(**_config)

        # if footprint_attr, coverage_attr or attr_tfbs needed, run the attribution on a sample to estimate the attr normalization
        # save the normalization value in output_path
        if any([footprint_attr, coverage_attr, attr_tfbs]):
            dataset = self._prerun_transform(
                inferencer=inferencer,
                output_path=output_path,
                bed_path=bed_path,
                footprint_attr=footprint_attr,
                coverage_attr=coverage_attr,
            )

        bed = understand_regions(bed_path, as_df=True)
        if bed.shape[0] > _chunk_size:
            for chunk_id, chunk_start in enumerate(range(0, bed.shape[0], _chunk_size)):
                _bed = bed.iloc[chunk_start : chunk_start + _chunk_size]
                _chunk_output_path = output_path / f"chunk_{chunk_id}"
                dataset = inferencer.transform(
                    _bed,
                    output_path=_chunk_output_path,
                    _pre_run=False,
                    footprint_tfbs=footprint_tfbs,
                    footprint_attr=footprint_attr,
                    coverage_attr=coverage_attr,
                    attr_tfbs=attr_tfbs,
                    _save_columns=_save_columns,
                    save_format=save_format,
                    **write_parquet_kwargs,
                )
        else:
            dataset = inferencer.transform(
                bed,
                output_path=output_path,
                _pre_run=False,
                footprint_tfbs=footprint_tfbs,
                footprint_attr=footprint_attr,
                coverage_attr=coverage_attr,
                attr_tfbs=attr_tfbs,
                _save_columns=_save_columns,
                save_format=save_format,
                **write_parquet_kwargs,
            )

        if output_path is not None:
            success_flag.touch()
        return dataset

    @staticmethod
    def _prerun_transform(
        inferencer: BaseFootprintInferencer,
        output_path: str,
        bed_path: str,
        footprint_attr: bool,
        coverage_attr: bool,
    ):
        norm_path = output_path / "attr_norm.joblib"
        if norm_path.exists():
            norm_dict = joblib.load(norm_path)
            inferencer.fp_attr_norm = norm_dict["fp_attr_norm"]
            inferencer.cov_attr_norm = norm_dict["cov_attr_norm"]
            return

        sample_n = 1000
        print(
            f"Pre-run attribution on {sample_n} regions to estimate the attr score normalization"
        )
        _bed = pr.read_bed(bed_path, as_df=True)
        if len(_bed) < sample_n:
            sample_bed = _bed
        else:
            sample_bed = _bed.sample(sample_n)
        inferencer.transform(
            sample_bed,
            _pre_run=True,
            output_path=None,
            footprint_tfbs=False,
            footprint_attr=footprint_attr,
            coverage_attr=coverage_attr,
            attr_tfbs=False,
        )

        # save norm value
        norm_dict = {
            "fp_attr_norm": inferencer.fp_attr_norm,
            "cov_attr_norm": inferencer.cov_attr_norm,
        }
        joblib.dump(norm_dict, norm_path)
        return

    @property
    def pseudobulk_names(self):
        """Get the possible names of the pseudobulks."""
        return self.model.pseudobulk_names



class scPrinterSNPInferencer(BorzoiSNPInferencer):
    def __init__(
        self,
        genome,
        checkpoint_path,
        peak_length=512,
        attr_length=1024,
        batch_size=8,
    ):
        super().__init__(
            genome=genome,
            checkpoint_path=checkpoint_path,
            peak_length=peak_length,
            attr_length=attr_length,
            batch_size=batch_size,
        )

        
    def _snp_predict(self, model, bed, modality='atac', mode='peak', effect_mode='mean', non_dual=False):
        """Run prediction for SNP variants.
        
        Parameters
        ----------
        model : torch.nn.Module
            Collapsed model for prediction.
        bed : pd.DataFrame
            Bed file with variant information.
        modality : str, optional
            Modality to extract from model output. Default is 'atac'.
            
        Returns
        -------
        dict
            Dictionary with reference and alternate predictions.
        """
        print(f"Running {mode} prediction for {modality}, with effect mode {effect_mode}...")
        all_data = {'ref': [], 'alt': [], 'peak':[], 'rev': []}

        for i in tqdm(range(0, bed.shape[0], self.batch_size)):
            batch_bed = bed.iloc[i : i + self.batch_size]
            
            batch = self._get_snp_dna_one_hot(batch_bed)
            outputs_ref = self._forward_pass(model, batch['ref'])
            outputs_alt = self._forward_pass(model, batch['alt'])
            all_data['rev'].extend(batch['rev'])
            
            if non_dual:
                                
                # Extract only the peak regions for each batch item
                batch_peak_effects = []
                for j in range(len(batch_bed)):
                    row = batch_bed.iloc[j]
                    # Calculate relative peak positions
                    # Convert from bp to 32bp bins and account for the 512bp padding
                    relative_peak_start = max(0, (row["peak-start"] - row["Start"] + 512) // 32)
                    relative_peak_end = min(outputs_ref.shape[-1] - 2, (row["peak-end"] - row["Start"] + 512) // 32)
                    
                    # Store the tensors as they are, without trying to stack them yet
                    ref_tensor = torch.sum(outputs_ref[j, :, relative_peak_start:relative_peak_end+1], axis=-1).cpu()
                    alt_tensor = torch.sum(outputs_alt[j, :, relative_peak_start:relative_peak_end+1],axis=-1).cpu()
                    
                    all_data['ref'].append(ref_tensor)
                    all_data['alt'].append(alt_tensor)
                                   
            
            else:
                if mode == 'snp':
                    # import pdb; breakpoint()
                    # Modified code for snp mode
                    for j in range(len(batch_bed)):
                        row = batch_bed.iloc[j]
                        # Calculate relative peak positions
                        # Convert from bp to 32bp bins and account for the 512bp padding
                        relative_peak_start = max(0, (row["peak-start"] - row["Start"] + 512) // 32)
                        relative_peak_end = min(outputs_ref[modality].shape[-1] - 2, (row["peak-end"] - row["Start"] + 512) // 32)
                        
                        # Store the tensors as they are, without trying to stack them yet
                        ref_tensor = torch.sum(outputs_ref[modality][j, :, relative_peak_start:relative_peak_end+1], axis=-1).cpu()
                        alt_tensor = torch.sum(outputs_alt[modality][j, :, relative_peak_start:relative_peak_end+1],axis=-1).cpu()
                        
                        all_data['ref'].append(ref_tensor)
                        all_data['alt'].append(alt_tensor)
                        
                        # Store the peak length
                        # all_data['peak_lengths'].append(ref_tensor.shape[-1])

                if mode == 'scprinter':
                        effect = torch.log2(outputs_alt['pred_coverage'] / outputs_ref['pred_coverage'])
                        all_data['peak'].append(effect.reshape(-1,1).cpu())
                        all_data['ref'].append(outputs_ref['pred_coverage'].reshape(-1,1).cpu())
                        all_data['alt'].append(outputs_alt['pred_coverage'].reshape(-1,1).cpu())
                
                elif mode == 'peak':
                    # # Peak mode code remains unchanged
                    effect = outputs_alt[modality] - outputs_ref[modality]
                    # Extract only the peak regions for each batch item
                    batch_peak_effects = []
                    for j in range(len(batch_bed)):
                        row = batch_bed.iloc[j]
                        # Calculate relative peak positions
                        # Convert from bp to 32bp bins and account for the 512bp padding
                        relative_peak_start = max(0, (row["peak-start"] - row["Start"] + 512) // 32)
                        relative_peak_end = min(effect.shape[-1] - 2, (row["peak-end"] - row["Start"] + 512) // 32)
                        
                        # Extract peak region effects
                        if effect_mode == 'mean':
                            peak_effect = torch.mean(effect[j, :, relative_peak_start:relative_peak_end+1],axis=-1)
                        elif effect_mode == 'sum':
                            peak_effect = torch.sum(effect[j, :, relative_peak_start:relative_peak_end+1],axis=-1)
                        elif effect_mode == 'log2foldchange':
                            ref_pred = torch.sum(outputs_ref[modality][j, :, relative_peak_start:relative_peak_end+1],axis=-1)
                            alt_pred = torch.sum(outputs_alt[modality][j, :, relative_peak_start:relative_peak_end+1],axis=-1)
                            peak_effect = torch.log2(alt_pred / ref_pred)
                        else:
                            raise ValueError(f"Unknown effect mode {effect_mode}")
                        # Append to batch results
                        batch_peak_effects.append(peak_effect)

                    # Stack batch results
                    if batch_peak_effects:
                        all_data['peak'].append(torch.stack(batch_peak_effects, dim=0).cpu())
                
                else:
                    raise ValueError(f"Unknown mode {mode}")

        if mode == 'snp':
            # Return the list of tensors as is, without stacking
            return all_data
        
        elif mode == 'peak' or mode == 'scprinter':
            all_data['peak'] = torch.cat(all_data['peak'], dim=0)
            all_data['ref'] = torch.cat(all_data['ref'], dim=0)
            all_data['alt'] = torch.cat(all_data['alt'], dim=0)
            return all_data
        

    def infer_snp(
        self, celltype: str, embedding: np.array, bed_path: str, progress_bar=True, mode='peak', effect_mode='mean', non_dual=False
    ) -> xr.Dataset:
        """Inference of variant effect for given embedding and bed files.
        
        Parameters
        ----------
        celltype : str
            The cell type identifier (e.g., 'ASC')
        embedding : np.array
            Embedding array for the cell type
        bed_path : str
            Path to the bed file with variant information
        progress_bar : bool, optional
            Show progress bar. Default is True.
            
        Returns
        -------
        BorzoiInferenceDataset
            Dataset with SNP effect predictions.
        """
        # Read and process the BED file
        bed = pd.read_csv(bed_path, header=0, sep="\t")
        
        # Standardize columns
        _columns = ["Chromosome", "Start", "End", "peak-start", "peak-end", "ref", "snp", "pos2start", "beta", "id", "PIP"]
        bed.columns = _columns
        
        # Process the embedding
        emb_model = self._collapse_model(embedding)
        data = self._snp_predict(emb_model, bed, mode=mode, effect_mode=effect_mode, non_dual=non_dual)
        
        del emb_model
        torch.cuda.empty_cache()
        
        if mode == 'snp':

            # Handle variable length peaks
            ref_data = data['ref']  # List of tensors with variable lengths
            alt_data = data['alt']  # List of tensors with variable lengths
            
            # Create a region index from the BED file
            region_index = np.arange(len(bed))
        
            # Build an xarray Dataset
            ds = xr.Dataset(
                {
                    "ref_effect": (["region", "effect"], np.array(ref_data)),
                    "alt_effect": (["region", "effect"], np.array(alt_data)),
                },
                coords={
                    "sample": [celltype],  # List with single cell type
                    "region": region_index
                },
                attrs={
                    "description": f"Peak effect predictions for {celltype}"
                }
            )
        
        elif mode == 'peak' or mode == 'scprinter':
            peak_effects = data['peak']
            ref_data = data['ref']  # List of tensors with variable lengths
            alt_data = data['alt']  # List of tensors with variable lengths

            # Create a region index from the BED file
            region_index = np.arange(len(bed))
            
            # Build an xarray Dataset
            ds = xr.Dataset(
                {
                    "peak_effect": (["region", "peak"], np.array(peak_effects)),
                    "ref_effect": (["region", "peak"], np.array(ref_data)),
                    "alt_effect": (["region", "peak"], np.array(alt_data)),
                },
                coords={
                    "sample": [celltype],  # List with single cell type
                    "region": region_index
                },
                attrs={
                    "description": f"Peak effect predictions for {celltype}"
                }
            )
        
        # Attach region metadata
        for col_name in bed.columns:
            col_values = bed[col_name].values
            if col_name == "Chromosome":
                col_values = col_values.astype(str)
            ds.coords[col_name] = (("region",), col_values)

        # reverse direction info
        ds.coords['direction'] = (("region",), data['rev'])

        return BorzoiInferenceDataset(ds, self.genome)