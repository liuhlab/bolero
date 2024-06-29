import gc
import pathlib
import time

import torch

from bolero.pp.genome import Genome
from bolero.tl.dataset.ray_dataset import RayRegionDataset
from bolero.tl.footprint.tfbs import FootPrintScoreModel
from bolero.tl.model.scprinter.attribution import BatchAttribution
from bolero.utils import try_gpu


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
    def __init__(self, dna_len, output_len, keys):
        self.radius = (dna_len - output_len) // 2
        self.keys = keys

    def __call__(self, data: dict):
        """
        Slice the DNA matrix to the output length.
        """
        for key in self.keys:
            mat = data[key]
            data[key] = mat[..., self.radius : -self.radius]
        return data


class scPrinterInferencer:
    """Class for getting the inference or attribution dataset for scPrinter model."""

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
            "modes": range(0, 30),
            "decay": 0.85,
            "prefix": prefix,
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
        kwargs = {"dna_len": self.dna_len, "output_len": self.output_len, "keys": keys}
        dataset = dataset.map_batches(
            fn=fn, fn_constructor_kwargs=kwargs, concurrency=1
        )
        return dataset

    def transform(
        self,
        bed: str,
        output_path: str = None,
        footprint_tfbs: bool = True,
        footprint_attr: bool = True,
        coverage_attr: bool = True,
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

        Returns
        -------
        xr.Dataset
            The transformed dataset.
        """
        ray_ds = RayRegionDataset(
            bed=bed, genome=self.genome, standard_length=self.dna_len
        )
        dataset = ray_ds.get_processed_dataset()
        key_to_slice = ["dna_one_hot"]

        dataset = self.add_inferencer(dataset, tfbs=footprint_tfbs)

        if footprint_attr:
            dataset = self.add_attributor(
                dataset=dataset,
                prefix="pred_footprint",
                wrapper="just_sum",
                num_gpus=0.2,
            )
            key_to_slice += [
                "pred_footprint:attributions",
                "pred_footprint:attributions_1d",
            ]

        if coverage_attr:
            dataset = self.add_attributor(
                dataset=dataset,
                prefix="pred_coverage",
                wrapper="count",
                num_gpus=0.2,
            )
            key_to_slice += [
                "pred_coverage:attributions",
                "pred_coverage:attributions_1d",
            ]

        dataset = self.add_slice(dataset, keys=key_to_slice)

        if output_path is not None:
            dataset.write_parquet(output_path)
            return dataset
        else:
            return dataset

    def _cleanup_env(self):
        time.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        return
