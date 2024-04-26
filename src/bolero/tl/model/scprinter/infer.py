import pathlib

import numpy as np
import pandas as pd
import torch
import xarray as xr

from bolero.pp.genome import Genome
from bolero.tl.dataset.ray_dataset import RayRegionDataset
from bolero.tl.footprint.footprint import postprocess_footprint
from bolero.tl.model.scprinter.attribution import BatchAttribution


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

    def __init__(self, model: torch.nn.Module, postprocess: bool = True):
        self.model = model
        self.postprocess = postprocess

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
        with torch.inference_mode():
            footprint, coverage = self.model(one_hot)
        if self.postprocess:
            footprint = postprocess_footprint(footprint=footprint, smooth_radius=5)
        data["footprint"] = footprint
        data["coverage"] = coverage.cpu().numpy()
        return data


class scPrinterInferencer:
    """Class for getting the inference or attribution dataset for scPrinter model."""

    def __init__(
        self,
        model: object,
        genome: object,
        **kwargs,
    ) -> None:
        """
        Initialize the scPrinterInferencer.

        Parameters
        ----------
        bed : str
            The bed file.
        genome : str
            The genome file.
        standard_length : int, optional
            The standard length (default is 1840).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        if isinstance(model, (str, pathlib.Path)):
            model = torch.load(model)
        self.model = model
        self.dna_len = model.dna_len
        self.output_len = model.output_len

        if isinstance(genome, str):
            genome = Genome(genome)
            # trigger loading of genome one hot zarr
            _ = genome.genome_one_hot
        self.genome = genome

    def _slice_dna_to_output_len(self, mat, as_numpy=True):
        if as_numpy:
            if isinstance(mat, torch.Tensor):
                mat = mat.cpu().numpy()
        radius = (self.dna_len - self.output_len) // 2
        return mat[..., radius:-radius]

    def get_footprint_attributor(
        self,
        wrapper: str = "just_sum",
        method: str = "shap_hypo",
        modes: range = range(0, 30),
        decay: float = 0.85,
    ):
        """
        Get the attributor for analyzing the footprint.

        Parameters
        ----------
        wrapper : str, optional
            The wrapper type (default is "just_sum").
        method : str, optional
            The attribution method (default is "shap_hypo").
        modes : range, optional
            The range of modes (default is range(0, 30)).
        decay : float, optional
            The decay value (default is 0.85).

        Returns
        -------
        Dataset
            The attributions dataset.
        """
        attributor = BatchAttribution(
            model=self.model,
            wrapper=wrapper,
            method=method,
            modes=modes,
            decay=decay,
            prefix="footprint",
        )
        return attributor

    def get_coverage_attributor(
        self,
        wrapper: str = "count",
        method: str = "shap_hypo",
    ):
        """
        Get the attributor for analyzing the coverage.
        """
        attributor = BatchAttribution(
            model=self.model, wrapper=wrapper, method=method, prefix="coverage"
        )
        return attributor

    def get_inferencer(self, postprocess=True):
        """Get the inferencer for the model."""
        inferencer = BatchInference(model=self.model, postprocess=postprocess)
        return inferencer

    def transform(
        self,
        bed: str,
        inference=True,
        infer_postprocess=True,
        footprint_attr=True,
        fp_attr_method="shap_hypo",
        fp_attr_modes=range(0, 30),
        fp_attr_decay=0.85,
        coverage_attr=True,
        cov_attr_method="shap_hypo",
        batch_size=64,
    ):
        """Transform the dataset."""
        dataset = RayRegionDataset(
            bed=bed, genome=self.genome, standard_length=self.model.dna_len
        )

        if inference:
            inferencer = self.get_inferencer(postprocess=infer_postprocess)
        else:
            inferencer = None
        if footprint_attr:
            footprint_attributor = self.get_footprint_attributor(
                wrapper="just_sum",
                method=fp_attr_method,
                modes=fp_attr_modes,
                decay=fp_attr_decay,
            )
        else:
            footprint_attributor = None
        if coverage_attr:
            coverage_attributor = self.get_coverage_attributor(
                wrapper="count", method=cov_attr_method
            )
        else:
            coverage_attributor = None

        loader = dataset.get_dataloader(batch_size=batch_size)

        batch_ds_list = []
        for batch in loader:
            batch["dna_one_hot"] = torch.from_numpy(batch["dna_one_hot"]).to("cuda")
            if inference:
                batch = inferencer(batch)
            if footprint_attr:
                batch = footprint_attributor(batch)
            if coverage_attr:
                batch = coverage_attributor(batch)

            batch_ds = self._batch_to_xarray(batch)
            batch_ds_list.append(batch_ds)
        total_ds = xr.concat(batch_ds_list, dim="region")
        return total_ds

    def _batch_to_xarray(self, batch, region_name=None):
        """Convert the batch to xarray."""
        key_to_dims = {
            "Name": ["region"],
            "Original_Name": ["region"],
            "dna_one_hot": ["region", "base", "pos"],
            "footprint": ["region", "mode", "pos"],
            "coverage": ["region"],
            "footprint_attributions": ["region", "base", "pos"],
            "footprint_attributions_1d": ["region", "pos"],
            "coverage_attributions": ["region", "base", "pos"],
            "coverage_attributions_1d": ["region", "pos"],
        }
        batch_clipped = {}
        for k, v in batch.items():
            if k == "dna_one_hot" or "attributions" in k:
                v = self._slice_dna_to_output_len(v, as_numpy=True)

            # change dtype while preventing overflow
            drange = np.finfo("float16")
            if np.issubdtype(v.dtype, np.floating):
                v = np.clip(v, drange.min, drange.max).astype(np.float16)

            batch_clipped[k] = (key_to_dims[k], v)
        ds = xr.Dataset(batch_clipped)

        regions = None
        if region_name is None:
            if "Original_Name" in batch:
                regions = pd.Index(batch["Original_Name"], name="region")
            elif "Name" in batch:
                regions = pd.Index(batch["Name"], name="region")
        if regions is not None:
            ds.coords["region"] = regions
        return ds
