import pathlib
import shutil
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import torch
import xarray as xr
from tqdm import tqdm

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
        self.model = model
        self.dna_len = model.dna_len
        self.output_len = model.output_len

        if isinstance(genome, str):
            genome = Genome(genome)
            # trigger loading of genome one hot zarr
            _ = genome.genome_one_hot
        self.genome = genome

    def _slice_dna_to_output_len(
        self, mat: Union[torch.Tensor, np.ndarray], as_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Slice the DNA matrix to the output length.

        Parameters
        ----------
        mat : torch.Tensor or np.ndarray
            The DNA matrix.
        as_numpy : bool, optional
            Flag indicating whether to return the result as a numpy array. Default is True.

        Returns
        -------
        torch.Tensor or np.ndarray
            The sliced DNA matrix.
        """
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
    ) -> BatchAttribution:
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
        BatchAttribution
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
    ) -> BatchAttribution:
        """
        Get the attributor for analyzing the coverage.

        Parameters
        ----------
        wrapper : str, optional
            The wrapper type (default is "count").
        method : str, optional
            The attribution method (default is "shap_hypo").

        Returns
        -------
        BatchAttribution
            The attributions dataset.
        """
        attributor = BatchAttribution(
            model=self.model, wrapper=wrapper, method=method, prefix="coverage"
        )
        return attributor

    def get_inferencer(self, postprocess: bool = True) -> BatchInference:
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
        inferencer = BatchInference(model=self.model, postprocess=postprocess)
        return inferencer

    def transform(
        self,
        bed: str,
        inference: bool = True,
        infer_postprocess: bool = True,
        footprint_attr: bool = True,
        fp_attr_method: str = "shap_hypo",
        fp_attr_modes: range = range(0, 30),
        fp_attr_decay: float = 0.85,
        coverage_attr: bool = True,
        cov_attr_method: str = "shap_hypo",
        batch_size: int = 64,
    ) -> xr.Dataset:
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

    def _batch_to_xarray(
        self, batch: dict, region_name: Optional[str] = None
    ) -> xr.Dataset:
        """
        Convert the batch to xarray.

        Parameters
        ----------
        batch : dict
            The batch data.
        region_name : str, optional
            The name of the region. Default is None.

        Returns
        -------
        xr.Dataset
            The converted xarray dataset.
        """
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

    def offline_transform(self, bed_path: str, output_path: str) -> None:
        """
        Perform offline transformation.

        Parameters
        ----------
        bed_path : str
            The path to the bed file.
        output_path : str
            The output path.
        chunk_size : int, optional
            The size of each chunk. Default is 20.

        Returns
        -------
        None
        """
        bed = pr.read_bed(bed_path, as_df=True)
        output_path = pathlib.Path(output_path.rstrip("/"))
        output_path.mkdir(parents=True, exist_ok=True)
        success_flag = output_path / ".success"
        if success_flag.exists():
            print(
                f"Output path {output_path} already exists with a success flag. Skipping..."
            )
            return

        batch_size = 64
        chunk_size = int(5 * batch_size)

        temp_dir = pathlib.Path(f"{output_path}_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        chunk_starts = list(range(0, len(bed), chunk_size))
        for chunk_start in tqdm(chunk_starts, desc="Transforming regions"):
            chunk_bed = bed.iloc[chunk_start : chunk_start + chunk_size]
            chunk_out_path = temp_dir / f"chunk_{chunk_start}.joblib"

            if pathlib.Path(chunk_out_path).exists():
                continue
            chunk_ds = self.transform(chunk_bed, batch_size=batch_size)
            joblib.dump(chunk_ds, f"{chunk_out_path}.temp", compress=1)
            pathlib.Path(f"{chunk_out_path}.temp").rename(chunk_out_path)

        for chunk_start in tqdm(chunk_starts, desc="Merging chunks"):
            chunk_out_path = temp_dir / f"chunk_{chunk_start}.joblib"
            chunk_ds = joblib.load(chunk_out_path)
            chunk_ds = chunk_ds.chunk(
                {"region": chunk_size, "base": 4, "pos": self.output_len, "mode": 99}
            )
            if chunk_start == 0:
                chunk_ds.to_zarr(output_path, mode="w")
            else:
                chunk_ds.to_zarr(output_path, append_dim="region")
        success_flag.touch()
        shutil.rmtree(temp_dir)
        return
