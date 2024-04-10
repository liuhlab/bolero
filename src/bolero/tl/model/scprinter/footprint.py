import pathlib
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pyBigWig
import scprinter as scp
import torch
from scprinter.seq.minimum_footprint import dispModel as _dispModel

from bolero.utils import try_gpu


def get_dispmodel(device):
    """Get the dispersion model."""
    model_path = scp.datasets.pretrained_dispersion_model
    disp_model = scp.utils.loadDispModel(model_path)
    disp_model = _dispModel(deepcopy(disp_model)).to(device)
    return disp_model


class FootPrintModel(_dispModel):
    """Footprint model convering the ATAC-seq data to the footprint."""

    def __init__(
        self,
        bias_bw_path: str = None,
        dispmodels: Optional[List] = None,
        modes: List[str] = None,
        device=None,
    ):
        """
        Initialize the FootPrintModel.

        Parameters
        ----------
        bias_bw_path : str, optional
            The path to the bias bigWig file.
        dispmodels : List[DispModel], optional
            A list of DispModel objects.
        modes : List[str], optional
            A list of modes.
        device : object, optional
            The device to use for computation.

        Returns
        -------
        None
        """
        # rename the original footprint function
        self._original_footprint = super().footprint

        if dispmodels is None:
            model_path = scp.datasets.pretrained_dispersion_model
            dispmodels = scp.utils.loadDispModel(model_path)
            dispmodels = deepcopy(dispmodels)
        super().__init__(dispmodels=dispmodels)

        if device is None:
            device = try_gpu()
            self.to(device)

        self.bias_bw_path = bias_bw_path
        self._bias_handle = None

        if modes is None:
            self.modes = np.arange(2, 101, 1)
        else:
            self.modes = modes

        self.atac_handles = {}

    def _calculate_footprint(self, atac, bias, *args, **kwargs):
        """
        Calculate the footprint.

        Parameters
        ----------
        atac : torch.Tensor
            A tensor containing the ATAC-seq data.
        bias : torch.Tensor
            A tensor containing the bias values.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            A tensor containing the computed footprint.
        """
        # add batch dimension if necessary
        if len(atac.shape) == 1:
            atac = atac.unsqueeze(0)
        if len(bias.shape) == 1:
            bias = bias.unsqueeze(0)
        return self._original_footprint(atac, bias, *args, **kwargs)

    @property
    def bias_handle(self):
        """
        Return the bias bigWig file handle.

        Returns
        -------
        pyBigWig
            The bias bigWig file handle.
        """
        if self.bias_bw_path is None:
            raise ValueError("No bias bigWig file provided. Please set the bias_bw_path attribute.")
        if self._bias_handle is None:
            self._bias_handle = pyBigWig.open(self.bias_bw_path)
        return self._bias_handle

    def add_atac_bw(self, atac_bw_path: str, name=None):
        """
        Add an ATAC bigWig file to the atac_handles dictionary. If name is not provided, the name of the file will be used.

        Parameters
        ----------
        atac_bw_path : str
            The path to the ATAC bigWig file.
        name : str, optional
            The name of the ATAC bigWig file.

        Returns
        -------
        None
        """
        if name is None:
            name = pathlib.Path(str(atac_bw_path)).name
        assert name not in self.atac_handles, f"ATAC bigWig file with name {name} already exists."
        self.atac_handles[name] = pyBigWig.open(atac_bw_path)

    def close(self):
        """
        Close the bigWig files.

        Returns
        -------
        None
        """
        self._bias_handle.close()
        for handle in self.atac_handles.values():
            handle.close()

    def fetch_bias(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """
        Fetch the bias values for a given region.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        start : int
            The start position of the region.
        end : int
            The end position of the region.

        Returns
        -------
        torch.Tensor
            A tensor containing the bias values.
        """
        start, end = int(start), int(end)
        values = self.bias_handle.values(chrom, start, end, numpy=True)
        np.nan_to_num(values, copy=False)
        return torch.tensor(values).float()

    def fetch_atac(self, chrom: str, start: int, end: int, name: str) -> torch.Tensor:
        """
        Fetch the ATAC-seq data for a given region.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        start : int
            The start position of the region.
        end : int
            The end position of the region.
        name : str
            The name of the ATAC bigWig file.

        Returns
        -------
        torch.Tensor
            A tensor containing the ATAC-seq data.
        """
        start, end = int(start), int(end)
        values = self.atac_handles[name].values(chrom, start, end, numpy=True)
        np.nan_to_num(values, copy=False)
        return torch.tensor(values).float()

    def footprint(
        self,
        chrom: str,
        start: int,
        end: int,
        name: str = None,
        modes: Optional[List[str]] = None,
        clip_min: int = -10,
        clip_max: int = 10,
    ) -> torch.Tensor:
        """
        Compute the footprint.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        start : int
            The start position of the region.
        end : int
            The end position of the region.
        name : str, optional
            The name of the ATAC bigWig file. If not provided, the first ATAC bigWig file will be used.
        modes : List[str], optional
            A list of modes. If not provided, the default modes will be used.
        clip_min : int, optional
            The minimum value to clip the output to.
        clip_max : int, optional
            The maximum value to clip the output to.

        Returns
        -------
        torch.Tensor
            A tensor containing the computed footprint.
        """
        if modes is None:
            modes = self.modes
        else:
            modes = np.array(modes)

        if name is None:
            assert (
                len(self.atac_handles) == 1
            ), "Multiple ATAC bigWig files found. Please provide the name of the file."
            name = list(self.atac_handles.keys())[0]

        atac = self.fetch_atac(chrom, start, end, name)
        bias = self.fetch_bias(chrom, start, end)
        _fp = self._calculate_footprint(
            atac=atac, bias=bias, modes=modes, clip_min=clip_min, clip_max=clip_max
        )
        return _fp

    def footprint_all(
        self,
        chrom: str,
        start: int,
        end: int,
        atac_names: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        clip_min: int = -10,
        clip_max: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the footprint for all ATAC bigWig files.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        start : int
            The start position of the region.
        end : int
            The end position of the region.
        atac_names : List[str], optional
            A list of ATAC bigWig file names. If not provided, all available ATAC bigWig files will be used.
        modes : List[str], optional
            A list of modes. If not provided, the default modes will be used.
        clip_min : int, optional
            The minimum value to clip the output to.
        clip_max : int, optional
            The maximum value to clip the output to.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the computed footprints for each ATAC bigWig file.
        """
        if modes is None:
            modes = self.modes
        else:
            modes = np.array(modes)

        if atac_names is None:
            atac_names = list(self.atac_handles.keys())

        bias = self.fetch_bias(chrom, start, end)

        fp_dict = {}
        for name in atac_names:
            atac = self.fetch_atac(chrom, start, end, name)
            _fp = self._calculate_footprint(
                atac=atac, bias=bias, modes=modes, clip_min=clip_min, clip_max=clip_max
            )
            fp_dict[name] = _fp
        return fp_dict

    def footprint_from_data(self, atac_data, bias_data, modes=None, clip_min=-10, clip_max=10):
        """
        Compute the footprint from given ATAC-seq and bias data.

        Parameters
        ----------
        atac_data : torch.Tensor
            A tensor containing the ATAC-seq data.
        bias_data : torch.Tensor
            A tensor containing the bias values.
        modes : List[str], optional
            A list of modes. If not provided, the default modes will be used.
        clip_min : int, optional
            The minimum value to clip the output to.
        clip_max : int, optional
            The maximum value to clip the output to.

        Returns
        -------
        torch.Tensor
            A tensor containing the computed footprint.
        """
        if modes is None:
            modes = self.modes
        else:
            modes = np.array(modes)

        atac_data = atac_data.float()
        bias_data = bias_data.float()

        _fp = self._calculate_footprint(
            atac=atac_data,
            bias=bias_data,
            modes=modes,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        return _fp
