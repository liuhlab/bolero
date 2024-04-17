import pathlib
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pyBigWig
import scipy
import torch
from scipy.ndimage import maximum_filter

try:
    # TODO: scprinter is not publicly available currently, remove this try-except block when it is available
    import scprinter as scp
    from scprinter.seq.minimum_footprint import dispModel as _dispModel
except ImportError:
    pass

from bolero.utils import try_gpu


def get_dispmodel(device) -> torch.nn.Module:
    """Get the dispersion model."""
    model_path = scp.datasets.pretrained_dispersion_model
    disp_model = scp.utils.loadDispModel(model_path)
    disp_model = _dispModel(deepcopy(disp_model)).to(device)
    return disp_model


def zscore2pval_torch(footprint):
    """
    Convert z-scores to p-values using the torch library.

    Parameters
    ----------
    - footprint (torch.Tensor): A tensor containing z-scores.

    Returns
    -------
    - pval_log (torch.Tensor): A tensor containing the corresponding p-values in logarithmic scale.
    """
    # fill nan with 0
    footprint[torch.isnan(footprint)] = 0

    # Calculate the CDF of the normal distribution for the given footprint
    pval = torch.distributions.Normal(0, 1).cdf(footprint)

    # Clamp pval to prevent log(0) which leads to -inf. Use a very small value as the lower bound.
    eps = torch.finfo(pval.dtype).eps
    pval_clamped = torch.clamp(pval, min=eps)

    # Compute the negative log10, using the clamped values to avoid -inf
    pval_log = -torch.log10(pval_clamped)

    # Optionally, to handle values very close to 1 (which would result in a negative pval_log),
    # you can clamp the output to be non-negative. This is a design choice depending on your requirements.
    pval_log = torch.clamp(pval_log, min=0, max=10)

    return pval_log


def zscore2pval(footprint: np.ndarray) -> np.ndarray:
    """
    Convert z-scores to p-values using the scipy library.

    Parameters
    ----------
    - footprint (np.ndarray): An array containing z-scores.

    Returns
    -------
    - pval (np.ndarray): An array containing the corresponding p-values in logarithmic scale.
    """
    pval = scipy.stats.norm.cdf(footprint, 0, 1)
    pval = -np.log10(pval)
    pval[np.isnan(pval)] = 0
    return pval


def rz_conv(a: np.ndarray, n: int = 2) -> np.ndarray:
    """
    Apply convolution to the input array on the last dimension.

    Parameters
    ----------
    - a (np.ndarray): The input array.
    - n (int): The number of elements to convolve on.

    Returns
    -------
    - np.ndarray: The convolved array.
    """
    if n == 0:
        return a
    # a can be shape of (batch, sample,...,  x) and x will be the dim to be conv on
    # pad first:
    shapes = np.array(a.shape)
    shapes[-1] = n
    a = np.concatenate([np.zeros(shapes), a, np.zeros(shapes)], axis=-1)
    ret = np.cumsum(a, axis=-1)
    # ret[..., n * 2:] = ret[..., n * 2:] - ret[..., :-n * 2]
    # ret = ret[..., n * 2:]
    ret = ret[..., n * 2 :] - ret[..., : -n * 2]
    return ret


def smooth_footprint(pval_log: np.ndarray, smooth_radius: int = 5) -> np.ndarray:
    """
    Smooths the given pval_log array using a maximum filter and a convolution operation.

    Parameters
    ----------
    - pval_log (ndarray): The input array to be smoothed.
    - smooth_radius (int): The radius of the smoothing operation. Default is 5.

    Returns
    -------
    - smoothed_array (ndarray): The smoothed array.

    """
    pval_log[np.isnan(pval_log)] = 0
    pval_log[np.isinf(pval_log)] = 20

    maximum_filter_size = [0] * len(pval_log.shape)
    maximum_filter_size[-1] = 2 * smooth_radius
    pval_log = maximum_filter(pval_log, tuple(maximum_filter_size), origin=-1)
    # Changed to smoothRadius.
    pval_log = rz_conv(pval_log, smooth_radius) / (2 * smooth_radius)

    pval_log[np.isnan(pval_log)] = 0
    pval_log[np.isinf(pval_log)] = 20
    return pval_log


class FootPrintModel(_dispModel):
    """Footprint model convering the ATAC-seq data to the footprint."""

    def __init__(
        self,
        bias_bw_path: str = None,
        dispmodels: Optional[list] = None,
        modes: list[str] = None,
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
        self.forward = super().footprint

        if dispmodels is None:
            model_path = scp.datasets.pretrained_dispersion_model
            dispmodels = scp.utils.loadDispModel(model_path)
            dispmodels = deepcopy(dispmodels)
        super().__init__(dispmodels=dispmodels)

        if device is None:
            device = try_gpu()
            self.to(device)
        self.device = next(self.parameters()).device

        self.bias_bw_path = bias_bw_path
        self._bias_handle = None

        if modes is None:
            self.modes = np.arange(2, 101, 1)
        else:
            self.modes = modes

        self.atac_handles = {}

    def _calculate_footprint(
        self,
        atac,
        bias,
        modes=None,
        clip_min: int = -10,
        clip_max: int = 10,
        return_pval: bool = False,
        smooth_radius: int = None,
        numpy=False,
    ):
        """
        Calculate the footprint.

        Parameters
        ----------
        atac : torch.Tensor, np.ndarray
            A tensor containing the ATAC-seq data.
        bias : torch.Tensor, np.ndarray
            A tensor containing the bias values.
        clip_min : int, optional
            The minimum value to clip the computed footprint, by default -10.
        clip_max : int, optional
            The maximum value to clip the computed footprint, by default 10.
        return_pval : bool, optional
            Whether to return the p-value transformed footprint, the default value is zscore, by default False.
        smooth_radius : int, optional
            The radius for smoothing the footprint, by default None.
        numpy : bool, optional
            Whether to return the footprint as a numpy array, by default False.

        Returns
        -------
        torch.Tensor or np.ndarray
            A tensor or numpy array containing the computed footprint.
        """
        if isinstance(atac, torch.Tensor):
            atac = atac.float().to(self.device)
        else:
            atac = torch.as_tensor(atac, dtype=torch.float32, device=self.device)

        if isinstance(bias, torch.Tensor):
            bias = bias.float().to(self.device)
        else:
            bias = torch.as_tensor(bias, dtype=torch.float32, device=self.device)

        if modes is None:
            modes = self.modes

        # add batch dimension if necessary
        if len(atac.shape) == 1:
            atac = atac.unsqueeze(0)
        if len(bias.shape) == 1:
            bias = bias.unsqueeze(0)

        with torch.inference_mode():
            _fp = self.forward(
                atac=atac,
                bias=bias,
                modes=modes,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            if return_pval:
                _fp = zscore2pval_torch(_fp)
                if smooth_radius is not None:
                    _device = _fp.device
                    _fp = _fp.cpu().numpy()
                    _fp = smooth_footprint(_fp, smooth_radius)
                    if not numpy:
                        _fp = torch.as_tensor(_fp, device=_device)

        if numpy:
            if isinstance(_fp, torch.Tensor):
                _fp = _fp.detach().cpu().numpy()
            return _fp
        else:
            return _fp

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
            raise ValueError(
                "No bias bigWig file provided. Please set the bias_bw_path attribute."
            )
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
        assert (
            name not in self.atac_handles
        ), f"ATAC bigWig file with name {name} already exists."
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
        return values

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
        return values

    def footprint(
        self,
        chrom: str,
        start: int,
        end: int,
        name: str = None,
        modes: Optional[list[str]] = None,
        clip_min: int = -10,
        clip_max: int = 10,
        return_pval: bool = False,
        smooth_radius: int = None,
        numpy=False,
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
        return_pval : bool, optional
            Whether to return p-values along with the computed footprint.
        smooth_radius : int, optional
            The radius for smoothing the footprint.
        numpy : bool, optional
            Whether to return the output as a NumPy array instead of a torch.Tensor.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            A tensor or array containing the computed footprint.

        Notes
        -----
        This method computes the footprint for a given region on a specific chromosome. It uses the ATAC bigWig file
        and bias information to calculate the footprint. The footprint represents the signal intensity at each position
        within the region.

        If the `name` parameter is not provided, the method will use the first ATAC bigWig file available. If the `modes`
        parameter is not provided, the method will use the default modes.

        The `clip_min` and `clip_max` parameters can be used to clip the output values to a specific range.

        If `return_pval` is set to True, the method will also return p-values along with the computed footprint.

        If `smooth_radius` is provided, the method will apply smoothing to the footprint using a rolling window of the
        specified radius.

        If `numpy` is set to True, the output will be returned as a NumPy array instead of a torch.Tensor.
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
            atac=atac,
            bias=bias,
            clip_min=clip_min,
            clip_max=clip_max,
            return_pval=return_pval,
            smooth_radius=smooth_radius,
            numpy=numpy,
        )
        return _fp

    def footprint_all(
        self,
        chrom: str,
        start: int,
        end: int,
        atac_names: Optional[list[str]] = None,
        modes: Optional[list[str]] = None,
        clip_min: int = -10,
        clip_max: int = 10,
        return_pval: bool = False,
        smooth_radius: int = None,
        numpy=False,
    ) -> dict[str, torch.Tensor]:
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
        return_pval : bool, optional
            Whether to return p-values along with the footprints.
        smooth_radius : int, optional
            The radius for smoothing the footprints.
        numpy : bool, optional
            Whether to return the footprints as numpy arrays instead of torch tensors.

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
                atac=atac,
                bias=bias,
                clip_min=clip_min,
                clip_max=clip_max,
                return_pval=return_pval,
                smooth_radius=smooth_radius,
                numpy=numpy,
            )
            fp_dict[name] = _fp
        return fp_dict

    def footprint_from_data(
        self,
        atac_data: torch.Tensor,
        bias_data: torch.Tensor,
        modes: list[str] = None,
        clip_min: int = -10,
        clip_max: int = 10,
        return_pval: bool = False,
        smooth_radius: int = None,
        numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute the footprint from given ATAC-seq and bias data.

        Parameters
        ----------
        atac_data : torch.Tensor, np.ndarray
            A tensor containing the ATAC-seq data.
        bias_data : torch.Tensor, np.ndarray
            A tensor containing the bias values.
        modes : List[str], optional
            A list of modes. If not provided, the default modes will be used.
        clip_min : int, optional
            The minimum value to clip the output to.
        clip_max : int, optional
            The maximum value to clip the output to.
        return_pval : bool, optional
            Whether to return p-values along with the computed footprint.
        smooth_radius : int, optional
            The radius for smoothing the footprint.
        numpy : bool, optional
            Whether to return the result as a NumPy array.

        Returns
        -------
        torch.Tensor or np.ndarray
            A tensor or array containing the computed footprint.
        """
        _fp = self._calculate_footprint(
            atac=atac_data,
            bias=bias_data,
            clip_min=clip_min,
            clip_max=clip_max,
            return_pval=return_pval,
            smooth_radius=smooth_radius,
            numpy=numpy,
            modes=modes,
        )
        return _fp

    @staticmethod
    def postprocess_footprint(
        footprint: Union[torch.Tensor, np.ndarray], smooth_radius: int = 5
    ) -> np.ndarray:
        """
        Postprocess the computed footprint.

        Parameters
        ----------
        footprint : torch.Tensor or np.ndarray
            The computed footprint.
        smooth_radius : int, optional
            The radius for smoothing the footprint.

        Returns
        -------
        np.ndarray
            The postprocessed footprint.

        Notes
        -----
        This method takes the computed footprint and performs postprocessing steps on it. If the footprint is a torch.Tensor,
        it is converted to a numpy array and then postprocessed. The postprocessing steps include converting the z-scores to p-values,
        smoothing the footprint using a rolling window of the specified radius.

        The smoothed footprint is returned as a numpy array.
        """
        if isinstance(footprint, torch.Tensor):
            footprint = footprint.clone()
            footprint = zscore2pval_torch(footprint)
            footprint = footprint.cpu().numpy()
        else:
            footprint = zscore2pval(footprint)

        footprint = smooth_footprint(footprint, smooth_radius)
        return footprint
