import pathlib
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyBigWig
import scipy
import torch
from scipy.ndimage import maximum_filter

from bolero.tl.footprint.segment import get_masks, get_peaks_df_pval_fp

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


# modal trained all TF chip data
TFBS_MODEL_PATH = "/mnt/filestore/scprinter/footprint_to_TFBS_conv_model.pt"
# model trained on TFs with strong footprint (Class I) data
TFBS_MODEL_CLASS1_PATH = (
    "/mnt/filestore/scprinter/footprint_to_TFBS_class1_conv_model.pt"
)
# model trained on nucleosome data
NUCLEOSOME_MODEL_PATH = "/mnt/filestore/scprinter/footprint_to_nucleosome_conv_model.pt"


def get_footprint_to_tfbs_model(model) -> torch.nn.Module:
    """Get the footprint to TFBS model."""
    if model == "all_tf":
        model_path = TFBS_MODEL_PATH
    elif model == "class1_tf":
        model_path = TFBS_MODEL_CLASS1_PATH
    elif model == "nucleosome":
        model_path = NUCLEOSOME_MODEL_PATH
    else:
        raise ValueError(
            f"Invalid model: {model}, needs to be one of 'all_tf', 'class1_tf', 'nucleosome'."
        )
    model = torch.jit.load(model_path)
    return model


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


class FootPrintScoreModel:
    """Calculate the TFBS score for the given footprint."""

    def __init__(self, modes=None, device=None):
        self._all_tf_tfbs_model = None
        self._class1_tf_tfbs_model = None
        self._nucleosome_tfbs_model = None
        if modes is None:
            modes = np.arange(2, 101, 1)
        self.modes = modes
        if device is None:
            device = try_gpu()
        self.device = device

    @property
    def all_tf_tfbs_model(self):
        """Get the TFBS model for all TFs."""
        if self._all_tf_tfbs_model is None:
            model = get_footprint_to_tfbs_model("all_tf").to(self.device)
            model.eval()
            self._all_tf_tfbs_model = model
        else:
            model = self._all_tf_tfbs_model
        mode_idx = self._mode_to_mode_idx(model.scales)
        return model, mode_idx

    @property
    def class1_tf_tfbs_model(self):
        """Get the TFBS model for TFs with strong footprint (Class I)."""
        if self._class1_tf_tfbs_model is None:
            model = get_footprint_to_tfbs_model("class1_tf").to(self.device)
            model.eval()
            self._class1_tf_tfbs_model = model
        else:
            model = self._class1_tf_tfbs_model
        mode_idx = self._mode_to_mode_idx(model.scales)
        return model, mode_idx

    @property
    def nucleosome_tfbs_model(self):
        """Get the nucleosome model."""
        if self._nucleosome_tfbs_model is None:
            model = get_footprint_to_tfbs_model("nucleosome").to(self.device)
            model.eval()
            self._nucleosome_tfbs_model = model
        else:
            model = self._nucleosome_tfbs_model
        mode_idx = self._mode_to_mode_idx(model.scales)
        return model, mode_idx

    def get_tfbs_score(
        self, model: str, mode_idx: int, fp: torch.Tensor, numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the TFBS score for the given model.

        Parameters
        ----------
        model : Model
            The model used for scoring.
        mode_idx : int
            The index of the mode.
        fp : torch.Tensor
            The footprint tensor.
        numpy : bool, optional
            Whether to return the score as a numpy array, by default False.

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            The TFBS score.

        """
        # post process fp for the score prediction
        fp = fp[:, mode_idx].cpu().numpy()
        model_modes = model.scales
        final = []
        for i in range(fp.shape[1]):
            footprintPvalMatrix = fp[:, i]
            footprintPvalMatrix = scipy.stats.norm.cdf(footprintPvalMatrix, 0, 1)
            footprintRadius = model_modes[i]
            smoothRadius = int(footprintRadius / 2)
            footprintPvalMatrix[np.isnan(footprintPvalMatrix)] = (
                1  # Set NA values to be pvalue = 1
            )
            pvalScoreMatrix = -np.log10(footprintPvalMatrix)
            pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
            pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20
            if smoothRadius > 0:
                maximum_filter_size = [0] * len(pvalScoreMatrix.shape)
                maximum_filter_size[-1] = 2 * smoothRadius
                pvalScoreMatrix = maximum_filter(
                    pvalScoreMatrix, tuple(maximum_filter_size), origin=-1
                )
                # Changed to smoothRadius.
                pvalScoreMatrix = rz_conv(pvalScoreMatrix, smoothRadius) / (
                    2 * smoothRadius
                )
            pvalScoreMatrix[np.isnan(pvalScoreMatrix)] = 0
            pvalScoreMatrix[np.isinf(pvalScoreMatrix)] = 20
            final.append(pvalScoreMatrix)
        final = np.stack(final, axis=1)
        final = torch.as_tensor(final, device=self.device)

        with torch.inference_mode():
            score = model(final)
        if numpy:
            score = score.cpu().numpy()
        return score

    def get_tfbs_score_all_tf(
        self, fp: torch.Tensor, numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the TFBS score for all TFs.

        Parameters
        ----------
        fp : torch.Tensor
            The footprint tensor.
        numpy : bool, optional
            Whether to return the score as a numpy array, by default False.

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            The TFBS score.
        """
        model, mode_idx = self.all_tf_tfbs_model
        return self.get_tfbs_score(model, mode_idx, fp, numpy)

    def get_tfbs_score_class1_tf(
        self, fp: torch.Tensor, numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the TFBS score for TFs with strong footprint (Class I).

        Parameters
        ----------
        fp : torch.Tensor
            The footprint tensor.
        numpy : bool, optional
            Whether to return the score as a numpy array, by default False.

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            The TFBS score.
        """
        model, mode_idx = self.class1_tf_tfbs_model
        return self.get_tfbs_score(model, mode_idx, fp, numpy)

    def get_nucleosome_score(
        self, fp: torch.Tensor, numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the nucleosome score.

        Parameters
        ----------
        fp : torch.Tensor
            The footprint tensor.
        numpy : bool, optional
            Whether to return the score as a numpy array, by default False.

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            The nucleosome score.
        """
        model, mode_idx = self.nucleosome_tfbs_model
        return self.get_tfbs_score(model, mode_idx, fp, numpy)

    def _mode_to_mode_idx(self, modes):
        return np.where(pd.Index(self.modes).isin(np.array(modes)))[0]


class FootPrintModel(_dispModel, FootPrintScoreModel):
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
        FootPrintScoreModel.__init__(self, modes=modes, device=device)

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
        atac: Union[torch.Tensor, np.ndarray],
        bias: Union[torch.Tensor, np.ndarray],
        modes: Optional[Union[torch.Tensor, np.ndarray]] = None,
        clip_min: int = -10,
        clip_max: int = 10,
        return_pval: bool = False,
        smooth_radius: Optional[int] = None,
        numpy: bool = False,
        tfbs_score_all: bool = False,
        tfbs_score_class1: bool = False,
        nucleosome_score: bool = False,
    ) -> Union[
        torch.Tensor,
        np.ndarray,
        tuple[
            Union[torch.Tensor, np.ndarray], dict[str, Union[torch.Tensor, np.ndarray]]
        ],
    ]:
        """
        Calculate the footprint.

        Parameters
        ----------
        atac : torch.Tensor or np.ndarray
            A tensor or numpy array containing the ATAC-seq data.
        bias : torch.Tensor or np.ndarray
            A tensor or numpy array containing the bias values.
        modes : torch.Tensor or np.ndarray, optional
            A tensor or numpy array containing the modes, by default None.
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
        tfbs_score_all : bool, optional
            Whether to return the TFBS score for all TFs, by default False.
        tfbs_score_class1 : bool, optional
            Whether to return the TFBS score for TFs with strong footprint (Class I), by default False.
        nucleosome_score : bool, optional
            Whether to return the nucleosome score, by default False.

        Returns
        -------
        torch.Tensor or np.ndarray or Tuple[Union[torch.Tensor, np.ndarray], Dict[str, Union[torch.Tensor, np.ndarray]]]]
            A tensor or numpy array containing the computed footprint.
            If any of the optional parameters `tfbs_score_all`, `tfbs_score_class1`, or `nucleosome_score` is set to True,
            a dictionary containing the corresponding scores will also be returned.
        """
        atac = torch.as_tensor(atac, dtype=torch.float32, device=self.device)
        bias = torch.as_tensor(bias, dtype=torch.float32, device=self.device)

        if modes is None:
            modes = self.modes
        modes = torch.as_tensor(modes, device=self.device)

        # add batch dimension if necessary
        if len(atac.shape) == 1:
            atac = atac.unsqueeze(0)
        if len(bias.shape) == 1:
            bias = bias.unsqueeze(0)

        with torch.inference_mode():
            raw_fp = self.forward(
                atac=atac,
                bias=bias,
                modes=modes,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            _fp = raw_fp.clone()
            if return_pval:
                _fp = zscore2pval_torch(_fp)
                if smooth_radius is not None:
                    _device = _fp.device
                    _fp = _fp.cpu().numpy()
                    _fp = smooth_footprint(_fp, smooth_radius)
                    if not numpy:
                        _fp = torch.as_tensor(_fp, device=_device)

        if numpy and isinstance(_fp, torch.Tensor):
            _fp = _fp.detach().cpu().numpy()

        score_dict = self._get_score_dict(
            raw_fp=raw_fp,
            numpy=numpy,
            tfbs_score_all=tfbs_score_all,
            tfbs_score_class1=tfbs_score_class1,
            nucleosome_score=nucleosome_score,
        )

        if len(score_dict) == 0:
            return _fp
        return _fp, score_dict

    def _get_score_dict(
        self, raw_fp, numpy, tfbs_score_all, tfbs_score_class1, nucleosome_score
    ):
        score_dict = {}
        if tfbs_score_all:
            tfbs_score = self.get_tfbs_score_all_tf(raw_fp, numpy=numpy)
            score_dict["tfbs_score_all_tf"] = tfbs_score
        if tfbs_score_class1:
            tfbs_score = self.get_tfbs_score_class1_tf(raw_fp, numpy=numpy)
            score_dict["tfbs_score_class1_tf"] = tfbs_score
        if nucleosome_score:
            tfbs_score = self.get_nucleosome_score(raw_fp, numpy=numpy)
            score_dict["nucleosome_score"] = tfbs_score
        return score_dict

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

    def add_atac_bw(self, *args, **kwargs):
        """
        Add an ATAC bigWig file to the atac_handles dictionary. If name is not provided, the name of the file will be used.

        Parameters
        ----------
        *args : List[str]
            The paths to the ATAC bigWig files.
        **kwargs : Dict[str, str]
            The names and paths to the ATAC bigWig files.
        """
        bw_to_add = kwargs
        for arg in args:
            name = pathlib.Path(str(arg)).name
            bw_to_add[name] = arg
        for name, path in bw_to_add.items():
            assert (
                name not in self.atac_handles
            ), f"ATAC bigWig file with name {name} already exists."
            self.atac_handles[name] = pyBigWig.open(path)

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
        atac_names: Optional[list[str]] = None,
        modes: Optional[list[str]] = None,
        clip_min: int = -10,
        clip_max: int = 10,
        return_pval: bool = False,
        smooth_radius: int = None,
        numpy=False,
        return_signal: bool = False,
        tfbs_score_all: bool = False,
        tfbs_score_class1: bool = False,
        nucleosome_score: bool = False,
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
        return_signal : bool, optional
            Whether to return the atac and bias signals along with the footprints.
        tfbs_score_all : bool, optional
            Whether to return the TFBS score for all TFs along with the footprints.
        tfbs_score_class1 : bool, optional
            Whether to return the TFBS score for TFs with strong footprint (Class I) along with the footprints.
        nucleosome_score : bool, optional
            Whether to return the nucleosome score along with the footprints.

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
        if return_signal:
            fp_dict["tn5_bias"] = bias
        for name in atac_names:
            atac = self.fetch_atac(chrom, start, end, name)
            result = self._calculate_footprint(
                atac=atac,
                bias=bias,
                clip_min=clip_min,
                clip_max=clip_max,
                return_pval=return_pval,
                smooth_radius=smooth_radius,
                numpy=numpy,
                tfbs_score_all=tfbs_score_all,
                tfbs_score_class1=tfbs_score_class1,
                nucleosome_score=nucleosome_score,
            )

            _tfbs = {}
            if len(result) == 1:
                _fp = result
            else:
                _fp, _tfbs = result

            fp_dict[f"{name}_footprint"] = _fp
            for key, value in _tfbs.items():
                fp_dict[f"{name}_{key}"] = value

            if return_signal:
                fp_dict[f"{name}_atac"] = atac
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
        tfbs_score_all: bool = False,
        tfbs_score_class1: bool = False,
        nucleosome_score: bool = False,
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
        tfbs_score_all : bool, optional
            Whether to return the TFBS score for all TFs along with the footprints.
        tfbs_score_class1 : bool, optional
            Whether to return the TFBS score for TFs with strong footprint (Class I) along with the footprints.
        nucleosome_score : bool, optional
            Whether to return the nucleosome score along with the footprints.

        Returns
        -------
        torch.Tensor or np.ndarray or Tuple[np.ndarray, np.ndarray]
            The computed footprint or a tuple containing the footprint and the TFBS score dict.
        """
        result = self._calculate_footprint(
            atac=atac_data,
            bias=bias_data,
            clip_min=clip_min,
            clip_max=clip_max,
            return_pval=return_pval,
            smooth_radius=smooth_radius,
            numpy=numpy,
            modes=modes,
            tfbs_score_all=tfbs_score_all,
            tfbs_score_class1=tfbs_score_class1,
            nucleosome_score=nucleosome_score,
        )
        return result

    def _get_footprint_mask(self, raw_fp, as_torch=True):
        footprint_pval = self.postprocess_footprint(raw_fp.clone())
        scores = self._get_score_dict(
            raw_fp=raw_fp,
            numpy=True,
            tfbs_score_all=True,
            tfbs_score_class1=True,
            nucleosome_score=True,
        )

        batch_size = footprint_pval.shape[0]

        masks = []
        for i in range(batch_size):
            use_fp = footprint_pval[i]
            use_scores = {k: v[i] for k, v in scores.items()}
            peaks_df = get_peaks_df_pval_fp(use_fp, use_scores)
            mask = get_masks(use_fp, peaks_df)
            masks.append(mask)
        masks = np.stack(masks, axis=0)
        if as_torch:
            masks = torch.as_tensor(masks, device=self.device)
        return masks

    @staticmethod
    def postprocess_footprint(
        footprint: Union[torch.Tensor, np.ndarray], smooth_radius: int = 5
    ) -> np.ndarray:
        """Run postprocessing on the computed footprint."""
        return postprocess_footprint(footprint, smooth_radius)
