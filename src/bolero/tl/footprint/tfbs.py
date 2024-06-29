import os
from typing import Union

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.ndimage import maximum_filter

from bolero.utils import try_gpu

from .utils import rz_conv

# get env viriable SCPRINTER_DATA
SCPRINTER_DATA = os.getenv("SCPRINTER_DATA")

# modal trained all TF chip data
TFBS_MODEL_PATH = f"{SCPRINTER_DATA}/footprint_to_TFBS_conv_model.pt"
# model trained on TFs with strong footprint (Class I) data
TFBS_MODEL_CLASS1_PATH = f"{SCPRINTER_DATA}/footprint_to_TFBS_class1_conv_model.pt"
# model trained on nucleosome data
NUCLEOSOME_MODEL_PATH = f"{SCPRINTER_DATA}/footprint_to_nucleosome_conv_model.pt"
# model trained on all TF chip data, using projected footprint attribution score as input.
ATTR_FP_TFBS_MODEL_PATH = f"{SCPRINTER_DATA}/TFBS_0_conv_v2.pt"
# model trained on all TF chip data, using projected coverage attribution score as input.
ATTR_COV_TFBS_MODEL_PATH = f"{SCPRINTER_DATA}/TFBS_1_conv_v2.pt"


def get_footprint_to_tfbs_model(model) -> torch.nn.Module:
    """Get the footprint to TFBS model."""
    if model == "all_tf":
        model_path = TFBS_MODEL_PATH
    elif model == "class1_tf":
        model_path = TFBS_MODEL_CLASS1_PATH
    elif model == "nucleosome":
        model_path = NUCLEOSOME_MODEL_PATH
    elif model == "attr_fp":
        model_path = ATTR_FP_TFBS_MODEL_PATH
    elif model == "attr_cov":
        model_path = ATTR_COV_TFBS_MODEL_PATH
    else:
        raise ValueError(
            f"Invalid model: {model}, needs to be one of 'all_tf', 'class1_tf', 'nucleosome'."
        )
    model = torch.jit.load(model_path)
    return model


class FootPrintScoreModel:
    """Calculate the TFBS score for the given footprint."""

    def __init__(self, modes=None, device=None, load=True):
        self._all_tf_tfbs_model = None
        self._class1_tf_tfbs_model = None
        self._nucleosome_tfbs_model = None
        if modes is None:
            modes = np.arange(2, 101, 1)
        self.modes = modes
        if device is None:
            device = try_gpu()
        self.device = device

        if load:
            # trigger model load
            _ = self.all_tf_tfbs_model
            _ = self.class1_tf_tfbs_model
            _ = self.nucleosome_tfbs_model

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

    def _get_tfbs_score(
        self, model: str, mode_idx: int, fp: torch.Tensor, numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the TFBS score for the given model.

        Parameters
        ----------
        model : Model
            The model used for scoring.
        mode_idx : int
            The index of the mode to be used for TFBS score.
            This index is generated along with the model, user should not provide it.
        fp : torch.Tensor
            The raw footprint tensor generated from ATAC insertion track and Tn5 bias OR from model prediction.
            The value should be z-score, and modes are the same as self.modes.
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
        return self._get_tfbs_score(model, mode_idx, fp, numpy)

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
        return self._get_tfbs_score(model, mode_idx, fp, numpy)

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
        return self._get_tfbs_score(model, mode_idx, fp, numpy)

    def _mode_to_mode_idx(self, modes):
        return np.where(pd.Index(self.modes).isin(np.array(modes)))[0]

    def get_all_scores(self, fp: torch.Tensor, numpy: bool = False):
        """
        Get the TFBS scores for all models.

        Parameters
        ----------
        fp : torch.Tensor
            The footprint tensor.
        numpy : bool, optional
            Whether to return the score as a numpy array, by default False.

        Returns
        -------
        dict
            The TFBS scores for all models.
        """
        scores = {
            "tfbs_score_all_tf": self.get_tfbs_score_all_tf(fp, numpy),
            "tfbs_score_class1_tf": self.get_tfbs_score_class1_tf(fp, numpy),
            "nucleosome_score": self.get_nucleosome_score(fp, numpy),
        }
        return scores


class AttrobutionScoreModel:
    """Calculate the TFBS score for the given 1-D projected attribution."""

    def __init__(self, score_type="attr_fp", device=None):
        if device is None:
            device = try_gpu()
        self.device = device

        if score_type == "attr_fp":
            self.tfbs_model = get_footprint_to_tfbs_model("attr_fp")
        elif score_type == "attr_cov":
            self.tfbs_model = get_footprint_to_tfbs_model("attr_cov")
        else:
            raise ValueError(
                f"Invalid score_type: {score_type}, needs to be one of 'attr_fp', 'attr_cov'."
            )
        self.tfbs_model = self.tfbs_model.to(self.device)

    def __call__(
        self, attr_score: torch.Tensor, numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the TFBS score for the given model.

        Parameters
        ----------
        model : Model
            The model used for scoring.
        mode_idx : int
            The index of the mode to be used for TFBS score.
            This index is generated along with the model, user should not provide it.
        fp : torch.Tensor
            The raw footprint tensor generated from ATAC insertion track and Tn5 bias OR from model prediction.
            The value should be z-score, and modes are the same as self.modes.
        numpy : bool, optional
            Whether to return the score as a numpy array, by default False.

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            The TFBS score.

        """
        # post process fp for the score prediction
        with torch.inference_mode():
            tfbs_score = self.tfbs_model(attr_score)
        if numpy:
            tfbs_score = tfbs_score.cpu().numpy()
        return tfbs_score
