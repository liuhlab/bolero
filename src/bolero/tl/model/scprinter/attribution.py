from functools import partial

import numpy as np
import torch
from scprinter.seq.attribution_wrapper import (
    CountWrapper,
    JustSumWrapper,
    ProfileWrapperFootprint,
    ProfileWrapperFootprintClass,
)
from scprinter.seq.attributions import calculate_attributions, projected_shap

from bolero.utils import try_gpu


class BatchAttribution:
    """Class for performing batch attribution on sequence data."""

    def __init__(
        self,
        model: torch.nn.Module,
        wrapper: str,
        method: str,
        prefix: str,
        modes: range = range(0, 30),
        decay: float = 0.85,
    ):
        """
        Initialize the BatchAttribution class.

        Args:
            model (torch.nn.Module): The model to be used for attribution.
            wrapper (str): The type of wrapper to be used.
            method (str): The attribution method to be used.
            prefix (str): The prefix to be used for the output key.
            modes (range, optional): The range of modes to be considered. Defaults to range(0, 30).
            decay (float, optional): The decay factor. Defaults to 0.85.
        """
        self.device = str(try_gpu())
        self.use_cuda = self.device != "cpu"
        self.model = self._prepare_model(
            model=model, wrapper=wrapper, modes=modes, decay=decay
        )
        self.method = method
        self.prefix = prefix

        self.attributor = partial(
            calculate_attributions,
            n_shuffles=20,
            method=self.method,
            verbose=False,
            model=self.model,
        )
        # project channel-by-sequence 2D attributions to sequence 1D attributions
        self.projector = partial(projected_shap, bs=64, device="cpu")

    def _prepare_model(
        self, model: torch.nn.Module, wrapper: str, modes: range, decay: float
    ) -> torch.nn.Module:
        """
        Prepare the model with the specified wrapper.

        Args:
            model (torch.nn.Module): The model to be wrapped.
            wrapper (str): The type of wrapper to be used.
            modes (range): The range of modes to be considered.
            decay (float): The decay factor.

        Returns
        -------
            torch.nn.Module: The wrapped model.
        """
        n_out = torch.from_numpy(np.array(modes))
        if wrapper == "classification":
            model = ProfileWrapperFootprintClass(
                model,
                nth_output=n_out,
                res=1,
                reduce_mean=False,
                decay=decay,
            )
        elif wrapper == "classification_reduce":
            model = ProfileWrapperFootprintClass(
                model,
                nth_output=n_out,
                res=1,
                reduce_mean=True,
                decay=decay,
            )
        elif wrapper == "regression":
            model = ProfileWrapperFootprint(
                model, nth_output=n_out, res=1, reduce_mean=False
            )
        elif wrapper == "regression_reduce":
            model = ProfileWrapperFootprint(
                model, nth_output=n_out, res=1, reduce_mean=True
            )
        elif wrapper == "just_sum":
            model = JustSumWrapper(model, nth_output=n_out, res=1, threshold=0.301)
        elif wrapper == "count":
            model = CountWrapper(model)
        else:
            raise ValueError(f"Unknown wrapper type {wrapper}")

        if self.use_cuda:
            model = model.cuda()
        return model

    def __call__(self, data: dict) -> dict:
        """
        Perform attribution on the given data.

        Args:
            data (dict): The input data.

        Returns
        -------
            dict: The data with attributions added.
        """
        _one_hot = data["dna_one_hot"]

        if isinstance(_one_hot, np.ndarray):
            _one_hot = torch.from_numpy(_one_hot).float()
        _one_hot = _one_hot.cpu()
        attrs = self.attributor(X=_one_hot)
        data[f"{self.prefix}_attributions"] = attrs.cpu().numpy()

        attrs_1d: np.array = self.projector(attributions=attrs, seqs=_one_hot)
        data[f"{self.prefix}_attributions_1d"] = attrs_1d
        return data
