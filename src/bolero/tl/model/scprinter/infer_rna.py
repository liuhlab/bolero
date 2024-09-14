import pathlib
from typing import Union

import torch

from bolero.tl.pseudobulk.rna_atac_pseudobulk import RNAVQPseudobulker

from .infer import BaseFootprintInferencer, scPrinterPseudobulkInferencer


class TrainedRNALoraModel:
    """
    TrainedLoraModel class represents a trained Lora model.

    Parameters
    ----------
        model (Union[str, pathlib.Path, torch.nn.Module]): The trained Lora model.
        pseudobulker (RNAVQPseudobulker): The pseudobulk data handler.

    Attributes
    ----------
        model (torch.nn.Module): The trained Lora model.
        pseudobulker (RNAVQPseudobulker): The pseudobulk data handler.

    Methods
    -------
        get_collapsed_model(key: str) -> torch.nn.Module: Get the collapsed model for a given key.

    """

    default_config: dict = {
        "model": "REQUIRED",
        "vq_records": "REQUIRED",
        "use_vq_emb": True,
    }

    def __init__(
        self,
        model: Union[str, pathlib.Path, torch.nn.Module],
        vq_records,
        use_vq_emb=True,
    ) -> None:
        if isinstance(model, (str, pathlib.Path)):
            model = torch.load(model, map_location="cpu", weights_only=False).eval()
        print(type(model), "model loaded")
        self.model: torch.nn.Module = model
        self.pseudobulker: RNAVQPseudobulker = RNAVQPseudobulker.create_from_config(
            vq_records=vq_records, use_vq_emb=use_vq_emb
        )

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
        *_, emb, _ = self.pseudobulker.take_by_name(key)

        # add bs dimension
        emb = torch.Tensor(emb).unsqueeze(0)
        _model = self.model.collapse(
            cell_embedding=emb, region_embedding=None, requires_grad=False
        )
        return _model


class scPrinterRNAPseudobulkInferencer(scPrinterPseudobulkInferencer):
    """
    Class for performing pseudobulk inference using scPrinter model and RNA VQ pseudobulk.
    """

    model_class: type = TrainedRNALoraModel
    infer_class: type = BaseFootprintInferencer
    default_config: dict = {**infer_class.default_config, **model_class.default_config}
