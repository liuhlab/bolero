import torch

from bolero.tl.model.borzoi.model import Borzoi
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.model.scprinter.model import seq2PRINT, seq2PRINTLoRA

_model_cls = Borzoi | BorzoiLoRA | seq2PRINT | seq2PRINTLoRA


def _get_device():
    """
    Get the device to be used for PyTorch operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class GenericPredictor:
    def __init__(self, config, model_class):
        self.config = config
        self.device = _get_device()

        self.model_class: _model_cls = model_class
        self._model = None

    def _create_model(self) -> _model_cls:
        model_config = self.config["model"]

        default_cfg = self.model_class.get_default_config()
        model_config = {k: v for k, v in model_config.items() if k in default_cfg}
        model_config = {**default_cfg, **model_config}
        model = self.model_class.create_from_config(model_config)
        return model

    def _load_ckeckpoint(self, model: _model_cls) -> _model_cls:
        checkpoint_path = self.config["checkpoint_path"]
        state = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state, strict=False)

        model.to(self.device)
        model.eval()
        return model

    @property
    def model(self) -> _model_cls:
        """
        Get the model, loading it if it hasn't been loaded yet.
        """
        if self._model is None:
            model = self._create_model()
            model = self._load_ckeckpoint(model)
            self._model = model
        return self._model


class BorzoiLoRAPredictor(GenericPredictor):
    def _create_model(self) -> BorzoiLoRA:
        model: BorzoiLoRA = super()._create_model()
        model.convert_to_lora()
        return model
