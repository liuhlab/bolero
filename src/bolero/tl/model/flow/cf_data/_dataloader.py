import abc
from typing import Any, Literal, Optional

import numpy as np
import torch

from ._data import PredictionData, TrainingData, ValidationData

__all__ = ["TrainSampler", "ValidationSampler", "PredictionSampler"]


def _batch_to_tensor(
    batch: dict[str, Any], device: str | torch.device
) -> dict[str, torch.Tensor]:
    """Convert a batch of data to a tensor."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, np.ndarray):
            batch[k] = torch.as_tensor(v).to(device)
        elif isinstance(v, list):
            batch[k] = [torch.as_tensor(el).to(device) for el in v]
        elif isinstance(v, dict):
            batch[k] = _batch_to_tensor(v, device)
        else:
            batch[k] = v
    return batch


class TrainSampler:
    """Data sampler for :class:`~cellflow.data.TrainingData`.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.
    """

    def __init__(
        self,
        data: TrainingData,
        batch_size: int = 1024,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)
        self.device = torch.device(device)

        self._data = data
        # indices of all cells
        self._data_idcs = torch.arange(data.cell_data.shape[0], dtype=torch.long)
        self.batch_size = batch_size

        # number of source / target distributions
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations

        # for each source i, a LongTensor of its possible target IDs
        self.conditional_samplings = [
            torch.as_tensor(data.control_to_perturbation[i], dtype=torch.long)
            for i in range(self.n_source_dists)
        ]

        # function to grab the per‑target embedding (matches your JAX get_embeddings)
        if data.condition_data is not None:
            self.get_embeddings = lambda idx: {
                name: torch.as_tensor(arr[idx]).to(torch.float32).unsqueeze(0)
                for name, arr in data.condition_data.items()
            }
        else:
            self.get_embeddings = None

        # masks and raw cell data
        self.split_covariates_mask = data.split_covariates_mask
        self.perturbation_covariates_mask = data.perturbation_covariates_mask
        self.cell_data: torch.Tensor = data.cell_data

    def sample(self) -> dict[str, torch.Tensor]:
        """Sample one batch"""
        # — pick a random source distribution
        src_dist = torch.randint(self.n_source_dists, ()).item()

        # — sample batch_size source cells from that dist
        src_mask = (self.split_covariates_mask == src_dist).float()
        src_p = src_mask / src_mask.sum()
        src_idcs = torch.multinomial(src_p, self.batch_size, replacement=True)
        src_batch = self.cell_data[src_idcs]

        # — pick one target dist conditioned on source_dist
        candidates = self.conditional_samplings[src_dist]
        choice = torch.randint(candidates.size(0), ()).item()
        tgt_dist = candidates[choice].item()

        # — sample batch_size target cells
        tgt_mask = (self.perturbation_covariates_mask == tgt_dist).float()
        tgt_p = tgt_mask / tgt_mask.sum()
        tgt_idcs = torch.multinomial(tgt_p, self.batch_size, replacement=True)
        tgt_batch = self.cell_data[tgt_idcs]

        out: dict[str, Any] = {
            "src_cell_data": src_batch,
            "tgt_cell_data": tgt_batch,
        }

        # — attach condition if present
        if self.get_embeddings is not None:
            out["condition"] = self.get_embeddings(tgt_dist)

        out = _batch_to_tensor(out, self.device)
        return out

    @property
    def data(self) -> TrainingData:
        """The training data."""
        return self._data


class BaseValidSampler(abc.ABC):
    @property
    def device(self) -> torch.device:
        """The device to use for the data."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @abc.abstractmethod
    def sample(*args, **kwargs):
        pass

    def _get_key(self, cond_idx: int) -> tuple[str, ...]:
        if len(self._data.perturbation_idx_to_id):  # type: ignore[attr-defined]
            return self._data.perturbation_idx_to_id[cond_idx]  # type: ignore[attr-defined]
        cov_combination = self._data.perturbation_idx_to_covariates[cond_idx]  # type: ignore[attr-defined]
        return tuple(cov_combination[i] for i in range(len(cov_combination)))

    def _get_perturbation_to_control(
        self, data: ValidationData | PredictionData
    ) -> dict[int, int]:
        d = {}
        for k, v in data.control_to_perturbation.items():
            for el in v:
                d[el] = k
        return d

    def _get_condition_data(self, cond_idx: int) -> torch.Tensor:
        return {
            k: torch.from_numpy(v[[cond_idx], ...]).to(torch.float32)
            for k, v in self._data.condition_data.items()
        }  # type: ignore[attr-defined]


class ValidationSampler(BaseValidSampler):
    """Data sampler for :class:`~cellflow.data.ValidationData`.

    Parameters
    ----------
    val_data
        The validation data.
    seed
        Random seed.
    """

    def __init__(self, val_data: ValidationData, seed: int = 0) -> None:
        self._data = val_data
        self.perturbation_to_control = self._get_perturbation_to_control(val_data)
        self.n_conditions_on_log_iteration = (
            val_data.n_conditions_on_log_iteration
            if val_data.n_conditions_on_log_iteration is not None
            else val_data.n_perturbations
        )
        self.n_conditions_on_train_end = (
            val_data.n_conditions_on_train_end
            if val_data.n_conditions_on_train_end is not None
            else val_data.n_perturbations
        )
        self.rng = np.random.default_rng(seed)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(
        self, mode: Literal["on_log_iteration", "on_train_end"], device=None
    ) -> Any:
        """Sample data for validation.

        Parameters
        ----------
        mode
            Sampling mode. Either ``"on_log_iteration"`` or ``"on_train_end"``.

        Returns
        -------
        Dictionary with source, condition, and target data from the validation data.
        """
        if device is None:
            device = self.device

        size = (
            self.n_conditions_on_log_iteration
            if mode == "on_log_iteration"
            else self.n_conditions_on_train_end
        )
        condition_idcs = self.rng.choice(
            self._data.n_perturbations, size=(size,), replace=False
        )

        source_idcs = [
            self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs
        ]
        source_cells_mask = [
            self._data.split_covariates_mask == source_idx for source_idx in source_idcs
        ]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        target_cells_mask = [
            cond_idx == self._data.perturbation_covariates_mask
            for cond_idx in condition_idcs
        ]
        target_cells = [self._data.cell_data[mask] for mask in target_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        true_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]
            true_dict[k] = target_cells[i]

        data = {"source": cell_rep_dict, "condition": cond_dict, "target": true_dict}
        data = _batch_to_tensor(data, device)
        return data

    @property
    def data(self) -> ValidationData:
        """The validation data."""
        return self._data


class PredictionSampler(BaseValidSampler):
    """Data sampler for :class:`~cellflow.data.PredictionData`.

    Parameters
    ----------
    pred_data
        The prediction data.

    """

    def __init__(self, pred_data: PredictionData) -> None:
        self._data = pred_data
        self.perturbation_to_control = self._get_perturbation_to_control(pred_data)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self) -> Any:
        """Sample data for prediction.

        Returns
        -------
        Dictionary with source and condition data from the prediction data.
        """
        condition_idcs = range(self._data.n_perturbations)

        source_idcs = [
            self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs
        ]
        source_cells_mask = [
            self._data.split_covariates_mask == source_idx for source_idx in source_idcs
        ]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]

        data = {
            "source": cell_rep_dict,
            "condition": cond_dict,
        }
        data = _batch_to_tensor(data, self.device)
        return data

    @property
    def data(self) -> PredictionData:
        """The training data."""
        return self._data
