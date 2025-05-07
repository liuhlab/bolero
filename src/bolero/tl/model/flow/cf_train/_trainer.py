from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import wandb
from numpy.typing import ArrayLike
from tqdm import tqdm

from bolero.tl.model.flow._otfm import OTFlowMatching
from bolero.tl.model.flow.cf_data import TrainSampler, ValidationSampler

from ._callbacks import BaseCallback, CallbackRunner


def _init_wandb(wandb_project, wandb_run, cfg):
    if wandb_project is None:
        return None
    return wandb.init(project=wandb_project, name=wandb_run, config=cfg, reinit=True)


def _log_wandb(output, step, wandb_run, index=None):
    if wandb_run is None:
        return

    metrics_to_log = ["loss", "grad_norm"]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT solver with a conditional velocity field.

    Parameters
    ----------
        dataloader
            Data sampler.
        solver
            OTFM/GENOT solver with a conditional velocity field.
        seed
            Random seed for subsampling validation data.

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        solver: OTFlowMatching,
        lr: float = 1e-4,
        accumulate_grad: int = 20,
    ):
        if not isinstance(solver, OTFlowMatching):
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching or GENOT, got {type(solver)}"
            )
        self.solver = solver
        self.training_logs: dict[str, Any] = {}
        self.device = solver.device

        self._optimizer = None
        self.lr = lr
        self.accumulate_grad = accumulate_grad

    def _validation_step(
        self,
        val_data: dict[str, ValidationSampler],
        mode: Literal["on_log_iteration", "on_train_end"] = "on_log_iteration",
    ) -> tuple[
        dict[str, dict[str, ArrayLike]],
        dict[str, dict[str, ArrayLike]],
    ]:
        """Compute predictions for validation data."""
        # TODO: Sample fixed number of conditions to validate on

        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}
        for val_key, vdl in val_data.items():
            batch = vdl.sample(mode=mode)
            src = batch["source"]
            condition = batch.get("condition", None)
            true_tgt = batch["target"]
            valid_pred_data[val_key] = self.solver.predict(src, condition)
            valid_true_data[val_key] = true_tgt

        return valid_true_data, valid_pred_data

    def _prepare_train(self, wandb_project, wandb_run, cfg) -> None:
        """Prepare the training process."""
        self._optimizer = torch.optim.Adam(
            self.solver.vf.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )

        wandb_run = _init_wandb(wandb_project, wandb_run, cfg)
        return wandb_run

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: TrainSampler,
        num_iterations: int,
        valid_freq: int,
        valid_loaders: dict[str, ValidationSampler] | None = None,
        monitor_metrics: Sequence[str] = (),
        callbacks: Sequence[BaseCallback] = (),
        wandb_project: str = None,
        wandb_run: str = None,
        cfg: dict[str, Any] = None,
    ) -> OTFlowMatching:
        """Trains the model.

        Parameters
        ----------
            dataloader
                Dataloader used.
            num_iterations
                Number of iterations to train the model.
            valid_freq
                Frequency of validation.
            valid_loaders
                Valid loaders.
            callbacks
                Callback functions.
            monitor_metrics
                Metrics to monitor.

        Returns
        -------
            The trained model.
        """
        cfg = cfg or {}
        wandb_run = self._prepare_train(wandb_project, wandb_run, cfg)

        self.training_logs = {"loss": []}

        # Initiate callbacks
        valid_loaders = valid_loaders or {}
        crun = CallbackRunner(
            callbacks=callbacks,
        )
        crun.on_train_begin()

        pbar = tqdm(range(num_iterations))
        for it in pbar:
            batch = dataloader.sample()
            loss = self.solver.step_fn(batch)
            self.training_logs["loss"].append(float(loss))
            if np.isnan(loss.item()):
                raise ValueError("Loss is NaN. Check your data and model parameters.")
            loss.backward()

            if ((it - 1) % valid_freq == 0) and (it > 1):
                with torch.inference_mode():
                    # Get predictions from validation data
                    valid_true_data, valid_pred_data = self._validation_step(
                        valid_loaders, mode="on_log_iteration"
                    )

                    # Run callbacks
                    metrics = crun.on_log_iteration(valid_true_data, valid_pred_data)  # type: ignore[arg-type]
                    self._update_logs(metrics)

                    # Update progress bar
                    mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                    postfix_dict = {
                        metric: round(self.training_logs[metric][-1], 3)
                        for metric in monitor_metrics
                    }
                    postfix_dict["loss"] = round(mean_loss, 3)
                    pbar.set_postfix(postfix_dict)

            if (it + 1) % self.accumulate_grad == 0:
                # clip gradients
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.solver.vf.parameters(), 1.0
                )
                self._optimizer.step()
                self._optimizer.zero_grad()

                _log_wandb(
                    {"loss": loss.detach().cpu(), "grad_norm": total_norm},
                    it,
                    wandb_run,
                )

        if num_iterations > 0:
            with torch.inference_mode():
                # Get predictions from validation data
                valid_true_data, valid_pred_data = self._validation_step(
                    valid_loaders, mode="on_train_end"
                )
                metrics = crun.on_train_end(valid_true_data, valid_pred_data)
                self._update_logs(metrics)

        self.solver.is_trained = True
        return self.solver
