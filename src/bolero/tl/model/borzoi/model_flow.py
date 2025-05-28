from typing import Any

import torch
import torch.nn as nn
from torchdiffeq import odeint

from bolero.tl.model.borzoi.model_lora import BorzoiLoRA


class BorzoiLoRAFlowWrapperForODE(nn.Module):
    """
    Wrapper for BorzoiLoRA to use with ODE.
    """

    def __init__(self, model: BorzoiLoRA, cell_emb, cond_emb, dna_one_hot):
        super().__init__()
        self.model = model
        self.cond_flow_module = model.cond_flow_module
        self.cell_emb = cell_emb
        self.cond_emb = cond_emb
        self.dna_one_hot = dna_one_hot

    def forward(self, t: torch.Tensor, x_t: torch.Tensor):
        """
        Forward pass for the ODE wrapper.
        """
        # Aggregate cell, condition and time embeddings
        emb = self.cond_flow_module(
            cell_emb=self.cell_emb, time=t, cond_emb=self.cond_emb
        )

        # Compute the velocity field
        vt = self.model(x=self.dna_one_hot, signal=x_t, embedding=emb, crop=False)
        return vt


class BorzoiLoRAFlowPredictor:
    def __init__(
        self,
        model: BorzoiLoRA,
        cell_emb: torch.Tensor = None,
        cond_emb: torch.Tensor = None,
        dna_one_hot: torch.Tensor = None,
        **ode_solver_kwargs,
    ):
        self.model = model
        self.vf_model = self._set_vf_model(cell_emb, cond_emb, dna_one_hot)
        self.ode_solver_kwargs = {
            "method": "dopri5",
            "rtol": 1e-4,
            "atol": 1e-4,
            **ode_solver_kwargs,
        }

    def _set_vf_model(
        self,
        cell_emb: torch.Tensor,
        cond_emb: torch.Tensor,
        dna_one_hot: torch.Tensor,
    ):
        if cell_emb is None or cond_emb is None or dna_one_hot is None:
            return None

        vf_model = BorzoiLoRAFlowWrapperForODE(
            model=self.model,
            cell_emb=cell_emb,
            cond_emb=cond_emb,
            dna_one_hot=dna_one_hot,
        )
        return vf_model

    def predict(
        self,
        x_0: torch.Tensor,
        t_range: torch.Tensor = None,
        cell_emb: torch.Tensor = None,
        cond_emb: torch.Tensor = None,
        dna_one_hot: torch.Tensor = None,
        return_traj: bool = False,
        **kwargs: Any,
    ):
        """
        Predict the trajectory of the ODE given an initial condition.
        """
        if self.vf_model is None:
            vf_model = self._set_vf_model(cell_emb, cond_emb, dna_one_hot)
            assert vf_model is not None, (
                "Velocity field model must be set before prediction, "
                "please provide cell_emb, cond_emb, dna_one_hot."
            )
        else:
            vf_model = self.vf_model

        if t_range is None:
            t_range = [0.0, 1.0]

        t_span = torch.tensor(
            [t_range[0], t_range[-1]], dtype=torch.float32, device=x_0.device
        )
        x_pred_traj = odeint(
            func=vf_model,
            y0=x_0,
            t=t_span,
            **self.ode_solver_kwargs,
            **kwargs,
        )
        if return_traj:
            return x_pred_traj
        else:
            return x_pred_traj[-1]
