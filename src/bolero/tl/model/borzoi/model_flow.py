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
        emb = self.cond_flow_module(cell_emb=self.cell_emb, t=t, cond_emb=self.cond_emb)

        # Compute the velocity field
        vt = self.model(x=self.dna_one_hot, signal=x_t, embedding=emb)
        return vt


class BorzoiLoRAFlowPredictor:
    def __init__(
        self,
        model: BorzoiLoRA,
        cell_emb: torch.Tensor,
        cond_emb: torch.Tensor,
        **ode_solver_kwargs,
    ):
        self.device = model.device
        self.vf_model = BorzoiLoRAFlowWrapperForODE(
            model=model, cell_emb=cell_emb, cond_emb=cond_emb
        )
        self.ode_solver_kwargs = {
            "method": "dopri5",
            "rtol": 1e-4,
            "atol": 1e-4,
            **ode_solver_kwargs,
        }

    def predict(
        self,
        x_0: torch.Tensor,
        t_range: torch.Tensor = None,
        return_traj: bool = False,
        **kwargs: Any,
    ):
        """
        Predict the trajectory of the ODE given an initial condition.
        """
        if t_range is None:
            t_range = torch.tensor([0.0, 1.0], device=self.vf_model.device)

        t_span = torch.tensor([t_range[0], t_range[-1]], device=self.device)
        x_pred_traj = odeint(
            func=self.vf_model,
            y0=x_0,
            t=t_span,
            **self.ode_solver_kwargs,
            **kwargs,
        )
        if return_traj:
            return x_pred_traj
        else:
            return x_pred_traj[-1]
