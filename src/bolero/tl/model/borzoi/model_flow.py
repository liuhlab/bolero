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

        # Data loader provides coverage count scale
        # model takes log1p scale as input
        if x_t is not None:
            x_t = torch.log1p(x_t)

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
        method: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-4,
        **ode_solver_kwargs,
    ):
        self.model = model
        self.vf_model = self._set_vf_model(cell_emb, cond_emb, dna_one_hot)
        self.ode_solver_kwargs = {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            **ode_solver_kwargs,
        }

    def _set_vf_model(
        self,
        cell_emb: torch.Tensor,
        cond_emb: torch.Tensor,
        dna_one_hot: torch.Tensor,
    ):
        if cell_emb is None or dna_one_hot is None:
            # cond_emb is optional
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

    def predict_vt(
        self,
        x_0: torch.Tensor,
        cell_emb: torch.Tensor = None,
        cond_emb: torch.Tensor = None,
        dna_one_hot: torch.Tensor = None,
    ):
        """
        Predict the velocity field at a time 0 given an initial condition.
        """
        t = torch.tensor([0], dtype=torch.float32, device=cell_emb.device)

        if self.vf_model is None:
            vf_model = self._set_vf_model(cell_emb, cond_emb, dna_one_hot)
            assert vf_model is not None, (
                "Velocity field model must be set before prediction, "
                "please provide cell_emb, cond_emb, dna_one_hot."
            )
        else:
            vf_model = self.vf_model

        vt = vf_model(t=t, x_t=x_0).float()
        return vt


class BorzoiLoRAFlowPredictorFP:
    def __init__(
        self,
        model: BorzoiLoRA,
        cell_emb: torch.Tensor = None,
        cond_emb: torch.Tensor = None,
        dna_one_hot: torch.Tensor = None,
        trange: torch.Tensor = None,
        steps: int = 100,
        flow_sigma: float = 1,
    ):
        self.model = model
        self.vf_model = self._set_vf_model(cell_emb, cond_emb, dna_one_hot)
        if trange is None:
            trange = [0, 0.999]
        tmin, tmax = trange
        self.ts = torch.linspace(tmin, tmax, steps, dtype=torch.float32)
        self.dt = self.ts[1] - self.ts[0]
        self.flow_sigma = flow_sigma

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

    def _step(self, vf_model, t, xt):
        vt = vf_model(t, xt)
        sigma = self.flow_sigma * (1 - t)
        dt = self.dt

        mu = xt + vt * dt  # add drift
        xt = mu + sigma * torch.randn_like(mu) * dt.sqrt()  # add score
        return xt, mu

    def predict(self, x_0, cell_emb=None, cond_emb=None, dna_one_hot=None):
        """
        Predict the trajectory of the flow given an initial condition.
        """
        xt = x_0
        if self.vf_model is None:
            vf_model = self._set_vf_model(cell_emb, cond_emb, dna_one_hot)
            assert vf_model is not None, (
                "Velocity field model must be set before prediction, "
                "please provide cell_emb, cond_emb, dna_one_hot."
            )
        else:
            vf_model = self.vf_model
        ones = torch.ones_like(xt)

        for tscalar in self.ts:
            t = tscalar * ones
            xt, mu = self._step(vf_model, t, xt)
        return mu
