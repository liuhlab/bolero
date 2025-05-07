from typing import Any, Optional

import torch
from torch import nn
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
)
from torchdiffeq import odeint

from ._velocity_field import ConditionalVelocityField


class _VelocityFieldWrapperForODE(nn.Module):
    def __init__(self, vf: ConditionalVelocityField, condition, encoder_noise):
        super().__init__()
        self.vf = vf
        self.condition = condition
        self.encoder_noise = encoder_noise

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x_t.shape[0], 1)

        v_t, *_ = self.vf(
            t=t,
            x_t=x_t,
            cond=self.condition,
            encoder_noise=self.encoder_noise,
        )
        return v_t


class OTFlowMatching:
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        probability_path
            Probability path between the source and the target distributions.
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`cellflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        kwargs
            Keyword arguments for :meth:`cellflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        matcher_class: str = "otcfm",
        matcher_kwargs: dict[str, Any] = None,
        ode_solver_kwargs: dict[str, Any] = None,
        device: torch.device = None,
        seed: Optional[int] = None,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization

        if matcher_class == "otcfm":
            matcher_class = ExactOptimalTransportConditionalFlowMatcher
        elif matcher_class == "cfm":
            matcher_class = ConditionalFlowMatcher
        elif matcher_class == "sbcfm":
            matcher_class = SchrodingerBridgeConditionalFlowMatcher
        else:
            raise ValueError(
                f"Unknown matcher class {matcher_class}. "
                "Use 'otcfm', 'cfm', or 'sbcfm'."
            )
        _matcher_kwargs = {
            "sigma": 0.0,
        }
        if matcher_kwargs is not None:
            _matcher_kwargs.update(matcher_kwargs)
        self.flow_matcher: ConditionalFlowMatcher = matcher_class(**_matcher_kwargs)

        _ode_solver_kwargs = {
            "method": "dopri5",
            "rtol": 1e-5,
            "atol": 1e-5,
        }
        if ode_solver_kwargs is not None:
            _ode_solver_kwargs.update(ode_solver_kwargs)
        self.ode_solver_kwargs = _ode_solver_kwargs

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.vf.to(device)

        if seed is None:
            seed = torch.seed()
        self.generator = torch.Generator(device).manual_seed(seed)

    def _calc_encoder_loss(
        self,
        mean_cond: torch.Tensor,
        logvar_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Regularization term for the condition encoder."""
        if self.condition_encoder_mode == "deterministic":
            if self.condition_encoder_regularization > 0:
                loss = 0.5 * torch.mean(mean_cond**2)
            else:
                loss = 0.0
        elif self.condition_encoder_mode == "stochastic":
            if self.condition_encoder_regularization > 0:
                mean_reg = 0.5 * torch.mean(mean_cond**2)
                var_reg = -0.5 * torch.mean(1 + logvar_cond - torch.exp(logvar_cond))
                loss = mean_reg + var_reg
            else:
                loss = 0.0
        else:
            raise ValueError(
                f"Unknown condition encoder mode {self.condition_encoder_mode}. "
                "Use 'deterministic' or 'stochastic'."
            )
        return loss

    def loss_fn(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        conditions: dict[str, torch.Tensor],
        encoder_noise: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> torch.Tensor:
        # The flow matcher class from torchcfm implements the algorithm to
        # sample locations and conditional flows from a random or given time
        # Optimal Transport (OT) is implemented inside the flow matcher class
        with torch.no_grad():
            t, x_t, u_t = self.flow_matcher.sample_location_and_conditional_flow(
                x0=x_0, x1=x_1, t=t
            )
            t = t.unsqueeze(-1)  # [batch_size, 1]

        v_t, mean_cond, logvar_cond = self.vf(
            t=t,
            x_t=x_t,
            cond=conditions,
            encoder_noise=encoder_noise,
        )

        flow_matching_loss = torch.mean((v_t - u_t) ** 2)
        encoder_loss = self._calc_encoder_loss(
            mean_cond=mean_cond,
            logvar_cond=logvar_cond,
        )
        return flow_matching_loss + encoder_loss

    def step_fn(
        self,
        batch: dict[str, torch.Tensor],
    ) -> float:
        """Single step function of the solver.

        Parameters
        ----------
        batch
            Data batch with keys ``x0``, ``x1``, and
            optionally ``condition``.

        Returns
        -------
        Loss value.
        """
        x_0, x_1 = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")

        n = x_0.shape[0]
        encoder_noise = torch.randn(
            n,
            self.vf.condition_embedding_dim,
            generator=self.generator,
            device=self.device,  # e.g. "cpu" or "cuda"
        )
        # TODO: test whether it's better to sample the same noise for all samples or different ones

        loss = self.loss_fn(
            x_0=x_0,
            x_1=x_1,
            conditions=condition,
            encoder_noise=encoder_noise,
            t=None,
        )
        return loss

    @torch.no_grad()
    def get_condition_embedding(
        self, condition: dict[str, torch.Tensor], return_as_numpy=True
    ) -> torch.Tensor:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.

        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.get_condition_embedding(condition)

        if return_as_numpy:
            return cond_mean.cpu().numpy(), cond_logvar.cpu().numpy()
        return cond_mean, cond_logvar

    @torch.no_grad()
    def predict(
        self,
        x_0: torch.Tensor,
        condition: dict[str, torch.Tensor],
        t_range: tuple[float, float] = (0.0, 1.0),
        t_steps: int = 512,
        return_traj: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Predict the translated source ``x`` under condition ``condition``.

        Parameters
        ----------
        x_0 : Tensor
            Input data of shape [batch_size, ...].
        condition : dict
            Dictionary of condition tensors.
        kwargs : Any
            Extra args for torchdiffeq.odeint

        Returns
        -------
        x_pred : Tensor
            Transformed data after solving ODE.
        """
        device = self.device
        batch_size = x_0.shape[0]

        # Prepare encoder noise
        if self.condition_encoder_mode == "deterministic":
            encoder_noise = torch.zeros(
                (batch_size, self.vf.condition_embedding_dim), device=device
            )
        else:
            encoder_noise = torch.randn(
                (batch_size, self.vf.condition_embedding_dim),
                generator=self.generator,
                device=device,
            )

        # Define ODE function
        vf_wrapper = _VelocityFieldWrapperForODE(self.vf, condition, encoder_noise)

        t_span = torch.linspace(*t_range, t_steps, device=device)
        x_pred_traj = odeint(
            func=vf_wrapper,
            y0=x_0,
            t=t_span,
            **self.ode_solver_kwargs,
            **kwargs,
        )
        if return_traj:
            return x_pred_traj

        else:
            return x_pred_traj[-1]

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
