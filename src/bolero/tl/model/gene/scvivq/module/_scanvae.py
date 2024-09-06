"""
Adapted from https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_scanvae.py
"""

import torch
from scvi.module import SCANVAE
from scvi.module.base import LossOutput
from torch.distributions import Distribution

from ._vae import VQVAE


class VQSCANVAE(VQVAE, SCANVAE):
    """Single-cell annotation using variational inference.

    This is an implementation of the scANVI model described in :cite:p:`Xu21`,
    inspired from M1 + M2 model, as described in (https://arxiv.org/pdf/1406.5298.pdf).

    The z_encoder contains an VQ encoder applied on the latent embedding of the data.

    see VQVAE.__init__ for parameters about VQ
    """

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_ouputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
    ):
        loss_output = SCANVAE.loss(
            self,
            tensors,
            inference_outputs,
            generative_ouputs,
            kl_weight,
            labelled_tensors,
            classification_ratio,
        )
        # LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_divergence)
        loss = loss_output.loss
        reconst_loss = loss_output.reconstruction_loss
        kl_local = loss_output.kl_local

        # vq loss
        vq_loss = inference_outputs["commit_loss"] * self.vq_loss_weight
        loss += vq_loss

        extra_metrics = {
            "commit_loss": inference_outputs["commit_loss"].clone().detach()
        }
        if len(inference_outputs["loss_breakdown"]) > 0:
            for k, v in inference_outputs["loss_breakdown"].items():
                extra_metrics[k] = v.clone().detach()

        new_loss_output = LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
        )
        return new_loss_output
