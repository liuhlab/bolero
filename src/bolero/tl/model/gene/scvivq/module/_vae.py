"""
Adapted from https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_vae.py
"""

from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
from scvi._types import Tunable
from scvi.module._vae import VAE
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import Encoder
from torch import nn
from torch.distributions import Normal
from vector_quantize_pytorch import VectorQuantize

from bolero.tl.model.gene.scvivq.utils import get_init_signature


# Encoder
class VQEncoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.
    Uses vector quantization to discretize the latent space.

    Parameters
    ----------
    encoder: scvi.nn.Encoder
        encoder network from scvi
    vq_kwargs: dict
        Keyword arguments for the VectorQuantize module
    """

    def __init__(
        self,
        encoder: Encoder,
        vq_kwargs: dict,
    ):
        super().__init__()
        self.vq_class = vq_kwargs.pop("vq_class", VectorQuantize)
        self.vq_loss_breakdown = vq_kwargs.pop("loss_breakdown", True)

        if "return_loss_breakdown" not in get_init_signature(self.vq_class.forward):
            self.vq_loss_breakdown = False

        self.vq_init_kwargs = get_init_signature(self.vq_class.__init__)
        self.vq_init_kwargs.update(vq_kwargs)
        self.vq = self.vq_class(**self.vq_init_kwargs)

        self.encoder = encoder
        return

    def forward(self, x: torch.Tensor, *cat_list: int, return_vq: bool = False):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Applies vector quantization to the latent space
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal
            \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample
        """
        encoder = self.encoder

        # Parameters for latent distribution
        q = encoder.encoder(x, *cat_list)

        q_m = encoder.mean_encoder(q)
        q_v = encoder.var_activation(encoder.var_encoder(q)) + encoder.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = encoder.z_transformation(dist.rsample())

        # Vector quantization
        q_latent, _vq_dict = self._vq_forward(latent)

        if return_vq:
            if encoder.return_dist:
                return dist, q_latent, _vq_dict
            return q_m, q_v, q_latent, _vq_dict
        else:
            if encoder.return_dist:
                return dist, q_latent
            return q_m, q_v, q_latent

    def _vq_forward(self, z):
        if self.vq_loss_breakdown:
            qz, embed_ind, commit_loss, loss_breakdown = self.vq(
                z, return_loss_breakdown=True
            )
            loss_breakdown = loss_breakdown._asdict()
        else:
            qz, embed_ind, commit_loss = self.vq(z)
            loss_breakdown = {}

        if commit_loss.numel() > 1:
            commit_loss = commit_loss.mean()

        _vq_dict = {
            "embed_ind": embed_ind,
            "commit_loss": commit_loss.squeeze(),
            "loss_breakdown": loss_breakdown,
        }
        return qz, _vq_dict

    def forward_n_samples(self, qz, n_samples, return_vq: bool = False):
        """Sample n_samples from the latent space and get quantized latent vector."""
        untran_z = qz.sample((n_samples,))
        z = self.z_encoder.z_transformation(untran_z)
        q_latent, _vq_dict = self._vq_forward(z)

        if return_vq:
            return q_latent, _vq_dict
        else:
            return q_latent


class VQVAE(VAE):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: Tunable[float] = 0.1,
        dispersion: Tunable[
            Literal["gene", "gene-batch", "gene-label", "gene-cell"]
        ] = "gene",
        log_variational: Tunable[bool] = True,
        gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        encode_covariates: Tunable[bool] = False,
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: Tunable[bool] = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Tunable[Callable] = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
        vq_kwargs: dict = None,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            var_activation=var_activation,
            extra_encoder_kwargs=extra_encoder_kwargs,
            extra_decoder_kwargs=extra_decoder_kwargs,
        )

        _default_vq_kwargs = {
            "dim": n_latent,
            "codebook_dim": int(n_latent * 2),
            "layernorm_after_project_in": True,
            "kmeans_init": True,
            "heads": 1,  # total codebook input dim is codebook_dim * heads
            "codebook_size": 256,
            "decay": 0.8,
            "commitment_weight": 1,
            "orthogonal_reg_weight": 0,
            "orthogonal_reg_active_codes_only": False,
            "orthogonal_reg_max_codes": None,
            "codebook_diversity_loss_weight": 0.0,
            "codebook_diversity_temperature": 100.0,
            "straight_through": False,
            "reinmax": False,
            "stochastic_sample_codes": False,
            "sample_codebook_temp": 1.0,
            "loss_breakdown": True,
            "loss_weight": 10.0,  # overall VQ loss weight before adding to the VAELoss
        }
        if vq_kwargs is not None:
            _default_vq_kwargs.update(vq_kwargs)
        vq_kwargs = _default_vq_kwargs

        self.vq_loss_weight = vq_kwargs.pop("loss_weight")
        self.z_encoder = VQEncoder(
            encoder=self.z_encoder,
            vq_kwargs=vq_kwargs,
        )

    @auto_move_data
    def _regular_inference(
        self,
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        qz, z, _vq_dict = self.z_encoder(
            encoder_input, batch_index, *categorical_input, return_vq=True
        )

        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            z, _vq_dict = self.z_encoder.forward_n_samples(
                qz, n_samples, return_vq=True
            )

            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        outputs = {"z": z, "qz": qz, "ql": ql, "library": library, **_vq_dict}
        return outputs

    def _cached_inference(self, *args, **kwargs):
        raise NotImplementedError("Cached inference not implemented for VQVAE")

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Computes the loss function for the model's VQ part."""
        loss_output: LossOutput = super().loss(
            tensors=tensors,
            inference_outputs=inference_outputs,
            generative_outputs=generative_outputs,
            kl_weight=kl_weight,
        )
        loss = loss_output.loss
        reconst_loss = loss_output.reconstruction_loss
        kl_local = loss_output.kl_local

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
