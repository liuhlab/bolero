import warnings
from collections.abc import Sequence
from typing import Optional

import torch
from anndata import AnnData
from scvi import settings
from scvi.data._utils import _is_minified
from scvi.model import SCVI

from bolero.tl.model.gene.scvivq.module._vae import VQVAE


class SCVIVQ(SCVI):
    _module_cls = VQVAE

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVI,
        adata: AnnData | None = None,
        **scvivq_kwargs,
    ):
        """Initialize scanVI model with weights from pretrained :class:`~scvi.model.SCVI` model.

        Adapted from scvi.model._scanvi.SCANVI.from_scvi_model

        Parameters
        ----------
        scvi_model
            Pretrained scvi model
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
        scvivq_kwargs
            kwargs for scVIVQ model
        """
        scvi_model._check_if_trained(
            message="Passed in scvi model hasn't been trained yet."
        )

        init_params = scvi_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in scvivq_kwargs.keys():
                warnings.warn(
                    f"Ignoring param '{k}' as it was already passed in to pretrained "
                    f"SCVI model with value {v}.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                del scvivq_kwargs[k]

        if scvi_model.minified_data_type is not None:
            raise ValueError(
                "We cannot use the given scvi model to initialize scanvi because it has a "
                "minified adata."
            )

        if adata is None:
            adata = scvi_model.adata
        else:
            if _is_minified(adata):
                raise ValueError(
                    "Please provide a non-minified `adata` to initialize scanvi."
                )
            # validate new anndata against old model
            scvi_model._validate_anndata(adata)

        scanvi_model = cls(adata, **non_kwargs, **kwargs, **scvivq_kwargs)
        scvi_state_dict = scvi_model.module.state_dict()
        scanvi_model.module.load_state_dict(scvi_state_dict, strict=False)
        scanvi_model.was_pretrained = True

        return scanvi_model

    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        use_vq=True,
        return_ind=False,
    ):
        # in scVI default, give_mean = True, and it will use qz.loc, which is not quantized
        # set give_mean = False will let super().get_latent_representation return the z in inferece output, which is quantized
        latent = super().get_latent_representation(
            adata=adata,
            indices=indices,
            give_mean=give_mean,
            mc_samples=mc_samples,
            batch_size=batch_size,
            return_dist=False,
        )
        if use_vq and give_mean:
            with torch.no_grad():
                latent = torch.from_numpy(latent).to(self.module.device)
                latent, vq_dict = self.module.z_encoder._vq_forward(latent)
                vq_ind = vq_dict["embed_ind"].cpu().numpy()
                latent = latent.cpu().numpy()
        if return_ind and use_vq:
            return latent, vq_ind
        else:
            return latent
