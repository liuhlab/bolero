"""
This POISSONANVI is adapted from the scvi-tools library.

SCANVI class
https://github.com/scverse/scvi-tools/blob/0142c4ad704efaec0063c4275aa0940a053d56c2/src/scvi/model/_scanvi.py
"""

import torch
from anndata import AnnData
from scvi.external.poissonvi import POISSONVI
from scvi.model._scanvi import SCANVI


class POISSONANVI(SCANVI):
    """
    Single-cell annotation using variational inference and PoissonVI as base model.

    For more details, see SCANVI class in scvi-tools.
    """

    def __init__(self, *args, **model_kwargs):
        # to be consistent with POISSONVI
        model_kwargs["gene_likelihood"] = "poisson"
        model_kwargs["use_batch_norm"] = "none"
        model_kwargs["use_layer_norm"] = "both"
        model_kwargs["extra_encoder_kwargs"] = {"activation_fn": torch.nn.LeakyReLU}
        model_kwargs["extra_decoder_kwargs"] = {"activation_fn": torch.nn.LeakyReLU}

        super().__init__(*args, **model_kwargs)

    @classmethod
    def from_poissonvi_model(
        cls,
        poissonvi_model: POISSONVI,
        unlabeled_category: str,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        **scanvi_kwargs,
    ):
        """Initialize scanVI model with weights from pretrained :class:`~scvi.model.SCVI` model.

        Parameters
        ----------
        poissonvi_model
            Pretrained poissonvi model
        labels_key
            key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the SCVI model. If None, uses the `labels_key` used to
            setup the SCVI model. If that was None, and error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with scvi.
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        scanvi_kwargs
            kwargs for scANVI model
        """
        poissonvi_model.minified_data_type = None
        n_hidden = poissonvi_model.module.z_encoder.encoder.fc_layers[0][0].out_features
        n_latent = poissonvi_model.module.z_encoder.mean_encoder.out_features
        poissonvi_model.init_params_["non_kwargs"]["n_hidden"] = n_hidden
        poissonvi_model.init_params_["non_kwargs"]["n_latent"] = n_latent

        scanvi_model = super().from_scvi_model(
            poissonvi_model,
            unlabeled_category=unlabeled_category,
            labels_key=labels_key,
            adata=adata,
            **scanvi_kwargs,
        )

        # some hacky stuff to get the right parameters for scVI's BaseModelClass.load function
        # model.init_params_ = {
        #     "kwargs": {
        #         "model_kwargs": {...},
        #     },
        #    "non_kwargs": {
        #        "n_hidden": __,
        #        "n_latent": __,
        #        ...
        #    },
        # }
        scanvi_model.init_params_["non_kwargs"].update(
            {
                k: v
                for k, v in poissonvi_model.init_params_["non_kwargs"].items()
                if k not in scanvi_model.init_params_["kwargs"]["model_kwargs"]
            }
        )
        return scanvi_model
