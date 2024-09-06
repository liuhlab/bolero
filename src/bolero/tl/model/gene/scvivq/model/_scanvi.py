from anndata import AnnData
from scvi.model import SCANVI

from bolero.tl.model.gene.scvivq.module._scanvae import VQSCANVAE

from ._scvi import SCVIVQ


class SCANVIVQ(SCANVI):
    _module_cls = VQSCANVAE

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVIVQ,
        unlabeled_category: str,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        **scanvi_kwargs,
    ):
        assert isinstance(
            scvi_model, SCVIVQ
        ), "Passed in model is not an instance of SCVIVQ."

        return SCANVI.from_scvi_model(
            scvi_model,
            unlabeled_category,
            labels_key,
            adata,
            **scanvi_kwargs,
        )
