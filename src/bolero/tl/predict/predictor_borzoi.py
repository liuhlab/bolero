import torch
from einops import rearrange

from bolero.tl.model.borzoi.model import Borzoi
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.utils import understand_regions

from .predictor import GenericPredictor


class BorzoiPredictor(GenericPredictor):
    def __init__(self, config):
        super().__init__(config, BorzoiLoRA)

    def _create_model(self) -> BorzoiLoRA:
        model: Borzoi | BorzoiLoRA = super()._create_model()
        if isinstance(model, BorzoiLoRA):
            model.convert_to_lora()
        return model

    def _lora_model_prediction_step(
        self,
        batch,
        dna_key="__dna__",
        embedding_key="__embedding__",
        batch_size=16,
    ):
        """
        Forward pass through the model.
        """
        # prepare data
        dna = batch[dna_key]
        emb = batch[embedding_key]

        n_emb = emb.shape[0]
        n_region = dna.shape[0]

        emb_idx = torch.arange(n_emb).repeat_interleave(n_region)
        region_idx = torch.arange(n_region).repeat(n_emb)

        pred_col = []
        for i in range(0, len(emb_idx), batch_size):
            emb_mini_batch = emb[emb_idx[i : i + batch_size]]
            dna_mini_batch = dna[region_idx[i : i + batch_size]]

            with self._autocast_context():
                y_pred_mini_batch = self.model(dna_mini_batch, embedding=emb_mini_batch)
                pred_col.append(y_pred_mini_batch)
        y_pred = torch.cat(pred_col, dim=0)
        # reshape to (n_region, n_emb, n_pseudobulk)
        y_pred = rearrange(
            y_pred,
            "(n_region n_emb) seq_len -> n_region n_emb seq_len",
            n_region=n_region,
            n_emb=n_emb,
        )

        batch["__ypred__"] = y_pred.detach()
        return batch

    def get_prediction_dataloader(
        self,
        regions="test_regions",
        pseudobulk_ids=None,
        add_true_data=False,
        dna_key="dna",
        embedding_key="embedding",
        batch_size=16,
    ):
        """
        Get the dataloader for prediction.
        """
        if isinstance(regions, str) and regions == "test_regions":
            regions = self.get_fold_regions(test_only=True)
        else:
            regions = understand_regions(regions)
        regions: list[str] = self._valid_and_sort_regions(regions, return_list=True)

        da_prefix = self.datamanager._get_data_prefixs()
        assert (
            len(da_prefix) == 1
        ), "Currently only one data prefix is supported for prediction."
        data_key = da_prefix[0]

        def _collate_fn(batch, add_data=add_true_data):
            # rename keys
            batch["__dna__"] = batch.pop(dna_key)
            batch["__embedding__"] = batch.pop(embedding_key)

            if add_data:
                # coverage normalize true data
                data = batch.pop(data_key)
                logscale = batch[f"{data_key}:cov_scale"]
                data = data / 2 ** logscale[None, :, None]
                batch["__ytrue__"] = data
            return batch

        dataloader = self.datamanager.get_dataloader(
            regions=regions,
            batch_size=batch_size,
            add_dna=True,
            add_data=add_true_data,
            pseudobulk_subset=pseudobulk_ids,
            pseudobulk_info_keys=["cov_scale", embedding_key],
            collate_fn=_collate_fn,
        )
        return dataloader
