import time
from typing import Generator

import numpy as np
import pandas as pd
import pyranges as pr
import torch
from einops import rearrange

from bolero.tl.model.borzoi.model import Borzoi
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.utils import understand_regions

from .callbacks import CALLBACK_NAME_TO_CLASS
from .datamanager import GenericGenomeDataManager
from .predictor import GenericPredictor


class BorzoiPredictor(GenericPredictor):
    def __init__(self, config):
        super().__init__(config, BorzoiLoRA)
        self._create_datamanager()

        peak_path = self.config.get("peak_path", None)
        if peak_path is not None:
            peak_df = pr.read_bed(peak_path).df
        else:
            peak_df = None
        self.peak_df: pd.DataFrame | None = peak_df

    def _create_datamanager(self):
        config = self.config
        genome = config["genome"]
        db_path = config["dataset_path"]
        pseudobulk_records_path = config["pseudobulk_records_path"]
        parallel = config.get("parallel", 8)

        dm = GenericGenomeDataManager(genome=genome)
        _ = dm.genome.genome_one_hot
        dm.add_pseudobulk_records(pseudobulk_records_path)
        dm.add_parquet_dataset("parquet", db_path, parallel=parallel)
        self.add_datamanager(dm)
        return

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

        emb_idx = torch.arange(n_emb).repeat(n_region)
        # [0, 1, ..., n_emb-1, ..., 0, 1, ..., n_emb-1]
        region_idx = torch.arange(n_region).repeat_interleave(n_emb)
        # [0, ..., 0, 1, ..., 1, ..., n_region-1, ..., n_region-1]

        pred_col = []
        for i in range(0, len(emb_idx), batch_size):
            emb_mini_batch = emb[emb_idx[i : i + batch_size]]
            dna_mini_batch = dna[region_idx[i : i + batch_size]]

            with self._autocast_context():
                y_pred_mini_batch = self.model(dna_mini_batch, embedding=emb_mini_batch)
                pred_col.append(y_pred_mini_batch)
        y_pred = torch.cat(pred_col, dim=0)
        # reshape to (n_region, n_emb, seq_len)
        # here only deal with one modality case
        y_pred = rearrange(
            y_pred,
            "(n_region n_emb) 1 seq_len -> n_region n_emb seq_len",
            n_region=n_region,
            n_emb=n_emb,
        )

        batch["__ypred__"] = y_pred.detach()
        return batch

    def prepare_post_inference_callbacks(
        self, callbacks: str | list[str] | list[tuple[str, dict]]
    ):
        """
        Prepare the post inference callbacks.

        callbacks: list[tuple[str, dict]]
            A list of tuples, where each tuple contains the name of the callback and its arguments.
            The callback name should be one of the keys in CALLBACK_NAME_TO_CLASS.
        """
        if isinstance(callbacks, str):
            callbacks = [callbacks]

        callback_list = []
        for name_and_kwargs in callbacks:
            if isinstance(name_and_kwargs, str):
                name_and_kwargs = (name_and_kwargs, {})
            name, kwargs = name_and_kwargs
            callback = CALLBACK_NAME_TO_CLASS[name](**kwargs)
            callback_list.append(callback)
        return callback_list

    def _get_prediction_callbacks(self):
        callbacks = [("pearsonr", {"output_key": "profile_r"})]
        if self.peak_df is not None:
            peak_level_callbacks = [
                # extract peak data from borzoi regions
                ("extract_peak", {"peak_bed": self.peak_df}),
                # this step adds the peak data to the batch:
                # "__ytrue__:peak" and "__ypred__:peak"
                #
                # calculate peak level profile correlation
                (
                    "pearsonr",
                    {
                        "ytrue_key": "__ytrue__:peak",
                        "ypred_key": "__ypred__:peak",
                        "output_key": "peak_profile_r",
                    },
                ),
                # calculate peak level sample correlation
                (
                    "pearsonr",
                    {
                        "ytrue_key": "__ytrue__:peak",
                        "ypred_key": "__ypred__:peak",
                        "permute": (1, 0),  # (peak, sample) -> (sample, peak)
                        "output_key": "peak_sample_r",
                    },
                ),
            ]
            callbacks.extend(peak_level_callbacks)
        return self.prepare_post_inference_callbacks(callbacks)

    def get_prediction_dataloader(
        self,
        regions="test_regions",
        pseudobulk_ids=None,
        add_true_data=False,
        dna_key="dna",
        embedding_key="embedding",
        batch_size=32,
        verbose=True,
    ) -> Generator:
        """
        Get the dataloader for prediction.
        """
        # 1. Get regions
        if isinstance(regions, str) and regions == "test_regions":
            regions = self.get_fold_regions(test_only=True)
        else:
            regions = understand_regions(regions)
        regions: list[str] = self._valid_and_sort_regions(regions, return_list=True)

        # 2. Get data loader
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
                data = data.float()

                # crop to the same size as the model output
                crop_radius = (data.shape[-1] - self.model.crop_to_length) // 2
                batch["__ytrue__"] = data[..., crop_radius:-crop_radius]
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
        pid_array = (
            dataloader.dataset.pseudobulk_ids
            if pseudobulk_ids is None
            else pseudobulk_ids
        )
        pid_array = np.array(pid_array)

        # 3. Get callbacks
        callbacks = self._get_prediction_callbacks()

        # trigger model load
        _ = self.model

        # 4. Inference and callback
        timer = {"infer": 0, "callback": 0, "total": 0, "counter": 0}
        start = time.time()
        for batch in dataloader:
            with torch.inference_mode():
                t = time.time()
                batch = self._lora_model_prediction_step(
                    batch,
                    dna_key="__dna__",
                    embedding_key="__embedding__",
                    batch_size=batch_size,
                )
                timer["infer"] += time.time() - t

                t = time.time()
                batch = self.apply_callbacks(batch, callbacks)
                timer["callback"] += time.time() - t
                timer["counter"] += 1

            batch["pseudobulk_ids"] = pid_array
            yield batch
        timer["total"] = time.time() - start

        if verbose:
            print(
                f"Total time: {timer['total']:.2f}s, "
                f"Inference time: {timer['infer']:.2f}s or "
                f"{timer['infer']/timer['counter']:.3f}s per batch, "
                f"Callback time: {timer['callback']:.2f}s or "
                f"{timer['callback']/timer['counter']:.3f}s per batch, "
                f"(total {timer['counter']} batches)"
            )
