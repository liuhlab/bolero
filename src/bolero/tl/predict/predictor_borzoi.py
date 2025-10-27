import gc
import json
import pathlib
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from shutil import rmtree
from typing import Generator

import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import torch
import xarray as xr
from einops import rearrange

from bolero.tl.model.borzoi.dataset_multi import DatasetRecordManager
from bolero.tl.model.borzoi.model import Borzoi
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA, BorzoiLoRAMulti
from bolero.tl.model.borzoi.utils import BorzoiGeneQTLRegions
from bolero.tl.pseudobulk.paired_pseudobulk import PAIRED_PSEUDOBULKER_CLS_DICT
from bolero.utils import understand_regions

from .datamanager import GenericGenomeDataManager
from .predictor import GenericPredictor
from .utils import gather_gene_data, gather_peak_data, load_config


def _get_cur_bid(batch_dir):
    batch_dir = pathlib.Path(batch_dir)
    bids = []
    for path in batch_dir.glob("*.joblib.gz"):
        bids.append(int(path.name.split(".")[0].split("_")[1]))
    return max(bids) + 1 if len(bids) > 0 else 0


def _clip(data: torch.Tensor, clip_length: int) -> torch.Tensor:
    seq_len = data.shape[-1]
    radius = clip_length // 2
    start = seq_len // 2 - radius
    end = start + clip_length
    return data[..., start:end]


def _clip_at_center(
    data, clip_length
) -> torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor]:
    if isinstance(data, dict):
        return {k: _clip(v, clip_length) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_clip(d, clip_length) for d in data]
    else:
        return _clip(data, clip_length)


class BorzoiInputXGradient:
    def __init__(
        self,
        model,
        peak_length=512,
        attr_length=1024,
        model_dna_length=524288,
        model_resolution=32,
    ):
        self.model = model
        self.model_dna_length = model_dna_length
        self.model_resolution = model_resolution
        self.peak_length = peak_length
        self.peak_bins = peak_length // self.model_resolution
        self.attr_length = attr_length

        def _forward_hook(*args):
            dna, *signal = args
            if len(signal) == 0:
                signal = None
            else:
                signal = signal[0]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if signal is None:
                    outputs = model(x=dna)  # Shape: (batch_size, 1, 16352)
                else:
                    outputs = model(x=dna, signal=signal)
            # Shape: (batch_size, )
            outputs = _clip_at_center(outputs, self.peak_bins)
            peak_outputs = outputs.sum(dim=-1)
            return peak_outputs

        from captum.attr import InputXGradient

        self.attributor = InputXGradient(_forward_hook)

    def __call__(
        self,
        *args,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Compute the input gradients for the given DNA sequence and other inputs.
        """
        if len(args) == 1:
            # only one input, assume it is the DNA sequence
            args = (args[0],)
        else:
            # two inputs, assume the first is DNA and the second is signal
            args = tuple(args)
        args = tuple([a for a in args if a is not None])

        attr_data = self.attributor.attribute(inputs=args)
        attr_data = _clip_at_center(attr_data, self.attr_length)
        return attr_data


class BorzoiGeneInputXGradient:
    def __init__(
        self,
        model,
        model_dna_length=524288,
        model_resolution=32,
    ):
        self.model = model
        self.model_dna_length = model_dna_length
        self.model_resolution = model_resolution

        def _forward_hook(
            *args,
        ):
            dna, gene_mask, *signal = args
            if len(signal) == 0:
                signal = None
            else:
                signal = signal[0]
            kwargs = {
                "x": dna,
                "gene_mask": gene_mask,
                "signal": signal,
                "crop": False,
                "return_dna_embedding": False,
                "embedding": None,
            }
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # take gene output only
                _, outputs = model.forward_gene(**kwargs)  # Shape: (batch_size, 1, 1)

            outputs = rearrange(outputs, "bs 1 1 -> bs")
            return outputs

        from captum.attr import InputXGradient

        self.attributor = InputXGradient(_forward_hook)

    def __call__(self, *args) -> torch.Tensor | list[torch.Tensor]:
        """
        Compute the input gradients for the given DNA sequence and other inputs.
        """
        args = tuple(args)
        args = tuple([a for a in args if a is not None])
        attr_data = self.attributor.attribute(inputs=args)
        return attr_data


class BorzoiPredictor(GenericPredictor):
    model_class = BorzoiLoRA

    def __init__(self, config):
        super().__init__(config, self.model_class)
        self._create_datamanager()

        peak_path = self.config.get("peak_path", None)
        if peak_path is not None:
            peak_df = pr.read_bed(peak_path).df
        else:
            peak_df = None
        self.peak_df: pd.DataFrame | None = peak_df

        self._callbacks = []
        self.qtl_manager = None

    def register_qtl_manager(self, qtl_table, qtl_type, channel_weights_path=None):
        """
        Register the QTL manager with the given QTL table.

        This is required in qtl task
        """
        if qtl_type in ("caqtl", "caqtl_multihead"):
            if qtl_type == "caqtl_multihead":
                from .task_manager import caQTLMultiheadManager

                channel_weights = joblib.load(channel_weights_path)
                self.qtl_manager = caQTLMultiheadManager(
                    qtl_table,
                    channel_weights=channel_weights,
                    qtl_type="caqtl_multihead",
                )
            else:
                from .task_manager import caQTLManager

                self.qtl_manager = caQTLManager(qtl_table, qtl_type="caqtl")
        elif qtl_type == "eqtl":
            from .task_manager import eQTLManager

            self.qtl_manager = eQTLManager(qtl_table)
        else:
            raise ValueError(f"Invalid qtl type: {qtl_type}")
        return

    def register_peak_manager(self, peak_table):
        """
        Register the peak manager with the given peak table.

        This is required in peak task
        """
        from .task_manager import PeakManager

        self.peak_manager = PeakManager(peak_table, resolution=32)
        return

    def _create_datamanager(self):
        config = self.config
        genome = config["genome"]
        db_path = config["dataset_path"]
        pseudobulk_records_path = config["pseudobulk_records_path"]
        parallel = config.get("parallel", 8)

        dm = GenericGenomeDataManager(genome=genome)
        dm.add_pseudobulk_records(pseudobulk_records_path)
        if not self._embedding_only_mode:
            dm.add_parquet_dataset("parquet", db_path, parallel=parallel)
        # add bigwig if any
        bw_path_dict = config.get("bigwig_paths", {})
        for name, path in bw_path_dict.items():
            dm.add_bigwig_dataset(dataset_name=name, dataset_path=path, resolution=32)

        self.add_datamanager(dm)
        return

    def _create_model(self) -> BorzoiLoRA:
        model: Borzoi | BorzoiLoRA = super()._create_model()
        if isinstance(model, BorzoiLoRA):
            model.convert_to_lora()
        return model

    def _model_prediction_step(
        self,
        batch,
        dna_key="__dna__",
        embedding_key="__embedding__",
        ypred_key="__ypred__",
        batch_size=16,
        crop=True,
        _gene_mode=False,
    ):
        """
        Forward pass through the model.
        """
        # prepare data
        dna = batch[dna_key]
        emb = batch[embedding_key]

        if _gene_mode:
            gene_mask = self.borzoi_gene_regions.get_gene_mask(batch, dna=dna)

        n_emb = emb.shape[0]
        n_region = dna.shape[0]

        emb_idx = torch.arange(n_emb).repeat(n_region)
        # [0, 1, ..., n_emb-1, ..., 0, 1, ..., n_emb-1]
        region_idx = torch.arange(n_region).repeat_interleave(n_emb)
        # [0, ..., 0, 1, ..., 1, ..., n_region-1, ..., n_region-1]

        pred_col = []
        gene_count_col = []
        for i in range(0, len(emb_idx), batch_size):
            emb_mini_batch = emb[emb_idx[i : i + batch_size]]
            dna_mini_batch = dna[region_idx[i : i + batch_size]]

            with self._autocast_context():
                if _gene_mode:
                    gene_mask_mini_batch = gene_mask[region_idx[i : i + batch_size]]
                    y_pred_mini_batch, gene_count_mini_batch = self.model.forward_gene(
                        dna_mini_batch,
                        embedding=emb_mini_batch,
                        crop=crop,
                        gene_mask=gene_mask_mini_batch,
                    )
                    gene_count_col.append(gene_count_mini_batch)
                else:
                    y_pred_mini_batch = self.model(
                        dna_mini_batch,
                        embedding=emb_mini_batch,
                        crop=crop,
                    )
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

        batch[ypred_key] = y_pred.detach()

        if _gene_mode:
            gene_count = torch.cat(gene_count_col, dim=0)
            gene_count = rearrange(
                gene_count,
                "(n_region n_emb) 1 1 -> n_region n_emb",
                n_region=n_region,
                n_emb=n_emb,
            )
            batch[f"{ypred_key}:gene_count"] = gene_count.detach()
        return batch

    def _model_gene_count_prediction_step(self, *args, **kwargs):
        return self._model_prediction_step(*args, _gene_mode=True, **kwargs)

    def _model_qtl_step(self, batch, batch_size):
        # turn "__dna__" into "__dna__:ref" and "__dna__:alt"
        qtl = self.qtl_manager
        assert qtl is not None, "QTL manager is not registered."
        batch = qtl.mutate_dna(batch)
        mutation_cols = ["ref", "alt"]
        with torch.inference_mode():
            for mutation_col in mutation_cols:
                if qtl.qtl_type == "caqtl":
                    batch = self._model_prediction_step(
                        batch,
                        dna_key=f"__dna__:{mutation_col}",
                        batch_size=batch_size,
                        ypred_key=f"__ypred__:{mutation_col}",
                        # QTL manager takes full borzoi region
                        crop=False,
                    )
                    batch = qtl.get_peak_sum(
                        batch,
                        ypred_key=f"__ypred__:{mutation_col}",
                    )
                elif qtl.qtl_type == "eqtl":
                    batch = self._model_gene_count_prediction_step(
                        batch,
                        dna_key=f"__dna__:{mutation_col}",
                        batch_size=batch_size,
                        ypred_key=f"__ypred__:{mutation_col}",
                        crop=False,
                    )
                else:
                    raise ValueError(f"Invalid qtl type: {qtl.qtl_type}")
        # add peak information to the batch
        batch = qtl.add_qtl_info(batch)
        return batch

    def _model_peak_step(self, batch, batch_size):
        # turn "__dna__" into "__dna__:ref" and "__dna__:alt"
        peaker = self.peak_manager
        assert peaker is not None, "Peak manager is not registered."

        with torch.inference_mode():
            batch = self._model_prediction_step(
                batch,
                dna_key="__dna__",
                batch_size=batch_size,
                ypred_key="__ypred__",
                # QTL manager takes full borzoi region
                crop=False,
            )

            batch = peaker.get_peak_sum(
                batch,
                ypred_key="__ypred__",
            )
        # add peak information to the batch
        batch = peaker.add_peak_info(batch)
        return batch

    def _get_post_attribution_callbacks(self):
        return []

    def _get_post_prediction_callbacks(self, mode="prediction"):
        callbacks = [
            # calc pearsonr on last dim (seq_len)
            # the metrics callback class always calculate on the first dim
            # permute parameter puts seq_len to first
            (
                "pearsonr",
                {
                    "output_key": "profile_pearsonr",
                    "permute": (2, 0, 1),
                },
            ),
            (
                "r2_score",
                {
                    "output_key": "profile_r2",
                    "permute": (2, 0, 1),
                },
            ),
        ]
        if self.peak_df is not None:
            peak_level_callbacks = [
                # extract peak data from borzoi regions
                (
                    "extract_peak",
                    {
                        "peak_bed": self.peak_df,
                        "_is_gene_region": mode == "gene_count_prediction",
                    },
                ),
                # this step adds the peak data to the batch:
                # "__ytrue__:peak" and "__ypred__:peak"
                # calculate peak cumulative profile correlation
                (
                    "pearsonr",
                    {
                        "ytrue_key": "__ytrue__:peak",
                        "ypred_key": "__ypred__:peak",
                        "permute": (1, 0),  # (sample, peak) -> (peak, sample)
                        "output_key": "peak_cum_profile_pearsonr",
                        "cumulative": True,
                    },
                ),
                (
                    "r2_score",
                    {
                        "ytrue_key": "__ytrue__:peak",
                        "ypred_key": "__ypred__:peak",
                        "permute": (1, 0),
                        "output_key": "peak_cum_profile_r2",
                        "cumulative": True,
                    },
                ),
                # calculate peak sample correlation
                (
                    "pearsonr",
                    {
                        "ytrue_key": "__ytrue__:peak",
                        "ypred_key": "__ypred__:peak",
                        "output_key": "peak_sample_pearsonr",
                    },
                ),
                (
                    "r2_score",
                    {
                        "ytrue_key": "__ytrue__:peak",
                        "ypred_key": "__ypred__:peak",
                        "output_key": "peak_sample_r2",
                    },
                ),
            ]
            callbacks.extend(peak_level_callbacks)
        return self._prepare_callbacks(callbacks)

    def _get_post_qtl_callbacks(self):
        callbacks = []
        return self._prepare_callbacks(callbacks)

    def _get_post_callbacks(self, mode="prediction"):
        if mode == "prediction":
            return self._get_post_prediction_callbacks(mode)
        elif mode == "attribution":
            return self._get_post_attribution_callbacks()
        elif mode in ("qtl", "eqtl"):
            return self._get_post_qtl_callbacks()
        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Must be one of 'prediction', 'qtl' or 'attribution'."
            )

    def _create_fn_and_dataloader(
        self,
        dna_key,
        data_key,
        embedding_key,
        regions,
        region_names,
        pseudobulk_ids,
        add_true_data,
        batch_size,
    ):
        if self._embedding_only_mode:
            add_true_data = False

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
            region_names=region_names,
            batch_size=batch_size,
            add_dna=True,
            add_data=add_true_data,
            pseudobulk_subset=pseudobulk_ids,
            pseudobulk_info_keys=["cov_scale", embedding_key],
            collate_fn=_collate_fn,
        )
        return dataloader

    def iter_debug_batch(
        self,
        dna_key="dna",
        embedding_key="embedding",
        pseudobulk_ids=None,
        add_true_data=True,
        batch_size=2,
        mode="prediction",
        regions=None,
        trigger_model=False,
    ):
        """Iterable for debugging"""
        # 1. Get regions
        if regions is None:
            regions = self.get_fold_regions(test_only=True, mode=mode)
        regions, region_names = self._valid_and_sort_regions(regions, return_list=True)

        # 2. Get data loader
        da_prefix = self.datamanager._get_data_prefixs()
        assert (
            len(da_prefix) == 1
        ), "Currently only one data prefix is supported for prediction."
        data_key = da_prefix[0]

        dataloader = self._create_fn_and_dataloader(
            dna_key=dna_key,
            data_key=data_key,
            embedding_key=embedding_key,
            regions=regions,
            region_names=region_names,
            pseudobulk_ids=pseudobulk_ids,
            add_true_data=add_true_data,
            batch_size=batch_size,
        )

        pid_array = (
            self.pseudobulk_manager.pseudobulk_ids
            if pseudobulk_ids is None
            else pseudobulk_ids
        )
        pid_array = np.array(pid_array)

        # 3. Get callbacks
        self._pre_callbacks = self._get_pre_callbacks(mode)
        self._post_callbacks = self._get_post_callbacks(mode)

        # trigger model load
        if trigger_model:
            _ = self.model

        # 4. Inference and callback
        for batch in dataloader:
            batch["pseudobulk_ids"] = pid_array

            # with torch.inference_mode():
            #     batch = self.apply_callbacks(batch, "pre")
            yield batch

    def get_prediction_dataloader(
        self,
        regions,
        batch_dir,
        pseudobulk_ids=None,
        add_true_data=False,
        dna_key="dna",
        embedding_key="embedding",
        batch_size=32,
        verbose=True,
        mode="prediction",
    ) -> Generator:
        """
        Get the dataloader for prediction.
        """
        assert mode in ["prediction", "gene_count_prediction", "qtl", "eqtl", "peak"], (
            f"Invalid mode: {mode}. "
            "Must be one of 'prediction', 'qtl', 'eqtl' or 'peak'."
        )
        # 1. Get regions
        regions, region_names = self._valid_and_sort_regions(
            regions, return_list=True, batch_dir=batch_dir
        )

        # 2. Get data loader
        da_prefix = self.datamanager._get_data_prefixs()
        assert (
            len(da_prefix) == 1
        ), "Currently only one data prefix is supported for prediction."
        data_key = da_prefix[0]

        n_pseudobulks = (
            len(pseudobulk_ids)
            if pseudobulk_ids is not None
            else len(self.pseudobulk_manager.pseudobulk_ids)
        )
        dataloader_batch_size = max(2, int(batch_size / n_pseudobulks * 100))
        dataloader_batch_size = min(16, dataloader_batch_size)
        print(f"Data loader batch size {dataloader_batch_size}")
        dataloader = self._create_fn_and_dataloader(
            dna_key=dna_key,
            data_key=data_key,
            embedding_key=embedding_key,
            regions=regions,
            region_names=region_names,
            pseudobulk_ids=pseudobulk_ids,
            add_true_data=add_true_data,
            batch_size=dataloader_batch_size,
        )

        pid_array = (
            self.pseudobulk_manager.pseudobulk_ids
            if pseudobulk_ids is None
            else pseudobulk_ids
        )
        pid_array = np.array(pid_array)

        # 3. Get callbacks
        self._pre_callbacks = self._get_pre_callbacks(mode)
        self._post_callbacks = self._get_post_callbacks(mode)

        # trigger model load
        _ = self.model

        # 4. Inference and callback
        timer = {"infer": 0, "callback": 0, "total": 0, "counter": 0}
        start = time.time()
        for batch in dataloader:
            batch["pseudobulk_ids"] = pid_array

            with torch.inference_mode():
                # Pre-inference callbacks
                t = time.time()
                batch = self.apply_callbacks(batch, "pre")
                timer["callback"] += time.time() - t

                # Inference step
                t = time.time()
                if mode == "prediction":
                    step_fn = self._model_prediction_step
                elif mode == "gene_count_prediction":
                    step_fn = self._model_gene_count_prediction_step
                elif mode in ("qtl", "eqtl"):
                    step_fn = self._model_qtl_step
                elif mode == "peak":
                    step_fn = self._model_peak_step
                else:
                    raise ValueError(f"Bad mode: {mode}.")
                batch = step_fn(batch, batch_size=batch_size)
                timer["infer"] += time.time() - t

                # Post-inference callbacks
                t = time.time()
                batch = self.apply_callbacks(batch, "post")
                timer["callback"] += time.time() - t
                timer["counter"] += 1
            yield batch
        timer["total"] = time.time() - start

        if verbose:
            print(
                f"Total time: {timer['total']:.2f}s\n"
                f"Inference time: {timer['infer']:.2f}s or "
                f"{timer['infer']/(timer['counter']+1e-5):.3f}s per batch\n"
                f"Callback time: {timer['callback']:.2f}s or "
                f"{timer['callback']/(timer['counter']+1e-5):.3f}s per batch\n"
                f"(total {timer['counter']} batches)\n"
            )

    def _collapse_model(self, emb):
        """Collapse model for inference given an embedding vector."""
        if isinstance(emb, pd.Series):
            emb = torch.from_numpy(emb.values)
        elif isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        emb = emb.cuda().unsqueeze(0)

        emb_model = self.model.collapse_lora(emb)
        emb_model = emb_model.eval().cuda()
        return emb_model

    def _prepare_attr_model(self, batch, mode="attribution") -> torch.nn.Module:
        if mode == "attribution":
            embedding = batch["__embedding__"]
            model = self._collapse_model(embedding)
            attr_model = BorzoiInputXGradient(model=model)
        else:
            raise ValueError(f"Invalid mode: {mode}. ")
        return attr_model

    def _no_training_randomness(self):
        """
        Disable any dropout or batch normalization updates
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            elif isinstance(module, torch.nn.BatchNorm1d):
                module.track_running_stats = False
                # module.running_mean.zero_()
                # module.running_var.fill_(1.0)

    def _model_attribution_step(
        self, model, batch, attr_batch_size, mode="attribution"
    ):
        assert (
            mode == "attribution"
        ), "Only attribution mode is supported for non-signal model now."

        dna = batch["__dna__"].float()
        dna.requires_grad_()

        bs = dna.shape[0]
        attr_col = []
        for i in range(0, bs, attr_batch_size):
            dna_mini_batch = dna[i : i + attr_batch_size]
            # calculate the attribution
            result = model(dna_mini_batch)
            dna_attr = result[0].detach().cpu().numpy()
            attr_col.append(dna_attr)
        attr_col = np.concatenate(attr_col, axis=0)
        batch["__dna__:attr"] = attr_col

        gc.collect()
        torch.cuda.empty_cache()
        return batch

    def _prepare_attr_regions(
        self, pseudobulk_ids, regions_per_pseudobulk
    ) -> dict[str, list[str]]:
        if isinstance(regions_per_pseudobulk, dict):
            # use different regions for each pseudobulk
            missing = []
            for pid in pseudobulk_ids:
                if pid not in regions_per_pseudobulk:
                    missing.append(pid)
            assert len(missing) == 0, (
                f"Missing regions for pseudobulks: {missing}. "
                "Please provide regions for all pseudobulks."
            )
            regions_per_pseudobulk = {
                pid: self._valid_and_sort_regions(
                    regions_per_pseudobulk[pid],
                    return_list=True,
                    standard_size=524288,
                )
                for pid in pseudobulk_ids
            }
        else:
            regions = regions_per_pseudobulk.copy()
            regions, region_names = self._valid_and_sort_regions(
                regions, return_list=True, standard_size=524288
            )
            regions_per_pseudobulk = {
                pid: (regions, region_names) for pid in pseudobulk_ids
            }
        return regions_per_pseudobulk

    def get_attribution_dataloader(
        self,
        regions_per_pseudobulk,
        pseudobulk_ids,
        dna_key="dna",
        embedding_key="embedding",
        batch_size=6,
        verbose=True,
        mode="attribution",
    ) -> Generator:
        """
        Get the dataloader for attribution.

        The main difference on data loader side is that prediction dataloader
        iterates all region and all pseudobulk together;
        attribution dataloader iterate all regions for one pseudobulk at a time.
        This is because
        1. predition task fetch parquet only once by putting all pseudobulk together,
            attribution task will not use parquet data.
        2. attribution task needs to collapse lora model into base,
            this can only be done one pseudobulk at a time.
        """
        da_prefix = self.datamanager._get_data_prefixs()
        assert (
            len(da_prefix) == 1
        ), "Currently only one data prefix is supported for prediction."
        data_key = da_prefix[0]

        # trigger model load
        _ = self.model
        self._no_training_randomness()

        if mode == "gene_count_attribution":
            dataloader_batch_size = batch_size * 5
        else:
            dataloader_batch_size = batch_size * 100
        timer = {"infer": 0, "callback": 0, "total": 0, "counter": 0}
        start = time.time()
        for pseudobulk_id in pseudobulk_ids:
            if isinstance(regions_per_pseudobulk, dict):
                regions, region_names = regions_per_pseudobulk[pseudobulk_id]
            else:
                regions, region_names = regions_per_pseudobulk

            dataloader = self._create_fn_and_dataloader(
                dna_key=dna_key,
                data_key=data_key,
                embedding_key=embedding_key,
                regions=regions,
                region_names=region_names,
                # attribution task will iterate one pseudobulk at a time
                pseudobulk_ids=[pseudobulk_id],
                # attribution task will not use true data in parquet
                # For Borzoi Signal mode, we will use reference bigwig for true data
                add_true_data=False,
                # batch size for attr is small, but we set it larger here so we save less batch files
                batch_size=dataloader_batch_size,
            )

            # 3. Get callbacks
            self._pre_callbacks = self._get_pre_callbacks(mode=mode)
            self._post_callbacks = self._get_post_callbacks(mode=mode)

            attr_model = None

            # 4. Inference and callback
            for batch in dataloader:
                batch["pseudobulk_id"] = pseudobulk_id

                if attr_model is None:
                    model = self._prepare_attr_model(batch, mode=mode)
                # Pre-inference callbacks
                t = time.time()
                batch = self.apply_callbacks(batch, "pre")
                timer["callback"] += time.time() - t

                # Inference step
                t = time.time()
                batch = self._model_attribution_step(
                    model, batch, attr_batch_size=batch_size, mode=mode
                )
                timer["infer"] += time.time() - t

                # Post-inference callbacks
                t = time.time()
                batch = self.apply_callbacks(batch, "post")
                timer["callback"] += time.time() - t
                timer["counter"] += 1
                yield batch
        timer["total"] = time.time() - start

        if verbose and timer["counter"] > 0:
            print(
                f"Total time: {timer['total']:.2f}s\n"
                f"Inference time: {timer['infer']:.2f}s or "
                f"{timer['infer']/timer['counter']:.3f}s per batch\n"
                f"Callback time: {timer['callback']:.2f}s or "
                f"{timer['callback']/timer['counter']:.3f}s per batch\n"
                f"(total {timer['counter']} batches)\n"
            )

    def _get_default_stats_keys(self):
        STATS_KEYS = [
            "region",
            "profile_pearsonr",
            "profile_r2",
            "peak",
            "peak_sample_pearsonr",
            "peak_sample_r2",
            "pseudobulk_ids",
            "peak_cum_profile_pearsonr",
            "peak_cum_profile_r2",
        ]
        return STATS_KEYS

    def _get_default_prediction_save_keys(self, mode="prediction"):
        SAVE_KEYS = [
            "__ytrue__:peak",
            "__ypred__:peak",
            "__embedding__",
            "peak",
            "region",
            "region_name",
            "pseudobulk_ids",
        ]

        if mode == "gene_count_prediction":
            SAVE_KEYS.append("__ypred__:gene_count")
        return SAVE_KEYS

    def prediction_task(
        self,
        output_dir,
        regions="test_regions",
        downsample_regions=None,
        downsample_seed=0,
        pseudobulk_ids=None,
        batch_size=16,
        save_keys="default",
        stats_keys=None,
        verbose=True,
        save_first_batch=False,
        mode="prediction",
        filter_valid_regions=True,
    ):
        """
        Prediction task for Borzoi.
        Compute the prediction on a set of regions and pseudobulk records.
        Then compute the stats and save them to a file.

        Parameters
        ----------
        output_dir: str
            The output directory to save the results.
        regions: str or pd.DataFrame or list[str]
            The regions to predict. If "test_regions",
            use the borzoi test regions based on fold in config.
        downsample_regions: int
            The number of regions to downsample. If None, use all regions.
        downsample_seed: int
            The seed for downsampling.
        pseudobulk_ids: list[str]
            The pseudobulk ids to use. If None, use all pseudobulk ids.
        batch_size: int
            The batch size for prediction.
        save_keys: list[str]
            The keys to save in the output batch file. If None, nothing will be saved.
        verbose: bool
            Whether to print the progress.
        save_first_batch: bool
            Whether to save the first full batch for debugging purposes.
        mode: str
            The mode of the task. One of "prediction", "gene_count_prediction".
            If "prediction", predict the genome tracks.
            If "gene_count_prediction", predict the gene counts along with genome tracks.
        filter_valid_regions: bool
            Whether to filter the regions to valid regions.
        """
        if mode == "gene_count_prediction":
            assert hasattr(self.model, "gene_count_output_head") or hasattr(
                self.model, "qtl_slope_output_head"
            ), (
                "Model does not have gene count output head or qtl slope output head. "
                "Please use a model with gene count output head or qtl slope output head for gene count prediction."
            )
            assert self.borzoi_gene_regions is not None, (
                "Borzoi gene regions is not set. "
                "Please check config['train_config']['use_regions']."
            )

        if isinstance(regions, str) and regions == "test_regions":
            regions = self.get_fold_regions(test_only=True, mode=mode)
        else:
            regions = understand_regions(regions)
            if filter_valid_regions:
                regions = self._filter_valid_regions(regions, mode=mode)
        if downsample_regions is not None:
            regions = regions.sample(n=downsample_regions, random_state=downsample_seed)

        output_dir = pathlib.Path(output_dir).absolute().resolve()
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        stats_path = output_dir / "summary_stats.joblib.gz"
        stats_tmpdir = tempfile.mkdtemp()
        stats_tmpdir = pathlib.Path(stats_tmpdir)
        if verbose:
            print(f"Saving batches to {batch_dir}")
            print(f"Saving stats to {stats_path}")
            print(f"Using temporary directory {stats_tmpdir}")

        dataloader = self.get_prediction_dataloader(
            regions=regions,
            batch_dir=batch_dir,
            add_true_data=True,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            verbose=verbose,
            mode=mode,
        )

        config_path = output_dir / "config.joblib.gz"
        self._save_task_configs(
            task_config={
                "regions": regions,
                "pseudobulk_ids": pseudobulk_ids,
                "save_keys": save_keys,
            },
            output_path=config_path,
        )

        # data to save for each batch
        if isinstance(save_keys, str) and save_keys == "default":
            save_keys = self._get_default_prediction_save_keys(mode=mode)
        save_keys = [] if save_keys is None else save_keys
        # stats to collect across batches
        default_stats_keys = self._get_default_stats_keys()
        if stats_keys is not None:
            stats_keys = list(set(default_stats_keys + stats_keys))
        else:
            stats_keys = default_stats_keys

        batch_stats_paths = []
        save_batch = {}
        cur_bid = _get_cur_bid(batch_dir)
        for idx, batch in enumerate(dataloader):
            idx = idx + cur_bid
            if idx == 0 and verbose:
                self._print_batch(batch, prefix="Dataloader")
                if save_first_batch:
                    # save the first complete batch into output dir
                    joblib.dump(batch, output_dir / "first_batch.joblib.gz")

            # save the batch to a file
            save_batch = {}
            stats_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if k in save_keys:
                    save_batch[k] = v
                if k in stats_keys:
                    stats_batch[k] = v
            # data to save for each batch
            save_path = batch_dir / f"batch_{idx}.joblib.gz"
            joblib.dump(save_batch, save_path)
            # stats to collect across batches
            batch_stats_path = stats_tmpdir / f"batch_{idx}.joblib"
            joblib.dump(stats_batch, batch_stats_path)
            batch_stats_paths.append(batch_stats_path)

        if verbose and len(save_batch) > 0:
            self._print_batch(save_batch, prefix="Saved")

        # collect the stats across batches
        total_stats = defaultdict(list)
        for _path in batch_stats_paths:
            batch_stats = joblib.load(_path)
            for k, v in batch_stats.items():
                total_stats[k].append(v)

        # concatenate the stats
        for k in total_stats.keys():
            v = total_stats[k]
            if k == "pseudobulk_ids":
                total_stats[k] = v[0]
            elif isinstance(v[0], pd.DataFrame):
                total_stats[k] = pd.concat(v)
            else:
                total_stats[k] = np.concatenate(v)

        if len(batch_stats_paths) > 0 and not self._embedding_only_mode:
            # add cumulative stats to final stats
            cum_data = self.compute_cumulative_callbacks()
            for k, v in cum_data.items():
                if k in stats_keys:
                    total_stats[k] = v

        # save the stats
        joblib.dump(total_stats, stats_path)

        # remove the temporary files
        rmtree(stats_tmpdir)
        if verbose:
            self._print_batch(total_stats, prefix="Final Stats")
            print(f"Removed temporary files in {stats_tmpdir}")

        # gather peak data into single dataframe
        self._gather_peak_data(output_dir)
        return

    @staticmethod
    def gather_gene_data(output_dir, gene_data_path):
        """
        Gather gene data into single dataframe from all batches in output_dir.
        """
        return gather_gene_data(
            output_dir,
            gene_data_path=gene_data_path,
        )

    def select_top_std_genes(self, gene_data_path, top_n=5000):
        """
        Select top variable genes, intersect with test fold.
        """
        true_gene_data = pd.read_feather(gene_data_path)
        *_, test_regions = self.borzoi_gene_regions.get_train_valid_test_regions(0)
        use_genes = (
            true_gene_data.std(axis=0).sort_values(ascending=False)[:top_n].index
        )

        use_test_regions = test_regions[test_regions["Name"].isin(use_genes)].copy()
        return use_test_regions

    @staticmethod
    def _gather_peak_data(output_dir):
        return gather_peak_data(
            output_dir,
            true_peak_key="__ytrue__:peak",
            pred_peak_key="__ypred__:peak",
        )

    def _filter_valid_regions(self, regions=None, mode="qtl"):
        db = self.datamanager.datasets["parquet"]
        if regions is None:
            if mode in ("qtl", "eqtl"):
                regions = self.qtl_manager.regions
            elif mode == "peak":
                regions = self.peak_manager.regions
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be one of 'qtl', 'eqtl' or 'peak'."
                )

        regions_bed = regions.reset_index().iloc[:, [1, 2, 3, 0]]
        regions_bed["Name"] = list(range(regions_bed.shape[0]))
        regions_bed = pr.PyRanges(regions_bed)
        valid_regions_bed = regions_bed.overlap(
            db.region_lookup_bed.merge(), how="containment"
        )
        new_regions = regions.iloc[valid_regions_bed.df["Name"].values].copy()
        if new_regions.shape[0] != regions.shape[0]:
            print(
                f"Filtered {regions.shape[0] - new_regions.shape[0]} invalid regions "
                "from input regions that are not fully overlapped with the parquet dataset."
            )

        if "Name" not in new_regions.columns:
            new_regions["Name"] = new_regions.index.astype(str)
        return new_regions

    def _gather_qtl_data(self, output_dir, qtl_type="caqtl"):
        batch_paths = list(
            pathlib.Path(f"{output_dir}/batch/").glob("batch_*.joblib.gz")
        )
        config = joblib.load(f"{output_dir}/config.joblib.gz")

        # Create pid_mapping, handling cases where __pid__ annotation might be missing
        pid_mapping = {}
        for k, v in config["pseudobulk_records"].items():
            if "__pid__" in v:
                pid_mapping[k] = v["__pid__"]
            else:
                # Fallback: use the pseudobulk ID itself if __pid__ annotation is missing
                pid_mapping[k] = k

        if qtl_type == "caqtl":
            suffix = ":peak"
        elif qtl_type == "eqtl":
            suffix = ":gene_count"
        else:
            raise ValueError(f"Invalid qtl type: {qtl_type}")

        all_qtl_data = []
        all_ref_data = []
        all_alt_data = []
        for path in batch_paths:
            batch = joblib.load(path)
            if len(pid_mapping.keys()) != batch[f"__ypred__:ref{suffix}"].shape[1]:
                pids = pd.Index(batch["pseudobulk_ids"][::2]).map(pid_mapping)
            else:
                pids = pd.Index(batch["pseudobulk_ids"]).map(pid_mapping)
            ref_data = pd.DataFrame(
                batch[f"__ypred__:ref{suffix}"],
                index=batch["region_name"],
                columns=pids,
            )
            alt_data = pd.DataFrame(
                batch[f"__ypred__:alt{suffix}"],
                index=batch["region_name"],
                columns=pids,
            )
            logfc = np.log2(alt_data / ref_data)
            all_ref_data.append(ref_data)
            all_alt_data.append(alt_data)
            all_qtl_data.append(logfc)

        all_qtl_data = pd.concat(all_qtl_data)
        all_alt_data = pd.concat(all_alt_data)
        all_ref_data = pd.concat(all_ref_data)

        all_ref_data.to_feather(f"{output_dir}/ref_data.feather")
        all_alt_data.to_feather(f"{output_dir}/alt_data.feather")
        all_qtl_data.to_feather(f"{output_dir}/ref_alt_logfc.feather")
        return

    def caqtl_task(
        self,
        output_dir,
        qtl_table,
        pseudobulk_ids=None,
        batch_size=16,
        save_keys="default",
        add_true_data=False,
        verbose=True,
        save_first_batch=False,
        qtl_type="caqtl",
        channel_weights_path=None,
    ):
        """
        QTL Prediction task for Borzoi.
        Compute the ref and alt prediction of a QTL dataset
        Then compute the stats and save them to a file.

        Parameters
        ----------
        output_dir: str
            The output directory to save the results.
        qtl_table: str or pd.DataFrame
            The QTL table path to use. QTL table should contain inference regions, qtl mutations and qtl peaks.
        pseudobulk_ids: list[str]
            The pseudobulk ids to use. If None, use all pseudobulk ids.
        batch_size: int
            The batch size for prediction.
        save_keys: list[str]
            The keys to save in the output file. If None, save all keys.
        verbose: bool
            Whether to print the progress.
        save_first_batch: bool
            Whether to save the first complete batch into output dir.
        """
        if isinstance(save_keys, str) and save_keys == "default":
            save_keys = [
                "region",
                "region_name",
                "pseudobulk_ids",
                "mutation_id",
                "qtl_peaks",
                "__ypred__:ref:peak",
                "__ypred__:alt:peak",
            ]

        # add qtl manager and get regions from the qtl manager
        self.register_qtl_manager(
            qtl_table, qtl_type=qtl_type, channel_weights_path=channel_weights_path
        )
        regions = self._filter_valid_regions(mode="qtl")

        output_dir = pathlib.Path(output_dir).absolute().resolve()
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving batches to {batch_dir}")

        dataloader = self.get_prediction_dataloader(
            regions=regions,
            batch_dir=batch_dir,
            add_true_data=add_true_data,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            verbose=verbose,
            mode="qtl",
        )

        config_path = output_dir / "config.joblib.gz"
        self._save_task_configs(
            task_config={
                "regions": regions,
                "pseudobulk_ids": pseudobulk_ids,
                "save_keys": save_keys,
                "qtl_table": qtl_table,
            },
            output_path=config_path,
        )

        # data to save for each batch
        save_keys = [] if save_keys is None else save_keys

        save_batch = {}
        cur_bid = _get_cur_bid(batch_dir)
        for idx, batch in enumerate(dataloader):
            idx = idx + cur_bid
            if idx == 0 and verbose:
                self._print_batch(batch, prefix="Dataloader")
                if save_first_batch:
                    # save the first complete batch into output dir
                    joblib.dump(batch, output_dir / "first_batch.joblib.gz")

            save_batch = self._save_batch(
                batch=batch, batch_dir=batch_dir, idx=idx, save_keys=save_keys
            )

        if verbose and len(save_batch) > 0:
            self._print_batch(save_batch, prefix="Saved")

        self._gather_qtl_data(output_dir, qtl_type="caqtl")
        return

    def qtl_task(self, *args, **kwargs):
        """Alias for caqtl_task"""
        print("qtl_task is deprecated. Use caqtl_task instead.")
        return self.caqtl_task(*args, **kwargs)

    def eqtl_task(
        self,
        output_dir,
        qtl_table,
        pseudobulk_ids=None,
        batch_size=16,
        save_keys="default",
        add_true_data=False,
        verbose=True,
        save_first_batch=False,
    ):
        """
        eQTL task for Borzoi - predict effect of variants on gene expression.

        Parameters
        ----------
        output_dir: str
            The output directory to save the results.
        qtl_table : str or pd.DataFrame
            eQTL table with variant info, gene info, and regions
        pseudobulk_ids: list[str]
            The pseudobulk ids to use. If None, use all pseudobulk ids.
        batch_size: int
            The batch size for prediction.
        save_keys: list[str]
            The keys to save in the output file. If None, save all keys.
        verbose: bool
            Whether to print the progress.
        add_true_data: bool
            Whether to add true data to the batch.
        save_first_batch: bool
            Whether to save the first complete batch into output dir.
        """
        if isinstance(save_keys, str) and save_keys == "default":
            save_keys = [
                "region",
                "region_name",
                "pseudobulk_ids",
                "mutation_id",
                "eqtl_genes",
                "__ypred__:ref:gene_count",
                "__ypred__:alt:gene_count",
            ]

        self.register_qtl_manager(qtl_table, qtl_type="eqtl")
        regions = self._filter_valid_regions(mode="eqtl")

        output_dir = pathlib.Path(output_dir).absolute().resolve()
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving batches to {batch_dir}")

        dataloader = self.get_prediction_dataloader(
            regions=regions,
            batch_dir=batch_dir,
            add_true_data=add_true_data,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            verbose=verbose,
            mode="eqtl",
        )

        config_path = output_dir / "config.joblib.gz"
        self._save_task_configs(
            task_config={
                "regions": regions,
                "pseudobulk_ids": pseudobulk_ids,
                "save_keys": save_keys,
                "qtl_table": qtl_table,
            },
            output_path=config_path,
        )

        # data to save for each batch
        save_keys = [] if save_keys is None else save_keys
        save_batch = {}
        cur_bid = _get_cur_bid(batch_dir)
        for idx, batch in enumerate(dataloader):
            idx = idx + cur_bid
            if idx == 0 and verbose:
                self._print_batch(batch, prefix="Dataloader")
                if save_first_batch:
                    # save the first complete batch into output dir
                    joblib.dump(batch, output_dir / "first_batch.joblib.gz")

            # save the batch to a file
            save_batch = self._save_batch(
                batch=batch, batch_dir=batch_dir, idx=idx, save_keys=save_keys
            )

        if verbose and len(save_batch) > 0:
            self._print_batch(save_batch, prefix="Saved")
        return

    def peak_task(
        self,
        output_dir,
        peak_table,
        pseudobulk_ids=None,
        batch_size=16,
        save_keys="default",
        add_true_data=True,
        verbose=True,
    ):
        """
        Prediction task for Borzoi.
        Compute the prediction on a peak of interests.

        Parameters
        ----------
        output_dir: str
            The output directory to save the results.
        peak_table: str or pd.DataFrame
            The peak table path to use. Peak table should contain borzoi input regions and peak regions.
        pseudobulk_ids: list[str]
            The pseudobulk ids to use. If None, use all pseudobulk ids.
        batch_size: int
            The batch size for prediction.
        save_keys: list[str]
            The keys to save in the output file. If None, save all keys.
        verbose: bool
            Whether to print the progress.
        """
        if isinstance(save_keys, str) and save_keys == "default":
            save_keys = [
                "region",
                "region_name",
                "pseudobulk_ids",
                "peaks",
                "__ypred__:peak",
            ]

        # add peak manager and get regions from the peak manager
        self.register_peak_manager(peak_table)
        regions = self._filter_valid_regions(mode="peak")

        output_dir = pathlib.Path(output_dir).absolute().resolve()
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving batches to {batch_dir}")

        dataloader = self.get_prediction_dataloader(
            regions=regions,
            batch_dir=batch_dir,
            add_true_data=add_true_data,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            verbose=verbose,
            mode="peak",
        )

        config_path = output_dir / "config.joblib.gz"
        self._save_task_configs(
            task_config={
                "regions": regions,
                "pseudobulk_ids": pseudobulk_ids,
                "save_keys": save_keys,
                "peak_table": peak_table,
            },
            output_path=config_path,
        )

        # data to save for each batch
        save_keys = [] if save_keys is None else save_keys

        save_batch = {}
        cur_bid = _get_cur_bid(batch_dir)
        for idx, batch in enumerate(dataloader):
            idx = idx + cur_bid
            if idx == 0 and verbose:
                self._print_batch(batch, prefix="Dataloader")
                # save the first complete batch into output dir
                joblib.dump(batch, output_dir / "first_batch.joblib.gz")

            # save the batch to a file
            save_batch = self._save_batch(
                batch=batch, batch_dir=batch_dir, idx=idx, save_keys=save_keys
            )

        if verbose and len(save_batch) > 0:
            self._print_batch(save_batch, prefix="Saved")
        return

    def _check_finished_attribution_batches(self, batch_dir, regions_per_pseudobulk):
        # make a pid-by-region bool table
        pid_region_table_log = {}
        for pid, (_, region_names) in regions_per_pseudobulk.items():
            pid_region_table_log[pid] = pd.Series(
                np.ones(len(region_names), dtype=bool), index=region_names
            )

        # collect all finished batches
        batch_paths = batch_dir.glob("*.joblib.gz")
        bids = []
        for path in batch_paths:
            bids.append(int(path.name.split(".")[0].split("_")[1]))
            batch = joblib.load(path)
            region_names = batch["region_name"]
            pid = batch["pseudobulk_id"]
            if pid not in pid_region_table_log:
                continue
            pid_region_table_log[pid].loc[region_names] = False

        # only run unfinished pid and regions
        def sel_list_with_bool(rl, bool_sel):
            rl = np.array(rl)[bool_sel]
            rl = rl.tolist()
            return rl

        cur_bid = max(bids) + 1 if len(bids) > 0 else 0
        regions_per_pseudobulk_torun = {}
        for pid, (regions, region_names) in regions_per_pseudobulk.items():
            regions_bool = pid_region_table_log[pid]
            if regions_bool.any():
                regions_per_pseudobulk_torun[pid] = (
                    sel_list_with_bool(regions, regions_bool),
                    sel_list_with_bool(region_names, regions_bool),
                )
        return regions_per_pseudobulk_torun, cur_bid

    def _save_pseudobulk_attr_ds(self, pseudobulk_ids, batch_dir):
        _all_attr_keys = ["__dna__:attr", "__signal__:attr"]
        _attr_keys = None
        for pid in pseudobulk_ids:
            pid_batch_paths = list(batch_dir.glob(f"*.{pid}.joblib.gz"))

            temp_path = batch_dir / f"{pid}.attr.zarr.temp"
            zarr_path = batch_dir / f"{pid}.attr.zarr"
            if not zarr_path.exists():
                pid_data = defaultdict(list)
                for p in pid_batch_paths:
                    batch = joblib.load(p)
                    if _attr_keys is None:
                        _attr_keys = [k for k in _all_attr_keys if k in batch]
                    region = batch["region"]
                    pid = batch["pseudobulk_id"]
                    for key in _attr_keys:
                        attr = batch[key]
                        if key == "__dna__:attr":
                            da = xr.DataArray(attr, dims=["region", "base", "pos"])
                            da.coords["region"] = region
                            da.coords["base"] = list("ACGT")
                        else:
                            da = xr.DataArray(attr, dims=["region", "channel", "pos"])
                            da.coords["region"] = region
                        pid_data[key].append(da)
                pid_data = {k: xr.concat(v, dim="region") for k, v in pid_data.items()}
                pid_ds = xr.Dataset(pid_data).chunk(region=10000)

                # change regions (524288) into attr regions (1024)
                full_regions = pid_ds.get_index("region")
                attr_length = attr.shape[-1]
                regions = self.genome.standard_region_length(
                    full_regions, length=attr_length, keep_original=True
                )
                regions = regions.set_index("Original_Name").reindex(full_regions)
                pid_ds.coords["region"] = regions["Name"].values

                pid_ds.to_zarr(temp_path, mode="w")
                temp_path.rename(zarr_path)

            for path in pid_batch_paths:
                path.unlink()
        return

    def attribution_task(
        self,
        output_dir,
        regions_per_pseudobulk,
        pseudobulk_ids=None,
        batch_size=6,
        save_keys=("__dna__:attr", "region", "pseudobulk_id", "region_name"),
        verbose=True,
        save_first_batch=False,
        mode="attribution",
    ):
        """
        Prediction task for Borzoi.
        Compute the prediction on a set of regions and pseudobulk records.
        Then compute the stats and save them to a file.

        Parameters
        ----------
        output_dir: str
            The output directory to save the results.
        regions_per_pseudobulk: str or pd.DataFrame or list[str]
            The regions to run attribution on. Regions center should be a peak.
        pseudobulk_ids: list[str]
            The pseudobulk ids to use. If None, use all pseudobulk ids.
        batch_size: int
            The batch size for prediction.
        save_keys: list[str]
            The keys to save in the output file. If None, save all keys.
        verbose: bool
            Whether to print the progress.
        save_first_batch: bool
            Whether to save the first full batch for debugging purposes.
        mode: str
            The mode of the task. One of "attribution", "gene_count_attribution".
            If "attribution", compute the attribution for the DNA input.
            If "gene_count_attribution", compute the attribution for the DNA input and gene count input.
        """
        output_dir = pathlib.Path(output_dir).absolute().resolve()
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving batches to {batch_dir}")

        # prepare regions for each pseudobulk
        if pseudobulk_ids is None:
            pseudobulk_ids = self.pseudobulk_manager.pseudobulk_ids
        if isinstance(regions_per_pseudobulk, str):
            regions_per_pseudobulk = pr.read_bed(regions_per_pseudobulk, as_df=True)
        regions_per_pseudobulk = self._prepare_attr_regions(
            pseudobulk_ids=pseudobulk_ids, regions_per_pseudobulk=regions_per_pseudobulk
        )
        # check and skip finished regions for each pseudobulk
        regions_per_pseudobulk_torun, cur_bid = (
            self._check_finished_attribution_batches(
                batch_dir=batch_dir,
                regions_per_pseudobulk=regions_per_pseudobulk,
            )
        )
        pseudobulk_ids_torun = list(regions_per_pseudobulk_torun.keys())
        if len(regions_per_pseudobulk_torun) != 0:
            config_path = output_dir / "config.joblib.gz"
            self._save_task_configs(
                task_config={
                    "regions": regions_per_pseudobulk,
                    "pseudobulk_ids": pseudobulk_ids,
                    "save_keys": save_keys,
                },
                output_path=config_path,
            )

            dataloader = self.get_attribution_dataloader(
                regions_per_pseudobulk=regions_per_pseudobulk_torun,
                pseudobulk_ids=pseudobulk_ids_torun,
                dna_key="dna",
                embedding_key="embedding",
                batch_size=batch_size,
                verbose=verbose,
                mode=mode,
            )

            save_batch = {}
            for idx, batch in enumerate(dataloader):
                idx = idx + cur_bid
                if idx == 0 and verbose:
                    self._print_batch(batch, prefix="Dataloader")
                    # save the first complete batch into output dir
                    if save_first_batch:
                        joblib.dump(batch, output_dir / "first_batch.joblib.gz")

                # save the batch to a file
                pid = batch["pseudobulk_id"]
                save_batch = self._save_batch(
                    batch=batch,
                    batch_dir=batch_dir,
                    idx=f"{idx}.{pid}",
                    save_keys=save_keys,
                )

            if verbose and len(save_batch) > 0:
                self._print_batch(save_batch, prefix="Saved")

        if mode == "attribution":
            self._save_pseudobulk_attr_ds(
                pseudobulk_ids=pseudobulk_ids, batch_dir=batch_dir
            )
        return


class BorzoiMultiHeadPredictor(BorzoiPredictor):
    def _model_prediction_step(
        self, batch, dna_key="__dna__", ypred_key="__ypred__", batch_size=16, crop=True
    ):
        """
        Forward pass through the model.
        """
        # prepare data
        dna = batch[dna_key]  # (n_region, 4, seq_len)

        n_region = dna.shape[0]
        pred_col = []
        for i in range(0, n_region, batch_size):
            dna_mini_batch = dna[i : i + batch_size]

            with self._autocast_context():
                y_pred_mini_batch = self.model(dna_mini_batch, crop=crop)
                pred_col.append(y_pred_mini_batch)

        y_pred = torch.cat(pred_col, dim=0)
        # y_pred shape (n_region, n_emb/n_pseudobulks, seq_len)

        batch[ypred_key] = y_pred.detach()
        return batch

    def _model_gene_count_prediction_step(self, *args, **kwargs):
        raise NotImplementedError(
            "Gene count prediction is not implemented for multi-head models."
        )


class BorzoiPairPredictor(BorzoiPredictor):
    def __init__(self, config):
        paired_pseudobulker, config = self._create_pseudobulk_records_from_design(
            config
        )
        self.paired_pseudobulker = paired_pseudobulker

        super().__init__(config)

    def _create_pseudobulk_records_from_design(self, config):
        """
        Create paired pseudobulk records from the design.
        """
        _train_config = load_config(config["train_config"])
        paired_mode = _train_config.get("paired_mode", "ensemble")
        pseudobulker_cls = PAIRED_PSEUDOBULKER_CLS_DICT[paired_mode]
        paired_pseudobulker = pseudobulker_cls(
            pseudobulk_and_ot_info=config["pseudobulk_records_path"],
            emb_key="embedding",
            downsample_pseudobulk=None,
            barcode_order=None,
            seed=42,
        )
        # if designs is None, will use all pids in the pseudobulk records
        designs = config.get("designs", None)
        config["pseudobulk_records_path"] = (
            paired_pseudobulker.create_pseudobulk_records_from_design(designs)
        )
        return paired_pseudobulker, config

    def _get_post_prediction_callbacks(self):
        # get the default prediction callbacks first
        pred_callbacks = super()._get_post_prediction_callbacks()

        # add paired task callbacks
        paired_callbacks = [
            # calculate paired profile correlation
            (
                "process_paired_data",
                {
                    "data_keys": ["__ytrue__:peak", "__ypred__:peak"],
                },
            ),
            (
                "pearsonr",
                {
                    "ytrue_key": "__ytrue__:peak:cond1",
                    "ypred_key": "__ypred__:peak:cond1",
                    "permute": (1, 0),  # (sample, peak) -> (peak, sample)
                    "output_key": "peak_pearsonr",
                    "cumulative": True,
                },
            ),
            (
                "r2_score",
                {
                    "ytrue_key": "__ytrue__:peak:cond1",
                    "ypred_key": "__ypred__:peak:cond1",
                    "permute": (1, 0),
                    "output_key": "peak_r2",
                    "cumulative": True,
                },
            ),
        ]
        cb = self._prepare_callbacks(paired_callbacks)
        pred_callbacks.extend(cb)
        return pred_callbacks

    def _get_default_stats_keys(self):
        STATS_KEYS = super()._get_default_stats_keys()
        STATS_KEYS += [
            "peak_pearsonr",
            "peak_r2",
        ]
        return STATS_KEYS

    @staticmethod
    def _gather_peak_data(output_dir):
        return gather_peak_data(
            output_dir,
            true_peak_key="__ytrue__:peak:cond1",
            pred_peak_key="__ypred__:peak:cond1",
        )


class BorzoiSignalPredictor(BorzoiPairPredictor):
    """
    BorzoiSignalPredictor is a predictor for Borzoi models that uses ODEs to predict
    the trajectory of the model given an initial condition.
    It is used for signal-based models.
    """

    def __init__(self, config):
        super().__init__(config)
        self.has_cond_emb = self.config.get("cond_emb_dim") is not None
        self._forward_without_signal = self.config.get("nosignal", False)

    def _get_pre_prediction_callbacks(self):
        _data_keys = ["__ytrue__", "__embedding__"]
        split_dim = [1, 0]
        if self.has_cond_emb:
            _data_keys.append("__conditionemb__")
            split_dim.append(0)
        if getattr(self, "has_shared_data", False):
            _data_keys.append("__shared_data__")
            split_dim.append(0)

        callback_configs = [
            # calculate paired profile correlation
            (
                "process_paired_data",
                {
                    "data_keys": _data_keys,
                    "split_dim": split_dim,
                },
            ),
        ]
        callbacks = self._prepare_callbacks(callback_configs)
        return callbacks

    def _get_pre_attribution_callbacks(self):
        return []

    def _get_pre_prediction_gene_count_callbacks(self, mode):
        if mode == "eqtl":
            region_name_to_strand = self.qtl_manager.region_to_strand
        else:
            region_name_to_strand = self.borzoi_gene_regions.borzoi_regions.set_index(
                "Name"
            )["Strand"].to_dict()

        callback_configs = [
            # calculate paired profile correlation
            (
                "reverse_complement_minus_strand",
                {
                    "dna_key": "__dna__",
                    "signal_key": [] if self._forward_without_signal else "__ytrue__",
                    "region_name_to_strand": region_name_to_strand,
                },
            ),
        ]
        callbacks = self._prepare_callbacks(callback_configs)
        return callbacks

    def _get_pre_callbacks(self, mode="prediction"):
        if mode in ("prediction", "qtl", "peak"):
            return self._get_pre_prediction_callbacks()
        elif mode in ("gene_count_prediction", "eqtl"):
            _gene_call_back = self._get_pre_prediction_gene_count_callbacks(mode)
            _pred_call_back = self._get_pre_prediction_callbacks()
            return _gene_call_back + _pred_call_back
        elif mode == "attribution":
            return self._get_pre_attribution_callbacks()
        elif mode == "gene_count_attribution":
            _gene_call_back = self._get_pre_prediction_gene_count_callbacks(mode)
            _attr_call_back = self._get_pre_attribution_callbacks()
            return _gene_call_back + _attr_call_back
        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Must be one of 'prediction' or 'attribution'."
            )

    def _get_post_attribution_callbacks(self, mode="attribution"):
        if mode == "attribution":
            return []
        elif mode == "gene_count_attribution":
            callbacks = [
                (
                    "gene_count_attr_post_process",
                    {
                        "seqlet_center_flank": 25,
                        "save_full_attr": False,
                        "save_full_attr1d": True,
                        "save_top_q": 0.02,
                        "threshold": 0.001,
                    },
                )
            ]
            return self._prepare_callbacks(callbacks)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Must be one of 'attribution' or 'gene_count_attribution'."
            )

    def _get_post_prediction_callbacks(self, mode="prediction"):
        # add paired task callbacks
        callbacks = [
            (
                "extract_peak",
                {
                    "peak_bed": self.peak_df,
                    "data_keys": [
                        "__ytrue__:cond0",
                        "__ytrue__:cond1",
                        "__ypred__:cond1",
                    ],
                    "_is_gene_region": mode == "gene_count_prediction",
                },
            ),
            (
                "rename",
                {
                    "name_map": {
                        "__ytrue__:cond0:peak": "__ytrue__:peak:cond0",
                        "__ytrue__:cond1:peak": "__ytrue__:peak:cond1",
                        "__ypred__:cond1:peak": "__ypred__:peak:cond1",
                    }
                },
            ),
            # calculate paired profile correlation
            (
                "pearsonr",
                {
                    "ytrue_key": "__ytrue__:peak:cond1",
                    "ypred_key": "__ypred__:peak:cond1",
                    "permute": (1, 0),  # (sample, peak) -> (peak, sample)
                    "output_key": "peak_pearsonr",
                    "cumulative": True,
                },
            ),
            (
                "r2_score",
                {
                    "ytrue_key": "__ytrue__:peak:cond1",
                    "ypred_key": "__ypred__:peak:cond1",
                    "permute": (1, 0),
                    "output_key": "peak_r2",
                    "cumulative": True,
                },
            ),
        ]
        cb = self._prepare_callbacks(callbacks)
        return cb

    def _get_post_null_callbacks(self):
        return []

    def _get_post_callbacks(self, mode="prediction"):
        """
        Get the post callbacks for the model.
        """
        if mode in ("prediction", "gene_count_prediction"):
            return self._get_post_prediction_callbacks(mode)
        elif mode in ("attribution", "gene_count_attribution"):
            return self._get_post_attribution_callbacks(mode)
        elif mode in ("qtl", "eqtl"):
            return self._get_post_null_callbacks()
        elif mode == "peak":
            return self._get_post_null_callbacks()
        else:
            raise ValueError(f"Invalid mode: {mode}. ")

    def _split_cond_emb_to_terms(self, cond_emb):
        # split the cond_emb into dict of terms using cond_encoder in pseudobulker
        pseduobulker = self.paired_pseudobulker
        condition_encoder = getattr(pseduobulker, "condition_encoder", None)
        if condition_encoder is not None:
            cond_emb_terms = condition_encoder.split_cond_emb(cond_emb)
        else:
            cond_emb_terms = cond_emb
        return cond_emb_terms

    def _prepare_attr_model(self, batch, mode="attribution"):
        embedding = batch["__embedding__"]
        if self.has_cond_emb:
            cond_emb = batch["__conditionemb__"]
            cond_emb = self._split_cond_emb_to_terms(cond_emb)
        else:
            cond_emb = None
        with torch.no_grad():
            agg_emb = self._forward_cond_emb_module(
                cell_emb=embedding, cond_emb=cond_emb
            )
            # TODO: add shared data for multi model with score data
        model = self._collapse_model(agg_emb)

        if mode == "attribution":
            attr_model = BorzoiInputXGradient(model=model)
        elif mode == "gene_count_attribution":
            attr_model = BorzoiGeneInputXGradient(model=model)
        else:
            raise ValueError(f"Invalid mode: {mode}. ")
        return attr_model

    def _dna_attr_step(self, model, batch, attr_batch_size):
        """
        Calculate the attribution for DNA input.
        """
        dna = batch["__dna__"].float()
        dna.requires_grad_()

        bs = dna.shape[0]
        dna_attr_col = []
        for i in range(0, bs, attr_batch_size):
            dna_mini_batch = dna[i : i + attr_batch_size]
            # calculate the attribution
            dna_attr = model(dna_mini_batch)[0]
            dna_attr_col.append(dna_attr.detach().cpu().numpy())
        dna_attr_col = np.concatenate(dna_attr_col, axis=0)
        batch["__dna__:attr"] = dna_attr_col
        return batch

    def _dna_gene_count_attr_step(self, model, batch, attr_batch_size):
        dna = batch["__dna__"].float()
        dna.requires_grad_()

        gene_mask = self.borzoi_gene_regions.get_gene_mask(batch, dna=dna)
        gene_mask = gene_mask.float()
        gene_mask.requires_grad_()

        bs = dna.shape[0]
        dna_attr_col = []
        for i in range(0, bs, attr_batch_size):
            dna_mini_batch = dna[i : i + attr_batch_size]
            gene_mask_mini_batch = gene_mask[i : i + attr_batch_size]
            # calculate the attribution
            dna_attr, mask_attr = model(dna_mini_batch, gene_mask_mini_batch)
            dna_attr_col.append(dna_attr.detach().cpu().numpy())
        dna_attr_col = np.concatenate(dna_attr_col, axis=0)
        batch["__dna__:attr"] = dna_attr_col
        return batch

    def _dna_sig_attr_step(self, model, batch, attr_batch_size):
        dna = batch["__dna__"].float()
        dna.requires_grad_()

        signal = batch["reference"].unsqueeze(1)
        signal = torch.log1p(signal)
        signal.requires_grad_()

        bs = dna.shape[0]
        dna_attr_col = []
        signal_attr_col = []
        for i in range(0, bs, attr_batch_size):
            dna_mini_batch = dna[i : i + attr_batch_size]
            signal_mini_batch = signal[i : i + attr_batch_size]
            # calculate the attribution
            dna_attr, signal_attr = model(dna_mini_batch, signal_mini_batch)
            dna_attr_col.append(dna_attr.detach().cpu().numpy())
            signal_attr_col.append(signal_attr.detach().cpu().numpy())
        dna_attr_col = np.concatenate(dna_attr_col, axis=0)
        signal_attr_col = np.concatenate(signal_attr_col, axis=0)
        batch["__dna__:attr"] = dna_attr_col
        batch["__signal__:attr"] = signal_attr_col
        return batch

    def _dna_sig_gene_count_attr_step(self, model, batch, attr_batch_size):
        raise NotImplementedError(
            "gene count attribution with signal is not implemented"
        )

    def _model_attribution_step(
        self, model, batch, attr_batch_size, mode="attribution"
    ):
        """
        Forward pass through the model to get the attribution.
        """
        if mode == "attribution":
            if self._forward_without_signal:
                batch = self._dna_attr_step(model, batch, attr_batch_size)
            else:
                batch = self._dna_sig_attr_step(model, batch, attr_batch_size)
        elif mode == "gene_count_attribution":
            if self._forward_without_signal:
                batch = self._dna_gene_count_attr_step(model, batch, attr_batch_size)
            else:
                batch = self._dna_sig_gene_count_attr_step(
                    model, batch, attr_batch_size
                )
        else:
            raise ValueError(f"Invalid mode: {mode}. ")

        # clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        return batch

    def _forward_cond_emb_module(self, cell_emb, cond_emb):
        if self.model.cond_emb_module is None:
            return cell_emb

        cond_ensemble = self.model.cond_emb_module(cell_emb=cell_emb, cond_emb=cond_emb)
        return cond_ensemble

    def _model_prediction_step(
        self,
        batch,
        dna_key="__dna__",
        x0_key="__ytrue__:cond0",
        cell_embedding_key="__embedding__:cond0",
        cond_embedding_key="__conditionemb__:cond1",
        batch_size=16,
        ypred_key="__ypred__:cond1",
        crop=True,
        gene_mode=False,
    ):
        """
        Forward pass through the model.
        """
        # prepare data
        dna = batch[dna_key]
        cell_emb = batch[cell_embedding_key]
        if getattr(self, "has_shared_data", False):
            # cond0 and cond1 has the same shared data value
            shared_data = batch["__shared_data__:cond1"].to(cell_emb.dtype)
            if shared_data.ndim == 1:
                shared_data = shared_data.unsqueeze(1)
                # shared_data shape (bs, 1)
        else:
            shared_data = None

        if self.has_cond_emb:
            cond_emb = batch[cond_embedding_key]
        else:
            cond_emb = None

        if gene_mode:
            gene_mask = self.borzoi_gene_regions.get_gene_mask(batch, dna=dna)

        # take reference signal and log1p scale
        x0 = batch.get(x0_key, None)
        if x0 is None:
            # x0 key doesn't exist, try reference signal key
            x0 = batch.get("reference", None)
        if x0 is None:
            raise KeyError(
                f"x0 key {x0_key} or reference signal key not found in batch, "
                f"current batch keys are: {batch.keys()}"
            )
        x0 = torch.log1p(x0)

        n_emb = cell_emb.shape[0]
        n_region = dna.shape[0]

        emb_idx = torch.arange(n_emb).repeat(n_region)
        # [0, 1, ..., n_emb-1, ..., 0, 1, ..., n_emb-1]
        region_idx = torch.arange(n_region).repeat_interleave(n_emb)
        # [0, ..., 0, 1, ..., 1, ..., n_region-1, ..., n_region-1]

        pred_col = []
        gene_count_col = []
        for i in range(0, len(emb_idx), batch_size):
            use_emb = emb_idx[i : i + batch_size]
            use_reg = region_idx[i : i + batch_size]

            # cell_emb shape (n_emb, d_cell)
            cell_emb_mini_batch = cell_emb[use_emb].contiguous()
            # cond_emb shape (n_emb, d_cond)
            if cond_emb is None:
                cond_emb_mini_batch = None
            else:
                cond_emb_mini_batch = cond_emb[use_emb].contiguous()
                cond_emb_mini_batch = self._split_cond_emb_to_terms(cond_emb_mini_batch)

            # shared_data shape (n_emb, d_shared)
            if shared_data is None:
                shared_data_mini_batch = None
            else:
                shared_data_mini_batch = shared_data[use_emb].contiguous()
            _cond_emb_module_kwargs = (
                {} if shared_data is None else {"shared_data": shared_data_mini_batch}
            )

            # dna has shape (n_region, 4, seq_len)
            dna_mini_batch = dna[use_reg].contiguous()
            if gene_mode:
                gene_mask_mini_batch = gene_mask[use_reg].contiguous()
            # x0 has shape (n_region, n_emb, seq_len)
            # we need to select the paired region and embedding and get (bs, 1, seq_len)
            if self._forward_without_signal:
                x0_mini_batch = None
            else:
                if x0.ndim == 2:
                    x0_mini_batch = x0[use_reg].contiguous().unsqueeze(1)
                else:
                    x0_mini_batch = x0[use_reg, use_emb].contiguous().unsqueeze(1)

            with self._autocast_context():
                cond_ensemble = self._forward_cond_emb_module(
                    cell_emb=cell_emb_mini_batch,
                    cond_emb=cond_emb_mini_batch,
                    **_cond_emb_module_kwargs,
                )
                if gene_mode:
                    if isinstance(self.borzoi_gene_regions, BorzoiGeneQTLRegions):
                        y_pred_mini_batch, _stats, _slope = self.model.forward_qtl(
                            dna_mini_batch,
                            gene_mask=gene_mask_mini_batch,
                            embedding=cond_ensemble,
                            signal=x0_mini_batch,
                            crop=crop,
                        )
                        _stats = torch.nn.functional.sigmoid(_stats)
                        gene_count_col.append(
                            torch.concat([_stats, _slope], dim=1).detach()
                        )
                    else:
                        y_pred_mini_batch, gene_count_mini_batch = (
                            self.model.forward_gene(
                                dna_mini_batch,
                                gene_mask=gene_mask_mini_batch,
                                embedding=cond_ensemble,
                                signal=x0_mini_batch,
                                crop=crop,
                            )
                        )
                        gene_count_col.append(gene_count_mini_batch)
                else:
                    y_pred_mini_batch = self.model(
                        dna_mini_batch,
                        embedding=cond_ensemble,
                        signal=x0_mini_batch,
                        crop=crop,
                    )
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
        if gene_mode:
            if isinstance(self.borzoi_gene_regions, BorzoiGeneQTLRegions):
                gene_count = torch.cat(gene_count_col, dim=0)
                gene_count = rearrange(
                    gene_count,
                    "(n_region n_emb) d -> n_region n_emb d",
                    n_region=n_region,
                    n_emb=n_emb,
                )
            else:
                gene_count = torch.cat(gene_count_col, dim=0)
                gene_count = rearrange(
                    gene_count,
                    "(n_region n_emb) 1 1 -> n_region n_emb",
                    n_region=n_region,
                    n_emb=n_emb,
                ).detach()
            batch[f"{ypred_key}:gene_count"] = gene_count

        if crop and not self._embedding_only_mode:
            # also crop ytrue to allow metric calculation
            for key in ["__ytrue__:cond0", "__ytrue__:cond1"]:
                _data = batch[key]
                crop_radius = (_data.shape[-1] - self.model.crop_to_length) // 2
                batch[key] = _data[..., crop_radius:-crop_radius]

        batch[ypred_key] = y_pred.detach()
        return batch

    def _get_default_prediction_save_keys(self, mode="prediction"):
        SAVE_KEYS = [
            "__ytrue__:peak:cond0",
            "__ytrue__:peak:cond1",
            "__ypred__:peak:cond1",
            "__embedding__",
            "condition_pairs",
            "peak",
            "region",
            "region_name",
            "pseudobulk_ids",
        ]

        if mode == "gene_count_prediction":
            SAVE_KEYS.append("__ypred__:cond1:gene_count")
        return SAVE_KEYS

    def _model_gene_count_prediction_step(self, *args, **kwargs):
        return self._model_prediction_step(*args, gene_mode=True, **kwargs)

    def _create_fn_and_dataloader(
        self,
        dna_key,
        data_key,
        embedding_key,
        regions,
        region_names,
        pseudobulk_ids,
        add_true_data,
        batch_size,
    ):
        if self._embedding_only_mode:
            add_true_data = False

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
                batch["__ytrue__"] = data
            return batch

        pseudobulk_info_keys = ["cov_scale", embedding_key]
        if self.has_cond_emb:
            pseudobulk_info_keys.append("__conditionemb__")
        if getattr(self, "has_shared_data", False):
            pseudobulk_info_keys.append("__shared_data__")

        dataloader = self.datamanager.get_dataloader(
            regions=regions,
            region_names=region_names,
            batch_size=batch_size,
            add_dna=True,
            add_data=add_true_data,
            pseudobulk_subset=pseudobulk_ids,
            pseudobulk_info_keys=pseudobulk_info_keys,
            collate_fn=_collate_fn,
        )
        return dataloader

    def _get_default_stats_keys(self):
        STATS_KEYS = [
            "region",
            "peak",
            "pseudobulk_ids",
            "condition_pairs",
            "peak_pearsonr",
            "peak_r2",
        ]
        return STATS_KEYS

    def attribution_task(
        self,
        output_dir,
        regions_per_pseudobulk,
        pseudobulk_ids=None,
        batch_size=6,
        save_keys=(
            "__dna__:attr",
            "__signal__:attr",
            "region",
            "pseudobulk_id",
            "region_name",
        ),
        verbose=True,
        mode="attribution",
        save_first_batch=False,
    ):
        """
        Attribution task for BorzoiSignalPredictor.
        Compute the attribution on a set of regions and pseudobulk records.
        Then save the results to a file.
        """
        original_pid_to_pid = {}
        for pid, rec in self.pseudobulk_manager.items():
            pid_type = pid.split(":")[-1].split("-")[0]
            if pid_type == "ensemble":
                continue
            original_pid = rec.annotation["__pid__"]
            original_pid_to_pid[original_pid] = pid

        if pseudobulk_ids is not None:
            # translate pid
            pseudobulk_ids = [
                original_pid_to_pid.get(pid, pid) for pid in pseudobulk_ids
            ]
        else:
            pseudobulk_ids = list(original_pid_to_pid.values())

        if isinstance(regions_per_pseudobulk, dict):
            regions_per_pseudobulk = {
                original_pid_to_pid.get(k, k): v
                for k, v in regions_per_pseudobulk.items()
            }

        super().attribution_task(
            output_dir=output_dir,
            regions_per_pseudobulk=regions_per_pseudobulk,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            save_keys=save_keys,
            verbose=verbose,
            mode=mode,
            save_first_batch=save_first_batch,
        )

    def qtl_task(
        self,
        output_dir,
        qtl_table,
        pseudobulk_ids=None,
        batch_size=16,
        save_keys="default",
        verbose=True,
    ):
        """
        QTL task for BorzoiSignalPredictor.

        # TODO: since we do not use signal input for QTL task, we can disable add_true_data
        # this need later code assume x0 key can be missing
        """
        # change add_true_data default to True
        return super().qtl_task(
            output_dir=output_dir,
            qtl_table=qtl_table,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            save_keys=save_keys,
            add_true_data=True,
            verbose=verbose,
        )


class BorzoiSignalPredictorMultiModel(BorzoiSignalPredictor):
    """
    this class handles prediction task for trained BorzoiLoRAMulti model

    Although the model is trained for multi datasets, this class will only handle prediction of one dataset.
    """

    model_class = BorzoiLoRAMulti
    _embedding_only_mode = False

    def __init__(self, dataset_key, config):
        self._embedding_only_mode = config.get("embedding_only_mode", False)
        self.dataset_key = dataset_key
        config = self._prepare_multi_dataset_manager(config)

        super().__init__(config)

        # overwrite BorzoiSignalPredictor
        self.has_cond_emb = True  # always set true for Multi model
        self.has_shared_data = (
            self.config["train_config"]["shared_data_paths"] is not None
        )
        # check if the pseudobulk records have shared data in annotation
        if self.has_shared_data:
            pm = self.datamanager.pseudobulk_manager
            assert (
                "__shared_data__"
                in pm.pseudobulk_records[pm.pseudobulk_ids[0]].annotation
            ), (
                "Pseudobulk records must have '__shared_data__' in annotation, "
                "make sure the pseudobulk records in file has a '__shared_data__' key"
            )

    def _prepare_multi_dataset_manager(self, config):
        train_config = deepcopy(config["train_config"])

        # extract dataset information and put into config
        # after this, pass the config to BorzoiSignalPredictor.__init__
        if not isinstance(train_config, dict):
            with open(train_config) as f:
                train_config = json.load(f)

        # create the same dataset record manager as BorzoiMultiDataset in training
        # use this class to retrive dataset shared information, such as genome embedding
        dataset_records = train_config["dataset_records"]
        self._multi_dataset_record_manager = DatasetRecordManager(
            dataset_records=dataset_records,
            pseudobulker_cls=train_config["paired_mode"],
            shared_data_paths=train_config.get("shared_data_paths", None),
            use_regions=train_config["use_regions"],
        )

        # update config with cond_module_kwargs using DatasetRecordManager
        cond_module_kwargs = (
            self._multi_dataset_record_manager.make_cond_module_kwargs()
        )
        cur_kwargs = train_config["cond_module_kwargs"] or {}
        cur_kwargs.update(cond_module_kwargs)
        train_config["cond_module_kwargs"] = cur_kwargs

        dataset_key_info = dataset_records[self.dataset_key]

        # replace pseudobulk_path in train_config with predictor provided pseudobulk_records_path
        train_config["dataset_records"][self.dataset_key]["pseudobulk_path"] = config[
            "pseudobulk_records_path"
        ]
        config.setdefault("pseudobulk_records_path", config["pseudobulk_records_path"])
        config.setdefault("dataset_path", dataset_key_info["data_path"])
        config.setdefault("genome", dataset_key_info["genome"])

        config["train_config"] = train_config
        return config

    def _forward_cond_emb_module(
        self, cell_emb: torch.Tensor, cond_emb: torch.Tensor, **kwargs
    ):
        """
        Forward pass for the conditional embedding module.
        """
        dataset_idx = self._multi_dataset_record_manager.data_keys.index(
            self.dataset_key
        )
        # create shared embedding info
        dataset_genome_emb = (
            self._multi_dataset_record_manager.dataset_idx_to_genome_emb[dataset_idx]
        )

        bs = cell_emb.shape[0]
        genome_emb = torch.tensor(
            dataset_genome_emb, device=cell_emb.device, dtype=torch.float
        ).repeat(bs, 1)

        shared_emb = {"__genome__": genome_emb}
        if "shared_data" in kwargs:
            shared_emb["__shared_data__"] = kwargs["shared_data"]

        # forward cond emb module using single dataset mode
        cond_ensemble = self.model.cond_emb_module.forward_single_dataset(
            cell_emb=cell_emb,
            cond_emb=cond_emb,
            shared_emb=shared_emb,
            dataset_key=dataset_idx,
        )
        return cond_ensemble
