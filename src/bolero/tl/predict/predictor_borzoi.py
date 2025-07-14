import gc
import pathlib
import tempfile
import time
from collections import defaultdict
from shutil import rmtree
from typing import Generator

import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import torch
import xarray as xr
from einops import rearrange

from bolero.tl.model.borzoi.model import Borzoi
from bolero.tl.model.borzoi.model_flow import (
    BorzoiLoRAFlowPredictor as _BorzoiFlowModelWithODESolver,
)
from bolero.tl.model.borzoi.model_flow import (
    BorzoiLoRAFlowPredictorFP as _BorzoiFlowModelWithSDESolverFP,
)
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.pseudobulk.paired_pseudobulk import PAIRED_PSEUDOBULKER_CLS_DICT
from bolero.utils import understand_regions

from .datamanager import GenericGenomeDataManager
from .predictor import GenericPredictor
from .utils import load_config


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
    ):
        self.model = model
        self.model_dna_length = 524288
        self.model_resolution = 32
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

        self._callbacks = []
        self.qtl_manager = None

    def register_qtl_manager(self, qtl_table):
        """
        Register the QTL manager with the given QTL table.

        This is required in qtl task
        """
        from .qtlmanager import QTLManager

        self.qtl_manager = QTLManager(qtl_table, resolution=32)
        return

    def _create_datamanager(self):
        config = self.config
        genome = config["genome"]
        db_path = config["dataset_path"]
        pseudobulk_records_path = config["pseudobulk_records_path"]
        parallel = config.get("parallel", 8)

        dm = GenericGenomeDataManager(genome=genome)
        dm.add_pseudobulk_records(pseudobulk_records_path)
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

        batch[ypred_key] = y_pred.detach()
        return batch

    def _model_qtl_step(self, batch, batch_size):
        # turn "__dna__" into "__dna__:ref" and "__dna__:alt"
        qtl = self.qtl_manager
        assert qtl is not None, "QTL manager is not registered."

        batch = qtl.mutate_dna(batch)
        mutation_cols = ["ref", "alt"]
        with torch.inference_mode():
            for mutation_col in mutation_cols:
                batch = self._model_prediction_step(
                    batch,
                    dna_key=f"__dna__:{mutation_col}",
                    batch_size=batch_size,
                    ypred_key=f"__ypred__:{mutation_col}",
                )
                batch = qtl.get_peak_sum(
                    batch,
                    ypred_key=f"__ypred__:{mutation_col}",
                )
        # add peak information to the batch
        batch = qtl.add_peak_info(batch)
        return batch

    def _get_post_attribution_callbacks(self):
        return []

    def _get_post_prediction_callbacks(self):
        callbacks = [
            # calc pearsonr on last dim (seq_len)
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
                ("extract_peak", {"peak_bed": self.peak_df}),
                # this step adds the peak data to the batch:
                # "__ytrue__:peak" and "__ypred__:peak"
                #
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
            return self._get_post_prediction_callbacks()
        elif mode == "attribution":
            return self._get_post_attribution_callbacks()
        elif mode == "qtl":
            return self._get_post_qtl_callbacks()
        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Must be one of 'prediction' or 'attribution'."
            )

    def _create_fn_and_dataloader(
        self,
        dna_key,
        data_key,
        embedding_key,
        regions,
        pseudobulk_ids,
        add_true_data,
        batch_size,
    ):
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
        return dataloader

    def iter_debug_batch(
        self,
        dna_key="dna",
        embedding_key="embedding",
        pseudobulk_ids=None,
        add_true_data=True,
        batch_size=16,
        mode="prediction",
        regions=None,
        trigger_model=False,
    ):
        """Iterable for debugging"""
        # 1. Get regions
        if regions is None:
            regions = self.get_fold_regions(test_only=True, minimize_overlap=True)
        regions: list[str] = self._valid_and_sort_regions(regions, return_list=True)

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
        assert mode in ["prediction", "qtl"], (
            f"Invalid mode: {mode}. " "Must be one of 'prediction' or 'qtl'."
        )
        # 1. Get regions
        regions: list[str] = self._valid_and_sort_regions(regions, return_list=True)

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
                elif mode == "qtl":
                    step_fn = self._model_qtl_step
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
                f"{timer['infer']/timer['counter']:.3f}s per batch\n"
                f"Callback time: {timer['callback']:.2f}s or "
                f"{timer['callback']/timer['counter']:.3f}s per batch\n"
                f"(total {timer['counter']} batches)\n"
            )

    def _collapse_model(self, emb):
        """Collapse model for inference given an embedding vector."""
        # TODO: get pseudobulk information and collapse model
        if isinstance(emb, pd.Series):
            emb = torch.from_numpy(emb.values)
        elif isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        emb = emb.cuda().unsqueeze(0)

        emb_model = self.model.collapse_lora(emb)
        emb_model = emb_model.eval().cuda()
        return emb_model

    def _prepare_attr_model(self, batch) -> torch.nn.Module:
        embedding = batch["__embedding__"]
        model = self._collapse_model(embedding)
        attr_model = BorzoiInputXGradient(model=model)
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

    def _model_attribution_step(self, model, batch, attr_batch_size):
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
            regions = regions_per_pseudobulk
            regions: list[str] = self._valid_and_sort_regions(
                regions, return_list=True, standard_size=524288
            )
            regions_per_pseudobulk = {pid: regions for pid in pseudobulk_ids}
        return regions_per_pseudobulk

    def get_attribution_dataloader(
        self,
        regions_per_pseudobulk,
        pseudobulk_ids,
        dna_key="dna",
        embedding_key="embedding",
        batch_size=6,
        verbose=True,
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

        timer = {"infer": 0, "callback": 0, "total": 0, "counter": 0}
        start = time.time()
        for pseudobulk_id in pseudobulk_ids:
            if isinstance(regions_per_pseudobulk, dict):
                regions = regions_per_pseudobulk[pseudobulk_id]
            else:
                regions = regions_per_pseudobulk

            dataloader = self._create_fn_and_dataloader(
                dna_key=dna_key,
                data_key=data_key,
                embedding_key=embedding_key,
                regions=regions,
                # attribution task will iterate one pseudobulk at a time
                pseudobulk_ids=[pseudobulk_id],
                # attribution task will not use true data in parquet
                add_true_data=False,
                # batch size for attr is small, but we set it larger here so we save less batch files
                batch_size=batch_size * 100,
            )

            # 3. Get callbacks
            self._pre_callbacks = self._get_pre_callbacks("attribution")
            self._post_callbacks = self._get_post_callbacks("attribution")

            attr_model = None

            # 4. Inference and callback
            for batch in dataloader:
                batch["pseudobulk_id"] = pseudobulk_id

                if attr_model is None:
                    model = self._prepare_attr_model(batch)
                # Pre-inference callbacks
                t = time.time()
                batch = self.apply_callbacks(batch, "pre")
                timer["callback"] += time.time() - t

                # Inference step
                t = time.time()
                batch = self._model_attribution_step(
                    model, batch, attr_batch_size=batch_size
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
        # TODO: clean up memory during batches

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

    def prediction_task(
        self,
        output_dir,
        regions="test_regions",
        downsample_regions=None,
        downsample_seed=0,
        pseudobulk_ids=None,
        batch_size=16,
        save_keys=None,
        stats_keys=None,
        verbose=True,
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
            The regions to predict. If "test_regions", use the test regions.
        downsample_regions: int
            The number of regions to downsample. If None, use all regions.
        downsample_seed: int
            The seed for downsampling.
        pseudobulk_ids: list[str]
            The pseudobulk ids to use. If None, use all pseudobulk ids.
        batch_size: int
            The batch size for prediction.
        save_keys: list[str]
            The keys to save in the output file. If None, save all keys.
        verbose: bool
            Whether to print the progress.
        """
        if isinstance(regions, str) and regions == "test_regions":
            regions = self.get_fold_regions(test_only=True, minimize_overlap=True)
        else:
            regions = understand_regions(regions)
        if downsample_regions is not None:
            regions = regions.sample(n=downsample_regions, random_state=downsample_seed)

        dataloader = self.get_prediction_dataloader(
            regions=regions,
            add_true_data=True,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            verbose=verbose,
        )

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
        save_keys = [] if save_keys is None else save_keys
        # stats to collect across batches
        default_stats_keys = self._get_default_stats_keys()
        if stats_keys is not None:
            stats_keys = list(set(default_stats_keys + stats_keys))
        else:
            stats_keys = default_stats_keys

        batch_stats_paths = []
        save_batch = {}
        for idx, batch in enumerate(dataloader):
            if idx == 0 and verbose:
                self._print_batch(batch, prefix="Dataloader")

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
        return

    def _filter_valid_qtl_regions(self):
        db = self.datamanager.datasets["parquet"]
        regions = self.qtl_manager.regions
        regions_bed = pr.PyRanges(regions.reset_index().iloc[:, [1, 2, 3, 0]])
        valid_regions_bed = regions_bed.overlap(
            db.region_lookup_bed.merge(), how="containment"
        )
        new_regions = regions.reindex(valid_regions_bed.df["Name"].values)
        if new_regions.shape[0] != regions.shape[0]:
            print(
                f"Filtered {regions.shape[0] - new_regions.shape[0]} invalid regions "
                "from QTL table that are not fully overlapped with the parquet dataset."
            )
        return new_regions

    def qtl_task(
        self,
        output_dir,
        qtl_table,
        pseudobulk_ids=None,
        batch_size=16,
        save_keys="default",
        add_true_data=False,
        verbose=True,
    ):
        """
        Prediction task for Borzoi.
        Compute the prediction on a set of regions and pseudobulk records.
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
        """
        if isinstance(save_keys, str) and save_keys == "default":
            save_keys = [
                "region",
                "pseudobulk_ids",
                "mutation_id",
                "qtl_peaks",
                "__ypred__:ref:peak",
                "__ypred__:alt:peak",
            ]

        # add qtl manager and get regions from the qtl manager
        self.register_qtl_manager(qtl_table)
        regions = self._filter_valid_qtl_regions()

        dataloader = self.get_prediction_dataloader(
            regions=regions,
            add_true_data=add_true_data,
            pseudobulk_ids=pseudobulk_ids,
            batch_size=batch_size,
            verbose=verbose,
            mode="qtl",
        )

        output_dir = pathlib.Path(output_dir).absolute().resolve()
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving batches to {batch_dir}")

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
        for idx, batch in enumerate(dataloader):
            if idx == 0 and verbose:
                self._print_batch(batch, prefix="Dataloader")

            # save the batch to a file
            save_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if k in save_keys:
                    save_batch[k] = v
            # data to save for each batch
            save_path = batch_dir / f"batch_{idx}.joblib.gz"
            joblib.dump(save_batch, save_path)

        if verbose and len(save_batch) > 0:
            self._print_batch(save_batch, prefix="Saved")
        return

    def _check_finished_attribution_batches(self, batch_dir, regions_per_pseudobulk):
        # make a pid-by-region bool table
        pid_region_table_log = {}
        for pid, regions in regions_per_pseudobulk.items():
            pid_region_table_log[pid] = pd.Series(
                np.ones(len(regions), dtype=bool), index=regions
            )
        pid_region_table_log = pd.DataFrame(pid_region_table_log)

        # collect all finished batches
        batch_paths = batch_dir.glob("*.joblib.gz")
        bids = []
        for path in batch_paths:
            bids.append(int(path.name.split(".")[0].split("_")[1]))
            batch = joblib.load(path)
            regions = batch["region"]
            pid = batch["pseudobulk_id"]
            pid_region_table_log.loc[regions, pid] = False

        # only run unfinished pid and regions
        cur_bid = max(bids) + 1 if len(bids) > 0 else 0
        regions_per_pseudobulk_torun = {
            pid: regions_bool[regions_bool].index.tolist()
            for pid, regions_bool in pid_region_table_log.items()
            if regions_bool.any()
        }
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
        save_keys=("__dna__:attr", "region", "pseudobulk_ids"),
        verbose=True,
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
        """
        output_dir = pathlib.Path(output_dir).absolute().resolve()
        batch_dir = output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Saving batches to {batch_dir}")

        if pseudobulk_ids is None:
            pseudobulk_ids = self.pseudobulk_manager.pseudobulk_ids
        if isinstance(regions_per_pseudobulk, str):
            regions_per_pseudobulk = understand_regions(regions_per_pseudobulk)
        regions_per_pseudobulk = self._prepare_attr_regions(
            pseudobulk_ids=pseudobulk_ids, regions_per_pseudobulk=regions_per_pseudobulk
        )
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
            )

            save_batch = {}
            for idx, batch in enumerate(dataloader):
                idx = idx + cur_bid
                if idx == 0 and verbose:
                    self._print_batch(batch, prefix="Dataloader")

                # save the batch to a file
                save_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    if k in save_keys:
                        save_batch[k] = v
                # data to save for each batch
                pid = batch["pseudobulk_id"]
                save_path = batch_dir / f"batch_{idx}.{pid}.joblib.gz"
                joblib.dump(save_batch, save_path)

            if verbose and len(save_batch) > 0:
                self._print_batch(save_batch, prefix="Saved")

        self._save_pseudobulk_attr_ds(
            pseudobulk_ids=pseudobulk_ids, batch_dir=batch_dir
        )
        return


class BorzoiMultiHeadPredictor(BorzoiPredictor):
    def _model_prediction_step(
        self,
        batch,
        dna_key="__dna__",
        ypred_key="__ypred__",
        batch_size=16,
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
                y_pred_mini_batch = self.model(dna_mini_batch)
                pred_col.append(y_pred_mini_batch)
        y_pred = torch.cat(pred_col, dim=0)
        # y_pred shape (n_region, n_emb, seq_len)

        batch[ypred_key] = y_pred.detach()
        return batch


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
        paired_mode = _train_config.get("paired_mode", "condition")
        pseudobulker_cls = PAIRED_PSEUDOBULKER_CLS_DICT[paired_mode]
        paired_pseudobulker = pseudobulker_cls(
            pseudobulk_and_ot_info=config["pseudobulk_records_path"],
            emb_key="embedding",
            downsample_pseudobulk=None,
            barcode_order=None,
            seed=42,
        )
        designs = config["designs"]
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
                    "ytrue_key": "__ytrue__:peak:delta",
                    "ypred_key": "__ypred__:peak:delta",
                    "permute": (1, 0),  # (sample, peak) -> (peak, sample)
                    "output_key": "peak_delta_pearsonr",
                    "cumulative": True,
                },
            ),
            (
                "r2_score",
                {
                    "ytrue_key": "__ytrue__:peak:delta",
                    "ypred_key": "__ypred__:peak:delta",
                    "permute": (1, 0),
                    "output_key": "peak_delta_r2",
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
            "peak_delta_pearsonr",
            "peak_delta_r2",
        ]
        return STATS_KEYS


class BorzoiFlowPredictor(BorzoiPairPredictor):
    """
    BorzoiFlowPredictor is a predictor for Borzoi models that uses ODEs to predict
    the trajectory of the model given an initial condition.
    It is used for flow-based models.
    """

    def __init__(self, config):
        super().__init__(config)
        self._ode_solver = None

        self.has_cond_emb = self.config.get("cond_emb_dim") is not None

        # prediction mode can be "ode" or "velocity"
        # "ode" means using the ODE solver to predict the trajectory
        # "velocity" means using the model to predict the velocity field
        self._prediction_mode = self.config.get("prediction_mode", "ode")
        assert self._prediction_mode in ("ode", "velocity"), (
            f"Invalid prediction mode: {self._prediction_mode}. "
            "Must be one of 'ode' or 'velocity'."
        )
        self._model_without_signal = self.config.get("_nosignal", False)

    def _create_solver(self) -> _BorzoiFlowModelWithODESolver:
        cfm_class = self.config.get("cfm_class", "cfm")
        if cfm_class == "fp":
            solver = _BorzoiFlowModelWithSDESolverFP(
                model=self.model,
                **self.config.get("solver_kwargs", {}),
            )
        else:
            solver = _BorzoiFlowModelWithODESolver(
                model=self.model,
                **self.config.get("solver_kwargs", {}),
            )
        return solver

    @property
    def ode_solver(self) -> _BorzoiFlowModelWithODESolver:
        """
        Get the ODE solver for the model.
        If not set, create a new one.
        """
        if self._ode_solver is None:
            self._ode_solver = self._create_solver()
        return self._ode_solver

    def _get_pre_prediction_callbacks(self):
        _data_keys = ["__ytrue__", "__embedding__"]
        split_dim = [1, 0]
        if self.has_cond_emb:
            _data_keys.append("__conditionemb__")
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

    def _get_pre_callbacks(self, mode="prediction"):
        if mode == "prediction":
            return self._get_pre_prediction_callbacks()
        elif mode == "attribution":
            return self._get_pre_attribution_callbacks()
        elif mode == "qtl":
            # same as prediction
            return self._get_pre_prediction_callbacks()
        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Must be one of 'prediction' or 'attribution'."
            )

    def _get_post_attribution_callbacks(self):
        return []

    def _get_post_prediction_callbacks(self, prediction_mode="prediction"):
        # add paired task callbacks
        callbacks = [
            (
                "extract_peak",
                {
                    "peak_bed": self.peak_df,
                    "data_keys": [
                        "__ytrue__:cond0",
                        "__ytrue__:cond1",
                        "__ytrue__:delta",
                        "__ypred__:cond1",
                        "__ypred__:delta",
                    ],
                },
            ),
            (
                "rename",
                {
                    "name_map": {
                        "__ytrue__:cond0:peak": "__ytrue__:peak:cond0",
                        "__ytrue__:cond1:peak": "__ytrue__:peak:cond1",
                        "__ytrue__:delta:peak": "__ytrue__:peak:delta",
                        "__ypred__:cond1:peak": "__ypred__:peak:cond1",
                        "__ypred__:delta:peak": "__ypred__:peak:delta",
                    }
                },
            ),
            # calculate paired profile correlation
            (
                "pearsonr",
                {
                    "ytrue_key": "__ytrue__:peak:delta",
                    "ypred_key": "__ypred__:peak:delta",
                    "permute": (1, 0),  # (sample, peak) -> (peak, sample)
                    "output_key": "peak_delta_pearsonr",
                    "cumulative": True,
                },
            ),
            (
                "r2_score",
                {
                    "ytrue_key": "__ytrue__:peak:delta",
                    "ypred_key": "__ypred__:peak:delta",
                    "permute": (1, 0),
                    "output_key": "peak_delta_r2",
                    "cumulative": True,
                },
            ),
        ]
        cb = self._prepare_callbacks(callbacks)
        return cb

    def _get_post_qtl_callbacks(self):
        return []

    def _get_post_callbacks(self, mode="prediction"):
        """
        Get the post callbacks for the model.
        """
        if mode == "prediction":
            return self._get_post_prediction_callbacks()
        elif mode == "attribution":
            return self._get_post_attribution_callbacks()
        elif mode == "qtl":
            return self._get_post_qtl_callbacks()
        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                "Must be one of 'prediction' or 'attribution'."
            )

    def _split_cond_emb_to_terms(self, cond_emb):
        # split the cond_emb into dict of terms using cond_encoder in pseudobulker
        pseduobulker = self.paired_pseudobulker
        condition_encoder = getattr(pseduobulker, "condition_encoder", None)
        if condition_encoder is not None:
            cond_emb_terms = condition_encoder.split_cond_emb(cond_emb)
        else:
            cond_emb_terms = cond_emb
        return cond_emb_terms

    def _prepare_attr_model(self, batch):
        embedding = batch["__embedding__"]
        if self.has_cond_emb:
            cond_emb = batch["__conditionemb__"]
            cond_emb = self._split_cond_emb_to_terms(cond_emb)
        else:
            cond_emb = None
        time = torch.zeros([embedding.shape[0], 1])
        time = time.type_as(embedding).to(embedding.device)

        agg_emb = self.model.cond_flow_module(
            cell_emb=embedding, cond_emb=cond_emb, time=time
        )
        model = self._collapse_model(agg_emb)
        attr_model = BorzoiInputXGradient(model=model)
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

    def _model_attribution_step(self, model, batch, attr_batch_size):
        """
        Forward pass through the model to get the attribution.
        """
        if self._model_without_signal:
            batch = self._dna_attr_step(model, batch, attr_batch_size)
        else:
            batch = self._dna_sig_attr_step(model, batch, attr_batch_size)

        # clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        return batch

    def _model_prediction_step(
        self,
        batch,
        dna_key="__dna__",
        x0_key="__ytrue__:cond0",
        cell_embedding_key="__embedding__:cond0",
        cond_embedding_key="__conditionemb__:cond1",
        time_range_key="__timerange__",
        batch_size=16,
        ypred_key="__ypred__:cond1",
    ):
        """
        Forward pass through the model.
        """
        # prepare data
        dna = batch[dna_key]
        cell_emb = batch[cell_embedding_key]
        if self.has_cond_emb:
            cond_emb = batch[cond_embedding_key]
        else:
            cond_emb = None
        x0 = batch[x0_key]
        t_range = batch[time_range_key]

        # x0 = torch.log1p(x0)

        n_emb = cell_emb.shape[0]
        n_region = dna.shape[0]

        emb_idx = torch.arange(n_emb).repeat(n_region)
        # [0, 1, ..., n_emb-1, ..., 0, 1, ..., n_emb-1]
        region_idx = torch.arange(n_region).repeat_interleave(n_emb)
        # [0, ..., 0, 1, ..., 1, ..., n_region-1, ..., n_region-1]

        pred_col = []
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
            # dna has shape (n_region, 4, seq_len)
            dna_mini_batch = dna[use_reg].contiguous()
            # x0 has shape (n_region, n_emb, seq_len)
            # we need to select the paired region and embedding and get (bs, 1, seq_len)
            if self._model_without_signal:
                x0_mini_batch = None
            else:
                x0_mini_batch = x0[use_reg, use_emb].contiguous().unsqueeze(1)

            with self._autocast_context():
                if self._prediction_mode == "velocity":
                    y_pred_mini_batch = self.ode_solver.predict_vt(
                        x_0=x0_mini_batch,
                        cell_emb=cell_emb_mini_batch,
                        cond_emb=cond_emb_mini_batch,
                        dna_one_hot=dna_mini_batch,
                    )
                    # This is equivalent to single step ODE approximation
                    # y_pred_mini_batch here is actual y_pred delta
                else:
                    y_pred_mini_batch = self.ode_solver.predict(
                        x_0=x0_mini_batch,
                        t_range=t_range,
                        cell_emb=cell_emb_mini_batch,
                        cond_emb=cond_emb_mini_batch,
                        dna_one_hot=dna_mini_batch,
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

        # if self._prediction_mode == "velocity":
        #     # y_pred is the velocity, we need to add it to the initial condition
        #     x1 = x0 + y_pred.detach()
        # else:
        #     x1 = y_pred.detach()
        # batch["__ypred__:cond1"] = torch.expm1(x1).clamp(min=0.0)

        if self._prediction_mode == "velocity":
            # y_pred is the velocity, we need to add it to the initial condition
            batch[ypred_key] = x0 + y_pred.detach()
        else:
            batch[ypred_key] = y_pred.detach()

        try:
            batch["__ypred__:delta"] = (
                batch["__ypred__:cond1"] - batch["__ytrue__:cond0"]
            )
        except KeyError:
            pass
        # joblib.dump(batch, "debug_batch.joblib.gz")
        return batch

    def _create_fn_and_dataloader(
        self,
        dna_key,
        data_key,
        embedding_key,
        regions,
        pseudobulk_ids,
        add_true_data,
        batch_size,
    ):
        def _collate_fn(batch, add_data=add_true_data):
            # rename keys
            batch["__dna__"] = batch.pop(dna_key)
            batch["__embedding__"] = batch.pop(embedding_key)
            batch["__timerange__"] = [0, 1]

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

        dataloader = self.datamanager.get_dataloader(
            regions=regions,
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
            "peak_delta_pearsonr",
            "peak_delta_r2",
        ]
        return STATS_KEYS

    def attribution_task(
        self,
        output_dir,
        regions_per_pseudobulk,
        pseudobulk_ids=None,
        batch_size=6,
        save_keys=("__dna__:attr", "region", "pseudobulk_ids"),
        verbose=True,
    ):
        """
        Attribution task for BorzoiFlowPredictor.
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
        QTL task for BorzoiFlowPredictor.
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
