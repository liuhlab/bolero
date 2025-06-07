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

    def _create_datamanager(self):
        config = self.config
        genome = config["genome"]
        db_path = config["dataset_path"]
        pseudobulk_records_path = config["pseudobulk_records_path"]
        parallel = config.get("parallel", 8)

        dm = GenericGenomeDataManager(genome=genome)
        dm.add_pseudobulk_records(pseudobulk_records_path)
        dm.add_parquet_dataset("parquet", db_path, parallel=parallel)
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

    def get_prediction_dataloader(
        self,
        regions,
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
        self._pre_callbacks = self._get_pre_prediction_callbacks()
        self._post_callbacks = self._get_post_prediction_callbacks()

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
                batch = self._model_prediction_step(
                    batch,
                    batch_size=batch_size,
                )
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

        config_path = output_dir / "config.joblib.gz"
        self._save_task_configs(
            task_config={
                "regions": regions,
                "pseudobulk_ids": pseudobulk_ids,
                "save_keys": save_keys,
            },
            output_path=config_path,
        )
        return


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
        paired_mode = self.config.get("paired_mode", "condition")
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

        # prediction mode can be "ode" or "velocity"
        # "ode" means using the ODE solver to predict the trajectory
        # "velocity" means using the model to predict the velocity field
        self._prediction_mode = self.config.get("prediction_mode", "ode")
        assert self._prediction_mode in ("ode", "velocity"), (
            f"Invalid prediction mode: {self._prediction_mode}. "
            "Must be one of 'ode' or 'velocity'."
        )

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
        callback_configs = [
            # calculate paired profile correlation
            (
                "process_paired_data",
                {
                    "data_keys": ["__ytrue__", "__conditionemb__", "__embedding__"],
                    "split_dim": [1, 0, 0],
                },
            ),
        ]
        callbacks = self._prepare_callbacks(callback_configs)
        return callbacks

    def _get_post_prediction_callbacks(self):
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

    def _model_prediction_step(
        self,
        batch,
        dna_key="__dna__",
        x0_key="__ytrue__:cond0",
        cell_embedding_key="__embedding__:cond0",
        cond_embedding_key="__conditionemb__:cond1",
        time_range_key="__timerange__",
        batch_size=16,
    ):
        """
        Forward pass through the model.
        """
        # prepare data
        dna = batch[dna_key]
        cell_emb = batch[cell_embedding_key]
        cond_emb = batch[cond_embedding_key]
        x0 = batch[x0_key]
        t_range = batch[time_range_key]

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
            cond_emb_mini_batch = cond_emb[use_emb].contiguous()
            # dna has shape (n_region, 4, seq_len)
            dna_mini_batch = dna[use_reg].contiguous()
            # x0 has shape (n_region, n_emb, seq_len)
            # we need to select the paired region and embedding and get (bs, 1, seq_len)
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

        if self._prediction_mode == "velocity":
            # y_pred is the velocity, we need to add it to the initial condition
            batch["__ypred__:cond1"] = x0 + y_pred.detach()
        else:
            batch["__ypred__:cond1"] = y_pred.detach()
        batch["__ypred__:delta"] = batch["__ypred__:cond1"] - batch["__ytrue__:cond0"]
        return batch

    def _model_prediction_step_nosde(
        self,
    ):
        """
        Forward pass through the model without ODE solver, just use the velocity prediction.
        """

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

        dataloader = self.datamanager.get_dataloader(
            regions=regions,
            batch_size=batch_size,
            add_dna=True,
            add_data=add_true_data,
            pseudobulk_subset=pseudobulk_ids,
            pseudobulk_info_keys=["cov_scale", embedding_key, "__conditionemb__"],
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


# TODO: attribution task - just prediction, collapse lora, no true data
