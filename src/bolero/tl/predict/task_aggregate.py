import pathlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from bolero.utils import understand_regions


def _get_attr_regions(batch):
    if "__dna__:attr" in batch:
        attr_size = batch["__dna__:attr"].shape[-1]
    else:
        attr_size = 524288

    attr_regions = understand_regions(batch["region"])
    attr_regions["Pos0Base"] = attr_regions[["Start", "End"]].sum(axis=1) // 2
    attr_regions["Start"] = attr_regions["Pos0Base"] - attr_size // 2
    attr_regions["End"] = attr_regions["Pos0Base"] + attr_size // 2
    attr_regions["Name"] = batch["region_name"]
    attr_regions.index = attr_regions["Name"]
    attr_regions.index.name = "region"
    return attr_regions


def _create_attr_ds(batch):
    batch_attr_regions = _get_attr_regions(batch)
    attr_keys = ["__dna__:attr", "__dna__:attr_seq"]
    batch_attr_ds = xr.Dataset(
        {
            key: xr.DataArray(
                batch[key],
                dims=["region", "base", "position"],
            )
            for key in attr_keys
        },
        coords={
            "region": batch_attr_regions.index,
            "base": list("ACGT"),
        },
    )
    return batch_attr_regions, batch_attr_ds


def _create_ref_alt_attr_ds(ref_batch, alt_batch):
    batch_attr_regions = _get_attr_regions(ref_batch)
    attr_keys = ["__dna__:attr", "__dna__:attr_seq"]
    batch_attr_ds = xr.Dataset(
        {
            key: xr.DataArray(
                np.stack([ref_batch[key], alt_batch[key]]),
                dims=["genotype", "region", "base", "position"],
            )
            for key in attr_keys
        },
        coords={
            "region": batch_attr_regions.index,
            "base": list("ACGT"),
            "genotype": ["ref", "alt"],
        },
    )
    return batch_attr_regions, batch_attr_ds


def _create_seqlet_ds(batch, batch_attr_regions):
    seqlets_info = batch["seqlets_info"]
    seqlets_info["attr_region"] = batch_attr_regions.index[seqlets_info["example_idx"]]
    seqlets_info["attr_region"] = seqlets_info["attr_region"].astype("category")
    seqlets_info["attr_region_chrom"] = seqlets_info["attr_region"].map(
        batch_attr_regions["Chromosome"]
    )
    seqlets_info["attr_region_start"] = (
        seqlets_info["attr_region"].map(batch_attr_regions["Start"]).astype(int)
    )
    del seqlets_info["example_idx"]

    keys = ["seqlets_attr", "seqlets_dna"]
    seqlet_ds = xr.Dataset(
        {
            key: xr.DataArray(
                batch[key],
                dims=["seqlet", "base", "position"],
            )
            for key in keys
        },
        coords={
            "base": list("ACGT"),
        },
    )
    return seqlets_info, seqlet_ds


def _create_ref_alt_seqlet_ds(ref_batch, alt_batch, batch_attr_regions):
    ref_seqlets_info = ref_batch["seqlets_info"]
    ref_seqlets_info["genotype"] = "ref"
    alt_seqlets_info = alt_batch["seqlets_info"]
    alt_seqlets_info["genotype"] = "alt"

    seqlets_info = pd.concat([ref_seqlets_info, alt_seqlets_info])
    seqlets_info["attr_region"] = batch_attr_regions.index[seqlets_info["example_idx"]]
    seqlets_info["attr_region"] = seqlets_info["attr_region"].astype("category")
    seqlets_info["attr_region_chrom"] = seqlets_info["attr_region"].map(
        batch_attr_regions["Chromosome"]
    )
    seqlets_info["attr_region_start"] = (
        seqlets_info["attr_region"].map(batch_attr_regions["Start"]).astype(int)
    )
    del seqlets_info["example_idx"]

    keys = ["seqlets_attr", "seqlets_dna"]
    seqlet_ds = xr.Dataset(
        {
            key: xr.DataArray(
                np.concatenate([ref_batch[key], alt_batch[key]]),
                dims=["seqlet", "base", "position"],
            )
            for key in keys
        },
        coords={
            "base": list("ACGT"),
        },
    )
    return seqlets_info, seqlet_ds


def _collect_attr_results(output_dir, pid_map, has_attr_ds=True):
    attr_region_col = []
    attr_ds_col = []
    seqlets_info_col = []
    seqlets_ds_col = []
    batch_paths = (output_dir / "batch").glob("*.joblib.gz")
    for batch_path in batch_paths:
        original_pid = pid_map[batch_path.name.split(".")[1]]
        batch = joblib.load(batch_path)

        if has_attr_ds:
            # full region attr scores (clipped at center 1024)
            batch_attr_regions, batch_attr_ds = _create_attr_ds(batch)
            batch_attr_regions["pseudobulk_id"] = original_pid
            attr_region_col.append(batch_attr_regions)
            attr_ds_col.append(batch_attr_ds)
        else:
            batch_attr_regions = _get_attr_regions(batch)

        # seqlet attr scores
        batch_seqlets_info, batch_seqlets_ds = _create_seqlet_ds(
            batch, batch_attr_regions
        )
        batch_seqlets_info["pseudobulk_id"] = original_pid
        seqlets_info_col.append(batch_seqlets_info)
        seqlets_ds_col.append(batch_seqlets_ds)

    return attr_region_col, attr_ds_col, seqlets_info_col, seqlets_ds_col


def _collect_mutation_attr_results(output_dir, pid_map):
    attr_region_col = []
    attr_ds_col = []
    seqlets_info_col = []
    seqlets_ds_col = []
    ref_batch_paths = (output_dir / "batch").glob("*.ref.joblib.gz")
    for batch_path in ref_batch_paths:
        original_pid = pid_map[batch_path.name.split(".")[1]]
        ref_batch = joblib.load(batch_path)
        alt_batch_path = batch_path.parent / batch_path.name.replace(
            ".ref.joblib.gz", ".alt.joblib.gz"
        )
        alt_batch = joblib.load(alt_batch_path)

        # full region attr scores (clipped at center 1024)
        batch_attr_regions, batch_attr_ds = _create_ref_alt_attr_ds(
            ref_batch, alt_batch
        )
        batch_attr_regions["pseudobulk_id"] = original_pid
        attr_region_col.append(batch_attr_regions)
        attr_ds_col.append(batch_attr_ds)

        # seqlet attr scores
        batch_seqlets_info, batch_seqlets_ds = _create_ref_alt_seqlet_ds(
            ref_batch, alt_batch, batch_attr_regions
        )
        batch_seqlets_info["pseudobulk_id"] = original_pid
        seqlets_info_col.append(batch_seqlets_info)
        seqlets_ds_col.append(batch_seqlets_ds)
    return attr_region_col, attr_ds_col, seqlets_info_col, seqlets_ds_col


def _get_pid_map(config):
    pid_map = pd.Series(
        {k: v.get("__pid__", k) for k, v in config["pseudobulk_records"].items()}
    )
    return pid_map


def _get_config(output_dir):
    config = joblib.load(output_dir / "config.joblib.gz")
    return config


class AggregateMixin:
    @staticmethod
    def aggregate_atac_attribution_results(output_dir):
        """
        Aggregate ATAC attribution results from all batches with ref and alt alleles.
        """
        output_dir = pathlib.Path(output_dir)
        config = _get_config(output_dir)
        pid_map = _get_pid_map(config)

        has_mutation = config["task_config"].get("qtl_table", None) is not None
        if has_mutation:
            attr_region_col, attr_ds_col, seqlets_info_col, seqlets_ds_col = (
                _collect_mutation_attr_results(output_dir, pid_map)
            )
        else:
            attr_region_col, attr_ds_col, seqlets_info_col, seqlets_ds_col = (
                _collect_attr_results(output_dir, pid_map)
            )

        attr_regions = pd.concat(attr_region_col)
        attr_ds = xr.concat(attr_ds_col, dim="region")
        attr_ds["__dna__:attr"] = attr_ds["__dna__:attr"].astype("float16")
        attr_ds["__dna__:attr_seq"] = attr_ds["__dna__:attr_seq"].astype("bool")
        columns = ["Chromosome", "Start", "End"]
        for col in columns:
            attr_ds[col] = attr_regions[col]
        attr_ds["pseudobulk_id"] = attr_regions["pseudobulk_id"].astype("object")

        seqlets_info = pd.concat(seqlets_info_col)
        seqlets_info.index = pd.Index(range(seqlets_info.shape[0]), name="seqlet")
        seqlets_ds = xr.concat(seqlets_ds_col, dim="seqlet")
        seqlets_ds["seqlets_attr"] = seqlets_ds["seqlets_attr"].astype("float16")
        seqlets_ds["seqlets_dna"] = seqlets_ds["seqlets_dna"].astype("bool")
        seqlets_ds["pseudobulk_id"] = seqlets_info["pseudobulk_id"].astype("object")
        seqlets_ds["Chromosome"] = seqlets_info["attr_region_chrom"]
        seqlets_ds["Start"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["flank_start"]
        )
        seqlets_ds["End"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["flank_end"]
        )
        seqlets_ds["SeqletStart"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["start"]
        )
        seqlets_ds["SeqletEnd"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["end"]
        )
        if has_mutation:
            seqlets_ds["genotype"] = seqlets_info["genotype"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            attr_regions.to_feather(
                output_dir / "attribution_regions.feather", compression="zstd"
            )
            attr_ds.to_zarr(output_dir / "region_attribution_and_seq.zarr", mode="w")

            seqlets_info.to_feather(
                output_dir / "seqlets_info.feather", compression="zstd"
            )
            seqlets_ds.to_zarr(
                output_dir / "seqlets_attribution_and_seq.zarr", mode="w"
            )
        return

    @staticmethod
    def aggregate_seqlet_attribution_results(output_dir):
        """
        Aggregate seqlet only attribution results from all batches.
        """
        output_dir = pathlib.Path(output_dir)
        config = _get_config(output_dir)
        pid_map = _get_pid_map(config)

        *_, seqlets_info_col, seqlets_ds_col = _collect_attr_results(
            output_dir, pid_map, has_attr_ds=False
        )

        seqlets_info = pd.concat(seqlets_info_col)
        seqlets_info.index = pd.Index(range(seqlets_info.shape[0]), name="seqlet")
        seqlets_ds = xr.concat(seqlets_ds_col, dim="seqlet")
        seqlets_ds["seqlets_attr"] = seqlets_ds["seqlets_attr"].astype("float16")
        seqlets_ds["seqlets_dna"] = seqlets_ds["seqlets_dna"].astype("bool")
        seqlets_ds["pseudobulk_id"] = seqlets_info["pseudobulk_id"].astype("object")
        seqlets_ds["Chromosome"] = seqlets_info["attr_region_chrom"]
        seqlets_ds["Start"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["flank_start"]
        )
        seqlets_ds["End"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["flank_end"]
        )
        seqlets_ds["SeqletStart"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["start"]
        )
        seqlets_ds["SeqletEnd"] = (
            seqlets_info["attr_region_start"].astype(int) + seqlets_info["end"]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            seqlets_info.to_feather(
                output_dir / "seqlets_info.feather", compression="zstd"
            )
            seqlets_ds.to_zarr(
                output_dir / "seqlets_attribution_and_seq.zarr", mode="w"
            )
        return

    @staticmethod
    def aggregate_qtl_results(output_dir, qtl_type="eqtl", save=True):
        """
        Aggregate eQTL results from all batches.
        """
        output_dir = pathlib.Path(output_dir)
        config = _get_config(output_dir)
        pid_map = _get_pid_map(config)

        ref_data_col = []
        alt_data_col = []
        region_col = []

        if qtl_type == "caqtl":
            suffix = ":peak"
        elif qtl_type == "eqtl":
            suffix = ":gene_count"
        else:
            raise ValueError(f"Invalid qtl type: {qtl_type}")

        batch_paths = (output_dir / "batch").glob("*.joblib.gz")
        for batch_path in batch_paths:
            batch = joblib.load(batch_path)
            ref_data_col.append(batch[f"__ypred__:ref{suffix}"])
            alt_data_col.append(batch[f"__ypred__:alt{suffix}"])
            region_col.append(batch["region_name"])

        pids = pd.Index(batch["pseudobulk_ids"][::2]).map(pid_map)
        regions = np.concatenate(region_col, axis=0)
        ref_data = pd.DataFrame(
            np.concatenate(ref_data_col, axis=0), index=regions, columns=pids
        )
        alt_data = pd.DataFrame(
            np.concatenate(alt_data_col, axis=0), index=regions, columns=pids
        )
        logfc = np.log2(alt_data / (ref_data + 1e-6))

        if save:
            ref_data.to_feather(output_dir / "ref_data.feather", compression="zstd")
            alt_data.to_feather(output_dir / "alt_data.feather", compression="zstd")
            logfc.to_feather(output_dir / "ref_alt_logfc.feather", compression="zstd")
        return ref_data, alt_data, logfc

    @staticmethod
    def aggregate_inference_results(output_dir, mode="prediction"):
        """
        Aggregate peak results from all batches.
        """
        output_dir = pathlib.Path(output_dir)
        config = _get_config(output_dir)
        pid_map = _get_pid_map(config)

        if mode.startswith("prediction"):
            pred_key = "__ypred__:peak"
            save_name = "pred_peak_data.feather"
        elif mode.startswith("gene_count_prediction"):
            pred_key = "__ypred__:cond1:gene_count"
            save_name = "pred_gene_count_data.feather"
        elif mode == "peak":
            pred_key = "__ypred__:peak"
            save_name = "pred_peak_data.feather"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        batch_paths = (output_dir / "batch").glob("*.joblib.gz")
        all_pred_data = []
        for path in batch_paths:
            batch = joblib.load(path)
            regions = batch["region_name"]
            pids = pd.Index(batch["pseudobulk_ids"][::2]).map(pid_map)
            pred_data = pd.DataFrame(batch[pred_key], index=regions, columns=pids)
            all_pred_data.append(pred_data)
        pred_data = pd.concat(all_pred_data)
        pred_data.to_feather(output_dir / save_name, compression="zstd")
        return pred_data

    @staticmethod
    def gather_peak_data(output_dir: str) -> None:
        """
        Gather peak true and prediction data into a single dataframe from all batches in output_dir.
        """
        batch_paths = list(Path(f"{output_dir}/batch/").glob("batch_*.joblib.gz"))
        print(f"{len(batch_paths)} batches in {output_dir}")

        # put back original pid
        first_batch = joblib.load(batch_paths[0])
        pids = pd.Index(first_batch["pseudobulk_ids"])
        if pids[0].startswith("ensemble|data:"):
            # get original pids for signal model
            pids = pids[1::2]  # only use data pid, skip ensembles (which is the cond0)
            config = joblib.load(f"{output_dir}/config.joblib.gz")
            prec = config["pseudobulk_records"]
            original_pid_map = {k: v["__pid__"] for k, v in prec.items()}
            pids = pids.map(original_pid_map)
        if "__ytrue__:peak:cond1" in first_batch:
            true_peak_key = "__ytrue__:peak:cond1"
            pred_peak_key = "__ypred__:peak:cond1"
        else:
            true_peak_key = "__ytrue__:peak"
            pred_peak_key = "__ypred__:peak"

        has_true_data = True
        true_peak_data = []
        pred_peak_data = []
        peak_bed = []
        for path in batch_paths:
            batch = joblib.load(path)
            if has_true_data:
                try:
                    true_peak_data.append(batch[true_peak_key])
                except KeyError:
                    has_true_data = False
                    print(
                        "True peak data not found in batch, skip true peak data saving."
                    )
            pred_peak_data.append(batch[pred_peak_key])
            peak_bed.append(batch["peak"])
        peak_bed = pd.concat(peak_bed)

        pred_peak_data = np.concatenate(pred_peak_data, axis=1).T.astype("float32")
        pred_peak_data = pd.DataFrame(
            pred_peak_data, index=peak_bed["Name"].values, columns=pids
        )
        pred_peak_data = pred_peak_data[~pred_peak_data.index.duplicated()].sort_index()
        pred_peak_data.to_feather(f"{output_dir}/pred_peak_data.feather")

        if has_true_data:
            true_peak_data = np.concatenate(true_peak_data, axis=1).T.astype("float32")
            true_peak_data = pd.DataFrame(
                true_peak_data, index=peak_bed["Name"].values, columns=pids
            )
            true_peak_data = true_peak_data[
                ~true_peak_data.index.duplicated()
            ].sort_index()
            true_peak_data.to_feather(f"{output_dir}/true_peak_data.feather")

        # remove duplicate peak bed rows
        peak_bed = peak_bed[~peak_bed["Name"].duplicated()].copy()
        return

    @staticmethod
    def gather_gene_data(output_dir: str) -> None:
        """
        Gather gene data into single dataframe from all batches in output_dir.
        """
        output_dir = pathlib.Path(output_dir)
        config = _get_config(output_dir)
        pid_map = _get_pid_map(config)

        batch_paths = (output_dir / "batch").glob("*.joblib.gz")

        all_pred_data = []
        for path in batch_paths:
            batch = joblib.load(path)
            columns = batch["pseudobulk_ids"][1::2]
            index = batch["region_name"]
            pred_data = pd.DataFrame(
                batch["__ypred__:cond1:gene_count"], index=index, columns=columns
            )
            all_pred_data.append(pred_data)
        all_pred_data = pd.concat(all_pred_data).T

        all_pred_data.index = all_pred_data.index.map(pid_map)
        all_pred_data.to_feather(
            output_dir / "pred_gene_data.feather", compression="zstd"
        )
        return
