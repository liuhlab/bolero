import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from bolero.utils import understand_regions


def _get_attr_regions(batch):
    attr_size = batch["__dna__:attr"].shape[-1]
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


def _collect_attr_results(output_dir, pid_map):
    attr_region_col = []
    attr_ds_col = []
    seqlets_info_col = []
    seqlets_ds_col = []
    batch_paths = (output_dir / "batch").glob("*.joblib.gz")
    for batch_path in batch_paths:
        original_pid = pid_map[batch_path.name.split(".")[1]]
        batch = joblib.load(batch_path)

        # full region attr scores (clipped at center 1024)
        batch_attr_regions, batch_attr_ds = _create_attr_ds(batch)
        batch_attr_regions["pseudobulk_id"] = original_pid
        attr_region_col.append(batch_attr_regions)
        attr_ds_col.append(batch_attr_ds)

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


class AggregateMixin:
    @staticmethod
    def aggregate_atac_attribution_results(output_dir):
        """
        Aggregate ATAC attribution results from all batches with ref and alt alleles.
        """
        output_dir = pathlib.Path(output_dir)
        config = joblib.load(output_dir / "config.joblib.gz")
        pid_map = pd.Series(
            {k: v["__pid__"] for k, v in config["pseudobulk_records"].items()}
        )

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            attr_regions.to_feather(output_dir / "attribution_regions.feather")
            attr_ds.to_zarr(output_dir / "region_attribution_and_seq.zarr", mode="w")

            seqlets_info.to_feather(output_dir / "seqlets_info.feather")
            seqlets_ds.to_zarr(
                output_dir / "seqlets_attribution_and_seq.zarr", mode="w"
            )
        return
