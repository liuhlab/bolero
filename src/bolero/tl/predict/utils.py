import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyranges as pr
import torch
from torchmetrics.functional import r2_score


def validate_region(region_bed: pr.PyRanges, chrom_sizes: dict[str, int]):
    """
    Validate the region bed file.

    1. start < end
    2. region chrom and coordinates are within the chromosome sizes
    """
    rdf = region_bed.df
    assert rdf["Start"].min() >= 0, "Start coordinate must be >= 0"

    assert (
        rdf["End"] > rdf["Start"]
    ).all(), "End coordinate must be greater than Start coordinate"

    for chrom, rdf_chrom in rdf.groupby("Chromosome", observed=True):
        try:
            size = chrom_sizes[chrom]
        except KeyError as e:
            raise ValueError(f"Chromosome {chrom} not found in chromosome sizes") from e

        assert (
            rdf_chrom["End"].max() <= size
        ), f"End coordinate must be <= {size} for chromosome {chrom}"
    return


def get_device():
    """
    Get the device to be used for PyTorch operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def convert_np_to_torch(x):
    """
    Convert numpy arrays to PyTorch tensors, while preserving the original type
    """
    if isinstance(x, np.ndarray):
        # Check if it's numeric or boolean
        if np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.bool_):
            return torch.from_numpy(x)
        else:
            return x  # Keep as numpy array (e.g., string dtype)
    elif isinstance(x, (list, tuple)):
        # Recursively apply for sequences
        return type(x)(convert_np_to_torch(i) for i in x)
    elif isinstance(x, dict):
        # Recursively apply for dictionaries
        return {k: convert_np_to_torch(v) for k, v in x.items()}
    else:
        return x  # Leave unchanged


def load_config(config) -> dict:
    """
    Load the config file.
    """
    if isinstance(config, str):
        config_path = Path(config)
        if config_path.suffix == ".json":
            with open(config_path) as f:
                config = json.load(f)
        elif config_path.suffix in (".joblib", "joblib.gz"):
            config = joblib.load(config_path)
        elif config_path.suffix == ".pkl":
            with open(config_path, "rb") as f:
                config = joblib.load(f)
        else:
            raise ValueError("Config file must be a .json or .pkl file")
    else:
        config = deepcopy(config)
    return config


def multi_level_peak_stats(output_dir, precomputed_region_group_path=None):
    """
    Generate five quantile cutoffs based on true value across sample STD.
    For each cutoff, select corresponding peaks and calculate profile/sample pearson corr and R2 metrics.

    Parameters
    ----------
    output_dir : str
        The output directory containing the true and predicted peak data.
    precomputed_region_group_path : str, optional
        The path to a precomputed region group file, if provided, peak group in this file will be used.
    """
    # 1. Load true and pred peaks
    all_true_peak_data = pd.read_feather(f"{output_dir}/true_peak_data.feather")
    all_pred_peak_data = pd.read_feather(f"{output_dir}/pred_peak_data.feather")
    # make sure true and pred has exact order
    all_pred_peak_data = all_pred_peak_data.reindex(
        index=all_true_peak_data.index, columns=all_true_peak_data.columns
    )
    assert not all_pred_peak_data.isna().values.any()

    # 2. Compute region groups based on true value sample std OR use pre-computed groups
    # small idx group is lowly variable / accessible regions
    # large idx group is highly variable / accessible regions
    if precomputed_region_group_path is None:
        region_groups = pd.qcut(
            all_true_peak_data.std(axis=1), [0, 0.2, 0.4, 0.6, 0.8, 1]
        ).cat.codes
    else:
        region_groups = joblib.load(precomputed_region_group_path)["region_groups"]

    # 3. Compute the four metrics
    keys = ["profile_corr", "sample_corr", "profile_r2", "sample_r2"]
    region_group_and_stats = {"region_groups": region_groups, **{k: [] for k in keys}}
    for group in region_groups.unique():
        regions = region_groups[region_groups >= group]
        true_peak_data = all_true_peak_data.reindex(regions.index)
        pred_peak_data = all_pred_peak_data.reindex(regions.index)

        # R2
        true = torch.as_tensor(true_peak_data.values)
        pred = torch.as_tensor(pred_peak_data.values)
        profile_r2 = r2_score(pred, true, multioutput="raw_values").numpy()
        profile_r2 = pd.Series(profile_r2, index=true_peak_data.columns)
        sample_r2 = r2_score(pred.T, true.T, multioutput="raw_values").numpy()
        sample_r2 = pd.Series(sample_r2, index=true_peak_data.index)

        # correlation
        profile_corr = true_peak_data.corrwith(pred_peak_data)
        sample_corr = true_peak_data.T.corrwith(pred_peak_data.T)

        # collect
        region_group_and_stats["profile_corr"].append(
            pd.DataFrame({"value": profile_corr, "group": group})
        )
        region_group_and_stats["sample_corr"].append(
            pd.DataFrame({"value": sample_corr, "group": group})
        )
        region_group_and_stats["profile_r2"].append(
            pd.DataFrame({"value": profile_r2, "group": group})
        )
        region_group_and_stats["sample_r2"].append(
            pd.DataFrame({"value": sample_r2, "group": group})
        )

    # 4. Compute metric summary
    stats_summary = []
    for k in keys:
        data = pd.concat(region_group_and_stats[k])
        region_group_and_stats[k] = data

        value_mean = data.groupby("group")["value"].mean()
        value_std = data.groupby("group")["value"].std()
        summary = (
            pd.DataFrame({"std": value_std, "mean": value_mean}).unstack().reset_index()
        )
        summary.columns = ["stat", "group", "value"]
        summary["metric"] = k
        stats_summary.append(summary)
    stats_summary = pd.concat(stats_summary)
    region_group_and_stats["stats_summary"] = stats_summary

    # 5. save all stats
    joblib.dump(region_group_and_stats, f"{output_dir}/region_group_and_stats.joblib")
