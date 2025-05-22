import json
from pathlib import Path

import joblib
import numpy as np
import pyranges as pr
import torch


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
    return config
