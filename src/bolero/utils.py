import pyranges as pr
import pandas as pd
import pathlib
import torch
import numpy as np
from typing import Union


def try_gpu():
    """
    Try to use GPU if available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def understand_regions(regions, as_df=False, return_names=False):
    """
    From various inputs, return a clear output. Return pyranges by default.
    """
    if isinstance(regions, pr.PyRanges):
        pass
    elif isinstance(regions, pd.DataFrame):
        regions = pr.PyRanges(regions)
    elif isinstance(regions, Union[str, pathlib.Path]):
        regions = pr.read_bed(regions)
    elif isinstance(regions, Union[list, tuple, pd.Index, np.ndarray, pd.Series]):
        regions = parse_region_names(regions)
    else:
        raise ValueError("bed must be a PyRanges, DataFrame, str or Path")
    if as_df:
        return regions.df
    if return_names:
        return regions.Name.to_list()
    return regions


def parse_region_names(names, as_df=False):
    bed_record = []
    for name in names:
        c, se = name.split(":")
        s, e = se.split("-")
        bed_record.append([c, s, e, name])
    bed = pr.PyRanges(
        pd.DataFrame(bed_record, columns=["Chromosome", "Start", "End", "Name"])
    )
    if as_df:
        return bed.df
    return bed


def parse_region_name(name):
    c, se = name.split(":")
    s, e = se.split("-")
    s = int(s)
    e = int(e)
    return c, s, e
