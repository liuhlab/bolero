import pyranges as pr
import pandas as pd
import pathlib
import torch
import numpy as np
from typing import Union
import bolero
import shutil
import subprocess


from bolero.utils import pathlib


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


def get_package_dir():
    package_dir = pathlib.Path(bolero.__file__).parent
    return package_dir


def get_default_save_dir(save_dir):
    """Get the default save directory for bolero."""
    if save_dir is None:
        # check if "/ref/bolero" exists
        _my_default = pathlib.Path("/ref/bolero")
        if _my_default.exists():
            save_dir = _my_default
        else:
            save_dir = get_package_dir()
    save_dir = pathlib.Path(save_dir).absolute()
    return save_dir


def get_file_size_gbs(self, url):
    """Get the file size from a URL."""
    cmd = f"curl -sI {url} | grep -i Content-Length | awk '{{print $2}}'"
    size = subprocess.check_output(cmd, shell=True).decode().strip()
    size = int(size) / 1024**3
    return size


def download_file(url, local_path):
    """Download a file from a url to a local path using wget or curl"""
    local_path = pathlib.Path(local_path)

    if local_path.exists():
        return

    temp_path = local_path.parent / (local_path.name + ".temp")
    # download with wget
    if shutil.which("wget"):
        subprocess.check_call(
            ["wget", "-O", temp_path, url],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
    # download with curl
    elif shutil.which("curl"):
        subprocess.check_call(
            ["curl", "-o", temp_path, url],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
    else:
        raise RuntimeError("Neither wget nor curl found on system")
    # rename temp file to final file
    temp_path.rename(local_path)
    return
