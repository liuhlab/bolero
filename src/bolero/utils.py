import pathlib
import shutil
import subprocess
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pyranges as pr
from pyarrow import ArrowInvalid
from pyarrow.fs import FileSystem, LocalFileSystem

import bolero


def get_fs_and_path(path: Union[str, pathlib.Path]) -> Tuple[FileSystem, str]:
    """
    Get the file system and path from a given URI or local path.

    Parameters
    ----------
    path : str or pathlib.Path
        The URI or local path.

    Returns
    -------
    Tuple[FileSystem, str]
        A tuple containing the file system and the resolved path.

    Raises
    ------
    ArrowInvalid
        If the given path is not a valid URI.

    Notes
    -----
    If the given path is a valid URI, the function will use `FileSystem.from_uri()`
    to get the file system and resolved path. If the given path is not a valid URI,
    the function will use `LocalFileSystem()` and `pathlib.Path()` to get the file system
    and resolved path respectively.
    """
    try:
        fs, path = FileSystem.from_uri(path)
    except ArrowInvalid:
        fs = LocalFileSystem()
        path = str(pathlib.Path(path).absolute().resolve())
    return fs, path


def try_gpu():
    """
    Try to use GPU if available.
    """
    import torch

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
    """
    Parse a list of region names into a PyRanges object or a DataFrame.

    Parameters
    ----------
        names (list): A list of region names in the format "chromosome:start-end".
        as_df (bool, optional): If True, return the result as a DataFrame. Default is False.

    Returns
    -------
        PyRanges or DataFrame: A PyRanges object representing the parsed regions, or a DataFrame if `as_df` is True.
    """
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
    """
    Parse a region name in the format 'c:s-e' and return the components.

    Parameters
    ----------
    name : str
        The region name to parse.

    Returns
    -------
    tuple
        A tuple containing the components of the region name:
        - c : str
            The first component of the region name.
        - s : int
            The start position of the region.
        - e : int
            The end position of the region.
    """
    c, se = name.split(":")
    s, e = se.split("-")
    s = int(s)
    e = int(e)
    return c, s, e


def get_package_dir():
    """
    Get the directory path of the bolero package.

    Returns
    -------
    package_dir : pathlib.Path
        The directory path of the bolero package.
    """
    package_dir = pathlib.Path(bolero.__file__).parent
    return package_dir


def get_default_save_dir(save_dir):
    """
    Get the default save directory for bolero.

    Parameters
    ----------
    save_dir : str or pathlib.Path, optional
        The save directory to use. If not provided, the function will attempt
        to find a default save directory.

    Returns
    -------
    pathlib.Path
        The default save directory for bolero.

    Notes
    -----
    If `save_dir` is not provided, the function will first check if the
    directory "/ref/bolero" exists. If it does, that directory will be used
    as the default save directory. If not, it will check if the directory
    "{home_dir}/ref/bolero" exists, where `home_dir` is the user's home
    directory. If that directory exists, it will be used as the default save
    directory. If neither directory exists, the function will fall back to
    `get_package_dir()` to determine the default save directory.

    The returned save directory will be an absolute `pathlib.Path` object.

    """
    if save_dir is None:
        _my_default = pathlib.Path("/ref/bolero")
        home_dir = pathlib.Path.home()
        _my_default2 = pathlib.Path(f"{home_dir}/ref/bolero")
        if _my_default.exists():
            save_dir = _my_default
        elif _my_default2.exists():
            save_dir = _my_default2
        else:
            save_dir = get_package_dir()
    save_dir = pathlib.Path(save_dir).absolute()
    return save_dir


def get_file_size_gbs(url):
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
