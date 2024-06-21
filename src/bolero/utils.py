import pathlib
import shutil
import subprocess
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pyranges as pr
import torch
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
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
        home_dir = pathlib.Path.home()
        _my_defaults = [
            pathlib.Path("/ref/bolero"),
            pathlib.Path(f"{home_dir}/ref/bolero"),
            pathlib.Path(f"{home_dir}/data/bolero"),
        ]
        save_dir = None
        for _default in _my_defaults:
            if _default.exists():
                save_dir = _default
                break
        if save_dir is None:
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


def init(
    visible_devices: tuple[int] = None,
    verbose=False,
    object_spilling=True,
    num_cpus=None,
    object_store_memory_gb=None,
    object_store_memory_ratio=0.5,
    _enable_lineage_reconstruction=False,
    _ray_max_errored_blocks=3,
):
    """
    Set up the environment for bolero.

    Parameters
    ----------
    visible_devices : tuple[int], optional
        The visible GPU devices to use. Default is None.
    verbose : bool, optional
        If True, enable verbose output. Default is False.
    object_spilling : bool, optional
        If True, enable ray object spilling. Default is False.
    num_cpus : int, optional
        The number of CPUs to use in ray.init job. Default is None, which will use
        `os.cpu_count() - 1`.
    object_store_memory_gb : int, optional
        The amount of memory in GBs to use for ray's object store. Default is None.
    """
    # CUDA
    import os

    if visible_devices is not None:
        if isinstance(visible_devices, int):
            visible_devices = (visible_devices,)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_devices))
        if verbose:
            print(
                "Setting CUDA_VISIBLE_DEVICES to:", os.environ["CUDA_VISIBLE_DEVICES"]
            )

    import torch

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if verbose:
        print("Enabled torch cudnn")

    import ray

    # ray core
    if not _enable_lineage_reconstruction:
        # disable ray's lineage reconstruction, as this produces memory leaks
        # https://github.com/ray-project/ray/issues/31421#issuecomment-1371865009
        # Note that disable lineage reconstruction will disallow object fault tolerance
        os.environ["RAY_lineage_pinning_enabled"] = "0"
        # https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html
        os.environ["RAY_TASK_MAX_RETRIES"] = "0"
        if verbose:
            print("Disabled ray lineage reconstruction and task retries")

    # get number of cpus
    if num_cpus is None:
        num_cpus = max(1, os.cpu_count() - 1)
    # get system memory size
    if object_store_memory_gb is None:
        # in bytes
        sys_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2)
        object_store_memory = int(sys_memory * object_store_memory_ratio)
    else:
        object_store_memory = int(object_store_memory_gb * 1024**3)
    ray.init(
        num_cpus=num_cpus,
        object_store_memory=object_store_memory,
        ignore_reinit_error=True,
        _system_config={
            "automatic_object_spilling_enabled": object_spilling,
        },
        runtime_env={},
    )

    # ray data
    from ray.data import DataContext

    context = DataContext.get_current()
    context.enable_progress_bars = verbose
    context.max_errored_blocks = _ray_max_errored_blocks
    return
