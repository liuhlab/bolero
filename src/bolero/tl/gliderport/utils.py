import pathlib
import re
import subprocess
from functools import lru_cache


@lru_cache
def get_bucket_to_local_map():
    """
    Get the mapping from GCS bucket to local mount point.
    """
    cmd = ["cat", "/etc/mtab"]
    p = subprocess.run(cmd, capture_output=True, encoding="utf8")
    bucket_to_local = {}
    for line in p.stdout.split("\n"):
        ll = line.split(" ")
        if len(ll) < 3:
            continue
        if ll[2] != "fuse.gcsfuse":
            continue
        bucket_to_local[ll[0]] = ll[1]
    return bucket_to_local


def gcsfuse_friendly_copy(source_path, target_path, create_target_dir=True):
    """
    Copy data from source_path to target_dir, if target_dir is a GCS bucket mount with gcsfuse, then use gsutil to copy the data to its actual gcs location.

    Parameters
    ----------
    source_path : str
        Source path.
    target_path : str
        Target path.
    """
    source_path = pathlib.Path(source_path).absolute().resolve()
    suffix = "/*" if source_path.is_dir() else ""
    source_path = str(source_path).rstrip("/") + suffix
    bucket_to_local = get_bucket_to_local_map()
    for bucket, local_path in bucket_to_local.items():
        gsutil = True
        if source_path.startswith(local_path):
            source_path = re.sub(f"^{local_path}", f"gs://{bucket}", source_path)
            break
    else:
        gsutil = False

    target_path = pathlib.Path(target_path).absolute().resolve()
    if not target_path.parent.exists():
        target_path.parent.mkdir(exist_ok=True, parents=True)
    suffix = "/" if target_path.is_dir() else ""
    target_path = str(target_path).rstrip("/") + suffix
    for bucket, local_path in bucket_to_local.items():
        gsutil = True
        if target_path.startswith(local_path):
            target_path = re.sub(f"^{local_path}", f"gs://{bucket}", target_path)
            break
    else:
        gsutil = False

    if gsutil:
        cp_cmd = f"gsutil -m cp -r {source_path} {target_path}"
        subprocess.check_call(
            cp_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        # only a normal cp
        cp_cmd = f"cp -r {source_path} {target_path}"
        subprocess.check_call(
            cp_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return
