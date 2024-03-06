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


def gcsfuse_friendly_copy(source_path, target_dir):
    """
    Copy data from source_path to target_dir, if target_dir is a GCS bucket mount with gcsfuse, then use gsutil to copy the data to its actual gcs location.

    Parameters
    ----------
    source_path : str
        Source path.
    target_dir : str
        Target directory.
    """
    source_path = str(pathlib.Path(source_path).absolute().resolve())
    cur_dir = str(pathlib.Path(target_dir).absolute().resolve())
    
    bucket_to_local = get_bucket_to_local_map()
    for bucket, local_path in bucket_to_local.items():
        if cur_dir.startswith(local_path):
            gsurl = re.sub(f"^{local_path}", f"gs://{bucket}", cur_dir)
            break
    else:
        gsurl = None
        
    if gsurl is not None:
        cp_cmd = f"gsutil -m cp -r {source_path} {gsurl}/"
        subprocess.check_call(
            cp_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        # only a normal cp
        cp_cmd = f"cp -r {source_path} {cur_dir}/"
        subprocess.check_call(
            cp_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return