import pathlib
from tempfile import mkdtemp

import joblib
import numpy as np
import pandas as pd
import pyBigWig
import ray
from scipy.sparse import csc_matrix, lil_matrix


@ray.remote
def _aggregate_pid_data(pid, pid_files, chrom_sizes, temp_dir, use_keys, resolution=32):
    """
    Aggregate region data into chromosomal CSC matrices for each pseudobulk.
    """
    pid_dicts = [joblib.load(path) for path in pid_files]
    pid_data = {}
    for use_key in use_keys:
        # Use LIL matrix for efficient construction, then convert to CSC
        chrom_lil_matrix = {
            chrom: lil_matrix((1, size // resolution))
            for chrom, size in chrom_sizes.items()
        }
        for pid_dict in pid_dicts:
            data = pid_dict[use_key]
            regions = pid_dict["regions"]
            for rid, (chrom, start, end) in enumerate(regions):
                bin_start = int(np.round(start / resolution))
                size = int((end - start) / resolution)
                bin_end = bin_start + size
                chrom_lil_matrix[chrom][0, bin_start:bin_end] = data[rid, :]

        # Convert LIL to CSC for final storage (efficient for column operations)
        chrom_csc_matrix = {}
        for chrom in chrom_lil_matrix.keys():
            chrom_csc_matrix[chrom] = csc_matrix(chrom_lil_matrix[chrom])
            chrom_csc_matrix[chrom].eliminate_zeros()
        pid_data[use_key] = chrom_csc_matrix

    joblib.dump(pid_data["ytrue"], f"{temp_dir}/{pid}.ytrue_chrom_csc_matrix.joblib.gz")
    joblib.dump(pid_data["ypred"], f"{temp_dir}/{pid}.ypred_chrom_csc_matrix.joblib.gz")


def _csc_to_bigwig(csc_file, chrom_sizes, resolution, output_file):
    # Load the CSC matrix data
    chrom_csc_data = joblib.load(csc_file)

    # Create BigWig file
    with pyBigWig.open(str(output_file), "w") as bw:
        # Write header with chromosome sizes
        bw.addHeader(list(chrom_sizes.items()))

        # Write data for each chromosome
        for chrom in chrom_sizes.keys():
            csc_matrix = chrom_csc_data[chrom]
            if csc_matrix.nnz == 0:  # Skip empty matrices
                continue

            # Convert CSC matrix to dense array for BigWig
            coo = csc_matrix.tocoo()
            starts = coo.col * resolution
            values = coo.data

            # Add intervals to BigWig
            bw.addEntries(
                chrom, starts.tolist(), values=values.tolist(), span=resolution
            )
    return


# TODO: deal with batch region overlap issue


class TrackAggregator:
    def __init__(
        self,
        output_dir,
        chrom_sizes,
        resolution=32,
        ytrue_key="__ytrue__:cond1",
        ypred_key="__ypred__:cond1",
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.track_dir = self.output_dir / "tracks"
        self.track_dir.mkdir(exist_ok=True)
        self._tmp_dir = None

        self.chrom_sizes = chrom_sizes
        self.resolution = resolution
        self.ytrue_key = ytrue_key
        self.ypred_key = ypred_key
        _use_keys = []
        if ytrue_key is not None:
            _use_keys.append(ytrue_key)
        if ypred_key is not None:
            _use_keys.append(ypred_key)
        self.use_keys = _use_keys
        assert (
            len(self.use_keys) > 0
        ), "At least one of ytrue_key or ypred_key must be provided"

        self.batch_paths = list(self.output_dir.glob("batch/batch*.gz"))

        config = joblib.load(self.output_dir / "config.joblib.gz")
        self.name_map = {
            k: v["__pid__"] for k, v in config["pseudobulk_records"].items()
        }

    @property
    def tmp_dir(self):
        """
        Get the temporary directory for the track aggregator.
        """
        if self._tmp_dir is None or not self._tmp_dir.exists():
            self._tmp_dir = pathlib.Path(mkdtemp())
        return self._tmp_dir

    def _dump_pid_data(self):
        # dump pid data from each batch
        for bid, path in enumerate(self.batch_paths):
            batch = joblib.load(path)

            pseudobulk_ids = batch["pseudobulk_ids"]
            pseudobulk_ids = pd.Index(pseudobulk_ids[1::2]).map(self.name_map)
            for pid in pseudobulk_ids:
                pid_dir = pathlib.Path(self.tmp_dir) / pid
                pid_dir.mkdir(exist_ok=True)

            regions = []
            for region in batch["region"]:
                chrom, coords = region.split(":")
                start, end = map(int, coords.split("-"))
                # adjust start and end due to clip
                regions.append([chrom, start + 512, end - 512])

            ytrue_track = batch[self.ytrue_key].swapaxes(
                0, 1
            )  # shape (n_pseudobulk, n_region, seq_len)
            ypred_track = batch[self.ypred_key].swapaxes(
                0, 1
            )  # shape (n_pseudobulk, n_region, seq_len)
            for idx, pid in enumerate(pseudobulk_ids):
                pid_dict = {
                    "ytrue": ytrue_track[idx],
                    "ypred": ypred_track[idx],
                    "regions": regions,
                }
                joblib.dump(pid_dict, f"{self.tmp_dir}/{pid}/{bid}.joblib.gz")

        futures = []
        for pid in pseudobulk_ids:
            # load all files for one pid, save data into true and pred bigwig file
            pid_files = list(pathlib.Path(self.tmp_dir).glob(f"{pid}/*.joblib.gz"))
            f = _aggregate_pid_data.remote(
                pid,
                pid_files,
                chrom_sizes=self.chrom_sizes,
                temp_dir=self.tmp_dir,
                resolution=self.resolution,
                use_keys=self.use_keys,
            )
            futures.append(f)
        _ = ray.get(futures)

    def _csc_to_bigwig(self):
        """
        Convert CSC matrix files to BigWig format.
        Loads files matching pattern *.*_chrom_csc_matrix.joblib.gz from tmp_dir
        and saves as BigWig files with pattern *.*_chrom_csc_matrix.bw in output_dir.
        """
        # Find all CSC matrix files
        csc_files = list(self.tmp_dir.glob("*.*_chrom_csc_matrix.joblib.gz"))

        for csc_file in csc_files:
            # Extract the base name for output file
            base_name = csc_file.stem.replace(".joblib.gz", "")
            output_file = self.track_dir / f"{base_name}.bw"

            _csc_to_bigwig(csc_file, self.chrom_sizes, self.resolution, output_file)

    def dump_track(self):
        """Dump the track data from separate batches into pseudobulk separated bigwig files."""
        self._dump_pid_data()
        self._csc_to_bigwig()
