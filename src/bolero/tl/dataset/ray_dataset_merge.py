import pathlib
import shutil
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import ray
from scipy.sparse import csc_matrix, csr_matrix, hstack, vstack
from tqdm import tqdm

from bolero import Genome
from bolero.pp.genome_chunk_dataset import (
    array_to_compressed_bytes_dict,
    csr_matrix_to_compressed_bytes_dict,
)
from bolero.tl.dataset.ray_dataset import RayGenomeChunkDataset
from bolero.utils import understand_regions


class PseudobulkMerge:
    def __init__(
        self,
        cell_to_cluster: pd.Series,
        barcode_order: dict[pd.Index],
        output_prefix="pseudobulk",
    ):
        self.n_clusters = cell_to_cluster.unique().size
        self.output_prefix = output_prefix

        cell_to_prefix = pd.Series(
            cell_to_cluster.index.map(lambda i: i.split(":")[0]),
            index=cell_to_cluster.index,
        )

        prefix_to_row_to_cluster = {}
        for prefix, barcode_to_cluster in cell_to_cluster.groupby(cell_to_prefix):
            barcode_to_cluster.index = barcode_to_cluster.index.map(
                lambda i: i.split(":")[1]
            )
            row_to_cluster = (
                barcode_to_cluster.reindex(barcode_order[prefix]).fillna(-1).astype(int)
            )
            row_to_cluster.index = range(row_to_cluster.size)
            prefix_to_row_to_cluster[prefix] = row_to_cluster
        self.prefix_to_row_to_cluster = prefix_to_row_to_cluster

    def __call__(self, row):
        """Generate pseudobulk from single cell rows."""
        prefix_to_row_to_cluster = self.prefix_to_row_to_cluster

        region_chunk_size = row[list(prefix_to_row_to_cluster.keys())[0]].shape[1]

        # only take rows but not sum, sum after vstack is much faster
        cluster_arr_col = defaultdict(list)
        for prefix, row_to_cluster in prefix_to_row_to_cluster.items():
            prefix_arr = row[prefix].astype("float32")
            for cluster, rows in row_to_cluster.groupby(row_to_cluster, observed=True):
                if cluster == -1:
                    continue
                cluster_arr_col[cluster].append(prefix_arr[rows.index])

        total_matrix = np.zeros((self.n_clusters, region_chunk_size), dtype="float32")
        for cluster, parts in cluster_arr_col.items():
            if len(parts) == 0:
                continue
            elif len(parts) == 1:
                parts = parts[0]
            else:
                parts = vstack(parts)
            if parts.shape[0] > 1:
                total_matrix[cluster] = parts.sum(axis=0).A1
            else:
                total_matrix[cluster] = parts.toarray()[0]

        final_dict = csr_matrix_to_compressed_bytes_dict(
            self.output_prefix, csr_matrix(total_matrix)
        )

        for k, v in row.items():
            if k in prefix_to_row_to_cluster:
                continue
            if isinstance(v, np.ndarray):
                final_dict.update(array_to_compressed_bytes_dict(k, v))
            else:
                final_dict[k] = v
        return final_dict


class RayGenomeChunkDatasetPseudobulkMerge(RayGenomeChunkDataset):
    def _merge_pseudobulk(self, dataset, cell_to_cluster, concurrency):
        fn = PseudobulkMerge
        fn_constructor_kwargs = {
            "cell_to_cluster": cell_to_cluster,
            "barcode_order": self.barcode_order,
        }

        dataset = dataset.map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )
        return dataset

    def get_processed_dataset(self, chroms, cell_to_cluster, concurrency=8) -> None:
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        concurrency = max(concurrency - 4, 8)

        dataset = self._read_parquet(chroms=chroms)

        # filter meta region length equal to self.window_size
        dataset = self._filter_meta_region_length(dataset=dataset)

        decompress_cpu = int(max(2, concurrency * 0.25))
        merge_cpu = int(max(2, concurrency * 0.75))

        # from compressed bytes to tensor (cell/sample by meta-region matrix) and other information
        dataset = self._compressed_bytes_to_tensor(
            dataset=dataset,
            concurrency=(1, decompress_cpu),
        )

        # merge cell into pseudobulk
        dataset = self._merge_pseudobulk(
            dataset=dataset,
            cell_to_cluster=cell_to_cluster,
            concurrency=(1, merge_cpu),
        )
        return dataset

    def merge_chrom_and_write_parquet(
        self,
        output_dir,
        chrom,
        cell_to_cluster,
        concurrency=8,
        num_rows_per_file=10,
        rows_per_chunk=200,
    ):
        """
        Merge single rows into pseudobulk and write to parquet per chromosome.
        """
        output_dir = pathlib.Path(f"{output_dir}/{chrom}")
        output_flag = output_dir / ".SUCCESS"
        if output_flag.exists():
            print(f"Skip {chrom} because {output_flag} exists.")
            return
        else:
            if output_dir.exists():
                shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True)
        work_ds = self.get_processed_dataset([chrom], cell_to_cluster, concurrency)
        chunk_paths = []
        row_col = []
        for idx, row in enumerate(work_ds.iter_rows()):
            row_col.append(row)
            if (idx + 1) % rows_per_chunk == 0:
                chunk_path = f"{output_dir}/{idx}.joblib"
                joblib.dump(row_col, chunk_path)
                row_col = []
                chunk_paths.append(chunk_path)
        if len(row_col) > 0:
            chunk_path = f"{output_dir}/{idx}.joblib"
            joblib.dump(row_col, chunk_path)
            chunk_paths.append(chunk_path)
        del work_ds

        # write parquet
        for chunk_path in chunk_paths:
            ds = ray.data.from_items(joblib.load(chunk_path))
            idx = pathlib.Path(chunk_path).name.split(".")[0]
            ds.write_parquet(f"{output_dir}/{idx}", num_rows_per_file=num_rows_per_file)
            pathlib.Path(chunk_path).unlink()
        output_flag.touch()
        return

    def merge(self, output_dir, cell_to_cluster, concurrency=64, num_rows_per_file=10):
        """
        Merge single rows into pseudobulk and write to parquet per chromosome.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chroms = self.get_chroms()
        for chrom in chroms:
            self.merge_chrom_and_write_parquet(
                output_dir=output_dir,
                chrom=chrom,
                cell_to_cluster=cell_to_cluster,
                concurrency=concurrency,
                num_rows_per_file=num_rows_per_file,
                rows_per_chunk=200,
            )

        n_cluster = cell_to_cluster.unique().size
        row_names = {
            "tn5_bias": pd.Index(["tn5_bias"]),
            "pseudobulk": pd.Index(list(range(n_cluster))),
        }
        joblib.dump(row_names, f"{output_dir}/row_names.joblib")

        config = self._ds_config.copy()
        config["num_rows_per_file"] = num_rows_per_file
        joblib.dump(config, f"{output_dir}/config.joblib")
        return


def _merge_bp_to_bins(window_data, merge_resolution):
    nrow, length = window_data.shape
    bins = length // merge_resolution
    assert length % merge_resolution == 0, "length % resolution needs to be 0"

    final_data = csc_matrix((nrow, bins))
    for k in range(merge_resolution):
        idx = np.arange(k, length, merge_resolution)
        final_data += window_data[:, idx]
    return final_data


class MergeGenomeChunkDatasetByBins:
    """
    This generator takes in pseudobulk-by-bases RayGenomeChunkDataset and merge them into pseudobulk-by-bins per chromosome.

    This is preparing for Borzoi model training.
    """

    def __init__(
        self,
        genome,
        merge_window_size=600000,
        merge_step_size=200000,
        merge_resolution=32,
        prefix="pseudobulk",
    ):
        self.genome = Genome(genome)
        self.window_bed = self.genome.make_windows(
            window_size=merge_window_size, step=merge_step_size, as_df=True
        )
        self.merge_step_size = merge_step_size
        self.merge_window_size = merge_window_size
        self.merge_resolution = merge_resolution
        self.prefix = prefix

    def _get_region_data(self, dataset_path, chrom):
        """
        Here I take out data from whole chromosome,
        which is memory intensive (not really at pseudobulk level).
        """
        ds = RayGenomeChunkDataset(dataset_path, genome=self.genome)
        dataset = ds._read_parquet(chroms=[chrom])

        # filter meta region length equal to self.window_size
        dataset = ds._filter_meta_region_length(dataset=dataset)

        # from compressed bytes to tensor (cell/sample by meta-region matrix) and other information
        dataset = ds._compressed_bytes_to_tensor(
            dataset=dataset,
            concurrency=(1, 8),
        )
        region_to_data = {}

        for batch in dataset.iter_batches(batch_size=50):
            for region, csr_mat in zip(batch["region"], batch[self.prefix]):
                region_to_data[region] = csr_mat.tocsc()

        return region_to_data

    def merge_chrom(self, dataset_path, chrom, output_dir, num_rows_per_file=20):
        """Merge a single chromosome"""
        chrom_out_path = pathlib.Path(f"{output_dir}/{chrom}")
        success_flag = chrom_out_path / ".SUCCESS"
        if success_flag.exists():
            print(f"Skip {chrom} because {success_flag} exists.")
            return

        window_bed = self.window_bed

        region_to_data = self._get_region_data(dataset_path, chrom)
        region_bed = (
            understand_regions(list(region_to_data.keys()))
            .sort_values("Start")
            .reset_index(drop=True)
        )

        window_data_col = []
        chrom_window_bed = window_bed[window_bed["Chromosome"] == chrom]
        for _, (chrom, start, end) in tqdm(
            chrom_window_bed.iterrows(), total=chrom_window_bed.shape[0]
        ):
            region = f"{chrom}:{start}-{end}"
            length = end - start
            if length != self.merge_window_size:
                continue
            use_regions = region_bed[
                (region_bed["Start"] < end) & (region_bed["End"] > start)
            ]

            # concatenate small meta regions (e.g., 100K) into a window data (e.g., 600K)
            # because the meta regions are overlapped, we need to trim the overlapped regions
            cur_start = start
            window_data = []
            for _, (_, rstart, rend, rname) in use_regions.iterrows():
                csc_data = region_to_data[rname]
                left_trim = cur_start - rstart
                right_trim = max(0, rend - end)
                if right_trim == 0:
                    csc_data = csc_data[:, left_trim:]
                    cur_start = rend
                else:
                    csc_data = csc_data[:, left_trim:-right_trim]
                    cur_start = rend - right_trim
                window_data.append(csc_data)
            window_data = hstack(window_data).tocsr()
            if window_data.shape[1] != length:
                print(f"Warning: {region} has shape {window_data.shape[1]} != {length}")
                continue

            # merge bp to bins at given resolution
            if self.merge_resolution > 1:
                window_data = _merge_bp_to_bins(window_data, self.merge_resolution)
            window_data = window_data.astype("float32").tocsr()
            data_dict = csr_matrix_to_compressed_bytes_dict(
                prefix=self.prefix, matrix=window_data
            )
            data_dict["region"] = region
            window_data_col.append(data_dict)

        # dump to parquet
        ds = ray.data.from_items(window_data_col)
        ds.write_parquet(f"{output_dir}/{chrom}", num_rows_per_file=num_rows_per_file)

        success_flag.touch()
        return

    def merge(self, dataset_path, output_dir, num_rows_per_file=20):
        """Merge all chromosomes"""
        chroms = [p.name for p in pathlib.Path(dataset_path).glob("*") if p.is_dir()]
        for chrom in chroms:
            print("Merging", chrom)
            self.merge_chrom(dataset_path, chrom, output_dir, num_rows_per_file)

        # config
        config = joblib.load(f"{dataset_path}/config.joblib")
        config["window_size"] = self.merge_window_size
        config["step_size"] = self.merge_step_size
        config["num_rows_per_file"] = num_rows_per_file
        joblib.dump(config, f"{output_dir}/config.joblib")

        # row names
        _row_names = joblib.load(f"{dataset_path}/row_names.joblib")
        row_names = {
            self.prefix: _row_names[self.prefix],
        }
        joblib.dump(row_names, f"{output_dir}/row_names.joblib")
