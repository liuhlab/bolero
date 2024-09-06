import pathlib
import shutil
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import ray
from scipy.sparse import csr_matrix, vstack

from bolero.pp.genome_chunk_dataset import (
    array_to_compressed_bytes_dict,
    csr_matrix_to_compressed_bytes_dict,
)
from bolero.tl.dataset.ray_dataset import RayGenomeChunkDataset


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
