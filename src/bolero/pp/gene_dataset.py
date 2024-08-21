import pathlib
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import ray
import xarray as xr


@ray.remote(resources={"bolero_100": 10})
def _save_ds_chunk(
    cell_slice,
    data_array,
    cell_annot,
    cell_folds,
    output_dir,
    chunk_id,
    data_key="gene_exp",
    num_rows_per_file=2500,
):
    _chunk_data = data_array.isel(cell=cell_slice).values
    _chunk_annot = cell_annot.iloc[cell_slice]
    _chunk_folds = cell_folds.iloc[cell_slice]

    row_col = defaultdict(list)
    for row, row_gene_data in enumerate(_chunk_data):
        row_data = {f"obs:{k}": v for k, v in _chunk_annot.iloc[row].items()}
        row_data[data_key] = row_gene_data
        fold = _chunk_folds.values[row]
        row_col[fold].append(row_data)

    for fold, rows in row_col.items():
        this_output_dir = pathlib.Path(f"{output_dir}/fold{fold}/chunk{chunk_id}")

        dataset = ray.data.from_items(rows)
        dataset.write_parquet(
            this_output_dir,
            num_rows_per_file=num_rows_per_file,
            concurrency=1,
        )
    return


class GeneDatasetGenerator:
    def __init__(
        self,
        gene_data_path,
        cell_annot_path,
        genome,
        gtf_version,
        da_name="gene_da",
        gene_dim="gene",
        cell_dim="cell",
        data_key="gene_exp",
        use_annot_cols=None,
    ):
        self.data_key = data_key
        self.genome = genome
        self.gtf_version = gtf_version

        self.gene_dim = gene_dim
        self.cell_dim = cell_dim
        ds = xr.open_zarr(gene_data_path)

        cell_annot = pd.read_feather(cell_annot_path)
        cell_annot.set_index(cell_annot.columns[0], inplace=True)
        if use_annot_cols is not None:
            cell_annot = cell_annot[use_annot_cols].copy()

        self.ds, self.cell_annot = self._sync_ds_and_annot(ds, cell_annot)
        self.da = self.ds[da_name]

        self.total_cells = self.cell_annot.shape[0]
        self.total_genes = self.da.get_index(gene_dim).size
        print(f"Total cells: {self.total_cells}")

    def _sync_ds_and_annot(self, ds, cell_annot):
        all_cells = ds.get_index(self.cell_dim)

        use_cells = all_cells.isin(cell_annot.index)
        ds = ds.sel(cell=use_cells)
        cell_annot = cell_annot.reindex(all_cells[use_cells])

        assert (ds.get_index(self.cell_dim) == cell_annot.index).all()
        return ds, cell_annot

    def _annot_to_numbers(self):
        cell_annot = self.cell_annot

        indices_to_category_map = {}

        # turn cell annot into int index
        for col, dtype in cell_annot.dtypes.items():
            if not pd.api.types.is_numeric_dtype(dtype):
                _data = cell_annot[col].astype("category")
                idx_to_cat = dict(enumerate(_data.cat.categories))
                indices_to_category_map[f"obs:{col}"] = idx_to_cat
                cell_annot[col] = _data.cat.codes

        # also turn cell id into int index
        idx_to_cat = dict(enumerate(cell_annot.index))
        indices_to_category_map["obs:cell"] = idx_to_cat
        cell_annot.reset_index(drop=True, inplace=True)
        cell_annot.index.name = "cell"
        cell_annot.reset_index(inplace=True)

        self.cell_annot = cell_annot
        return indices_to_category_map

    def _dump_config(self, output_dir, folds, chunks, num_rows_per_file):
        meta_data = {
            "num_cells": self.total_cells,
            "num_genes": self.total_genes,
            "data_key": self.data_key,
            "metadata_cols": list(self.cell_annot.columns),
            "n_folds": folds,
            "n_chunks": chunks,
            "num_rows_per_file": num_rows_per_file,
            "genome": self.genome,
            "gtf_version": self.gtf_version,
        }
        joblib.dump(meta_data, f"{output_dir}/config.joblib")

    def generate(self, output_dir, folds=10, chunk_size=100000, num_rows_per_file=2500):
        """Generate the dataset in parquet format"""
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

        # turn cell
        indices_to_category_map = self._annot_to_numbers()
        joblib.dump(indices_to_category_map, f"{output_dir}/categories.joblib")

        # save gene order
        gene_list = self.ds.get_index(self.gene_dim).tolist()
        joblib.dump(gene_list, f"{output_dir}/gene_list.joblib")

        cell_folds = pd.Series(np.random.randint(0, folds, self.total_cells))

        da_ref = ray.put(self.da)
        cell_annot_ref = ray.put(self.cell_annot)
        cell_folds_ref = ray.put(cell_folds)

        chunk_starts = list(range(0, self.total_cells, chunk_size))

        tasks = []
        for chunk_idx, chunk_start in enumerate(chunk_starts):
            _cell_slice = slice(
                chunk_start, min(chunk_start + chunk_size, self.total_cells)
            )
            task = _save_ds_chunk.remote(
                cell_slice=_cell_slice,
                data_array=da_ref,
                cell_annot=cell_annot_ref,
                cell_folds=cell_folds_ref,
                output_dir=output_dir,
                chunk_id=chunk_idx,
                data_key=self.data_key,
                num_rows_per_file=num_rows_per_file,
            )
            tasks.append(task)

        _ = ray.get(tasks)

        self._dump_config(output_dir, folds, len(chunk_starts), num_rows_per_file)
        return
