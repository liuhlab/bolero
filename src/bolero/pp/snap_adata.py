import pathlib
import tempfile
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import ray
import snapatac2 as snap
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack

from bolero import Genome


@ray.remote
def _dump_grouped_csr(
    sample, adata_path, key, pseudobulk_order, groupping, output_path
):
    adata = snap.read(adata_path, backed="r")
    adata_bcs = pd.Index(adata.obs_names)

    cell_row_to_bulk_row = np.zeros(adata_bcs.size, dtype="int32") - 1
    for bulk_row_id, group in enumerate(pseudobulk_order):
        # we intersect adata obs_names with each group's bcs,
        # this means cells are hard assigned to groups, one cell only occurs in one group
        bool_row_sel = adata_bcs.isin(groupping.get((sample, group), []))
        if bool_row_sel.sum() == 0:
            continue
        cell_row_to_bulk_row[bool_row_sel] = bulk_row_id

    data = adata.obsm[key].tocsr()
    group_mat = {}
    for bulk_row in range(len(pseudobulk_order)):
        cell_rows = np.where(cell_row_to_bulk_row == bulk_row)[0]
        if cell_rows.size > 0:
            group_mat[bulk_row] = data[cell_rows]

    joblib.dump(group_mat, output_path)
    return


@ray.remote
def _csr_row_sum(csr_list, save_path):
    data = csr_matrix(vstack(csr_list).sum(axis=0), dtype="uint32")
    save_npz(save_path, data, compressed=True)


class AdataPseudobulkMerger:
    def __init__(
        self,
        pseudobulk_meta,
        adata_path_dict,
        output_dir,
        sparse_format="csc",
    ):
        """
        Merge pseudobulk data from multiple samples/adata_files and groups into a single directory.

        It took ~30min to merge 173k cell into 1.3k unbalanced pseudobulks, using 60 cpus, memory max < 30GB.
        The last csc step is slow and unparalleled, can parallel if memory is enough.

        Parameters
        ----------
        pseudobulk_meta : str
            Path to the pseudobulk metadata file (feather format).
            It should have cell_id as index, and three columns:
            - sample (match with adata_path_dict)
            - groups (grouping for pseudobulk)
            - bc (barcode for each cell, match with each snap adata file)
        adata_path_dict : dict
            Dictionary mapping sample names to their corresponding snap adata file paths.
        output_dir : str
            Directory to save the merged pseudobulk data. Final file will be:
        sparse_format : str
            The sparse format to save the merged data. Options are 'csr' or 'csc'.

        """
        self.output_dir = pathlib.Path(output_dir)

        # Prepare merge info
        if not isinstance(pseudobulk_meta, pd.DataFrame):
            pseudobulk_meta = pd.read_feather(pseudobulk_meta)
        self.pseudobulk_meta = pseudobulk_meta
        self.pseudobulk_meta.columns = ["sample", "groups", "bc"]

        n_sample = self.pseudobulk_meta["sample"].nunique()
        n_groups = self.pseudobulk_meta["groups"].nunique()
        n_cells = self.pseudobulk_meta.shape[0]
        print(
            f"pseudobulk meta table: {n_sample} samples, {n_groups} groups, {n_cells} cells"
        )
        self.pseudobulk_order = sorted(self.pseudobulk_meta["groups"].unique())

        self.groupping = {
            (sample, group): df["bc"].tolist()
            for (sample, group), df in self.pseudobulk_meta.groupby(
                ["sample", "groups"]
            )
        }
        self.adata_path_dict = adata_path_dict
        samples_in_table = self.pseudobulk_meta["sample"].unique()
        assert all(
            k in samples_in_table for k in adata_path_dict.keys()
        ), "not all samples in pseudobulk meta table"

        adata = snap.read(list(adata_path_dict.values())[0], backed="r")
        self.chrom_keys = [k for k in adata.obsm.keys() if k.startswith("insertion_")]
        adata.close()

        self.sparse_format = sparse_format
        assert self.sparse_format in [
            "csr",
            "csc",
        ], f"unsupported sparse format {sparse_format}"
        return

    def _save_metadata(self):
        joblib.dump(
            self.pseudobulk_order, self.output_dir / "pseudobulk_order.list.joblib"
        )
        joblib.dump(self.chrom_keys, self.output_dir / "chrom_keys.list.joblib")
        self.pseudobulk_meta.to_feather(self.output_dir / "pseudobulk_metadata.feather")

    def _extract_by_group(self, temp_dir, chrom_key):
        # select rows for each sample and each group,
        # save into group:csr_mat dict for each sample.
        fs = []
        for sample, adata_path in self.adata_path_dict.items():
            output_path = temp_dir / f"{sample}_{chrom_key}"
            f = _dump_grouped_csr.remote(
                sample=sample,
                adata_path=adata_path,
                key=chrom_key,
                pseudobulk_order=self.pseudobulk_order,
                groupping=self.groupping,
                output_path=output_path,
            )
            fs.append(f)
        _ = ray.get(fs)
        return

    def _merge_single_chrom(self, temp_dir, chrom_key):
        # merge samples for each chrom and each group
        print(f"dump {chrom_key} groups merge")
        to_del = []
        chrom_col = defaultdict(list)
        for sample in self.adata_path_dict.keys():
            output_path = temp_dir / f"{sample}_{chrom_key}"
            to_del.append(output_path)
            sample_data = joblib.load(output_path)
            for k, v in sample_data.items():
                chrom_col[k].append(v)
        chrom_size = next(iter(chrom_col.values()))[0].shape[1]

        fs = []
        for k, v in chrom_col.items():
            output_path = temp_dir / f"group_{k}_{chrom_key}"
            f = _csr_row_sum.remote(v, output_path)
            fs.append(f)
        _ = ray.get(fs)

        print("merging done, now loading group data")
        group_datas = []
        for gid in range(len(self.pseudobulk_order)):
            group_path = temp_dir / f"group_{gid}_{chrom_key}.npz"
            to_del.append(group_path)
            try:
                group_datas.append(load_npz(group_path).tocsr())
            except FileNotFoundError:
                group_datas.append(csr_matrix((1, chrom_size), dtype="uint32"))
        group_datas = vstack(group_datas)

        group_datas.data = np.clip(group_datas.data, 0, np.iinfo("uint16").max)
        group_datas = group_datas.astype("uint16")

        if self.sparse_format == "csc":
            group_datas = group_datas.tocsc()

        temp_path = self.output_dir / f"{chrom_key}.{self.sparse_format}.temp.joblib"
        final_path = self.output_dir / f"{chrom_key}.{self.sparse_format}.joblib"
        joblib.dump(group_datas, self.output_dir / temp_path)
        final_path.rename(temp_path)

        for path in to_del:
            if pathlib.Path(path).exists():
                pathlib.Path(path).unlink()
        return

    def merge(self):
        """Merge all chroms for all samples and groups."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)

            for chrom_key in self.chrom_keys:
                chrom_path = (
                    self.output_dir / f"{chrom_key}.{self.sparse_format}.joblib"
                )
                if chrom_path.exists():
                    print(f"{chrom_key} already exists, skip")
                    continue

                self._extract_by_group(temp_dir, chrom_key)
                self._merge_single_chrom(temp_dir, chrom_key)
        return


class PseudobulkAdata:
    def __init__(self, pseudobulk_dir, genome, catch_chroms=False):
        """
        Load the pseudobulk data from the specified directory.

        Parameters
        ----------
        pseudobulk_dir : str
            Directory containing the merged pseudobulk data files.

        """
        self.pseudobulk_dir = pathlib.Path(pseudobulk_dir)
        self.pseudobulk_order = joblib.load(
            self.pseudobulk_dir / "pseudobulk_order.list.joblib"
        )
        self.pseudobulk_meta = pd.read_feather(
            self.pseudobulk_dir / "pseudobulk_metadata.feather"
        )
        self.chrom_keys = joblib.load(self.pseudobulk_dir / "chrom_keys.list.joblib")
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome
        self.catch_chroms = catch_chroms
        self.chrom_cache = {}

    def get_chrom(self, chrom_key, mat_type="csc"):
        """Get the pseudobulk whole chromosome sparse matrix."""
        cache_key = (chrom_key, mat_type)
        if cache_key in self.chrom_cache:
            return self.chrom_cache[cache_key]

        assert chrom_key in self.chrom_keys, f"chrom_key {chrom_key} not found"
        assert mat_type in ["csr", "csc"], f"mat_type {mat_type} not supported"

        data = joblib.load(self.pseudobulk_dir / f"{chrom_key}.{mat_type}.joblib")
        if self.catch_chroms:
            self.chrom_cache[cache_key] = data
        return data

    def clean_cache(self):
        """Clear the cached chromosome data."""
        self.chrom_cache.clear()
