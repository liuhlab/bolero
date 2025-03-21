import pathlib
from collections import defaultdict
from shutil import rmtree

import joblib
import numpy as np
import pandas as pd
import ray
import snapatac2 as snap
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack
from tqdm import tqdm

from bolero import Genome


@ray.remote
def _dump_grouped_csr(
    sample, adata_path, key, pseudobulk_order, groupping, output_path
):
    if pathlib.Path(output_path).exists():
        return

    adata = snap.read(adata_path, backed="r")
    adata_bcs = pd.Index(adata.obs_names)

    cell_row_to_bulk_row = np.zeros(adata_bcs.size, dtype="int32") - 1
    for row_id, group in enumerate(pseudobulk_order):
        bool_row_sel = adata_bcs.isin(groupping.get((sample, group), []))
        if bool_row_sel.sum() == 0:
            continue
        cell_row_to_bulk_row[bool_row_sel] = row_id

    data = adata.obsm[key].tocsr()
    group_mat = {}
    for bulk_row in range(len(pseudobulk_order)):
        cell_rows = np.where(cell_row_to_bulk_row == bulk_row)[0]
        if cell_rows.size > 0:
            group_mat[bulk_row] = data[cell_rows]

    joblib.dump(group_mat, output_path)
    return


@ray.remote
def _merge_csr(csr_list, save_path):
    data = csr_matrix(vstack(csr_list).sum(axis=0), dtype="uint32")
    save_npz(save_path, data, compressed=True)


class AdataPseudobulkMerger:
    def __init__(
        self,
        pseudobulk_meta_path,
        adata_path_dict,
        output_dir="pseudobulk/",
    ):
        """
        Merge pseudobulk data from multiple samples/adata_files and groups into a single directory.

        It took ~30min to merge 173k cell into 1.3k unbalanced pseudobulks, using 60 cpus, memory max < 30GB.
        The last csc step is slow and unparalleled, can parallel if memory is enough.

        Parameters
        ----------
        pseudobulk_meta_path : str
            Path to the pseudobulk metadata file (feather format).
            It should have cell_id as index, and three columns:
            - sample (match with adata_path_dict)
            - groups (grouping for pseudobulk)
            - bc (barcode for each cell, match with each adata file)
        adata_path_dict : dict
            Dictionary mapping sample names to their corresponding adata file paths.
        output_dir : str
            Directory to save the merged pseudobulk data. Final file will be:
            - csr matrix for each chrom_key (e.g. "insertion_1.csr.joblib")
            - csc matrix for each chrom_key (if save_csc=True)

        """
        self.output_dir = pathlib.Path(output_dir)
        self.temp_dir = self.output_dir / "temp/"

        # Prepare merge info
        self.pseudobulk_meta = pd.read_feather(pseudobulk_meta_path)
        self.pseudobulk_meta.columns = ["sample", "groups", "bc"]
        print("Getting pseudobulk meta table")
        print(self.pseudobulk_meta.shape)
        print(self.pseudobulk_meta.head())
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

    def _save_metadata(self):
        joblib.dump(
            self.pseudobulk_order, self.output_dir / "pseudobulk_order.list.joblib"
        )
        joblib.dump(self.chrom_keys, self.output_dir / "chrom_keys.list.joblib")
        self.pseudobulk_meta.to_feather(self.output_dir / "pseudobulk_metadata.feather")

    def _extract_by_group(self):
        # select rows for each sample and each group,
        # save into group:csr_mat dict for each sample.
        fs = []
        for sample, adata_path in self.adata_path_dict.items():
            for chrom_key in self.chrom_keys:
                output_path = self.temp_dir / f"{sample}_{chrom_key}"
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

    def _merge_single_chrom(self, chrom_key):
        # merge samples for each chrom and each group

        print(f"dump {chrom_key} groups merge")
        to_del = []
        chrom_col = defaultdict(list)
        for sample in self.adata_path_dict.keys():
            output_path = self.temp_dir / f"{sample}_{chrom_key}"
            to_del.append(output_path)
            sample_data = joblib.load(output_path)
            for k, v in sample_data.items():
                chrom_col[k].append(v)
        chrom_size = next(iter(chrom_col.values()))[0].shape[1]

        fs = []
        for k, v in chrom_col.items():
            output_path = self.temp_dir / f"group_{k}_{chrom_key}"
            f = _merge_csr.remote(v, output_path)
            fs.append(f)
        _ = ray.get(fs)

        print("merging done, now loading group data")
        group_datas = []
        for gid, _ in tqdm(
            enumerate(self.pseudobulk_order),
            total=len(self.pseudobulk_order),
        ):
            group_path = self.temp_dir / f"group_{gid}_{chrom_key}.npz"
            to_del.append(group_path)
            try:
                group_datas.append(load_npz(group_path).tocsr())
            except FileNotFoundError:
                group_datas.append(csr_matrix((1, chrom_size), dtype="uint32"))
        group_datas = vstack(group_datas)

        group_datas.data = np.clip(group_datas.data, 0, np.iinfo("uint16").max)
        group_datas = group_datas.astype("uint16")
        joblib.dump(group_datas, self.output_dir / f"{chrom_key}.csr.joblib")

        for path in to_del:
            if pathlib.Path(path).exists():
                pathlib.Path(path).unlink()
        return

    def _to_csc(self):
        # save a copy of CSC matrix
        for chrom_key in self.chrom_keys:
            group_datas = joblib.load(self.output_dir / f"{chrom_key}.csr.joblib")
            group_datas = group_datas.tocsc()
            joblib.dump(group_datas, self.output_dir / f"{chrom_key}.csc.joblib")

    def merge(self, save_csc=True):
        """Merge all chroms for all samples and groups."""
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        self._save_metadata()

        for chrom_key in self.chrom_keys:
            self._merge_single_chrom(chrom_key)

        if save_csc:
            self._to_csc()

        if self.temp_dir.exists():
            rmtree(str(self.temp_dir))


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
