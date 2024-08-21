import pathlib
from collections import defaultdict
from typing import Union

import joblib
import numpy as np
import pandas as pd
import ray
from einops import repeat

from bolero import Genome
from bolero.tl.generic.dataset import GenericDataset


class ConvertCategories:
    def __init__(self, categories):
        self.categories = categories

    def __call__(self, data_dict):
        """Convert int categories to strings."""
        row_data = {}
        for key, value in data_dict.items():
            try:
                row_data[key] = self.categories[key][value]
            except KeyError:
                row_data[key] = value
        return row_data


class CountPreprocess:
    def __init__(
        self, total_count=1e5, log1p=True, dtype="float32", data_key="gene_exp"
    ):
        self.total_count = total_count
        self.log1p = log1p
        self.data_key = data_key
        self.dtype = dtype

    def __call__(self, data_dict):
        """Normalize and log1p transform the gene expression data."""
        gene_data = data_dict.pop(self.data_key)

        gene_total_count = gene_data.sum(axis=-1)
        gene_data = gene_data * self.total_count / gene_total_count[:, None]

        if self.log1p:
            gene_data = np.log1p(gene_data)

        data_dict[self.data_key] = gene_data.astype(self.dtype)
        return data_dict


class SelectGenes:
    def __init__(self, gene_bool_sel, data_key):
        self.gene_bool_sel = gene_bool_sel
        self.data_key = data_key

    def __call__(self, data_dict):
        """Select genes."""
        data_dict[self.data_key] = data_dict[self.data_key][
            :, self.gene_bool_sel
        ].copy()
        return data_dict


class FilterByObs:
    def __init__(self, filter_by_obs_idx: dict[set[int]]):
        self.filter_dict = {k: list(set(v)) for k, v in filter_by_obs_idx.items()}

    def __call__(self, data_dict):
        """Filter rows by obs metadata categories."""
        row_sel = []
        for col, allow_idx in self.filter_dict.items():
            try:
                bool_sel = np.isin(data_dict[col], allow_idx)
                row_sel.append(bool_sel)
            except KeyError:
                continue
        row_sel = np.all(row_sel, axis=0)
        sel_data = {k: v[row_sel] for k, v in data_dict.items()}
        return sel_data


class RayGeneDataset(GenericDataset):
    default_config = {
        "dataset_path": "REQUIRED",
        "shuffle_files": True,
        "read_parquet_kwargs": None,
    }

    def __init__(
        self,
        dataset_path,
        genome=None,
        shuffle_files=True,
        read_parquet_kwargs=None,
    ):
        self.data_key = "gene_exp"

        self.dataset_path = dataset_path

        if not shuffle_files:
            print("File shuffle is disabled!!!")

        _kwargs = {
            "shuffle": "files" if shuffle_files else None,
            "file_extensions": ["parquet"],
        }
        if read_parquet_kwargs is not None:
            _kwargs.update(read_parquet_kwargs)
        self.read_parquet_kwargs = _kwargs

        # get categories info
        self._idx_to_category_map = None
        # gene info
        self.gene_list = joblib.load(f"{dataset_path}/gene_list.joblib")
        self.gene_order = pd.Index(self.gene_list, name="gene")
        self.gene_to_idx = pd.Series(
            {gene: idx for idx, gene in enumerate(self.gene_order)}
        )
        # get metadata
        self.config = joblib.load(f"{dataset_path}/config.joblib")

        if genome is None:
            if isinstance(genome, str):
                self.genome = Genome(genome)
        else:
            self.genome = Genome(self.config["genome"])

    @property
    def idx_to_category_map(self):
        """Get the index to category map."""
        if self._idx_to_category_map is None:
            self._idx_to_category_map = joblib.load(
                f"{self.dataset_path}/categories.joblib"
            )
        return self._idx_to_category_map

    @property
    def category_keys(self):
        """Get the category keys."""
        # remove "obs:" prefix
        return [k[4:] for k in self.idx_to_category_map.keys()]

    def get_categories(self, key) -> pd.Series:
        """Get the categories for an obs key."""
        if not key.startswith("obs:"):
            key = f"obs:{key}"
            return pd.Series(self.idx_to_category_map[key])

    def _read_parquet(self, folds, _conc=4):
        if folds is None:
            _path = self.dataset_path
        else:
            if isinstance(folds, int):
                folds = [folds]
            _path = [f"{self.dataset_path}/fold{fold}/" for fold in folds]

        _kwargs = self.read_parquet_kwargs.copy()
        _kwargs["concurrency"] = _conc
        dataset = ray.data.read_parquet(_path, **_kwargs)
        return dataset

    def _sel_genes(self, dataset, cur_gene_order, sel_genes, _bs, _conc):
        gene_bool_sel = cur_gene_order.isin(sel_genes)
        print(f"Selecting {gene_bool_sel.sum()}/{len(cur_gene_order)} genes")
        dataset = dataset.map_batches(
            SelectGenes,
            fn_constructor_kwargs={
                "gene_bool_sel": gene_bool_sel,
                "data_key": self.data_key,
            },
            batch_size=_bs,
            concurrency=_conc,
        )
        new_gene_order = cur_gene_order[gene_bool_sel].copy()
        return dataset, new_gene_order

    def _count_preprocess(self, dataset, total_count, log1p, _bs, _conc):
        dataset = dataset.map_batches(
            CountPreprocess,
            fn_constructor_kwargs={
                "total_count": total_count,
                "log1p": log1p,
                "dtype": "float32",
            },
            batch_size=_bs,
            concurrency=_conc,
        )
        return dataset

    def _add_gene_idx_to_batch(self, dataset, cur_gene_order, bs=1024):
        # add gene index to dataset
        # gene index is the col index of the original gene order
        # gene order is current gene ids
        cur_gene_idx = cur_gene_order.map(self.gene_to_idx).values.astype("int32")
        cur_gene_idx = repeat(cur_gene_idx, "gene -> batch gene", batch=bs)

        def _add_gene_idx(data_dict):
            data_dict["gene_idx"] = cur_gene_idx.copy()
            return data_dict

        dataset = dataset.map_batches(
            _add_gene_idx,
            batch_size=bs,
            concurrency=max(self.concurency // 4, 1),
        )
        return dataset

    def _dump_obs(self, dataset):
        def _dump(data_dict):
            keys = data_dict.keys()
            for k in keys:
                if k.startswith("obs:"):
                    data_dict.pop(k)
            return data_dict

        dataset = dataset.map_batches(
            _dump,
            batch_size=1024,
            concurrency=max(self.concurency // 4, 1),
        )
        return dataset

    def _filter_by_obs(
        self, dataset, filter_by_obs: dict[set[str] | str], _bs=1024, _conc=(1, 4)
    ):
        # prepare filter dict, input is category names, convert to category index
        # e.g., filter_by_obs = {"MajorRegion": "HPF"}
        # e.g., filter_by_obs_idx = {'obs:MajorRegion': {4}}
        filter_by_obs_idx = {}
        for key, values in filter_by_obs.items():
            if isinstance(values, str):
                values = {values}
            else:
                values = set(values)
            idx_to_cat = self.idx_to_category_map[f"obs:{key}"]
            cat_to_idx = {v: k for k, v in idx_to_cat.items() if v in values}
            filter_by_obs_idx[f"obs:{key}"] = set(cat_to_idx.values())

        dataset = dataset.map_batches(
            FilterByObs,
            fn_constructor_kwargs={"filter_by_obs_idx": filter_by_obs_idx},
            batch_size=_bs,
            concurrency=_conc,
        )
        return dataset

    def pre_estimate_genes(self, qc_genes, sel_genes):
        """Pre-estimate gene order to help setup model."""
        cur_gene_index = self.gene_order.copy()

        if qc_genes is not None:
            if isinstance(qc_genes, (str, pathlib.Path)):
                qc_genes = pd.read_csv(qc_genes, header=None, index_col=0).index
            qc_genes = pd.Index(qc_genes)
            cur_gene_index = cur_gene_index[cur_gene_index.isin(qc_genes)].copy()

        if sel_genes is not None:
            if isinstance(sel_genes, (str, pathlib.Path)):
                sel_genes = pd.read_csv(sel_genes, header=None, index_col=0).index
            sel_genes = pd.Index(sel_genes)
            cur_gene_index = cur_gene_index[cur_gene_index.isin(sel_genes)].copy()

        return cur_gene_index

    def _get_processed_dataset(
        self,
        folds: Union[int, list[int]],
        dataset=None,
        cur_gene_order=None,
        normalize=True,
        convert_categories=False,
        cell_target_count=1e5,
        log1p=True,
        max_step_concurrency=4,
        qc_genes=None,
        sel_genes=None,
        filter_by_obs=None,
        _oprator_batch_size=1024,
    ):
        """
        Get a processed dataset.

        Parameters
        ----------
        folds : Union[int, list[int]]
            Folds to read.
        dataset : ray.data.dataset.Dataset, optional
            An pre-opened dataset, by default None.
        cur_gene_order : pd.Index, optional
            Current gene order for pre-opened dataset, by default None.
        convert_categories : bool, optional
            Convert int categories to strings, by default False.
        normalize : bool, optional
            Whether to normalize the cell count, by default True.
        cell_target_count : float, optional
            Target count for normalization, by default 1e5.
        log1p : bool, optional
            Log1p transform, by default True.
        max_step_concurrency : int, optional
            Maximum concurrency for each step, by default 4.
        qc_genes : pd.Index, optional
            Genes passed basic QC, this step will be done before normalization, by default None.
        sel_genes : pd.Index, optional
            Genes to select, this step will be done after normalization, by default None.

        Returns
        -------
        ray.data.dataset.Dataset, pd.Index
            Processed dataset and the gene order.
        """
        if dataset is None:
            dataset = self._read_parquet(folds, _conc=max_step_concurrency)
            cur_gene_order = self.gene_order.copy()
        else:
            assert (
                cur_gene_order is not None
            ), "cur_gene_order must be provided if dataset is provided."

        if filter_by_obs is not None:
            dataset = self._filter_by_obs(
                dataset,
                filter_by_obs,
                _bs=_oprator_batch_size,
                _conc=(1, max_step_concurrency),
            )

        if convert_categories:
            dataset = dataset.map(
                ConvertCategories,
                fn_constructor_kwargs={"categories": self.idx_to_category_map},
                concurrency=(1, max_step_concurrency),
            )

        if qc_genes is not None:
            dataset, cur_gene_order = self._sel_genes(
                dataset=dataset,
                sel_genes=qc_genes,
                cur_gene_order=cur_gene_order,
                _bs=_oprator_batch_size,
                _conc=(1, max_step_concurrency),
            )

        if normalize:
            dataset = self._count_preprocess(
                dataset,
                cell_target_count,
                log1p,
                _bs=_oprator_batch_size,
                _conc=(1, max_step_concurrency),
            )

        if sel_genes is not None:
            dataset, cur_gene_order = self._sel_genes(
                dataset=dataset,
                sel_genes=sel_genes,
                cur_gene_order=cur_gene_order,
                _bs=_oprator_batch_size,
                _conc=(1, max_step_concurrency),
            )
        return dataset, cur_gene_order

    def get_sample_adata(
        self,
        n_cells,
        folds,
        qc_genes=None,
        sel_genes=None,
        filter_by_obs=None,
        sparse=True,
        normalize=True,
        cell_target_count=1e5,
        log1p=True,
        local_shuffle_buffer_size=10000,
        concurrency=4,
    ):
        """
        Get a sample of cells as an AnnData object.
        """
        import anndata
        from scipy.sparse import csr_matrix, vstack

        data_key = self.config["data_key"]

        dataset, cur_gene_order = self._get_processed_dataset(
            folds=folds,
            dataset=None,
            cur_gene_order=None,
            normalize=normalize,
            convert_categories=False,
            cell_target_count=cell_target_count,
            log1p=log1p,
            max_step_concurrency=concurrency,
            qc_genes=qc_genes,
            sel_genes=sel_genes,
            filter_by_obs=filter_by_obs,
        )
        print(f"{cur_gene_order.size} genes are selected.")

        obs_col = defaultdict(list)
        data_col = []
        loader = dataset.limit(n_cells).iter_batches(
            batch_size=1024, local_shuffle_buffer_size=local_shuffle_buffer_size
        )
        for batch in loader:
            _exp_data = batch[data_key]
            if sparse:
                data_col.append(csr_matrix(_exp_data))
            else:
                data_col.append(_exp_data)

            for key, value in batch.items():
                if key.startswith("obs:"):
                    obs_col[key[4:]].extend(list(value))
        del loader
        del dataset

        if len(data_col) == 0:
            raise ValueError(
                "No cells returned by the data loader, "
                "please check filter and other parameters."
            )

        obs = pd.DataFrame(obs_col)
        obs.index = obs.index.astype("str")
        var = pd.DataFrame(index=cur_gene_order)
        adata = anndata.AnnData(
            X=vstack(data_col) if sparse else np.vstack(data_col),
            obs=obs,
            var=var,
        )

        # map categories and cell ids
        idx_to_category_map = self.idx_to_category_map
        cell_int = adata.obs.pop("cell")
        cell_index = pd.Index(cell_int.map(idx_to_category_map["obs:cell"]))
        adata.obs_names = cell_index
        for col in adata.obs.columns:
            key = f"obs:{col}"
            if key in idx_to_category_map:
                adata.obs[col] = (
                    adata.obs[col].map(idx_to_category_map[key]).astype("category")
                )

        if adata.shape[0] < n_cells:
            print(
                f"Warning: Only {adata.shape[0]} cells are returned, "
                f"instead of the requested {n_cells}."
            )
        return adata

    def estimate_highly_variable_genes(
        self,
        folds: Union[int, list[int]],
        use_cells=100000,
        min_genes=100,
        min_cell_ratio=0.0005,
        n_top_hvg=None,
        filter_by_obs=None,
        subset=False,
        return_adata=False,
        concurrency=4,
    ):
        """
        Estimate Highly Variable Genes by sampling a subset of cells
        and filtering genes with basic scanpy preprocessing.

        Parameters
        ----------
        folds : Union[int, list[int]]
            Folds to sample cells from.
        use_cells : int, optional
            Number of cells to sample, by default 100000.
        min_genes : int, optional
            Minimum number of genes to keep a cell, by default 100.
        min_cell_ratio : float, optional
            Minimum ratio of cells a gene must be expressed in to keep it, by default 0.0005.
        n_top_hvg : int, optional
            Number of highly variable genes to keep, by default None.
        filter_by_obs : dict[set[str] | str], optional
            Filter cells by obs metadata categories, by default None.
            Example: {"MajorRegion": "HPF"} or {"MajorRegion": {"HPF", "TH"}}
        subset : bool, optional
            Whether to subset the dataset with HVG, by default False.
        return_adata : bool, optional
            Whether to return the AnnData object, by default False.
        concurrency : int, optional
            Maximum concurrency for each step, by default 4.

        Returns
        -------
        pd.Index
            Genes that pass the filtering.
        """
        import scanpy as sc

        print(f"Sampling {use_cells} cells to generate an AnnData object.")
        # raw adata without and preprocessing and gene selection
        adata = self.get_sample_adata(
            use_cells,
            folds=folds,
            filter_by_obs=filter_by_obs,
            concurrency=concurrency,
            normalize=False,
            qc_genes=None,
            sel_genes=None,
        )

        print("Filtering genes with basic scanpy preprocessing.")
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=int(min_cell_ratio * use_cells))
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_hvg, subset=subset)

        _hvgs = adata.var_names[adata.var["highly_variable"]]
        remain_genes = self.gene_order[self.gene_order.isin(_hvgs)].copy()
        print(f"Remaining {len(remain_genes)} genes.")

        if return_adata:
            return remain_genes, adata
        else:
            return remain_genes, None

    def get_dataloader(
        self, work_ds, data_iter_kwargs, n_batches, batch_size=256, as_torch=True
    ):
        """Get a dataloader."""
        print(f"Get dataloader with {self.dataset_mode} mode")
        if n_batches is not None:
            n_rows = (n_batches + 1) * batch_size
            work_ds = work_ds.limit(n_rows)

        _kwargs = {
            "batch_size": batch_size,
            "prefetch_batches": 3,
            "drop_last": True,  # helps to avoid the last batch with less than batch_size
            "local_shuffle_buffer_size": (
                10000 if self.dataset_mode == "train" else None
            ),
        }
        _kwargs.update(data_iter_kwargs)
        print("Data loader kwargs", _kwargs)

        if as_torch:
            loader = work_ds.iter_torch_batches(**_kwargs)
        else:
            loader = work_ds.iter_batches(**_kwargs)

        return loader
