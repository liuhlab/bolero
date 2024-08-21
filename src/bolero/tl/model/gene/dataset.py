import hashlib
import pathlib

import pandas as pd

from bolero.tl.dataset.ray_gene_dataset import RayGeneDataset

from .scaler import GeneDataScaler, IdentityScaler


class ScaleGene:
    def __init__(self, scaler, data_keys):
        self.scaler = scaler
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        self.data_keys = data_keys

    def __call__(self, data):
        """Scale the gene data."""
        for key in self.data_keys:
            data[key] = self.scaler.transform(data[key])
        return data


def list_to_hash(my_list):
    """Turn iterable into a 16 char hash string."""
    joined_string = "".join(map(str, sorted(my_list)))
    hash_object = hashlib.md5(joined_string.encode())
    short_hash_string = hash_object.hexdigest()[:16]
    return short_hash_string


class GeneDataset(RayGeneDataset):
    default_config = {
        "dataset_path": "REQUIRED",
        "shuffle_files": True,
        "read_parquet_kwargs": None,
        "convert_categories": False,
        # data preprocessing
        "qc_genes": None,
        "sel_genes": None,
        "normalize": True,
        "log1p": True,
        "cell_target_count": 100000,
        "batch_size": 64,
        "concurency": 8,
        "scale_gene": True,
        "pca": True,
        "pca_n_components": 512,
        "scale_pc": True,
    }

    def __init__(
        self,
        dataset_path,
        shuffle_files=True,
        read_parquet_kwargs=None,
        convert_categories=False,
        qc_genes=None,
        sel_genes=None,
        normalize=True,
        log1p=True,
        cell_target_count=100000,
        batch_size=64,
        concurency=8,
        scale_gene=True,
        pca=True,
        pca_n_components=512,
        scale_pc=True,
    ):
        super().__init__(
            dataset_path=dataset_path,
            shuffle_files=shuffle_files,
            read_parquet_kwargs=read_parquet_kwargs,
        )

        self.convert_categories = convert_categories

        if isinstance(qc_genes, (str, pathlib.Path)):
            qc_genes = pd.read_csv(qc_genes, index_col=0, header=None).index
        self.qc_genes = qc_genes

        if isinstance(sel_genes, (str, pathlib.Path)):
            sel_genes = pd.read_csv(sel_genes, index_col=0, header=None).index
        self.sel_genes = sel_genes

        # cell level and count pre processing
        self.normalize = normalize
        self.log1p = log1p
        self.cell_target_count = cell_target_count
        self.batch_size = batch_size
        self.concurency = concurency

        # gene level pre processing
        self.scale_gene = scale_gene
        self.pca = pca
        self.pca_n_components = pca_n_components
        self.scale_pc = scale_pc
        self.scaler_dir = pathlib.Path(self.dataset_path) / "scaler"
        self.scaler_dir.mkdir(exist_ok=True)
        self.add_scaler = any([self.scale_gene, self.pca])

    # ================== Gene Scaler ==================

    def _get_scaler_path_from_folds(self, folds, gene_order):
        folds_hash = list_to_hash(folds)
        gene_hash = list_to_hash(gene_order)
        signature = (
            f"{folds_hash+gene_hash}+{self.pca}+"
            f"{self.pca_n_components}+"
            f"{self.scale_pc}+{self.scale_gene}"
        )
        path = self.scaler_dir / f"scaler_{signature}.joblib"
        return path

    def _get_scaler_or_fit(
        self,
        scaler_path=None,
        folds=None,
        use_cells=100000,
        gene_order=None,
    ):
        if not any([self.scale_gene, self.pca]):
            return IdentityScaler()

        need_fit = True
        if scaler_path is not None and scaler_path.exists():
            scaler = GeneDataScaler.load(scaler_path)
            # validate to make sure the scaler is compatible with the current config
            valid = scaler.validate(
                pca=self.pca,
                n_components=self.pca_n_components,
                scale_pc=self.scale_pc,
                scale_gene=self.scale_gene,
                gene_index=gene_order,
            )
            if not valid:
                print(
                    "The provided scaler is not compatible with the current config. "
                    "Refitting the scaler."
                )
                need_fit = True
            else:
                print("Loaded gene value scaler from", scaler_path)
                need_fit = False

        if need_fit:
            if folds is None:
                raise ValueError("folds must be provided to fit the scaler.")

            print(
                f"Fitting gene value scaler with {use_cells} "
                f"cells from folds {folds}..."
            )
            # init scaler and fit with adata
            scaler = GeneDataScaler(
                pca=self.pca,
                n_components=self.pca_n_components,
                scale_pc=self.scale_pc,
                scale_gene=self.scale_gene,
            )
            # adata after preprocessing and gene selection
            adata = self.get_sample_adata(
                n_cells=use_cells,
                folds=folds,
                qc_genes=self.qc_genes,
                sel_genes=self.sel_genes,
                filter_by_obs=None,
                sparse=True,
                normalize=self.normalize,
                cell_target_count=self.cell_target_count,
                log1p=self.log1p,
                local_shuffle_buffer_size=10000,
                concurrency=4,
            )
            scaler.fit(adata)

            if scaler_path is not None:
                scaler.dump(scaler_path)
        return scaler

    def get_gene_scaler(self, folds):
        """
        Get gene scaler with the given folds.
        The fitted scaler will be saved in the self.dataset_path/scaler folder
        with a file name containing the configuration of the scaler, later if the same
        configuration is used, the scaler will be loaded from the saved file.

        Parameters
        ----------
        folds: int | list[int]
            Folds to use to fit the scaler, usually the train folds.

        Returns
        -------
        scaler: GeneDataScaler
            Fitted gene scaler.
        """
        gene_order = self.pre_estimate_genes(
            qc_genes=self.qc_genes, sel_genes=self.sel_genes
        )
        scaler_path = self._get_scaler_path_from_folds(folds, gene_order)
        scaler = self._get_scaler_or_fit(
            folds=folds,
            use_cells=100000,
            scaler_path=scaler_path,
            gene_order=gene_order,
        )
        return scaler

    def _add_scaler_to_dataset(self, dataset, scaler, bs=1024):
        dataset = dataset.map_batches(
            ScaleGene,
            fn_constructor_kwargs={"scaler": scaler, "data_keys": [self.data_key]},
            batch_size=bs,
            concurrency=(1, self.concurency),
        )
        return dataset

    # ================== Ray Dataset ==================

    def get_processed_dataset(
        self,
        folds: int | list[int],
        dump_obs: bool = True,
        add_gene_idx_to_batch: bool = False,
        scaler=None,
    ):
        """Get processed dataset."""
        dataset, gene_order = super()._get_processed_dataset(
            folds=folds,
            dataset=None,
            cur_gene_order=None,
            convert_categories=self.convert_categories,
            normalize=self.normalize,
            cell_target_count=self.cell_target_count,
            log1p=self.log1p,
            max_step_concurrency=self.concurency,
            qc_genes=self.qc_genes,
            sel_genes=self.sel_genes,
        )

        if isinstance(scaler, GeneDataScaler):
            # here we need to pass in the scaler instead of getting it
            # with current folds using self.get_gene_scaler()
            # because scaler needs the train folds,
            # but here the folds may be from validation or test
            dataset = self._add_scaler_to_dataset(dataset, scaler)

        if add_gene_idx_to_batch:
            dataset = self._add_gene_idx_to_batch(dataset, gene_order)
        if dump_obs:
            self._dump_obs(dataset)
        return dataset

    def get_dataloader(self, folds, n_batches, scaler):
        """Get dataloader."""
        work_ds = self.get_processed_dataset(folds=folds, scaler=scaler)
        return super().get_dataloader(
            work_ds=work_ds,
            data_iter_kwargs={},
            n_batches=n_batches,
            batch_size=self.batch_size,
            as_torch=True,
        )
