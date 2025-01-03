from collections import OrderedDict
from typing import Any, Iterable

import anndata
import joblib
import numpy as np
import pandas as pd
import torch

from bolero import Genome
from bolero.tl.dataset.ray_dataset import (
    RayGenomeChunkDataset,
)
from bolero.tl.model.borzoi.dataset import BorzoiDataset
from bolero.tl.model.borzoi.utils import BorzoiRegions
from bolero.tl.model.creformer.utils import atac_pre_processing
from bolero.utils import get_package_dir, understand_regions

DNA_NAME = "dna_one_hot"


class PrepareCREFormerBatches:
    def __init__(
        self,
        genome,
        atac_key,
        sample_key,
        gene_count_data,
        peak_token_path,
        use_genes: pd.Index = None,
        atac_scale=0.1,
        pseudobulk_ids=None,
        chunk_step_size=200000,
    ):
        if isinstance(genome, str):
            self.genome = Genome(genome)
        else:
            self.genome = genome

        pkg_dir = get_package_dir()
        genome_dict_path = f"{pkg_dir}/pkg_data/creformer/kmer_dict.pkl"
        self.genome_dict = torch.load(genome_dict_path, weights_only=True)

        gene_bed = genome.gtf_db.gene_bed
        gene_bed["TSS"] = gene_bed.apply(
            lambda row: row["Start"] if row["Strand"] == "+" else row["End"], axis=1
        )
        if use_genes is not None:
            gene_bed = gene_bed[gene_bed["Name"].isin(use_genes)].copy()
        self.gene_bed = gene_bed

        self.peak_token = pd.read_feather(peak_token_path).set_index("index")
        self.peak_bed = understand_regions(self.peak_token.index)

        # gene count data CPM
        self.gene_count_df: pd.DataFrame = anndata.read_h5ad(gene_count_data).to_df()
        if pseudobulk_ids is not None:
            pseudobulk_ids = pseudobulk_ids.astype(str)
            self.gene_count_df = self.gene_count_df.loc[pseudobulk_ids].copy()
            self.gene_count_df.index = self.gene_count_df.index.map(
                {pid: idx for idx, pid in enumerate(pseudobulk_ids)}
            )

        self.min_peaks = 8
        self.tss_flanking = 150000
        self.chunk_step_size = chunk_step_size
        self.max_peaks = 150
        self.atac_res = 32
        self.atac_key = atac_key
        self.sample_key = sample_key
        self.atac_scale = atac_scale
        self._gene_peak_records_cache = {}
        return

    def get_gene_peak_records(
        self,
        region: str,
    ):
        """
        For a given genome chunk region (e.g. 600k),
        return genes and peaks located +/- 151k around TSS
        """
        if region in self._gene_peak_records_cache:
            return self._gene_peak_records_cache[region]

        gene_bed = self.gene_bed
        peak_bed = self.peak_bed

        chrom, coords = region.split(":")
        region_start, region_end = map(int, coords.split("-"))

        gene_peak_records = {}

        # select relevant genes
        genes = gene_bed[
            (
                (gene_bed["Chromosome"] == chrom)
                & (
                    gene_bed["TSS"]
                    < (region_start + self.chunk_step_size + self.tss_flanking)
                )
                & (gene_bed["TSS"] - self.tss_flanking > region_start)
                & (gene_bed["TSS"] + self.tss_flanking < region_end)
            )
        ].copy()
        for _, (gene_id, gene_strand, gene_tss) in genes[
            ["Name", "Strand", "TSS"]
        ].iterrows():
            gene_strand = 1 if gene_strand == "+" else 0
            # select gene tss flanking peaks
            peaks = peak_bed[
                (
                    (peak_bed["Chromosome"] == chrom)
                    & (peak_bed["Start"] > gene_tss - self.tss_flanking + 500)
                    & (peak_bed["End"] < gene_tss + self.tss_flanking - 500)
                )
            ].copy()
            if peaks.shape[0] < self.min_peaks:
                continue

            peaks["center_to_tss"] = (peaks["End"] + peaks["Start"]) // 2 - gene_tss
            # select upto 150 peaks closest to TSS
            use_peaks = (
                peaks["center_to_tss"].abs().sort_values()[: self.max_peaks].index
            )
            peaks = peaks[peaks.index.isin(use_peaks)].copy()
            tss_pos = peaks["center_to_tss"].abs().argmin()
            gene_peak_records[(gene_id, gene_strand, tss_pos, region_start)] = (
                peaks.sort_values("center_to_tss").reset_index(drop=True)
            )

        # cache the result
        self._gene_peak_records_cache[region] = gene_peak_records
        return gene_peak_records

    def _get_peaks_dna_seq(self, peaks):
        peaks_idx = peaks["Name"].values
        all_seq = self.peak_token.loc[peaks_idx].values
        # shape (n_peaks, 1024)
        return all_seq

    def _get_peaks_atac(self, atac_data, peaks, region_start):
        rel_peaks = peaks.reset_index(drop=True)
        rel_peaks["StartBin"] = (rel_peaks["Start"] - region_start) // self.atac_res
        rel_peaks["EndBin"] = (rel_peaks["End"] - region_start) // self.atac_res + 5
        rel_peaks["Offset"] = (rel_peaks["Start"] - region_start) % self.atac_res

        peaks_atac = np.zeros((peaks.shape[0], 1024))
        for idx, (offset, sb, eb) in rel_peaks[
            ["Offset", "StartBin", "EndBin"]
        ].iterrows():
            peak_atac = atac_data[sb:eb] * self.atac_scale
            peak_atac = np.convolve(peak_atac, np.ones(3), mode="same")
            peak_atac = np.repeat(peak_atac, self.atac_res)
            peak_atac = atac_pre_processing(peak_atac)
            peaks_atac[idx] = peak_atac[offset : offset + 1024]
        # shape (n_peaks, 1024)
        return peaks_atac

    def _pad_to_fix_size(self, data):
        n_peaks = data.shape[0]
        pad_peaks = np.zeros((self.max_peaks, 1024))
        pad_peaks[:n_peaks] = data
        return n_peaks, pad_peaks

    def _process_single_gene(
        self, gene_id, gene_strand, tss_pos, region_start, peaks, batch_dict
    ) -> dict[str, Any]:
        peaks_dna = self._get_peaks_dna_seq(peaks)
        _, peaks_dna = self._pad_to_fix_size(peaks_dna)

        atac_data = batch_dict[self.atac_key].ravel()
        peaks_atac = self._get_peaks_atac(atac_data, peaks, region_start)
        n_peaks, peaks_atac = self._pad_to_fix_size(peaks_atac)

        gene_dict = {
            "gene_id": gene_id,
            "gene_strand": gene_strand,
            "tss_pos": tss_pos,
            "dna_in": peaks_dna,
            "atac_in": peaks_atac,
            "n_peaks": n_peaks,
        }
        for k, v in batch_dict.items():
            if k not in ["region", self.atac_key]:
                gene_dict[k] = v

        # add gene data
        gene_dict = self._add_gene_data(gene_dict)

        # add batch size dim
        gene_dict = {
            k: np.expand_dims(v, 0) if isinstance(v, np.ndarray) else np.array([v])
            for k, v in gene_dict.items()
        }
        return gene_dict

    def _add_gene_data(self, batch_dict):
        pid = batch_dict[self.sample_key]
        gid = batch_dict["gene_id"]
        gene_data = self.gene_count_df.loc[pid, gid]
        batch_dict["gene_cpm_data"] = gene_data
        return batch_dict

    def _process_single_batch(self, batch_dict: dict) -> list[dict[str, Any]]:
        # get genes and peaks
        gene_peak_records = self.get_gene_peak_records(batch_dict["region"])
        list_of_dict = []
        for (
            gene_id,
            gene_strand,
            tss_pos,
            region_start,
        ), peaks in gene_peak_records.items():
            single_gene_dict = self._process_single_gene(
                gene_id, gene_strand, tss_pos, region_start, peaks, batch_dict
            )
            list_of_dict.append(single_gene_dict)
        return list_of_dict

    def __call__(self, batch_dict):
        """Prepare a gene level batch."""
        batch_size = batch_dict[self.atac_key].shape[0]
        list_of_dict = []

        for i in range(batch_size):
            single_input = {
                k: v[i] if isinstance(v, np.ndarray) else v
                for k, v in batch_dict.items()
            }
            single_output = self._process_single_batch(single_input)
            list_of_dict.extend(single_output)
        return list_of_dict


class CREFormerDataset(BorzoiDataset, RayGenomeChunkDataset):
    """Singel cell dataset for CREFormer model."""

    default_config = {
        "genome": "REQUIRED",
        "dataset_path": "REQUIRED",
        "gene_data_path": "REQUIRED",
        "peak_token_path": "REQUIRED",
        "use_genes": None,
        "atac_scale": 0.1,
        "n_pseudobulks": 10,
        "read_parquet_kwargs": None,
        "batch_size": 1,
    }

    def __init__(
        self,
        genome,
        dataset_path: str,
        gene_data_path: str,
        peak_token_path: str,
        use_genes: str = None,
        atac_scale: float = 0.1,
        n_pseudobulks: int = 10,
        read_parquet_kwargs=None,
        batch_size: int = 1,
    ):
        RayGenomeChunkDataset.__init__(
            self,
            dataset_path=dataset_path,
            genome=genome,
            shuffle_files=True,
            read_parquet_kwargs=read_parquet_kwargs,
            load_one_hot=False,
        )
        # region properties
        self.n_pseudobulks = n_pseudobulks
        self.name_to_pseudobulker = OrderedDict()
        self.borzoi_regions = BorzoiRegions(self.genome)

        # gene data
        self.gene_data_path = gene_data_path
        self.peak_token_path = peak_token_path
        if use_genes is not None:
            self.use_genes = joblib.load(use_genes)
        else:
            self.use_genes = None
        self.atac_scale = atac_scale

        # Borzoi dataset requires these attributes but not used in scPrinter
        self.pos_resolution = 32  # use Borzoi Dataset
        self.reduce_resolution = False
        self.normalize_cov = True
        self.paired_data = False
        self.batch_size = batch_size
        return

    def __repr__(self) -> str:
        _str = f"scPrinterDataset\n" f"Dataset directory: {self.dataset_path}\n"
        return _str

    def _prepare_gene_batch(self, dataset, prefix="pseudobulk", concurrency=4):
        fn = PrepareCREFormerBatches
        fn_constructor_kwargs = {
            "genome": self.genome,
            "gene_count_data": self.gene_data_path,
            "atac_key": f"{prefix}:bulk_data",
            "sample_key": f"{prefix}:pseudobulk_ids",
            "use_genes": self.use_genes,
            "atac_scale": 0.1,
            "peak_token_path": self.peak_token_path,
            "pseudobulk_ids": self.name_to_pseudobulker[prefix].pseudobulk_ids,
        }
        dataset = dataset.flat_map(
            fn=fn,
            fn_constructor_kwargs=fn_constructor_kwargs,
            concurrency=concurrency,
        )
        return dataset

    def get_processed_dataset(self, folds, concurrency=32):
        """
        Preprocess the dataset to return pseudobulk region rows.
        """
        compressed_bytes_to_tensor_concurrency = (1, concurrency // 8)
        generate_pseudobulk_concurrency = (1, concurrency // 8)
        generate_gene_concurrency = (1, concurrency // 2)

        dataset = self._read_parquet(folds=folds)

        # filter meta region length equal to self.window_size
        dataset = self._filter_meta_region_length(dataset=dataset)

        # from compressed bytes to tensor (cell/sample by meta-region matrix) and other information
        dataset = self._compressed_bytes_to_tensor(
            dataset=dataset,
            concurrency=compressed_bytes_to_tensor_concurrency,
        )

        # generate pseudobulk
        self.signal_columns = []
        name_to_pseudobulker = self.name_to_pseudobulker
        if len(name_to_pseudobulker) > 0:
            dataset = self._generate_pseudobulk(
                dataset=dataset,
                concurrency=generate_pseudobulk_concurrency,
                normalize_cov=True,
            )

        # prepare gene
        dataset = self._prepare_gene_batch(
            dataset, concurrency=generate_gene_concurrency
        )
        return dataset

    def get_dataloader(
        self,
        folds: list[int],
        n_batches: int = None,
        batch_size: int = None,
        shuffle_rows: int = 100,
        concurrency: int = 32,
        **dataloader_kwargs: Any,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader.

        Parameters
        ----------
        folds : list
            List of folds to include in the dataset.
        n_batches : int, optional
            Number of batches to return, by default None.
        batch_size : int, optional
            Batch size, by default None.
        shuffle_rows : int, optional
            The size of the local shuffle buffer, by default 500.
        concurrency : int, optional
            The number of workers to use for processing the dataset, by default 32.
        **dataloader_kwargs
            Additional keyword arguments pass to ray.data.Dataset.iter_batches.

        Returns
        -------
        DataLoader
            The dataloader.
        """
        # dataset_kwargs will be passed to self.get_processed_dataset method
        dataset_kwargs = {
            "folds": folds,  # for borzoi we don't split train/valid/test via chromosomes, so all chromosomes are included
            "concurrency": concurrency,
        }
        data_iter_kwargs = dataloader_kwargs

        loader = self._get_dataloader_with_wrapper(
            dataset_kwargs=dataset_kwargs,
            data_iter_kwargs=data_iter_kwargs,
            as_torch=False,
            shuffle_rows=shuffle_rows,
            shuffle_eval=True,
            n_batches=n_batches,
            batch_size=self.batch_size if batch_size is None else batch_size,
        )
        return loader

    def get_train_valid_test(self, fold):
        """Get the train, valid, and test folds and regions for the given fold."""
        fold_split = self.borzoi_regions.fold_splits[fold]
        train_folds = fold_split["train"]
        valid_folds = fold_split["valid"]
        test_folds = fold_split["test"]
        return train_folds, valid_folds, test_folds
