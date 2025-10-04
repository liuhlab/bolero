from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from bolero.pp.seq import one_hot_encoding_torch


def prepare_qtl_table(qtl_table, resolution, stats_cols=None):
    """
    Parse the QTL table and return
    regions, region_to_mutation, mutations, and peaks.
    """
    stats_cols = stats_cols or ["beta", "PIP"]
    table = pd.read_csv(qtl_table, sep="\t")

    table = table.rename(columns={"id": "mutation_id"})

    # use peak region to represent peak id
    table["peak_id"] = (
        table["chr"]
        + ":"
        + table["peak-start"].astype(str)
        + "-"
        + table["peak-end"].astype(str)
    )
    # use snp id + peak id as the name of caqtl item
    table["qtl_id"] = table["mutation_id"] + "+" + table["peak_id"]
    # make sure the combination of snp id and peak id is unique
    assert table["qtl_id"].duplicated().sum() == 0

    # Borzoi Region information
    regions = table[["chr", "524k-start", "524k-end", "qtl_id"]].set_index("qtl_id")
    regions.columns = ["Chromosome", "Start", "End"]
    regions.index.name = "Name"

    # Region to mutation mapping
    region_to_mutation = table.set_index("qtl_id")["mutation_id"].to_dict()
    assert len(region_to_mutation) == table.shape[0]

    # Mutation information
    mutations = (
        table[["ref", "alt", "pos2start", "mutation_id"]]
        .drop_duplicates()
        .set_index("mutation_id")
    )
    assert (
        mutations.index.duplicated().sum() == 0
    ), "mutations with same id has different information."
    # mutation pos2start is 1 based (VCF), adjust to 0 based
    mutations["pos2start"] -= 1

    # Peak information
    peak_cols = ["chr", "peak-start", "peak-end", "peak_id", "qtl_id"] + [
        c for c in stats_cols if c in table.columns
    ]
    peaks = table[peak_cols].copy()
    peaks.columns = [
        "Chromosome",
        "Start",
        "End",
        "peak_id",
        "qtl_id",
    ] + peaks.columns.tolist()[5:]
    peaks = peaks.set_index("qtl_id")

    # Here we don't do any coordinates adjustment,
    # assuming the borzoi region should always be uncliped and
    # at length 524288 (1bp res) OR 16384 (32bp res)
    peaks["bin_start"] = (peaks["Start"] - table["524k-start"].values) / resolution
    peaks["bin_end"] = (peaks["End"] - table["524k-start"].values) / resolution
    peaks["bin_start"] = peaks["bin_start"].round().astype(int)
    peaks["bin_end"] = peaks["bin_end"].round().astype(int)
    assert peaks["bin_end"].max() <= 16384
    assert peaks["bin_start"].min() >= 0
    assert (
        peaks["bin_end"] - peaks["bin_start"] > 0
    ).all(), "peak bin end should be greater than start"
    return regions, region_to_mutation, mutations, peaks


def dna_substitution_(
    dna: torch.Tensor, mutation_start_pos: int, mut_seq: str
) -> torch.Tensor:
    """
    Substitute the DNA sequence at the given position with the mutation sequence.

    dna shape should be (4, seq_len), no batch dimension.
    """
    mut_one_hot = one_hot_encoding_torch(mut_seq, batch_dim=False, device=dna.device)
    mut_len = mut_one_hot.shape[1]
    dna[:, mutation_start_pos : mutation_start_pos + mut_len] = mut_one_hot
    return dna


class QTLManager:
    def __init__(
        self,
        qtl_table: str,
        resolution: int = 32,
        qtl_stats_cols=None,
        ypred_seq_len=16384,
    ):
        if qtl_stats_cols is None:
            qtl_stats_cols = ["beta", "PIP"]

        regions, region_to_mutation, mutations, peaks = prepare_qtl_table(
            qtl_table, resolution=resolution, stats_cols=qtl_stats_cols
        )
        self.resolution: int = resolution
        self.ypred_seq_len: int = ypred_seq_len
        # columns: ["Chromosome", "Start", "End"], index by region id
        self.regions: pd.DataFrame = regions
        self.qtl_ids = regions.index.tolist()

        # map qtl id to mutation id, one-to-one mapping
        self.region_to_mutation: dict[str, str] = region_to_mutation

        # columns: ["ref", "alt", "pos2start"], index by mutation id
        self.mutations: pd.DataFrame = mutations
        self.mutation_ids = mutations.index.tolist()

        # columns: ["Chromosome", "Start", "End", "Name"], index by qtl id
        self.peaks: pd.DataFrame = peaks
        return

    def mutate_dna(
        self,
        batch: dict[str, torch.Tensor],
        dna_key="__dna__",
        region_key="region_name",
        mutation_col="alt",
    ) -> dict[str, torch.Tensor]:
        """
        Mutate the DNA sequence in the batch according to the QTL regions.
        """
        dna = batch.pop(dna_key)
        regions = batch[region_key]  # list of qtl ids

        mut_dna_col = defaultdict(list)
        mutation_cols = ["ref", "alt"]
        for qtl_id, region_dna in zip(regions, dna):
            mutation_id = self.region_to_mutation[qtl_id]
            mut_dna_col["mutation_id"].append(mutation_id)
            mutation = self.mutations.loc[mutation_id]
            for mutation_col in mutation_cols:
                mut_seq = mutation[mutation_col]
                pos2start = mutation["pos2start"]
                # make substitution for both ref and alt,
                # because some times ref and alt are swapped
                region_dna = dna_substitution_(region_dna.clone(), pos2start, mut_seq)
                mut_dna_col[f"{dna_key}:{mutation_col}"].append(region_dna)

        for mutation_col in mutation_cols:
            _data = mut_dna_col[f"{dna_key}:{mutation_col}"]
            mut_dna_col[f"{dna_key}:{mutation_col}"] = torch.stack(_data, dim=0)
        mut_dna_col["mutation_id"] = np.array(mut_dna_col["mutation_id"])
        batch.update(mut_dna_col)
        return batch

    def get_peak_sum(self, batch: dict, ypred_key):
        """
        Calculate the sum of predictions over the QTL peak regions for each region/mutation in the batch.
        """
        batch_peaks = self.peaks.loc[batch["region_name"]]

        ypred = batch[ypred_key]
        assert ypred.shape[-1] == self.ypred_seq_len, (
            f"Expected ypred to have last dimension of {self.ypred_seq_len}, "
            f"but got {ypred.shape[-1]}"
        )
        ypred = ypred.clamp(min=0.0)  # Ensure non-negative predictions

        peak_sum = []
        for rid, (bs, be) in enumerate(batch_peaks[["bin_start", "bin_end"]].values):
            r_peak_sum = ypred[rid, :, bs:be].sum(axis=-1)
            peak_sum.append(r_peak_sum)
        peak_sum = torch.stack(peak_sum, dim=0)
        batch[f"{ypred_key}:peak"] = peak_sum
        return batch

    def add_peak_info(self, batch: dict):
        """
        Add peak information to the batch.
        """
        batch_peaks = self.peaks.loc[batch["region_name"]]
        batch["qtl_peaks"] = batch_peaks
        return batch


def prepare_peak_table(
    peak_table: str | pd.DataFrame, resolution: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Parse borzoi region and peak region from a predefined peak file.

    This file is expected to contain the following columns:
    - Chromosome
    - Start: Borzoi region start, target length 524288
    - End: Borzoi region end, target length 524288
    - Name: Borzoi region name
    - PeakStart: Peak region start, length are variable.
    - PeakEnd: Peak region end, length are variable.
    - Original_Name: Peak Name
    """
    if isinstance(peak_table, str):
        peak_table = pd.read_feather(peak_table)
    # Original_Name should be a unique peak name
    borzoi_region = peak_table[
        ["Chromosome", "Start", "End", "Original_Name"]
    ].reset_index(drop=True)
    peak_region = peak_table[
        ["Chromosome", "PeakStart", "PeakEnd", "Original_Name"]
    ].reset_index(drop=True)
    peak_region.columns = borzoi_region.columns.copy()

    # Here we don't do any coordinates adjustment,
    # assuming the borzoi region should always be uncliped and
    # at length 524288 (1bp res) OR 16384 (32bp res)
    peak_region["bin_start"] = (
        peak_region["Start"] - borzoi_region["Start"].values
    ) / resolution
    peak_region["bin_end"] = (
        peak_region["End"] - borzoi_region["Start"].values
    ) / resolution
    peak_region["bin_start"] = peak_region["bin_start"].round().astype(int)
    peak_region["bin_end"] = peak_region["bin_end"].round().astype(int)
    return borzoi_region, peak_region


class PeakManager:
    def __init__(
        self,
        peak_table: str,
        resolution: int = 32,
        ypred_seq_len=16384,
    ):
        borzoi_region, peak_region = prepare_peak_table(
            peak_table, resolution=resolution
        )
        self.resolution: int = resolution
        self.ypred_seq_len: int = ypred_seq_len
        # columns: ["Chromosome", "Start", "End"], index by region id
        self.regions: pd.DataFrame = borzoi_region
        self.region_ids = borzoi_region.index.tolist()

        # columns: ["Chromosome", "Start", "End", "Name"], index by region id and mutation id
        self.peaks: pd.DataFrame = peak_region
        self.peaks.index = self.peaks["Original_Name"]
        self.peak_ids = peak_region.index.tolist()
        return

    # def mutate_dna(
    #     self,
    #     batch: dict[str, torch.Tensor],
    #     *args,
    #     **kwargs
    # ) -> dict[str, torch.Tensor]:
    #     """
    #     # TODO Substitute DNA sequence with mutated peak sequence
    #     """
    #     return batch

    def get_peak_sum(self, batch: dict, ypred_key):
        """
        Calculate the sum of predictions over the QTL peak regions for each region/mutation in the batch.
        """
        # region idx should be one-to-one match to peak idx
        batch_peaks = self.peaks.loc[batch["region_name"]]

        ypred = batch[ypred_key]
        assert ypred.shape[-1] == self.ypred_seq_len, (
            f"Expected ypred to have last dimension of {self.ypred_seq_len}, "
            f"but got {ypred.shape[-1]}"
        )
        ypred = ypred.clamp(min=0.0)  # Ensure non-negative predictions

        peak_sum = []
        for rid, (bs, be) in enumerate(batch_peaks[["bin_start", "bin_end"]].values):
            r_peak_sum = ypred[rid, :, bs:be].sum(axis=-1)
            peak_sum.append(r_peak_sum)
        peak_sum = torch.stack(peak_sum, dim=0)
        batch[f"{ypred_key}:peak"] = peak_sum
        return batch

    def add_peak_info(self, batch: dict):
        """
        Add peak information to the batch.
        """
        batch_peaks = self.peaks.loc[batch["region_name"]]
        batch["peaks"] = batch_peaks
        return batch
