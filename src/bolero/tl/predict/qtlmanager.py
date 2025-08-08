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

    table["name"] = (
        table["chr"]
        + ":"
        + table["524k-start"].astype(str)
        + "-"
        + table["524k-end"].astype(str)
    )

    regions = (
        table[["chr", "524k-start", "524k-end", "name"]]
        .set_index("name")
        .drop_duplicates()
    )
    regions.columns = ["Chromosome", "Start", "End"]
    regions.index.name = "Name"

    region_and_mut = table[["id", "name"]].drop_duplicates()
    # make sure region and mutation is one-to-one match
    assert region_and_mut["id"].duplicated().sum() == 0
    assert region_and_mut["name"].duplicated().sum() == 0
    region_to_mutation = region_and_mut.set_index("name")["id"].to_dict()
    mutations = table[["ref", "alt", "pos2start", "id"]].set_index("id")

    peak_cols = ["chr", "peak-end", "peak-start", "name", "id"] + [
        c for c in stats_cols if c in table.columns
    ]
    peaks = table[peak_cols].copy()
    peaks.columns = [
        "Chromosome",
        "End",
        "Start",
        "region_id",
        "mutation_id",
    ] + peaks.columns.tolist()[5:]
    peaks = peaks.set_index(["region_id", "mutation_id"])
    peaks["Name"] = (
        peaks["Chromosome"].astype(str)
        + ":"
        + peaks["Start"].astype(str)
        + "-"
        + peaks["End"].astype(str)
    )
    # peaks["bin_start"] = (peaks["Start"] - table["524k-start"].values) / resolution
    # peaks["bin_end"] = (peaks["End"] - table["524k-start"].values) / resolution
    peaks["bin_start"] = ((peaks["Start"] - table["524k-start"].values) - 512) / resolution #TODO TG: Cleaner clipping
    peaks["bin_end"] = ((peaks["End"] - table["524k-start"].values) - 512) / resolution #TODO TG: Cleaner clipping
    peaks["bin_start"] = peaks["bin_start"].round().astype(int)
    peaks["bin_end"] = peaks["bin_end"].round().astype(int)
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
    def __init__(self, qtl_table: str, resolution: int = 32, qtl_stats_cols=None):
        if qtl_stats_cols is None:
            qtl_stats_cols = ["beta", "PIP"]

        regions, region_to_mutation, mutations, peaks = prepare_qtl_table(
            qtl_table, resolution=resolution, stats_cols=qtl_stats_cols
        )
        self.resolution: int = resolution
        # columns: ["Chromosome", "Start", "End"], index by region id
        self.regions: pd.DataFrame = regions
        self.region_ids = regions.index.tolist()

        # map region name to mutation id, one-to-one mapping
        self.region_to_mutation: dict[str, str] = region_to_mutation

        # columns: ["ref", "alt", "pos2start"], index by mutation id
        self.mutations: pd.DataFrame = mutations
        self.mutation_ids = mutations.index.tolist()

        # columns: ["Chromosome", "Start", "End", "Name"], index by region id and mutation id
        self.peaks: pd.DataFrame = peaks
        self.peak_ids = peaks.index.tolist()
        return

    def mutate_dna(
        self,
        batch: dict[str, torch.Tensor],
        dna_key="__dna__",
        region_key="region",
        mutation_col="alt",
    ) -> dict[str, torch.Tensor]:
        """
        Mutate the DNA sequence in the batch according to the QTL regions.
        """
        dna = batch.pop(dna_key)
        regions = batch[region_key]

        mut_dna_col = defaultdict(list)
        mutation_cols = ["ref", "alt"]
        for region, region_dna in zip(regions, dna):
            mutation_id = self.region_to_mutation[region]
            mut_dna_col["mutation_id"].append(mutation_id)
            mutation = self.mutations.loc[mutation_id]
            for mutation_col in mutation_cols:
                mut_seq = mutation[mutation_col]
                pos2start = mutation["pos2start"]
                # make substitution for both ref and alt,
                # because some times ref and alt are swapped
                # region_dna = dna_substitution_(region_dna.clone(), pos2start, mut_seq)
                region_dna = dna_substitution_(region_dna.clone(), pos2start-1, mut_seq)
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
        batch_peaks = self.peaks.loc[list(zip(batch["region"], batch["mutation_id"]))]
        ypred = batch[ypred_key]
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
        batch_peaks = self.peaks.loc[list(zip(batch["region"], batch["mutation_id"]))]
        batch["qtl_peaks"] = batch_peaks
        return batch
