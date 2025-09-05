import pathlib

import numpy as np
import pandas as pd
import pyranges as pr
import torch

from bolero.pp.genome import Genome
from bolero.utils import get_package_dir

BORZOI_DATA_DIR = get_package_dir() / "pkg_data/borzoi"
# TODO: change this dir to pkg dir
BORZOI_GENE_DIR = pathlib.Path(
    "/large_storage/zhoulab/hanliu/250901-GeneCountPred/prepare/regions"
)
BORZOI_REGION_SIZE = 524288

# as said by the author: https://github.com/calico/borzoi/issues/11
FOLD_SPLITS = [
    {
        "train": [2, 3, 4, 5, 6, 7],
        "valid": [0],
        "test": [1],
    },
    {
        "train": [0, 3, 4, 5, 6, 7],
        "valid": [1],
        "test": [2],
    },
    {
        "train": [0, 1, 4, 5, 6, 7],
        "valid": [2],
        "test": [3],
    },
    {
        "train": [0, 1, 2, 5, 6, 7],
        "valid": [3],
        "test": [4],
    },
]


class BorzoiRegions:
    """
    Original source code from
    https://github.com/calico/baskerville/blob/main/src/baskerville/bed.py
    """

    fold_splits = FOLD_SPLITS

    def __init__(self, genome):
        if isinstance(genome, Genome):
            self.genome = genome
            self.genome_name = genome.name
        else:
            self.genome = Genome(genome)
            self.genome_name = genome

        self._borzoi_regions = None
        self.cur_idmap = None
        self._cur_effective_regions = None

    @property
    def borzoi_regions(self):
        """Return borzoi regions."""
        if self._borzoi_regions is None:
            _path = BORZOI_DATA_DIR / f"{self.genome_name}_sequences.bed.gz"
            assert _path.exists(), f"Genome {self.genome_name} does not have borzoi regions file in {BORZOI_DATA_DIR}."

            bed = pr.read_bed(str(_path), as_df=True)
            if self.genome_name not in ["hg38", "mm10"]:
                # for custom genome, skip first and last region of each chromosome to avoid border issue
                bed = (
                    bed.groupby("Chromosome", observed=True)
                    .apply(lambda df: df.sort_values("Start").iloc[1:-1])
                    .reset_index(drop=True)
                )

            self._borzoi_regions = bed
            self._borzoi_regions.columns = ["Chromosome", "Start", "End", "Fold"]
            self._borzoi_regions["Fold"] = (
                self._borzoi_regions["Fold"].str[4:].astype(int)
            )
            self.cur_idmap = dict(enumerate(self._borzoi_regions.index))
        return self._borzoi_regions

    def _remove_overlap(self, bed1, bed2, bed3):
        """Remove overlap region from bed1 that overlaps with bed2 or bed3."""
        to_remove = []
        overlap1 = bed1.overlap(bed2).df
        overlap2 = bed1.overlap(bed3).df
        for overlap in [overlap1, overlap2]:
            if not overlap.empty:
                to_remove.extend(overlap["Original_Name"].values)
        return bed1.df[~bed1.df["Original_Name"].isin(to_remove)].copy()

    def get_train_valid_test_regions(self, split_id, region_length=524288, **kwargs):
        """
        Get train, valid, test regions for a given genome and split id.
        """
        regions = self.borzoi_regions.copy()

        id_to_fold = regions["Fold"].to_dict()
        regions["Name"] = regions.index
        # blacklist regions not exist in any folds
        null_regions_bed = self.genome.genome_bed.subtract(pr.PyRanges(regions))
        sized_regions = self.genome.standard_region_length(
            regions,
            region_length,
            remove_blacklist=False,
            keep_original=True,
            boarder_strategy="drop",
        )

        # remove null regions
        sized_regions_bed = pr.PyRanges(sized_regions)
        null_regions = sized_regions_bed.overlap(null_regions_bed).df
        if null_regions.shape[0] > 0:
            null_ids = null_regions["Original_Name"].values
            sized_regions = sized_regions[
                ~sized_regions["Original_Name"].isin(null_ids)
            ].copy()
        sized_regions["fold"] = sized_regions["Original_Name"].map(id_to_fold)

        # split to folds
        fold_split = self.fold_splits[split_id]
        train_regions = sized_regions[
            sized_regions["fold"].isin(fold_split["train"])
        ].copy()
        valid_regions = sized_regions[
            sized_regions["fold"].isin(fold_split["valid"])
        ].copy()
        test_regions = sized_regions[
            sized_regions["fold"].isin(fold_split["test"])
        ].copy()

        # make sure the regions are not overlapping
        train_bed = pr.PyRanges(train_regions)
        valid_bed = pr.PyRanges(valid_regions)
        test_bed = pr.PyRanges(test_regions)

        train_regions = self._remove_overlap(train_bed, valid_bed, test_bed)
        valid_regions = self._remove_overlap(valid_bed, train_bed, test_bed)
        test_regions = self._remove_overlap(test_bed, train_bed, valid_bed)
        return train_regions, valid_regions, test_regions


class BorzoiGeneRegions(BorzoiRegions):
    _gene_to_mask: pd.DataFrame

    @staticmethod
    def _add_mask_coords(regions):
        records = []
        for _, row in regions.iterrows():
            if row["Strand"] == "+":
                ms = row["GeneStart"] - row["Start"]
                me = min(row["GeneEnd"] - row["Start"], 524288)
            else:
                # for "-" strand gene, we will reverse comp the DNA input
                # (see the ReverseComplementMinusStrand class),
                # therefore we take distance to the "End" to calculate mask
                ms = row["End"] - row["GeneEnd"]
                me = min(row["End"] - row["GeneStart"], 524288)
            records.append([ms, me])
        records = pd.DataFrame(
            records, index=regions.index, columns=["MaskStart", "MaskEnd"]
        )
        records["MaskStart"] -= 1

        regions = pd.concat([regions, records], axis=1)
        return regions

    @property
    def borzoi_regions(self):
        """Return borzoi regions."""
        if self._borzoi_regions is None:
            _path = BORZOI_GENE_DIR / f"{self.genome_name}.GeneBorzoiRegion.feather"
            assert _path.exists(), f"Genome {self.genome_name} does not have borzoi regions file in {BORZOI_GENE_DIR}."

            bed = pd.read_feather(_path)
            bed["Name"] = bed["Name"].map(
                lambda name: name.split(".")[0]
            )  # remove version number in gene id
            # bed.columns: [
            #     'Chromosome', 'Start', 'End', 'Name',
            #     'Strand', 'TSS', 'GeneStart', 'GeneEnd', 'Fold'
            # ]
            # Start and End are borzoi region coords (524288)

            # skip first and last region of each chromosome to avoid border issue
            bed = (
                bed.groupby("Chromosome", observed=True)
                .apply(lambda df: df.sort_values("Start").iloc[1:-1])
                .reset_index(drop=True)
            )
            self._borzoi_regions = bed.reset_index(drop=True).drop_duplicates(
                subset="Name"
            )
            self._borzoi_regions = self._add_mask_coords(self._borzoi_regions)
            self._gene_to_mask: pd.DataFrame = self._borzoi_regions.set_index("Name")[
                ["MaskStart", "MaskEnd"]
            ].copy()
            self.cur_idmap: dict[int, str] = bed["Name"].to_dict()
        return self._borzoi_regions

    def get_gene_mask(self, genes, dna) -> torch.Tensor:
        """
        Get mask bins for given genes.
        """
        mask_coords = self._gene_to_mask.loc[genes].values
        mask = gene_mask_coords_to_mask(mask_coords, dna)
        return mask

    def get_train_valid_test_regions(self, split_id, deg_list=None, **kwargs):
        """
        Get train, valid, test regions for a given genome and split id.
        """
        regions = self.borzoi_regions[
            [
                "Chromosome",
                "Start",
                "End",
                "Name",
                "Strand",
                "Fold",
                "MaskStart",
                "MaskEnd",
            ]
        ].copy()
        regions = regions[regions["Fold"] != "."].copy()
        regions["Fold"] = regions["Fold"].astype(int)
        regions["Original_Name"] = regions.index

        if deg_list is not None:
            regions = regions[regions["Name"].isin(deg_list)].copy()
            print(f"After deg_list filter, {regions.shape[0]} gene regions remain.")
            fold_count_min = regions["Fold"].value_counts().min()
            if fold_count_min == 0:
                raise ValueError(
                    "Some fold has no gene left after deg_list filter, "
                    "make sure deg_list are matched with gene ids."
                )
            if fold_count_min < 10:
                print(
                    "Some fold has < 10 gene regions remain after deg_list filter, "
                    "consider adjusting your deg_list."
                )

        # split to folds
        fold_split = self.fold_splits[split_id]
        train_regions = regions[regions["Fold"].isin(fold_split["train"])].copy()
        valid_regions = regions[regions["Fold"].isin(fold_split["valid"])].copy()
        test_regions = regions[regions["Fold"].isin(fold_split["test"])].copy()

        # make sure the regions are not overlapping
        train_bed = pr.PyRanges(train_regions)
        valid_bed = pr.PyRanges(valid_regions)
        test_bed = pr.PyRanges(test_regions)

        train_regions = self._remove_overlap(train_bed, valid_bed, test_bed)
        valid_regions = self._remove_overlap(valid_bed, train_bed, test_bed)
        test_regions = self._remove_overlap(test_bed, train_bed, valid_bed)
        return train_regions, valid_regions, test_regions


def gene_mask_coords_to_mask(
    gene_mask: torch.Tensor | np.ndarray, dna: torch.Tensor
) -> torch.Tensor:
    """Turn gene mask coordinates to gene mask tensor."""
    if isinstance(gene_mask, torch.Tensor):
        gene_mask = gene_mask.cpu().numpy()

    # coords (bs, 2) to gene mask (bs, 1, seq_len)
    bs, _, seq_len = dna.shape
    gene_mask_tensor = torch.zeros((bs, 1, seq_len), dtype=dna.dtype, device=dna.device)
    for i, (start, end) in enumerate(gene_mask):
        gene_mask_tensor[i, 0, start:end] = 1.0
    return gene_mask_tensor


class MultiBorzoiRegions:
    """
    A class to handle multiple BorzoiRegions for different keys.
    """

    fold_splits = FOLD_SPLITS

    def __init__(self, key_to_genome):
        self.key_to_genome = key_to_genome
        self.regions = {
            key: BorzoiRegions(genome) for key, genome in key_to_genome.items()
        }
        self.fold_splits = BorzoiRegions.fold_splits

    def get_train_valid_test_regions(self, split_id, **kwargs):
        """
        Get train, valid, test regions for each key, and add key to the region bed.
        Concate them in the end.
        """
        train_regions, valid_regions, test_regions = [], [], []
        for key, br in self.regions.items():
            tr, vr, te = br.get_train_valid_test_regions(split_id=split_id, **kwargs)
            tr["key"] = key
            vr["key"] = key
            te["key"] = key
            train_regions.append(tr)
            valid_regions.append(vr)
            test_regions.append(te)
        train_regions = pd.concat(train_regions, ignore_index=True)
        valid_regions = pd.concat(valid_regions, ignore_index=True)
        test_regions = pd.concat(test_regions, ignore_index=True)
        return train_regions, valid_regions, test_regions


def compute_total_grad_norm(parameters, norm_type=2):
    """Compute the total norm of the gradients for a list of parameters"""
    total_norm = 0.0
    norm_type = float(norm_type)

    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    return total_norm ** (1.0 / norm_type)


def clamp_sqrt_large_value(
    data: torch.Tensor, power=0.75, threshold=200, effective_bool=None
):
    """
    This is special data transformation used for RNA-seq in the Borzoi model.
    Power the data, then clamp and sqrt the large residual values in the data.

    Parameters
    ----------
    data : torch.Tensor
        Count data of shape (bs, channels, length).
    power : float, optional
        The power value, by default 0.75.
    threshold : int, optional
        The threshold value, by default 200.
        Values <= threshold are not changed.
        Values > threshold will become threshold + sqrt(value - threshold).
    effective_bool : torch.Tensor, optional
        A boolean tensor to indicate which channels are effective, by default None.
        If None, all channels are effective.
    """
    if isinstance(effective_bool, str):
        bool_list = [True if char == "1" else False for char in effective_bool]
        effective_bool = torch.tensor(bool_list)  # shape: (channels,)

    # data: (bs, channels, length)
    if effective_bool is None:
        result = torch.pow(data, power)
        result = torch.clamp_max(result, threshold) + torch.sqrt(
            torch.clamp_min(result - threshold, 0)
        )
    else:
        # effective bool is a bool tensor with the same length as the channels
        result = data.clone()
        effective_data = data[:, effective_bool, :]
        effective_data = torch.pow(effective_data, power)
        effective_data = torch.clamp_max(effective_data, threshold) + torch.sqrt(
            torch.clamp_min(effective_data - threshold, 0)
        )
        result[:, effective_bool, :] = effective_data
    return result


def reverse_clamp_sqrt(data: torch.Tensor, power=0.75, threshold=200):
    """
    Reverse the clamp and sqrt operation in the Borzoi model.

    Parameters
    ----------
    data : torch.Tensor
        Count data.
    power : float, optional
        The power value, by default 0.75.
    threshold : int, optional
        The threshold value, by default 200.
        Values <= threshold are not changed.
        Values > threshold will become threshold + (value - threshold)^2.
    """
    data = torch.clamp_max(data, threshold) + torch.pow(
        torch.clamp_min(data - threshold, 0), 2
    )
    data = torch.pow(data, 1 / power)
    return data


class MovingMetric:
    """
    Compute moving metrics of an FIFO queue.
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.data = []
        self.full = False

    def __len__(self):
        return len(self.data)

    def update(self, value):
        """Update the data with FIFO."""
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)
            self.full = True

    def quantile(self, q):
        """Moving quantile."""
        return torch.quantile(torch.tensor(self.data), q)

    def mean(self):
        """Moving mean."""
        return torch.mean(torch.tensor(self.data))

    def std(self):
        """Moving std."""
        return torch.std(torch.tensor(self.data))
