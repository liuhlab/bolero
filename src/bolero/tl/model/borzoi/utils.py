import numpy as np
import pandas as pd
import pyranges as pr
import torch

from bolero.pp.genome import Genome
from bolero.pp.gtf import GTFDB, MM10_GTFDB_PATH
from bolero.utils import get_package_dir

BORZOI_DATA_DIR = get_package_dir() / "pkg_data/borzoi"


class BorzoiRegions:
    """
    Original source code from
    https://github.com/calico/baskerville/blob/main/src/baskerville/bed.py
    """

    # as said by the author: https://github.com/calico/borzoi/issues/11
    fold_splits = [
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

    def __init__(self):
        self._hg38 = None
        self._hg38_regions = None
        self._hg38_gene_regions = None
        self._hg38_gene_effective_regions = None

        self._mm10 = None
        self._mm10_regions = None
        self._mm10_gene_regions = None
        self._mm10_gene_effective_regions = None

        self.cur_idmap = None
        self.cur_effective_regions = None

    @property
    def hg38_regions(self):
        """Return hg38 borzoi regions."""
        if self._hg38_regions is None:
            self._hg38_regions = pr.read_bed(
                str(BORZOI_DATA_DIR / "hg38_sequences.bed"), as_df=True
            )
            self._hg38_regions.columns = ["Chromosome", "Start", "End", "Fold"]
            self._hg38_regions["Fold"] = self._hg38_regions["Fold"].str[4:].astype(int)
            self._hg38_regions_idmap = dict(enumerate(self._hg38_regions.index))
        return self._hg38_regions

    @property
    def hg38_gene_regions(self):
        """Return hg38 borzoi gene regions."""
        raise NotImplementedError("hg38_gene_regions is not implemented yet.")

    @property
    def mm10_regions(self):
        """Return mm10 borzoi regions."""
        if self._mm10_regions is None:
            self._mm10_regions = pr.read_bed(
                str(BORZOI_DATA_DIR / "mm10_sequences.bed"), as_df=True
            )
            self._mm10_regions.columns = ["Chromosome", "Start", "End", "Fold"]
            self._mm10_regions["Fold"] = self._mm10_regions["Fold"].str[4:].astype(int)
            self._mm10_regions_idmap = dict(enumerate(self._mm10_regions.index))
        return self._mm10_regions

    @property
    def mm10_gene_regions(self):
        """Return mm10 borzoi gene regions."""
        if self._mm10_gene_regions is None:
            self._mm10_gene_regions = pr.read_bed(
                str(BORZOI_DATA_DIR / "borzoi_mm10_gene.biccn_vm23.bed.gz"), as_df=True
            )
            self._mm10_gene_regions.columns = [
                "Chromosome",
                "Start",
                "End",
                "Name",
                "Fold",
            ]
            self._mm10_gene_regions["Fold"] = (
                self._mm10_gene_regions["Fold"].str[4:].astype(int)
            )
            self._mm10_gene_regions.set_index("Name", inplace=True)
            self._mm10_gene_regions_idmap = dict(
                enumerate(self._mm10_gene_regions.index)
            )
            self._mm10_gene_regions.reset_index(inplace=True, drop=True)

            # effective gene regions
            self._mm10_gene_effective_regions = pr.read_bed(
                str(
                    BORZOI_DATA_DIR
                    / "borzoi_mm10_gene.biccn_vm23.effective_gene_region_ext1k.bed.gz"
                ),
                as_df=True,
            )
            self._mm10_gene_effective_regions.set_index("Name", inplace=True)

        return self._mm10_gene_regions

    @property
    def hg38(self):
        """Return hg38 genome."""
        if self._hg38 is None:
            self._hg38 = Genome("hg38")
        return self._hg38

    @property
    def mm10(self):
        """Return mm10 genome."""
        if self._mm10 is None:
            self._mm10 = Genome("mm10")
        return self._mm10

    def _remove_overlap(self, bed1, bed2, bed3):
        """Remove overlap region from bed1 that overlaps with bed2 or bed3."""
        to_remove = []
        overlap1 = bed1.overlap(bed2).df
        overlap2 = bed1.overlap(bed3).df
        for overlap in [overlap1, overlap2]:
            if not overlap.empty:
                to_remove.extend(overlap["Original_Name"].values)
        return bed1.df[~bed1.df["Original_Name"].isin(to_remove)].copy()

    def _filter_gene_regions(self, regions, deg_list, idmap):
        """Filter gene regions by deg_list."""
        if isinstance(deg_list, str):
            deg_list = pd.read_csv(deg_list, header=None, index_col=0).index
        genes = regions.index.map(lambda x: idmap[x].split("_")[0])
        final_regions = regions.loc[genes.isin(deg_list)].copy()

        n_genes = genes[genes.isin(deg_list)].nunique()

        print(
            f"DEG list provided. Found {len(final_regions)} regions with {n_genes} genes."
        )
        return final_regions

    def get_train_valid_test_regions(
        self,
        genome,
        split_id,
        region_length=524288,
        use_gene_regions=False,
        deg_list=None,
    ):
        """
        Get train, valid, test regions for a given genome and split id.
        """
        if genome == "hg38":
            if use_gene_regions:
                regions = self.hg38_gene_regions.copy()
                self.cur_idmap = {}
            else:
                regions = self.hg38_regions.copy()
                self.cur_idmap = self._hg38_regions_idmap
            genome = self.hg38
        elif genome == "mm10":
            if use_gene_regions:
                regions = self.mm10_gene_regions.copy()
                self.cur_idmap = self._mm10_gene_regions_idmap
                self.cur_effective_regions = self._mm10_gene_effective_regions
                self.cur_gtf_db = GTFDB(MM10_GTFDB_PATH)
            else:
                regions = self.mm10_regions.copy()
                self.cur_idmap = self._mm10_regions_idmap
            genome = self.mm10
        else:
            raise ValueError(f"Invalid genome: {genome}, choose from ['hg38', 'mm10']")

        if use_gene_regions and deg_list is not None:
            regions = self._filter_gene_regions(regions, deg_list, self.cur_idmap)

        id_to_fold = regions["Fold"].to_dict()
        regions["Name"] = regions.index
        # blacklist regions not exist in any folds
        null_regions_bed = genome.genome_bed.subtract(pr.PyRanges(regions))
        sized_regions = genome.standard_region_length(
            regions,
            region_length,
            remove_blacklist=False,
            keep_original=True,
            boarder_strategy="drop",
        )
        # remove null regions
        sized_regions_bed = pr.PyRanges(sized_regions)
        null_ids = (
            sized_regions_bed.overlap(null_regions_bed).df["Original_Name"].values
        )
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


def create_gene_weights(r1, r2, resolution=32, r2_value=1, other_value=0.001):
    """
    Create gene weights for a region where position bins
    belong to r2 has r2_value and other positions has other_value.
    """
    r1s, r1e = r1
    r2s, r2e = r2

    size = (r1e - r1s) // resolution
    weights = np.full(size, other_value)

    # Determine the overlapping region
    overlap_start = max(r1s, r2s)
    overlap_end = min(r1e, r2e)

    # Check if there is an overlap
    if overlap_start < overlap_end:
        start_index = (overlap_start - r1s) // resolution
        end_index = (overlap_end - r1s) // resolution
        weights[start_index:end_index] = r2_value
    return weights


def add_position_weights_to_batch(
    batch,
    genome,
    region_id_map,
    effective_regions,
    resolution=32,
    gene_value=1,
    other_value=0.001,
):
    """Create region weights for the batch."""
    region_names = pd.Index(batch["original_name"].cpu().numpy()).map(region_id_map)
    effective_global_coords = genome.get_global_coords(
        effective_regions.loc[region_names]
    )
    region_global_coords = batch["region"].cpu().numpy()

    weight_array = []
    for r1, r2 in zip(region_global_coords, effective_global_coords):
        weights = create_gene_weights(
            r1, r2, resolution=resolution, r2_value=gene_value, other_value=other_value
        )
        weight_array.append(weights)
    weight_array = np.array(weight_array)
    weight_tensor = (
        torch.Tensor(weight_array).half().unsqueeze(1).to(batch["region"].device)
    )
    # shape: (bs, 1, length)
    batch["position_weights"] = weight_tensor
    return batch


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
