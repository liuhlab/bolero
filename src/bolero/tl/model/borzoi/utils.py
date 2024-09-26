import pyranges as pr
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from bolero.pp.genome import Genome
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
        self.hg38_regions = pr.read_bed(
            str(BORZOI_DATA_DIR / "hg38_sequences.bed"), as_df=True
        )
        self.hg38_regions.columns = ["Chromosome", "Start", "End", "Fold"]
        self.hg38_regions["Fold"] = self.hg38_regions["Fold"].str[4:].astype(int)

        self._mm10 = None
        self.mm10_regions = pr.read_bed(
            str(BORZOI_DATA_DIR / "mm10_sequences.bed"), as_df=True
        )
        self.mm10_regions.columns = ["Chromosome", "Start", "End", "Fold"]
        self.mm10_regions["Fold"] = self.mm10_regions["Fold"].str[4:].astype(int)

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

    def get_train_valid_test_regions(self, genome, split_id, region_length=524288):
        """
        Get train, valid, test regions for a given genome and split id.
        """
        if genome == "hg38":
            regions = self.hg38_regions.copy()
            genome = self.hg38
        elif genome == "mm10":
            regions = self.mm10_regions.copy()
            genome = self.mm10
        else:
            raise ValueError(f"Invalid genome: {genome}, choose from ['hg38', 'mm10']")

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


def compute_total_grad_norm(parameters, norm_type=2):
    """Compute the total norm of the gradients for a list of parameters"""
    total_norm = 0.0
    norm_type = float(norm_type)

    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    return total_norm ** (1.0 / norm_type)


def clamp_sqrt_large_value(data: torch.Tensor, threshold=50):
    """
    Clamp and sqrt the large values in the data, as performed in the Borzoi model.

    Parameters
    ----------
    data : torch.Tensor
        Count data.
    threshold : int, optional
        The threshold value, by default 50.
        Values <= threshold are not changed.
        Values > threshold will become threshold + sqrt(value - threshold).
    """
    data = torch.clamp_max(data, threshold) + torch.sqrt(
        torch.clamp_min(data - threshold, 0)
    )
    return data


def reverse_clamp_sqrt(data: torch.Tensor, threshold=50):
    """
    Reverse the clamp and sqrt operation in the Borzoi model.

    Parameters
    ----------
    data : torch.Tensor
        Count data.
    threshold : int, optional
        The threshold value, by default 50.
        Values <= threshold are not changed.
        Values > threshold will become threshold + (value - threshold)^2.
    """
    data = torch.clamp_max(data, threshold) + torch.pow(
        torch.clamp_min(data - threshold, 0), 2
    )
    return data


def make_warmup_scheduler(optimizer, warmup_steps):
    """
    Create a learning rate scheduler with warmup.
    """

    def _lr_lambda(step):
        if step < warmup_steps:
            lr_rate = (step + 1) / (warmup_steps + 1)
        else:
            lr_rate = 1.0
        return lr_rate

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def freeze_batchnorms_(model):
    """
    Freeze batchnorms in the model.

    # https://github.com/lucidrains/tf-bind-transformer/blob/main/tf_bind_transformer/tf_bind_transformer.py#L468-L470
    When finetune Enformer or Borzoi, it is recommended to freeze the batchnorms.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]

    for bn in bns:
        bn.eval()
        bn.track_running_stats = False
        for p in bn.parameters():
            p.requires_grad = False


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
