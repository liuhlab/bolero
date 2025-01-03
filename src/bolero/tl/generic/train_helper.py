import math
import pathlib
from collections import defaultdict
from typing import Union

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from tqdm import tqdm

hg38_splits = [None] * 5
hg38_splits[0] = {
    "test": ["chr1", "chr3", "chr6"],
    "valid": ["chr8", "chr20"],
    "train": [
        "chr2",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr21",
        "chr22",
        # "chrX",
        # "chrY",
    ],
}
hg38_splits[1] = {
    "test": ["chr2", "chr8", "chr9", "chr16"],
    "valid": ["chr12", "chr17"],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr11",
        "chr13",
        "chr14",
        "chr15",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
        # "chrY",
    ],
}
hg38_splits[2] = {
    "test": [
        "chr4",
        "chr11",
        "chr12",
        "chr15",
        # "chrY",
    ],
    "valid": ["chr22", "chr7"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr10",
        "chr13",
        "chr14",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chrX",
    ],
}
hg38_splits[3] = {
    "test": ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
    "valid": ["chr6", "chr21"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        # "chrY",
    ],
}
hg38_splits[4] = {
    "test": ["chr7", "chr13", "chr17", "chr19", "chr21", "chrX"],
    "valid": ["chr10", "chr18"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr14",
        "chr15",
        "chr16",
        "chr20",
        "chr22",
        # "chrY",
    ],
}


mm10_splits = [None] * 5
mm10_splits[0] = {
    "test": ["chr1", "chr6", "chr12", "chr13", "chr16"],
    "valid": ["chr8", "chr11", "chr18", "chr19", "chrX"],
    "train": [
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr14",
        "chr15",
        "chr17",
    ],
}
mm10_splits[1] = {
    "test": ["chr2", "chr7", "chr10", "chr14", "chr17"],
    "valid": [
        "chr5",
        "chr9",
        "chr13",
        "chr15",
        # "chrY",
    ],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr6",
        "chr8",
        "chr11",
        "chr12",
        "chr16",
        "chr18",
        "chr19",
        "chrX",
    ],
}
mm10_splits[2] = {
    "test": ["chr3", "chr8", "chr13", "chr15", "chr17"],
    "valid": [
        "chr2",
        "chr9",
        "chr11",
        "chr12",
        # "chrY",
    ],
    "train": [
        "chr1",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr14",
        "chr16",
        "chr18",
        "chr19",
        "chrX",
    ],
}
mm10_splits[3] = {
    "test": ["chr4", "chr9", "chr11", "chr14", "chr19"],
    "valid": [
        "chr1",
        "chr7",
        "chr12",
        "chr13",
        # "chrY",
    ],
    "train": [
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr10",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chrX",
    ],
}
mm10_splits[4] = {
    "test": [
        "chr5",
        "chr10",
        "chr12",
        "chr16",
        # "chrY",
    ],
    "valid": ["chr3", "chr7", "chr14", "chr15", "chr18"],
    "train": [
        "chr1",
        "chr2",
        "chr4",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr13",
        "chr17",
        "chr19",
        "chrX",
    ],
}

corigami_hg38_splits = [None]
corigami_hg38_splits[0] = {
    "test": [
        "chr15",
    ],
    "valid": ["chr10"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
    ],
}


def get_splits(genome: str, split_id: int) -> dict[str, Union[list, None]]:
    """
    Get the splits for a given genome and split ID.

    Parameters
    ----------
        genome (str): The genome (either "hg38" or "mm10").
        split_id (int): The split ID (0 to 4).

    Returns
    -------
        dict: A dictionary containing the splits for the given genome and split ID.
              The dictionary has keys "test", "valid", and "train", each mapping to a list of chromosome names.
              The key "test" maps to the chromosomes used for testing,
              the key "valid" maps to the chromosomes used for validation,
              and the key "train" maps to the chromosomes used for training.

    Raises
    ------
        ValueError: If the split ID is invalid or the genome is unknown.
    """
    if split_id < 0 or split_id >= 5:
        raise ValueError(f"Invalid split_id {split_id}")
    if genome == "hg38":
        return hg38_splits[split_id]
    elif genome == "mm10":
        return mm10_splits[split_id]
    else:
        raise ValueError(f"Unknown genome {genome}")


class CumulativeCounter:
    """Cumulative counter for calculating mean and sum of values."""

    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update the cumulative counter with a new value.

        Parameters
        ----------
            value (np.ndarray or torch.Tensor): The value to be added to the counter.
        """
        if isinstance(value, (int, float)):
            self.total += value
            self.count += 1
            return
        else:
            try:
                self.total += float(np.nansum(value))
            except TypeError:
                # torch
                self.total += float(torch.nansum(value).detach().cpu().item())
            # both numpy and torch will work
            self.count += np.prod(value.shape)
            return

    def mean(self) -> float:
        """
        Calculate the mean of the values in the counter.

        Returns
        -------
            float: The mean value.
        """
        if self.count == 0:
            return 0
        return self.total / self.count

    def sum(self) -> float:
        """
        Calculate the sum of the values in the counter.

        Returns
        -------
            float: The sum value.
        """
        return self.total


class CumulativeCounterPerChannel:
    """Cumulative counter for calculating mean values for an array along the dimension 1."""

    def __init__(self):
        self.sum_array = None
        self.count = 0

    def update(self, value: torch.Tensor) -> None:
        """
        Update the cumulative counter with a new value.

        Parameters
        ----------
            value (torch.Tensor): The value to be added to the counter.
        """
        if self.sum_array is None:
            self.sum_array = torch.sum(value, dim=1)[None, :]
        else:
            self.sum_array += torch.sum(value, dim=1)[None, :]

        # both numpy and torch will work
        self.count += value.shape[1]  # batch size

    def mean(self) -> float:
        """
        Calculate the mean of the values in the counter.

        Returns
        -------
            float: The mean value.
        """
        if self.count == 0:
            return 0
        return (self.sum_array / self.count).detach().cpu().tolist()


class CumulativePearson:
    """Cumulative pearson counter for calculating the pearson correlation coefficient."""

    def __init__(self):
        self.count = 0
        self.x_counter = CumulativeCounter()
        self.y_counter = CumulativeCounter()
        self.xy_counter = CumulativeCounter()
        self.x2_counter = CumulativeCounter()
        self.y2_counter = CumulativeCounter()

    def update(
        self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Update the cumulative pearson counter with new values.

        Parameters
        ----------
            x (np.ndarray or torch.Tensor): The x values to be added to the counter.
            y (np.ndarray or torch.Tensor): The y values to be added to the counter.
        """
        assert (
            x.shape == y.shape
        ), f"Shape mismatch between x and y, {x.shape} != {y.shape}"
        self.x_counter.update(x)
        self.y_counter.update(y)
        self.xy_counter.update(x * y)
        self.x2_counter.update(x**2)
        self.y2_counter.update(y**2)

    def corr(self) -> float:
        """
        Calculate the pearson correlation coefficient.

        Returns
        -------
            float: The pearson correlation coefficient.
        """
        nx = self.x_counter.count
        ny = self.y_counter.count
        assert nx == ny, f"Length mismatch between x and y, {nx} != {ny}"
        count = nx

        if nx == 0:
            return 0

        sum_x = self.x_counter.sum()
        mean_x = self.x_counter.mean()
        sum_y = self.y_counter.sum()
        mean_y = self.y_counter.mean()
        sum_xy = self.xy_counter.sum()
        sum_x2 = self.x2_counter.sum()
        sum_y2 = self.y2_counter.sum()

        covariance = sum_xy - mean_x * sum_y - mean_y * sum_x + count * mean_x * mean_y
        variance_x = sum_x2 - 2 * mean_x * sum_x + count * mean_x**2
        variance_y = sum_y2 - 2 * mean_y * sum_y + count * mean_y**2

        # Pearson correlation
        correlation = covariance / (
            math.sqrt(variance_x * variance_y) + 1e-8
        )  # Adding small value for numerical stability

        assert (
            -1 <= correlation <= 1
        ), f"Invalid correlation value {correlation}, {covariance}, {variance_x}, {variance_y}"
        return correlation


def batch_pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the batch Pearson correlation coefficient between two tensors.

    Parameters
    ----------
        x (Tensor): The input tensor x of shape (batch_size, features).
        y (Tensor): The input tensor y of shape (batch_size, features).

    Returns
    -------
        Tensor: The batch Pearson correlation coefficients of shape (batch_size,).

    Notes
    -----
        The Pearson correlation coefficient measures the linear relationship between two variables.
        It is computed as the covariance of x and y divided by the product of their standard deviations.

    """
    bs = x.shape[0]
    x = x.view(bs, -1)
    y = y.view(bs, -1)

    # Compute means along the batch dimension
    mean_x = torch.mean(x, dim=-1, keepdim=True)
    mean_y = torch.mean(y, dim=-1, keepdim=True)

    diff_x = x - mean_x
    diff_y = y - mean_y

    # Compute covariance and variance
    covariance = torch.sum(diff_x * diff_y, dim=-1)
    variance_x = torch.sum((diff_x) ** 2, dim=-1)
    variance_y = torch.sum((diff_y) ** 2, dim=-1)

    # Pearson correlation
    correlation = covariance / (
        torch.sqrt(variance_x * variance_y) + 1e-8
    )  # Adding small value for numerical stability
    return correlation


def batch_pearson_correlation_per_channel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    Compute the batch Pearson correlation coefficient between two tensors along the 2nd dimension

    Parameters
    ----------
        x (Tensor): The input tensor x of shape (batch_size, features).
        y (Tensor): The input tensor y of shape (batch_size, features).

    Returns
    -------
        Tensor: The batch Pearson correlation coefficients of shape (batch_size,).

    Notes
    -----
        The Pearson correlation coefficient measures the linear relationship between two variables.
        It is computed as the covariance of x and y divided by the product of their standard deviations.

    """
    # Assuming your data tensor has shape [64, 9, 1000]
    _, num_cell_types, _ = x.shape

    # Initialize a dictionary to store correlations for each cell type
    correlations_lst = []

    for cell_type_idx in range(num_cell_types):
        # Extract data for this cell type
        x_cell_type = x[:, cell_type_idx, :]  # Shape: [64, 1000]
        y_cell_type = y[:, cell_type_idx, :]  # Shape: [64, 1000]

        # Compute Pearson correlation for this cell type
        corr_cell_type = batch_pearson_correlation(x_cell_type, y_cell_type)

        # Store the correlation coefficient
        correlations_lst.append(corr_cell_type[None, :])

    cell_type_correlation = torch.cat(correlations_lst, dim=0)  # Shape: [9, 64]

    return cell_type_correlation


def chr_score(matrix, res=10000, radius=500000, pseudocount_coeff=30):
    """
    Calculate the score for each locus in the matrix based on the surrounding loci.
    """
    pseudocount = matrix.mean() * pseudocount_coeff
    pixel_radius = int(radius / res)
    scores = []
    for _, loc in enumerate(range(len(matrix))):
        scores.append(point_score(loc, pixel_radius, matrix, pseudocount))
    return scores


def point_score(locus, radius, matrix, pseudocount):
    """
    Calculate the score for a single locus in the matrix based on the surrounding loci.
    """
    l_edge = max(locus - radius, 0)
    r_edge = min(locus + radius, len(matrix))
    l_mask = matrix[l_edge:locus, l_edge:locus]
    r_mask = matrix[locus:r_edge, locus:r_edge]
    center_mask = matrix[l_edge:locus, locus:r_edge]
    score = (max(l_mask.mean(), r_mask.mean()) + pseudocount) / (
        center_mask.mean() + pseudocount
    )
    return score


def insulation_pearson(preds: torch.Tensor, targets: torch.Tensor):
    """
    Calculate the pearson correlation between the predicted insulation score and the target insulation score
    """
    scores = []
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    for pred, target in zip(preds, tqdm(targets)):
        pred_insu = np.array(chr_score(pred))
        label_insu = np.array(chr_score(target))
        nas = np.logical_or(np.isnan(pred_insu), np.isnan(label_insu))
        if nas.sum() == len(pred):
            scores.append(np.nan)
        else:
            metric, p_val = pearsonr(pred_insu[~nas], label_insu[~nas])
            scores.append(metric)
    results = scores
    return results


def safe_save(obj: torch.Tensor, path: str) -> None:
    """
    Save the given object to the specified path in a safe manner.

    Parameters
    ----------
        obj (torch.Tensor): The object to be saved.
        path (str): The path where the object will be saved.

    Returns
    -------
        None
    """
    temp_path = f"{path}.temp"
    torch.save(obj, temp_path)
    pathlib.Path(temp_path).rename(path)
    return


def compare_configs(config1, config2):
    """
    Compare two dictionaries to see if they are identical, considering only
    supported data types (numbers, strings, lists of numbers and strings, bools, and None).
    Other data types are ignored in the comparison.
    """

    def _is_valid_value(value):
        """Check if the value is of a supported type."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return True
        if isinstance(value, list):
            return all(isinstance(item, (int, float, str)) for item in value)
        if isinstance(value, tuple):
            return all(isinstance(item, (int, float, str)) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(item, (int, float, str, list)) for item in value.values()
            ) and all(isinstance(key, str) for key in value.keys())
        return False

    def _is_equal(value1, value2):
        """Check if two values are equal."""
        if isinstance(value1, list) and isinstance(value2, list):
            return value1.sort() == value2.sort()
        if isinstance(value1, tuple) or isinstance(value2, tuple):
            return list(value1) == list(value2)
        return value1 == value2

    # Extract keys from both dictionaries considering only supported value types
    keys1 = {key for key, value in config1.items() if _is_valid_value(value)}
    keys2 = {key for key, value in config2.items() if _is_valid_value(value)}

    # Check for identical sets of keys
    if keys1 != keys2:
        return False

    # Compare values for each key
    for key in keys1:
        value1 = config1[key]
        value2 = config2[key]

        # Check for list to handle potential unordered elements
        if not _is_equal(value1, value2):
            return False
    return True


def check_wandb_success(wandb_path):
    """
    Check if the wandb run was successful by checking the run state in the API.
    """
    import wandb

    api = wandb.Api()

    # run = api.run("your_entity/your_project_name/your_run_id")
    run = api.run(wandb_path)
    run_success = run.state == "finished" and run.summary.get("success", False)
    return run_success


class FakeWandb:
    """
    A fake wandb context manager that does nothing.
    """

    def __init__(self):
        self.config = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class GradNormCollector:
    """
    A class to collect the gradient norms of a model during training.
    """

    def __init__(self):
        self.grad_norms = defaultdict(list)
        self.param_norms = {}

    @torch.no_grad()
    def collect(self, model):
        """
        Collect the gradient norms of the model.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.grad_norms[name].append(param.grad.norm().item())

        # record param stats only once in the beginning
        if len(self.param_norms) == 0:
            for name, param in model.named_parameters():
                try:
                    param_norm = param.data.norm().item()
                except RuntimeError as e:
                    print(f"Error in calculating norm for {name} {param} {param.dtype}")
                    raise e
                self.param_norms[name] = param_norm
        return

    def __getitem__(self, key):
        return self.grad_norms[key]

    def items(self):
        """Iterate over the collected gradient norms."""
        return self.grad_norms.items()

    def save(self, path):
        """
        Save the collected gradient norms to a file.
        """
        out_dict = {"grad_norms": self.grad_norms, "param_norms": self.param_norms}
        joblib.dump(out_dict, path)

    @classmethod
    def load(cls, path):
        """
        Load the collected gradient norms from a file.
        """
        collector = cls()
        saved_data = joblib.load(path)
        collector.grad_norms = saved_data["grad_norms"]
        collector.param_norms = saved_data["param_norms"]
        return collector

    def reset(self):
        """
        Reset the collected gradient norms.
        """
        self.grad_norms = defaultdict(list)
        self.param_norms = {}
        return


def make_borzoi_scheduler(optimizer, warmup_steps=10000, total_steps=1000000):
    """
    Create a learning rate scheduler with warmup.
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-7,
        end_factor=1,
        total_iters=warmup_steps,
    )
    # this setup decreases lr to
    # ~10% in 40% of the (total-warmup) steps
    # ~1% in 60% of the (total-warmup) steps
    # ~0.1% in 75% of the (total-warmup) steps
    train_scheduler = PolynomialLR(
        optimizer,
        total_iters=total_steps - warmup_steps,
        power=5,
    )
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, train_scheduler], [warmup_steps]
    )
    return scheduler
