import math
import pathlib
from typing import Dict, Union

import numpy as np
import torch

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
        "chrX",
        "chrY",
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
        "chrY",
    ],
}
hg38_splits[2] = {
    "test": ["chr4", "chr11", "chr12", "chr15", "chrY"],
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
        "chrY",
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
        "chrY",
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
    "valid": ["chr5", "chr9", "chr13", "chr15", "chrY"],
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
    "valid": ["chr2", "chr9", "chr11", "chr12", "chrY"],
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
    "valid": ["chr1", "chr7", "chr12", "chr13", "chrY"],
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
    "test": ["chr5", "chr10", "chr12", "chr16", "chrY"],
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


def get_splits(genome: str, split_id: int) -> Dict[str, Union[list, None]]:
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
        try:
            self.total += float(np.nansum(value))
        except TypeError:
            # torch
            self.total += float(torch.nansum(value).detach().cpu().item())
        # both numpy and torch will work
        self.count += np.prod(value.shape)

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
        assert nx == ny, "Length mismatch between x and y"
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
        return False

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
        if isinstance(value1, list) and isinstance(value2, list):
            if value1 != value2:
                return False
        elif value1 != value2:
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
