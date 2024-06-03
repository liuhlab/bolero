"""
Filter functions for ray.data.Dataset objects.

Each filter is a function that dynamically creates a filter function for filtering rows in a Dataset object.
Aim to be used in ray.data.Dataset.filter() method.

The filter function takes a data dictionary and returns a boolean value.
"""

# TODO: change filter to map_batches


class RowSumFilter:
    """Filter rows based on the sum of a column in the data dictionary.

    Args:
        key (str): The key of the column to calculate the sum.
        min_sum (float): The minimum sum value for filtering.
        max_sum (float): The maximum sum value for filtering.

    Returns
    -------
        bool: True if the sum is within the specified range, False otherwise.
    """

    def __init__(self, key: str, min_sum: float, max_sum: float):
        self.key = key
        self.min_sum = min_sum
        self.max_sum = max_sum

    def __call__(self, data: dict) -> bool:
        """Filter rows based on the sum of a column in the data dictionary.

        Args:
            data (dict): The data dictionary containing the column.

        Returns
        -------
            bool: True if the sum is within the specified range, False otherwise.
        """
        _sum = data[self.key].sum()
        return (_sum > self.min_sum) & (_sum < self.max_sum)
