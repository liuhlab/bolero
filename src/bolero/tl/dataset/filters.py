"""
Filter functions for ray.data.Dataset objects.

Each filter is a function that dynamically creates a filter function for filtering rows in a Dataset object.
Aim to be used in ray.data.Dataset.filter() method.

The filter function takes a data dictionary and returns a boolean value.
"""


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


class MinMaxFilter:
    """Filter rows based on the min and max values of a column in the data dictionary.

    Args:
        key (str): The key of the column to calculate the min and max values.
        min_val (float): The minimum value for filtering.
        max_val (float): The maximum value for filtering.

    Returns
    -------
        bool: True if the min and max values are within the specified range, False otherwise.
    """

    def __init__(self, key: str, min_val: float, max_val: float):
        self.key = key
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data: dict) -> bool:
        """Filter rows based on the min and max values of a column in the data dictionary.

        Args:
            data (dict): The data dictionary containing the column.

        Returns
        -------
            bool: True if the min and max values are within the specified range, False otherwise.
        """
        _min = data[self.key].min()
        _max = data[self.key].max()
        return (_min > self.min_val) & (_max < self.max_val)
