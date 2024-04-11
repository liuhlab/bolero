"""
Filter functions for ray.data.Dataset objects.

Each filter is a function that dynamically creates a filter function for filtering rows in a Dataset object.
Aim to be used in ray.data.Dataset.filter() method.

The filter function takes a data dictionary and returns a boolean value.
"""


def sum_filter(key, min_sum, max_sum):
    """Filter rows based on the sum of the values in the specified column."""

    def _cov_filter(data):
        _cov = data[key].sum()
        return (_cov > min_sum) & (_cov < max_sum)

    return _cov_filter


def min_max_filter(key, min_val, max_val):
    """Filter rows based on the minimum and maximum values in the specified column."""

    def _min_max_filter(data):
        _min = data[key].min()
        _max = data[key].max()
        return (_min > min_val) & (_max < max_val)

    return _min_max_filter
