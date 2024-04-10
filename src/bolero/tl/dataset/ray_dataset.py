import pathlib
from typing import List, Union

import ray
from pyarrow.fs import FileSystem


class RayGenomeDataset:
    """RayDataset class for working with ray.data.Dataset objects."""

    def __init__(self, dataset: ray.data.Dataset) -> None:
        """
        Initialize a RayDataset object.

        Parameters
        ----------
            dataset (ray.data.Dataset): The Ray dataset.

        Returns
        -------
            None
        """
        self.dataset = dataset

    @classmethod
    def read_parquet(
        cls, path: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]], **kwargs
    ) -> "RayGenomeDataset":
        """
        Read a Parquet file into a RayDataset object.

        Parameters
        ----------
            path (Union[str, pathlib.Path, List[Union[str, pathlib.Path]]]): The path(s) to the Parquet file(s).
            **kwargs: Additional keyword arguments to pass to the `ray.data.read_parquet` function.

        Returns
        -------
            RayDataset: The RayDataset object containing the data from the Parquet file(s).
        """
        if isinstance(path, (str, pathlib.Path)):
            paths = [str(path)]
        else:
            paths = [str(p) for p in path]

        fs, _ = FileSystem.from_uri(paths[0])
        _ds = ray.data.read_parquet(path, filesystem=fs, **kwargs)
        return cls(_ds)
