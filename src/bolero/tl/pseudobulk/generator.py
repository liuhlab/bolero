from collections import defaultdict
from typing import Generator

import numpy as np
import pandas as pd


class PseudobulkGenerator:
    """Generate pseudobulks from embedding data."""

    def __init__(
        self, embedding: pd.DataFrame, barcode_order: dict[str, pd.Index]
    ) -> None:
        """
        Initialize the PseudobulkGenerator.

        Parameters
        ----------
        embedding (pd.DataFrame): The embedding data.
        barcode_order (dict[str, pd.Index]): The barcode order dictionary.

        Returns
        -------
        None
        """
        self.embedding = embedding.astype("float32")
        self.cells = embedding.index
        self.n_cells, self.n_features = embedding.shape

        self._predefined_pseudobulks = None

        self.barcode_order = barcode_order

    def add_predefined_pseudobulks(self, pseudobulks: dict[str, pd.Index]) -> None:
        """
        Add predefined pseudobulks.

        Parameters
        ----------
        pseudobulks (dict[str, pd.Index]): The predefined pseudobulks.

        Returns
        -------
        None
        """
        if self._predefined_pseudobulks is None:
            self._predefined_pseudobulks = list(pseudobulks.values())
        else:
            self._predefined_pseudobulks.extend(pseudobulks.values())

    def get_pseudobulk_centriods(
        self, cells: pd.Index, method: str = "mean"
    ) -> np.ndarray:
        """
        Get the centroids of pseudobulks.

        Parameters
        ----------
        cells (pd.Index): The cells to calculate centroids for.
        method (str): The method to calculate centroids. Default is "mean".

        Returns
        -------
        np.ndarray: The centroids of pseudobulks.
        """
        cells = pd.Index(cells)
        if method == "mean":
            return self.embedding.loc[cells].mean(axis=0).values
        elif method == "median":
            return self.embedding.loc[cells].median(axis=0).values
        else:
            raise ValueError(f"Unknown method {method}")

    def take_predefined_pseudobulk(
        self, n: int
    ) -> Generator[tuple[dict[str, pd.Index], np.ndarray], None, None]:
        """
        Take predefined pseudobulks.

        Parameters
        ----------
        n (int): The number of pseudobulks to take.

        Yields
        ------
        Tuple[dict[str, pd.Index], np.ndarray]: A tuple of prefix to rows dictionary and pseudobulk centroids.
        """
        if self._predefined_pseudobulks is None:
            raise ValueError("No predefined pseudobulks")

        random_idx = np.random.choice(
            len(self._predefined_pseudobulks), size=n, replace=False
        )
        for idx in random_idx:
            cells = self._predefined_pseudobulks[idx]
            prefix_to_rows = self._cells_to_prefix_dict(cells)
            embeddings = self.get_pseudobulk_centriods(cells)
            yield cells, prefix_to_rows, embeddings

    def _cells_to_prefix_dict(self, cells: pd.Index) -> dict[str, pd.Index]:
        """
        Convert cells to prefix to rows dictionary.

        Parameters
        ----------
        cells (pd.Index): The cells to convert.

        Returns
        -------
        dict[str, pd.Index]: The prefix to rows dictionary.
        """
        prefix_to_cells = defaultdict(list)
        for cell in cells:
            prefix, barcode = cell.split(":")
            prefix_to_cells[prefix].append(barcode)

        prefix_to_rows = {}
        for prefix, cells in prefix_to_cells.items():
            try:
                barcode_orders = self.barcode_order[prefix]
                prefix_to_rows[prefix] = barcode_orders.isin(cells)
            except KeyError:
                continue
        return prefix_to_rows

    def take(
        self, n: int, mode: str = "predefined"
    ) -> Generator[tuple[dict[str, pd.Index], np.ndarray], None, None]:
        """
        Take pseudobulks.

        Parameters
        ----------
        n (int): The number of pseudobulks to take.
        mode (str): The mode to take pseudobulks. Default is "predefined".

        Yields
        ------
        Tuple[dict[str, pd.Index], np.ndarray]: A tuple of prefix to rows dictionary and pseudobulk centroids.
        """
        if mode == "predefined":
            return self.take_predefined_pseudobulk(n)
        else:
            raise NotImplementedError(f"Unknown mode {mode}")
