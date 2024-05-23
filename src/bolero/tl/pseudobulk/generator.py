from collections import defaultdict
from typing import Generator

import numpy as np
import pandas as pd


class PseudobulkGenerator:
    """Generate pseudobulks from embedding data."""

    def __init__(
        self,
        embedding: pd.DataFrame,
        barcode_order: dict[str, pd.Index],
        cell_coverage: pd.Series,
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
        self.cell_coverage = cell_coverage

        self._predefined_pseudobulks = None

        self.barcode_order = barcode_order

    def add_predefined_pseudobulks(
        self, pseudobulks: dict[str, pd.Index], standard_cells=2500
    ) -> None:
        """
        Add predefined pseudobulks.

        Parameters
        ----------
        pseudobulks (dict[str, pd.Index]): The predefined pseudobulks.

        Returns
        -------
        None
        """
        use_pseudobulks = {}
        for k, cells in pseudobulks.items():
            cells = pd.Series(list(cells))
            if standard_cells is not None:
                if cells.size >= standard_cells:
                    cells = cells.sample(standard_cells, random_state=0)
                    use_pseudobulks[k] = cells
                else:
                    continue
            else:
                use_pseudobulks[k] = cells

        print(
            f"{len(use_pseudobulks)} predefined pseudobulks are used, standard cell number is {standard_cells}."
        )

        if self._predefined_pseudobulks is None:
            self._predefined_pseudobulks = list(use_pseudobulks.values())
        else:
            self._predefined_pseudobulks.extend(use_pseudobulks.values())

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

    def get_pseudobulk_coverage(self, cells: pd.Index) -> float:
        """
        Get the coverage of pseudobulks.

        Parameters
        ----------
        cells (pd.Index): The cells to calculate coverage for.

        Returns
        -------
        float: The coverage of pseudobulks.
        """
        return np.log10(self.cell_coverage.loc[cells].sum() + 1)

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

        n_defined = len(self._predefined_pseudobulks)
        actual_n = min(n, n_defined)
        random_idx = np.random.choice(n_defined, size=actual_n, replace=False)
        for idx in random_idx:
            cells = self._predefined_pseudobulks[idx]
            prefix_to_rows = self._cells_to_prefix_dict(cells)
            embeddings = self.get_pseudobulk_centriods(cells)
            coverage = self.get_pseudobulk_coverage(cells)
            embeddings = np.concatenate([embeddings, [coverage]])
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
