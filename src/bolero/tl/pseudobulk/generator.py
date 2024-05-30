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
        standard_cells: int = 2500,
    ) -> None:
        """
        Initialize the PseudobulkGenerator.

        Parameters
        ----------
        embedding (pd.DataFrame): The embedding data.
        barcode_order (dict[str, pd.Index]): The barcode order dictionary.
        cell_coverage (pd.Series): The cell coverage.
        standard_cells (int): The standard cell number. Default is 2500.

        Returns
        -------
        None
        """
        self.embedding = embedding.astype("float32")
        self.cells = embedding.index
        self.n_cells, self.n_features = embedding.shape
        self.cell_coverage = cell_coverage

        self._predefined_pseudobulks = None
        self._predefined_pseudobulks_names = None
        self.standard_cells = standard_cells
        # TODO: use standard_total_reads instead of standard_cells

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
        use_pseudobulks = {}
        for k, cells in pseudobulks.items():
            cells = pd.Series(list(cells))
            if cells.size >= self.standard_cells:
                use_pseudobulks[k] = cells

        print(
            f"{len(use_pseudobulks)} predefined pseudobulks are used, standard cell number is {self.standard_cells}."
        )

        pseudobulk_list = []
        pseudobulk_names = []
        for k, cells in use_pseudobulks.items():
            pseudobulk_list.append(cells)
            pseudobulk_names.append(k)

        if self._predefined_pseudobulks is None:
            self._predefined_pseudobulks = pseudobulk_list
            self._predefined_pseudobulks_names = pseudobulk_names
        else:
            self._predefined_pseudobulks.extend(pseudobulk_list)
            self._predefined_pseudobulks_names.extend(pseudobulk_names)

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
            
            if cells.size > self.standard_cells:
                cells = pd.Index(np.random.choice(cells, size=self.standard_cells, replace=False))

            prefix_to_rows = self._cells_to_prefix_dict(cells)
            embeddings = self.get_pseudobulk_centriods(cells)
            coverage = self.get_pseudobulk_coverage(cells)
            embeddings = np.concatenate([embeddings, [coverage]])
            yield cells, prefix_to_rows, embeddings, idx

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

    def pseudobulk_id_to_name(self, idx):
        if isinstance(idx, int):
            idx = np.array([idx])
        return [self._predefined_pseudobulks_names[i] for i in idx]
