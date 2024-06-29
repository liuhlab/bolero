"""
PredefinedPseudobulkGenerator
-----------------------------
Prepare:
1. embedding data: prefix:cell_id as index
2. cell coverage data: prefix:cell_id as index
3. barcode order: prefix to cell_id index without prefix
4. predefined pseudobulk data: dict of pseudobulk name to prefix:cell_id index

Rule is that whenever data is not separated by prefix, it should be prefix:cell_id as index

"""

import pathlib
from typing import Generator, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler, StandardScaler

from bolero.tl.model.generic.train_helper import validate_config


class PseudobulkGenerator:
    """Base class for pseudobulk generator."""

    default_config = {}

    standard_cov: int
    standard_cell: int
    cell_coverage: Union[int, pd.Series]
    barcode_order: dict[str, pd.Index]

    @classmethod
    def get_default_config(cls):
        """Get the default configuration."""
        return cls.default_config

    @classmethod
    def create_from_config(cls, **config):
        """Create the pseudobulk generator from configuration."""
        config = {k: v for k, v in config.items() if k in cls.default_config}
        validate_config(config, cls.default_config)
        return cls(**config)

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
        # assume prefix:cell_id only have one ":"
        cell_split = cells.str.split(":")
        prefix_to_cells: dict[list] = cell_split.str[1].groupby(cell_split.str[0])

        prefix_to_rows = {}
        found_cells = 0
        for prefix, cells in prefix_to_cells.items():
            try:
                barcode_orders = self.barcode_order[prefix]
                bool_index = barcode_orders.isin(cells)
                found_cells += bool_index.sum()
                prefix_to_rows[prefix] = bool_index
            except KeyError:
                continue

        # check if all cells are in the dataset
        if found_cells != len(cells):
            print(
                f"Cell number doesn't match between pseudobulk and dataset! "
                f"Pseudobulk size: {len(cells)}, Found cells: {found_cells}."
            )
        return prefix_to_rows

    def _select_cells(self, cells: pd.Index, replace=False) -> pd.Index:
        # Try replace == True to mimic bootstrap sampling in base or lora training
        if self.standard_cov:
            if isinstance(self.cell_coverage, int):
                # cell coverage is a constant number
                n_cells = self.standard_cov // self.cell_coverage
                cells = pd.Index(np.random.choice(cells, n_cells, replace=replace))
            else:
                # select random cells to reach the standard coverage
                random_cumsum = (
                    self.cell_coverage.loc[cells]
                    .sample(cells.size, replace=replace)
                    .cumsum()
                )
                cells = random_cumsum[random_cumsum < self.standard_cov].index
        else:
            # select random cells to reach the standard cell number
            cells = pd.Index(
                np.random.choice(cells, self.standard_cell, replace=replace)
            )
        return cells


class SinglePseudobulkGenerator(PseudobulkGenerator):
    """Generate single pseudobulk with predefined cell ids.

    Parameters
    ----------
    cells : list or array-like
        The predefined cell ids for generating the pseudobulk.
    cell_coverage : pd.Series, optional
        The coverage of each cell. Required when `standard_cell` or `standard_cov` is set.
    standard_cov : float, optional
        The standard coverage value for sampling cells.
    standard_cell : str, optional
        The standard cell id for sampling cells.

    Attributes
    ----------
    cells : pd.Index
        The predefined cell ids.
    cell_coverage : pd.Series
        The coverage of each cell.
    """

    default_config = {
        "cells": "REQUIRED",
        "cell_coverage": None,
        "barcode_order": "REQUIRED",
    }

    def __init__(
        self,
        cells,
        barcode_order,
        cell_coverage=None,
        standard_cov=None,
        standard_cell=None,
    ):
        self.cells = pd.Index(cells)
        assert self.cells.duplicated().sum() == 0, "Duplicated cell ids."

        self.barcode_order = barcode_order

        if standard_cell is not None or standard_cov is not None:
            self._sample = True
            assert (
                cell_coverage is not None
            ), "Cell coverage is required when standard_cell or standard_cov is set."
        else:
            self._sample = False
        self.cell_coverage: Union[int, pd.Series] = cell_coverage

    def take(self, *args, **kwargs):
        """Generate the pseudobulk.

        Returns
        -------
        cells : pd.Index
            The selected cells for the pseudobulk.
        prefix_to_rows : dict
            A dictionary mapping prefix to rows for the selected cells.
        """
        if self._sample:
            cells = self._select_cells(self.cells)
        else:
            cells = self.cells

        prefix_to_rows = self._cells_to_prefix_dict(cells)

        # cells, prefix_to_rows, embeddings, idx
        records = [(cells, prefix_to_rows, 0, 0)]
        return records


class PredefinedPseudobulkGenerator(PseudobulkGenerator):
    """Generate pseudobulks from embedding data."""

    default_config = {
        "cell_embedding": "REQUIRED",
        "cell_coverage": "REQUIRED",
        # although barcode_order is required,
        # user don't need to provide it, RayGenomeChunkDataset will provide it
        # the predefined pseudobulk cell id needs to be consistent with the barcode_order's cell id
        "barcode_order": "REQUIRED",
        "predefined_pseudobulk_path": None,
        "standard_cov": 10e6,
        "standard_cell": None,
    }

    @classmethod
    def create_from_config(
        cls,
        cell_embedding,
        cell_coverage,
        barcode_order,
        predefined_pseudobulk_path=None,
        standard_cov: int = 1e7,
        standard_cell: int = None,
    ) -> "PredefinedPseudobulkGenerator":
        """
        Prepare the pseudobulker.

        Parameters
        ----------
        cell_embedding : Union[str, pathlib.Path, pd.DataFrame]
            The cell embedding data, cell id should contain prefix and unique.
        cell_coverage : Union[str, pathlib.Path, pd.Series]
            The cell coverage data. Index should be cell id.
        barcode_order : dict[str, pd.Index]
            The barcode order dictionary. Key is the prefix, value is the barcode index without prefix.
            This dict is part of the ray dataset, stored at "dataset_dir/row"
        predefined_pseudobulk : Optional[dict], optional
            Predefined pseudobulk data, by default None.
        standard_cov : int, optional
            The standard total pseudobulk coverage, by default 1e7.
            Pseudobulk cells will be randowmly sampled to reach this coverage.
            If a predefined pseudobulk's total coverage is bellow this value,
            it will be discarded when adding predefined pseudobulks.
            Only one of standard_cov and standard_cell can be set.
        standard_cell : int, optional
            The standard total pseudobulk cell number, by default None.
            Pseudobulk cells will be randowmly sampled to reach this cell number.
            If a predefined pseudobulk's total cell number is bellow this value,
            it will be discarded when adding predefined pseudobulks.
            Only one of standard_cov and standard_cell can be set.

        Returns
        -------
        None
        """
        if isinstance(cell_embedding, (str, pathlib.Path)):
            _embedding = pd.read_feather(cell_embedding)
            _embedding = _embedding.set_index(_embedding.columns[0])
        elif isinstance(cell_embedding, pd.DataFrame):
            _embedding = cell_embedding.copy()

        if isinstance(cell_coverage, (str, pathlib.Path)):
            cell_coverage = pd.read_feather(cell_coverage)
            cell_coverage = cell_coverage.set_index(cell_coverage.columns[0]).squeeze()
            cell_coverage = cell_coverage.reindex(_embedding.index)
        else:
            if isinstance(cell_coverage, pd.Series):
                cell_coverage = cell_coverage.reindex(_embedding.index)
                # check coverage nan
                assert (
                    not cell_coverage.isna().values.any()
                ), "coverage contains nan values."

        # check cell id format
        cells = _embedding.index
        assert (
            cells.str.count(":").max() == 1
        ), "cell id should be prefix:cell_id, no more ':' in prefix or cell_id is allowed."
        # check embedding nan
        assert not _embedding.isna().values.any(), "embedding contains nan values."

        pseudobulker = cls(
            embedding=_embedding,
            barcode_order=barcode_order,
            cell_coverage=cell_coverage,
            standard_cov=standard_cov,
            standard_cell=standard_cell,
        )
        if predefined_pseudobulk_path is not None:
            if isinstance(predefined_pseudobulk_path, (str, pathlib.Path)):
                predefined_pseudobulk_path = [predefined_pseudobulk_path]
            for path in predefined_pseudobulk_path:
                _d = joblib.load(path)
                pseudobulker.add_predefined_pseudobulks(_d)

        if len(pseudobulker.predefined_pseudobulks) > 0:
            pseudobulker.prepare_scaler()
        return pseudobulker

    def __init__(
        self,
        embedding: pd.DataFrame,
        barcode_order: dict[str, pd.Index],
        cell_coverage: Union[int, pd.Series],
        standard_cov: int = 1e7,
        standard_cell: int = None,
    ) -> None:
        """
        Initialize the pseudobulk generator.

        Parameters
        ----------
        embedding (pd.DataFrame): The embedding data.
        barcode_order (dict[str, pd.Index]): The barcode order dictionary.
        cell_coverage (pd.Series): The cell coverage.
        standard_cov (int): The standard total pseudobulk coverage. Default is 1e7.
        standard_cell (int): The standard total pseudobulk cell number. Default is None.

        Returns
        -------
        None
        """
        self.embedding = embedding.astype("float32")
        self.cells = embedding.index
        self.n_cells, self.n_features = embedding.shape
        self.cell_coverage = cell_coverage

        self.predefined_pseudobulks = None
        self.predefined_pseudobulks_names = None
        self.standard_cov = standard_cov
        self.standard_cell = standard_cell
        assert not (
            standard_cov and standard_cell
        ), "Only one of standard_cov and standard_cell can be set."
        self.barcode_order = barcode_order
        self.scaler = EmbeddingScaler()

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

            if self.standard_cov:
                # cov mode
                if isinstance(self.cell_coverage, int):
                    # cell coverage is a constant number
                    total_coverage = self.cell_coverage * len(cells)
                else:
                    total_coverage = self.cell_coverage.loc[cells.values].sum()
                if total_coverage >= self.standard_cov:
                    use_pseudobulks[k] = cells
            else:
                # cell mode
                if len(cells) >= self.standard_cell:
                    use_pseudobulks[k] = cells

        print(
            f"{len(use_pseudobulks)} predefined pseudobulks are used, "
            f"standard pseudobulk coverage is {self.standard_cov}, "
            f"standard cell is {self.standard_cell}."
        )

        pseudobulk_list = []
        pseudobulk_names = []
        for k, cells in use_pseudobulks.items():
            pseudobulk_list.append(cells)
            pseudobulk_names.append(k)

        if self.predefined_pseudobulks is None:
            self.predefined_pseudobulks = pseudobulk_list
            self.predefined_pseudobulks_names = pseudobulk_names
        else:
            self.predefined_pseudobulks.extend(pseudobulk_list)
            self.predefined_pseudobulks_names.extend(pseudobulk_names)
        return

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
            embedding = self.embedding.loc[cells].mean(axis=0).values
        elif method == "median":
            embedding = self.embedding.loc[cells].median(axis=0).values
        else:
            raise ValueError(f"Unknown method {method}")

        # if cell mode, add the pseudobulk coverage log1p to the end
        if not self.standard_cov:
            embedding = np.append(embedding, self.get_pseudobulk_coverage(cells))

        try:
            # Normalize the embedding
            embedding = self.scaler.transform(embedding)
        except NotFittedError:
            pass
        return embedding

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
        if isinstance(self.cell_coverage, int):
            return np.log10(self.cell_coverage * len(cells) + 1)
        else:
            return np.log10(self.cell_coverage.loc[cells].sum() + 1)

    def take_predefined_pseudobulk(
        self,
    ) -> Generator[tuple[dict[str, pd.Index], np.ndarray], None, None]:
        """
        Take one predefined pseudobulk.

        Yields
        ------
        Tuple[dict[str, pd.Index], np.ndarray]: A tuple of prefix to rows dictionary and pseudobulk centroids.
        """
        if self.predefined_pseudobulks is None:
            raise ValueError("No predefined pseudobulks")

        n_defined = len(self.predefined_pseudobulks)

        idx = np.random.choice(n_defined)
        cells = pd.Index(self.predefined_pseudobulks[idx])
        cells = self._select_cells(cells)

        prefix_to_rows = self._cells_to_prefix_dict(cells)
        embeddings = self.get_pseudobulk_centriods(cells)
        return cells, prefix_to_rows, embeddings, idx

    def take(
        self,
        n: int,
        mode: str = "predefined",
    ) -> tuple[dict[str, pd.Index], np.ndarray]:
        """
        Take pseudobulks.

        Parameters
        ----------
        n (int): The number of pseudobulks to take.
        mode (str): The mode to take pseudobulks. Default is "predefined".

        Yields
        ------
        Tuple[pd.Index, dict[pd.Index], np.ndarray, int]: A tuple of four objects,
        containing cell index, prefix_to_rows, embeddings, pseudobulk idx
        """
        records = []
        n = min(n, len(self.predefined_pseudobulks))
        for _ in range(n):
            if mode == "predefined":
                records.append(self.take_predefined_pseudobulk())
            else:
                raise NotImplementedError(f"Unknown mode {mode}")
        return records

    def _pseudobulk_id_to_name(self, idx):
        if isinstance(idx, int):
            idx = np.array([idx])
        return [self.predefined_pseudobulks_names[i] for i in idx]

    def prepare_scaler(self):
        """
        Fit the scaler using predefined pseudobulks.
        """
        rows = []
        for cells in self.predefined_pseudobulks:
            embedding = self.embedding.loc[cells.values].mean()

            if not self.standard_cov:
                # for standard_cell mode, we need to select the correct cell number
                # in order to make the coverage embedding dim consistent
                #
                # add the coverage log1p to the end of the embedding
                _cells = np.random.choice(cells, self.standard_cell, replace=False)
                embedding = np.append(embedding, self.get_pseudobulk_coverage(_cells))

            rows.append(embedding)
        example_embedding = pd.DataFrame(rows, index=self.predefined_pseudobulks_names)
        self.scaler.fit(example_embedding)
        return

    def save_scaler(self, path):
        """
        Save the scaler to path.

        Parameters
        ----------
        path (str): The path to save the scaler.
        """
        if not self.scaler.fitted:
            raise NotFittedError("Scaler is not fitted yet.")

        joblib.dump(self.scaler, path)
        return

    def save(self, path):
        """
        Save the pseudobulk generator to path.

        Parameters
        ----------
        path (str): The path to save the pseudobulk generator.
        """
        joblib.dump(self, path)
        return


class EmbeddingScaler:
    """
    A class for scaling embeddings and
    also save the fitting data as (scaled) example embedding.

    Attributes
    ----------
    scaler1 : RobustScaler
        The first scaler for robust scaling.
    scaler2 : StandardScaler
        The second scaler for standard scaling.
    fitted : bool
        Indicates whether the scaler has been fitted.
    _example_embedding : pd.DataFrame or None
        An example embedding used for scaling.

    Methods
    -------
    example_embedding()
        Get the example embedding.
    fit(embedding)
        Fit the scaler using the given embedding.
    transform(embedding)
        Transform the given embedding using the fitted scaler.
    """

    def __init__(self):
        self.scaler1 = RobustScaler(quantile_range=(5, 95))
        self.scaler2 = StandardScaler()
        self.fitted = False
        self._example_embedding = None

    @property
    def example_embedding(self) -> pd.DataFrame:
        """
        Get the example embedding.

        Returns
        -------
        pd.DataFrame
            The example embedding.
        """
        if not self.fitted:
            raise NotFittedError
        return self._example_embedding

    def fit(self, embedding: Union[pd.DataFrame, np.ndarray]) -> "EmbeddingScaler":
        """
        Fit the scaler using the given embedding.

        Parameters
        ----------
        embedding : pd.DataFrame or np.ndarray
            The embedding to fit the scaler.

        Returns
        -------
        EmbeddingScaler
            The fitted scaler.
        """
        if isinstance(embedding, pd.DataFrame):
            index = embedding.index
            columns = embedding.columns
            embedding = embedding.values
        else:
            index = None
            columns = None

        if self.fitted:
            print(
                "Warning: this EmbeddingScaler has already been fitted, "
                "its state will be overwritten due to re-fit."
            )
        embedding = np.clip(self.scaler1.fit_transform(embedding), -1, 1)
        embedding = self.scaler2.fit_transform(embedding.reshape((-1, 1))).reshape(
            embedding.shape
        )
        self.fitted = True

        if index is not None:
            embedding = pd.DataFrame(embedding, index=index, columns=columns)
        self._example_embedding = embedding
        return self

    def transform(
        self, embedding: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform the given embedding using the fitted scaler.

        Parameters
        ----------
        embedding : pd.DataFrame, pd.Series, or np.ndarray
            The embedding to transform.

        Returns
        -------
        Union[pd.DataFrame, pd.Series]
            The transformed embedding.
        """
        if isinstance(embedding, pd.DataFrame):
            index = embedding.index
            columns = embedding.columns
            embedding = embedding.values
        if isinstance(embedding, pd.Series):
            index = embedding.index
            embedding = embedding.values
            columns = None
        else:
            index = None
            columns = None

        reshape = len(embedding.shape) == 1
        if reshape:
            embedding = embedding.reshape((1, -1))
        embedding = np.clip(self.scaler1.transform(embedding), -1, 1)
        embedding = self.scaler2.transform(embedding.reshape((-1, 1))).reshape(
            embedding.shape
        )
        if reshape:
            embedding = embedding[0]

        if index is not None and columns is not None:
            embedding = pd.DataFrame(embedding, index=index, columns=columns)
        elif index is not None:
            embedding = pd.Series(embedding, index=index)

        return embedding
