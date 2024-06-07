import pathlib
from typing import Generator

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler, StandardScaler

from bolero.tl.model.generic.train_helper import validate_config


class PseudobulkGenerator:
    """Base class for pseudobulk generator."""

    default_config = {}

    @classmethod
    def get_default_config(cls):
        """Get the default configuration."""
        return cls.default_config

    @classmethod
    def create_from_config(cls, config):
        """Create the pseudobulk generator from configuration."""
        config = {k: v for k, v in config.items() if v in cls.default_config}
        validate_config(config, cls.default_config)
        return cls(**config)


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
        standard_cov: int = 10e6,
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
            The standard total pseudobulk coverage, by default 10e6.
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
        else:
            cell_coverage = cell_coverage.copy()

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
            for i, path in enumerate(predefined_pseudobulk_path):
                _d = {f"{k}_{i}": v for k, v in joblib.load(path).items()}
                pseudobulker.add_predefined_pseudobulks(_d)

        if len(pseudobulker._predefined_pseudobulks) > 0:
            pseudobulker.prepare_scaler()
        return pseudobulker

    def __init__(
        self,
        embedding: pd.DataFrame,
        barcode_order: dict[str, pd.Index],
        cell_coverage: pd.Series,
        standard_cov: int = 10e6,
        standard_cell: int = None,
    ) -> None:
        """
        Initialize the pseudobulk generator.

        Parameters
        ----------
        embedding (pd.DataFrame): The embedding data.
        barcode_order (dict[str, pd.Index]): The barcode order dictionary.
        cell_coverage (pd.Series): The cell coverage.
        standard_cov (int): The standard total pseudobulk coverage. Default is 10e6.
        standard_cell (int): The standard total pseudobulk cell number. Default is None.

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
        self.standard_cov = standard_cov
        self.standard_cell = standard_cell
        assert not (
            standard_cov and standard_cell
        ), "Only one of standard_cov and standard_cell can be set."
        self.barcode_order = barcode_order

        self.scaler1 = RobustScaler(quantile_range=(5, 95))
        self.scaler2 = StandardScaler()
        self.example_embedding = None

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
                total_coverage = self.cell_coverage.loc[cells.values].sum()
                if total_coverage >= self.standard_cov:
                    use_pseudobulks[k] = cells
            else:
                # cell mode
                if len(cells) >= self.standard_cell:
                    use_pseudobulks[k] = cells

        print(
            f"{len(use_pseudobulks)} predefined pseudobulks are used, "
            f"standard pseudobulk coverage is {int(self.standard_cov)}, "
            f"standard cell is {self.standard_cell}."
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
            embedding = self._scaler(embedding)
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
        if self._predefined_pseudobulks is None:
            raise ValueError("No predefined pseudobulks")

        n_defined = len(self._predefined_pseudobulks)

        idx = np.random.choice(n_defined)
        cells = pd.Index(self._predefined_pseudobulks[idx])

        if self.standard_cov:
            # select random cells to reach the standard coverage
            random_cumsum = (
                self.cell_coverage.loc[cells].sample(cells.size, replace=False).cumsum()
            )
            cells = random_cumsum[random_cumsum < self.standard_cov].index
        else:
            # select random cells to reach the standard cell number
            cells = pd.Index(np.random.choice(cells, self.standard_cell, replace=False))

        prefix_to_rows = self._cells_to_prefix_dict(cells)
        embeddings = self.get_pseudobulk_centriods(cells)
        return cells, prefix_to_rows, embeddings, idx

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
        prefix_to_cells_series = pd.Series(
            cells.map(lambda x: x.split(":")[0], index=cells)
        )
        prefix_to_cells = {
            k: v.index
            for k, v in prefix_to_cells_series.groupby(prefix_to_cells_series)
        }

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
                f"Not all cells are in the dataset! Pseudobulk size: {len(cells)}, Found cells: {found_cells}."
            )

        return prefix_to_rows

    def take(
        self, n: int, mode: str = "predefined"
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
        for _ in range(n):
            if mode == "predefined":
                records.append(self.take_predefined_pseudobulk())
            else:
                raise NotImplementedError(f"Unknown mode {mode}")
        return records

    def _pseudobulk_id_to_name(self, idx):
        if isinstance(idx, int):
            idx = np.array([idx])
        return [self._predefined_pseudobulks_names[i] for i in idx]

    def prepare_scaler(self):
        """
        Fit the scaler using predefined pseudobulks.
        """
        col = []
        for cells in self._predefined_pseudobulks:
            embedding = self.embedding.loc[cells.values].mean()

            if not self.standard_cov:
                # add the coverage log1p to the end of the embedding
                _cells = np.random.choice(cells, self.standard_cell, replace=False)
                embedding = np.append(embedding, self.get_pseudobulk_coverage(_cells))

            col.append(embedding)
        embedding = np.array(col)
        embedding = np.clip(self.scaler1.fit_transform(embedding), -1, 1)
        embedding = self.scaler2.fit_transform(embedding.reshape((-1, 1))).reshape(
            embedding.shape
        )

        self.example_embedding = embedding
        return

    def _scaler(self, embedding):
        reshape = len(embedding.shape) == 1
        if reshape:
            embedding = embedding.reshape((1, -1))
        embedding = np.clip(self.scaler1.transform(embedding), -1, 1)
        embedding = self.scaler2.transform(embedding.reshape((-1, 1))).reshape(
            embedding.shape
        )
        if reshape:
            embedding = embedding[0]
        return embedding
