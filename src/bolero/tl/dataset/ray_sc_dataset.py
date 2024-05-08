# TODO: scRayDataset inherits from RayDataset, and then in scprinter, scPrinterscDataset inherits from scPrinterDataset.
# Once the sc dataset processed pseudobulk and provides region and pseudobulk data dict, the remaining preprocess step should be the same as bulk train model.
import pathlib
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import ray

from bolero import Genome
from bolero.tl.dataset.sc_transforms import scMetaRegionToBulkRegion
from bolero.tl.dataset.transforms import FetchRegionOneHot
from bolero.tl.pseudobulk.generator import PseudobulkGenerator


class RaySingleCellDataset:
    """Single cell dataset for cell-by-meta-region data."""

    def __init__(
        self, dataset_path: str, use_prefixs: Optional[list[str]] = None
    ) -> None:
        """
        Initialize the RaySingleCellDataset.

        Parameters
        ----------
        dataset_path : str
            The path to the dataset.
        use_prefixs : Optional[List[str]], optional
            The list of prefixes to use, by default None.

        Returns
        -------
        None
        """
        self._dataset = ray.data.read_parquet(
            dataset_path, file_extensions=["parquet"], shuffle="file"
        )
        _schema = self._dataset.schema()
        self.schema: dict = dict(zip(_schema.names, _schema.types))

        # get prefix
        self.prefixs = list({key.split(":")[0] for key in self.schema.keys()})
        if use_prefixs is not None:
            self.prefixs = [prefix for prefix in self.prefixs if prefix in use_prefixs]

        # get barcode order for each prefix
        self.barcode_order = {
            name: pd.Index(cells)
            for name, cells in np.load(f"{dataset_path}/barcodes.npz").items()
            if name in self.prefixs
        }
        self.pseudobulker = None

        # get genome
        with open(f"{dataset_path}/genome.flag") as f:
            genome = f.read().strip()
        self.genome = Genome(genome)
        # trigger one hot loading
        _ = self.genome.genome_one_hot

        self._dataset_mode = None

    def __repr__(self) -> str:
        """
        Return a string representation of the dataset.

        Returns
        -------
        str
            The string representation of the dataset.
        """
        return self._dataset.__repr__()

    def prepare_pseudobulker(
        self,
        embedding: Union[str, pathlib.Path, pd.DataFrame],
        predefined_pseudobulk: Optional[dict] = None,
    ) -> None:
        """
        Prepare the pseudobulker.

        Parameters
        ----------
        embedding : Union[str, pathlib.Path, pd.DataFrame]
            The embedding data.
        predefined_pseudobulk : Optional[dict], optional
            Predefined pseudobulk data, by default None.

        Returns
        -------
        None
        """
        if isinstance(embedding, (str, pathlib.Path)):
            _embedding = pd.read_feather(embedding)
            _embedding = _embedding.set_index(_embedding.columns[0])
        elif isinstance(embedding, pd.DataFrame):
            _embedding = embedding

        pseudobulker = PseudobulkGenerator(
            embedding=_embedding, barcode_order=self.barcode_order
        )
        if predefined_pseudobulk:
            pseudobulker.add_predefined_pseudobulks(predefined_pseudobulk)
        self.pseudobulker = pseudobulker
        return

    def _dataset_preprocess(
        self,
        sample_regions: int,
        n_pseudobulks: int,
        min_cov: int,
        max_cov: int,
        low_cov_ratio: float,
    ) -> None:
        """
        Preprocess the dataset.

        Parameters
        ----------
        sample_regions : int
            The number of sample regions.
        n_pseudobulks : int
            The number of pseudobulks.
        min_cov : int
            The minimum coverage.
        max_cov : int
            The maximum coverage.
        low_cov_ratio : float
            The low coverage ratio.

        Returns
        -------
        None
        """
        self._pseudobulk_and_extract_regions(
            sample_regions=sample_regions,
            n_pseudobulks=n_pseudobulks,
            min_cov=min_cov,
            max_cov=max_cov,
            low_cov_ratio=low_cov_ratio,
        )
        self._fetch_dna_one_hot()
        return

    def _pseudobulk_and_extract_regions(
        self,
        sample_regions: int,
        n_pseudobulks: int,
        min_cov: int,
        max_cov: int,
        low_cov_ratio: float,
    ) -> None:
        """
        Perform pseudobulking and extract regions.

        Parameters
        ----------
        sample_regions : int
            The number of sample regions.
        n_pseudobulks : int
            The number of pseudobulks.
        min_cov : int
            The minimum coverage.
        max_cov : int
            The maximum coverage.
        low_cov_ratio : float
            The low coverage ratio.

        Returns
        -------
        None
        """
        if self.pseudobulker is None:
            raise ValueError(
                "Pseudobulker not prepared yet, call self.prepare_pseudobulker() first."
            )

        # merge cell into pseudobulk and
        # split large meta region (storage) into smaller final regions (data consumption)
        processor = scMetaRegionToBulkRegion(
            prefixs=self.prefixs,
            pseudobulker=self.pseudobulker,
            sample_regions=sample_regions,
            min_cov=min_cov,
            max_cov=max_cov,
            low_cov_ratio=low_cov_ratio,
            n_pseudobulks=n_pseudobulks,
        )
        self._working_dataset = self._working_dataset.flat_map(processor)
        # after flat_map processor, each row in working_dataset is a dict with keys:
        # ["bulk_embedding", "bulk_data", "region"]

    def _fetch_dna_one_hot(self) -> None:
        """
        Fetch the DNA one hot.

        Returns
        -------
        None
        """
        # add DNA one hot
        one_hot_processor = FetchRegionOneHot(
            genome=self.genome,
            region_key="region",
            output_key="dna_one_hot",
            dtype="float32",
        )
        self._working_dataset = self._working_dataset.map_batches(one_hot_processor)
        return

    def get_dataloader(
        self,
        batch_size: int = 64,
        sample_regions: int = 200,
        n_pseudobulks: int = 10,
        min_cov: int = 10,
        max_cov: int = 100000,
        low_cov_ratio: float = 0.1,
        **kwargs,
    ) -> Iterable[dict[str, Any]]:
        """
        Get the dataloader.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, by default 64.
        sample_regions : int, optional
            The number of sample regions, by default 200.
        n_pseudobulks : int, optional
            The number of pseudobulks, by default 10.
        min_cov : int, optional
            The minimum coverage, by default 10.
        max_cov : int, optional
            The maximum coverage, by default 100000.
        low_cov_ratio : float, optional
            The low coverage ratio, by default 0.1.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DataLoader
            The dataloader.
        """
        self._working_dataset = self._dataset
        self._dataset_preprocess(
            sample_regions=sample_regions,
            n_pseudobulks=n_pseudobulks,
            min_cov=min_cov,
            max_cov=max_cov,
            low_cov_ratio=low_cov_ratio,
        )

        _default = {
            "prefetch_batches": 5,
            "local_shuffle_buffer_size": 10000,
            "drop_last": True,
        }
        _default.update(kwargs)
        loader = self._working_dataset.iter_batches(batch_size=batch_size, **_default)
        return loader

    def train(self) -> None:
        """
        Set the dataset mode to "train".

        Returns
        -------
        None
        """
        self._dataset_mode = "train"
        return

    def eval(self) -> None:
        """
        Set the dataset mode to "eval".

        Returns
        -------
        None
        """
        self._dataset_mode = "eval"
        return
