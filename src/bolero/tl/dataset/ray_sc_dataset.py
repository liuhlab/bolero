# TODO: scRayDataset inherits from RayDataset, and then in scprinter, scPrinterscDataset inherits from scPrinterDataset.
# Once the sc dataset processed pseudobulk and provides region and pseudobulk data dict, the remaining preprocess step should be the same as bulk train model.
import pathlib
from typing import Optional, Union

import numpy as np
import pandas as pd
import ray

from bolero import Genome
from bolero.tl.dataset.sc_transforms import scMetaRegionToBulkRegion
from bolero.tl.pseudobulk.generator import PseudobulkGenerator


class RaySingleCellDataset:
    """Single cell dataset for cell-by-meta-region data."""

    def __init__(
        self,
        dataset_path: str,
        use_prefixs: Optional[list[str]] = None,
        override_num_blocks=None,
        chroms=None,
        shuffle_files=True,
        genome: str = None,
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
        if chroms is None:
            chrom_dirs = [str(p) for p in pathlib.Path(dataset_path).glob("chr*")]
        else:
            if isinstance(chroms, str):
                chroms = [chroms]
            chrom_dirs = [f"{dataset_path}/{chrom}" for chrom in chroms]

        self._dataset = ray.data.read_parquet(
            chrom_dirs,
            file_extensions=["parquet"],
            shuffle="file" if shuffle_files else None,
            override_num_blocks=override_num_blocks,
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
        if genome is None:
            with open(f"{dataset_path}/genome.flag") as f:
                genome = f.read().strip()
        if isinstance(genome, str):
            self.genome = Genome(genome)
        else:
            self.genome = genome

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
        cell_embedding: Union[str, pathlib.Path, pd.DataFrame],
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
        if isinstance(cell_embedding, (str, pathlib.Path)):
            _embedding = pd.read_feather(cell_embedding)
            _embedding = _embedding.set_index(_embedding.columns[0])
        elif isinstance(cell_embedding, pd.DataFrame):
            _embedding = cell_embedding

        pseudobulker = PseudobulkGenerator(
            embedding=_embedding, barcode_order=self.barcode_order
        )
        if predefined_pseudobulk:
            pseudobulker.add_predefined_pseudobulks(predefined_pseudobulk)
        self.pseudobulker = pseudobulker

        # TODO: check pseudobulk prefix, cell barcode with the dataset's prefix and barcode
        # all pseudobulk cells should occured in the dataset
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
        return

    def _pseudobulk_and_extract_regions(
        self,
        sample_regions: int,
        n_pseudobulks: int,
        min_cov: int,
        max_cov: int,
        low_cov_ratio: float,
        num_cpus: int = 1,
        memory: float = "auto",
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
        # TODO: determine flat_map memory dynamically based on the size of the dataset
        if memory == "auto":
            memory = 5 * 1024**3  # 4GB

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
        self._working_dataset = self._working_dataset.flat_map(
            processor, num_cpus=num_cpus, memory=memory
        )
        # after flat_map processor, each row in working_dataset is a dict with keys:
        # ["bulk_embedding", "bulk_data", "region"]

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
