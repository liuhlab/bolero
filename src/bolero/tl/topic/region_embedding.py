import pathlib
from typing import Union

import pandas as pd

from bolero.utils import understand_regions


class RegionEmbedder:
    """
    Class for embedding regions.

    Parameters
    ----------
    dtype : str, optional
        The data type of the embedding, by default "float32".

    Attributes
    ----------
    predefined_region_embedding : pandas.DataFrame or None
        The predefined region embedding.
    dtype : str
        The data type of the embedding.

    Methods
    -------
    add_predefined_embedding(region_embedding)
        Add a predefined region embedding.
    _get_predefined_region_embedding(regions)
        Get the predefined region embedding for the given regions.
    __call__(regions, predefined=True)
        Embed the regions.

    """

    def __init__(self, dtype: str = "float32"):
        """
        Initialize the RegionEmbedder.

        Parameters
        ----------
        dtype : str, optional
            The data type of the embedding, by default "float32".

        """
        self.predefined_region_embedding = None
        self.dtype = dtype

    def add_predefined_embedding(
        self, region_embedding: Union[str, pathlib.Path, pd.DataFrame]
    ) -> None:
        """
        Add a predefined region embedding.

        Parameters
        ----------
        region_embedding : str or pathlib.Path or pandas.DataFrame
            The path to the region embedding file or the region embedding DataFrame.

        """
        if isinstance(region_embedding, (str, pathlib.Path)):
            df = pd.read_feather(region_embedding)
            df = df.set_index(df.columns[0])
        else:
            df = region_embedding
        self.predefined_region_embedding = df.astype(self.dtype)
        return

    def _get_predefined_region_embedding(self, regions: pd.DataFrame) -> pd.DataFrame:
        """
        Get the predefined region embedding for the given regions.

        Parameters
        ----------
        regions : pandas.DataFrame
            The regions to get the embedding for.

        Returns
        -------
        pandas.DataFrame
            The region embedding for the given regions.

        Raises
        ------
        ValueError
            If no predefined region embedding is found.

        """
        if self.predefined_region_embedding is None:
            raise ValueError("No predefined region embedding found.")
        regions.index = pd.Index(
            regions["Chromosome"].astype(str)
            + ":"
            + regions["Start"].astype(str)
            + "-"
            + regions["End"].astype(str)
        )
        return self.predefined_region_embedding.loc[regions.index]

    def __call__(self, regions: pd.DataFrame, predefined: bool = True) -> pd.DataFrame:
        """
        Embed the regions.

        Parameters
        ----------
        regions : pandas.DataFrame
            The regions to embed.
        predefined : bool, optional
            Whether to use the predefined region embedding, by default True.

        Returns
        -------
        pandas.DataFrame
            The embedded regions.

        Raises
        ------
        NotImplementedError
            If the embedding is not implemented yet.

        """
        regions = understand_regions(regions, as_df=True)

        if predefined:
            return self._get_predefined_region_embedding(regions)
        else:
            raise NotImplementedError("Not implemented yet.")
