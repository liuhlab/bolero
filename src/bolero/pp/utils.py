import numpy as np
import pandas as pd


def get_global_coords(
    chrom_offsets: pd.DataFrame, region_bed_df: pd.DataFrame
) -> np.ndarray:
    """
    Calculate the global coordinates of regions in a DataFrame.

    Parameters
    ----------
        chrom_offsets (pd.DataFrame): A dictionary mapping chromosome names to their global offsets.
        region_bed_df (pd.DataFrame): A DataFrame containing region information.

    Returns
    -------
        np.ndarray: An array of global coordinates for each region.

    """
    add_start = (
        region_bed_df["Chromosome"].map(chrom_offsets["global_start"]).astype(int)
    )
    start = region_bed_df["Start"] + add_start
    end = region_bed_df["End"] + add_start
    global_coords = np.hstack([start.values[:, None], end.values[:, None]])
    return global_coords
