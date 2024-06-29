import pathlib
from typing import Any, Dict, List, Union

import numpy as np
import pysam

from bolero.pp.genome_chunk_dataset import query_allc_region
from bolero.utils import understand_regions


def _open_allc(allc_path):
    handle = pysam.TabixFile(allc_path, mode="r")
    return handle


class FetchRegionALLCs:
    def __init__(
        self,
        allc_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        region_key: str = "region",
    ) -> None:
        """
        Initialize FetchRegionALLCs.

        Parameters
        ----------
        - allc_paths: Path(s) to the allc file(s).
        - region_key: Key in the data_dict that represents the region.

        Returns
        -------
        None
        """
        if isinstance(allc_paths, (str, pathlib.Path)):
            allc_paths = [allc_paths]
        self.allc_paths = allc_paths
        self.region_key = region_key
        self.allc_handles = [_open_allc(path) for path in allc_paths]

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch region ALLCs.

        Parameters
        ----------
        - data_dict: Dictionary containing the data.

        Returns
        -------
        Dictionary containing the updated data.
        """
        region_ = data_dict[self.region_key]
        if isinstance(region_, str):
            region_ = [region_]
        regions = understand_regions(region_)
        assert (regions["End"] - regions["Start"]).unique().shape[
            0
        ] == 1, "Regions must have the same length."

        n_regions = len(region_)
        region_length = regions["End"].iloc[0] - regions["Start"].iloc[0]
        n_allc = len(self.allc_paths)

        total_mc_values = np.zeros(
            shape=(n_regions, n_allc, region_length), dtype=np.float32
        )
        total_cov_values = np.zeros(
            shape=(n_regions, n_allc, region_length), dtype=np.float32
        )
        for idx, (_, (chrom, start, end, *_)) in enumerate(regions.iterrows()):
            for idy, allc_handle in enumerate(self.allc_handles):
                mc_values, cov_values = query_allc_region(
                    allc_handle, chrom, start, end
                )
                total_mc_values[idx, idy, :] = mc_values
                total_cov_values[idx, idy, :] = cov_values
        data_dict["mc_values"] = total_mc_values
        data_dict["cov_values"] = total_cov_values
        return data_dict

    def close(self) -> None:
        """
        Close allc handles.

        Returns
        -------
        None
        """
        for handle in self.allc_handles:
            handle.close()
