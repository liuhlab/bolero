import json
import pathlib
from typing import Generator, Union

import duckdb
import pandas as pd
import ray
from scipy.sparse import csr_matrix

from bolero.pp.snap_adata import CSRRowMerge
from bolero.tl.dataset.sc_transforms import CompressedBytesToTensor
from bolero.utils import understand_regions


@ray.remote
class ParallelRowProcessor:
    def __init__(self, row_merge_plan, n_input, n_output):
        self.byte_to_csr = CompressedBytesToTensor()
        self.row_merge_plan = row_merge_plan
        self.row_to_pseudobulk = CSRRowMerge(
            row_merge_plan, n_input=n_input, n_output=n_output
        )

    def convert(self, data_dict):
        """Convert the data_dict to a pseudobulk format."""
        # byte to csr matrix
        data_dict = self.byte_to_csr(data_dict)

        # merge rows
        _dict = {}
        for key, value in data_dict.items():
            if isinstance(value, csr_matrix):
                value = self.row_to_pseudobulk(value)
            _dict[key] = value
        return _dict


class GenomeParquetDB:
    """
    A DuckDB-based interface for querying genomic regions from a Parquet dataset.
    """

    def __init__(self, dataset_dir: str, parallel: Union[dict, int] = 1):
        """
        dataset_dir: path to the dataset directory containing the Parquet files and the region lookup table.
        """
        self.con = duckdb.connect(database=":memory:")
        self.dataset_dir = pathlib.Path(dataset_dir)
        self._register_region_lookup()

    def _register_region_lookup(self):
        """
        Register the region lookup table in DuckDB.
        """
        lookup_path = self.dataset_dir / "parquet_row_regions.feather"
        # Read the Feather file into a Pandas DataFrame
        region_lookup = pd.read_feather(lookup_path)
        # Adjust parquet_file column to be full paths
        region_lookup["parquet_file"] = (
            str(self.dataset_dir) + "/" + region_lookup["parquet_file"]
        )

        # add genome coords
        region_bed = understand_regions(region_lookup["region"], as_df=True)
        region_lookup = pd.concat([region_bed, region_lookup], axis=1)

        self.region_lookup = region_lookup
        # Register the DataFrame as a table in DuckDB
        self.con.register("region_lookup", region_lookup)

    def query_regions(self, regions: list[str]) -> Generator[dict]:
        """
        Given a list of region strings, find which Parquet file(s) contain any of those regions,
        read the data from those files, and yield the rows as dictionaries.
        """
        # Build a comma-separated list of quoted region strings for SQL
        region_list_sql = ", ".join(f"'{r}'" for r in regions)

        # 1) Gather the Parquet file paths for these regions
        files_query = f"""
            SELECT DISTINCT parquet_file
            FROM region_lookup
            WHERE region IN ({region_list_sql})
        """
        parquet_files = [row[0] for row in self.con.execute(files_query).fetchall()]
        if not parquet_files:
            raise ValueError("No regions found in the dataset.")

        # 2) Query the data from the list of parquet files and filter on the regions
        parquet_files_json = json.dumps(parquet_files)
        sql = f"""
            SELECT *
            FROM read_parquet({parquet_files_json})
            WHERE region IN ({region_list_sql});
        """
        cursor = self.con.execute(sql)
        # Get column names from the cursor description
        columns = [desc[0] for desc in cursor.description]

        # Yield one row at a time as a dict
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield dict(zip(columns, row))
