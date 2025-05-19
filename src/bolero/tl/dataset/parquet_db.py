import json
import pathlib
from collections import defaultdict
from typing import Generator

import duckdb
import joblib
import pandas as pd
import pyranges as pr
import ray
from scipy.sparse import csc_matrix, hstack, isspmatrix

from bolero.pp.snap_adata import CSRRowMerge
from bolero.tl.dataset.sc_transforms import CompressedBytesToTensor
from bolero.utils import understand_regions


@ray.remote
class ParallelRowProcessor:
    def __init__(self, row_merge_plan, n_input, n_output):
        self.byte_to_csr = CompressedBytesToTensor()
        self.row_merge_plan = row_merge_plan
        if row_merge_plan is None:
            self.row_to_pseudobulk = {}
        else:
            self.row_to_pseudobulk = {
                prefix: CSRRowMerge(
                    prefix_merge_plan,
                    n_input=n_input[prefix],
                    n_output=n_output[prefix],
                )
                for prefix, prefix_merge_plan in row_merge_plan.items()
            }

    def convert(self, data_dict):
        """Convert the data_dict to a pseudobulk matrix."""
        # byte to csr matrix
        data_dict = self.byte_to_csr(data_dict)

        # merge rows
        _dict = {}
        for key, value in data_dict.items():
            if key in self.row_to_pseudobulk:
                value = self.row_to_pseudobulk[key](value)
            _dict[key] = value
        return _dict


@ray.remote
class _RegionExtractor:
    def extract(
        self,
        cluster_data: dict[str, str | csc_matrix],
        specs: list[tuple[str, int, int]],
    ) -> list[dict[str, str | csc_matrix]]:
        """
        cluster_data: { key: sparse_matrix or array, ... }
        specs: List of (region_name, rel_start, rel_end)
        """
        out = []
        for region_name, rel_start, rel_end in specs:
            rd = {}
            for key, value in cluster_data.items():
                if isspmatrix(value):
                    # slice and copy
                    rd[key] = value[:, rel_start:rel_end].copy()
            rd["region"] = region_name
            out.append(rd)
        return out


class GenomeParquetDB:
    """
    A DuckDB-based interface for querying genomic regions from a Parquet dataset.
    """

    def __init__(self, dataset_dir: str, parallel: int = 1, merge_plan=None):
        """
        dataset_dir: path to the dataset directory containing the Parquet files and the region lookup table.
        """
        self.con = duckdb.connect(database=":memory:")
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.region_lookup: pd.DataFrame = self._register_region_lookup()
        self.region_lookup_bed = pr.PyRanges(
            self.region_lookup[["Chromosome", "Start", "End", "Name"]]
        )

        self.row_names: dict[str, pd.Index] = joblib.load(
            self.dataset_dir / "row_names.joblib"
        )
        self.original_merge_plan: dict[str, list[str]] | None = merge_plan
        merge_plan, pseudobulk_ids = self._register_merge_plan(merge_plan)
        self.merge_plan: dict[str, dict[int, int]] | None = merge_plan
        self.pseudobulk_ids: pd.Index = pseudobulk_ids

        self.parallel = parallel
        self._actor_pool = self._create_row_actor_pool()
        self._extractor_pool = self._create_extractor_pool()

    def _register_region_lookup(self):
        """
        Register the region lookup table in DuckDB to query parquet dataset by regions in "chrom:start-end".
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
        region_lookup = pd.concat([region_bed, region_lookup], axis=1)  # type: ignore[arg-type]

        # Register the DataFrame as a table in DuckDB
        self.con.register("region_lookup", region_lookup)
        return region_lookup

    def _register_merge_plan(
        self, merge_plan: dict[str, list[str]] | None
    ) -> tuple[dict[str, dict[int, int]] | None, pd.Index]:
        """
        Register the merge plan from {pseudobulk_id: [row_names]} to {row_id: pseudobulk_row_id}
        This conversion generates merge_plan for CSRRowMerge.
        CSRRowMerge will use this to merge cell rows into pseudobulk rows.

        Currently, assuming the merge plan applies to all prefix in the parquet dataset.
        """
        if merge_plan is None:
            return None, pd.Index([])

        row_name_mapping = self.row_names

        all_merge_plan = {}
        pseudobulk_ids = list(merge_plan.keys())
        for prefix, cell_row_names in row_name_mapping.items():
            prefix_plan = defaultdict(list)
            for pseudobulk_id, pseudobulk_row_names in merge_plan.items():
                pseudobulk_idx = pseudobulk_ids.index(pseudobulk_id)
                cell_row_indices = sorted(
                    cell_row_names.get_indexer(pseudobulk_row_names)
                )
                for cell_row_idx in cell_row_indices:
                    prefix_plan[cell_row_idx].append(pseudobulk_idx)
            all_merge_plan[prefix] = prefix_plan
        pseudobulk_ids = pd.Index(pseudobulk_ids)
        return all_merge_plan, pseudobulk_ids

    def _create_row_actor_pool(self) -> ray.util.ActorPool:
        """
        Create a pool of actors for parallel processing of rows.
        Each actor will handle the conversion of rows to pseudobulk format.
        """
        n_input: dict[str:int] = {
            prefix: len(row_names) for prefix, row_names in self.row_names.items()
        }
        n_output: dict[str, int] = {
            prefix: self.pseudobulk_ids.size for prefix in n_input.keys()
        }

        actors = [
            ParallelRowProcessor.remote(
                row_merge_plan=self.merge_plan, n_input=n_input, n_output=n_output
            )  # type: ignore[arg-type]
            for _ in range(self.parallel)
        ]
        actor_pool = ray.util.ActorPool(actors)
        return actor_pool

    def _create_extractor_pool(self) -> ray.util.ActorPool:
        """
        Create a pool of actors for parallel processing of regions.
        Each actor will handle the extraction of regions from the cluster data.
        """
        actors = [_RegionExtractor.remote() for _ in range(self.parallel)]
        actor_pool = ray.util.ActorPool(actors)
        return actor_pool

    def query_parquet_regions(
        self, regions: list[str], return_ordered=True, tocsc=False
    ) -> Generator:
        """
        Given a list of regions, find which Parquet file(s) contain any of those regions,
        read the data from those files, and yield the rows as dictionaries.

        IMPORTANT: the regions will be deduplicated in the SQL query, so if repeated regions
        are passed in, they will be returned only once.

        Parameters
        ----------
        regions: List of genomic regions in the format "chrom:start-end",
            the region coordinates must be exactly the same as regions in parquet files.
            for more general region query, use "query_regions" method.
        return_ordered: If True, the results are returned in the order of the input regions.
                        If False, the results are returned in an unordered fashion.
        tocsc: If True, the data will be converted to a sparse matrix in CSC format.
        """
        # Build a comma-separated list of quoted region strings for SQL
        regions = list(set(regions))  # Deduplicate regions
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
        cols = [desc[0] for desc in cursor.description]  # type: ignore[union-attr]

        # --- round‐robin tasks in flight, flush as soon as a batch completes ---
        actor_pool = self._actor_pool
        runing_task = 0
        while True:
            row = cursor.fetchone()
            if row is None:
                break

            row_dict = dict(zip(cols, row))
            actor_pool.submit(lambda a, x: a.convert.remote(x), row_dict)
            runing_task += 1

            # once we've launched at least one task per actor, wait for that many
            if runing_task >= self.parallel * 4:
                if return_ordered:
                    _data = actor_pool.get_next()
                else:
                    _data = actor_pool.get_next_unordered()
                if tocsc:
                    _data = {
                        k: v.tocsc() if isspmatrix(v) else v for k, v in _data.items()
                    }
                yield _data
                actor_pool.submit(lambda a, x: a.convert.remote(x), row_dict)

        while runing_task > 0:
            if return_ordered:
                _data = actor_pool.get_next()
            else:
                _data = actor_pool.get_next_unordered()
            if tocsc:
                _data = {k: v.tocsc() if isspmatrix(v) else v for k, v in _data.items()}
            yield _data
            runing_task -= 1
        return

    def _get_non_overlap_parquet_clusters(
        self, parquet_regions_bed: pr.PyRanges
    ) -> tuple[dict[str, dict], pr.PyRanges]:
        """
        Given a parquet_regions bed, query the parquet dataset for the regions,
        and return the non-overlapping parquet clusters by merging the overlapping parquet regions.
        """
        # query parquet regions
        parquet_regions_data = {}
        for data in self.query_parquet_regions(parquet_regions_bed.df["Name"].tolist()):
            parquet_regions_data[data["region"]] = data

        # Merge overlapping parquet regions into non-overlapping parquet cluster regions
        parquet_clusters_data = {}
        parquet_cluster_bed = []
        parquet_region_with_cluster = parquet_regions_bed.cluster().df
        for _, cluster_regions in parquet_region_with_cluster.groupby("Cluster"):
            cchrom = cluster_regions["Chromosome"].iloc[0]
            cstart = cluster_regions["Start"].min()
            cend = cluster_regions["End"].max()
            cregion = f"{cchrom}:{cstart}-{cend}"
            cdata = {"region": cregion}
            parquet_cluster_bed.append([cchrom, cstart, cend, cregion])

            cur_end = cstart
            for _, (_, start, end, name, *_) in cluster_regions.iterrows():
                region_data = parquet_regions_data[name]
                for k, v in region_data.items():
                    if k == "region":
                        continue
                    v = v.tocsc()
                    cut_left = cur_end - start
                    if cut_left > 0:
                        v = v[:, cut_left:]
                    if k not in cdata:
                        cdata[k] = [v]
                    else:
                        cdata[k].append(v)
                    cur_end = end
            cdata = {
                k: hstack(v) if isinstance(v, list) else v for k, v in cdata.items()
            }
            parquet_clusters_data[cregion] = cdata
        parquet_cluster_bed = pr.PyRanges(
            pd.DataFrame(
                parquet_cluster_bed, columns=["Chromosome", "Start", "End", "Name"]
            )
        )
        return parquet_clusters_data, parquet_cluster_bed

    def _extract_region_from_cluster(
        self,
        regions_to_get: pr.PyRanges,
        cluster_data: dict[str, str | csc_matrix],
        cluster_bed: pr.PyRanges,
    ) -> list[dict[str, str | csc_matrix]]:
        # Build a map: cluster_name -> list of (region_name, rel_start, rel_end)
        joined = regions_to_get.join(
            cluster_bed,
            how="left",  # keep every region, even if no cluster (shouldn't happen)
        )
        jdf = joined.df
        jdf["rel_start"] = jdf["Start"] - jdf["Start_b"]  # _b is the cluster
        jdf["rel_end"] = jdf["End"] - jdf["Start_b"]
        specs_per_cluster: dict[str, list[tuple[str, int, int]]] = (
            jdf.groupby("Name_b")
            .apply(
                lambda g: list(
                    zip(
                        g["Name"].tolist(),  # region_name
                        g["rel_start"].astype(int),  # rel_start
                        g["rel_end"].astype(int),  # rel_end
                    )
                )
            )
            .to_dict()
        )

        cluster_data_refs = {
            cname: ray.put(data) for cname, data in cluster_data.items()
        }
        task_inputs = []
        for cluster, specs in specs_per_cluster.items():
            chunk_size = 100
            for i in range(0, len(specs), chunk_size):
                task_inputs.append(
                    (cluster_data_refs[cluster], specs[i : i + chunk_size])
                )

        # Step D: launch all extraction tasks in parallel
        region_data_dicts = []
        tasks = self._extractor_pool.map_unordered(
            lambda a, x: a.extract.remote(*x), task_inputs
        )
        for result in tasks:
            region_data_dicts.extend(result)
        return region_data_dicts

    def query_regions(
        self,
        regions: list[str] | pr.PyRanges | pd.DataFrame | pd.Index,
    ):
        """
        Load data from the Parquet dataset for any regions.

        Returns
        -------
        region_data_dicts: List of dictionaries, each containing the data for a region.
            The keys of the dictionaries are the same as the columns in the Parquet dataset.
            The values are either sparse matrices or arrays, depending on the data type.
        """
        regions_to_get: pr.PyRanges = pr.PyRanges(understand_regions(regions)).sort()

        # Load relevant parquet regions and merge them into non-overlapping clusters
        parquet_regions_to_get: pr.PyRanges = self.region_lookup_bed.overlap(
            regions_to_get
        ).sort()
        cluster_data, cluster_bed = self._get_non_overlap_parquet_clusters(
            parquet_regions_to_get
        )

        region_data_dicts = self._extract_region_from_cluster(
            regions_to_get, cluster_data, cluster_bed
        )
        return region_data_dicts
