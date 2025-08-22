import pathlib
import tempfile

import anndata
import joblib
import numpy as np
import pandas as pd
import ray
from scipy.sparse import isspmatrix

from bolero.utils import deprecated


@ray.remote
class _SumSingleAdata:
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir

    def merge(self, path: str, cell_mapping: pd.Series) -> str | None:
        """
        Merge single adata object by cell_to_group mapping.
        """
        adata = anndata.read_h5ad(path, backed="r")
        use_cells = adata.obs_names.isin(cell_mapping.index)
        if use_cells.sum() == 0:
            return None

        adata = adata[use_cells].to_memory()
        adata.obs["group"] = cell_mapping
        adata_group_sum = {}

        if isspmatrix(adata.X):
            adata.X = adata.X.tocsr()
        for group, cells in adata.obs.groupby("group", observed=True):
            gsum = np.array(adata[cells.index].X.sum(axis=0).A1).astype("float32")
            adata_group_sum[group] = gsum
        out_path = pathlib.Path(self.temp_dir) / path.name
        joblib.dump(adata_group_sum, out_path)
        return out_path


@deprecated
class AdataPseudobulker:
    def __init__(
        self,
        adata_paths: list[str],
        group_cols: str | list[str] | None = None,
        cell_mapping: pd.Series | None = None,
        cell_meta: pd.DataFrame | None = None,
        cell_obs_agg: dict[str, str] | None = None,
    ):
        self.adata_paths = adata_paths
        if cell_mapping is None:
            assert (
                group_cols is not None
            ), "group_cols must be provided if cell_mapping is None."
        else:
            assert (
                group_cols is None
            ), "If cell_mapping is provided, group_cols must be None."

        self.var_names, self.obs_names, self.cell_mapping, self.cell_obs = (
            self._get_obs_and_var_names(
                group_cols=group_cols,
                cell_mapping=cell_mapping,
                cell_meta=cell_meta,
            )
        )
        self.cell_obs_agg = cell_obs_agg or {}

    def _get_obs_and_var_names(
        self,
        group_cols: str | list[str],
        cell_mapping: pd.Series | None,
        cell_meta: pd.DataFrame | None = None,
    ) -> pd.Series:
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        var_names = None
        obs_col = []
        cell_mapping_col = []
        for path in self.adata_paths:
            adata = anndata.read_h5ad(path, backed="r")
            obs_col.append(adata.obs)

            if var_names is None:
                var_names = adata.var_names
            else:
                # make sure all adatas have the same var_names
                assert adata.var_names.equals(
                    var_names
                ), "All adatas must have the same var_names."

            if cell_mapping is None:
                if cell_meta is not None:
                    for col in group_cols:
                        if col in cell_meta.columns:
                            adata.obs[col] = cell_meta[col]

                _adata_mapping = (
                    adata.obs[group_cols]
                    .dropna(subset=group_cols, how="any")
                    .agg(lambda x: "_".join(x.astype(str)), axis=1)
                )
                cell_mapping_col.append(_adata_mapping)
        if cell_mapping is None:
            cell_mapping = pd.concat(cell_mapping_col)

        cell_mapping = cell_mapping.astype("category")
        obs_names = pd.Index(cell_mapping.unique(), name="obs_names").astype(str)
        total_cell_obs = pd.concat(obs_col).reindex(cell_mapping.index)
        if cell_meta is not None:
            for col, data in cell_meta.items():
                total_cell_obs[col] = data
        return var_names, obs_names, cell_mapping, total_cell_obs

    def _group_metadata(self) -> pd.DataFrame:
        """
        Prepare group metadata and add into merged adata object.
        """
        _agg = {}
        for col, agg in self.cell_obs_agg.items():
            if agg == "first":
                _agg[col] = lambda x: x.iloc[0]
            elif agg == "major":
                _agg[col] = lambda x: x.value_counts().idxmax()
            else:
                _agg[col] = agg

        group_meta = {}
        cell_count = self.cell_mapping.value_counts()
        group_meta["cell_count"] = cell_count

        group_agg = self.cell_obs.groupby(self.cell_mapping).agg(_agg)
        for col, data in group_agg.items():
            group_meta[col] = data

        group_meta = pd.DataFrame(group_meta).reindex(self.obs_names)
        group_meta.index.name = "group"
        return group_meta

    def run(self, output_path: str, parallel: int = 1):
        """
        Run the pseudobulking process on the adata objects.

        Parameters
        ----------
        parallel : int
            Number of parallel processes to use. If 1, will run in serial.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            # Merge each adata object by cell_mapping
            act_pool = ray.util.ActorPool(
                [_SumSingleAdata.remote(temp_dir=tempdir) for _ in range(parallel)]
            )
            inputs = []
            for path in self.adata_paths:
                inputs.append((path, self.cell_mapping))
            temp_paths = act_pool.map(lambda a, x: a.merge.remote(*x), inputs)

            # sum all the results
            total_col = {}
            for temp_path in temp_paths:
                if temp_path is None:
                    continue
                result = joblib.load(temp_path)
                for group, gsum in result.items():
                    if group not in total_col:
                        total_col[group] = gsum
                    else:
                        total_col[group] += gsum

            gsum = np.concatenate(
                [total_col[k][None, :] for k in self.obs_names], axis=0
            )

        total_adata = anndata.AnnData(
            X=gsum,
            obs=self._group_metadata(),
            var=pd.DataFrame(index=self.var_names),
        )

        total_adata.write_h5ad(output_path)
