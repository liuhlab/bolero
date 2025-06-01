import pathlib
from collections import defaultdict
from shutil import rmtree

import anndata
import pandas as pd
import pertpy as pt
import ray

DE_TABLE_DTYPES = {
    "variable": "category",
    "log_fc": "float32",
    "logCPM": "float32",
    "F": "float32",
    "p_value": "float32",
    "adj_p_value": "float32",
    "contrast": "category",
    "base": "category",
    "cond": "category",
    "diff_group": "category",
}


def _single_test_contrasts(
    da,
    diff_group,
    column,
    base,
    cond,
    output_path,
    sig_only,
    pval_cutoff,
    log_fc_cutoff,
):
    temp_path = pathlib.Path(f"{output_path}.tmp")
    if pathlib.Path(output_path).exists():
        return

    try:
        res_df = da.test_contrasts(
            da.contrast(column=column, baseline=base, group_to_compare=cond)
        )
    except Exception as e:
        print(diff_group, column, base, cond)
        raise e

    if sig_only:
        is_sig = (res_df["adj_p_value"] < pval_cutoff) & (
            res_df["log_fc"].abs() > log_fc_cutoff
        )
        res_df = res_df[is_sig].copy()

    res_df["contrast"] = column
    res_df["base"] = base
    res_df["cond"] = cond
    res_df["diff_group"] = diff_group
    res_df = res_df.astype(DE_TABLE_DTYPES)
    res_df.to_feather(temp_path, compression="zstd")
    pathlib.Path(temp_path).rename(output_path)
    return


@ray.remote
def _run_edger(
    pdata_path,
    obs_names,
    design,
    test_groups,
    diff_group,
    output_dir,
    sig_only,
    pval_cutoff,
    log_fc_cutoff,
):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    success_flag = output_dir / ".success"
    if success_flag.exists():
        return

    # fit the EdgeR model
    pdata = anndata.read_h5ad(pdata_path, backed="r")
    sel_pdata = pdata[obs_names].to_memory()
    da = pt.tl.EdgeR(adata=sel_pdata, design=design)
    da.fit()

    # test contrasts
    for column, pairs in test_groups.items():
        for base, cond in pairs:
            output_path = output_dir / f"{diff_group}.{column}.{base}.{cond}.feather"
            _single_test_contrasts(
                da,
                diff_group,
                column,
                base,
                cond,
                output_path,
                sig_only=sig_only,
                pval_cutoff=pval_cutoff,
                log_fc_cutoff=log_fc_cutoff,
            )
    success_flag.touch()
    return


class EdgeR:
    def __init__(
        self,
        pdata_path: str,
        test_policy: dict,
        output_prefix: str,
        sig_only=True,
        pval_cutoff=0.1,
        log_fc_cutoff=0,
    ):
        """
        Initialize the EdgeR class for differential expression analysis.

        Parameters
        ----------
        pdata_path : str
            Path to the pseudobulk level raw count data in AnnData format.
        test_policy : dict
            Dictionary containing the test policy, including groupby, test_design, and test_groups.
            Schema:
            {
                "policy1_name": {
                    "groupby": ["pdata_obs_column1", "pdata_obs_column2", ...] OR "pdata_obs_column",
                    "test_design": "design formula", # e.g., "~ condition + batch",
                    "test_groups": {
                        "contrast_column": [(base_group, cond_group), ...]
                    }
                },
                ...
            }
        output_prefix : str
            Prefix for the output files. The results will be saved with name `<output_prefix>_{policy_name}.feather`.
        sig_only : bool, optional
            If True, only significant results will be saved. Default is True.
        pval_cutoff : float, optional
            Adjusted P-value cutoff for significance. Default is 0.1.
        log_fc_cutoff : float, optional
            Log fold change cutoff for significance. Default is 0.
        """
        self.pdata_path = pdata_path
        self.test_policy = test_policy
        self.pval_cutoff = pval_cutoff
        self.sig_only = sig_only
        self.log_fc_cutoff = log_fc_cutoff
        self.output_prefix = output_prefix
        self.output_dir = pathlib.Path(f"{output_prefix}_edger")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_policy()

    def _concat_results(self, policy_dirs):
        for policy_name, dir_list in policy_dirs.items():
            policy_results = []
            for policy_dir in dir_list:
                for path in policy_dir.glob("*.feather"):
                    policy_results.append(pd.read_feather(path))
            if len(policy_results) == 0:
                continue
            policy_results = pd.concat(policy_results, ignore_index=True).astype(
                DE_TABLE_DTYPES
            )
            policy_results.to_feather(f"{self.output_prefix}.{policy_name}.feather")
        return

    def print_policy_and_group_pairs(self):
        """
        Print the test policy and group pairs for debugging purposes.
        """
        pdata = anndata.read_h5ad(self.pdata_path, backed="r")
        group_strs = set()
        for policy_name, policy_dict in self.test_policy.items():
            pdata_groupby = policy_dict["groupby"]
            for group, group_df in pdata.obs.groupby(pdata_groupby, observed=True):
                obs_names = group_df.index
                group_str = "-".join(map(str, group))
                group_strs.add(group_str)
                print(
                    f"Policy: {policy_name}, Group: {group_str}, Samples: {len(obs_names)}"
                )
        return group_strs

    def _validate_policy(self):
        """
        Validate the test policy to ensure it contains the required fields.
        """
        pdata = anndata.read_h5ad(self.pdata_path, backed="r")

        for policy_name, policy_dict in self.test_policy.items():
            assert (
                "test_design" in policy_dict
            ), f"Policy '{policy_name}' is missing 'test_design'."
            test_design = (
                policy_dict["test_design"].lstrip("~").replace(" ", "").split("+")
            )
            terms = set()
            for term in test_design:
                if "+" in term:
                    terms.update(term.split("+"))
                else:
                    terms.add(term)
            for term in terms:
                if term not in pdata.obs.columns:
                    raise ValueError(
                        f"Term '{term}' in 'test_design' of policy '{policy_name}' does not exist in pdata.obs."
                    )

            assert (
                "test_groups" in policy_dict
            ), f"Policy '{policy_name}' is missing 'test_groups'."
            test_groups = policy_dict["test_groups"]
            if not isinstance(test_groups, dict):
                raise ValueError(
                    f"'test_groups' in policy '{policy_name}' must be a dict."
                )
            col_to_cats = defaultdict(set)
            for column, pairs in test_groups.items():
                assert (
                    column in pdata.obs.columns
                ), f"Column '{column}' in 'test_groups' of policy '{policy_name}' does not exist in pdata.obs."
                if not isinstance(pairs, list):
                    raise ValueError(
                        f"Pairs for column '{column}' in 'test_groups' of policy '{policy_name}' must be a list."
                    )
                column_category = pdata.obs[column].unique()
                for base, cond in pairs:
                    if base not in column_category or cond not in column_category:
                        raise ValueError(
                            f"Base '{base}' or condition '{cond}' in 'test_groups' of policy '{policy_name}' "
                            f"does not exist in pdata.obs column '{column}'."
                        )
                    col_to_cats[column].add(base)
                    col_to_cats[column].add(cond)

            assert (
                "groupby" in policy_dict
            ), f"Policy '{policy_name}' is missing 'groupby'."
            groupby = policy_dict["groupby"]
            if isinstance(groupby, str):
                groupby = [groupby]
            for col in groupby:
                assert (
                    col in pdata.obs.columns
                ), f"Column '{col}' in 'groupby' of policy '{policy_name}' does not exist in pdata.obs."
        return

    def fit(self):
        """
        Fit the EdgeR model and perform differential expression analysis.
        """
        pdata = anndata.read_h5ad(self.pdata_path, backed="r")

        tasks = []
        policy_dirs = {}
        for policy_name, policy_dict in self.test_policy.items():
            design = policy_dict["test_design"]
            pdata_groupby = policy_dict["groupby"]
            test_groups = policy_dict["test_groups"]
            dirs = []
            for group, group_df in pdata.obs.groupby(pdata_groupby, observed=True):
                obs_names = group_df.index
                if len(obs_names) < 2:
                    print(f"Skipping group {group} with less than 2 samples.")
                    continue
                group_str = "-".join(map(str, group))
                _this_output_dir = self.output_dir / f"{policy_name}_{group_str}"
                task = _run_edger.remote(
                    pdata_path=self.pdata_path,
                    obs_names=obs_names,
                    design=design,
                    test_groups=test_groups,
                    diff_group=group_str,
                    output_dir=_this_output_dir,
                    sig_only=self.sig_only,
                    pval_cutoff=self.pval_cutoff,
                    log_fc_cutoff=self.log_fc_cutoff,
                )
                dirs.append(_this_output_dir)
                tasks.append(task)
            policy_dirs[policy_name] = dirs
        ray.get(tasks)

        self._concat_results(policy_dirs)
        rmtree(self.output_dir, ignore_errors=True)
        return
