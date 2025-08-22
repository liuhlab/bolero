from typing import Union

import joblib
import numpy as np
import pandas as pd

from bolero.utils import validate_config


# pseudobulk_records schema:
# pseudobulk_records[pseudobulk] = {
#     "cluster_ids": dict[str, np.array],
#     "pseudobulk_emb": pseudobulk_emb, np.array
#     "n_frags": dict[str, int]
#     "cov_scale": dict[str, float] log2 fold change of pseudobulk coverage to target coverage
#     **kwargs
# }
class SinglePseudobulker:
    default_config = {
        "pseudobulk_records": "REQUIRED",
        # this prefix will be the final output prefix that occurs in the data dict
        # When parquet has only one prefix, this is also the prefix name in the parquet
        "downsample_pseudobulks": None,
        "prefix_name": "pseudobulk",
        "add_cov_to_emb": False,
        "emb_key": "embedding",
        # barcode order is associated with each parquet dataset, it will be
        # added by the parquet dataset class who maintain the pseudobulker object
        "barcode_order": None,
    }

    @classmethod
    def create_from_config(cls, **config):
        """Create the pseudobulk generator from configuration."""
        config = {k: v for k, v in config.items() if k in cls.default_config}
        validate_config(config, cls.default_config)
        pseudobulker = cls(**config)
        return pseudobulker

    def __init__(
        self,
        pseudobulk_records: Union[str, dict],
        downsample_pseudobulks: int = None,
        prefix_name: str = "pseudobulk",
        add_cov_to_emb: bool = False,
        barcode_order: Union[str, list] = None,
        emb_key="embedding",
        seed=42,
    ):
        """
        Load pseudobulk records and prepare pseudobulk data.

        Parameters
        ----------
        pseudobulk_records (dict[str, dict]):
            The prefix name (in ray dataset) to pseudobulk records file path mapping.
        downsample_pseudobulks (int): Number of pseudobulks to downsample to.
        prefix_name (str): The prefix name to use in the output data dict.
        """
        self.local_rng = np.random.default_rng(seed=seed)
        if emb_key is None:
            emb_key = "pseudobulk_emb"
        cov_key = "cov_scale"

        pseudobulk_records = self._load_pseudobulk_records(
            pseudobulk_records, downsample_pseudobulks
        )
        self.pseudobulk_records = pseudobulk_records

        record_keys = list(pseudobulk_records.values())[0].keys()

        self.barcode_order = barcode_order
        if barcode_order is not None:
            self.barcode_order = {
                k: {c: i for i, c in enumerate(vl)} for k, vl in barcode_order.items()
            }
        # check if the pseudobulk records cluster_ids need to be converted to int
        pseudobulk_example = pseudobulk_records[list(pseudobulk_records.keys())[0]]
        cells_example = list(pseudobulk_example["cluster_ids"].values())[0]
        if isinstance(cells_example[0], str):
            need_int_convert = True
            assert (
                self.barcode_order is not None
            ), "barcode_order must be provided when cluster_ids are str."
        else:
            need_int_convert = False

        if emb_key not in record_keys:
            raise ValueError(
                f"Pseudobulk records must contain {emb_key} key. Found {record_keys}."
            )

        pseudobulk_keys = list(pseudobulk_records.keys())
        # process pseudobulk records
        self.predefined_pseudobulks = {}
        self.pseudobulk_ids = pd.Index(pseudobulk_keys)
        self.n_pids = self.pseudobulk_ids.size
        self.sampling_weights = {}

        self.prefix_order = None
        self.pseudobulk_data_type = None
        for idx, pid in enumerate(self.pseudobulk_ids):
            data = pseudobulk_records[pid]
            emb_data = data[emb_key]
            cov_value = data[cov_key]
            self.sampling_weights[pid] = data.get("sample_weight", 1)
            rows = data["cluster_ids"]
            if isinstance(rows, dict):
                parquet_prefix_to_rows = rows
                if self.prefix_order is None:
                    self.pseudobulk_data_type = "multi_prefix"
                    self.prefix_order = list(parquet_prefix_to_rows.keys())
                cov_value_list = [cov_value[prefix] for prefix in self.prefix_order]
                if add_cov_to_emb:
                    emb_data = np.concatenate(
                        [emb_data, np.array(cov_value_list)]
                    ).astype("float32")
            else:
                parquet_prefix_to_rows = {prefix_name: data["cluster_ids"]}
                if self.prefix_order is None:
                    self.pseudobulk_data_type = "single_prefix"
                    self.prefix_order = [prefix_name]
                cov_value_list = [cov_value]
                if add_cov_to_emb:
                    emb_data = np.concatenate(
                        [emb_data, np.array(cov_value_list)]
                    ).astype("float32")
            self.emb_dims = emb_data.shape[-1] - len(self.prefix_order)

            if need_int_convert:
                # convert str barcode to int row index
                parquet_prefix_to_rows = {
                    k: sorted([self.barcode_order[k][c] for c in v])
                    for k, v in parquet_prefix_to_rows.items()
                }

            self.predefined_pseudobulks[pid] = [
                parquet_prefix_to_rows,
                emb_data,
                np.array(cov_value_list),
                idx,
            ]

        self.sampling_weights = (
            pd.Series(self.sampling_weights).reindex(self.pseudobulk_ids).values
        )
        self.sampling_weights = self.sampling_weights / self.sampling_weights.sum()
        return

    def _load_pseudobulk_records(self, pseudobulk_records, downsample_pseudobulks):
        if isinstance(pseudobulk_records, str):
            pseudobulk_records = joblib.load(pseudobulk_records)
            if "pseudobulk_records" in pseudobulk_records:
                pseudobulk_records = pseudobulk_records["pseudobulk_records"]
        if downsample_pseudobulks is not None:
            use_pid = self.local_rng.choice(
                list(pseudobulk_records.keys()), downsample_pseudobulks, replace=False
            )
            pseudobulk_records = {
                k: v for k, v in pseudobulk_records.items() if k in use_pid
            }
            print(f"Downsampled to {len(pseudobulk_records)} pseudobulks.")
        return pseudobulk_records

    def take_by_name(self, name):
        """Take a pseudobulk by name."""
        data = self.predefined_pseudobulks[name]
        return data

    def take(self, n):
        """Take n pseudobulks from the random pool."""
        if n > self.n_pids:
            raise ValueError(
                f"Cannot take {n} pseudobulks, only {self.n_pids} available."
            )
        use_pids = self.local_rng.choice(
            self.pseudobulk_ids, n, replace=False, p=self.sampling_weights
        )
        pseudobulks = [self.predefined_pseudobulks[pid] for pid in use_pids]

        # each item in pseudobulks is
        # Type 1 format (single prefix in the parquet):
        # self.pseudobulk_data_type = 'single_prefix'
        # [
        #     prefix_to_rows: dict[str, np.array],  # parquet prefix csr matrix to rows in that csr, only one prefix in this case
        #     emb_data: np.array, # embedding data concatenated with cov_value
        #     cov_value: np.array, # coverage value
        #     idx: int # index of the pseudobulk
        # ]
        # Type 2 format (multiple prefixes in the parquet):
        # self.pseudobulk_pseudobulk_data_type = 'multi_prefix'
        # [
        #     prefix_to_rows: dict[str, dict[str, np.array]],  # output prefix to parquet prefix csr matrix to rows in that csr
        #     emb_data: np.array, # embedding data concatenated with cov_value
        #     cov_value: np.array, # coverage value
        #     idx: int # index of the pseudobulk
        # ]
        return pseudobulks
