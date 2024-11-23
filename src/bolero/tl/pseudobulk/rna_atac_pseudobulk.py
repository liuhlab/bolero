from typing import Union

import joblib
import numpy as np
import pandas as pd

from bolero.utils import validate_config


# vq_records schema:
# vq_records[vq] = {
#     "cluster_ids": np.array or dict[str, np.array],
#     "vq_ind": vq_ind, np.array
#     "vq_emb": vq_emb, np.array
#     "n_frags": n_frags, int or dict[str, int]
#     "cov_scale": n_frags_log2fc, float or dict[str, float] log2 fold change of psedobulk coverage to target coverage
#     **kwargs
# }
class RNAVQPseudobulker:
    default_config = {
        "vq_records": "REQUIRED",
        "use_vq_emb": False,
        # this prefix will be the final output prefix that occurs in the data dict
        # When parquet has only one prefix, this is also the prefix name in the parquet
        "downsample_vq": None,
        "prefix_name": "pseudobulk",
        "add_cov_to_emb": False,
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
        vq_records: Union[str, dict],
        use_vq_emb: bool = True,
        downsample_vq: int = None,
        prefix_name: str = "pseudobulk",
        add_cov_to_emb: bool = False,
        seed=42,
    ):
        """
        Load VQ records and prepare pseudobulk data.

        Parameters
        ----------
        vq_records (dict[str, dict]):
            The prefix name (in ray dataset) to VQ records file path mapping.
        use_vq_emb (bool): Whether to use VQ embeddings.
        downsample_vq (int): Number of VQs to downsample to.
        prefix_name (str): The prefix name to use in the output data dict.
        """
        self.local_rng = np.random.default_rng(seed=seed)
        self.use_vq_emb = use_vq_emb
        emb_key = "vq_emb" if use_vq_emb else "vq_ind"
        cov_key = "cov_scale"

        vq_records = self._load_vq_records(vq_records, downsample_vq)
        vq_keys = list(vq_records.keys())
        # process VQ records
        self.predefined_pseudobulks = {}
        self.pseudobulk_ids = pd.Index(vq_keys)
        self.n_pids = self.pseudobulk_ids.size

        self.prefix_order = None
        self.pseudobulk_vq_data_type = None
        for idx, vq in enumerate(self.pseudobulk_ids):
            data = vq_records[vq]
            emb_data = data[emb_key]
            cov_value = data[cov_key]
            rows = data["cluster_ids"]
            if isinstance(rows, dict):
                parquet_prefix_to_rows = rows
                if self.prefix_order is None:
                    self.pseudobulk_vq_data_type = "multi_prefix"
                    self.prefix_order = list(parquet_prefix_to_rows.keys())
                cov_value_list = [cov_value[prefix] for prefix in self.prefix_order]
                if add_cov_to_emb:
                    emb_data = np.concatenate(
                        [emb_data, np.array(cov_value_list)]
                    ).astype("float32")
            else:
                parquet_prefix_to_rows = {prefix_name: data["cluster_ids"]}
                if self.prefix_order is None:
                    self.pseudobulk_vq_data_type = "single_prefix"
                    self.prefix_order = [prefix_name]
                cov_value_list = [cov_value]
                if add_cov_to_emb:
                    emb_data = np.concatenate(
                        [emb_data, np.array(cov_value_list)]
                    ).astype("float32")
            self.vq_emb_dims = emb_data.size - len(self.prefix_order)
            self.predefined_pseudobulks[vq] = [
                parquet_prefix_to_rows,
                emb_data,
                np.array(cov_value_list),
                idx,
            ]

        # create a random pool of pseudobulks
        self.random_pid = self.local_rng.choice(
            self.pseudobulk_ids, self.n_pids, replace=False
        )
        return

    def _load_vq_records(self, vq_records, downsample_vq):
        if isinstance(vq_records, str):
            vq_records = joblib.load(vq_records)
        if downsample_vq is not None:
            use_vq_id = self.local_rng.choice(
                list(vq_records.keys()), downsample_vq, replace=False
            )
            vq_records = {k: v for k, v in vq_records.items() if k in use_vq_id}
            print(f"Downsampled to {len(vq_records)} VQs.")
        return vq_records

    def take_by_name(self, name):
        """Take a VQ pseudobulk by name."""
        data = self.predefined_pseudobulks[name]
        return data

    def take(self, n):
        """Take n pseudobulks from the random pool."""
        while self.random_pid.size < n:
            _random_pid = self.local_rng.choice(
                self.pseudobulk_ids, self.n_pids, replace=False
            )
            self.random_pid = np.concatenate([self.random_pid, _random_pid])
        use_pids = self.random_pid[:n]
        self.random_pid = self.random_pid[n:].copy()

        pseudobulks = [self.predefined_pseudobulks[pid] for pid in use_pids]

        # each item in pseudobulks is
        # Type 1 format (single prefix in the parquet):
        # self.pseudobulk_vq_data_type = 'single_prefix'
        # [
        #     prefix_to_rows: dict[str, np.array],  # parquet prefix csr matrix to rows in that csr, only one prefix in this case
        #     emb_data: np.array, # embedding data concatenated with cov_value
        #     cov_value: np.array, # coverage value
        #     idx: int # index of the pseudobulk
        # ]
        # Type 2 format (multiple prefixes in the parquet):
        # self.pseudobulk_vq_data_type = 'multi_prefix'
        # [
        #     prefix_to_rows: dict[str, dict[str, np.array]],  # output prefix to parquet prefix csr matrix to rows in that csr
        #     emb_data: np.array, # embedding data concatenated with cov_value
        #     cov_value: np.array, # coverage value
        #     idx: int # index of the pseudobulk
        # ]
        return pseudobulks
