from copy import deepcopy
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

from bolero.tl.model.flow.matcher import ConditionalFlowMatcher
from bolero.utils import validate_config

# pseudobulk_records schema:
# {
#    "pseudobulk_id": {
#        'cluster_ids': {
#            'DA_NAME':
#                [
#                    'meta_cell_1',
#                    'meta_cell_2',
#                    ...
#                 ]
#        },
#    'n_frags': {
#        'DA_NAME': N_FRAGS,
#    },
#    'cov_scale': {
#        'DA_NAME': np.log2(N_FRAGS / TARGET_COV),
#    },
#    'embedding': np.array(emb_dim) # mean embedding of meta cells
#    'embedding_multi': np.array(max(n_meta_cell, 128), emb_dim), # separate embedding for each meta cell
#    'sample_weight': float, # optional
# }

# pseudobulk_and_ot_info schema:
# {
#    'pseudobulk_records': dict of predefined coverage pseudobulk records
#    'meta_cell_emb': pd.DataFrame (n_meta_cell, emb_dim), # embedding of each meta cell
#    'meta_cell_n_frags': pd.Series (n_meta_cell,), # number of fragments for each meta cell
#    'ot_transition': dict of transition matrix between conditions
#    'condition_to_related_pseudobulk': dict of condition to related pseudobulks mapping
#    'target_cov': int, # target coverage for pseudobulk
#    'cond_pair_emb': dict of condition pair embedding
#    'condition_emb': optional, pre-computed condition embedding,
#        if not provided, will use one hot encoder on condition_to_related_pseudobulk.keys()
# }


def sample_mapping(tmat: pd.DataFrame) -> pd.Series:
    """
    Sample a one-to-one mapping from a transition matrix.

    Parameters
    ----------
    tmat : pd.DataFrame
        The transition matrix to sample from. The index is the source names and the columns are the target names.
        shape: (n_source, n_target)

    Returns
    -------
    mapping : pd.Series
        A mapping from source names to target names. The index is the source names and the values are the target names.
        shape: (n_source,)
    """
    _tmat = torch.from_numpy(tmat.values)
    src_names = tmat.index
    tgt_names = tmat.columns
    tgt_idx = torch.multinomial(_tmat, num_samples=1).squeeze(1)
    tgt_ordered = tgt_names[tgt_idx.cpu().numpy()]
    mapping = pd.Series(tgt_ordered, index=src_names)
    return mapping


def pad_or_chunk_emb(emb, target_n=128):
    """
    Pad or chunk the embedding to the target size.
    """
    n = emb.shape[0]
    pad_width = target_n - n
    if pad_width < 0:
        emb = emb[:target_n].copy()
    elif pad_width == 0:
        pass
    else:
        emb = np.pad(emb, ((0, pad_width), (0, 0)), mode="constant")
    return emb


class PredefinedCondEncoder:
    def __init__(self, cond_emb: dict[np.ndarray] = None, conditions=None):
        """
        Condition encoder from pre-defined condition embedding or one hot encoding.
        """
        if cond_emb is not None and conditions is not None:
            raise ValueError(
                "Either cond_emb or conditions must be provided, not both."
            )

        if cond_emb is not None:
            assert isinstance(cond_emb, dict), "cond_emb must be a dict."

            self.cond_emb = cond_emb
            self.onehot_encoder = None
        elif conditions is not None:
            category = np.array(conditions)
            if category.ndim < 2:
                category = category.reshape(-1, 1)
            self.onehot_encoder = OneHotEncoder(dtype="float32", sparse_output=False)
            self.onehot_encoder.fit(category)
        else:
            raise ValueError("Either cond_emb or conditions must be provided.")

        self.emb_dim = (
            list(self.cond_emb.values())[0].shape[-1]
            if cond_emb is not None
            else self.onehot_encoder.get_feature_names_out().size
        )
        return

    def transform(self, cond: str | list[str]) -> np.ndarray:
        """
        Transform the condition to the embedding.
        """
        if self.onehot_encoder is None:
            return self.cond_emb[cond]
        else:
            if not isinstance(cond, np.ndarray):
                cond = np.array(cond)
            if cond.ndim < 2:
                cond = cond.reshape(-1, 1)
            emb = self.onehot_encoder.transform(cond)
            return emb

    def __call__(self, *args, **kwds) -> np.ndarray:
        """
        Call the transform method.
        """
        return self.transform(*args, **kwds)


class PairedPseudobulker:
    default_config = {
        "pseudobulk_and_ot_info": "REQUIRED",
        "emb_key": "embedding",
        "downsample_pseudobulk": None,
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
        pseudobulk_and_ot_info: Union[str, dict],
        emb_key: str = "embedding",
        downsample_pseudobulk: int = None,
        barcode_order: Union[str, list] = None,
        seed=42,
    ):
        """
        Load VQ records and prepare pseudobulk data.

        Parameters
        ----------
        pseudobulk_records (dict[str, dict]):
            The prefix name (in ray dataset) to VQ records file path mapping.
        emb_key (str): The key to use for the embedding.
        downsample_pseudobulk (int): The number of pseudobulks to downsample to.
        barcode_order (dict): The order of barcodes for each prefix.
        flow_match_sigma (float): The sigma for the flow matcher.
        seed (int): The random seed for sampling.
        """
        self.local_rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.cov_key = "cov_scale"

        pseudobulk_and_ot_info = joblib.load(pseudobulk_and_ot_info)

        # 1. pseudobulk dict records
        self.pseudobulk_records: dict[dict[str, Any]] = self._load_records(
            pseudobulk_and_ot_info["pseudobulk_records"], downsample_pseudobulk
        )
        assert (
            emb_key in list(self.pseudobulk_records.values())[0]
        ), f"pseudobulk records must contain {emb_key} key."
        self.emb_key = emb_key
        pseudobulk_keys = list(self.pseudobulk_records.keys())
        # process records
        self.pseudobulk_ids = pd.Index(pseudobulk_keys)
        self.n_pids = self.pseudobulk_ids.size

        # 2. meta cell embedding, shape (n_meta_cell, emb_dim)
        self.meta_cell_emb: pd.DataFrame = pseudobulk_and_ot_info["meta_cell_emb"]

        # 3. meta cell N frags, shape (n_meta_cell,)
        self.meta_cell_n_frags = pseudobulk_and_ot_info["meta_cell_n_frags"]

        # 4. OT transition matrix between conditions
        # schema: {
        #     (source_condition, target_condition):
        #         pd.DataFrame (n_source_meta_cell, n_target_meta_cell)
        # }
        self.ot_transition: dict[str, pd.DataFrame] = pseudobulk_and_ot_info[
            "ot_transition"
        ]
        self.condition_pairs = list(self.ot_transition.keys())

        # 5. condition to related pseudobulk mapping
        self.condition_to_related_pseudobulk = pseudobulk_and_ot_info[
            "condition_to_related_pseudobulk"
        ]
        cond_emb = pseudobulk_and_ot_info.get("condition_emb", None)
        self.condition_encoder = self._make_condition_encoder(cond_emb)

        # 6. target coverage
        self.target_cov = pseudobulk_and_ot_info["target_cov"]

        self.barcode_order = barcode_order
        if barcode_order is not None:
            self.barcode_order = {
                k: {c: i for i, c in enumerate(vl)} for k, vl in barcode_order.items()
            }

        # collect pseudobulk related information
        sampling_weights = {}
        self.prefix_order = None
        for pid in self.pseudobulk_ids:
            data = self.pseudobulk_records[pid]
            sampling_weights[pid] = data.get("sample_weight", 1)

            parquet_prefix_to_rows = data["cluster_ids"]
            if self.prefix_order is None:
                self.prefix_order = list(parquet_prefix_to_rows.keys())

            assert len(self.prefix_order) == 1, "Only one prefix is supported."
            self.prefix_name = self.prefix_order[0]

        sampling_weights = pd.Series(sampling_weights).reindex(self.pseudobulk_ids)
        self.sampling_weights: pd.Series = sampling_weights / sampling_weights.sum()
        return

    def _make_condition_encoder(self, cond_emb):
        if cond_emb is None:
            conditions = list(self.condition_to_related_pseudobulk.keys())
        else:
            conditions = None

        encoder = PredefinedCondEncoder(cond_emb, conditions)
        return encoder

    def _load_records(self, pseudobulk_records, downsample_pseudobulk):
        if isinstance(pseudobulk_records, str):
            pseudobulk_records = joblib.load(pseudobulk_records)
        if downsample_pseudobulk is not None:
            use_id = self.local_rng.choice(
                list(pseudobulk_records.keys()), downsample_pseudobulk, replace=False
            )
            pseudobulk_records = {
                k: v for k, v in pseudobulk_records.items() if k in use_id
            }
            print(f"Downsampled to {len(pseudobulk_records)} pseudobulk records.")
        return pseudobulk_records

    def _sample_single_pseudobulk(self, skip_pids=None):
        # 1. sample a condition pair
        cond_pair = tuple(self.local_rng.choice(self.condition_pairs))
        tmat = self.ot_transition[cond_pair]
        cond_pair_pseudobulks = [None, None]

        # 2. select a predefined pseudobulk from either source or target condition
        p_sample = self.local_rng.choice([0, 1])
        related_pseudobulks = list(
            self.condition_to_related_pseudobulk[cond_pair[p_sample]]
        )
        if skip_pids is not None:
            related_pseudobulks = [
                pid for pid in related_pseudobulks if pid not in skip_pids
            ]
        related_sample_weights = self.sampling_weights.reindex(related_pseudobulks)
        related_sample_weights /= related_sample_weights.sum()
        pid_choice = self.local_rng.choice(
            related_pseudobulks, 1, replace=False, p=related_sample_weights.values
        )[0]
        sel_pseudobulk = deepcopy(self.pseudobulk_records[pid_choice])
        cond_pair_pseudobulks[p_sample] = sel_pseudobulk

        # 3. sample a matched meta cell list from the OT transition matrix
        if p_sample == 1:
            tmat = tmat.T
        p_ot = 0 if p_sample == 1 else 1
        mapping = sample_mapping(tmat)
        p_ot_meta_cells = mapping.loc[
            sel_pseudobulk["cluster_ids"][self.prefix_name]
        ].values.tolist()

        n_frags = self.meta_cell_n_frags.loc[p_ot_meta_cells].sum()
        pad_emb_to_n = sel_pseudobulk["embedding_multi"].shape[0]
        ot_pseudobulk = {
            "cluster_ids": {self.prefix_name: p_ot_meta_cells},
            "n_frags": {self.prefix_name: n_frags},
            "cov_scale": {self.prefix_name: np.log2(n_frags / self.target_cov)},
            "embedding": self.meta_cell_emb.loc[p_ot_meta_cells].mean().values,
            "embedding_multi": pad_or_chunk_emb(
                self.meta_cell_emb.loc[p_ot_meta_cells].values, pad_emb_to_n
            ),
            "sample_weight": sel_pseudobulk["sample_weight"],
        }
        cond_pair_pseudobulks[p_ot] = ot_pseudobulk

        # Following code is for generator to handle the pseudobulk records
        for d, cond in zip(cond_pair_pseudobulks, cond_pair):
            # 1. set the embedding for generator to use
            d["__embedding__"] = d[self.emb_key]
            d["__covlogfc__"] = d[self.cov_key]
            d["__conditionemb__"] = self.condition_encoder(cond)

            # 2. convert meta cell list to int row index
            if self.barcode_order is not None:
                parquet_prefix_to_rows: dict = d["cluster_ids"]
                parquet_prefix_to_rows = {
                    k: sorted([self.barcode_order[k][c] for c in v])
                    for k, v in parquet_prefix_to_rows.items()
                }
                d["cluster_ids"] = parquet_prefix_to_rows

            # self.pseudobulk_records[pid][f'{cov_key}_list'] = cov_value_list
            # self.pseudobulk_records[pid]['parquet_prefix_to_int_rows'] = parquet_prefix_to_rows
            # cov_value_list = [data[cov_key][prefix] for prefix in self.prefix_order]
        return cond_pair_pseudobulks, cond_pair, pid_choice

    def take_by_name(self, name):
        """Take a pseudobulk by name."""
        data = deepcopy(self.pseudobulk_records[name])
        return data

    def take(self, n):
        """Take n pseudobulks from the random pool."""
        if n > self.n_pids:
            raise ValueError(
                f"Cannot take {n} pseudobulks, only {self.n_pids} available."
            )

        records = []
        used_pids = set()
        for _ in range(n):
            # 1. sample a condition pair
            cond_pair_pseudobulks, cond_pair, pid_choice = (
                self._sample_single_pseudobulk(skip_pids=used_pids)
            )
            used_pids.add(pid_choice)
            # cond_pair_pseudobulk: list of two pseudobulk records, source and target
            # cond_pair: tuple of two condition names
            records.append(cond_pair_pseudobulks)
        return records


class GeneratePairedPseudobulk:
    """
    Transform meta region data into bulk region data.
    """

    def __init__(
        self,
        n_pseudobulks=10,
        return_rows=False,
        inplace=False,
        bypass_keys=None,
        normalize_cov=None,
        reduce_resolution=None,
        flow_matcher_sigma=0.0,
        **name_to_pseudobulker,
    ):
        self.name_to_pseudobulker = name_to_pseudobulker
        self.n_pseudobulks = n_pseudobulks
        self.return_rows = return_rows
        self.inplace = inplace
        self.flow_matcher = ConditionalFlowMatcher(sigma=flow_matcher_sigma)

        self.bypass_keys = ["region"]
        if bypass_keys is not None:
            if bypass_keys is str:
                self.bypass_keys.append(bypass_keys)
            else:
                self.bypass_keys.extend(list(bypass_keys))
        self.normalize_cov = normalize_cov
        self.reduce_resolution = reduce_resolution

        # suffix for p1 and p0 data keys
        self.suffix = ["_1", "_0"]
        return

    def _reduce_resolution(self, data):
        resolution = self.reduce_resolution
        # from (1, seq_len) to (1, seq_len // resolution) by summing
        data = data.reshape(1, -1, resolution).sum(axis=-1)
        return data

    def _sample_location_and_conditional_flow(self, data_dict, output_prefix):
        x0 = data_dict[f"{output_prefix}:bulk_data_0"]
        x1 = data_dict[f"{output_prefix}:bulk_data_1"]
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
            x0=torch.from_numpy(x0), x1=torch.from_numpy(x1), t=None, return_noise=False
        )
        data_dict["__t__"] = t.numpy()
        data_dict["__xt__"] = xt.numpy()
        data_dict["__ut__"] = ut.numpy()
        return data_dict

    def __call__(self, data_dict: dict[str, bytes]) -> list[dict[str, np.ndarray]]:
        """Generate pseudobulks for each output prefix."""
        list_of_dicts = []

        assert len(self.name_to_pseudobulker) == 1, "Only one pseudobulker is allowed"
        output_prefix, pseudobulker = list(self.name_to_pseudobulker.items())[0]
        pseudobulker: PairedPseudobulker

        # print("before pseudobulk", data_dict["pseudobulk"].shape)
        # merge rows (cell or sample) to bulk and also get embedding data
        for cond_pair_pseudobulks in pseudobulker.take(self.n_pseudobulks):
            this_bulk_dict = {}
            for pseudobulk, suffix in zip(cond_pair_pseudobulks, self.suffix):
                # 1. add condition embedding
                this_bulk_dict[f"{output_prefix}:condition_emb{suffix}"] = pseudobulk[
                    "__conditionemb__"
                ]

                # 2. add pseudobulk embedding
                row_embedding = pseudobulk["__embedding__"]
                this_bulk_dict[f"{output_prefix}:embedding_data{suffix}"] = (
                    row_embedding
                )

                # 3. add pseudobulk data with optional
                # coverage normalization and resolution reduction
                prefix_to_rows = pseudobulk["cluster_ids"]
                cov_logfc = pseudobulk["__covlogfc__"]
                combined_bulk_data = []
                for prefix in pseudobulker.prefix_order:
                    prefix_rows = prefix_to_rows[prefix]
                    # row_by_base is a csr_matrix of shape (n_rows, region_length)
                    try:
                        row_by_base = data_dict[prefix]
                    except KeyError as e:
                        raise KeyError(
                            f"Key {prefix} not found in data_dict, {data_dict.keys()}"
                        ) from e

                    _bulk_values = (
                        row_by_base[prefix_rows].sum(axis=0).A1
                    )  # (1, region_length)

                    if self.normalize_cov:
                        prefix_cov_logfc = cov_logfc[prefix]
                        _bulk_values /= 2**prefix_cov_logfc

                    if self.reduce_resolution:
                        _bulk_values = self._reduce_resolution(_bulk_values)

                    combined_bulk_data.append(_bulk_values)
                this_bulk_dict[f"{output_prefix}:bulk_data{suffix}"] = np.vstack(
                    combined_bulk_data
                )

                # 4. copy shared information to the bulk dict
                for key in self.bypass_keys:
                    if key in data_dict:
                        this_bulk_dict[key] = deepcopy(data_dict[key])

            # 5. add flow match sampling
            this_bulk_dict = self._sample_location_and_conditional_flow(
                this_bulk_dict, output_prefix
            )

            list_of_dicts.append(this_bulk_dict)
        return list_of_dicts
