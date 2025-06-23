from collections import OrderedDict
from copy import deepcopy
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from bolero.tl.model.flow.matcher import (
    ConditionalFlowMatcher,
    ConstantFlowMatcher,
    FollmerProcessFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
)
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
#                ]
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

# pseudobulk_and_ot_info schema for Pseudobulker:
# {
#    'pseudobulk_records': dict of predefined coverage pseudobulk records
#    'meta_cell_emb': pd.DataFrame (n_meta_cell, emb_dim), # embedding of each meta cell
#    'meta_cell_n_frags': pd.Series (n_meta_cell,), # number of fragments for each meta cell
#    'target_cov': int, # target coverage for pseudobulk
#    'condition_to_related_pseudobulk': dict of condition to related pseudobulks mapping
#    'condition_emb': optional, instruct how to create pre-defined condition encoder,
#        if not provided, will use one hot encoder on condition_to_related_pseudobulk.keys()
#        if provided, it should contain all terms and posible values as provided in condition_terms
#        or metacell_condition_terms
#    'condition_terms': dict of dict, mapping condition name in cond_pairs into dict of terms.
#        example: {'ctrl_0h': {'cond': 'ctrl', 't': '0h'}, 'pert_1h': {'cond': 'pert', 't': '1h'},
#    'metacell_condition_terms': dict of series, value per meta cell,
#        it provides meta cell level condition terms, each pseudobulk can use its cluster_ids
#        to get the aggregated meta cell condition terms, it will be aggregated and add into
#        __conditionterms__ with cond_terms together currently, meta_cell conditon
#    'metacell_condition_terms_agg': dict of callable, optional, specify how to aggregate
#        meta cell condition terms. if not provided, will use mean.
# }

# Additional pseudobulk_and_ot_info key for PairedPseudobulker:
# {
#    'ot_transition': dict of transition matrix between conditions
# }

# Additional pseudobulk_and_ot_info key for EnsemblePairedPseudobulker:
# This class uses pseudobulk_records as p1,
# while p0 is randomly sampled from all available meta cells.
# {
#    'ensemble_whitelist': list of meta cell id to use for p0 pseudobulk,
#        if not provided, all meta cells will be used.
# }


def sample_mapping(tmat: pd.DataFrame, seed=None) -> pd.Series:
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
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    _tmat = torch.from_numpy(tmat.values.copy())
    src_names = tmat.index
    tgt_names = tmat.columns
    tgt_idx = torch.multinomial(_tmat, num_samples=1, generator=generator).squeeze(1)
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
    def __init__(self, cond_emb: dict[list | str] | list[str] = None):
        """
        Condition encoder that saves one-hot encoding for categorical conditions.
        Keep numerical conditions as is.

        Parameters
        ----------
        cond_emb : dict[np.ndarray], optional
            A dictionary of cond_key and cond_values.
            If cond_key is categoricals, cond_values should be possible categories.
            If cond_key is numerical, cond_values should be "CONTINUOUS:dims".
            If provided, conditions will not be used.
        """
        if isinstance(cond_emb, list):
            cond_emb = {"DEFAULT": cond_emb}

        self.encoder_dict = OrderedDict()

        sorted_keys = sorted(cond_emb.keys())
        for cond_key in sorted_keys:
            cond_values = cond_emb[cond_key]
            if (
                isinstance(cond_values, str)
                and cond_values.split(":")[0] == "CONTINUOUS"
            ):
                # numerical condition
                _, *dim = cond_values.split(":")
                if len(dim) == 0:
                    dim = 1
                else:
                    dim = int(dim[0])
                self.encoder_dict[cond_key] = dim
            else:
                # categorical condition
                cond_values = np.array(cond_values)
                if cond_values.ndim < 2:
                    cond_values = cond_values.reshape(-1, 1)
                _encoder = OneHotEncoder(dtype="float32", sparse_output=False)
                _encoder.fit(cond_values)
                self.encoder_dict[cond_key] = _encoder

        self.cond_emb_dims = OrderedDict()
        for k, v in self.encoder_dict.items():
            self.cond_emb_dims[k] = (
                v.categories_[0].size if isinstance(v, OneHotEncoder) else v
            )
        return

    def transform(self, cond: str | list[str] | dict) -> np.ndarray:
        """
        Transform the condition to the embedding.

        When condition has multiple terms, the result will be concatenated.
        """
        if not isinstance(cond, dict):
            # single condition, using default key
            cond = {"DEFAULT": cond}

        all_emb = []
        # make sure all_emb follows the order of self.encoder_dict
        for cond_key, encoder in self.encoder_dict.items():
            try:
                cond_values = cond[cond_key]
            except KeyError as e:
                raise KeyError(
                    f"Condition key '{cond_key}' not found in cond dictionary. "
                    f"Available keys: {list(cond.keys())}, encoder keys: {self.encoder_dict.keys()}"
                ) from e

            if not isinstance(cond_values, np.ndarray):
                cond_values = np.array(cond_values)
            if cond_values.ndim < 2:
                cond_values = cond_values.reshape(-1, 1)
            if isinstance(encoder, OneHotEncoder):
                emb = encoder.transform(cond_values)
            else:
                emb = cond_values.astype("float32")
            all_emb.append(emb)

        all_emb = np.concatenate(all_emb, axis=-1)
        return all_emb

    def split_cond_emb(
        self, cond_emb: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Split the condition embedding back to the original terms.

        Parameters
        ----------
        cond_emb : np.ndarray or torch.Tensor
            The condition embedding to split.

        Returns
        -------
        cond_dict : dict
            A dictionary of condition terms.
        """
        if (len(self.cond_emb_dims) == 1) and ("DEFAULT" in self.cond_emb_dims):
            # single condition term, return as is
            return cond_emb

        # multiple condition terms, split and return as dict
        cond_dict = {}
        start_idx = 0
        for cond_key in self.encoder_dict.keys():
            dim = self.cond_emb_dims[cond_key]
            end_idx = start_idx + dim
            cond_dict[cond_key] = cond_emb[..., start_idx:end_idx]
            start_idx = end_idx
        assert end_idx == cond_emb.shape[-1], "Condition embedding dimension mismatch."
        return cond_dict

    def __call__(self, *args, **kwds) -> np.ndarray:
        """
        Call the transform method.
        """
        return self.transform(*args, **kwds)


class PseudobulkerMixin:
    """
    Mixin class for pseudobulker to provide common methods.
    """

    default_config = {
        "pseudobulk_and_ot_info": "REQUIRED",
        "emb_key": "embedding",
        "downsample_pseudobulk": None,
        "barcode_order": None,
    }
    local_rng: np.random.Generator
    cov_key: str
    pseudobulk_records: dict[dict[str, Any]]
    emb_key: str
    pseudobulk_ids: pd.Index
    n_pids: int
    meta_cell_emb: pd.DataFrame
    metacell_id_to_int: pd.Series
    meta_cell_n_frags: pd.Series
    target_cov: int
    barcode_order: dict[str, dict[str, int]] | None
    prefix_order: list[str] | None
    prefix_name: str
    sampling_weights: pd.Series

    @classmethod
    def create_from_config(cls, **config):
        """Create the pseudobulk generator from configuration."""
        config = {k: v for k, v in config.items() if k in cls.default_config}
        validate_config(config, cls.default_config)
        pseudobulker = cls(**config)
        return pseudobulker

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

    def _prepare_pseudobulk_and_meta_cell_records(
        self,
        pseudobulk_and_ot_info,
        emb_key,
        downsample_pseudobulk,
        barcode_order,
        seed,
    ):
        self.local_rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.cov_key = "cov_scale"

        pseudobulk_and_ot_info = joblib.load(pseudobulk_and_ot_info)

        # Pre-computed pseudobulk records
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
        # collect meta cell idx information
        self.n_pids = self.pseudobulk_ids.size

        # Meta cell
        self.meta_cell_emb: pd.DataFrame = pseudobulk_and_ot_info["meta_cell_emb"]
        all_meta_cells = self.meta_cell_emb.index.tolist()
        self.metacell_id_to_int = pd.Series(
            range(len(all_meta_cells)), index=all_meta_cells
        )
        self.meta_cell_n_frags = pseudobulk_and_ot_info["meta_cell_n_frags"]

        # Target coverage
        self.target_cov = pseudobulk_and_ot_info["target_cov"]

        # Barcode order
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
        return pseudobulk_and_ot_info

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
            cond_pair_pseudobulks, pid_choice = self._sample_single_pseudobulk(
                skip_pids=used_pids
            )
            if cond_pair_pseudobulks is None:
                break
            used_pids.add(pid_choice)
            # cond_pair_pseudobulk: list of two pseudobulk records, source and target
            # cond_pair: tuple of two condition names
            records.append(cond_pair_pseudobulks)
        return records

    def _make_condition_encoder(self, cond_dict):
        if cond_dict is None:
            cond_dict = list(self.condition_to_related_pseudobulk.keys())

        encoder = PredefinedCondEncoder(cond_dict)
        return encoder

    def _prepare_condition_terms_and_emb(self, pseudobulk_and_ot_info: dict):
        self.condition_terms: dict[str, dict] = pseudobulk_and_ot_info.get(
            "condition_terms", {}
        )
        self.metacell_condition_terms: dict[str, pd.Series] = (
            pseudobulk_and_ot_info.get("metacell_condition_terms", {})
        )
        self.metacell_condition_terms_agg = pseudobulk_and_ot_info.get(
            "metacell_condition_terms_agg", {}
        )
        for term in self.metacell_condition_terms.keys():
            self.metacell_condition_terms_agg.setdefault(term, lambda x: x.mean())

        # 3. condition to related pseudobulk mapping
        self.condition_to_related_pseudobulk = pseudobulk_and_ot_info[
            "condition_to_related_pseudobulk"
        ]
        pid_to_cond = {}
        for cond, pids in self.condition_to_related_pseudobulk.items():
            for pid in pids:
                assert pid not in pid_to_cond, (
                    f"Duplicate pseudobulk {pid} found in multiple conditions: "
                    f"{pid_to_cond[pid]} and {cond}."
                )
                pid_to_cond[pid] = cond
        self.pseudobulk_id_to_condition = pd.Series(
            pid_to_cond, index=self.pseudobulk_ids
        )

        cond_emb = pseudobulk_and_ot_info.get("condition_emb", None)
        for k in self.metacell_condition_terms.keys():
            cond_emb[k] = "CONTINUOUS"
        self.condition_encoder = self._make_condition_encoder(cond_emb)
        self.cond_emb_dims = self.condition_encoder.cond_emb_dims
        return

    def _add_cond_term_and_emb_to_pseudobulk(
        self, pseudobulk_rec: dict[str, Any], cond: str
    ):
        cond_terms = self.condition_terms.get(cond, cond)
        # aggregate meta cell condition terms (if any)
        cids = pseudobulk_rec["cluster_ids"][self.prefix_name]
        for term, term_series in self.metacell_condition_terms.items():
            agg_func = self.metacell_condition_terms_agg.get(term)
            cid_values = term_series.reindex(cids)
            agg_value = agg_func(cid_values)
            if np.isnan(agg_value):
                print(pseudobulk_rec)
                raise ValueError(
                    f"Condition term {term} has NaN value in {cond} pseudobulk, "
                    "please check the condition terms and meta cell records."
                )
            try:
                cond_terms[term] = agg_value
            except Exception as e:
                print(cond_terms)
                print(agg_value)
                print(term)
                print(pseudobulk_rec)
                raise e
        pseudobulk_rec["__conditionemb__"] = self.condition_encoder(cond_terms)
        pseudobulk_rec["__conditionterms__"] = cond_terms
        return pseudobulk_rec


class PairedPseudobulker(PseudobulkerMixin):
    """
    Generate paired pseudobulks from predefined pseudobulk records,
    condition pairs, and pre-computed OT transition matrix between each condition pair.
    """

    def __init__(
        self,
        pseudobulk_and_ot_info: Union[str, dict],
        emb_key: str = "embedding",
        downsample_pseudobulk: int = None,
        barcode_order: Union[str, list] = None,
        seed=42,
    ):
        """
        Load pseudobulk records and prepare pseudobulk embedding and OT data.

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
        # 1. pseudobulk dict records and meta cell information
        pseudobulk_and_ot_info = self._prepare_pseudobulk_and_meta_cell_records(
            pseudobulk_and_ot_info=pseudobulk_and_ot_info,
            emb_key=emb_key,
            downsample_pseudobulk=downsample_pseudobulk,
            barcode_order=barcode_order,
            seed=seed,
        )

        # 2. OT transition matrix between conditions
        # schema: {
        #     (source_condition, target_condition):
        #         pd.DataFrame (n_source_meta_cell, n_target_meta_cell)
        # }
        self.ot_transition: dict[str, pd.DataFrame] = pseudobulk_and_ot_info[
            "ot_transition"
        ]
        self.condition_pairs = list(self.ot_transition.keys())
        self.conditions = set()
        for cond_pair in self.condition_pairs:
            self.conditions.update(cond_pair)

        self._prepare_condition_terms_and_emb(pseudobulk_and_ot_info)
        return

    def _sample_single_pseudobulk(
        self,
        skip_pids=None,
        cond_pair=None,
        p_sample=None,
        pid_choice=None,
        ot_seed=None,
    ):
        """
        Sample a single pseudobulk pair from the predefined pseudobulk records.

        Parameters
        ----------
        skip_pids : set, optional
            A set of pseudobulk IDs to skip when sampling.
        cond_pair : tuple, optional
            A tuple of two condition names to use. If None, a random condition pair will be sampled.
        p_sample : int, optional
            The index of the condition to use (0 or 1). If None, a random condition will be sampled.
        pid_choice : str, optional
            A specific pseudobulk ID to use. If None, a random pseudobulk will be sampled.

        Returns
        -------
        cond_pair_pseudobulks : list[dict]
        """
        # 1. For a condition pair, get the transition matrix
        if cond_pair is None:
            cond_pair = tuple(self.local_rng.choice(self.condition_pairs))
        tmat = self.ot_transition[cond_pair]
        cond_pair_pseudobulks = [None, None]

        # 2. select a predefined pseudobulk from either source or target condition
        if p_sample is None:
            p_sample = self.local_rng.choice([0, 1])
        related_pseudobulks = set(
            self.condition_to_related_pseudobulk[cond_pair[p_sample]]
        )
        if skip_pids is not None:
            related_pseudobulks -= set(skip_pids)
        related_pseudobulks = list(related_pseudobulks)

        if len(related_pseudobulks) == 0:
            return None, None
        related_sample_weights = self.sampling_weights.reindex(
            related_pseudobulks
        ).dropna()
        related_sample_weights /= related_sample_weights.sum()
        if pid_choice is None:
            pid_choice = self.local_rng.choice(
                related_pseudobulks, 1, replace=False, p=related_sample_weights.values
            )[0]
        sel_pseudobulk = deepcopy(self.pseudobulk_records[pid_choice])
        sel_pseudobulk.setdefault("__t__", p_sample)
        cond_pair_pseudobulks[p_sample] = sel_pseudobulk

        # 3. sample a matched meta cell list from the OT transition matrix
        if p_sample == 1:
            tmat = tmat.T
        p_ot = 0 if p_sample == 1 else 1
        mapping = sample_mapping(tmat, seed=ot_seed)
        p_ot_meta_cells = mapping.loc[
            sel_pseudobulk["cluster_ids"][self.prefix_name]
        ].values.tolist()

        # 4. create a new pseudobulk record for the OT selected pseudobulk
        n_frags = self.meta_cell_n_frags.loc[p_ot_meta_cells].sum()
        pad_emb_to_n = sel_pseudobulk["embedding_multi"].shape[0]
        ot_pseudobulk = {
            "cluster_ids": {self.prefix_name: p_ot_meta_cells},
            "n_frags": {self.prefix_name: n_frags},
            "cov_scale": {self.prefix_name: np.log2(n_frags / self.target_cov)},
            # CAUTION: current pseudobulk records use the first meta cell's
            # embedding as embedding key, NOT mean embedding
            "embedding": self.meta_cell_emb.loc[p_ot_meta_cells].values[0],
            "embedding_multi": pad_or_chunk_emb(
                self.meta_cell_emb.loc[p_ot_meta_cells].values, pad_emb_to_n
            ),
            "sample_weight": sel_pseudobulk["sample_weight"],
        }
        ot_pseudobulk.setdefault("__t__", p_ot)
        cond_pair_pseudobulks[p_ot] = ot_pseudobulk

        # Following code is for generator to handle the pseudobulk records
        for d, cond in zip(cond_pair_pseudobulks, cond_pair):
            # 1. set the embedding for generator to use
            # add cell embedding and coverage
            d["__embedding__"] = d[self.emb_key]
            d["__covlogfc__"] = d[self.cov_key]
            # add condition embedding
            d = self._add_cond_term_and_emb_to_pseudobulk(d, cond)

            # 2. convert meta cell list to int row index
            if self.barcode_order is not None:
                parquet_prefix_to_rows: dict = d["cluster_ids"]
                parquet_prefix_to_rows = {
                    k: sorted([self.barcode_order[k][c] for c in v])
                    for k, v in parquet_prefix_to_rows.items()
                }
                meta_cell_index: np.array = self.metacell_id_to_int.loc[
                    d["cluster_ids"][self.prefix_name]
                ].values
                d["meta_cell_index"] = meta_cell_index
                d["cluster_ids"] = parquet_prefix_to_rows

            # self.pseudobulk_records[pid][f'{cov_key}_list'] = cov_value_list
            # self.pseudobulk_records[pid]['parquet_prefix_to_int_rows'] = parquet_prefix_to_rows
            # cov_value_list = [data[cov_key][prefix] for prefix in self.prefix_order]
        return cond_pair_pseudobulks, pid_choice

    def create_pseudobulk_records_from_design(
        self, designs: list[tuple]
    ) -> dict[str, dict]:
        """
        Create pseudobulk records from a design.

        Parameters
        ----------
        designs : list[tuple]
            A list of tuples, where each tuple contains:
            - cond_pair: tuple of two condition names to compare (e.g., ("control", "treatment")).
                Pairs should belong to self.condition_pairs, and unique among all designs.
            - p_sample: index of the condition to sample from (0 or 1)
            - n_pids: number of pseudobulks to sample from the condition pair
            Full example: [
                (("control", "treatment1"), 0, 5),
                (("control", "treatment2"), 0, 5),
            ]
            This example will create 10 * 2 = 20 pseudobulks in total.

        Returns
        -------
        pseudobulk_col : dict
            A dictionary where keys are pseudobulk names and values are the corresponding pseudobulk records.
        Each pseudobulk name is formatted as "{name0}|{name1}:{name0}-{idx}" and "{name0}|{name1}:{name1}-{idx}".
        """
        pseudobulk_col = OrderedDict()
        for cond_pair, p_sample, n_pids in designs:
            sample_from_cond = cond_pair[p_sample]
            related_pids = pd.Series(
                list(self.condition_to_related_pseudobulk[sample_from_cond])
            ).sort_values()
            if n_pids > related_pids.size:
                n_pids = related_pids.size
                print(
                    f"{sample_from_cond} only has {related_pids.size} pseudobulks, will use all of them."
                )
                pids = related_pids.tolist()
            else:
                pids = related_pids.sample(n_pids, random_state=0).tolist()

            for idx, pid in enumerate(pids):
                (p0, p1), _ = self._sample_single_pseudobulk(
                    skip_pids=None,
                    cond_pair=cond_pair,
                    p_sample=p_sample,
                    pid_choice=pid,
                    ot_seed=0,
                )
                name0, name1 = cond_pair

                p0_key = f"{name0}|{name1}:{name0}-{idx}"
                assert (
                    p0_key not in pseudobulk_col
                ), f"Duplicate key {p0_key} found in pseudobulk_col."
                p0.setdefault("__t__", 0)
                pseudobulk_col[p0_key] = p0

                p1_key = f"{name0}|{name1}:{name1}-{idx}"
                assert (
                    p1_key not in pseudobulk_col
                ), f"Duplicate key {p1_key} found in pseudobulk_col."
                p1.setdefault("__t__", 1)
                pseudobulk_col[p1_key] = p1
        return pseudobulk_col


class EnsemblePairedPseudobulker(PseudobulkerMixin):
    """
    Generate paired pseudobulks from predefined pseudobulk records,
    the predefined pseudobulk will be used as p1, while p0 will be randomly sampled from all available pseudobulks.
    """

    def __init__(
        self,
        pseudobulk_and_ot_info: Union[str, dict],
        emb_key: str = "embedding",
        downsample_pseudobulk: int = None,
        barcode_order: Union[str, list] = None,
        seed=42,
        p0_n_meta_cells=1000,
    ):
        """
        Load pseudobulk records and prepare pseudobulk embedding data.

        Parameters
        ----------
        pseudobulk_records (dict[str, dict]):
            The prefix name (in ray dataset) to VQ records file path mapping.
        emb_key (str): The key to use for the embedding.
        downsample_pseudobulk (int): The number of pseudobulks to downsample to.
        barcode_order (dict): The order of barcodes for each prefix.
        flow_match_sigma (float): The sigma for the flow matcher.
        seed (int): The random seed for sampling.
        p0_n_meta_cells (int): The number of meta cells to sample for p0 pseudobulk.
            If fix_p0_meta_cells is provided, this will be ignored.
        """
        pseudobulk_and_ot_info = self._prepare_pseudobulk_and_meta_cell_records(
            pseudobulk_and_ot_info=pseudobulk_and_ot_info,
            emb_key=emb_key,
            downsample_pseudobulk=downsample_pseudobulk,
            barcode_order=barcode_order,
            seed=seed,
        )

        self.p0_n_meta_cells = min(p0_n_meta_cells, self.meta_cell_emb.shape[0])
        self.fix_p0_meta_cells = list(
            pseudobulk_and_ot_info.get("fixed_p0_meta_cells", None)
        )
        self.ensemble_whitelist = list(
            pseudobulk_and_ot_info.get(
                "ensemble_whitelist", self.meta_cell_emb.index.copy()
            )
        )

        if "condition_to_related_pseudobulk" in pseudobulk_and_ot_info:
            # pseudobulk has condition information
            self.conditions = set(
                pseudobulk_and_ot_info["condition_to_related_pseudobulk"].keys()
            )
            self._prepare_condition_terms_and_emb(pseudobulk_and_ot_info)
        return

    def _sample_single_pseudobulk(
        self,
        skip_pids=None,
        pid_choice=None,
    ):
        """
        Sample a single pseudobulk pair from the predefined pseudobulk records.

        Parameters
        ----------
        skip_pids : set, optional
            A set of pseudobulk IDs to skip when sampling.
        pid_choice : str, optional
            A specific pseudobulk ID to use. If None, a random pseudobulk will be sampled.

        Returns
        -------
        cond_pair_pseudobulks : list[dict]
            A list of two pseudobulk records, p0 and p1.
        """
        if pid_choice is None:
            skip_pids = set() if skip_pids is None else set(skip_pids)
            use_pids = set(self.pseudobulk_ids) - skip_pids
            # TODO: use sampling weights to sample pids
            pid_choice = self.local_rng.choice(list(use_pids))

        p1_pseudobulk = self.take_by_name(pid_choice)
        p1_pseudobulk.setdefault("__t__", 1)
        p1_cell_emb = p1_pseudobulk[self.emb_key]

        if hasattr(self, "condition_encoder"):
            p1_cond = self.pseudobulk_id_to_condition[pid_choice]
            p1_pseudobulk = self._add_cond_term_and_emb_to_pseudobulk(
                p1_pseudobulk, p1_cond
            )

        if self.fix_p0_meta_cells is None:
            p0_meta_cells = self.local_rng.choice(
                self.ensemble_whitelist,
                size=min(self.p0_n_meta_cells, len(self.ensemble_whitelist)),
                replace=False,
            ).tolist()
        else:
            p0_meta_cells = self.fix_p0_meta_cells
        n_frags = self.meta_cell_n_frags.loc[p0_meta_cells].sum()
        cov_scale = np.log2(n_frags / self.target_cov)

        # use mean emb as p0 cell emb
        p0_embedding = self.meta_cell_emb.loc[p0_meta_cells].mean().values
        p0_p1_embedding = np.concatenate([p0_embedding, p1_cell_emb], axis=-1)
        # IMPORTANT: flow model use embedding of p0, and condition embedding of p1
        # Here we concat p0 and p1 embeddings as final embedding
        # we put p0_p1_embedding into both p0 and p1 to make sure they are in the same shape
        p1_pseudobulk[self.emb_key] = p0_p1_embedding

        p0_pseudobulk = {
            "cluster_ids": {self.prefix_name: p0_meta_cells},
            "n_frags": {self.prefix_name: n_frags},
            "cov_scale": {self.prefix_name: cov_scale},
            self.emb_key: p0_p1_embedding,
        }
        if hasattr(self, "condition_encoder"):
            # copy p1 condition embedding to p0
            p0_pseudobulk["__conditionemb__"] = p1_pseudobulk["__conditionemb__"]
            p0_pseudobulk["__conditionterms__"] = p1_pseudobulk["__conditionterms__"]

        p0_pseudobulk.setdefault("__t__", 0)

        cond_pair_pseudobulks = [p0_pseudobulk, p1_pseudobulk]
        for d in cond_pair_pseudobulks:
            # 1. set the embedding for generator to use
            d["__embedding__"] = d[self.emb_key]
            d["__covlogfc__"] = d[self.cov_key]

            # 2. convert meta cell list to int row index
            if self.barcode_order is not None:
                parquet_prefix_to_rows: dict = d["cluster_ids"]
                parquet_prefix_to_rows = {
                    k: sorted([self.barcode_order[k][c] for c in v])
                    for k, v in parquet_prefix_to_rows.items()
                }
                meta_cell_index: np.array = self.metacell_id_to_int.loc[
                    d["cluster_ids"][self.prefix_name]
                ].values
                d["meta_cell_index"] = meta_cell_index
                d["cluster_ids"] = parquet_prefix_to_rows
        return cond_pair_pseudobulks, pid_choice

    def create_pseudobulk_records_from_design(self, designs) -> dict[str, dict]:
        """
        Create pseudobulk pairs for each pseudobulk in the self.pseudobulk_records.
        """
        if designs is not None:
            use_pids = pd.Index(designs).intersection(self.pseudobulk_ids)
        else:
            use_pids = self.pseudobulk_ids

        # assume designs is None or is a list of pids to use
        pseudobulk_col = OrderedDict()

        for idx, pid in enumerate(use_pids):
            (p0, p1), _ = self._sample_single_pseudobulk(skip_pids=None, pid_choice=pid)
            p0_key = f"ensemble|data:ensemble-{idx}"
            assert (
                p0_key not in pseudobulk_col
            ), f"Duplicate key {p0_key} found in pseudobulk_col."
            p0.setdefault("__t__", 0)
            p0["__pid__"] = pid
            pseudobulk_col[p0_key] = p0

            p1_key = f"ensemble|data:data-{idx}"
            assert (
                p1_key not in pseudobulk_col
            ), f"Duplicate key {p1_key} found in pseudobulk_col."
            p1.setdefault("__t__", 1)
            p1["__pid__"] = pid
            pseudobulk_col[p1_key] = p1

        assert (
            len(pseudobulk_col) != 0
        ), f"no pseudobulk records after design, make sure designs is a list of pid to use; designs: {designs}"
        return pseudobulk_col


PAIRED_PSEUDOBULKER_CLS_DICT = {
    "condition": PairedPseudobulker,
    "ensemble": EnsemblePairedPseudobulker,
}


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
        flow_matcher_class="cfm",
        flow_matcher_kwargs=None,
        **name_to_pseudobulker,
    ):
        self.name_to_pseudobulker = name_to_pseudobulker
        self.n_pseudobulks = n_pseudobulks
        self.return_rows = return_rows
        self.inplace = inplace

        flow_matcher_kwargs = {} if flow_matcher_kwargs is None else flow_matcher_kwargs
        if flow_matcher_class == "cfm":
            cfm_cls = ConditionalFlowMatcher
        elif flow_matcher_class == "sb":
            cfm_cls = SchrodingerBridgeConditionalFlowMatcher
        elif flow_matcher_class == "fp":
            cfm_cls = FollmerProcessFlowMatcher
        elif flow_matcher_class == "constant":
            cfm_cls = ConstantFlowMatcher
        else:
            raise ValueError(
                f"Unknown flow_matcher_class: {flow_matcher_class}. "
                "Supported classes are 'cfm', 'sb', 'fp', and 'constant'."
            )
        self.flow_matcher = cfm_cls(**flow_matcher_kwargs)

        self.bypass_keys = ["region"]
        if bypass_keys is not None:
            if bypass_keys is str:
                self.bypass_keys.append(bypass_keys)
            else:
                self.bypass_keys.extend(list(bypass_keys))
        self.normalize_cov = normalize_cov
        self.reduce_resolution = reduce_resolution

        # suffix for p0 and p1 data keys
        self.suffix = ["_0", "_1"]
        return

    def _reduce_resolution(self, data):
        resolution = self.reduce_resolution
        # from (1, seq_len) to (1, seq_len // resolution) by summing
        data = data.reshape(1, -1, resolution).sum(axis=-1)
        return data

    def _sample_location_and_conditional_flow(self, data_dict, output_prefix):
        x0 = data_dict[f"{output_prefix}:bulk_data_0"]
        x1 = data_dict[f"{output_prefix}:bulk_data_1"]
        t_start = data_dict.get("__t_0", 0)
        t_end = data_dict.get("__t_1", 1)

        x0 = torch.from_numpy(x0)
        x1 = torch.from_numpy(x1)

        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
            x0=x0, x1=x1, t=None, return_noise=False
        )
        # scale t to the range [t_start, t_end]
        # default trange is [0, 1], so t is unchanged
        t = t_start + t * (t_end - t_start)

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
                if "__conditionemb__" in pseudobulk:
                    this_bulk_dict[f"{output_prefix}:condition_emb{suffix}"] = (
                        pseudobulk["__conditionemb__"]
                    )

                # 2. add pseudobulk embedding
                row_embedding = pseudobulk["__embedding__"]
                this_bulk_dict[f"{output_prefix}:embedding_data{suffix}"] = (
                    row_embedding
                )

                # 3. add trange if available
                this_bulk_dict[f"__t{suffix}"] = pseudobulk["__t__"]

                # 4. add pseudobulk data with optional
                # coverage normalization and resolution reduction
                prefix_to_rows = pseudobulk["cluster_ids"]
                cov_logfc = pseudobulk["__covlogfc__"]
                combined_bulk_data = []
                for prefix in pseudobulker.prefix_order:
                    prefix_rows = prefix_to_rows[prefix]
                    # row_by_base is a csr_matrix of shape (n_rows, region_length)
                    try:
                        row_by_base: csr_matrix = data_dict[prefix]
                        assert isinstance(
                            row_by_base, csr_matrix
                        ), f"Expected csr_matrix for prefix {prefix}, got {type(row_by_base)}"
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

                # 5. copy shared information to the bulk dict
                for key in self.bypass_keys:
                    if key in data_dict:
                        this_bulk_dict[key] = deepcopy(data_dict[key])

            # 6. add flow match sampling
            this_bulk_dict = self._sample_location_and_conditional_flow(
                this_bulk_dict, output_prefix
            )

            list_of_dicts.append(this_bulk_dict)
        return list_of_dicts
