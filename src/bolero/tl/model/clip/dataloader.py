import pathlib
import random

import joblib
import numpy as np
import pandas as pd
import ray
import torch
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding
from esm.utils.misc import stack_variable_length_tensors
from tqdm import tqdm


def _filter_3did_ddi_table(ddi_pdb, max_chain_size):
    # this filter does not change unique count of ddi_id, it just remove some outlier domain and broken index
    ddi_pdb["chain1_size"] = ddi_pdb["chain1_end"] - ddi_pdb["chain1_start"]
    ddi_pdb["chain2_size"] = ddi_pdb["chain2_end"] - ddi_pdb["chain2_start"]
    ddi_pdb = ddi_pdb[
        (ddi_pdb["chain1_size"] > 0)
        & (ddi_pdb["chain2_size"] > 0)
        & (ddi_pdb["chain1_size"] < max_chain_size)
        & (ddi_pdb["chain2_size"] < max_chain_size)
    ].copy()
    return ddi_pdb


def _add_start_end(table, count_col):
    table["chain_end"] = table[count_col].cumsum()
    table["chain_start"] = np.concatenate(
        [np.array([0]), table["chain_end"].values[:-1]]
    )
    return


@ray.remote
class BatchSeqGenerator:
    def __init__(
        self,
        pdb_chain_size_path,
        pdb_residue_token_path,
        pdb_codebook_path,
    ):
        self.pdb_chain_size = pd.read_feather(pdb_chain_size_path)
        _add_start_end(self.pdb_chain_size, "size")
        self.pdb_residue_token = np.load(pdb_residue_token_path)["data"]
        self.pdb_codebook = joblib.load(pdb_codebook_path)
        self.pdb_decodebook = {v: k for k, v in self.pdb_codebook.items()}

    def _get_chain_seq(
        self,
        pdb_id,
        chain_id,
        domain_start,
        domain_end,
    ):
        hits = self.pdb_chain_size.query(
            f"(pdb_id == '{pdb_id}') & (chain == '{chain_id}')"
        )[["chain_start", "chain_end"]]
        if hits.shape[0] != 1:
            return None
        else:
            cstart, cend = hits.iloc[0].values
        # (chain_size, 2), columns are [residue_id, residue_code]
        # residue_id is ordered, but may not start from 0, so we use compare to select
        chain_data = self.pdb_residue_token[cstart:cend]

        chain_seq = "".join(
            pd.Index(
                chain_data[
                    (chain_data[:, 0] >= domain_start)
                    & (chain_data[:, 0] <= domain_end)
                ][:, 1]
            )
            .map(self.pdb_decodebook)
            .values
        )
        return chain_seq

    def get_batch_seqs(self, batch_records):
        """Get sequences for a batch of ddi_pdb records"""
        chain1_seqs = []
        chain2_seqs = []
        ids = []
        for _, record in batch_records.iterrows():
            ddi_id = record["ddi_id"]
            pdb_id = record["pdb_id"]
            chain1 = record["chain1"]
            chain2 = record["chain2"]
            cs1 = record["chain1_start"]
            ce1 = record["chain1_end"]
            cs2 = record["chain2_start"]
            ce2 = record["chain2_end"]

            chain1_seq = self._get_chain_seq(
                pdb_id,
                chain1,
                cs1,
                ce1,
            )
            chain2_seq = self._get_chain_seq(
                pdb_id,
                chain2,
                cs2,
                ce2,
            )
            if chain1_seq is None or chain2_seq is None:
                continue
            chain1_seqs.append(chain1_seq)
            chain2_seqs.append(chain2_seq)
            ids.append((ddi_id, pdb_id, chain1, chain2))
        return {"ids": ids, "chain1": chain1_seqs, "chain2": chain2_seqs}


class DDIDataloader:
    def __init__(
        self,
        did_pdb_table,
        pdb_chain_size_path,
        pdb_residue_token_path,
        pdb_codebook_path,
        mmcif_dir,
        pdb_chain_contact_count_path=None,
        pdb_chain_contacts_path=None,
        max_chain_size=512,
        device="cuda",
        num_workers=10,
    ):
        ddi_pdb = pd.read_feather(did_pdb_table)
        self.ddi_pdb = _filter_3did_ddi_table(ddi_pdb, max_chain_size=max_chain_size)

        self.max_chain_size = max_chain_size
        self.esm_tokenizer = EsmSequenceTokenizer(clean_up_tokenization_spaces=True)

        # chain and residue token
        self.seq_generators = [
            BatchSeqGenerator.remote(
                pdb_chain_size_path,
                pdb_residue_token_path,
                pdb_codebook_path,
            )
            for _ in range(num_workers)
        ]
        self.num_workers = num_workers

        # chain contacts
        if pdb_chain_contact_count_path is not None:
            self.use_contacts = True
            assert (
                pdb_chain_contacts_path is not None
            ), "pdb_chain_contacts_path is None"
            self.pdb_chain_contact_count = pd.read_feather(pdb_chain_contact_count_path)
            _add_start_end(self.pdb_chain_contact_count, "contact_count")
            self.pdb_chain_contacts = np.load(
                pdb_chain_contacts_path,
                allow_pickle=True,
            )["data"]
        else:
            self.use_contacts = False
            self.pdb_chain_contact_count = None
            self.pdb_chain_contacts = None

        # mmcif
        self.mmcif_dir = mmcif_dir
        self.pdb_mmcif_paths = {
            p.name.split(".")[0]: str(p) for p in pathlib.Path(mmcif_dir).glob("*.cif*")
        }
        self.ddi_pdb = self.ddi_pdb[
            self.ddi_pdb["pdb_id"].isin(self.pdb_mmcif_paths.keys())
        ].copy()

        self.device = (
            torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        )
        return

    def _get_chain_contacts(
        self,
        pdb_id,
        chain1_id,
        chain2_id,
        domain1_start,
        domain1_end,
        domain2_start,
        domain2_end,
    ):
        pdb_chain_contact_count = self.pdb_chain_contact_count
        pdb_chain_contacts = self.pdb_chain_contacts

        pdb_sel = f"(pdb_id == '{pdb_id}')"
        sel1 = f"(chain1 == '{chain1_id}') & (chain2 == '{chain2_id}')"
        sel2 = f"(chain1 == '{chain2_id}') & (chain2 == '{chain1_id}')"
        pair_record = pdb_chain_contact_count.query(f"{pdb_sel} & ({sel1} | {sel2})")
        if pair_record.shape[0] < 1:
            return None
        else:
            pair_record = pair_record.iloc[0]
        if pair_record["chain1"] != chain2_id:
            # flip values
            domain1_start, domain2_start = domain2_start, domain1_start
            domain1_end, domain2_end = domain2_end, domain1_end

        cstart, cend = pair_record[["chain_start", "chain_end"]].values
        # (chain_size, 2), columns are [residue_id, residue_code]
        # residue_id is ordered, but may not start from 0, so we use compare to select
        chain_data = pdb_chain_contacts[cstart:cend]
        domain1_sel = (chain_data[:, 0] >= domain1_start) & (
            chain_data[:, 0] <= domain1_end
        )
        domain2_sel = (chain_data[:, 1] >= domain2_start) & (
            chain_data[:, 1] <= domain2_end
        )
        row_sel = domain1_sel & domain2_sel
        chain_contacts = chain_data[row_sel][:, 2]
        return chain_contacts

    def iter_batches(self, batch_size, nbatch=None):
        """
        Iterate over batches of ddi_pdb table
        """
        ddi_pdb = self.ddi_pdb
        num_batches = int(np.ceil(ddi_pdb.shape[0] / batch_size))
        if nbatch is not None:
            num_batches = min(num_batches, nbatch)

        refs = []
        for i in range(num_batches):
            batch_records = ddi_pdb.iloc[i * batch_size : (i + 1) * batch_size]
            actor = self.seq_generators[i % self.num_workers]
            refs.append(actor.get_batch_seqs.remote(batch_records))

        while refs:
            ready_refs, refs = ray.wait(refs, num_returns=1)
            for ref in ready_refs:
                batch = ray.get(ref)
                yield batch

    def dump_feather(self, output_prefix):
        """Dump single feather file for all pairs of ddi_pbd with sequences."""
        batch_size = 64

        batch_col = []
        chunk_paths = []
        n = self.ddi_pdb.shape[0] // batch_size + 1
        for idx, batch in tqdm(
            enumerate(self.iter_batches(batch_size, tokenize=False, shuffle=False)),
            total=n,
        ):
            batch_df = pd.DataFrame(
                batch["ids"], columns=["ddi_id", "pdb_id", "chain1", "chain2"]
            )
            batch_df["chain1_seq"] = pd.Series(batch["chain1"])
            batch_df["chain2_seq"] = pd.Series(batch["chain2"])
            batch_col.append(batch_df)
            if len(batch_col) % 100 == 0:
                batch_col = pd.concat(batch_col)
                chunk_path = f"{output_prefix}.{idx}.feather"
                chunk_paths.append(chunk_path)
                batch_col.to_feather(chunk_path)
                batch_col = []

        final_col = []
        for path in chunk_paths:
            df = pd.read_feather(path)
            final_col.append(df)
        final_col = pd.concat(final_col).reset_index(drop=True)
        final_col.to_feather(f"{output_prefix}.ddi_pdb_records.feather")

        for path in chunk_paths:
            pathlib.Path(path).unlink()
        return


class TokenizerMixin:
    def __init__(self, device):
        self.esm_tokenizer = EsmSequenceTokenizer(
            clean_up_tokenization_spaces=True,
        )
        self.device = device

    def _tokenize(self, sequence: list[str]) -> torch.Tensor:
        pad = self.esm_tokenizer.pad_token_id
        assert pad is not None
        return stack_variable_length_tensors(
            [
                encoding.tokenize_sequence(
                    x, self.esm_tokenizer, add_special_tokens=True
                )
                for x in sequence
            ],
            constant_value=pad,
        ).to(self.device)


def _get_device(device: str):
    return torch.device(device) if torch.cuda.is_available() else torch.device("cpu")


def random_batch_id_sampler(data, batch_size=64):
    """
    Generator that yields batches of items of size batch_size from a dictionary of lists.

    It maintains a separate list of keys (for efficiency) and updates it whenever a key is removed.
    For each batch:
      - Randomly selects a key from the maintained keys list.
      - Determines the number of items needed (batch_size minus current batch length).
      - If the selected key's list has more than the needed items, it takes exactly the needed items
        and updates that list.
      - If the key's list has fewer or equal items, it takes all of them and removes the key from
        both the dictionary and the keys list.

    Yields
    ------
        A list of exactly batch_size items.
    """
    keys = list(data.keys())

    batch = set()
    while keys:
        while len(batch) < batch_size and keys:
            # Choose a random key from the maintained keys list.
            key = random.choice(keys)
            needed = batch_size - len(batch)
            current_list = data[key]

            if len(current_list) > needed:
                # If more items are available than needed, take only the needed amount.
                batch.update(current_list[:needed])
                new_list = current_list[needed:]
                data[key] = new_list
            else:
                # If the list doesn't have enough, take all of them and remove the key.
                batch.update(current_list)
                del data[key]
                keys.remove(key)

        # Yield exactly batch_size items.
        batch_list = list(batch)
        yield batch_list[:batch_size]
        batch = set(batch_list[batch_size:])


class FeatherDataset(TokenizerMixin):
    def __init__(
        self,
        data,
        train_folds,
        eval_folds,
        pdb_go_tax_path=None,
        max_len=512,
        concat_dim="batch",
        device="cuda",
    ):
        self.device = _get_device(device)
        if isinstance(data, (str, pathlib.Path)):
            data = pd.read_feather(data)
        self.ddi_pdb = data

        seq1_filter = self.ddi_pdb["chain1_seq"].map(lambda s: len(s)) <= max_len
        seq2_filter = self.ddi_pdb["chain2_seq"].map(lambda s: len(s)) <= max_len
        self.ddi_pdb = self.ddi_pdb[seq1_filter & seq2_filter].copy()

        self.ddi_pdb_train = self.ddi_pdb[
            self.ddi_pdb["fold_id"].isin(train_folds)
        ].reset_index(drop=True)
        self.ddi_pdb_eval = self.ddi_pdb[
            self.ddi_pdb["fold_id"].isin(eval_folds)
        ].reset_index(drop=True)

        TokenizerMixin.__init__(self, device=self.device)
        self.max_len = max_len
        self.concat_dim = concat_dim

        if pdb_go_tax_path is not None:
            table = pd.read_feather(pdb_go_tax_path)
            self.pdb_go_tax = table[
                table["PDB"].isin(self.ddi_pdb["pdb_id"])
            ].reset_index(drop=True)
        else:
            self.pdb_go_tax = None

        self.training = True

    def train(self):
        """Train mode"""
        self.training = True

    def eval(self):
        """Eval mode"""
        self.training = False

    @staticmethod
    def _sample_epoch_ddi_pdb_table(table):
        """Sample one interaction record for each (ddi_id, pdb_id) pair"""
        use_rows = []
        id_table = table[["ddi_id", "pdb_id"]].copy()
        for _, df in id_table.groupby(["ddi_id", "pdb_id"], observed=True):
            if df.shape[0] == 1:
                use_rows.append(df.index[0])
            else:
                use_rows.append(np.random.choice(df.index))
        # shuffle rows
        np.random.shuffle(use_rows)
        use_ddi_pdb = table.loc[pd.Index(use_rows)].reset_index(drop=True)
        return use_ddi_pdb

    def _get_pdb_groups(self, ddi_pdb):
        pdb_go_tax = self.pdb_go_tax[
            self.pdb_go_tax["PDB"].isin(ddi_pdb["pdb_id"])
        ].copy()
        pdb_go_tax = pdb_go_tax.sample(frac=1, replace=False)
        pdb_groups = {
            (tax, go): df["PDB"].unique()
            for (tax, go), df in tqdm(
                pdb_go_tax.groupby(["TAX_ID", "GO_ID"], observed=True),
                desc="PDB ID Grouping",
            )
        }
        return pdb_groups

    def tokenize_batch(self, batch_df):
        """
        Tokenize a batch of protein sequence pairs, chain1, chain2 are concatenated

        Output shape (batch_size * 2, max_chain_size + 2)

        To split the batch back to chain1 and chain2, use torch.split(batch_tensor, batch_size)
        """
        match self.concat_dim:
            case "seq_len":
                seq_concat = (
                    batch_df["chain1_seq"] + "<eos>|<cls>" + batch_df["chain2_seq"]
                )
                seq_list = seq_concat.tolist()
            case "batch":
                seq_list = (
                    batch_df["chain1_seq"].tolist() + batch_df["chain2_seq"].tolist()
                )
            case _:
                raise ValueError(f"Invalid concat_dim: {self.concat_dim}")

        batch_tensor = self._tokenize(seq_list)
        return batch_tensor

    def _iter_records(self, ddi_pdb, batch_size):
        """
        Global random sample
        """
        num_batches = int(np.ceil(len(ddi_pdb) / batch_size))
        for i in range(num_batches):
            batch_records = ddi_pdb.iloc[i * batch_size : (i + 1) * batch_size]
            if batch_records.shape[0] < batch_size:
                break
            yield batch_records

    def _iter_records_with_group(self, ddi_pdb, batch_size):
        """
        Sample batch records that is grouped by GO and TAX,
        therefore prot seq are more likely coming from the same species and function group.
        """
        pdb_groups = self._get_pdb_groups(ddi_pdb)

        pdb_id_batches = list(
            tqdm(
                random_batch_id_sampler(pdb_groups, batch_size=batch_size),
                "PDB ID Batch",
            )
        )
        random.shuffle(pdb_id_batches)

        batch_rows = []
        batch_row_count = 0
        for pdb_id_batch in tqdm(pdb_id_batches):
            use_rows = ddi_pdb[ddi_pdb["pdb_id"].isin(pdb_id_batch)].drop_duplicates(
                subset=["ddi_id"]
            )
            if use_rows.shape[0] > batch_size:
                use_rows = use_rows.sample(batch_size, replace=False)
            batch_rows.append(use_rows)
            batch_row_count += use_rows.shape[0]
            if batch_row_count >= batch_size:
                batch_df = pd.concat(batch_rows)
                yield batch_df.iloc[:batch_size].copy()
                batch_rows = [batch_df.iloc[batch_size:].copy()]
                batch_row_count = 0
        return

    def iter_batches(self, batch_size, groupby_go_and_tax=False, nbatch=None):
        """
        Iterate over batches of ddi_pdb table
        """
        if self.training:
            ddi_pdb = self._sample_epoch_ddi_pdb_table(self.ddi_pdb_train)
        else:
            ddi_pdb = self._sample_epoch_ddi_pdb_table(self.ddi_pdb_eval)

        if groupby_go_and_tax:
            rec_iter = self._iter_records_with_group(ddi_pdb, batch_size)
        else:
            rec_iter = self._iter_records(ddi_pdb, batch_size)

        for idx, batch_records in enumerate(rec_iter):
            batch_records = self.tokenize_batch(batch_records)
            yield batch_records
            if nbatch is not None and idx >= nbatch:
                break


class SequenceDataset(TokenizerMixin):
    def __init__(
        self,
        chain_seqs: list[str],
        window=512,
        step_size=128,
    ):
        self.chain_seqs = chain_seqs
        self.window = window
        self.step_size = step_size
        self.chain_tiles = self._tile_seq(chain_seqs)
        self.device = _get_device("cuda")

        TokenizerMixin.__init__(self, device=self.device)
        return

    def _tile_seq(self, seqs):
        records = []
        for chain_id, seq in enumerate(seqs):
            for tile_id, tile_start in enumerate(range(0, len(seq), self.step_size)):
                tile_end = min(len(seq), tile_start + self.window)
                records.append([chain_id, tile_id, seq[tile_start:tile_end]])
                if tile_end == len(seq):
                    break
        return pd.DataFrame(records, columns=["chain_id", "tile_id", "seq"])

    def tokenize_batch(self, batch_df):
        """
        Tokenize a batch of protein sequence
        """
        seq_list = batch_df["seq"].tolist()
        batch_tensor = self._tokenize(seq_list)
        return batch_tensor

    def iter_batches(self, batch_size=64):
        """
        Iterate over batches of ddi_pdb table
        """
        num_batches = int(np.ceil(len(self.chain_tiles) / batch_size))
        for i in range(num_batches):
            batch_records = self.chain_tiles.iloc[i * batch_size : (i + 1) * batch_size]
            batch_records = self.tokenize_batch(batch_records)
            yield batch_records
