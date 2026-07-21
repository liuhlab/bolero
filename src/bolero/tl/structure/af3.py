"""Build AlphaFold3 inputs, run inference, and read back AF3 result bundles.

Workflow: assemble an :class:`AF3Input` (proteins with cached MSA/templates, DNA,
ligands), run it with :meth:`AF3Input.infer` (which drives :class:`AF3Runner` over a
Singularity image, data-pipeline-free), then load the compact result with
:class:`AF3ResultMinimum` (or :class:`AF3Result` for a full AF3 output directory).
"""

import itertools
import json
import os
import pathlib
import re
import string
import subprocess
import tempfile
from collections.abc import Sequence
from itertools import combinations, combinations_with_replacement

import joblib
import numpy as np
import pandas as pd

from .code import converter
from .mmcif import mmCIFStructure

StrPath = str | pathlib.Path

# AlphaFold3 requires a local install: the AF3 code, model parameters, the multi-GB
# public sequence databases, and a Singularity image — none are shipped with bolero (out
# of scope for this package). Point these at your own AF3 setup via environment variables:
#   BOLERO_AF3_CODE_DIR, BOLERO_AF3_MODEL_DIR, BOLERO_AF3_DB_DIR,
#   BOLERO_AF3_SINGULARITY_SIF, BOLERO_AF3_MSA_CACHE_DIR
# Any unset (or non-existent) path resolves to None.
_LOCAL_DEFAULT = {
    "af3_code_dir": os.environ.get("BOLERO_AF3_CODE_DIR"),
    "af3_model_dir": os.environ.get("BOLERO_AF3_MODEL_DIR"),
    "af3_db_dir": os.environ.get("BOLERO_AF3_DB_DIR"),
    "af3_singularity_sif": os.environ.get("BOLERO_AF3_SINGULARITY_SIF"),
    "msa_cache_dir": os.environ.get("BOLERO_AF3_MSA_CACHE_DIR"),
}
_LOCAL_DEFAULT = {
    k: p if p is not None and pathlib.Path(p).exists() else None
    for k, p in _LOCAL_DEFAULT.items()
}


def _json_load(path: StrPath) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def get_chain_interval(chain_ids: list[str]) -> dict[str, tuple[int, int]]:
    """
    Get the (start, end) index interval spanned by each chain in a token list.

    Chains are assumed contiguous, e.g. ``["A", "A", "A", "B", "B"]`` ->
    ``{"A": (0, 3), "B": (3, 5)}``.
    """
    arr = np.array(chain_ids)
    change_indices = (
        np.where(arr[:-1] != arr[1:])[0] + 1
    )  # Shift by +1 to mark the start of new segments
    start_indices = np.insert(change_indices, 0, 0)  # First segment starts at index 0
    end_indices = np.append(
        change_indices, len(arr)
    )  # Last segment ends at the last index
    intervals = {
        str(arr[start]): (int(start), int(end))
        for start, end in zip(start_indices, end_indices, strict=False)
    }
    return intervals


class AF3Result:
    """
    Read-access to a full AlphaFold3 output directory.

    Wraps the ``*_summary_confidences.json``, ``*_confidences.json``, ``*_data.json``
    and ``*_model.cif`` files produced by one AF3 run. ``chain_idmap`` optionally maps
    chain IDs back to source identifiers (gene/protein IDs, sequences, ligand codes).
    """

    def __init__(self, output_dir: StrPath, chain_idmap: dict | None = None) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.run_name = self._get_run_name()

        self.summary_data, self.confidences_data = self._load_data()
        self._input_data = None
        self._atom_table = None
        self._atom_plddts = None
        self._contact_probs = None
        self._pae = None
        self._chain_intervals = None
        self._structure = None
        self.converter = converter
        self.chain_idmap = chain_idmap if chain_idmap is not None else {}

    def _get_run_name(self) -> str:
        _input_jsons = list(self.output_dir.glob("*_data.json"))
        assert (
            len(_input_jsons) == 1
        ), f"Multiple or no input json found in output dir {self.output_dir}."
        name = _input_jsons[0].name[:-10]
        return name

    @property
    def _summary_confidences_path(self):
        return self.output_dir / f"{self.run_name}_summary_confidences.json"

    @property
    def _confidences_path(self):
        return self.output_dir / f"{self.run_name}_confidences.json"

    @property
    def _input_data_path(self):
        return self.output_dir / f"{self.run_name}_data.json"

    @property
    def _model_path(self):
        return self.output_dir / f"{self.run_name}_model.cif"

    def _load_data(self) -> tuple[dict, dict]:
        summary_data = _json_load(self._summary_confidences_path)
        confidences_data = _json_load(self._confidences_path)
        return summary_data, confidences_data

    @property
    def input_data(self) -> "AF3Input":
        """The :class:`AF3Input` reconstructed from the run's ``*_data.json``."""
        if self._input_data is None:
            self._input_data = AF3Input.load(self._input_data_path)
        return self._input_data

    @property
    def atom_plddts(self) -> dict:
        """Per-chain array of atom pLDDT values."""
        confidences_data = self.confidences_data

        atom_df = pd.DataFrame(
            {k: confidences_data[k] for k in ["atom_chain_ids", "atom_plddts"]}
        )
        atom_plddts = {
            chain: data["atom_plddts"].astype("float16").values
            for chain, data in atom_df.groupby("atom_chain_ids")
        }
        return atom_plddts

    @property
    def atom_table(self) -> pd.DataFrame:
        """Per-atom table from the model structure."""
        return self.structure.atom_table

    def get_residue_ave_plddts(self) -> pd.DataFrame:
        """Get per-residue pLDDT averaged over the residue's atoms."""
        return self.structure.get_residue_ave_plddts()

    def get_residue_ca_plddts(self) -> pd.DataFrame:
        """Get per-residue pLDDT taken from the alpha-carbon (CA) atom."""
        return self.structure.get_residue_ca_plddts()

    @property
    def contact_probs(self) -> np.ndarray:
        """Token-by-token contact probability matrix (symmetrized, float16)."""
        if self._contact_probs is None:
            data = np.array(self.confidences_data["contact_probs"])
            self._contact_probs = ((data + data.T) / 2).astype("float16")
        return self._contact_probs

    @property
    def pae(self) -> np.ndarray:
        """Token-by-token predicted alignment error (symmetrized, float16)."""
        if self._pae is None:
            data = np.array(self.confidences_data["pae"])
            # integer average, matching AF3ResultMinimum.pae
            self._pae = ((data + data.T) // 2).astype("float16")
        return self._pae

    @property
    def chain_intervals(self) -> dict[str, tuple[int, int]]:
        """Map each chain name to its (start, end) token interval."""
        if self._chain_intervals is None:
            self._chain_intervals = get_chain_interval(
                self.confidences_data["token_chain_ids"]
            )
        return self._chain_intervals

    @property
    def chain_names(self) -> list[str]:
        """List of chain names."""
        return list(self.chain_intervals.keys())

    def _get_chain_pairs(self, replace: bool = False) -> list[tuple[str, str]]:
        if replace:
            return list(combinations_with_replacement(self.chain_names, 2))
        else:
            return list(combinations(self.chain_names, 2))

    def _get_pairwise(
        self, matrix: np.ndarray, self_pair: bool
    ) -> dict[tuple[str, str], np.ndarray]:
        """Slice ``matrix`` into a per-chain-pair sub-matrix dict."""
        pairwise = {}
        for chain1, chain2 in self._get_chain_pairs(replace=self_pair):
            start1, end1 = self.chain_intervals[chain1]
            start2, end2 = self.chain_intervals[chain2]
            pairwise[chain1, chain2] = matrix[start1:end1, start2:end2].copy()
        return pairwise

    def get_pairwise_contact_probs(
        self, self_pair: bool = False
    ) -> dict[tuple[str, str], np.ndarray]:
        """Contact-probability sub-matrix for each chain pair (keyed by chain pair)."""
        return self._get_pairwise(self.contact_probs, self_pair)

    def get_pairwise_pae(
        self, self_pair: bool = False
    ) -> dict[tuple[str, str], np.ndarray]:
        """PAE sub-matrix for each chain pair (keyed by chain pair)."""
        return self._get_pairwise(self.pae, self_pair)

    @property
    def structure(self) -> mmCIFStructure:
        """The parsed mmCIF structure object."""
        if self._structure is None:
            self._structure = mmCIFStructure(self._model_path)
        return self._structure

    @property
    def num_atoms(self) -> int:
        """Number of atoms."""
        return len(self.confidences_data["atom_plddts"])

    @property
    def num_tokens(self) -> int:
        """Number of tokens (residues / nucleotides)."""
        return len(self.confidences_data["token_res_ids"])

    def view(self, **kwargs):
        """View the structure (see :meth:`mmCIFStructure.view`)."""
        return self.structure.view(**kwargs)

    def get_minimum_results(self) -> dict:
        """
        Get minimum results and save in a single dict.

        Minimum results include:
        - chain summary data like iptm, ptm, clash, frac_disorder etc.
        - chain name, size, sequence
        - Ca pLDDT at uint8
        - pae (not symetric) at uint8
        - contact_prob * 100 (not symetric) at uint8
        - mmCIF file content
        """
        # raw pae at uint8
        pae = np.round(self.confidences_data["pae"]).astype("uint8")

        # raw contact_prob * 100 at uint8
        contact_prob = np.round(
            np.array(self.confidences_data["contact_probs"]) * 100
        ).astype("uint8")

        # get pLDDT and chain info
        plddt = self.get_residue_ca_plddts()
        plddt["pLDDT"] = plddt["pLDDT"].round().astype("uint8")

        # add chain name, size, seq
        ave_plddt = (
            self.get_residue_ave_plddts()
        )  # we must use ave_plddt to get all protein and dna chains
        chain_size = ave_plddt["chain"].value_counts().sort_index()
        chains = chain_size.index
        chain_seqs = {}
        for chain in chains:
            chain_res = np.asarray(plddt.loc[plddt["chain"] == chain, "residue_name"])
            single = self.converter.triple_to_single(chain_res)
            chain_seq = "".join(np.asarray(single).tolist())
            if len(chain_seq) == 0:
                # dna chain, use ave_plddt to get the sequence
                chain_seq = ave_plddt.loc[
                    ave_plddt["chain"] == chain, "residue_name"
                ].str[-1]
                chain_seq = "".join(chain_seq.tolist())
            chain_seqs[chain] = chain_seq
        self.summary_data["chain_size"] = chain_size.values.tolist()
        self.summary_data["chains"] = chains.tolist()
        self.summary_data["chain_seqs"] = chain_seqs

        # add mmCIF file content
        with open(self._model_path) as f:
            mmcif_content = f.read()

        # gather all data
        core_metric = {
            **self.summary_data,
            "plddt": plddt["pLDDT"].values,
            "pae": pae,
            "contact_prob": contact_prob,
            "mmcif": mmcif_content,
            "chain_idmap": self.chain_idmap,
        }
        return core_metric


class AF3ResultMinimum(AF3Result):
    """
    Load the compact result bundle written by :meth:`AF3Input.infer`.

    Reads a single joblib dict (as produced by :meth:`AF3Result.get_minimum_results`)
    instead of a full AF3 output directory, reconstructing PAE / contact / structure
    access from the stored ``uint8`` arrays and embedded mmCIF content.
    """

    def __init__(self, minimum_path: str) -> None:
        self.minimum_path = minimum_path
        self.content = joblib.load(minimum_path)
        self._structure = None
        self.chain_idmap = self.content.get("chain_idmap", {})

    @property
    def chain_names(self) -> list[str]:
        """List of chain names."""
        return self.content["chains"]

    @property
    def chain_intervals(self) -> dict[str, tuple[int, int]]:
        """Map each chain name to its (start, end) token interval."""
        chain_intervals = {}
        cur_start = 0
        for chain, chain_size in zip(
            self.content["chains"], self.content["chain_size"], strict=False
        ):
            chain_intervals[chain] = (cur_start, cur_start + chain_size)
            cur_start += chain_size
        return chain_intervals

    @property
    def pae(self) -> np.ndarray:
        """Predicted alignment error (PAE), symmetrized from the stored uint8 matrix."""
        pae = self.content["pae"]
        pae = (pae + pae.T) // 2
        return pae

    @property
    def contact_probs(self) -> np.ndarray:
        """Contact probabilities, symmetrized and rescaled from the stored uint8 matrix."""
        contact_probs = self.content["contact_prob"]
        contact_probs = (contact_probs + contact_probs.T) / 200
        contact_probs = contact_probs.astype("float16")
        return contact_probs

    @property
    def structure(self) -> mmCIFStructure:
        """The structure object, parsed from the embedded mmCIF content."""
        if self._structure is None:
            self._structure = mmCIFStructure(self.content["mmcif"])
        return self._structure


def _id_generator():
    """Yield chain IDs A, B, ... Z, AA, AB, ... (up to length 4)."""
    alphabet = string.ascii_uppercase  # 'A' to 'Z'
    length = 1

    while length <= 4:
        for id_tuple in itertools.product(alphabet, repeat=length):
            yield "".join(id_tuple)
        length += 1


def _truncate_long_strings(obj, max_length: int = 30):
    """Recursively truncate long string values in JSON objects (for printing)."""
    if isinstance(obj, dict):
        return {
            k: (
                _truncate_long_strings(v, max_length)
                if k != "templates"
                else f"{len(v)} mmCIF templates"
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_truncate_long_strings(v, max_length) for v in obj]
    elif isinstance(obj, str) and len(obj) > max_length:
        return obj[:max_length] + "..."
    else:
        return obj


def _get_gene_af3_cache_path(gene: str, genome, af3_cache_dir: str) -> str:
    """Resolve the cached MSA/template joblib path for a gene or protein accession."""
    record = genome.get_gene_protein_sequence(gene, sel_longest=True)
    if record is None:
        acc = gene  # assume gene is a protein accession
    else:
        acc = record.id
    af3_cache_path = f"{af3_cache_dir}/{acc}.msa_and_template.joblib.gz"
    if not pathlib.Path(af3_cache_path).exists():
        raise FileNotFoundError(
            f"AF3 cache file not found: {af3_cache_path} for {gene}"
        )
    return af3_cache_path


def truncate_single_msa_seq(seq: str, start: int, end: int) -> str:
    """Slice one A3M row to columns [start, end), ignoring lowercase insertions."""
    # encode char
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    # pos not in lower char [a-z] (insertions)
    valid_pos = ~((arr >= 97) & (arr <= 122))
    # -1 because first valid pos should start from 0
    pos_idx = valid_pos.cumsum() - 1
    start_idx = pos_idx.searchsorted(start, side="left")
    end_idx = pos_idx.searchsorted(end, side="left")
    return seq[start_idx:end_idx]


def truncate_a3m_msa(unpaired_msa: str, start: int, end: int) -> str:
    """
    Truncate an A3M MSA string to a given start and end slice, handling variable sequence lengths.

    Parameters
    ----------
    unpaired_msa : str
        A3M formatted MSA string.
    start : int
        0-based start index (inclusive).
    end : int
        0-based end index (exclusive).

    Returns
    -------
    str
        Truncated A3M formatted MSA string.
    """
    lines = unpaired_msa.strip().split("\n")

    # Separate headers and sequences
    new_lines = []
    for line in lines:
        if not line.startswith(">"):
            line = truncate_single_msa_seq(line, start, end)
        new_lines.append(line)
    return "\n".join(new_lines)


def truncate_templates(
    templates: list, start: int, end: int, min_length: int = 10
) -> list:
    """
    Truncates a list of template dictionaries to match the given start and end slice.

    Parameters
    ----------
    templates : list
        A list of templates in AlphaFold 3 format.
    start : int
        The 0-based start index (inclusive).
    end : int
        The 0-based end index (exclusive).
    min_length : int
        Only include truncated templates with usable indices > min_length.

    Returns
    -------
    list
        The truncated template list.
    """
    truncated_templates = []
    for template in templates:
        temp_idx = np.array([template["queryIndices"], template["templateIndices"]])
        idx_start = temp_idx[0].searchsorted(start, side="left")
        idx_end = temp_idx[0].searchsorted(end, side="left")
        if (idx_end - idx_start) > min_length:
            # only return templates with enough indices remained
            # offset the query indices to start from 0
            template["queryIndices"] = (temp_idx[0, idx_start:idx_end] - start).tolist()
            template["templateIndices"] = temp_idx[1, idx_start:idx_end].tolist()
            truncated_templates.append(template)
    return truncated_templates


class AF3Input:
    """
    Builder for an AlphaFold3 input JSON.

    Add chains with :meth:`add_protein` / :meth:`add_protein_from_cache`,
    :meth:`add_dna`, :meth:`add_ligand_ccd` / :meth:`add_ligand_smiles` (or build the
    whole thing at once with :meth:`from_design`), then :meth:`dump` the JSON or
    :meth:`infer` directly. ``chain_idmap`` records what each auto-assigned chain ID
    corresponds to (gene/protein ID, DNA sequence, ligand code).
    """

    def __init__(
        self,
        name: str,
        modelSeeds: int | list[int] = 1,
        sequences: list[dict] | None = None,
        bondedAtomPairs: list | None = None,
        userCCD: str | None = None,
        dialect: str = "alphafold3",
        version: int = 2,
    ) -> None:
        """
        Init AF3Input json data.

        See documentation for more information:
        https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
        """
        self.chain_idmap = {}

        if isinstance(modelSeeds, int):
            modelSeeds = [modelSeeds]
        if sequences is None:
            sequences = []

        # af3 will do name convertion anyway, do it early here to avoid miss-match
        name = re.sub(r"[^a-z0-9-]", "-", name.lower())
        self.name = name
        self.json_data = {
            "name": name,
            "modelSeeds": modelSeeds,
            "sequences": sequences,
            "dialect": dialect,
            "version": version,
        }
        if userCCD is not None:
            self.json_data["userCCD"] = userCCD
        if bondedAtomPairs is not None:
            self.json_data["bondedAtomPairs"] = bondedAtomPairs

        self._chain_ids = set()
        for seq in sequences:
            for d in seq.values():
                _id = d["id"]
                if isinstance(_id, str):
                    self._chain_ids.add(_id)
                else:
                    for i in _id:
                        self._chain_ids.add(i)
        self._id_generator = _id_generator()
        return

    def _get_next_chain_id(self) -> str:
        while True:
            try:
                _id = next(self._id_generator)
                if _id not in self._chain_ids:
                    break
            except StopIteration:
                raise StopIteration(
                    "Chain ID generator exhausted, there seems to be huge numbre of chains in this input."
                ) from None
        return _id

    def _prepare_chain_id(
        self, chain_id: str | list[str] | None = None, repeat: int = 1
    ) -> str | list[str]:
        if chain_id is None:
            ids = [self._get_next_chain_id() for _ in range(repeat)]
        elif isinstance(chain_id, str):
            ids = [chain_id]
        else:
            ids = list(chain_id)
        for _id in ids:
            assert _id not in self._chain_ids, f"Chain ID {_id} already exists."
            self._chain_ids.add(_id)
        # return a bare str for a single chain, else the list (AF3 accepts both)
        return ids[0] if len(ids) == 1 else ids

    def add_protein(
        self,
        sequence: str,
        chain_id: str | list[str] | None = None,
        repeat: int = 1,
        **kwargs,
    ) -> str | list[str]:
        """
        Add a protein chain to AF3Input.

        Parameters
        ----------
        sequence : str
            Protein sequence.
        chain_id : list of str, optional
            Explicit chain IDs; auto-assigned when omitted.
        repeat : int
            Number of copies of this chain in the model.
        **kwargs
            Extra fields for the AF3 ``protein`` entry (e.g. ``unpairedMsa``,
            ``pairedMsa``, ``templates``).

        Returns
        -------
        str or list of str
            The assigned chain ID(s).
        """
        chain_id = self._prepare_chain_id(chain_id, repeat)
        protein = {"protein": {"id": chain_id, "sequence": sequence}}
        protein["protein"].update(kwargs)
        self.json_data["sequences"].append(protein)
        return chain_id

    def add_protein_from_cache(
        self,
        prot_id: str,
        cache_data: dict | str,
        chain_id: str | list[str] | None = None,
        repeat: int = 1,
        truncate_region: Sequence[int] | None = None,
        min_template_length: int = 10,
    ) -> None:
        """
        Add a protein chain with precomputed MSA and templates.

        Parameters
        ----------
        prot_id : str
            Source identifier recorded in ``chain_idmap`` for the added chain(s).
        cache_data : dict or str
            Cache dict, or a path to a joblib file holding one.
        chain_id : list of str, optional
            Explicit chain IDs; auto-assigned when omitted.
        repeat : int
            Number of copies of this chain.
        truncate_region : tuple of int, optional
            ``(start, end)`` to truncate the sequence, MSA and templates to.
        min_template_length : int
            Minimum retained length for a truncated template to be kept.
        """
        if isinstance(cache_data, dict):
            data = cache_data
        else:
            # assume its file path
            data = joblib.load(cache_data)

        chain_ids = []
        for _, chain_data in data.items():
            if truncate_region is not None and len(truncate_region) == 2:
                chain_data = self.truncate_protein_chain_data(
                    chain_data, truncate_region, min_template_length=min_template_length
                )
            # don't reuse `chain_id`: each cache entry gets its own id(s)
            new_id = self.add_protein(chain_id=chain_id, repeat=repeat, **chain_data)
            if isinstance(new_id, list):
                chain_ids.extend(new_id)
            else:
                chain_ids.append(new_id)

        for cid in chain_ids:
            self.chain_idmap[cid] = prot_id
        return

    @staticmethod
    def truncate_protein_chain_data(
        chain_data: dict,
        truncate_region: Sequence[int],
        min_template_length: int,
    ) -> dict:
        """Truncate a protein chain's sequence, MSA and templates in place."""
        start, end = truncate_region
        chain_data["unpairedMsa"] = truncate_a3m_msa(
            chain_data["unpairedMsa"], start, end
        )
        chain_data["pairedMsa"] = truncate_a3m_msa(chain_data["pairedMsa"], start, end)
        chain_data["templates"] = truncate_templates(
            chain_data["templates"], start, end, min_length=min_template_length
        )
        chain_data["sequence"] = chain_data["sequence"][start:end]
        return chain_data

    def add_dna(
        self,
        sequence: str,
        chain_id: str | list[str] | None = None,
        repeat: int = 1,
        modifications: list[dict] | None = None,
        add_rc_strand: bool = True,
    ) -> None:
        """
        Add a DNA chain to AF3Input.

        Parameters
        ----------
        sequence : str
            DNA sequence (upper-cased internally).
        chain_id : list of str, optional
            Explicit chain IDs; auto-assigned when omitted.
        repeat : int
            Number of copies of this chain.
        modifications : list of dict, optional
            AF3 DNA modification entries.
        add_rc_strand : bool
            If True, also add the reverse-complement strand as a separate chain.
        """
        sequence = sequence.upper()

        chain_id = self._prepare_chain_id(chain_id, repeat)
        dna = {"dna": {"id": chain_id, "sequence": sequence}}
        if modifications is not None:
            dna["dna"]["modifications"] = modifications
        self.json_data["sequences"].append(dna)
        if add_rc_strand:
            rc_sequence = sequence.translate(str.maketrans("ATCG", "TAGC"))[::-1]
            self.add_dna(sequence=rc_sequence, repeat=repeat, add_rc_strand=False)

        ids = [chain_id] if isinstance(chain_id, str) else chain_id
        for _id in ids:
            self.chain_idmap[_id] = sequence
        return

    def add_ligand_ccd(
        self,
        ccdCodes: str | list[str],
        chain_id: str | list[str] | None = None,
        repeat: int = 1,
    ) -> None:
        """
        Add a ligand to AF3Input by CCD code.

        Parameters
        ----------
        ccdCodes : str or list of str
            Chemical Component Dictionary code(s).
        chain_id : list of str, optional
            Explicit chain IDs; auto-assigned when omitted.
        repeat : int
            Number of copies of this ligand.
        """
        chain_id = self._prepare_chain_id(chain_id, repeat)
        if isinstance(ccdCodes, str):
            ccdCodes = [ccdCodes]
        ligand = {"ligand": {"id": chain_id, "ccdCodes": ccdCodes}}
        self.json_data["sequences"].append(ligand)
        ids = [chain_id] if isinstance(chain_id, str) else chain_id
        for _id in ids:
            self.chain_idmap[_id] = ccdCodes
        return

    def add_ligand_smiles(
        self, smiles: str, chain_id: str | list[str] | None = None, repeat: int = 1
    ) -> None:
        """
        Add a ligand to AF3Input by SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string.
        chain_id : list of str, optional
            Explicit chain IDs; auto-assigned when omitted.
        repeat : int
            Number of copies of this ligand.
        """
        chain_id = self._prepare_chain_id(chain_id, repeat)
        ligand = {"ligand": {"id": chain_id, "smiles": smiles}}
        self.json_data["sequences"].append(ligand)
        ids = [chain_id] if isinstance(chain_id, str) else chain_id
        for _id in ids:
            self.chain_idmap[_id] = smiles
        return

    def dump(self, path: StrPath) -> None:
        """
        Dump AF3Input json data to a file.
        """
        with open(path, "w") as f:
            json.dump(self.json_data, f, indent=4)
        return

    @classmethod
    def load(cls, path: StrPath) -> "AF3Input":
        """
        Load AF3Input json data from a file.
        """
        with open(path) as f:
            json_data = json.load(f)
        return cls(**json_data)

    @classmethod
    def from_design(
        cls,
        name: str,
        design: list[dict],
        genome,
        af3_cache_dir: str | None = None,
        **kwargs,
    ) -> "AF3Input":
        """
        Create AF3Input from a design list.

        Each design entry maps a sequence type to a list of items; repeated items are
        merged into a single multi-copy chain. Proteins are resolved through the MSA/
        template cache and may carry a ``GENE:start-end`` suffix to truncate them.

        Example
        -------
        >>> design = [
        ...     {"protein": ["Fos", "Fos", "Jun", "Jun"]},
        ...     {"DNA": ["TGACTCA", "TGACGTCA", "CACTGGCT"]},
        ... ]
        """
        if af3_cache_dir is None:
            af3_cache_dir = _LOCAL_DEFAULT["msa_cache_dir"]
        if af3_cache_dir is None:
            raise ValueError("No MSA cache dir provided and no local default found.")

        af3_input = AF3Input(name=name, **kwargs)
        for seq_dict in design:
            for seq_type, seq_list in seq_dict.items():
                seq_and_repeats = list(pd.Series(seq_list).value_counts().items())
                match seq_type.lower():
                    case "protein":
                        for prot, repeat in seq_and_repeats:
                            prot, *truncate = str(prot).split(":")
                            truncate_region = (
                                list(map(int, truncate[0].split("-")))
                                if len(truncate) == 1
                                else None
                            )
                            af3_cache_path = _get_gene_af3_cache_path(
                                prot, genome, af3_cache_dir
                            )
                            af3_input.add_protein_from_cache(
                                prot_id=prot,
                                cache_data=af3_cache_path,
                                repeat=repeat,
                                truncate_region=truncate_region,
                            )
                    case "dna":
                        for seq, repeat in seq_and_repeats:
                            af3_input.add_dna(sequence=str(seq), repeat=repeat)
                    case "ligand_ccd" | "ligand":
                        for lig, repeat in seq_and_repeats:
                            af3_input.add_ligand_ccd(str(lig), repeat=repeat)
                    case "ligand_smiles":
                        for lig, repeat in seq_and_repeats:
                            af3_input.add_ligand_smiles(str(lig), repeat=repeat)
        return af3_input

    def __repr__(self):
        truncated_data = _truncate_long_strings(self.json_data)
        head = f"AF3Input object with {len(self._chain_ids)} chains\n\n"
        print_str = head + json.dumps(truncated_data, indent=4, ensure_ascii=False)
        return print_str

    def infer(
        self,
        save_dir: StrPath,
        return_minimum: bool = True,
        delete_temp: bool = True,
        verbose: bool = False,
        redo: bool = False,
    ) -> pathlib.Path:
        """
        Run AlphaFold3 inference (no data pipeline) and save the result bundle.

        Parameters
        ----------
        save_dir : str or Path
            Directory to write ``{name}_af3_inference.gz`` into.
        return_minimum : bool
            Save only the compact minimum result (see :meth:`AF3Result.get_minimum_results`).
        delete_temp : bool
            Delete the raw AF3 output dir; set False to keep it for debugging.
        verbose : bool
            Stream AF3 stdout/stderr instead of capturing it.
        redo : bool
            Recompute even if the output file already exists.

        Returns
        -------
        pathlib.Path
            Path to the saved result bundle.
        """
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        final_path = save_dir / f"{self.name}_af3_inference.gz"
        temp_path = save_dir / f"{self.name}_af3_inference_temp.gz"
        if final_path.exists() and not redo:
            return final_path

        runner = AF3Runner()
        if delete_temp:
            # temp AF3 input/output path will be saved in a temp dir
            _output_dir = None
        else:
            _output_dir = save_dir / f"{self.name}_af3_inference"
            _output_dir.mkdir(parents=True, exist_ok=True)
        # only minimum results of the best model will be returned here;
        # the minimum result already carries chain_idmap (from get_minimum_results)
        result = runner.af3_inference_no_data_pipeline(
            self,
            return_minimum=return_minimum,
            delete_temp=delete_temp,
            verbose=verbose,
            output_dir=_output_dir,
        )

        joblib.dump(result, temp_path)
        temp_path.rename(final_path)
        return final_path


AF3_INFER_NO_DATA_PIPELINE = """
singularity exec --nv \
--bind {run_dir}:/root/af_run \
--bind {af3_code_dir}:/root/alphafold3 \
--bind /tmp:/root/tmp \
--bind {af3_model_dir}:/root/models \
--bind {af3_db_dir}:/root/public_databases \
{af3_singularity_sif} \
python /root/alphafold3/run_alphafold.py \
--json_path=/root/af_run/{input_file_name} \
--model_dir=/root/models \
--db_dir=/root/public_databases \
--output_dir=/root/af_run \
--jax_compilation_cache_dir=/root/tmp/ \
--norun_data_pipeline
"""


class AF3Runner:
    """
    Run AlphaFold3 inference via its Singularity image with ``--norun_data_pipeline``.

    Paths default to the lab-local locations in ``_LOCAL_DEFAULT`` and must all exist.
    """

    def __init__(
        self,
        af3_code_dir: str | None = None,
        af3_model_dir: str | None = None,
        af3_db_dir: str | None = None,
        af3_singularity_sif: str | None = None,
    ) -> None:
        self.af3_code_dir = self._resolve_path(
            af3_code_dir, _LOCAL_DEFAULT["af3_code_dir"], "AF3 code dir"
        )
        self.af3_model_dir = self._resolve_path(
            af3_model_dir, _LOCAL_DEFAULT["af3_model_dir"], "Model dir"
        )
        self.af3_db_dir = self._resolve_path(
            af3_db_dir, _LOCAL_DEFAULT["af3_db_dir"], "Database dir"
        )
        self.af3_singularity_sif = self._resolve_path(
            af3_singularity_sif,
            _LOCAL_DEFAULT["af3_singularity_sif"],
            "Singularity SIF",
        )

    @staticmethod
    def _resolve_path(value: str | None, default: str | None, what: str) -> str:
        """Return ``value`` or the local ``default``, asserting the path exists."""
        path = value or default
        assert (
            path and pathlib.Path(path).exists()
        ), f"{what} not provided or not exist."
        return path

    def af3_inference_no_data_pipeline(
        self,
        af3_input: "AF3Input",
        output_dir: StrPath | None = None,
        return_minimum: bool = True,
        delete_temp: bool = True,
        verbose: bool = False,
    ) -> "dict | AF3Result":
        """
        Run AlphaFold3 inference with no data pipeline.

        Writes the input JSON and run script into ``output_dir`` (a temp dir when
        None), executes the Singularity command, then returns the parsed result —
        the compact dict when ``return_minimum`` else an :class:`AF3Result`.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="af3_")
            is_temp_dir = True
        else:
            is_temp_dir = False

        input_json_path = f"{output_dir}/input.json"
        af3_input.dump(input_json_path)

        script_path = f"{output_dir}/run.sh"
        infer_script = AF3_INFER_NO_DATA_PIPELINE.format(
            af3_code_dir=self.af3_code_dir,
            af3_model_dir=self.af3_model_dir,
            af3_db_dir=self.af3_db_dir,
            af3_singularity_sif=self.af3_singularity_sif,
            run_dir=output_dir,
            input_file_name="input.json",
        )
        with open(script_path, "w") as f:
            f.write(infer_script)

        subprocess.run(
            ["bash", script_path],
            check=True,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
        )

        result = AF3Result(
            f"{output_dir}/{af3_input.name}", chain_idmap=af3_input.chain_idmap
        )
        if return_minimum:
            result = result.get_minimum_results()
        if is_temp_dir and delete_temp:
            subprocess.run(["rm", "-rf", output_dir], check=True)
        return result
