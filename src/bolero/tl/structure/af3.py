import itertools
import json
import pathlib
import re
import string
import subprocess
import tempfile
from itertools import combinations, combinations_with_replacement

import joblib
import numpy as np
import pandas as pd

from .code import converter
from .mmcif import mmCIFStructure

_LOCAL_DEFAULT = {
    "af3_code_dir": "/large_storage/zhoulab/hanliu/wmb/ref/af3/alphafold3/",
    "af3_model_dir": "/scratch/zhoulab/hanliu/af3/model",
    "af3_db_dir": "/scratch/zhoulab/hanliu/af3/public_databases",
    "af3_singularity_sif": "/scratch/zhoulab/hanliu/af3/alphafold3.sif",
    "msa_cache_dir": "/large_storage/zhoulab/hanliu/wmb/ref/af3/msa_cache/protein",
}
_LOCAL_DEFAULT = {
    k: p if pathlib.Path(p).exists() else None for k, p in _LOCAL_DEFAULT.items()
}


def _json_load(path):
    with open(path) as f:
        data = json.load(f)
    return data


def get_chain_interval(chain_ids):
    """
    Get start end interval for a list of chain name.
    e.g., ["A", "A", "A", "A", "B", "B", "B"]
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
        arr[start]: (start, end) for start, end in zip(start_indices, end_indices)
    }
    return intervals


class AF3Result:
    """AlphaFold3 result object"""

    def __init__(self, output_dir, chain_idmap=None):
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

    def _get_run_name(self):
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

    def _load_data(self):
        summary_data = _json_load(self._summary_confidences_path)
        confidences_data = _json_load(self._confidences_path)
        return summary_data, confidences_data

    @property
    def input_data(self):
        """Input data."""
        if self._input_data is None:
            self._input_data = AF3Input.load(self._input_data_path)
        return self._input_data

    @property
    def atom_plddts(self):
        """Atom pLDDT."""
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
    def atom_table(self):
        """Atom table."""
        return self.structure.atom_table

    def get_residue_ave_plddts(self):
        """Get residue atom average pLDDT."""
        return self.structure.get_residue_ave_plddts()

    def get_residue_ca_plddts(self):
        """Get residue alpha carbon (CA) pLDDT."""
        return self.structure.get_residue_ca_plddts()

    @property
    def contact_probs(self):
        """Contact probabilities."""
        if self._contact_probs is None:
            data = np.array(self.confidences_data["contact_probs"])
            self._contact_probs = ((data + data.T) / 2).astype("float16")
        return self._contact_probs

    @property
    def pae(self):
        """Predicted alignment error (PAE)."""
        if self._pae is None:
            data = np.array(self.confidences_data["pae"])
            self._pae = ((data + data.T) / 2).astype("float16")
        return self._pae

    @property
    def chain_intervals(self):
        """Chain intervals."""
        if self._chain_intervals is None:
            self._chain_intervals = get_chain_interval(
                self.confidences_data["token_chain_ids"]
            )
        return self._chain_intervals

    @property
    def chain_names(self):
        """List of chain names."""
        return list(self.chain_intervals.keys())

    def _get_chain_pair_contact_probs(self, chain1, chain2):
        start1, end1 = self.chain_intervals[chain1]
        start2, end2 = self.chain_intervals[chain2]
        return self.contact_probs[start1:end1, start2:end2].copy()

    def _get_chain_pair_pae(self, chain1, chain2):
        start1, end1 = self.chain_intervals[chain1]
        start2, end2 = self.chain_intervals[chain2]
        return self.pae[start1:end1, start2:end2].copy()

    def _get_chain_pairs(self, replace=False):
        if replace:
            return list(combinations_with_replacement(self.chain_names, 2))
        else:
            return list(combinations(self.chain_names, 2))

    def get_pairwise_contact_probs(self, self_pair=False):
        """Get chain pair contact probabilities."""
        pairs = self._get_chain_pairs(replace=self_pair)
        contact_probs = {
            pair: self._get_chain_pair_contact_probs(*pair) for pair in pairs
        }
        return contact_probs

    def get_pairwise_pae(self, self_pair=False):
        """Get chain pair predicted alignment error (PAE)."""
        pairs = self._get_chain_pairs(replace=self_pair)
        pae = {pair: self._get_chain_pair_pae(*pair) for pair in pairs}
        return pae

    @property
    def structure(self):
        """The structure object."""
        if self._structure is None:
            self._structure = mmCIFStructure(self._model_path)
        return self._structure

    @property
    def num_atoms(self):
        """Number of atoms."""
        return len(self.confidences_data["atom_plddts"])

    @property
    def num_tokens(self):
        """Number of residues."""
        return len(self.confidences_data["token_res_ids"])

    def view(self, **kwargs):
        """View the structure."""
        return self.structure.view(**kwargs)

    def get_minimum_results(self):
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
        chain_size = plddt["chain"].value_counts().sort_index()
        chains = chain_size.index
        chain_seqs = {}
        for chain in chains:
            chain_seq = plddt.loc[plddt["chain"] == chain, "residue_name"]
            chain_seq = "".join(self.converter.triple_to_single(chain_seq).tolist())
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
    def __init__(self, minimum_path):
        self.minimum_path = minimum_path
        self.content = joblib.load(minimum_path)
        self._structure = None
        self.chain_idmap = self.content.get("chain_idmap", {})

    @property
    def chain_names(self):
        """List of chain names."""
        return self.content["chains"]

    @property
    def chain_intervals(self):
        """Chain intervals."""
        chain_intervals = {}
        cur_start = 0
        for chain, chain_size in zip(
            self.content["chains"], self.content["chain_size"]
        ):
            chain_intervals[chain] = (cur_start, cur_start + chain_size)
            cur_start += chain_size
        return chain_intervals

    @property
    def pae(self):
        """Predicted alignment error (PAE) symetric."""
        pae = self.content["pae"]
        pae = (pae + pae.T) // 2
        return pae

    @property
    def contact_probs(self):
        """Contact probabilities symetric."""
        contact_probs = self.content["contact_prob"]
        contact_probs = (contact_probs + contact_probs.T) / 200
        contact_probs = contact_probs.astype("float16")
        return contact_probs

    @property
    def structure(self):
        """The structure object."""
        if self._structure is None:
            self._structure = mmCIFStructure(self.content["mmcif"])
        return self._structure


def _id_generator():
    alphabet = string.ascii_uppercase  # 'A' to 'Z'
    length = 1

    while length <= 4:
        for id_tuple in itertools.product(alphabet, repeat=length):
            yield "".join(id_tuple)
        length += 1


def _truncate_long_strings(obj, max_length=30):
    """Recursively truncate long string values in JSON objects."""
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


def _get_gene_af3_cache_path(gene, genome, af3_cache_dir):
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


def truncate_single_msa_seq(seq, start, end):
    """Function to get truncated MSA sequence (ignoring lowercase insertions)."""
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

    Args:
        unpaired_msa (str): A3M formatted MSA string.
        start (int): 0-based start index (inclusive).
        end (int): 0-based end index (exclusive).

    Returns
    -------
        str: Truncated A3M formatted MSA string.
    """
    lines = unpaired_msa.strip().split("\n")

    # Separate headers and sequences
    new_lines = []
    for line in lines:
        if not line.startswith(">"):
            line = truncate_single_msa_seq(line, start, end)
        new_lines.append(line)
    return "\n".join(new_lines)


def truncate_templates(templates: list, start: int, end: int, min_length=10) -> list:
    """
    Truncates a list of template dictionaries to match the given start and end slice.

    Args:
        templates (list): A list of templates in AlphaFold 3 format.
        start (int): The 0-based start index (inclusive).
        end (int): The 0-based end index (exclusive).
        min_length (int): Only include truncated templates with usable indices > min_length.

    Returns
    -------
        list: The truncated template list.
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
    chain_idmap = {}

    def __init__(
        self,
        name,
        modelSeeds=1,
        sequences=None,
        bondedAtomPairs=None,
        userCCD=None,
        dialect="alphafold3",
        version=2,
    ):
        """
        Init AF3Input json data.

        See documentation for more information:
        https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
        """
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

    def _get_next_chain_id(self):
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

    def _prepare_chain_id(self, chain_id=None, repeat=1):
        if chain_id is None:
            chain_id = [self._get_next_chain_id() for _ in range(repeat)]
        for _id in chain_id:
            assert _id not in self._chain_ids, f"Chain ID {chain_id} already exists."
            self._chain_ids.add(_id)
        if len(chain_id) == 1:
            chain_id = chain_id[0]
        return chain_id

    def add_protein(self, sequence, chain_id=None, repeat=1, **kwargs):
        """
        Add protein to AF3Input.

        Args:
            pid: Protein ID
            sequence: Protein sequence
            repeat: Make number of copies for this chain in the model
            **kwargs: Additional keyword arguments
        """
        chain_id = self._prepare_chain_id(chain_id, repeat)
        protein = {"protein": {"id": chain_id, "sequence": sequence}}
        protein["protein"].update(kwargs)
        self.json_data["sequences"].append(protein)
        return chain_id

    def add_protein_from_cache(
        self,
        prot_id,
        cache_data,
        chain_id=None,
        repeat=1,
        truncate_region=None,
        min_template_length=10,
    ):
        """
        Add protein with MSA and template cache from a file.
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
            chain_id = self.add_protein(chain_id=chain_id, repeat=repeat, **chain_data)
            if isinstance(chain_id, list):
                chain_ids.extend(chain_id)
            else:
                chain_ids.append(chain_id)

        for chain_id in chain_ids:
            self.chain_idmap[chain_id] = prot_id
        return

    @staticmethod
    def truncate_protein_chain_data(chain_data, truncate_region, min_template_length):
        """Truncate protein chain MSA and templates."""
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
        sequence,
        chain_id=None,
        repeat=1,
        modifications: list[dict] = None,
        add_rc_strand=True,
    ):
        """
        Add DNA to AF3Input.

        Args:
            sequence: DNA sequence
            modifications: List of modifications, each modification is a dict with keys:
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

        if repeat == 1:
            chain_id = [chain_id]
        for _id in chain_id:
            self.chain_idmap[_id] = sequence
        return

    def add_ligand_ccd(self, ccdCodes, chain_id=None, repeat=1):
        """
        Add ligand to AF3Input.

        Args:
            ccdCodes: List of chemical component dictionary codes
        """
        chain_id = self._prepare_chain_id(chain_id, repeat)
        if isinstance(ccdCodes, str):
            ccdCodes = [ccdCodes]
        ligand = {"ligand": {"id": chain_id, "ccdCodes": ccdCodes}}
        self.json_data["sequences"].append(ligand)
        if repeat == 1:
            chain_id = [chain_id]
        for _id in chain_id:
            self.chain_idmap[_id] = ccdCodes
        return

    def add_ligand_smiles(self, smiles, chain_id=None, repeat=1):
        """
        Add ligand to AF3Input.

        Args:
            smiles: SMILES string
        """
        chain_id = self._prepare_chain_id(chain_id, repeat)
        ligand = {"ligand": {"id": chain_id, "smiles": smiles}}
        self.json_data["sequences"].append(ligand)
        if repeat == 1:
            chain_id = [chain_id]
        for _id in chain_id:
            self.chain_idmap[_id] = smiles
        return

    def dump(self, path):
        """
        Dump AF3Input json data to a file.
        """
        with open(path, "w") as f:
            json.dump(self.json_data, f, indent=4)
        return

    @classmethod
    def load(cls, path):
        """
        Load AF3Input json data from a file.
        """
        with open(path) as f:
            json_data = json.load(f)
        return cls(**json_data)

    @classmethod
    def from_design(cls, name, design, genome, af3_cache_dir=None, **kwargs):
        """
        Create AF3Input from design list.

        Example design list:
        # repeated names will be merged atuomatically in the input json

        design = [
            {"protein": ["Fos", "Fos", "Jun", "Jun"]},
            {"DNA": ["TGACTCA", "TGACGTCA", "CACTGGCT"]},
        ]
        """
        if af3_cache_dir is None:
            af3_cache_dir = _LOCAL_DEFAULT["msa_cache_dir"]

        af3_input = AF3Input(name=name, **kwargs)
        for seq_dict in design:
            for seq_type, seq_list in seq_dict.items():
                seq_and_repeats = list(pd.Series(seq_list).value_counts().items())
                match seq_type.lower():
                    case "protein":
                        for prot, repeat in seq_and_repeats:
                            prot, *truncate_region = prot.split(":")
                            if len(truncate_region) == 1:
                                truncate_region = list(
                                    map(int, truncate_region[0].split("-"))
                                )
                            else:
                                truncate_region = None
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
                            af3_input.add_dna(sequence=seq, repeat=repeat)
                    case "ligand_ccd" | "ligand":
                        for lig, repeat in seq_and_repeats:
                            af3_input.add_ligand_ccd(lig, repeat=repeat)
                    case "ligand_smiles":
                        for lig, repeat in seq_and_repeats:
                            af3_input.add_ligand_smiles(lig, repeat=repeat)
        return af3_input

    def __repr__(self):
        truncated_data = _truncate_long_strings(self.json_data)
        head = f"AF3Input object with {len(self._chain_ids)} chains\n\n"
        print_str = head + json.dumps(truncated_data, indent=4, ensure_ascii=False)
        return print_str

    def infer(
        self, save_dir, return_minimum=True, delete_temp=True, verbose=False, redo=False
    ):
        """
        Run AlphaFold3 inference with no data pipeline.
        """
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        final_path = save_dir / f"{self.name}_af3_inference.gz"
        temp_path = save_dir / f"{self.name}_af3_inference_temp.gz"
        if final_path.exists() and not redo:
            return final_path

        runner = AF3Runner()
        # temp AF3 input/output path will be saved in a temp dir
        # only minimum results of the best model will be returned here
        result = runner.af3_inference_no_data_pipeline(
            self,
            return_minimum=return_minimum,
            delete_temp=delete_temp,
            verbose=verbose,
        )

        result["chain_idmap"] = self.chain_idmap

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
    def __init__(
        self,
        af3_code_dir=None,
        af3_model_dir=None,
        af3_db_dir=None,
        af3_singularity_sif=None,
    ):
        self.af3_code_dir = (
            af3_code_dir if af3_code_dir else _LOCAL_DEFAULT["af3_code_dir"]
        )
        assert pathlib.Path(
            self.af3_code_dir
        ).exists(), "AF3 path not provided or not exist."
        self.af3_model_dir = (
            af3_model_dir if af3_model_dir else _LOCAL_DEFAULT["af3_model_dir"]
        )
        assert pathlib.Path(
            self.af3_model_dir
        ).exists(), "Model dir not provided or not exist."
        self.af3_db_dir = af3_db_dir if af3_db_dir else _LOCAL_DEFAULT["af3_db_dir"]
        assert pathlib.Path(
            self.af3_db_dir
        ).exists(), "Database dir not provided or not exist."
        self.af3_singularity_sif = (
            af3_singularity_sif
            if af3_singularity_sif
            else _LOCAL_DEFAULT["af3_singularity_sif"]
        )
        assert pathlib.Path(
            self.af3_singularity_sif
        ).exists(), "Singularity SIF not provided or not exist."
        return

    def af3_inference_no_data_pipeline(
        self,
        af3_input,
        output_dir=None,
        return_minimum=True,
        delete_temp=True,
        verbose=False,
    ):
        """
        Run AlphaFold3 inference with no data pipeline.
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
