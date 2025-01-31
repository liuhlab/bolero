import json
import pathlib
from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser


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

    def __init__(self, output_dir):
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

    def _get_run_name(self):
        name = list(self.output_dir.glob("*_data.json"))[0].name[:-10]
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
            self._input_data = _json_load(self._input_data_path)
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
        if self._atom_table is None:
            atom_to_token = []
            for model in self.structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            atom_to_token.append(
                                [atom.name, residue.id[1], residue.resname, chain.id]
                            )
            atom_table = pd.DataFrame(
                atom_to_token,
                columns=["atom_name", "residue_chain_ids", "residue_name", "chain"],
            )
            atom_table["pLDDT"] = self.confidences_data["atom_plddts"]
        return atom_table

    def get_residue_ave_plddts(self):
        """Get residue atom average pLDDT."""
        res_plddt = (
            self.atom_table.groupby(["chain", "residue_chain_ids", "residue_name"])[
                "pLDDT"
            ]
            .mean()
            .reset_index()
        )
        return res_plddt

    def get_residue_ca_plddts(self):
        """Get residue alpha carbon (CA) pLDDT."""
        ca_plddt = self.atom_table[self.atom_table["atom_name"] == "CA"]
        ca_plddt.pop("atom_name")
        return ca_plddt

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
            # Load the mmCIF file
            parser = MMCIFParser()
            self._structure = parser.get_structure("af3", str(self._model_path))
        return self._structure

    @property
    def num_atoms(self):
        """Number of atoms."""
        return len(self.confidences_data["atom_plddts"])

    @property
    def num_tokens(self):
        """Number of residues."""
        return len(self.confidences_data["token_res_ids"])

    def set_atom_bfactors(self, value="atom_plddts"):
        """Set atom bfactor with atom or residue level values."""
        if isinstance(value, str):
            if value == "atom_plddts":
                for atom, score in zip(
                    self.structure.get_atoms(), self.confidences_data["atom_plddts"]
                ):
                    atom.set_bfactor(score)
            else:
                raise ValueError(
                    f"Unknown value: {value}, possible values: 'atom_plddts'"
                )
        else:
            # assume value is list or array or series
            if len(value) == self.num_atoms:
                for atom, score in zip(self.structure.get_atoms(), value):
                    atom.set_bfactor(score)
            elif len(value) == self.num_tokens:
                for residue, score in zip(self.structure.get_residues(), value):
                    for atom in residue:
                        atom.set_bfactor(score)
            else:
                raise ValueError(
                    f"Length of value ({len(value)}) does not match number of "
                    f"atoms ({self.num_atoms}) or tokens ({self.num_tokens})."
                )
        return

    def view(self, hue="chain", hue_norm=(0, 100), colormap="RdYlBu", **kwargs):
        """
        Simple visualization in jupyter using NGLView.
        """
        try:
            import nglview as nv
        except ImportError:
            raise ImportError("Please install nglview to visualize model.") from None

        use_bfactor = True
        if isinstance(hue, str):
            if hue == "chain":
                # default hue is chain
                use_bfactor = False
            elif hue.lower() in ("atom_plddt", "plddt"):
                self.set_atom_bfactors(value="atom_plddts")
            elif hue == "bfactor":
                pass
            else:
                raise ValueError(
                    f"Unknown hue: {hue}, possible values: 'chain', 'atom_plddt'"
                )
        else:
            self.set_atom_bfactors(value=hue)

        # Visualize using NGLView
        view = nv.show_biopython(self.structure, **kwargs)

        if use_bfactor:
            view.clear_representations()
            view.add_representation(
                "cartoon",
                color="bfactor",
                colorScale=colormap,  # Uses the Viridis colormap
                colorDomain=hue_norm,  # Maps pLDDT values correctly
            )
        return view
