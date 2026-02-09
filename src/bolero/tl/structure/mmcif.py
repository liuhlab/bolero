import gzip
import io
import os
import tempfile
from itertools import combinations, combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import MMCIFParser
from scipy.ndimage import gaussian_filter1d

from .code import converter


def merge_segments(segments, min_size=50):
    """
    Recursively merge sorted segments so that each merged segment spans at least min_size.
    """
    start, end = segments[0][0], segments[-1][1]
    total = end - start
    # Base case: one segment or span too small to split
    if len(segments) == 1 or total <= min_size:
        return [[start, end]]

    mid = start + total / 2
    # Locate the segment that covers the midpoint
    mid_i = 0
    for s, e in segments:
        if s <= mid <= e:
            break
        mid_i += 1

    # Decide group assignment based on overlap with mid
    left_overlap = mid - s
    right_overlap = e - mid
    if left_overlap >= right_overlap:
        left, right = segments[: mid_i + 1], segments[mid_i + 1 :]
    else:
        left, right = segments[:mid_i], segments[mid_i:]

    # Compute spans for both groups (if they exist)
    left_span = left[-1][1] - left[0][0] if left else 0
    right_span = right[-1][1] - right[0][0] if right else 0

    # If both groups are sufficiently large, split recursively; otherwise, merge all.
    if left and right and left_span >= min_size and right_span >= min_size:
        return merge_segments(left, min_size) + merge_segments(right, min_size)
    return [[start, end]]


def segment_protein_chain_plddt(
    ca_plddt_scores, threshold=70, smoothing_sigma=5, min_region_size=50
):
    """
    Segments a protein chain into contiguous regions based on AlphaFold pLDDT scores.
    Ensures all segments are at least `min_region_size` by merging small ones with the previous segment.

    Parameters
    ----------
        ca_plddt_scores (numpy array): 1D array of pLDDT values for each C-alpha position.
        threshold (float): pLDDT cutoff to define structured vs. disordered regions.
        smoothing_sigma (int): Gaussian smoothing factor to reduce noise.
        min_region_size (int): Minimum size of any segment.

    Returns
    -------
        list of tuples: Segments as (start, end, mean_pLDDT, is_folded), where
                        `start` is inclusive and `end` is exclusive (Python slicing).
    """
    # Smooth the pLDDT scores
    smoothed_plddt = gaussian_filter1d(ca_plddt_scores, sigma=smoothing_sigma)

    # Identify transition points
    is_folded = smoothed_plddt >= threshold
    change_points = (
        np.where(np.diff(is_folded.astype(int)) != 0)[0] + 1
    )  # Indices where state changes

    # Add first and last position to ensure full coverage
    segment_boundaries = np.concatenate(([0], change_points, [len(ca_plddt_scores)]))

    # Extract segments
    segments = []
    for i in range(len(segment_boundaries) - 1):
        start, end = segment_boundaries[i], segment_boundaries[i + 1]
        segments.append([start, end])  # Store as list for easy modification

    # Merge small segments
    merged_segments = merge_segments(segments, min_region_size)

    # add mean lddt to each segment
    for i, (start, end) in enumerate(merged_segments):
        mean_pLDDT = np.mean(ca_plddt_scores[start:end])
        merged_segments[i].append(mean_pLDDT)
    merged_segments = pd.DataFrame(
        merged_segments, columns=["start", "end", "mean_plddt"]
    )
    return merged_segments


def plot_plddt_segments(all_segments, ca_atom_table):
    """Plot pLDDT segments on top of raw pLDDT scores."""
    n_chain = len(all_segments["chain"].unique())
    _, axes = plt.subplots(
        figsize=(10, 1.5 * n_chain), nrows=n_chain, constrained_layout=True
    )
    if n_chain == 1:
        axes = [axes]
    for ax, (chain, chain_segments) in zip(axes, all_segments.groupby("chain")):
        chain_table = ca_atom_table[ca_atom_table["chain"] == chain].reset_index(
            drop=True
        )
        ax.plot(chain_table["pLDDT"], label="Raw pLDDT", alpha=0.7)
        for _, (start, end, score, _) in chain_segments.iterrows():
            ax.hlines(
                y=score,
                xmin=start,
                xmax=end,
                colors="grey",
                linestyle="--",
                linewidth=1,
            )
            ax.vlines(x=start, ymin=0, ymax=100, colors="grey", linewidth=1)
            print(
                f"Segment: {start}-{end}, Size: {end - start}, Mean pLDDT: {score:.2f}"
            )
        ax.vlines(x=end, ymin=0, ymax=100, colors="grey", linewidth=1)
        ax.set_title(f"Chain {chain}")
        ax.set_ylabel("pLDDT")
        ax.set_ylim(0, 100)
    ax = axes[-1]
    ax.set_xlabel("Residue Position")
    plt.show()


def get_segments_mean_pae(pae, segments):
    """
    Get mean PAE for each segment pair.
    """
    seg_pae = {}
    for seg_a, (starta, enda, *_) in segments.iterrows():
        for seg_b, (startb, endb, *_) in segments.iterrows():
            seg_mean_pae = pae[starta:enda, startb:endb].mean()
            seg_pae[seg_a, seg_b] = seg_mean_pae
    return pd.Series(seg_pae).unstack().astype("float32")


class mmCIFStructure:
    def __init__(self, mmcif_path, name="structure"):
        mmcif_path = str(mmcif_path)
        if "\n" in mmcif_path:
            # assume mmcif_path is the actual mmcif content
            self.structure = MMCIFParser().get_structure(name, io.StringIO(mmcif_path))
        else:
            if str(mmcif_path).endswith(".gz"):
                with gzip.open(mmcif_path, "rt") as f:
                    self.structure = MMCIFParser().get_structure(name, f)
            else:
                self.structure = MMCIFParser().get_structure(name, mmcif_path)

        self._atom_table = None
        self.pae = None
        self.converter = converter
        self.residual_offset = 0

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
                                [
                                    atom.name,
                                    residue.id[1],
                                    residue.resname,
                                    chain.id,
                                    atom.bfactor,
                                ]
                            )
            atom_table = pd.DataFrame(
                atom_to_token,
                columns=[
                    "atom_name",
                    "residue_chain_ids",
                    "residue_name",
                    "chain",
                    "pLDDT",
                ],
            )
            self._atom_table = atom_table
        return self._atom_table

    def view(self, hue="plddt", **kwargs):
        """
        Visualize the structure in Jupyter using ipymolstar (PDBeMolstar).

        Parameters
        ----------
        hue : str, default="plddt"
            'plddt' (or 'atom_plddt', 'bfactor') for pLDDT coloring,
            'chain' for chain-based coloring.
        **kwargs
            Passed to PDBeMolstar (e.g. theme, hide_water, alphafold_view).
        """
        try:
            from ipymolstar import PDBeMolstar
        except ImportError:
            raise ImportError(
                "Visualization requires ipymolstar: pip install ipymolstar"
            ) from None

        hue_lower = hue.lower() if isinstance(hue, str) else ""
        if hue_lower not in ("chain", "plddt", "atom_plddt", "bfactor"):
            raise ValueError(
                f"hue must be one of 'chain', 'plddt', 'atom_plddt', 'bfactor'; got {hue!r}"
            )
        alphafold_view = hue_lower != "chain"

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".cif", delete=False
            ) as tmp:
                tmp_path = tmp.name
            io_obj = MMCIFIO()
            io_obj.set_structure(self.structure)
            io_obj.save(tmp_path)
            with open(tmp_path, "rb") as f:
                mmcif_bytes = f.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        custom_data = {"data": mmcif_bytes, "format": "cif", "binary": False}
        kwargs = dict(kwargs)
        kwargs.setdefault("alphafold_view", alphafold_view)
        return PDBeMolstar(custom_data=custom_data, **kwargs)

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
        ca_plddt = self.atom_table[self.atom_table["atom_name"] == "CA"].reset_index(
            drop=True
        )
        ca_plddt.pop("atom_name")
        return ca_plddt

    def plot_plddt_and_pae(self):
        """Plot pLDDT and PAE."""
        plddt = self.get_residue_ca_plddts()["pLDDT"].values
        pae = self.pae
        length = plddt.size

        fig = plt.figure(figsize=(4, 4), dpi=200, constrained_layout=False)
        gs = fig.add_gridspec(8, 8, wspace=0.3, hspace=0.3)

        # plddt
        ax = fig.add_subplot(gs[0, :-1])
        ax.plot(np.arange(length), plddt, linewidth=1)
        ax.set(xlim=(0, length), ylim=(0, 103), xticklabels=[])
        sns.despine(ax=ax)
        ax = fig.add_subplot(gs[1:, -1])
        ax.plot(plddt, np.arange(length), linewidth=1)
        ax.set(xlim=(0, 103), ylim=(0, length), yticklabels=[])
        sns.despine(ax=ax)
        if pae is None:
            return fig

        # pae
        ax = fig.add_subplot(gs[1:, :-1])
        im = ax.imshow(pae, aspect="auto", cmap="Greens_r", vmin=0, vmax=30)
        ax.set(ylim=(0, length))
        cax = fig.add_subplot(gs[0, -1])
        fig.colorbar(im, cax=cax, orientation="vertical", label="")
        return fig

    def get_protein_plddt_segments(
        self, threshold=70, smoothing_sigma=5, min_region_size=30, plot=False
    ):
        """
        Get protein folding segments based on pLDDT scores.

        Parameters
        ----------
            threshold (float): pLDDT cutoff to define structured vs. disordered regions.
            smoothing_sigma (int): Gaussian smoothing factor to reduce noise.
            min_region_size (int): Minimum size of any segment.
            plot (bool): Whether to plot the results.

        Returns
        -------
            pd.DataFrame: Segmented regions with columns (start, end, mean_pLDDT, is_folded, chain).
        """
        atom_table = self.atom_table
        ca_atom_table = atom_table[atom_table["atom_name"] == "CA"]

        all_segments = []
        for chain, chain_table in ca_atom_table.groupby("chain"):
            segments = segment_protein_chain_plddt(
                chain_table["pLDDT"].values,
                threshold=threshold,
                smoothing_sigma=smoothing_sigma,
                min_region_size=min_region_size,
            )
            segments["chain"] = chain
            all_segments.append(segments)
        all_segments = pd.concat(all_segments)

        if plot:
            plot_plddt_segments(all_segments, self.get_residue_ca_plddts()["pLDDT"])
        return all_segments

    def get_sequence(self):
        """Get the protein sequence of the structure."""
        residule = self.get_residue_ca_plddts()["residue_name"]
        seq = "".join(self.converter.triple_to_single(residule).values)
        return seq

    def get_protein_chain_data(self):
        """
        Get chain data dict of protein residues information and coordinates.

        Schema:
        {
            chain_id: [
                {
                    "residue": Bio.PDB.Residue,
                    "residue_id": int,
                    "ca": np.array([x, y, z]),
                    "atoms": np.array([[x1, y1, z1], [x2, y2, z2], ...])
                },
                ...
            ],
            ...
        }
        """
        chain_data = {}

        for model in self.structure:
            for chain in model:
                chain_id = chain.get_id()
                residues = []
                for residue in chain:
                    # Only consider residues that contain a Cα atom
                    if "CA" in residue:
                        ca_atom = residue["CA"]
                        ca_coord = np.array(ca_atom.get_coord(), dtype=np.float32)
                        # Get coordinates for all atoms in the residue
                        atom_coords = []
                        for atom in residue:
                            atom_coords.append(
                                np.array(atom.get_coord(), dtype=np.float32)
                            )
                        atom_coords = np.array(atom_coords)  # shape: (n_atoms, 3)

                        residues.append(
                            {
                                "residue": residue,
                                "residue_id": residue.get_id()[
                                    1
                                ],  # e.g., (' ', resseq, icode)
                                "ca": ca_coord,
                                "atoms": atom_coords,
                            }
                        )
                if residues:
                    chain_data[chain_id] = residues
        return chain_data

    def calculate_residue_contacts(
        self,
        ca_threshold: float = 10.0,
        atom_threshold: float = 4.0,
        intra_chain: bool = False,
        intra_chain_min_aa_dist: int = 5,
    ) -> dict:
        """
        Calculates all pairwise residue-residue interactions between different chains.
        For every pair of residues (from different chains whose Ca atoms are within
        `ca_threshold`, the function computes the minimum atom-atom distance between
        the two residues.

        Parameters
        ----------
        mmCIF_file (str): Path to the mmCIF file.
        ca_threshold (float): Distance threshold (in angstrom) for Ca-Ca pre-filtering.
        atom_threshold (float): Distance threshold (in angstrom) for atom-atom distance.
        intra_chain (bool): If True, include intra-chain contacts.
        intra_chain_min_aa_dist (int): Minimum distance between residues in the same chain to be considered a contact.
            This parameter is used to filter out contacts between too close residues in the same chain,
            which is in contact due to closure or just secondary structure.
            The contact from this function refers to 3D long-range contacts.

        Returns
        -------
        contacts (dict): Keys are tuples of ((chain1, residue_id1), (chain2, residue_id2))
                        and values are the minimum atom-atom distance between these residues.
        """
        chain_data = self.get_protein_chain_data()

        # Store final contacts:
        # [[chain1, residue_id1, chain2, residue_id2, min_atom_dist]]
        contacts = []

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get list of chain IDs and compare all unique pairs
        chain_pairs = []
        for model in self.structure:
            model_chain_ids = [c.get_id() for c in model if c.get_id() in chain_data]
            if intra_chain:
                pairs = combinations_with_replacement(model_chain_ids, 2)
            else:
                pairs = combinations(model_chain_ids, 2)
            chain_pairs.extend(pairs)

        for chain_id1, chain_id2 in chain_pairs:
            residues1 = chain_data[chain_id1]
            residues2 = chain_data[chain_id2]

            # Convert the list of CA coordinates into a NumPy array, then to a torch tensor.
            ca_coords1 = torch.from_numpy(
                np.array([res["ca"] for res in residues1])
            ).to(device)
            # (n1, 3)
            ca_coords2 = torch.from_numpy(
                np.array([res["ca"] for res in residues2])
            ).to(device)
            # (n2, 3)
            ca_dists = torch.cdist(ca_coords1, ca_coords2)

            if intra_chain and (chain_id1 == chain_id2):
                # Intra-chain contacts: mask out close residues
                n_residue = ca_dists.shape[0]
                i = torch.arange(n_residue).unsqueeze(1)
                j = torch.arange(n_residue).unsqueeze(0)
                diff = torch.abs(i - j)
                # position to be masked will be 1
                mask = (diff < intra_chain_min_aa_dist).to(
                    dtype=ca_dists.dtype, device=ca_dists.device
                )
                ca_dists = ca_dists + (mask * (ca_threshold + 1))
                # add large value to the masked position so they won't be selected

            # Find residue pairs with Cα distance below threshold
            indices = torch.nonzero(ca_dists < ca_threshold, as_tuple=False)
            for index in indices:
                idx1, idx2 = index.tolist()
                res1 = residues1[idx1]
                res2 = residues2[idx2]

                # Compute the full atom-atom distance matrix for the residue pair
                atoms1 = torch.from_numpy(res1["atoms"]).to(device)  # (n_atoms1, 3)
                atoms2 = torch.from_numpy(res2["atoms"]).to(device)  # (n_atoms2, 3)
                atom_dists = torch.cdist(atoms1, atoms2)
                min_dist = torch.min(atom_dists).item()

                # Record the contact information:
                contacts.append(
                    [
                        chain_id1,
                        res1["residue_id"],
                        chain_id2,
                        res2["residue_id"],
                        min_dist,
                    ]
                )
        contacts = pd.DataFrame(
            contacts,
            columns=["chain1", "residue_id1", "chain2", "residue_id2", "min_atom_dist"],
        )
        # apply atom threshold
        contacts = contacts[contacts["min_atom_dist"] < atom_threshold].copy()
        return contacts
