import gzip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from scipy.ndimage import gaussian_filter1d


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


class mmCIFStructure:
    def __init__(self, mmcif_path, name="structure"):
        if str(mmcif_path).endswith(".gz"):
            with gzip.open(mmcif_path, "rt") as f:
                self.structure = MMCIFParser().get_structure(name, f)
        else:
            self.structure = MMCIFParser().get_structure(name, mmcif_path)

        self._atom_table = None

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
                hue = "bfactor"
            elif hue == "bfactor":
                pass
            else:
                raise ValueError(
                    f"Unknown hue: {hue}, possible values: 'chain', 'atom_plddt'"
                )
        else:
            raise ValueError("hue must be a string.")

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
            n_chain = len(all_segments["chain"].unique())
            _, axes = plt.subplots(
                figsize=(10, 1.5 * n_chain), nrows=n_chain, constrained_layout=True
            )
            if n_chain == 1:
                axes = [axes]
            for ax, (chain, chain_segments) in zip(axes, all_segments.groupby("chain")):
                chain_table = ca_atom_table[
                    ca_atom_table["chain"] == chain
                ].reset_index(drop=True)
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
        return all_segments
