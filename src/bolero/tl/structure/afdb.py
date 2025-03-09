import pathlib
import re

import numpy as np
import pandas as pd

from .code import converter
from .mmcif import get_segments_mean_pae, mmCIFStructure, segment_protein_chain_plddt


def find_seq_overlap(seq1, seq2, min_match=5):
    """
    Find the offset of seq2 in seq1.

    Args:
        seq1 (str): The reference sequence.
        seq2 (str): The query sequence.
        min_match (int): Initial number of characters to search.

    Returns
    -------
        int or None: The unique offset if found and verified, otherwise None.
    """
    for length in range(min_match, len(seq2) + 1):
        fragment = seq2[:length]  # Take progressively longer fragments
        positions = [m.start() for m in re.finditer(f"(?={fragment})", seq1)]

        if len(positions) == 1:  # Unique match found
            offset = positions[0]

            # Final check: Ensure the full overlap part of seq1 matches seq2
            overlap_length = min(
                len(seq1) - offset, len(seq2)
            )  # Overlapping region size
            if seq1[offset : offset + overlap_length] == seq2[:overlap_length]:
                return offset
            else:
                return None  # Overlap verification failed
        elif len(positions) == 0:
            return None

    return None


class AFDB:
    def __init__(self, db_dir):
        self.db_dir = pathlib.Path(db_dir)
        metadata = pd.read_feather(self.db_dir / "metadata.feather")
        self.uniprot_frame_count = metadata.value_counts("uniprot_id")
        self.uniprot_ids = self.uniprot_frame_count.index
        self.metadata = metadata.set_index(["uniprot_id", "frame"])
        self.metadata["frame_int"] = self.metadata.index.get_level_values("frame").map(
            lambda i: int(i[1:])
        )
        self.converter = converter

    def get_structure(self, prot_id, frame=None):
        """
        Get structure for a protein in AFDB with a uniprot ID and frame number.

        Parameters
        ----------
        prot_id : str
            Uniprot ID of the protein
        frame : int, optional
            Frame number of the protein, by default None.
            For long proteins, there will be multiple frames, specify the frame number to get the structure.

        Returns
        -------
        mmCIFStructure
            Structure of the protein
        """
        assert prot_id in self.uniprot_ids, f"Protein {prot_id} not found in AFDB"

        if frame is None:
            assert (
                self.uniprot_frame_count[prot_id] == 1
            ), f"Multiple frames for {prot_id}, please specify frame"
            frame = 1
        offset = int(200 * (frame - 1))
        frame = f"F{frame}"

        mmcif_file = self.metadata.loc[(prot_id, frame), "mmcif_file"]
        structure = mmCIFStructure(self.db_dir / "mmcif" / mmcif_file)

        # residule offset
        structure.residual_offset = offset

        pae_file = self.metadata.loc[(prot_id, frame), "pae_file"]
        if isinstance(pae_file, str):
            structure.pae = np.load(self.db_dir / "pae" / pae_file)["data"]
        return structure

    def get_structure_segments(
        self, prot_id, min_region_size=30, threshold=70, smoothing_sigma=5
    ):
        """
        Get structure segments for a protein in AFDB with a uniprot ID.

        Parameters
        ----------
        prot_id : str
            Uniprot ID of the protein
        min_region_size : int, optional
            Minimum region size for segments, by default 30.
        threshold : int, optional
            Threshold for pLDDT to cut segments, by default 70.
        smoothing_sigma : int, optional
            Sigma for smoothing pLDDT, by default 5.

        Returns
        -------
        segments : pd.DataFrame
            Segments of the protein
        segment_pae : pd.DataFrame
            PAE of the segments
        """
        frame_count = self.uniprot_frame_count[prot_id]

        # load structures (if multiple)
        structures = []
        for frame in range(1, frame_count + 1):
            structure = self.get_structure(prot_id, frame)
            structures.append(structure)

        # get whole chain pLDDT
        if frame_count > 1:
            # combine from multiple frames
            full_plddt = []
            for idx, structure in enumerate(structures):
                plddt = structure.get_residue_ca_plddts().copy()
                plddt["residue_chain_ids"] += structure.residual_offset
                if idx == 0:
                    # clip last 200 aa
                    full_plddt.append(plddt.iloc[:-200])
                elif idx != len(structures) - 1:
                    # clip first and last 200 aa
                    full_plddt.append(plddt.iloc[200:-200])
                else:
                    # clip first 200 aa
                    full_plddt.append(plddt.iloc[200:])
            full_plddt = pd.concat(full_plddt)

            full_plddt = (
                full_plddt.groupby(["residue_chain_ids", "residue_name", "chain"])
                .agg({"pLDDT": "mean"})
                .reset_index()
            )
            assert (
                full_plddt.value_counts(["chain", "residue_chain_ids"]).max() == 1
            ), "residue position is not unique after merge frames"
        else:
            # only a single frame for the whole protein
            full_plddt = structure.get_residue_ca_plddts()

        # create pLDDT segments
        segments = segment_protein_chain_plddt(
            full_plddt["pLDDT"].values,
            min_region_size=min_region_size,
            threshold=threshold,
            smoothing_sigma=smoothing_sigma,
        )
        segments["chain"] = full_plddt["chain"].values[0]

        # get segment mean pae
        if frame_count > 1:
            print("Multiple frames PAE not supported yet")
            pae = None
        else:
            pae = structure.pae
        if pae is not None:
            segment_pae = get_segments_mean_pae(pae, segments)
        else:
            segment_pae = None

        return segments, segment_pae
