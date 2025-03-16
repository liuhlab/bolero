import warnings

import torch
import torch.nn.functional as F
from esm.models.esmc import ESMC as ESMC_esm
from esm.sdk.api import ESMProtein, LogitsConfig


def make_segment_batch(emb, segments, max_segment_size):
    """
    Create a batch of segment embeddings from a protein embedding and segment ranges.

    Parameters
    ----------
    emb: torch.Tensor
        Protein embedding tensor of shape (1, protein_size, emb_dim).
    segments: torch.Tensor
        Segment ranges tensor of shape (n_segment, 2).
    max_segment_size: int
        Maximum segment size for padding.

    Returns
    -------
    seg_batch: torch.Tensor
        Segment embedding tensor of shape (n_segment, max_segment_size, emb_dim).
    valid_mask: torch.Tensor
        Mask tensor of shape (n_segment, max_segment_size) indicating valid positions.
    """
    if emb.ndim == 3:
        # remove batch dim
        emb = emb.squeeze(0)

    assert (
        emb.shape[0] >= segments.max().item()
    ), "Segment boarder exceed the protein size"

    emb = F.pad(emb, (0, 0, 0, max_segment_size))

    seg_start = segments[:, 0].unsqueeze(1)  # shape: (n_segment, 1)
    seg_len = segments[:, 1] - segments[:, 0]  # shape: (n_segment,)

    # Gather embeddings from the padded protein embedding for each segment.
    range_tensor = torch.arange(max_segment_size, device=emb.device).unsqueeze(0)
    indices = seg_start + range_tensor  # shape: (n_segment, max_segment_size)

    seg_batch = emb[indices]  # shape: (n_segment, max_segment_size, emb_dim)

    # Create a valid mask for true segment size
    # shape: (n_segment, max_segment_size)
    valid_mask = range_tensor < seg_len.unsqueeze(1)
    # Zero out positions that are not valid
    seg_batch = seg_batch * valid_mask.unsqueeze(-1).to(seg_batch.dtype)
    return seg_batch, valid_mask


class ESMC:
    def __init__(self, model_name="esmc_600m", device=None):
        assert model_name in (
            "esmc_600m",
            "esmc_300m",
        ), f"Unknown model name: {model_name}"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.model = ESMC_esm.from_pretrained(model_name, device)

    def get_embedding(self, sequence, numpy=False):
        """
        Get the ESMC embedding for a protein sequence.

        Parameters
        ----------
        sequence: str
            Protein sequence.
        numpy: bool
            Whether to return numpy array.

        Returns
        -------
        emb: torch.Tensor or numpy.ndarray
            Embedding tensor of shape (1, protein_size, emb_dim) or numpy array.
        """
        protein_tensor = self.tokenize_sequence(sequence)
        logits_output = self.model.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        emb = logits_output.embeddings
        if numpy:
            emb = emb.cpu().numpy()
        # shape: (1, protein_size, emb_dim)
        return emb

    def tokenize_sequence(self, sequence):
        """
        Tokenize a protein sequence.

        Parameters
        ----------
        sequence: str
            Protein sequence.

        Returns
        -------
        protein_tensor: torch.Tensor
            Tokenized protein tensor of shape (1, protein_size, emb_dim).
        """
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        return protein_tensor

    def get_segments_embedding(self, sequence, segments, max_segment_size, numpy=False):
        """
        Get the ESMC embedding for segments of a protein sequence.

        Parameters
        ----------
        sequence: str
            Protein sequence.
        segments: torch.Tensor
            Segment ranges tensor of shape (n_segment, 2).
        max_segment_size: int
            Maximum segment size for padding.
        numpy: bool
            Whether to return numpy array.

        Returns
        -------
        seg_batch: torch.Tensor or numpy.ndarray
            Segment embedding tensor of shape (n_segment, max_segment_size, emb_dim) or numpy array.
        valid_mask: torch.Tensor or numpy.ndarray
            Mask tensor of shape (n_segment, max_segment_size) indicating valid positions.
        """
        emb = self.get_embedding(sequence)
        segments = torch.tensor(segments, device=emb.device)

        seg_batch, valid_mask = make_segment_batch(emb, segments, max_segment_size)
        if numpy:
            seg_batch = seg_batch.cpu().numpy()
            valid_mask = valid_mask.cpu().numpy()
        return seg_batch, valid_mask


"""
Notes:

1. Get a protein embedding from a protein sequence.
2. Get segments from protein domain, peptide motif or structure informed annotation
3. Get domain-domain interaction annotation from PDB or AF3 prediction
4. Train CLIP model to predict domain-domain interaction from segment batches
"""
