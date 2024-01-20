import numpy as np
from Bio.Seq import Seq


def _one_hot_encoding(seq, order, dtype):
    one_hot = np.zeros((len(seq), 4), dtype=dtype)
    seq_array = np.array(list(seq.upper()))

    for i, base in enumerate(order.upper()):
        one_hot[:, i] = seq_array == base
    return one_hot


class Sequence(Seq):
    """Utility class for DNA sequence manipulation. Inherits from Bio.Seq.Seq."""

    def __init__(self, data, name=None, chrom=None, start=None, end=None, strand=None):
        super().__init__(data=data)
        self.name = name
        self.chrom = chrom
        self.start = start
        self.end = end
        self.strand = strand

    def one_hot_encoding(self, order="ATCG", dtype=np.int8) -> np.ndarray:
        """
        One-hot encoding of a DNA sequence string. Output is a numpy array of shape (len(seq), 4).

        Parameters
        ----------
        order : str, optional
            Order of the one-hot encoding base axis. Default is 'ATCG'.
        dtype : numpy.dtype, optional
            Data type of the output array. Default is np.int8.

        Returns
        -------
        one_hot : numpy.ndarray
            One-hot encoding of the sequence
        """
        one_hot = _one_hot_encoding(str(self), order, dtype)
        return one_hot

    def reverse_complement(self):
        """Returns the reverse complement of the sequence."""
        seq = super().reverse_complement()

        strand = self.strand
        if strand:
            if strand == "+":
                strand = "-"
            elif strand == "-":
                strand = "+"
        return Sequence(
            seq,
            name=self.name,
            chrom=self.chrom,
            start=self.start,
            end=self.end,
            strand=strand,
        )

    @property
    def sequence(self):
        """Returns the sequence as a string."""
        return str(self)
