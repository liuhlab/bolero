import numpy as np
from Bio.Seq import Seq

DEFAULT_ONE_HOT_ORDER = "ACGT"


def one_hot_decoding(one_hot: np.ndarray, order=DEFAULT_ONE_HOT_ORDER) -> str:
    """
    Decoding of a one-hot encoded DNA sequence. Output is a string.

    Parameters
    ----------
    one_hot : numpy.ndarray
        One-hot encoded DNA sequence. Shape must be (len(seq), 4).
    order : str, optional
        Order of the one-hot encoding base axis. Default is 'ACGT'
        so reverse the base axis will be equal to make a compelment conversion.

    Returns
    -------
    seq : str
        Decoded DNA sequence
    """
    seq = "".join([order[i] for i in np.argmax(one_hot, axis=1)])
    return seq


def one_hot_encoding(seq: str, order: str, dtype: np.dtype) -> np.ndarray:
    """
    One-hot encoding of a DNA sequence string. Output is a numpy array of shape (len(seq), 4).

    Parameters
    ----------
    seq : str
        DNA sequence string to be encoded.
    order : str
        Order of the one-hot encoding base axis. Default is 'ACGT'
        so reverse the base axis will be equal to make a compelment conversion.
    dtype : numpy.dtype
        Data type of the output array. Default is np.int8.

    Returns
    -------
    one_hot : numpy.ndarray
        One-hot encoding of the sequence
    """
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

    def one_hot_encoding(
        self, order=DEFAULT_ONE_HOT_ORDER, dtype=np.int8
    ) -> np.ndarray:
        """
        One-hot encoding of a DNA sequence string. Output is a numpy array of shape (len(seq), 4).

        Parameters
        ----------
        order : str, optional
            Order of the one-hot encoding base axis. Default is 'ACGT' so reverse the base axis will be equal to make a compelment conversion.
        dtype : numpy.dtype, optional
            Data type of the output array. Default is np.int8.

        Returns
        -------
        one_hot : numpy.ndarray
            One-hot encoding of the sequence
        """
        one_hot = one_hot_encoding(str(self), order, dtype)
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
