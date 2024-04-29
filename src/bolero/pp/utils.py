from bolero.pp.seq import DEFAULT_ONE_HOT_ORDER


import numpy as np


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


def get_global_coords(chrom_offsets, region_bed_df):
    add_start = (
        region_bed_df["Chromosome"].map(chrom_offsets["global_start"]).astype(int)
    )
    start = region_bed_df["Start"] + add_start
    end = region_bed_df["End"] + add_start
    global_coords = np.hstack([start.values[:, None], end.values[:, None]])
    return global_coords