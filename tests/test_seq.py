import numpy as np

from bolero.pp.seq import Sequence


def test_seq():
    seq = Sequence("ACTTGC")
    assert seq.reverse_complement() == "GCAAGT"

    oh = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert (seq.one_hot_encoding() != oh).sum() == 0
