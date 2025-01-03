import numpy as np
import pandas as pd
import torch

from bolero import Genome
from bolero.pp.seq import one_hot_decoding
from bolero.utils import get_package_dir


def _dna_tokenizer(x, k=6):
    tok = ""
    i = 0
    while i <= len(x) - k:
        for j in range(k):
            tok = tok + x[i + j]
        tok = tok + " "
        i += 1
    return tok


def _dna_pre_processing(x, genome_dict):
    x = x.split()
    for i in range(len(x)):
        x[i] = genome_dict[x[i]]
    return x


def atac_pre_processing(x, k=6):
    """Discretize the ATAC-seq signal into 37 levels."""
    y = x.copy()
    x = x[: len(x) - k + 1]
    for i in range(len(x)):
        x[i] = sum(y[i : i + k]) / k
        if x[i] == 0:
            x[i] = 1
            continue
        if x[i] > 0 and x[i] < 0.001:
            x[i] = 2
            continue
        if x[i] >= 0.001 and x[i] < 0.01:
            x[i] = 3
            continue
        if x[i] >= 0.01 and x[i] < 0.1:
            x[i] = 4
            continue
        if x[i] >= 0.1 and x[i] < 0.125:
            x[i] = 5
            continue
        if x[i] >= 0.125 and x[i] < 0.15:
            x[i] = 6
            continue
        if x[i] >= 0.15 and x[i] < 0.2:
            x[i] = 7
            continue
        if x[i] >= 0.2 and x[i] < 0.3:
            x[i] = 8
            continue
        if x[i] >= 0.3 and x[i] < 0.4:
            x[i] = 9
            continue
        if x[i] >= 0.4 and x[i] < 0.5:
            x[i] = 10
            continue
        if x[i] >= 0.5 and x[i] < 0.6:
            x[i] = 11
            continue
        if x[i] >= 0.6 and x[i] < 0.7:
            x[i] = 12
            continue
        if x[i] >= 0.7 and x[i] < 0.8:
            x[i] = 13
            continue
        if x[i] >= 0.8 and x[i] < 0.9:
            x[i] = 14
            continue
        if x[i] >= 0.9 and x[i] < 1.0:
            x[i] = 15
            continue
        if x[i] >= 1.0 and x[i] < 2.0:
            x[i] = 16
            continue
        if x[i] >= 2.0 and x[i] < 3.0:
            x[i] = 17
            continue
        if x[i] >= 3.0 and x[i] < 4.0:
            x[i] = 18
            continue
        if x[i] >= 4.0 and x[i] < 5.0:
            x[i] = 19
            continue
        if x[i] >= 5.0 and x[i] < 6.0:
            x[i] = 20
            continue
        if x[i] >= 6.0 and x[i] < 7.0:
            x[i] = 21
            continue
        if x[i] >= 7.0 and x[i] < 8.0:
            x[i] = 22
            continue
        if x[i] >= 8.0 and x[i] < 9.0:
            x[i] = 23
            continue
        if x[i] >= 9.0 and x[i] < 10.0:
            x[i] = 24
            continue
        if x[i] >= 10.0 and x[i] < 11.0:
            x[i] = 25
            continue
        if x[i] >= 11.0 and x[i] < 12.0:
            x[i] = 26
            continue
        if x[i] >= 12.0 and x[i] < 13.0:
            x[i] = 27
            continue
        if x[i] >= 13.0 and x[i] < 14.0:
            x[i] = 28
            continue
        if x[i] >= 14.0 and x[i] < 15.0:
            x[i] = 29
            continue
        if x[i] >= 15.0 and x[i] < 20.0:
            x[i] = 30
            continue
        if x[i] >= 20.0 and x[i] < 25.0:
            x[i] = 31
            continue
        if x[i] >= 25.0 and x[i] < 35.0:
            x[i] = 32
            continue
        if x[i] >= 35.0 and x[i] < 55.0:
            x[i] = 33
            continue
        if x[i] >= 55.0 and x[i] < 100.0:
            x[i] = 34
            continue
        if x[i] >= 100.0 and x[i] < 200.0:
            x[i] = 35
            continue
        if x[i] >= 200.0:
            x[i] = 36
            continue
    return x


def prepare_peak_token(peak_path, out_feather_path):
    """
    Pre-compute the tokenized peak sequences for the CREFormer model.

    peak_path: str
        The path to the peak file in BED format.
    out_feather_path: str
        The path to save the feather file containing the tokenized peak sequences
    """
    import ray

    pkg_dir = get_package_dir()
    genome_dict_path = f"{pkg_dir}/pkg_data/creformer/kmer_dict.pkl"
    genome_dict = torch.load(genome_dict_path, weights_only=True)
    remote_genome_dict = ray.put(genome_dict)
    genome = Genome("mm10")
    peak_bed = genome.standard_region_length(peak_path, 1024)
    peak_bed["End"] = peak_bed["Start"] + 1029  # 1024+5

    @ray.remote
    def process(peaks, dna_one_hot, genome_dict):
        all_seq = []
        for peak_one_hot in dna_one_hot:
            seq = one_hot_decoding(peak_one_hot)
            seq_in = _dna_pre_processing(_dna_tokenizer(seq), genome_dict)
            all_seq.append(seq_in)
        all_seq = np.array(all_seq).astype("uint16")
        return pd.DataFrame(all_seq, index=peaks)

    chunk = 10000
    fs = []
    for chunk_start in range(0, peak_bed.shape[0], chunk):
        peak_chunk = peak_bed.iloc[chunk_start : chunk_start + chunk]
        peaks = peak_chunk["Name"].values
        dna_one_hot = genome.get_regions_one_hot(peak_chunk)
        fs.append(process.remote(peaks, dna_one_hot, remote_genome_dict))

    total = pd.concat(ray.get(fs))
    total.columns = total.columns.astype(str)
    total.reset_index().to_feather(out_feather_path, compression="zstd")
    # shape: (n_peaks, 1024)
    return
