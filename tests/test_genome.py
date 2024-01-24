import os

import pandas as pd

from bolero.pp.genome import Genome


def test_genome():
    genome = Genome("sacCer3")
    assert genome.fasta_path.exists()
    assert genome.chrom_sizes_path.exists()

    test_bed = pd.DataFrame(
        {"chrom": ["chrI", "chrII", "chrIII"], "start": [1, 1, 1], "end": [10, 10, 10], "id": ["a", "b", "c"]}
    )
    test_bed_path = "test.bed"
    test_bed.to_csv(test_bed_path, sep="\t", header=False, index=False)
    genome.get_region_fasta(test_bed_path, output_path="test.fa", compress=True)
    assert os.path.exists("test.fa.gz")
    os.remove("test.fa.gz")
    os.remove("test.bed")
