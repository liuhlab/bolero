# Genome & Sequence

Assembly handles and DNA utilities. `Genome` resolves a UCSC assembly to its FASTA /
chrom.sizes, serves DNA as sequence or one-hot arrays, and exposes global coordinates and the
blacklist. `Sequence` wraps a single DNA string with one-hot helpers, and `understand_regions`
normalizes region specifications used throughout the package.

::: bolero.pp.genome.Genome

::: bolero.pp.seq.Sequence

::: bolero.utils.understand_regions
