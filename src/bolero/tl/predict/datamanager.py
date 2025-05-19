import pandas as pd
import pyranges as pr

from bolero.pp.genome import Genome


def _validate_region(region_bed: pr.PyRanges, chrom_sizes: dict[str, int]):
    """
    Validate the region bed file.

    1. start < end
    2. region coordinates are within the chromosome sizes
    """
    rdf = region_bed.df
    assert rdf["Start"].min() >= 0, "Start coordinate must be >= 0"

    assert (
        rdf["End"] > rdf["Start"]
    ).all(), "End coordinate must be greater than Start coordinate"

    for chrom, rdf_chrom in rdf.groupby("Chromosome"):
        try:
            size = chrom_sizes[chrom]
        except KeyError as e:
            raise ValueError(f"Chromosome {chrom} not found in chromosome sizes") from e

        assert (
            rdf_chrom["End"].max() <= size
        ), f"End coordinate must be <= {size} for chromosome {chrom}"
    return


class GenericGenomeDataManager:
    def __init__(self, genome: str | Genome, region_bed: pd.DataFrame):
        self.genome = Genome(genome) if isinstance(genome, str) else genome
        self.region_bed = region_bed

        # Signal dataset from parquet or bigwig
        self._parquet_dataset = None
        self._bigwig_dataset = None

        # Mutation information for mutation task
        self._mutation_table = None

    def add_mutations(self):
        """
        Add mutations to the mutation table.
        """
        raise NotImplementedError(
            "add_mutations method not implemented in GenericGenomeDataManager"
        )
