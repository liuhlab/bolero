from typing import Dict, Union

hg38_splits = [None] * 5
hg38_splits[0] = {
    "test": ["chr1", "chr3", "chr6"],
    "valid": ["chr8", "chr20"],
    "train": [
        "chr2",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr21",
        "chr22",
        "chrX",
        "chrY",
    ],
}
hg38_splits[1] = {
    "test": ["chr2", "chr8", "chr9", "chr16"],
    "valid": ["chr12", "chr17"],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr11",
        "chr13",
        "chr14",
        "chr15",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
        "chrY",
    ],
}
hg38_splits[2] = {
    "test": ["chr4", "chr11", "chr12", "chr15", "chrY"],
    "valid": ["chr22", "chr7"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr10",
        "chr13",
        "chr14",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chrX",
    ],
}
hg38_splits[3] = {
    "test": ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
    "valid": ["chr6", "chr21"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        "chrY",
    ],
}
hg38_splits[4] = {
    "test": ["chr7", "chr13", "chr17", "chr19", "chr21", "chrX"],
    "valid": ["chr10", "chr18"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr14",
        "chr15",
        "chr16",
        "chr20",
        "chr22",
        "chrY",
    ],
}


mm10_splits = [None] * 5
mm10_splits[0] = {
    "test": ["chr1", "chr6", "chr12", "chr13", "chr16"],
    "valid": ["chr8", "chr11", "chr18", "chr19", "chrX"],
    "train": [
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr14",
        "chr15",
        "chr17",
    ],
}
mm10_splits[1] = {
    "test": ["chr2", "chr7", "chr10", "chr14", "chr17"],
    "valid": ["chr5", "chr9", "chr13", "chr15", "chrY"],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr6",
        "chr8",
        "chr11",
        "chr12",
        "chr16",
        "chr18",
        "chr19",
        "chrX",
    ],
}
mm10_splits[2] = {
    "test": ["chr3", "chr8", "chr13", "chr15", "chr17"],
    "valid": ["chr2", "chr9", "chr11", "chr12", "chrY"],
    "train": [
        "chr1",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr14",
        "chr16",
        "chr18",
        "chr19",
        "chrX",
    ],
}
mm10_splits[3] = {
    "test": ["chr4", "chr9", "chr11", "chr14", "chr19"],
    "valid": ["chr1", "chr7", "chr12", "chr13", "chrY"],
    "train": [
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr10",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chrX",
    ],
}
mm10_splits[4] = {
    "test": ["chr5", "chr10", "chr12", "chr16", "chrY"],
    "valid": ["chr3", "chr7", "chr14", "chr15", "chr18"],
    "train": [
        "chr1",
        "chr2",
        "chr4",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr13",
        "chr17",
        "chr19",
        "chrX",
    ],
}


def get_splits(genome: str, split_id: int) -> Dict[str, Union[list, None]]:
    """
    Get the splits for a given genome and split ID.

    Parameters
    ----------
        genome (str): The genome (either "hg38" or "mm10").
        split_id (int): The split ID (0 to 4).

    Returns
    -------
        dict: A dictionary containing the splits for the given genome and split ID.
              The dictionary has keys "test", "valid", and "train", each mapping to a list of chromosome names.
              The key "test" maps to the chromosomes used for testing,
              the key "valid" maps to the chromosomes used for validation,
              and the key "train" maps to the chromosomes used for training.

    Raises
    ------
        ValueError: If the split ID is invalid or the genome is unknown.
    """
    if split_id < 0 or split_id >= 5:
        raise ValueError(f"Invalid split_id {split_id}")
    if genome == "hg38":
        return hg38_splits[split_id]
    elif genome == "mm10":
        return mm10_splits[split_id]
    else:
        raise ValueError(f"Unknown genome {genome}")
