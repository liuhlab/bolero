import pathlib
import subprocess

import h5py
import numpy as np
import pandas as pd
import pyranges as pr
import ray

from bolero import Genome

MODISCO_PIPELINE_TEMPLATE = """
genome = "{GENOME}"

sample_to_dataset_path_dict = {SAMPLE_TO_DATASET_PATH_DICT}
samples = list(sample_to_dataset_path_dict.keys())

tfbs_cutoff = {TFBS_CUTOFF}
modisco_n = {MODISCO_N}
attr_type = "{ATTR_TYPE}"
jaspar_meme_path = "{JASPAR_MEME_PATH}"
if jaspar_meme_path == "None":
    jaspar_meme_path = None
finemo_width=800


rule all:
    input:
        expand("modisco/{{sample}}/modisco.h5", sample=samples),
        expand("modisco/{{sample}}/modisco_report/motifs.html", sample=samples),
        expand("modisco/{{sample}}/finemo_hits/hits.tsv", sample=samples)

rule dump_npz:
    input:
        path=lambda wildcards: sample_to_dataset_path_dict[wildcards.sample],
    output:
        dna_path="modisco/{{sample}}/dna.npz",
        attr_path="modisco/{{sample}}/attr.npz",
        tfbs_path="modisco/{{sample}}/tfbs.npz",
        region_path="modisco/{{sample}}/region.bed"
    params:
        tfbs_cutoff=tfbs_cutoff,
        output_dir="modisco/{{sample}}",
        attr_type=attr_type
    run:
        from bolero.tl.model.scprinter.modisco import dump_npz_from_parquet
        dump_npz_from_parquet(genome, input.path, params.output_dir, tfbs_cutoff=params.tfbs_cutoff, attr_type=params.attr_type)

rule modisco:
    input:
        dna_path="modisco/{{sample}}/dna.npz",
        attr_path="modisco/{{sample}}/attr.npz",
    output:
        modisco_h5_path="modisco/{{sample}}/modisco.h5"
    params:
        modisco_n=modisco_n
    shell:
        "modisco motifs "
        "-s {{input.dna_path}} "
        "-a {{input.attr_path}} "
        "-n {{params.modisco_n}} "
        "-o {{output.modisco_h5_path}}"

rule modisco_report:
    input:
        modisco_h5_path="modisco/{{sample}}/modisco.h5"
    output:
        modisco_report="modisco/{{sample}}/modisco_report/motifs.html"
    params:
        jaspar_meme_path=f"-m {{jaspar_meme_path}}" if jaspar_meme_path is not None else "",
        modisco_report_dir=lambda wildcards: f"modisco/{{wildcards.sample}}/modisco_report"
    shell:
        "modisco report "
        "-i {{input.modisco_h5_path}} "
        "-o {{params.modisco_report_dir}} "
        "-t {{params.jaspar_meme_path}}"

rule finemo_dump:
    input:
        dna_path="modisco/{{sample}}/dna.npz",
        attr_path="modisco/{{sample}}/attr.npz",
    output:
        finemo_path=temp("modisco/{{sample}}/finemo_input.npz")
    params:
        finemo_width=finemo_width
    shell:
        "finemo extract-regions-modisco-fmt "
        "-s {{input.dna_path}} "
        "-a {{input.attr_path}} "
        "-o {{output.finemo_path}} "
        "-w {{params.finemo_width}}"

rule finemo:
    input:
        finemo_path="modisco/{{sample}}/finemo_input.npz",
        modisco_h5_path="modisco/{{sample}}/modisco.h5",
    output:
        finemo_output_path="modisco/{{sample}}/finemo_hits/hits.tsv"
    params:
        output_dir="modisco/{{sample}}/finemo_hits"
    shell:
        "finemo call-hits "
        "-M pp "
        "-r {{input.finemo_path}} "
        "-m {{input.modisco_h5_path}} "
        "-o {{params.output_dir}} "
        "-d cpu"
"""


def dump_npz_from_parquet(
    genome, path, output_dir, tfbs_cutoff=0.2, attr_type="footprint"
):
    """Dump npz files from inference parquet dataset for modisco input."""
    if isinstance(genome, str):
        genome = Genome(genome)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dna_path = output_dir / "dna"
    attr_path = output_dir / "attr"
    tfbs_path = output_dir / "tfbs"
    region_path = output_dir / "region.bed"

    dataset = ray.data.read_parquet(path, file_extensions=["parquet"], concurrency=1)
    # prepare modisco input
    all_dna = []
    all_attr = []
    all_tfbs = []
    all_region = []
    for batch in dataset.iter_batches(batch_size=500):
        dna = batch["dna_one_hot"]
        _attr = batch[f"pred_{attr_type}:attributions"]
        _tfbs = batch[f"pred_{attr_type}:attributions_1d:tfbs"]
        _tfbs = _tfbs.squeeze()
        _region = batch["region"]

        use_rows = _tfbs.max(axis=1) > tfbs_cutoff
        if use_rows.sum() == 0:
            continue

        all_dna.append(dna[use_rows])
        all_attr.append(_attr[use_rows])
        all_tfbs.append(_tfbs[use_rows])
        all_region.append(_region[use_rows])
    all_dna = np.concatenate(all_dna)
    all_attr = np.concatenate(all_attr)
    all_tfbs = np.concatenate(all_tfbs)
    all_region = np.concatenate(all_region)

    np.savez_compressed(dna_path, all_dna.astype("bool"))
    np.savez_compressed(attr_path, all_attr.astype("float32"))
    np.savez_compressed(tfbs_path, all_tfbs.astype("float32"))
    genome.standard_region_length(all_region, all_attr.shape[-1]).to_bed(region_path)
    return


def run_modisco_pipeline(
    genome: str,
    sample_to_dataset_path_dict: dict[str, str],
    output_dir: str = "./",
    tfbs_cutoff: float = 0.2,
    modisco_n: int = 1000000,
    attr_type: str = "footprint",
    jaspar_meme_path: str = None,
    cpu: int = 16,
) -> None:
    """
    Run modisco and finemo pipeline.

    Parameters
    ----------
    genome : str
        The genome to use for the pipeline.
    sample_to_dataset_path_dict : Dict[str, str]
        A dictionary mapping sample names to dataset paths.
    output_dir : str, optional
        The output directory for the pipeline (default is './').
    tfbs_cutoff : float, optional
        The cutoff value for TFBS (default is 0.2).
    modisco_n : int, optional
        The number of modisco iterations (default is 1000000).
    attr_type : str, optional
        The attribute type for modisco (default is 'footprint').
    jaspar_meme_path : str, optional
        The path to the JASPAR MEME file (default is None).
    cpu : int, optional
        The number of CPUs to use for the pipeline (default is 16).

    Returns
    -------
    None
    """
    snakefile_path = pathlib.Path(output_dir) / "Snakefile"
    if not snakefile_path.exists():
        pipeline = MODISCO_PIPELINE_TEMPLATE.format(
            GENOME=genome,
            SAMPLE_TO_DATASET_PATH_DICT=sample_to_dataset_path_dict,
            TFBS_CUTOFF=tfbs_cutoff,
            MODISCO_N=modisco_n,
            ATTR_TYPE=attr_type,
            JASPAR_MEME_PATH=jaspar_meme_path,
        )
        with open("Snakefile", "w") as f:
            f.write(pipeline)
    else:
        print("Snakefile already exists. Skipping writing Snakefile.")

    subprocess.run(
        [
            "snakemake",
            "-s",
            snakefile_path,
            "-j",
            str(cpu),
            "-d",
            output_dir,
            "--keep-going",
        ]
    )
    return


def parse_finemo_results(output_dir):
    """Parse finemo results."""
    output_dir = pathlib.Path(output_dir)
    finemo_hits = pd.read_table(output_dir / "finemo_hits/hits.tsv")
    region_bed = pr.read_bed(str(output_dir / "region.bed"), as_df=True)

    use_bed = region_bed.reindex(finemo_hits["peak_id"].values).reset_index(drop=True)
    finemo_hits["chr"] = use_bed["Chromosome"]
    finemo_hits["start"] += use_bed["Start"]
    finemo_hits["end"] += use_bed["Start"]
    finemo_hits.rename(
        columns={"chr": "Chromosome", "start": "Start", "end": "End"}, inplace=True
    )
    finemo_hits["start_untrimmed"] += use_bed["Start"]
    finemo_hits["end_untrimmed"] += use_bed["Start"]
    finemo_hits["peak_name"] = use_bed["Name"]

    finemo_hits.to_csv(
        output_dir / "finemo_hits.bed.gz", sep="\t", index=False, header=True
    )
    return finemo_hits


class ModiscoPattern:
    """
    Represents a Modisco pattern.

    Attributes
    ----------
        name (str): The name of the pattern.
        sequence (np.ndarray): The sequence of the pattern.
        contrib_scores (np.ndarray): The contribution scores of the pattern.
        hypothetical_contribs (np.ndarray): The hypothetical contribution scores of the pattern.
        seqlets (dict[np.ndarray]): The sequencelets of the pattern.
        subpatterns (list[ModiscoPattern]): The subpatterns of the pattern.
        tomtom_df (pd.DataFrame, optional): The TOMTOM dataframe associated with the pattern.
    """

    def __init__(
        self,
        name: str,
        sequence: np.ndarray,
        contrib_scores: np.ndarray,
        hypothetical_contribs: np.ndarray,
        seqlets: dict[np.ndarray],
        subpatterns: list["ModiscoPattern"],
        tomtom_df: pd.DataFrame = None,
    ):
        self.name: str = name
        self.sequence: np.ndarray = sequence
        self.contrib_scores: np.ndarray = contrib_scores
        self.hypothetical_contribs: np.ndarray = hypothetical_contribs
        self.seqlets: dict[np.ndarray] = seqlets
        self.subpatterns: list["ModiscoPattern"] = subpatterns
        self.tomtom_df: pd.DataFrame = tomtom_df

    @classmethod
    def from_modisco_group(cls, name, h5_group, tomtom_path):
        """
        Create a ModiscoPattern object from a modisco group.

        Args:
            name (str): The name of the pattern.
            h5_group: The modisco group.
            tomtom_path (str): The path to the TOMTOM file.

        Returns
        -------
            ModiscoPattern: The created ModiscoPattern object.
        """
        sequence = np.array(h5_group["sequence"], dtype=np.bool)
        contrib_scores = np.array(h5_group["contrib_scores"], dtype=np.float32)
        hypothetical_contribs = np.array(
            h5_group["hypothetical_contribs"], dtype=np.float32
        )
        seqlets = {k: np.array(v) for k, v in h5_group["seqlets"].items()}

        subpatterns = []
        for subpattern_name, subpattern_group in h5_group.items():
            if subpattern_name.startswith("subpattern_"):
                subpatterns.append(
                    cls.from_modisco_group(subpattern_name, subpattern_group)
                )

        tomtom = (
            pd.read_table(tomtom_path, comment="#") if tomtom_path is not None else None
        )

        return cls(
            name,
            sequence,
            contrib_scores,
            hypothetical_contribs,
            seqlets,
            subpatterns,
            tomtom,
        )


class ModiscoResults:
    """
    Represents the results of a Modisco analysis.

    Attributes
    ----------
        output_dir (str): The output directory of the analysis.
        pos_patterns (list[ModiscoPattern]): The positive patterns identified in the analysis.
        neg_patterns (list[ModiscoPattern]): The negative patterns identified in the analysis.
        _dna_one_hot (np.ndarray): The one-hot encoded DNA sequences.
        _attr (np.ndarray): The attribute scores.
        _attr_1d (np.ndarray): The 1D attribute scores.
        _tfbs (np.ndarray): The TFBS scores.
        _region (pd.DataFrame): The region information.
        _hits (pd.DataFrame): The motif hits.
    """

    def __init__(
        self,
        output_dir: str,
        pos_patterns: list["ModiscoPattern"],
        neg_patterns: list["ModiscoPattern"],
    ):
        """
        Initialize ModiscoResults.

        Args:
            output_dir (str): The output directory of the analysis.
            pos_patterns (list[ModiscoPattern]): The positive patterns identified in the analysis.
            neg_patterns (list[ModiscoPattern]): The negative patterns identified in the analysis.
        """
        self.output_dir: str = output_dir
        self.pos_patterns: list["ModiscoPattern"] = pos_patterns
        self.neg_patterns: list["ModiscoPattern"] = neg_patterns

        self._dna_one_hot: np.ndarray = None
        self._attr: np.ndarray = None
        self._attr_1d: np.ndarray = None
        self._tfbs: np.ndarray = None
        self._region: pd.DataFrame = None
        self._hits: pd.DataFrame = None
        return

    @classmethod
    def from_modisco_h5(cls, output_dir: str) -> "ModiscoResults":
        """
        Create a ModiscoResults object from a modisco.h5 file.

        Args:
            output_dir (str): The output directory of the analysis.

        Returns
        -------
            ModiscoResults: The created ModiscoResults object.
        """
        output_dir = pathlib.Path(output_dir)
        h5_path = output_dir / "modisco.h5"
        report_dir = output_dir / "modisco_report"

        with h5py.File(h5_path, "r") as h5:
            pos_patterns = []
            neg_patterns = []
            for type_name, type_group in h5.items():
                _patterns = []
                for name, pattern_group in type_group.items():
                    if name.startswith("pattern_"):
                        tomtom_path = (
                            report_dir / f"tomtom/{type_name}.{name}.tomtom.tsv"
                        )
                        tomtom_path = tomtom_path if tomtom_path.exists() else None

                        _patterns.append(
                            ModiscoPattern.from_modisco_group(
                                name, pattern_group, tomtom_path
                            )
                        )
                if type_name == "pos_patterns":
                    pos_patterns = _patterns
                elif type_name == "neg_patterns":
                    neg_patterns = _patterns
        return cls(output_dir, pos_patterns, neg_patterns)

    @property
    def dna_one_hot(self) -> np.ndarray:
        """
        Get the one-hot encoded DNA sequences.

        Returns
        -------
            np.ndarray: The one-hot encoded DNA sequences.
        """
        if self._dna_one_hot is None:
            self._dna_one_hot = np.load(self.output_dir / "dna.npz")["arr_0"]
        return self._dna_one_hot

    @property
    def attr(self) -> np.ndarray:
        """
        Get the attribute scores.

        Returns
        -------
            np.ndarray: The attribute scores.
        """
        if self._attr is None:
            self._attr = np.load(self.output_dir / "attr.npz")["arr_0"]
        return self._attr

    @property
    def attr_1d(self) -> np.ndarray:
        """
        Get the 1D attribute scores.

        Returns
        -------
            np.ndarray: The 1D attribute scores.
        """
        if self._attr_1d is None:
            self._attr_1d = (self.attr * self.dna_one_hot).sum(axis=1)
        return self._attr_1d

    @property
    def tfbs(self) -> np.ndarray:
        """
        Get the TFBS scores.

        Returns
        -------
            np.ndarray: The TFBS scores.
        """
        if self._tfbs is None:
            self._tfbs = np.load(self.output_dir / "tfbs.npz")["arr_0"]
        return self._tfbs

    @property
    def region(self) -> pd.DataFrame:
        """
        Get the region information.

        Returns
        -------
            pd.DataFrame: The region information.
        """
        if self._region is None:
            self._region = pr.read_bed(self.output_dir / "region.bed", as_df=True)
        return self._region

    @property
    def motif_hits(self) -> pd.DataFrame:
        """
        Get the motif hits.

        Returns
        -------
            pd.DataFrame: The motif hits.
        """
        if self._hits is None:
            self._hits = pr.read_bed(self.output_dir / "finemo_hits.bed.gz", as_df=True)
        return self._hits
