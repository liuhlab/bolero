import pathlib
import warnings

import h5py
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr
import ray
import seaborn as sns
import torch
from modiscolite.core import Seqlet

from bolero import Genome

MODISCO_PIPELINE_TEMPLATE = """
genome = "{GENOME}"

sample_dirs = {SAMPLE_DIRS}

modisco_n = {MODISCO_N}
jaspar_meme_path = "{JASPAR_MEME_PATH}"
if jaspar_meme_path == "None":
    jaspar_meme_path = None

rule all:
    input:
        expand("{{sample_dir}}/modisco.h5", sample_dir=sample_dirs),
        expand("{{sample_dir}}/modisco_report/motifs.html", sample_dir=sample_dirs),

rule modisco:
    input:
        dna_path="{{sample_dir}}/dna.npz",
        attr_path="{{sample_dir}}/attr.npz",
    output:
        modisco_h5_path="{{sample_dir}}/modisco.h5"
    params:
        modisco_n=modisco_n
    threads:
        15
    shell:
        "modisco motifs "
        "-s '{{input.dna_path}}' "
        "-a '{{input.attr_path}}' "
        "-n {{params.modisco_n}} "
        "-o '{{output.modisco_h5_path}}'"

rule modisco_report:
    input:
        modisco_h5_path="{{sample_dir}}/modisco.h5"
    output:
        modisco_report="{{sample_dir}}/modisco_report/motifs.html"
    params:
        jaspar_meme_path=f"-m {{jaspar_meme_path}}" if jaspar_meme_path is not None else "",
        modisco_report_dir=lambda wildcards: f"{{wildcards.sample_dir}}/modisco_report"
    shell:
        "modisco report "
        "-i '{{input.modisco_h5_path}}' "
        "-o '{{params.modisco_report_dir}}' "
"""

FINEMO_PIPELINE_TEMPLATE = """
sample_dirs = {SAMPLE_DIRS}
finemo_width= {FINEMO_WIDTH}

rule all:
    input:
        expand("{{sample_dir}}/finemo_hits/hits.tsv", sample_dir=sample_dirs)

rule finemo_dump:
    input:
        dna_path="{{sample_dir}}/dna.npz",
        attr_path="{{sample_dir}}/attr.npz",
    output:
        finemo_path=temp("{{sample_dir}}/finemo_input.npz")
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
        finemo_path="{{sample_dir}}/finemo_input.npz",
        modisco_h5_path="{{sample_dir}}/modisco.h5",
    output:
        finemo_output_path="{{sample_dir}}/finemo_hits/hits.tsv"
    params:
        output_dir="{{sample_dir}}/finemo_hits"
    resources:
        gpu_slots=1
    shell:
        "finemo call-hits "
        "-M pp "
        "-r {{input.finemo_path}} "
        "-m {{input.modisco_h5_path}} "
        "-o {{params.output_dir}} "
        "-d cuda"
"""


def dump_npz_from_parquet(
    genome, path, output_dir, tfbs_cutoff=0.2, attr_type="footprint", concurrency=1
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

    dataset = ray.data.read_parquet(
        path, file_extensions=["parquet"], concurrency=concurrency
    )
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

    bed_df = genome.standard_region_length(all_region, all_attr.shape[-1])
    bed_df.to_csv(region_path, sep="\t", index=False, header=False)

    print(f"Saved data for {len(all_region)} regions.")
    return


def prepare_modisco_pipeline(
    genome: str,
    finemo_width: int,
    sample_dirs: list[str],
    output_dir: str = "./",
    snakefile_suffix="",
    modisco_n: int = 100000,
    jaspar_meme_path: str = None,
    cpu: int = 16,
) -> None:
    """
    Prepare modisco and finemo pipeline.

    Parameters
    ----------
    genome : str
        The genome to use for the pipeline.
    finemo_width : int
        The width of the finemo regions.
    sample_dirs : list[str]
        A list of sample_dir containing modisco input npz files extracted by dump_npz_from_parquet
    output_dir : str, optional
        The output directory for the pipeline (default is './').
    modisco_n : int, optional
        The number of modisco iterations (default is 1000000).
    jaspar_meme_path : str, optional
        The path to the JASPAR MEME file (default is None).
    cpu : int, optional
        The number of CPUs to use for the pipeline (default is 16).

    Returns
    -------
    None
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    snakefile_path = output_dir / f"Snakefile{snakefile_suffix}"
    if not snakefile_path.exists():
        pipeline = MODISCO_PIPELINE_TEMPLATE.format(
            GENOME=genome,
            SAMPLE_DIRS=[str(path) for path in sample_dirs],
            MODISCO_N=modisco_n,
            JASPAR_MEME_PATH=jaspar_meme_path,
        )
        with open(snakefile_path, "w") as f:
            f.write(pipeline)
    else:
        print("Snakefile already exists. Skipping writing Snakefile.")

    snakefile_finemo_path = output_dir / f"Snakefile_finemo{snakefile_suffix}"
    if not snakefile_finemo_path.exists():
        pipeline = FINEMO_PIPELINE_TEMPLATE.format(
            SAMPLE_DIRS=[str(path) for path in sample_dirs], FINEMO_WIDTH=finemo_width
        )
        with open(snakefile_finemo_path, "w") as f:
            f.write(pipeline)
    else:
        print("Snakefile_finemo already exists. Skipping writing Snakefile.")

    cmd = f"snakemake -s {snakefile_path} -j {cpu} -d {output_dir} --keep-going"
    cmd2 = f"snakemake -s {snakefile_finemo_path} -j {cpu} -d {output_dir} --resources gpu_slots=1 --keep-going"
    return cmd, cmd2


def parse_finemo_results(output_dir):
    """Parse finemo results."""
    output_dir = pathlib.Path(output_dir)
    finemo_hits = pd.read_table(output_dir / "finemo_hits/hits.tsv")
    region_bed = pr.read_bed(str(output_dir / "region.bed"), as_df=True)

    use_bed = region_bed.reindex(finemo_hits["peak_id"].values).reset_index(drop=True)
    finemo_hits["chr"] = use_bed["Chromosome"]
    finemo_hits["Start"] = finemo_hits["start"] + use_bed["Start"]
    finemo_hits["End"] = finemo_hits["end"] + use_bed["Start"]
    finemo_hits.rename(
        columns={"chr": "Chromosome", "start": "rel_start", "end": "rel_end"},
        inplace=True,
    )
    finemo_hits["start_untrimmed"] += use_bed["Start"]
    finemo_hits["end_untrimmed"] += use_bed["Start"]
    finemo_hits["peak_name"] = use_bed["Name"]

    finemo_hits.to_csv(
        output_dir / "finemo_hits.bed.gz", sep="\t", index=False, header=True
    )
    return finemo_hits


class ModiscoSeqlet(Seqlet):
    def __init__(
        self,
        name,
        example_idx,
        start,
        end,
        is_revcomp,
        sequence=None,
        contrib_scores=None,
        hypothetical_contribs=None,
    ):
        self.name = name
        super().__init__(example_idx, start, end, is_revcomp)

        self.sequence = sequence  # shape (len, 4)
        self.contrib_scores = contrib_scores  # shape (len, 4)
        self.hypothetical_contribs = hypothetical_contribs  # shape (len, 4)

    def get_trimed_matrix(self, key="sequence", trim_threshold=0.3, additional_flank=0):
        """
        Trim the sequence by contribution score > max(contrib_scores) * trim_threshold.
        """
        try:
            ppm = getattr(self, key)
        except AttributeError as e:
            raise AttributeError(
                f"Key '{key}' not found in ModiscoSeqlet, "
                f"possible keys are ['sequence', 'contrib_scores', 'hypothetical_contribs']"
            ) from e

        # filter by information content
        ic = self._get_info_content(self.sequence)
        pass_ic = ic >= trim_threshold

        # filter by contribution score
        cwm = self.contrib_scores
        score = np.sum(np.abs(cwm), axis=1)
        trim_thresh = (
            np.max(score) * trim_threshold
        )  # Cut off anything less than 30% of max score
        pass_contrib = score >= trim_thresh

        pass_inds = np.where(pass_ic & pass_contrib)[0]
        if len(pass_inds) == 0:
            trimmed = np.zeros((0, 4))
        else:
            s = max(np.min(pass_inds) - additional_flank, 0)
            e = min(np.max(pass_inds) + additional_flank + 1, len(ppm))
            trimmed = ppm[s:e]
        return trimmed

    def get_tangermeme_input(self, key="sequence", trim_threshold=0.3):
        """
        Get the sequence and contribution scores for TangerMEME input.
        """
        data = self.get_trimed_matrix(key=key, trim_threshold=trim_threshold)
        data = torch.from_numpy(data.T)
        return data

    @staticmethod
    def _get_info_content(pwm):
        """
        Calculate the information content of a PWM.
        """
        pwm = pwm.T  # shape (4, len)
        entropy = -np.sum(pwm * np.log2(pwm + 1e-10), axis=0)
        max_ic = np.log2(4)  # max possible information content for 4 bases (DNA)
        info_content = max_ic - entropy
        return info_content

    def plot_on_ax(self, ax, key="sequence", trim_threshold=0.3, **kwargs):
        """
        Plot motif logo on an Axes
        """
        data = self.get_trimed_matrix(key=key, trim_threshold=trim_threshold)
        if key == "sequence":
            data_info = self._get_info_content(data)
            data = data * data_info[:, None]

        # Create a sequence logo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            data = pd.DataFrame(data, columns=list("ACGT"))
            logo = logomaker.Logo(data, ax=ax, **kwargs)
        return logo

    def plot(self, key="sequence", trim_threshold=0.3):
        """Visualize the motif."""
        data = self.get_trimed_matrix(key=key, trim_threshold=trim_threshold)

        _, ax = plt.subplots(figsize=(len(data) / 4, 1), dpi=100)
        self.plot_on_ax(ax, key=key, trim_threshold=trim_threshold)
        ax.set_title(f"{self.name}", fontsize=8)
        if key == "sequence":
            ax.set_ylabel("IC (Bits)", fontsize=8)
        else:
            ax.set_ylabel("Attribution", fontsize=8)
        sns.despine(ax=ax)
        return

    def get_consensus_sequence(self, trim_threshold=0.3, rc=False):
        """
        Get the consensus sequence of the motif.

        Returns
        -------
        - str: The consensus sequence.
        """
        seq = self.get_trimed_matrix(key="sequence", trim_threshold=trim_threshold)
        consensus = (
            pd.DataFrame(seq, columns=("A", "C", "G", "T")).idxmax(axis=1).values
        )
        consensus = "".join(consensus)

        if rc:
            consensus = consensus.translate(str.maketrans("AGCT", "TCGA"))[::-1]
        return consensus


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
        self.subclusters = None
        return

    def to_seqlet(self):
        """Return a modisco compatible seqlet object"""
        return ModiscoSeqlet(
            name=self.name,
            example_idx=None,
            start=None,
            end=None,
            is_revcomp=False,
            sequence=self.sequence,
            contrib_scores=self.contrib_scores,
            hypothetical_contribs=self.hypothetical_contribs,
        )

    def get_all_seqlet_instances(self):
        """Get all seqlet instances of the pattern."""
        instances = []
        for i, (sequence, contrib_scores, hypothetical_contribs) in enumerate(
            zip(
                self.seqlets["sequence"],
                self.seqlets["contrib_scores"],
                self.seqlets["hypothetical_contribs"],
            )
        ):
            instances.append(
                ModiscoSeqlet(
                    name=f"{self.name}_{i}",
                    example_idx=i,
                    start=None,
                    end=None,
                    is_revcomp=False,
                    sequence=sequence,
                    contrib_scores=contrib_scores,
                    hypothetical_contribs=hypothetical_contribs,
                )
            )
        return instances

    @classmethod
    def from_modisco_group(
        cls,
        name,
        h5_group,
        tomtom_path=None,
        load_subpatterns=False,
        load_seqlets=False,
    ):
        """
        Create a ModiscoPattern object from a modisco group.

        Args:
            name (str): The name of the pattern.
            h5_group: The modisco group.
            tomtom_path (str): The path to the TOMTOM file.
            load_subpatterns (bool): Whether to load subpatterns.

        Returns
        -------
            ModiscoPattern: The created ModiscoPattern object.
        """
        sequence = np.array(h5_group["sequence"])
        contrib_scores = np.array(h5_group["contrib_scores"], dtype=np.float32)
        hypothetical_contribs = np.array(
            h5_group["hypothetical_contribs"], dtype=np.float32
        )
        if load_seqlets:
            seqlets = {k: np.array(v) for k, v in h5_group["seqlets"].items()}
        else:
            seqlets = {}

        subpatterns = []
        if load_subpatterns:
            for subpattern_name, subpattern_group in h5_group.items():
                if subpattern_name.startswith("subpattern_"):
                    subpatterns.append(
                        cls.from_modisco_group(
                            subpattern_name,
                            subpattern_group,
                            tomtom_path=None,
                            load_subpatterns=False,
                            load_seqlets=load_seqlets,
                        )
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


def _get_patterns_from_h5(
    h5_path, tomtom_path=None, load_subpatterns=False, load_seqlets=False
):
    with h5py.File(h5_path, "r") as h5:
        pos_patterns = []
        neg_patterns = []
        for type_name, type_group in h5.items():
            _patterns = []
            for name, pattern_group in type_group.items():
                if name.startswith("pattern_"):
                    _patterns.append(
                        ModiscoPattern.from_modisco_group(
                            name,
                            pattern_group,
                            tomtom_path=tomtom_path,
                            load_subpatterns=load_subpatterns,
                            load_seqlets=load_seqlets,
                        )
                    )
            if type_name == "pos_patterns":
                pos_patterns = _patterns
            elif type_name == "neg_patterns":
                neg_patterns = _patterns
    return pos_patterns, neg_patterns


class ModiscoResults:
    """
    Represents the results of a Modisco analysis.

    Attributes
    ----------
        output_dir (str): The output directory of the analysis.
        _pos_patterns (list[ModiscoPattern]): The positive patterns identified in the analysis.
        _neg_patterns (list[ModiscoPattern]): The negative patterns identified in the analysis.
        _dna_one_hot (np.ndarray): The one-hot encoded DNA sequences.
        _attr (np.ndarray): The attribute scores.
        _attr_1d (np.ndarray): The 1D attribute scores.
        _tfbs (np.ndarray): The TFBS scores.
        _region (pd.DataFrame): The region information.
        _hits (pd.DataFrame): The motif hits.
    """

    def __init__(
        self, output_dir: str, load_h5_subpatterns=False, load_h5_seqlets=False
    ):
        """
        Initialize ModiscoResults.

        Args:
            output_dir (str): The output directory of the analysis.
        """
        self.output_dir: str = output_dir
        self._pos_patterns: list["ModiscoPattern"] = None
        self._neg_patterns: list["ModiscoPattern"] = None
        self._dna_one_hot: np.ndarray = None
        self._attr: np.ndarray = None
        self._attr_1d: np.ndarray = None
        self._tfbs: np.ndarray = None
        self._region: pd.DataFrame = None
        self._hits: pd.DataFrame = None
        self._load_h5_subpatterns = load_h5_subpatterns
        self._load_h5_seqlets = load_h5_seqlets
        return

    def _get_patterns_from_h5(self):
        pos_patterns, neg_patterns = _get_patterns_from_h5(
            self.output_dir / "modisco.h5",
            load_subpatterns=self._load_h5_subpatterns,
            load_seqlets=self._load_h5_seqlets,
        )
        return pos_patterns, neg_patterns

    @property
    def neg_patterns(self):
        """Modisco patterns with negative attribution scores."""
        if self._neg_patterns is None:
            self._pos_patterns, self._neg_patterns = self._get_patterns_from_h5()
        return self._neg_patterns

    @property
    def neg_seqlets(self):
        """Get the representative seqlet for each negative pattern."""
        return [p.to_seqlet() for p in self.neg_patterns]

    @property
    def pos_patterns(self):
        """Modisco patterns with positive attribution scores."""
        if self._pos_patterns is None:
            self._pos_patterns, self._neg_patterns = self._get_patterns_from_h5()
        return self._pos_patterns

    @property
    def pos_seqlets(self):
        """Get the representative seqlet for each positive pattern."""
        return [p.to_seqlet() for p in self.pos_patterns]

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
            self._region = pr.read_bed(str(self.output_dir / "region.bed"), as_df=True)
        return self._region

    def get_motif_hits(self, tfbs=False, attr_1d=False, slop=30) -> pd.DataFrame:
        """
        Get the motif hits.

        Returns
        -------
            pd.DataFrame: The motif hits.
        """
        if self._hits is None:
            self._hits = parse_finemo_results(self.output_dir)
            if tfbs:
                self._annotate_score_to_hits(slop=slop, score="tfbs", reduce="max")
            if attr_1d:
                self._annotate_score_to_hits(slop=slop, score="attr_1d", reduce="mean")
        return self._hits

    def _annotate_score_to_hits(self, slop=30, score="tfbs", reduce="max"):
        """Add max tfbs scores to hit regions."""
        if score == "tfbs":
            _score = self.tfbs
        elif score == "attr_1d":
            _score = self.attr_1d

        max_len = _score.shape[1]
        motif_score_col = []
        for _, (peak_id, rstart, rend) in self._hits[
            ["peak_id", "rel_start", "rel_end"]
        ].iterrows():
            motif_scores = _score[
                peak_id, max(rstart - slop, 0) : min(rend + slop, max_len)
            ]
            if reduce == "max":
                motif_scores = motif_scores.max()
            elif reduce == "mean":
                motif_scores = motif_scores.mean()
            elif reduce == "min":
                motif_scores = motif_scores.min()
            motif_score_col.append(motif_scores)
        self._hits[f"motif_{score}_{reduce}"] = motif_score_col
        return


class ModiscoHDF:
    def __init__(self, hdf5_path, load_subpatterns=False, load_seqlets=False):
        """
        Initialize ModiscoHDF.

        Args:
            hdf5_path (str): The path to the HDF5 file.
        """
        self.hdf5_path = hdf5_path
        self.load_subpatterns = load_subpatterns
        self.load_seqlets = load_seqlets
        self.pos_patterns, self.neg_patterns = self._get_patterns_from_h5()

    def _get_patterns_from_h5(self):
        """
        Get patterns from the HDF5 file.

        Returns
        -------
            tuple: A tuple containing positive and negative patterns.
        """
        pos_patterns, neg_patterns = _get_patterns_from_h5(
            self.hdf5_path,
            load_subpatterns=self.load_subpatterns,
            load_seqlets=self.load_seqlets,
        )
        return pos_patterns, neg_patterns
