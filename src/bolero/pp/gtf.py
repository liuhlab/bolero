import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gffutils import Feature, FeatureDB, FeatureNotFoundError, create_db


def universal_mapping(input, mapping: dict):
    """Map input to output using a mapping dict with support for various input types."""
    if isinstance(input, str):
        return mapping[input]
    elif isinstance(input, list):
        return [mapping[i] for i in input]
    elif isinstance(input, set):
        return {mapping[i] for i in input}
    elif isinstance(input, np.ndarray):
        return np.array([mapping[i] for i in input])
    elif isinstance(input, torch.Tensor):
        return torch.tensor([mapping[i] for i in input.numpy()])
    elif isinstance(input, pd.Series):
        return input.map(mapping)
    elif isinstance(input, pd.Index):
        return input.map(mapping)
    else:
        raise ValueError(f"Unsupported input type: {type(input)}")


def features_to_bed(features):
    """Convert gffutils.Feature to bed format."""
    columns = ["Chromosome", "Start", "End", "Name", "Strand", "GeneID", "FeatureType"]
    if len(features) == 0:
        # empty bed with same columns
        return pd.DataFrame(columns=columns)

    bed_df = []
    for f in features:
        bed_df.append(
            {
                "Chromosome": f.chrom,
                "Start": f.start - 1,
                "End": f.end,
                "Name": f.id,
                "Strand": f.strand,
                "GeneID": f["gene_id"][0],
                "FeatureType": f.featuretype,
            }
        )
    bed_df = pd.DataFrame(bed_df)[columns].copy()
    return bed_df


class GTFDB(FeatureDB):
    _gene_id_to_name: dict[str, str]
    _gene_name_to_id: dict[str, str]
    _gene_id_base_to_name: dict[str, str]
    _gene_name_to_id_base: dict[str, str]
    gene_ids: list[str]
    gene_names: list[str]
    transcript_ids: list[str]

    def __init__(self, dbfn, **kwargs):
        super().__init__(dbfn, **kwargs)
        self.db_path = pathlib.Path(dbfn).resolve().absolute()
        self.feature_types = list(self.featuretypes())
        self.chroms = list(self.seqids())

        try:
            basic_info_dict = self._prepare_basic_ids()
            for k, v in basic_info_dict.items():
                setattr(self, k, v)
        except KeyError:
            print("Failed to load basic info.")

        # place holder
        self._gene_bed = None
        return

    def _prepare_basic_ids(self):
        basic_info_path = self.db_path.with_suffix(".basic_info.joblib")
        if basic_info_path.exists():
            basic_info_dict = joblib.load(basic_info_path)
            return basic_info_dict

        # gene
        try:
            gene_id_to_name = {}
            for gene in self.features_of_type("gene"):
                gene_id_to_name[gene.id] = gene["gene_name"][0]
                if len(gene["gene_name"]) > 1:
                    print(gene["gene_name"])
        except KeyError:
            # disable gene id name convertion
            gene_id_to_name = {g.id: g.id for g in self.features_of_type("gene")}

        gene_name_to_id = {v: k for k, v in gene_id_to_name.items()}
        gene_ids = list(gene_id_to_name.keys())
        gene_names = list(set(gene_id_to_name.values()))

        # transcript
        transcript_ids = [t.id for t in self.features_of_type("transcript")]

        basic_info_dict = {
            "_gene_id_to_name": gene_id_to_name,
            "_gene_id_base_to_name": {
                k.split(".")[0]: v for k, v in gene_id_to_name.items()
            },
            "_gene_name_to_id": gene_name_to_id,
            "_gene_name_to_id_base": {
                v: k.split(".")[0] for k, v in gene_name_to_id.items()
            },
            "gene_ids": gene_ids,
            "gene_names": gene_names,
            "transcript_ids": transcript_ids,
        }
        joblib.dump(basic_info_dict, basic_info_path)
        return basic_info_dict

    def gene_name_to_id(self, gene_name):
        """Convert gene name to gene ID."""
        return universal_mapping(gene_name, self._gene_name_to_id)

    def gene_id_to_name(self, gene_id):
        """Convert gene ID to gene name."""
        return universal_mapping(gene_id, self._gene_id_to_name)

    def gene_id_base_to_name(self, gene_id):
        """Convert gene ID base to gene name."""
        return universal_mapping(gene_id, self._gene_id_base_to_name)

    def gene_name_to_id_base(self, gene_name):
        """Convert gene name to gene ID base."""
        return universal_mapping(gene_name, self._gene_name_to_id_base)

    @property
    def gene_bed(self):
        """Gene features in bed format."""
        if self._gene_bed is None:
            gene_bed = []
            for gene in self.features_of_type("gene"):
                gene_bed.append(
                    {
                        "Chromosome": gene.chrom,
                        "Start": gene.start - 1,
                        "End": gene.end,
                        "Name": gene.id,
                        "Score": ".",
                        "Strand": gene.strand,
                    }
                )
            self._gene_bed = pd.DataFrame(gene_bed)

        gene_bed = self._gene_bed
        return gene_bed

    def find_region_features(self, region, feature_types=("gene",), return_bed=True):
        """Find features in a region."""
        if feature_types is None:
            feature_types = self.feature_types
        features = list(self.features_of_type(featuretype=feature_types, limit=region))

        if return_bed:
            return features_to_bed(features)
        else:
            return features

    def find_gene_features(
        self, gene, feature_types=("transcript", "exon", "intron"), return_bed=True
    ):
        """Find features of a gene."""
        try:
            gene_id = self.gene_name_to_id(gene)
        except KeyError:
            gene_id = gene
        gene_features = list(self.children(gene_id, featuretype=feature_types))

        if return_bed:
            return features_to_bed(gene_features)
        else:
            return gene_features

    def _feature_returner(self, **kwargs):
        """This method overwrites the original one to provide more specific feature classes."""
        kwargs.setdefault("dialect", self.dialect)
        kwargs.setdefault("keep_order", self.keep_order)
        kwargs.setdefault("sort_attribute_values", self.sort_attribute_values)

        featuretype = kwargs.get("featuretype", ".")
        if featuretype == "gene":
            return Gene(self, **kwargs)
        elif featuretype == "transcript":
            return Transcript(self, **kwargs)
        else:
            return Feature(**kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except FeatureNotFoundError:
            # try gene name
            gene_id = self.gene_name_to_id(key)
            return super().__getitem__(gene_id)


class FeatureSharedPropertyMixin:
    @property
    def gene_name(self) -> str:
        """Gene name of the feature."""
        return self["gene_name"][0]

    @property
    def gene_type(self) -> str:
        """Gene type of the feature."""
        return self["gene_type"][0]

    @property
    def exons(self) -> list[Feature]:
        """Exons of the feature."""
        return list(self.gtf_db.children(self.id, featuretype="exon"))

    @property
    def exon_bed(self) -> pd.DataFrame:
        """Exons of the feature in bed format."""
        return features_to_bed(self.exons)

    @property
    def CDS(self) -> list[Feature]:
        """CDS of the feature."""
        return list(self.gtf_db.children(self.id, featuretype="CDS"))

    @property
    def cds_bed(self) -> pd.DataFrame:
        """CDS of the feature in bed format."""
        return features_to_bed(self.cds)

    @property
    def UTR(self) -> list[Feature]:
        """UTR of the feature."""
        return list(self.gtf_db.children(self.id, featuretype="UTR"))

    @property
    def utr_bed(self) -> pd.DataFrame:
        """UTR of the feature in bed format."""
        return features_to_bed(self.utr)

    @property
    def start_codon(self) -> list[Feature]:
        """Start codon of the feature."""
        return list(self.gtf_db.children(self.id, featuretype="start_codon"))

    @property
    def start_codon_bed(self) -> pd.DataFrame:
        """Start codon of the feature in bed format."""
        return features_to_bed(self.start_codon)

    @property
    def stop_codon(self) -> list[Feature]:
        """Stop codon of the feature."""
        return list(self.gtf_db.children(self.id, featuretype="stop_codon"))

    @property
    def stop_codon_bed(self) -> pd.DataFrame:
        """Stop codon of the feature in bed format."""
        return features_to_bed(self.stop_codon)

    @property
    def region_tuple(self) -> tuple[str, int, int]:
        """Region of the feature in tuple format."""
        return self.chrom, self.start - 1, self.end

    @property
    def region(self) -> str:
        """Region of the feature in string format."""
        return f"{self.chrom}:{self.start - 1}-{self.end}"

    @property
    def length(self) -> int:
        """Length of the feature."""
        return self.end - self.start + 1

    @property
    def chrom_corrds(self) -> tuple[int, int]:
        """Chromosome coordinates of the feature."""
        return self.start - 1, self.end

    def __lt__(self, other):
        return self.start < other.start

    def __gt__(self, other):
        return self.start > other.start


class Gene(Feature, FeatureSharedPropertyMixin):
    def __init__(self, gtf_db, *args, **kwargs):
        self.gtf_db = gtf_db
        super().__init__(*args, **kwargs)

    @property
    def transcripts(self) -> list[Feature]:
        """Transcripts of the gene."""
        return list(self.gtf_db.children(self.id, featuretype="transcript"))

    @property
    def transcript_ids(self) -> list[str]:
        """Transcript IDs of the gene."""
        return [t.id for t in self.transcripts]

    @property
    def transcript_bed(self) -> pd.DataFrame:
        """Transcripts of the gene in bed format."""
        return features_to_bed(self.transcripts)

    @property
    def introns(self) -> list[Feature]:
        """Introns of the gene."""
        return list(self.gtf_db.children(self.id, featuretype="intron"))

    @property
    def intron_bed(self) -> pd.DataFrame:
        """Introns of the gene in bed format."""
        return features_to_bed(self.introns)

    def plot_on_ax(
        self,
        ax,
        transcripts=None,
        height=0.5,
        label_tss=True,
        offset=0,
        fontsize=8,
        **kwargs,
    ):
        """
        Plot transcripts on ax.
        """
        kwargs.setdefault("color", "black")
        kwargs.setdefault("linewidth", 0.5)

        if transcripts is None:
            transcripts = sorted(self.transcripts)

        for i, transcript in enumerate(transcripts):
            transcript.plot_on_ax(
                ax,
                y=i,
                height=height,
                label_tss=label_tss,
                offset=offset,
                fontsize=fontsize,
                **kwargs,
            )

        extend = self.length * 0.05
        ax.set(
            xlim=(self.start - offset - extend, self.end - offset + extend),
            ylim=(-1, len(self.transcripts)),
        )
        return ax


class Transcript(Feature, FeatureSharedPropertyMixin):
    def __init__(self, gtf_db, *args, **kwargs):
        self.gtf_db = gtf_db
        super().__init__(*args, **kwargs)

    @property
    def transcript_name(self) -> str:
        """Transcript name of the transcript."""
        return self["transcript_name"][0]

    @property
    def tss_position(self) -> int:
        """TSS position of the transcript."""
        if self.strand == "+":
            return self.start
        else:
            return self.end

    def __lt__(self, other: "Transcript"):
        return self.tss_position < other.tss_position

    def __gt__(self, other: "Transcript"):
        return self.tss_position > other.tss_position

    def plot_on_ax(
        self, ax, y=0, height=0.5, label_tss=True, offset=0, fontsize=8, **kwargs
    ):
        """
        Plot a arrow line on ax to represent the transcript.
        Plot rectangles to represent exons.
        Plot small triangle marks to represent strand.
        """
        if self.strand == "+":
            start = self.start - 1 - offset
            end = self.end - offset
        else:
            start = self.end - offset
            end = self.start - 1 - offset

        # plot transcript rectangle
        t_height = height / 10
        ax.add_patch(
            plt.Rectangle(
                xy=(start, y - t_height / 2),
                width=end - start,
                height=t_height,
                **kwargs,
            )
        )
        # plot exons
        for exon in self.exons:
            ax.add_patch(
                plt.Rectangle(
                    xy=(exon.start - 1 - offset, y - height / 2),
                    width=exon.end - exon.start + 1,
                    height=height,
                    **kwargs,
                )
            )

        # label tss
        if label_tss:
            ax.text(
                self.tss_position - offset,
                y,
                self.transcript_name,
                ha="right" if self.strand == "+" else "left",
                va="center",
                fontsize=fontsize,
            )

        return ax


# Currently gtf path is hard coded
# look for a remote storage solution
MM10_GTF_PATH = "/large_storage/zhoulab/hanliu/wmb/ref/mm10/gtf/biccn/modified_gencode.vM23.primary_assembly.annotation.gtf"
MM10_GTFDB_PATH = "/large_storage/zhoulab/hanliu/wmb/ref/mm10/gtf/biccn/modified_gencode.vM23.primary_assembly.annotation.gffutils.db"
HG38_GTF_PATH = (
    "/large_storage/zhoulab/hanliu/wmb/ref/hg38/gtf/gencode.v30.annotation.gtf"
)
HG38_GTFDB_PATH = (
    "/large_storage/zhoulab/hanliu/wmb/ref/hg38/gtf/gencode.v30.annotation.gffutils.db"
)


def load_gtf(genome: str = "mm10", gtf_path: str = None):
    """Load GTFDB for genome."""
    if gtf_path is None:
        if genome == "mm10":
            return GTFDB(MM10_GTFDB_PATH)
        elif genome == "hg38":
            return GTFDB(HG38_GTFDB_PATH)
        else:
            raise ValueError(
                f"Please provide gtf_path for custom genome. Unsupported genome: {genome}"
            )
    else:
        if str(gtf_path).endswith(".db"):
            gtfdb_path = gtf_path
        else:
            # try to create GTFDB
            gtfdb_path = pathlib.Path(gtf_path).with_suffix(".gffutils.db")
            if not gtfdb_path.exists():
                print(f"Create GTFDB for {gtf_path}")
                create_db(str(gtf_path), dbfn=str(gtfdb_path))
        return GTFDB(gtfdb_path)
