import pathlib
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd
import pyBigWig
from Bio import motifs

# get pkg_data path from package root
import bolero
from bolero.pp.seq import DEFAULT_ONE_HOT_ORDER
from bolero.utils import download_file, get_default_save_dir, get_file_size_gbs

PKG_DATA_PATH = pathlib.Path(bolero.__file__).parent / "pkg_data"
JASPAR_MTOFI_DBS = {
    "_".join(p.name.split("_")[:3]): p
    for p in pathlib.Path(PKG_DATA_PATH).glob("jaspar/*.motif_pwm.dict")
}

# The above motif_pwm.dict files are generated from the JASPAR 2024 CORE motif database using dump_jaspar_motif_pwm_dict
# JASPAR 2024 CORE motif database
_JASPAR_URL_BASE = "https://jaspar.elixir.no/download/data/2024/CORE"
JASPAR_URLS = {
    "JASPAR2024_CORE_nematodes": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_nematodes_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_diatoms": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_diatoms_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_insects": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_insects_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_vertebrates": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_fungi": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_fungi_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_plants": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_plants_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_urochordates": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_urochordates_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_ALL": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_non-redundant_pfms_jaspar.zip",
}


def dump_jaspar_motif_pwm_dict(db, output_dir="."):
    """
    Download JASPAR motif database and dump the PWMs into a dictionary.

    Parameters
    ----------
    jaspar_url : str
        URL to the JASPAR database.
    output_dir : str
        Directory to save the motif PWMs.
    """
    jaspar_url = JASPAR_URLS[db]
    db_name = jaspar_url.split("/")[-1].split(".")[0]
    with tempfile.TemporaryDirectory(prefix="bolero_") as tmp_dir:
        db_name = jaspar_url.split("/")[-1].split(".")[0]
        subprocess.run(
            f"wget {jaspar_url} -P {tmp_dir}",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"unzip {tmp_dir}/{db_name}.zip -d {tmp_dir}",
            shell=True,
            check=True,
        )
        jaspar_paths = list(pathlib.Path(tmp_dir).glob("*.jaspar"))

        motif_pwms = {}
        for p in jaspar_paths:
            with open(p) as handle:
                motif_list = motifs.parse(handle, "jaspar")
                for motif in motif_list:
                    pwm = pd.DataFrame(motif.pwm)
                    motif_pwms[(motif.matrix_id, motif.name)] = pwm

    output_dir = pathlib.Path(output_dir).absolute()
    joblib.dump(motif_pwms, f"{db_name}.motif_pwm.dict", compress=1)
    return


def _calc_row_entropy(row):
    row = row[row > 0]
    e = -np.sum(row * np.log2(row))
    return e


class JASPARMotif:
    """Initialize a JASPARMotif object."""

    def __init__(self, motif_id, motif_name, pwm, base_order=DEFAULT_ONE_HOT_ORDER):
        """
        Initialize a JASPARMotif object.

        Parameters
        ----------
        - motif_id (str): The ID of the motif.
        - motif_name (str): The name of the motif.
        - pwm (pandas.DataFrame): The position weight matrix (PWM) of the motif.
        - base_order (list): The order of bases in the PWM.

        Returns
        -------
        - JASPARMotif: The initialized JASPARMotif object.
        """
        self.motif_id = motif_id
        self.motif_name = motif_name
        self.pwm = pwm.loc[:, list(base_order)].copy()

    def __len__(self):
        """
        Get the length of the motif.

        Returns
        -------
        - int: The length of the motif.
        """
        return self.pwm.shape[0]

    def pwm_entropy(self):
        """
        Calculate the entropy of each position in the PWM.

        Returns
        -------
        - pandas.Series: The entropy values for each position in the PWM.
        """
        entropy = self.pwm.apply(_calc_row_entropy, axis=1)
        return entropy

    def clip_pwm_by_entropy(self, max_length=24):
        """
        Clip the PWM by removing the end positions with the highest entropy.

        Parameters
        ----------
        - max_length (int): The maximum length of the clipped PWM.

        Returns
        -------
        - None
        """
        cur_length = len(self)
        if cur_length <= max_length:
            return

        # calculate the entropy at each position
        pwm = self.pwm.copy()
        entropy = self.pwm_entropy()
        while cur_length > max_length:
            start_e = entropy.values[0]
            end_e = entropy.values[-1]
            if start_e > end_e:
                entropy = entropy.iloc[1:]
                pwm = pwm.iloc[1:]
            else:
                entropy = entropy.iloc[:-1]
                pwm = pwm.iloc[:-1]
            cur_length -= 1
        self.pwm = pwm.copy()
        return


class JASPARMotifDatabase:
    """
    Represents a database of JASPAR motifs.

    Parameters
    ----------
    - db (str): The JASPAR database to use. Defaults to "JASPAR2024_CORE_vertebrates".
    - max_length (int): The maximum length of motifs. Defaults to 24.
    - base_order (str): The order of bases in the motifs. Defaults to DEFAULT_ONE_HOT_ORDER.

    Attributes
    ----------
    - db (str): The JASPAR database being used.
    - motifs (list): A list of JASPARMotif objects representing the motifs in the database.

    Methods
    -------
    - available_databases(): Returns a set of available JASPAR databases.

    """

    @classmethod
    def available_databases(cls):
        """
        Returns a set of available JASPAR databases.

        Returns
        -------
        - set: A set of available JASPAR databases.

        """
        return set(JASPAR_MTOFI_DBS.keys())

    def __init__(
        self,
        db="JASPAR2024_CORE_vertebrates",
        max_length=24,
        base_order=DEFAULT_ONE_HOT_ORDER,
    ):
        """
        Initializes a JASPARMotifDatabase object.

        Parameters
        ----------
        - db (str): The JASPAR database to use. Defaults to "JASPAR2024_CORE_vertebrates".
        - max_length (int): The maximum length of motifs. Defaults to 24.
        - base_order (str): The order of bases in the motifs. Defaults to DEFAULT_ONE_HOT_ORDER.

        Raises
        ------
        - ValueError: If the specified JASPAR database is invalid.

        """
        # check if db is valid using class method
        if db not in self.available_databases():
            raise ValueError(f"Invalid JASPAR database: {db}")

        self.db = db
        motif_pwms = joblib.load(JASPAR_MTOFI_DBS[db])
        self.base_order = base_order

        self.motifs = []
        for (motif_id, motif_name), pwm in motif_pwms.items():
            motif = JASPARMotif(
                motif_id, motif_name, pwm, base_order=DEFAULT_ONE_HOT_ORDER
            )

            motif.clip_pwm_by_entropy(max_length)
            self.motifs.append(motif)
        return


JASPAR_TFBS_GENOME_BIGBED_URL = (
    "https://frigg.uio.no/JASPAR/JASPAR_TFBSs/2024/JASPAR2024_{genome}.bb"
)


class JASPARMotifBigBed:
    """
    Class for working with JASPAR motif bigbed files.

    Parameters
    ----------
    genome : str
        Genome identifier.
    bb_file : str or None, optional
        Path to the bigbed file. If not provided, the file will be downloaded automatically.
    save_dir : str or None, optional
        Directory to save the bigbed file. If not provided, the default save directory will be used.

    Attributes
    ----------
    genome : str
        Genome identifier.
    bb_file : str
        Path to the bigbed file.
    bb_handle : pyBigWig.BigWigFile
        Handle to the bigbed file.

    Methods
    -------
    __enter__()
        Enter method for context management.
    __exit__(*args)
        Exit method for context management.
    close()
        Close the bigbed file.
    get_motifs(*args, use_genes=None)
        Get motifs for a genomic region from the bigbed file.
    get_motif_track_plotter(*args, plot_order=None, plot_genes=None)
        Get a MotifTrackPlotter instance for the motifs in the bigbed file.
    download_motif_bigbed(genome, save_dir=None)
        Download a genome fasta file from UCSC.
    """

    def __init__(self, genome, bb_file=None, save_dir=None):
        self.genome = genome

        if bb_file is None:
            self.save_dir = get_default_save_dir(save_dir)
            jaspar_tfbs_dir = self.save_dir / "jaspar/TFBSs"
            jaspar_tfbs_dir.mkdir(exist_ok=True, parents=True)
            self.bb_file = jaspar_tfbs_dir / f"JASPAR2024_{self.genome}.bb"
            if not self.bb_file.exists():
                raise FileNotFoundError(
                    f"BigBed file not found: {self.bb_file}, "
                    f"please download it with JAASPARMotifBigBed.download_motif_bigbed method "
                    f"or provide the path to the file using the bb_file argument."
                )

        self.bb_handle = pyBigWig.open(str(self.bb_file))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.bb_handle.__exit__()

    def close(self):
        """
        Close the bigbed file.
        """
        self.bb_handle.close()
        return

    def get_motifs(self, *args, use_genes=None):
        """
        Get motifs for a genomic region from the bigbed file.

        Parameters
        ----------
        args : str or tuple
            Genomic region in the format "chr:start-end" or "chr", start, end as separate arguments.
        use_genes : list of str, optional
            List of genes to include. Default is None, which includes all genes.

        Returns
        -------
        motifs_df : pandas.DataFrame
            DataFrame with the motifs in the genomic region.
            Columns are "Chromosome", "Start", "End", "Motif", "Score", "Strand", "Gene".
        """
        if len(args) == 1:
            chrom, coords = args[0].split(":")
            start, end = coords.split("-")
            start = int(start)
            end = int(end)
        else:
            chrom, start, end = args

        motifs_df = []
        for _start, _end, info in self.bb_handle.entries(chrom, start, end):
            motif, score, strand, gene = info.split("\t")
            motifs_df.append([chrom, _start, _end, motif, score, strand, gene])
        motifs_df = pd.DataFrame(
            motifs_df,
            columns=["Chromosome", "Start", "End", "Motif", "Score", "Strand", "Gene"],
        )
        motifs_df["Score"] = motifs_df["Score"].astype(int)

        if use_genes is not None:
            motifs_df = motifs_df[motifs_df["Gene"].isin(use_genes)].copy()
        return motifs_df

    def get_motif_track_plotter(self, *args, plot_order=None, plot_genes=None):
        """
        Get a MotifTrackPlotter instance for the motifs in the bigbed file.

        Parameters
        ----------
        args : str or tuple or pd.DataFrame
            Genomic region in the format "chr:start-end" or "chr", start, end as separate arguments
            or a DataFrame from the JAASPARMotifBigBed.get_motifs method.
        plot_order : list of str, optional
            Order of the genes to plot. Default is None, which uses the gene's average motif score to determine the order.
        plot_genes : list of str, optional
            List of genes to include. Default is None, which includes all genes.

        Returns
        -------
        MotifTrackPlotter
            MotifTrackPlotter instance for the motifs in the genomic region.
            MotifTrackPlotter.plot() method can be used to plot the motifs on an Axes.
        """
        from bolero.pl.motif_track import MotifTrackPlotter

        if isinstance(args[0], pd.DataFrame):
            motifs_df = args[0]
        else:
            motifs_df = self.get_motifs(*args, use_genes=plot_genes)
        return MotifTrackPlotter(
            motifs_df, name_col="Gene", plot_order=plot_order, plot_genes=plot_genes
        )

    @classmethod
    def download_motif_bigbed(cls, genome, save_dir=None):
        """
        Download a genome fasta file from UCSC.

        Parameters
        ----------
        genome : str
            Genome identifier.
        save_dir : str or None, optional
            Directory to save the bigbed file. If not provided, the default save directory will be used.

        Returns
        -------
        bb_file : str
            Path to the downloaded bigbed file.
        """
        # create a data directory within the package if it doesn't exist
        save_dir = get_default_save_dir(save_dir)
        bb_url = JASPAR_TFBS_GENOME_BIGBED_URL.format(genome=genome)
        jaspar_tfbs_dir = save_dir / "jaspar/TFBSs"
        jaspar_tfbs_dir.mkdir(exist_ok=True, parents=True)
        bb_file = jaspar_tfbs_dir / f"JASPAR2024_{genome}.bb"

        # download fasta file
        if not bb_file.exists():
            print(
                f"Downloading {genome} TFBSs BigBed file from JASPAR 2024"
                f"\nJASPAR url: {bb_url}"
                f"\nLocal path: {bb_file}"
                f"\nFile size: {get_file_size_gbs(bb_url):.2f} GB"
                "\nIf the download is too slow, you may manually download the file and "
                "provide the path to the file using the bb_file argument."
            )
            download_file(bb_url, bb_file)
        return bb_file
