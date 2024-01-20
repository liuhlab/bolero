import pathlib
import shutil
import subprocess

import dnaio
import numpy as np
import pandas as pd
import pyranges as pr
import xarray as xr

import bolero

from .seq import Sequence

UCSC_GENOME = "https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.fa.gz"
UCSC_CHROM_SIZES = "https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.chrom.sizes"


def _read_chrom_sizes(chrom_sizes_path, main=True):
    chrom_sizes = pd.read_csv(
        chrom_sizes_path,
        sep="\t",
        names=["chrom", "size"],
        dtype={"chrom": str, "size": np.int64},
    )
    chrom_sizes = chrom_sizes.set_index("chrom").squeeze().sort_index()

    if main:
        # only keep main chromosomes
        chrom_sizes = chrom_sizes[~chrom_sizes.index.str.contains("_|random|chrUn|chrEBV|chrM|chrU|hap")]

    return chrom_sizes


def _iter_fasta(fasta_path):
    with dnaio.open(fasta_path) as f:
        for record in f:
            yield Sequence(
                record.sequence,
                name=record.name.split("::")[0],
            )


def _get_package_dir():
    package_dir = pathlib.Path(bolero.__file__).parent
    return package_dir


def _download_file(url, local_path):
    """Download a file from a url to a local path using wget or curl"""
    local_path = pathlib.Path(local_path)

    if local_path.exists():
        return

    temp_path = local_path.parent / (local_path.name + ".temp")
    # download with wget
    if shutil.which("wget"):
        subprocess.check_call(["wget", "-O", temp_path, url])
    # download with curl
    elif shutil.which("curl"):
        subprocess.check_call(["curl", "-o", temp_path, url])
    else:
        raise RuntimeError("Neither wget nor curl found on system")
    # rename temp file to final file
    temp_path.rename(local_path)
    return


class Genome:
    """Class for utilities related to a genome."""

    def __init__(self, genome):
        self.genome = genome
        self.fasta_path, self.chrom_sizes_path = self.download_genome_fasta()
        self.chrom_sizes = _read_chrom_sizes(self.chrom_sizes_path, main=True)
        self.all_chrom_sizes = _read_chrom_sizes(self.chrom_sizes_path, main=False)

        # load blacklist if it exists
        package_dir = _get_package_dir()
        blacklist_path = package_dir / f"pkg_data/blacklist_v2/{genome}-blacklist.v2.bed.gz"
        if blacklist_path.exists():
            _df = pr.read_bed(str(blacklist_path), as_df=True)
            self.blacklist_bed = pr.PyRanges(_df.iloc[:, :3]).sort()
        else:
            self.blacklist_bed = None

    def download_genome_fasta(self):
        """Download a genome fasta file from UCSC"""
        genome = self.genome

        # create a data directory within the package if it doesn't exist
        package_dir = _get_package_dir()
        data_dir = package_dir / "data"
        fasta_dir = data_dir / genome / "fasta"
        fasta_dir.mkdir(exist_ok=True, parents=True)

        fasta_url = UCSC_GENOME.format(genome=genome)
        fasta_file = fasta_dir / f"{genome}.fa"
        chrom_sizes_url = UCSC_CHROM_SIZES.format(genome=genome)
        chrom_sizes_file = fasta_dir / f"{genome}.chrom.sizes"

        # download fasta file
        if not fasta_file.exists():
            fasta_gz_file = fasta_file.parent / (fasta_file.name + ".gz")
            print(
                f"Downloading {genome} fasta file from UCSC" f"\nUCSC url: {fasta_url}" f"\nLocal path: {fasta_file}\n"
            )
            _download_file(fasta_url, fasta_gz_file)
            _download_file(chrom_sizes_url, chrom_sizes_file)

            # unzip fasta file
            print(f"Unzipping {fasta_gz_file}")
            subprocess.check_call(["gunzip", fasta_gz_file])

        return fasta_file, chrom_sizes_file

    def get_region_fasta(self, bed_path, output_path=None, name=False, compress=True):
        """
        Extract fasta sequences from a bed file.

        Parameters
        ----------
        bed_path : str or pathlib.Path
            Path to a bed file, bed file must be sorted and have chrom, start, end and name columns.
        output_path : str or pathlib.Path, optional
            Path to output fasta file. If None, will be the same as bed_path with a .fa extension
        name : bool, optional
            If True, will use the fourth column of the bed file as the name of the region in the fasta file
        compress : bool, optional
            If True, will compress the fasta file with gzip

        Returns
        -------
        output_path : pathlib.Path
            Path to output fasta file
        """
        bed_path = pathlib.Path(bed_path)
        if output_path is None:
            output_path = bed_path.parent / (bed_path.stem + ".fa")
        else:
            # remove .gz extension if present
            output_path = str(output_path)
            if output_path.endswith(".gz"):
                output_path = output_path[:-3]
            output_path = pathlib.Path(output_path)

        subprocess.check_call(
            [
                "bedtools",
                "getfasta",
                "-name",
                "-fi",
                self.fasta_path,
                "-bed",
                bed_path,
                "-fo",
                output_path,
            ]
        )

        if compress:
            subprocess.check_call(["gzip", "-f", output_path])

        return output_path

    def prepare_bed(
        self,
        bed_path,
        output_path=None,
        main_chroms=True,
        remove_blacklist=True,
        window=True,
        window_size=1000,
        window_step=50,
        downsample=None,
    ):
        """
        Prepare a bed file for generating one-hot matrix.

        Parameters
        ----------
        bed_path : str or pathlib.Path
            Path to a bed file.
        output_path : str or pathlib.Path, optional
            Path to output bed file. If None, will be the same as bed_path with a .prepared.bed extension
        main_chroms : bool, optional
            If True, will only keep main chromosomes
        remove_blacklist : bool, optional
            If True, will remove blacklist regions
        window : bool, optional
            If True, will use genome windows with window_size and window_step to cover the entire bed file
        window_size : int, optional
            Window size
        window_step : int, optional
            Window step
        downsample : int, optional
            Number of regions to downsample to

        Returns
        -------
        output_path : pathlib.Path
            Path to output bed file
        """
        bed_path = pathlib.Path(bed_path)
        bed = pr.read_bed(str(bed_path)).sort()

        # filter chromosomes
        if main_chroms:
            bed = bed[bed.Chromosome.isin(self.chrom_sizes.index)].copy()
        else:
            bed = bed[bed.Chromosome.isin(self.all_chrom_sizes.index)].copy()

        # remove blacklist regions
        if remove_blacklist and self.blacklist_bed is not None:
            bed = bed.subtract(self.blacklist_bed)

        # use genome windows with window_size and window_step to cover the entire bed file
        if window:
            bed = bed.merge().window(window_step)
            bed.End = bed.Start + window_step
            left_shift = window_size // window_step // 2 * window_step
            right_shift = window_size - left_shift
            bed.Start -= left_shift
            bed.End += right_shift

        # check if bed file has name column
        no_name = False
        if window:
            no_name = True
        elif "Name" not in bed.df.columns:
            no_name = True
        else:
            if (bed.df["Name"].unique() == np.array(["."])).sum() == 1:
                no_name = True
        if no_name:
            bed.Name = (
                bed.df["Chromosome"].astype(str) + ":" + bed.df["Start"].astype(str) + "-" + bed.df["End"].astype(str)
            )

        # downsample
        if downsample is not None:
            bed = bed.sample(n=downsample, replace=False)

        # save bed to new file
        if output_path is None:
            output_path = bed_path.stem + ".prepared.bed"
        bed.to_bed(str(output_path))
        return output_path

    def get_region_sequences(self, bed_path, save_fasta=False):
        """
        Extract fasta sequences from a bed file.

        Parameters
        ----------
        bed_path : str or pathlib.Path
            Path to a bed file
        save_fasta : bool, optional
            If True, will save the fasta file to the same directory as the bed file

        Returns
        -------
        sequences : list of bolero.pp.seq.Sequence
            List of Sequence objects
        """
        fasta_path = self.get_region_fasta(bed_path, output_path=None, name=False, compress=save_fasta)
        sequences = list(_iter_fasta(fasta_path))
        if not save_fasta:
            fasta_path.unlink()

        return sequences

    def get_region_one_hot(
        self,
        bed_path=None,
        order="ATCG",
        dtype=np.int8,
        add_reverse_complement=True,
    ):
        """
        Extract one-hot encoded sequences from a bed file.

        Regions in the bed file must be sorted and have chrom, start, end and name columns.
        Regions also needs to have the same length.

        Parameters
        ----------
        bed_path : str or pathlib.Path, optional
            Path to a bed file, bed file must be sorted and have chrom, start, end and name columns.
            If None, will extract sequences from fasta_path
        region_id : str, optional
            Column name of the region ID in the bed file. If None, will use chrom:start-end as the ID
        order : str, optional
            Order of the one-hot encoding base axis. Default is 'ATCG'.
        dtype : numpy.dtype, optional
            Data type of the output array. Default is np.int8.
        add_reverse_complement : bool, optional
            If True, will add the reverse complement of each sequence to the output

        Returns
        -------
        one_hot : xarray.DataArray
            One-hot encoded sequences
        """
        bed_path = pathlib.Path(bed_path)
        bed = pr.read_bed(str(bed_path))

        sequences = self.get_region_sequences(bed_path, save_fasta=False)

        # make sure all sequences are the same length
        seq_len = len(sequences[0])
        for seq in sequences:
            assert len(seq) == seq_len, "All sequences must be the same length"

        one_hot = np.zeros((len(sequences), seq_len, len(order)), dtype=dtype)
        for i, seq in enumerate(sequences):
            one_hot[i] = seq.one_hot_encoding(order=order, dtype=dtype)

        if add_reverse_complement:
            one_hot_rc = np.zeros((len(sequences), seq_len, len(order)), dtype=dtype)
            for i, seq in enumerate(sequences):
                one_hot_rc[i] = seq.reverse_complement().one_hot_encoding(order=order, dtype=dtype)
            one_hot = np.concatenate([one_hot, one_hot_rc], axis=0)

        # construct xarray.DataArray
        region_index = [seq.name for seq in sequences]
        region_chrom = bed.Chromosome
        region_start = bed.Start
        region_end = bed.End
        is_rc = [False] * len(sequences)
        if add_reverse_complement:
            region_index = region_index + [seq.name + "_rc" for seq in sequences]
            region_chrom = pd.concat([region_chrom, region_chrom])
            region_start = pd.concat([region_start, region_start])
            region_end = pd.concat([region_end, region_end])
            is_rc = is_rc + [True] * len(sequences)

        one_hot = xr.DataArray(
            one_hot,
            dims=("region", "position", "base"),
            coords={
                "region": region_index,
                "position": np.arange(seq_len),
                "base": list(order),
            },
        )
        one_hot = one_hot.assign_coords(
            {
                "chrom": ("region", region_chrom),
                "start": ("region", region_start),
                "end": ("region", region_end),
                "is_rc": ("region", is_rc),
            }
        )
        return one_hot

    def delete_genome_data(self):
        """Delete genome data files"""
        package_dir = _get_package_dir()
        data_dir = package_dir / "data"
        genome_dir = data_dir / self.genome
        shutil.rmtree(genome_dir)
        return
