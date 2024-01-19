import pathlib
import shutil
import subprocess

import bolero

UCSC_GENOME = "https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.fa.gz"
UCSC_CHROM_SIZES = "https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.chrom.sizes"


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

    def get_region_fasta(self, bed_path, output_path=None, region_id=None):
        """
        Extract a fasta file for regions in a bed file

        Parameters
        ----------
        bed_path : str or pathlib.Path
            Path to a bed file
        output_path : str or pathlib.Path, optional
            Path to output fasta file. If None, will be the same as bed_path with a .fa extension
        region_id : str, optional
            Column position or name of the region id in the bed file. If None, region id will be
            generated automatically as chr:start-end

        Returns
        -------
        output_path : pathlib.Path
            Path to output fasta file
        """
        bed_path = pathlib.Path(bed_path)
        if output_path is None:
            output_path = bed_path.parent / (bed_path.stem + ".fa")
        else:
            output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        temp_output_path = output_path.parent / (output_path.name + ".temp")

        if region_id is None:
            name_param = ""
        else:
            name_param = f"-name {region_id}"

        if not output_path.exists():
            print(f"Extracting fasta file for {bed_path}")
            # bedtools getfasta command
            cmd = f"bedtools getfasta -fi {self.fasta_path} -bed {bed_path} " f"-fo {temp_output_path} {name_param}"
            subprocess.check_call(cmd, shell=True)
            temp_output_path.rename(output_path)

        return output_path
