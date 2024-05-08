import pathlib
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import pyranges as pr
import xarray as xr
import zarr
from numcodecs import Zstd
from pyfaidx import Fasta
from tqdm import tqdm

from bolero.pp.genome_dataset import GenomeOneHotZarr
from bolero.pp.seq import DEFAULT_ONE_HOT_ORDER, Sequence
from bolero.pp.utils import get_global_coords
from bolero.utils import (
    download_file,
    get_default_save_dir,
    get_package_dir,
    understand_regions,
)

zarr.storage.default_compressor = Zstd(level=3)


UCSC_GENOME = (
    "https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.fa.gz"
)
UCSC_CHROM_SIZES = (
    "https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.chrom.sizes"
)


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
        chrom_sizes = chrom_sizes[
            ~chrom_sizes.index.str.contains("_|random|chrUn|chrEBV|chrM|chrU|hap")
        ]

    return chrom_sizes


def _chrom_sizes_to_bed(chrom_sizes):
    genome_bed = chrom_sizes.reset_index()
    genome_bed.columns = ["Chromosome", "Size"]
    genome_bed["End"] = genome_bed["Size"]
    genome_bed["Start"] = 0
    genome_bed = pr.PyRanges(genome_bed[["Chromosome", "Start", "End"]])
    return genome_bed


def _chrom_size_to_chrom_offsets(chrom_sizes):
    cur_start = 0
    cur_end = 0
    records = []
    for chrom, size in chrom_sizes.items():
        cur_end += size
        records.append([chrom, cur_start, cur_end, size])
        cur_start += size
    chrom_offsets = pd.DataFrame(
        records, columns=["chrom", "global_start", "global_end", "size"]
    ).set_index("chrom")
    chrom_offsets.columns.name = "coords"
    return chrom_offsets


def _iter_fasta(fasta_path):
    with Fasta(fasta_path) as f:
        for record in f:
            yield Sequence(
                str(record[:]),
                name=record.name.split("::")[0],
            )


def _scan_bw(bw_path, bed_path, type="mean", dtype="float32"):
    regions = pr.read_bed(str(bed_path), as_df=True)
    with pyBigWig.open(str(bw_path)) as bw:
        values = []
        for _, (chrom, start, end, *_) in regions.iterrows():
            data = bw.stats(chrom, start, end, type=type)[0]
            values.append(data)
    values = pd.Series(values, dtype=dtype)
    return values


def _dump_fa(path, name, seq):
    with open(path, "w") as f:
        f.write(f">{name}\n")
        f.write(str(seq.seq).upper() + "\n")


def _process_cbust_bed(df):
    chrom, chunk_start, chunk_end, slop = df["# chrom"][0].split(":")
    chunk_start = int(chunk_start)
    chunk_end = int(chunk_end)
    slop = int(slop)
    seq_start = max(0, chunk_start - slop)

    # adjust to genome coords
    df["genomic_start__bed"] += seq_start
    df["genomic_end__bed"] += seq_start
    df["# chrom"] = chrom

    use_cols = [
        "# chrom",
        "genomic_start__bed",
        "genomic_end__bed",
        "cluster_id_or_motif_name",
        "cluster_or_motif_score",
        "strand",
        "cluster_or_motif",
        "motif_sequence",
        "motif_type_contribution_score",
    ]
    df = df[use_cols].copy()
    df = df.loc[
        (df["genomic_end__bed"] <= chunk_end) & (df["genomic_start__bed"] > chunk_start)
    ].copy()
    return df


def _run_cbust_chunk(
    output_dir, fasta_chunk_path, cbust_path, motif_path, min_cluster_score, b, r
):
    fasta_chunk_path = pathlib.Path(fasta_chunk_path)
    fa_name = fasta_chunk_path.name
    output_path = f"{output_dir}/{fa_name}.csv.gz"
    temp_path = f"{output_dir}/{fa_name}.temp.csv.gz"
    if pathlib.Path(output_path).exists():
        return

    cmd = f"{cbust_path} -f 5 -c {min_cluster_score} -b {b} -r {r} -t 1000000000 {motif_path} {fasta_chunk_path}"
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
        shell=True,
    )
    try:
        df = pd.read_csv(StringIO(p.stdout), sep="\t")
    except pd.errors.EmptyDataError:
        return

    df = _process_cbust_bed(df)

    df.to_csv(temp_path)
    pathlib.Path(temp_path).rename(output_path)
    return


def _combine_single_motif_scan_to_bigwig(
    output_dir, genome, chrom_sizes, save_motif_scan
):
    motif = pathlib.Path(output_dir).name
    all_chunk_paths = list(output_dir.glob("*.csv.gz"))
    total_results = []
    for path in tqdm(all_chunk_paths):
        df = pd.read_csv(path, index_col=0)
        total_results.append(df)
    total_results = pd.concat(total_results).rename(
        columns={
            "# chrom": "chrom",
            "genomic_start__bed": "start",
            "genomic_end__bed": "end",
        }
    )
    cluster_bed = total_results[total_results["cluster_or_motif"] == "cluster"]
    cluster_bed = cluster_bed.sort_values(["chrom", "start"])
    with pyBigWig.open(f"{genome}+{motif}.bw", "w") as bw:
        bw.addHeader(list(chrom_sizes.sort_index().items()))
        bw.addEntries(
            cluster_bed["chrom"].astype(str).tolist(),
            cluster_bed["start"].astype("int64").tolist(),
            ends=cluster_bed["end"].astype("int64").tolist(),
            values=cluster_bed["cluster_or_motif_score"].astype("float32").tolist(),
        )
    if save_motif_scan:
        total_results.to_csv(f"{genome}+{motif}.motif_scan.csv.gz")
    return


def _is_macos():
    import platform

    return platform.system() == "Darwin"


class Genome:
    """Class for utilities related to a genome."""

    def __init__(self, genome, save_dir=None):
        if isinstance(genome, self.__class__):
            return genome

        self.name = genome

        package_dir = get_package_dir()
        self.save_dir = get_default_save_dir(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.fasta_path, self.chrom_sizes_path = self.download_genome_fasta()
        self.chrom_sizes = _read_chrom_sizes(self.chrom_sizes_path, main=True)
        self.chrom_offsets = _chrom_size_to_chrom_offsets(self.chrom_sizes)
        self.chromosomes = self.chrom_sizes.index
        self.genome_bed = _chrom_sizes_to_bed(self.chrom_sizes)
        self.all_chrom_sizes = _read_chrom_sizes(self.chrom_sizes_path, main=False)
        self.all_genome_bed = _chrom_sizes_to_bed(self.all_chrom_sizes)
        self.all_chromosomes = self.all_chrom_sizes.index

        # load blacklist if it exists
        blacklist_path = (
            package_dir / f"pkg_data/blacklist_v2/{genome}-blacklist.v2.bed.gz"
        )
        if blacklist_path.exists():
            _df = pr.read_bed(str(blacklist_path), as_df=True)
            self.blacklist_bed = pr.PyRanges(_df.iloc[:, :3]).sort()
        else:
            self.blacklist_bed = None

        # one hot
        self._one_hot_obj = None
        self.genome_one_hot_path = (
            self.save_dir / "data" / self.name / f"{self.name}.onehot.zarr"
        )
        return

    def __repr__(self):
        name_str = f"Genome: {self.name}"
        fastq_path = f"Fasta Path: {self.fasta_path}"
        if self._one_hot_obj is None:
            one_hot_zarr = "Genome One Hot Zarr: Not created"
        else:
            one_hot_zarr = f"Genome One Hot Zarr:\n{self.genome_one_hot.__repr__()}"
        return f"{name_str}\n{fastq_path}\n{one_hot_zarr}"

    def download_genome_fasta(self):
        """Download a genome fasta file from UCSC"""
        _genome = self.name

        # create a data directory within the package if it doesn't exist
        save_dir = self.save_dir
        data_dir = save_dir / "data"
        fasta_dir = data_dir / _genome / "fasta"
        fasta_dir.mkdir(exist_ok=True, parents=True)

        fasta_url = UCSC_GENOME.format(genome=_genome)
        fasta_file = fasta_dir / f"{_genome}.fa"
        chrom_sizes_url = UCSC_CHROM_SIZES.format(genome=_genome)
        chrom_sizes_file = fasta_dir / f"{_genome}.chrom.sizes"

        # download fasta file
        if not fasta_file.exists():
            fasta_gz_file = fasta_file.parent / (fasta_file.name + ".gz")
            print(
                f"Downloading {_genome} fasta file from UCSC"
                f"\nUCSC url: {fasta_url}"
                f"\nLocal path: {fasta_file}\n"
            )
            download_file(fasta_url, fasta_gz_file)
            download_file(chrom_sizes_url, chrom_sizes_file)

            # unzip fasta file
            print(f"Unzipping {fasta_gz_file}")
            subprocess.check_call(["gunzip", fasta_gz_file])
        return fasta_file, chrom_sizes_file

    def get_region_fasta(self, bed_path, output_path=None, compress=True):
        """
        Extract fasta sequences from a bed file.

        Parameters
        ----------
        bed_path : str or pathlib.Path
            Path to a bed file, bed file must be sorted and have chrom, start, end and name columns.
        output_path : str or pathlib.Path, optional
            Path to output fasta file. If None, will be the same as bed_path with a .fa extension
        compress : bool, optional
            If True, will compress the fasta file with bgzip

        Returns
        -------
        output_path : pathlib.Path
            Path to output fasta file
        """
        bed_path = pathlib.Path(bed_path)

        # read head of bed file to check if it has a name column
        bed_df = pd.read_csv(bed_path, sep="\t", header=None, nrows=5)
        if bed_df.shape[1] == 3:
            name_param = []
        else:
            name_param = ["-name"]

        if output_path is None:
            output_path = bed_path.parent / (bed_path.stem + ".fa")
        else:
            # remove .gz extension if present
            output_path = str(output_path)
            if output_path.endswith(".gz"):
                output_path = output_path[:-3]
            output_path = pathlib.Path(output_path)

        subprocess.check_call(
            ["bedtools", "getfasta"]
            + name_param
            + [
                "-fi",
                self.fasta_path,
                "-bed",
                bed_path,
                "-fo",
                output_path,
            ]
        )

        if compress:
            subprocess.check_call(["bgzip", "-f", output_path])

        return output_path

    def _remove_blacklist(self, bed, slop_black=2000):
        """Remove blacklist regions from a bed file"""
        if self.blacklist_bed is not None:
            if slop_black > 0:
                _blacklist_bed = self.blacklist_bed.extend(slop_black)
            else:
                _blacklist_bed = self.blacklist_bed
            bed = bed.subtract(_blacklist_bed)
        return bed

    def prepare_window_bed(
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
        if remove_blacklist:
            bed = self._remove_blacklist(bed)

        # use genome windows with window_size and window_step to cover the entire bed file
        if window:
            bed = bed.merge().window(window_step)
            bed.End = bed.Start + window_step
            left_shift = window_size // window_step // 2 * window_step
            right_shift = window_size - left_shift
            s = bed.Start.copy()
            bed.End = s + right_shift
            bed.Start = s - left_shift

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
                bed.df["Chromosome"].astype(str)
                + ":"
                + bed.df["Start"].astype(str)
                + "-"
                + bed.df["End"].astype(str)
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
        fasta_path = self.get_region_fasta(
            bed_path, output_path=None, compress=save_fasta
        )
        sequences = list(_iter_fasta(fasta_path))
        if not save_fasta:
            fasta_path.unlink()
            fai_path = fasta_path.parent / (fasta_path.name + ".fai")
            fai_path.unlink()

        return sequences

    def delete_genome_data(self):
        """Delete genome data files"""
        data_dir = self.save_dir / "data"
        genome_dir = data_dir / self.name
        shutil.rmtree(genome_dir)
        return

    def _scan_bw_table(self, bw_table, bed_path, zarr_path, cpu=None):
        bw_paths = pd.read_csv(bw_table, index_col=0, header=None).squeeze()
        fs = {}
        with ProcessPoolExecutor(cpu) as p:
            for name, bw_path in bw_paths.items():
                bw_path = pathlib.Path(bw_path).absolute()
                name = pathlib.Path(bw_path).name.split(".")[0]
                f = p.submit(
                    _scan_bw,
                    bw_path=bw_path,
                    bed_path=bed_path,
                    type="mean",
                    dtype="float32",
                )
                fs[f] = name

            results = {}
            for f in as_completed(fs):
                name = fs[f]
                results[name] = f.result()

            results = pd.DataFrame(results[k] for k in bw_paths.index)

            regions = pr.read_bed(str(bed_path))
            results.columns = regions.Name
            results.columns.name = "region"
            results.index.name = "bigwig"

            da = xr.DataArray(results)
            da = da.assign_coords(
                {
                    "chrom": ("region", regions.Chromosome),
                    "start": ("region", regions.Start),
                    "end": ("region", regions.End),
                }
            )

        bw_len = bw_paths.size
        region_chunk_size = max(5000, 100000000 // bw_len // 10000 * 10000)
        da = da.chunk({"region": region_chunk_size, "bigwig": bw_len})

        for coord in list(da.coords.keys()):
            _coords = da.coords[coord]
            if coord == "region":
                da.coords[coord] = _coords.chunk({"region": 100000000})
            elif coord == "bigwig":
                da.coords[coord] = _coords.chunk({coord: len(_coords)})
            elif coord == "chrom":
                chrom_max_size = max([len(k) for k in self.chrom_sizes.index])
                da.coords[coord] = _coords.astype(f"<U{chrom_max_size}").chunk(
                    {"region": 100000000}
                )
            elif coord in {"start", "end"}:
                da.coords[coord] = _coords.chunk({"region": 100000000})

        da.to_zarr(zarr_path, mode="w")
        return

    def standard_region_length(
        self,
        regions,
        length,
        remove_blacklist=False,
        boarder_strategy="shift",
        as_df=False,
        keep_original=False,
    ):
        """
        Adjusts the length of regions to a standard length.

        Parameters
        ----------
        regions : PyRanges, DataFrame, str, Path, list, or Index
            The regions to be adjusted. It can be a PyRanges object, a DataFrame, a file path, a list, or an Index.
        length : int
            The desired length of the regions.
        remove_blacklist : bool, optional
            Whether to remove regions that overlap with the blacklist. Default is False.
        boarder_strategy : str, optional
            For regions that overlap with the chromosome boarder, the strategy to adjust the boarder.
            If 'shift', the region will be shifted to the left or right to fit into the chromosome.
            If 'drop', the region overlapping with the boarder will be dropped. The number of output regions may be less than the input regions.
            Default is 'shift'.
        as_df : bool, optional
            Whether to return the adjusted regions as a DataFrame. Default is False.

        Returns
        -------
        regions_bed : PyRanges
            The adjusted regions with the specified length.

        Raises
        ------
        ValueError
            If the regions parameter is not a PyRanges, DataFrame, str, Path, list, or Index.

        Notes
        -----
        - The method adjusts the length of the regions to the specified length.
        - It ensures that all regions have the same size by centering them around their midpoint.
        - It also ensures that the start and end positions of each region are within the range of the chromosome.
        - The method updates the 'Name' column of the regions to reflect the adjusted positions.

        """
        regions_bed = understand_regions(regions)

        if keep_original:
            regions_bed_df = regions_bed.df
            if "Name" in regions_bed_df:
                regions_bed_df["Original_Name"] = regions_bed_df["Name"]
            else:
                regions_bed_df["Name"] = (
                    regions_bed_df["Chromosome"].astype(str)
                    + ":"
                    + regions_bed_df["Start"].astype(str)
                    + "-"
                    + regions_bed_df["End"].astype(str)
                )
            regions_bed = pr.PyRanges(regions_bed_df)

        # make sure all regions have the same size
        regions_center = (regions_bed.Start + regions_bed.End) // 2
        regions_bed.Start = regions_center - length // 2
        regions_bed.End = regions_center + length // 2
        # make sure for each chrom, start and end are not out of range
        # only keep regions that are in range
        chrom_sizes = self.chrom_sizes
        use_regions = []
        for chrom, chrom_df in regions_bed.df.groupby("Chromosome"):
            chrom_size = chrom_sizes[chrom]
            if boarder_strategy == "shift":
                chrom_df.loc[chrom_df.Start < 0, ["Start", "End"]] -= chrom_df.loc[
                    chrom_df.Start < 0, "Start"
                ].values[:, None]
                chrom_df.loc[chrom_df.End > chrom_size, ["Start", "End"]] -= (
                    chrom_df.loc[chrom_df.End > chrom_size, "End"] - chrom_size
                ).values[:, None]
            elif boarder_strategy == "drop":
                chrom_df = chrom_df[
                    (chrom_df.Start >= 0) & (chrom_df.End <= chrom_size)
                ]
            else:
                raise ValueError("boarder_strategy must be 'shift' or 'drop'")
            use_regions.append(chrom_df)
        use_regions = pd.concat(use_regions)

        # update Name col
        use_regions["Name"] = (
            use_regions["Chromosome"].astype(str)
            + ":"
            + use_regions["Start"].astype(str)
            + "-"
            + use_regions["End"].astype(str)
        )
        use_cols = ["Chromosome", "Start", "End", "Name"]
        if keep_original:
            use_cols.append("Original_Name")
        regions_bed = pr.PyRanges(use_regions[use_cols])

        if remove_blacklist and self.blacklist_bed is not None:
            regions_bed = self._remove_blacklist(regions_bed)
            # region length may change after removing blacklist
            use_regions = regions_bed.df["End"] - regions_bed.df["Start"] == length
            regions_bed = regions_bed[use_regions].copy()

        if len(regions_bed) == 0:
            raise pd.errors.EmptyDataError("No regions left after processing")

        if as_df:
            return regions_bed.df
        return regions_bed

    @property
    def genome_one_hot(self):
        """
        Returns the one-hot encoded representation of the genome.

        If the one-hot encoded object is not already created, it generates it and saves it to a zarr file.
        The generated object is then stored in the `_one_hot_obj` attribute for future use.

        Returns
        -------
            GenomeOneHotZarr: The one-hot encoded representation of the genome.
        """
        if self._one_hot_obj is None:
            zarr_path = self.genome_one_hot_path
            success_flag_path = zarr_path / ".success"
            if not success_flag_path.exists():
                self.generate_genome_one_hot(zarr_path=zarr_path)
            genome_one_hot = GenomeOneHotZarr(zarr_path)
            self._one_hot_obj = genome_one_hot
        return self._one_hot_obj

    def generate_genome_one_hot(self, zarr_path=None):
        """
        Generate genome one-hot encoding.

        Parameters
        ----------
        - zarr_path (str): Path to save the Zarr file. If not provided, a default path will be used.

        Returns
        -------
        - None

        Raises
        ------
        - None
        """
        print("Generating genome one-hot encoding")
        if zarr_path is None:
            zarr_path = self.save_dir / "data" / self.name / f"{self.name}.onehot.zarr"
            zarr_path.mkdir(exist_ok=True, parents=True)

        success_flag_path = zarr_path / ".success"
        if success_flag_path.exists():
            return

        total_chrom_size = self.chrom_sizes.sum()
        one_hot_da = xr.DataArray(
            np.zeros([total_chrom_size, 4], dtype="bool"),
            dims=["pos", "base"],
            coords={"base": list(DEFAULT_ONE_HOT_ORDER)},
        )
        one_hot_ds = xr.Dataset({"X": one_hot_da, "offsets": self.chrom_offsets})
        one_hot_ds.to_zarr(
            zarr_path, encoding={"X": {"chunks": (50000000, 4)}}, mode="w"
        )
        zarr_da = zarr.open_array(f"{zarr_path}/X")
        with pyfaidx.Fasta(self.fasta_path) as fa:
            cur_start = 0
            for chrom in tqdm(self.chrom_sizes.index):
                seq = Sequence(str(fa[chrom]))
                seq_len = len(seq)
                one_hot = seq.one_hot_encoding(dtype=bool)

                zarr_da[cur_start : cur_start + seq_len, :] = one_hot
                cur_start += seq_len
        success_flag_path.touch()
        return

    def dump_region_bigwig_zarr(
        self,
        bw_table,
        bed_path,
        partition_dir,
        region_id=None,
        partition_size=50000000,
        cpu=None,
    ):
        """
        Dump bigwig values from a bed file into zarr files.
        """
        partition_dir = pathlib.Path(partition_dir)
        partition_dir.mkdir(exist_ok=True, parents=True)
        bed_df = pr.read_bed(str(bed_path), as_df=True)
        bed_df["Partition"] = (
            bed_df.Chromosome.astype(str)
            + "-"
            + (bed_df.Start // partition_size).astype(str)
        )
        if region_id is None:
            region_id = "Name"
            bed_df[region_id] = (
                bed_df.Chromosome.astype(str)
                + ":"
                + bed_df.Start.astype(str)
                + "-"
                + bed_df.End.astype(str)
            )
        bed_df = bed_df[["Chromosome", "Start", "End", region_id, "Partition"]]

        for chunk_name, chunk_bed in tqdm(bed_df.groupby("Partition")):
            chunk_bed_path = partition_dir / f"{chunk_name}.bed"
            chunk_zarr_path = partition_dir / f"{chunk_name}.zarr"
            chunk_bed.iloc[:, :4].to_csv(
                chunk_bed_path, sep="\t", index=None, header=None
            )

            self._scan_bw_table(
                bw_table=bw_table,
                bed_path=chunk_bed_path,
                zarr_path=chunk_zarr_path,
                cpu=cpu,
            )
            pathlib.Path(chunk_bed_path).unlink()
        return

    def split_genome_fasta(self, fasta_chunk_dir, chunk_size=10000000, slop_size=10000):
        """
        Split genome fasta into chunks.

        Parameters
        ----------
        fasta_chunk_dir : str or pathlib.Path
            Path to directory to save the fasta chunks
        chunk_size : int, optional
            Size of each chunk in base pairs
        slop_size : int, optional
            Size of slop for each chunk
        """
        fasta_chunk_dir = pathlib.Path(fasta_chunk_dir)
        fasta_chunk_dir.mkdir(exist_ok=True)
        success_flag_path = fasta_chunk_dir / ".success"

        if success_flag_path.exists():
            return

        with Fasta(self.fasta_path) as fasta:
            for chrom in fasta:
                if chrom.name not in self.chromosomes:
                    continue

                chrom_size = self.chrom_sizes[chrom.name]

                chunk_starts = list(range(0, chrom_size, chunk_size))
                slop = (
                    slop_size + 1000
                )  # slop this size for the -r parameter in cbust, estimating background motif occurance
                for chunk_start in chunk_starts:
                    seq_start = max(chunk_start - slop, 0)
                    chunk_end = min(chunk_start + chunk_size, chrom_size)
                    seq_end = min(chunk_start + chunk_size + slop, chrom_size)
                    _name = f"{chrom.name}:{chunk_start}:{chunk_end}:{slop}"
                    _path = f"{fasta_chunk_dir}/{_name}.fa"
                    _seq = chrom[seq_start:seq_end]
                    _dump_fa(path=_path, name=_name, seq=_seq)

        success_flag_path.touch()
        return

    def scan_motif_with_cbust(
        self,
        output_dir,
        motif_table,
        cpu=None,
        min_cluster_score=0,
        r=10000,
        b=0,
        save_motif_scan=False,
    ):
        """
        Scan motifs with cbust.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Path to directory to save the output bigwig files
        motif_table : str or pathlib.Path
            Path to a table of motif names and paths
        cpu : int, optional
            Number of cpus to use, if None, will use all available cpus
        min_cluster_score : int, optional
            Minimum cluster score
        r : int, optional
            cbust -r parameter. Range in bp for counting local nucleotide abundances.
        b : int, optional
            cbust -b parameter. Background padding in bp.
        save_motif_scan : bool, optional
            If True, will save the motif scan table file, which has exact motif locations and scores.
        """
        motif_paths = pd.read_csv(motif_table, index_col=0, header=None).squeeze()

        if _is_macos():
            cbust_path = self.save_dir / "pkg_data/cbust_macos"
        else:
            cbust_path = self.save_dir / "pkg_data/cbust"

        output_dir = pathlib.Path(output_dir)
        fasta_chunk_dir = output_dir / "fasta_chunks_for_motif_scan"
        fasta_chunk_dir.mkdir(exist_ok=True, parents=True)

        self.split_genome_fasta(fasta_chunk_dir=fasta_chunk_dir, slop_size=r)

        fasta_chunk_paths = list(pathlib.Path(fasta_chunk_dir).glob("*.fa"))

        with ProcessPoolExecutor(cpu) as pool:
            fs = []
            for motif, motif_path in motif_paths.items():
                motif_temp_dir = output_dir / (motif + "_temp")
                motif_temp_dir.mkdir(exist_ok=True, parents=True)

                for fasta_chunk_path in fasta_chunk_paths:
                    fs.append(
                        pool.submit(
                            _run_cbust_chunk,
                            output_dir=motif_temp_dir,
                            fasta_chunk_path=fasta_chunk_path,
                            cbust_path=cbust_path,
                            motif_path=motif_path,
                            min_cluster_score=min_cluster_score,
                            b=b,
                            r=r,
                        )
                    )

            for f in as_completed(fs):
                f.result()

        motif_temp_dirs = list(output_dir.glob("*_temp"))
        with ProcessPoolExecutor(cpu) as pool:
            fs = {}
            for motif_temp_dir in motif_temp_dirs:
                future = pool.submit(
                    _combine_single_motif_scan_to_bigwig,
                    output_dir=motif_temp_dir,
                    genome=self.name,
                    chrom_sizes=self.chrom_sizes,
                    save_motif_scan=save_motif_scan,
                )
                fs[future] = motif_temp_dir

            for f in as_completed(fs):
                f.result()
                motif_temp_dir = fs[f]
                shutil.rmtree(motif_temp_dir)
        return

    def get_region_one_hot(self, *args):
        """
        Returns the one-hot encoding of a genomic region.

        Parameters
        ----------
        *args: Variable length argument list specifying the genomic region.

        Returns
        -------
        numpy.ndarray: The one-hot encoding of the specified genomic region.

        Raises
        ------
        ValueError: If the genome one-hot encoding is not created. Please run `genome.get_genome_one_hot` first.
        """
        if self.genome_one_hot is None:
            raise ValueError(
                "Genome one-hot encoding is not created, please run genome.get_genome_one_hot first."
            )
        return self.genome_one_hot.get_region_one_hot(*args)

    def get_regions_one_hot(self, regions):
        """
        Get the one-hot encoding for the given regions.

        Parameters
        ----------
            regions (list): A list of regions for which to retrieve the one-hot encoding.

        Returns
        -------
            numpy.ndarray: The one-hot encoding for the given regions.

        Raises
        ------
            ValueError: If the genome one-hot encoding is not created. Please run `genome.get_genome_one_hot` first.
        """
        if self.genome_one_hot is None:
            raise ValueError(
                "Genome one-hot encoding is not created, please run genome.get_genome_one_hot first."
            )
        return self.genome_one_hot.get_regions_one_hot(regions)

    def get_global_coords(self, region_bed, chrom_offsets=None):
        """
        Convert the coordinates in the given region bed file to global coordinates.

        Parameters
        ----------
        region_bed : str
            The path to the region bed file.
        chrom_offsets : pd.DataFrame, optional
            A DataFrame containing the global start offsets for each chromosome. Default is None.

        Returns
        -------
        numpy.ndarray
            An array of global coordinates corresponding to the coordinates in the region bed file.

        Notes
        -----
        This method assumes that the `chrom_offsets` attribute has been properly set.

        The `region_bed` file should be in BED format, with columns for chromosome, start position, and end position.

        Examples
        --------
        >>> genome = Genome()
        >>> genome.chrom_offsets = {"chr1": 0, "chr2": 1000}
        >>> coords = genome.get_global_coords("regions.bed")
        >>> print(coords)
        [100 200 300 1100 1200 1300]
        """
        return get_global_coords(
            chrom_offsets=self.chrom_offsets,
            region_bed_df=understand_regions(region_bed, as_df=True),
        )
