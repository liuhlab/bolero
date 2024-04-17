import pathlib
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import pyranges as pr
import ray
import xarray as xr
import zarr
from numcodecs import Zstd
from pyfaidx import Fasta
from tqdm import tqdm

from bolero.pp.seq import DEFAULT_ONE_HOT_ORDER, Sequence
from bolero.utils import (
    download_file,
    get_default_save_dir,
    get_package_dir,
    parse_region_name,
    parse_region_names,
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


def _get_global_coords(chrom_offsets, region_bed_df):
    add_start = (
        region_bed_df["Chromosome"].map(chrom_offsets["global_start"]).astype(int)
    )
    start = region_bed_df["Start"] + add_start
    end = region_bed_df["End"] + add_start
    global_coords = np.hstack([start.values[:, None], end.values[:, None]])
    return global_coords


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
        if isinstance(regions, pr.PyRanges):
            regions_bed = regions
        elif isinstance(regions, pd.DataFrame):
            regions_bed = pr.PyRanges(regions)
        elif isinstance(regions, (str, pathlib.Path)):
            regions_bed = pr.read_bed(regions)
        elif isinstance(regions, (list, pd.Index)):
            regions_bed = parse_region_names(regions)
        else:
            raise ValueError(
                "regions must be a PyRanges, DataFrame, str, Path, list or Index"
            )

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
        regions_bed = pr.PyRanges(use_regions[["Chromosome", "Start", "End", "Name"]])

        if remove_blacklist and self.blacklist_bed is not None:
            regions_bed = self._remove_blacklist(regions_bed)

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

    def get_global_coords(self, region_bed):
        """
        Convert the coordinates in the given region bed file to global coordinates.

        Parameters
        ----------
        region_bed : str
            The path to the region bed file.

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
        return _get_global_coords(
            chrom_offsets=self.chrom_offsets,
            region_bed_df=understand_regions(region_bed, as_df=True),
        )


@ray.remote
def _remote_isel(da, dim, sel):
    # try first sel to get shape
    data_list = []
    for slice_ in enumerate(sel):
        data_list.append(da.isel({dim: slice_}).values)
    return data_list


class GenomeWideDataset:
    """
    Represents a dataset containing genome-wide data.

    Attributes
    ----------
        None

    Methods
    -------
        get_region_data: Retrieves data for a specific genomic region.
        get_regions_data: Retrieves data for multiple genomic regions.

    """

    def __init__(self):
        return

    def get_region_data(self, chrom: str, start: int, end: int) -> Any:
        """
        Retrieves data for a specific genomic region.

        Args:
            chrom (str): The chromosome of the genomic region.
            start (int): The start position of the genomic region.
            end (int): The end position of the genomic region.

        Returns
        -------
            Any: The data for the specified genomic region.

        Raises
        ------
            NotImplementedError: This method should be implemented by a subclass.

        """
        raise NotImplementedError

    def get_regions_data(
        self, regions: List[Tuple[str, int, int]], chunk_size: Optional[int] = None
    ) -> Any:
        """
        Retrieves data for multiple genomic regions.

        Args:
            regions (List[Tuple[str, int, int]]): A list of genomic regions specified as tuples of (chromosome, start, end).
            chunk_size (Optional[int]): The size of each chunk of data to retrieve.

        Returns
        -------
            Any: The data for the specified genomic regions.

        Raises
        ------
            NotImplementedError: This method should be implemented by a subclass.

        """
        raise NotImplementedError


class GenomePositionZarr(GenomeWideDataset):
    """
    Represents a genomic position in a Zarr dataset.

    Parameters
    ----------
    - da (xarray.DataArray): The Zarr dataset.
    - offsets (dict): A dictionary containing the global start offsets for each chromosome.
    - load (bool): Whether to load the dataset into memory. Default is False.
    - pos_dim (str): The name of the position dimension. Default is "pos".

    Attributes
    ----------
    - da (xarray.DataArray): The Zarr dataset.
    - load (bool): Whether the dataset is loaded into memory.
    - pos_dim (str): The name of the position dimension.
    - offsets (dict): The global start offsets for each chromosome.
    - global_start (dict): The global start positions for each chromosome.
    - _remote_da (ray.ObjectRef): The remote reference to the dataset (if not loaded).

    Methods
    -------
    - get_region_data(chrom, start, end): Get the region data for a specific chromosome and range.
    - get_regions_data(regions_df): Get the region data for multiple regions specified in a DataFrame.
    """

    def __init__(self, da, offsets, load=False, pos_dim="pos"):
        super().__init__()
        self.da = da
        self.load = load
        if load:
            self.da.load()

        if "position" in da.dims:
            pos_dim = "position"
        assert pos_dim in da.dims
        self.da = self.da.rename({pos_dim: "pos"})
        self.pos_dim = pos_dim

        self.offsets = offsets
        self.global_start = self.offsets["global_start"].to_dict()

        if load:
            self._remote_da = None
        else:
            self._remote_da = ray.put(self.da)

    def get_region_data(self, chrom, start, end):
        """
        Get the region data for a specific chromosome and range.

        Parameters
        ----------
        - chrom (str): The chromosome name.
        - start (int): The start position of the region.
        - end (int): The end position of the region.

        Returns
        -------
        - region_data (numpy.ndarray): The region data as a NumPy array.
        """
        add_start = self.global_start[chrom]
        global_start = start + add_start
        global_end = end + add_start

        region_data = self.da.isel(pos=slice(global_start, global_end)).values
        return region_data

    def get_regions_data(self, regions, chunk_size=None):
        """
        Get the region data for multiple regions specified in a DataFrame.

        Parameters
        ----------
        - regions_df (pandas.DataFrame): A DataFrame containing the regions to retrieve.

        Returns
        -------
        - regions_data (numpy.ndarray): The region data as a NumPy array.
        """
        if isinstance(regions, pr.PyRanges):
            regions_df = regions.df
        elif isinstance(regions, pd.DataFrame):
            regions_df = regions
        else:
            raise ValueError("regions must be a PyRanges or DataFrame")

        global_coords = _get_global_coords(
            chrom_offsets=self.offsets, region_bed_df=regions_df
        )

        # init an empty array, assume all regions have the same length
        n_regions = len(global_coords)
        region_size = global_coords[0, 1] - global_coords[0, 0]
        shape_list = [n_regions]
        for dim, size in self.da.sizes.items():
            if dim == "pos":
                shape_list.append(region_size)
            else:
                shape_list.append(size)

        regions_data = np.zeros(shape_list, dtype=self.da.dtype)
        if self.load:
            for i, (start, end) in enumerate(global_coords):
                _data = self.da.isel(pos=slice(start, end)).values
                regions_data[i] = _data
        else:
            chunk_size = regions_df.shape[0] if chunk_size is None else chunk_size
            futures = []
            chunk_slices = []
            for chunk_start in range(0, regions_df.shape[0], chunk_size):
                chunk_slice = slice(chunk_start, chunk_start + chunk_size)
                _slice_list = [
                    slice(start, end) for start, end in global_coords[chunk_slice]
                ]
                task = _remote_isel.remote(self._remote_da, "pos", _slice_list)
                futures.append(task)
                chunk_slices.append(chunk_slice)

            data_list = ray.get(futures)
            for chunk_slice, data in zip(chunk_slices, data_list):
                regions_data[chunk_slice] = data
        return regions_data


class GenomeRegionZarr(GenomeWideDataset):
    """
    Represents a genomic region in Zarr format.

    Parameters
    ----------
    da : xarray.DataArray
        The data array containing the genomic region.
    load : bool, optional
        Whether to load the data array into memory, by default False.
    region_dim : str, optional
        The name of the dimension representing the regions, by default "region".

    Attributes
    ----------
    da : xarray.DataArray
        The data array containing the genomic region.
    load : bool
        Whether the data array is loaded into memory.
    region_dim : str
        The name of the dimension representing the regions.
    _remote_da : ray.ObjectRef or None
        A reference to the remote data array if not loaded into memory, None otherwise.

    Methods
    -------
    get_region_data(region)
        Get the data for a specific region.
    get_regions_data(*regions)
        Get the data for multiple regions.

    """

    def __init__(self, da, load=False, region_dim="region"):
        super().__init__()
        self.da = da
        self.load = load
        if load:
            self.da = self.da.load()

        assert region_dim in self.da.dims
        self.da = self.da.rename({region_dim: "region"})
        self.region_dim = region_dim

        if load:
            self._remote_da = None
        else:
            self._remote_da = ray.put(self.da)

    def get_region_data(self, *args, **kwargs):
        """
        Get the data for a specific region.

        Parameters
        ----------
        region : int, slice, or str
            The region to retrieve the data for.

        Returns
        -------
        numpy.ndarray
            The data for the specified region.

        """
        if "chrom" in kwargs and "start" in kwargs and "end" in kwargs:
            chrom = kwargs["chrom"]
            start = kwargs["start"]
            end = kwargs["end"]
            region = f"{chrom}:{start}-{end}"
        else:
            if len(args) == 1:
                region = args[0]
            else:
                region = pd.Index(args)

        if isinstance(region, (int, slice)):
            region_data = self.da.isel(region=region).values
        else:
            region_data = self.da.sel(region=region).values
        return region_data

    def get_regions_data(self, regions, chunk_size=None):
        """
        Get the data for multiple regions.

        Parameters
        ----------
        regions : int, slice, or str
            The regions to retrieve the data for.

        Returns
        -------
        numpy.ndarray
            The data for the specified regions.

        """
        # chunk size is not really used here, just be consistent with other data classes
        _ = len(regions) if chunk_size is None else chunk_size

        if isinstance(regions, pr.PyRanges):
            regions_df = regions.df
        elif isinstance(regions, pd.DataFrame):
            regions_df = regions
        else:
            regions_df = None
        if regions_df is not None:
            if "Name" in regions_df.columns:
                regions = regions_df["Name"]
            else:
                regions = []
                for _, (chrom, start, end, *_) in regions_df.iterrows():
                    regions.append(f"{chrom}:{start}-{end}")

        _data = self.get_region_data(regions)
        return _data


class GenomeOneHotZarr(GenomePositionZarr):
    """
    A class for working with one-hot encoded genomic data stored in Zarr format.

    Parameters
    ----------
    ds_path : str
        The path to the Zarr dataset.
    load : bool, optional
        Whether to load the dataset into memory, by default True.

    Attributes
    ----------
    ds : xr.Dataset
        The Zarr dataset.
    one_hot : xr.DataArray
        The one-hot encoded genomic data.

    Methods
    -------
    __repr__()
        Returns a string representation of the Zarr dataset.
    get_region_one_hot(*args)
        Get the one-hot encoded representation of a genomic region.
    get_regions_one_hot(regions)
        Get the one-hot encoded representation of the given regions.

    """

    def __init__(self, ds_path, load=True):
        self.ds = xr.open_zarr(ds_path)
        self.one_hot = self.ds["X"]
        if load:
            print("Loading genome DNA one-hot encoding...")
            self.one_hot.load()
        super().__init__(
            da=self.one_hot,
            offsets=self.ds["offsets"].to_pandas(),
            load=load,
            pos_dim="pos",
        )

    def __repr__(self):
        """
        Returns a string representation of the Zarr dataset.

        Returns
        -------
        str
            The string representation of the Zarr dataset.

        """
        return self.ds.__repr__()

    def get_region_one_hot(self, *args):
        """
        Get the one-hot encoded representation of a genomic region.

        Parameters
        ----------
        args : tuple
            If a single argument is provided, it is assumed to be a region name
            and will be parsed into chromosome, start, and end coordinates.
            If three arguments are provided, they are assumed to be chromosome,
            start, and end coordinates directly.

        Returns
        -------
        region_one_hot : numpy.ndarray
            The one-hot encoded representation of the genomic region.

        Raises
        ------
        ValueError
            If the number of arguments is not 1 or 3.

        """
        if len(args) == 1:
            # assume it's a region name
            chrom, start, end = parse_region_name(args[0])
        elif len(args) == 3:
            # assume it's chrom, start, end
            chrom, start, end = args
        else:
            raise ValueError("args must be a region name or chrom, start, end")

        region_one_hot = self.get_region_data(chrom, start, end)
        return region_one_hot

    def get_regions_one_hot(self, regions):
        """
        Get the one-hot encoded representation of the given regions.

        Parameters
        ----------
        regions : pd.DataFrame or pr.PyRanges or str or list
            The regions to be encoded. It can be provided as a pandas DataFrame,
            a PyRanges object, a string representing a region name, or a list of region names.

        Returns
        -------
        np.ndarray
            The one-hot encoded representation of the regions.

        Raises
        ------
        AssertionError
            If the regions have different lengths.

        """
        # get global coords
        if isinstance(regions, pd.DataFrame):
            regions = regions[["Chromosome", "Start", "End"]]
        elif isinstance(regions, pr.PyRanges):
            regions = regions.df[["Chromosome", "Start", "End"]]
        elif isinstance(regions, str):
            regions = parse_region_names([regions]).df[["Chromosome", "Start", "End"]]
        else:
            regions = parse_region_names(regions).df[["Chromosome", "Start", "End"]]
        global_coords = _get_global_coords(
            chrom_offsets=self.offsets, region_bed_df=regions
        )

        # make sure regions are in the same length
        region_lengths = global_coords[:, 1] - global_coords[:, 0]
        assert (
            region_lengths == region_lengths[0]
        ).all(), "All regions must have the same length."

        region_one_hot = self.get_regions_data(regions)
        return region_one_hot


def _bw_values(bw, chrom, start, end):
    # inside bw, always keep numpy true
    _data = bw.values(chrom, start, end, numpy=True)
    return _data


def _bw_values_chunk(bw, regions):
    regions_data = []
    for _, (chrom, start, end, *_) in regions.iterrows():
        regions_data.append(_bw_values(bw=bw, chrom=chrom, start=start, end=end))
    regions_data = np.array(regions_data)
    regions_data.astype("float32", copy=False)
    np.nan_to_num(regions_data, copy=False)
    return regions_data


@ray.remote
def _remote_bw_values_chunk(bw_path, regions):
    with pyBigWig.open(bw_path) as bw:
        return _bw_values_chunk(bw, regions)


class GenomeBigWigDataset(GenomeWideDataset):
    """Represents a genomic dataset stored in BigWig format."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Represents a genomic dataset stored in BigWig format.

        Parameters
        ----------
        *args : str
            The paths to the BigWig files. The dataset names will be inferred from the file names.
        **kwargs : str
            The paths to the BigWig files, with the dataset names as the keys.
        """
        super().__init__()
        self.bigwig_path_dict = {}
        self.add_bigwig(*args, **kwargs)

        self._opened_bigwigs = {}

    def __repr__(self):
        repr_str = f"GenomeBigWigDataset ({len(self.bigwig_path_dict)} bigwig)\n"
        for name, path in self.bigwig_path_dict.items():
            repr_str += f"{name}: {path}\n"
        return repr_str

    def add_bigwig(self, *args, **kwargs):
        """
        Add a BigWig file to the dataset.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the BigWig file.
        name : str, optional
            The name of the dataset, by default None.
        """
        for key, value in kwargs.items():
            self.bigwig_path_dict[key] = str(value)
        for arg in args:
            name = pathlib.Path(arg).name
            self.bigwig_path_dict[name] = str(arg)

    def _open(self) -> None:
        """
        Open the BigWig files.
        """
        for name, path in self.bigwig_path_dict.items():
            self._opened_bigwigs[name] = pyBigWig.open(path)

    def _close(self) -> None:
        """
        Close the opened BigWig files.
        """
        for bw in self._opened_bigwigs.values():
            bw.close()
        self._opened_bigwigs = {}

    def __enter__(self) -> "GenomeBigWigDataset":
        """
        Enter the context manager and open the BigWig files.
        """
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the context manager and close the opened BigWig files.
        """
        self._close()

    def get_region_data(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> Dict[str, np.ndarray]:
        """
        Get the data for a specific genomic region.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        start : int
            The start position of the region.
        end : int
            The end position of the region.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the region data for each dataset,
            where the keys are the dataset names and the values are the data arrays.
        """
        with self:
            region_data = {}
            for name, bw in self._opened_bigwigs.items():
                region_data[name] = _bw_values(bw, chrom, start, end)
        return region_data

    def get_regions_data(
        self,
        regions: Union[pr.PyRanges, pd.DataFrame],
        chunk_size: Optional[int] = None,
    ) -> Dict[str, Union[np.ndarray, List[float]]]:
        """
        Get the data for multiple genomic regions.

        Parameters
        ----------
        regions : pr.PyRanges or pd.DataFrame
            The regions to retrieve data for.
        chunk_size : int, optional
            The number of regions to process in each chunk, by default None.

        Returns
        -------
        Dict[str, Union[np.ndarray, List[float]]]
            A dictionary containing the region data for each dataset,
            where the keys are the dataset names and the values are the data arrays or lists.

        Raises
        ------
        ValueError
            If the regions parameter is not a PyRanges or DataFrame.
        """
        if isinstance(regions, pr.PyRanges):
            regions_df = regions.df
        elif isinstance(regions, pd.DataFrame):
            regions_df = regions
        else:
            raise ValueError("regions must be a PyRanges or DataFrame")

        if chunk_size is None:
            chunk_size = regions_df.shape[0]

        names = []
        tasks = []
        for name, path in self.bigwig_path_dict.items():
            this_tasks = []
            for chunk_start in range(0, regions_df.shape[0], chunk_size):
                chunk_slice = slice(chunk_start, chunk_start + chunk_size)
                regions = regions_df.iloc[chunk_slice, :3].copy()
                this_tasks.append(_remote_bw_values_chunk.remote(path, regions))
            tasks.append(this_tasks)
            names.append(name)

        regions_data = {}
        for name, task in zip(names, tasks):
            regions_data[name] = np.concatenate(ray.get(task))
        return regions_data


class GenomeEnsembleDataset:
    """
    Represents an ensemble dataset for genomic data.

    Parameters
    ----------
        genome (str or Genome): The genome associated with the dataset.

    Attributes
    ----------
        genome (Genome): The genome associated with the dataset.
        datasets (dict): A dictionary of dataset names and corresponding GenomeWideDataset objects.

    """

    def __init__(self, genome: Union[str, Genome]):
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome

        self.datasets = {}
        # special slots
        self.add_genome_one_hot()
        self._bigwig_dataset = None

        self.regions = {}
        self.region_sizes = {}
        self._n_regions = None
        self._region_dataset_query_pairs = set()

    def __repr__(self):
        repr_str = f"GenomeEnsembleDataset(genome={self.genome.name})\n"

        _s = f"\nDatasets ({len(self.datasets)}):\n"
        repr_str += "\n" + "=" * (len(_s) - 1) + _s + "=" * (len(_s) - 1)

        for name, dataset in self.datasets.items():
            # also collect information about which regions will be queried in this dataset
            regions = []
            for region_name, dataset_name in self._region_dataset_query_pairs:
                if dataset_name == name:
                    regions.append(region_name)
            regions_str = "Query by regions: " + ", ".join(regions)
            dataset_str = dataset.__repr__().strip("\n")
            repr_str += (
                f"\n{name}: {type(dataset).__name__}\n{dataset_str}\n{regions_str}\n"
            )

        _s = f"\nRegions ({len(self.regions)}):\n"
        repr_str += "\n" + "=" * (len(_s) - 1) + _s + "=" * (len(_s) - 1) + "\n"

        for name, regions in self.regions.items():
            length = self.region_sizes[name]
            repr_str += f"{name}: {len(regions)} regions, region length {length} bp.\n"
        return repr_str

    def __getitem__(self, key):
        return self.datasets[key]

    def __setitem__(self, key, value):
        if key in self.datasets:
            raise ValueError(f"Dataset {key} already exists.")
        self.datasets[key] = value
        return

    def __delitem__(self, key):
        del self.datasets[key]
        return

    def add_regions(
        self,
        name,
        regions,
        length=2500,
        query_datasets="all",
        remove_blacklist=True,
        boarder_strategy="drop",
    ):
        """
        Adds regions to the ensemble.

        The regions will be used to retrieve region-level data from the genome datasets.

        Parameters
        ----------
            name (str): The name of the regions.
            regions (str, pathlib.Path, PyRanges or pd.DataFrame): The regions bed table to add.
            length (int): The length of the regions to standardize to.
                If None, the regions will not be standardized and
                user must ensure all regions have the same length
                and do not exceed the genome boarder. Default is 2500.
            query_datasets (str or List[str]): The datasets to query when retrieving region data.
                Default is 'all', which queries all datasets in self.datasets.
            check_length (bool): Whether to check if all regions have the same length. Default is False.
            remove_blacklist (bool): Whether to remove regions that overlap with blacklisted regions. Default is True.
            boarder_strategy (str): The stratagy to handle regions that go beyond the genome boarder. Default is 'drop'. See `Genome.standard_region_length` for more details.

        """
        regions = understand_regions(regions, as_df=True)
        if length is not None:
            regions = self.genome.standard_region_length(
                regions=regions,
                length=length,
                boarder_strategy=boarder_strategy,
                remove_blacklist=remove_blacklist,
                as_df=True,
            )
            if remove_blacklist:
                # region length may change after removing blacklisted regions
                regions = regions[(regions["End"] - regions["Start"]) == length].copy()
            region_size = length
        else:
            region_size = regions.iloc[0, 2] - regions.iloc[0, 1]
            region_lengths = regions["End"] - regions["Start"]
            assert (
                region_lengths == region_size
            ).all(), "All regions must have the same length."

        self.regions[name] = regions
        self.region_sizes[name] = region_size

        if self._n_regions is None:
            self._n_regions = len(regions)
        else:
            assert (
                len(regions) == self._n_regions
            ), "All region beds must have the same number of regions."

        if query_datasets == "all":
            query_datasets = list(self.datasets.keys())
        elif isinstance(query_datasets, str):
            query_datasets = [query_datasets]
        else:
            query_datasets = list(query_datasets)

        for dataset_name in query_datasets:
            assert dataset_name in self.datasets, f"Dataset {dataset_name} not found."
            self._region_dataset_query_pairs.add((name, dataset_name))
        return

    def add_bigwig(
        self,
        name="bigwig",
        *args,
        **kwargs,
    ):
        """
        Adds a BigWig dataset to the ensemble.

        Parameters
        ----------
            name (str): The name of the dataset.
            bigwig_path (str or List[str] or pathlib.Path): The path(s) to the BigWig file(s).

        """
        bigwig_dict = {}
        for key, value in kwargs.items():
            bigwig_dict[key] = value
        for arg in args:
            _path = pathlib.Path(arg)
            bigwig_dict[_path.name] = str(_path)

        if self._bigwig_dataset is None:
            self._bigwig_dataset = GenomeBigWigDataset(**bigwig_dict)
            self.datasets[name] = self._bigwig_dataset
        else:
            self._bigwig_dataset.add_bigwig(**bigwig_dict)
        return

    def add_position_zarr(
        self,
        zarr_path: Union[str, pathlib.Path, xr.Dataset, xr.DataArray],
        name: str = None,
        da_name: str = None,
        load: bool = False,
    ):
        """
        Adds a position Zarr dataset to the ensemble.

        Parameters
        ----------
            name (str): The name of the dataset.
            zarr_path (str or pathlib.Path): The path to the Zarr dataset.
            load (bool): Whether to load the dataset into memory.

        """
        if isinstance(zarr_path, (str, pathlib.Path)):
            zarr = xr.open_zarr(zarr_path)

        if isinstance(zarr, xr.Dataset):
            if da_name is None:
                _data_vars = list(zarr.data_vars)
                if len(_data_vars) != 1:
                    raise ValueError(
                        "da_name must be specified if there is more than one data variable in the Zarr dataset. "
                        "Available data variables: " + ", ".join(_data_vars)
                    )
                else:
                    da_name = list(zarr.data_vars)[0]
            zarr = zarr[da_name]

        if isinstance(zarr, xr.DataArray):
            self.datasets[name] = GenomePositionZarr(
                da=zarr, offsets=self.genome.chrom_offsets, load=load
            )
        else:
            raise ValueError(
                "zarr must be a path to a Zarr dataset or an xarray Dataset or DataArray."
            )
        return

    def add_genome_one_hot(self, name="dna_one_hot"):
        """
        Adds the genome one-hot encoding to the ensemble.

        Parameters
        ----------
            name (str): The name of the dataset.

        """
        self.datasets[name] = self.genome.genome_one_hot
        return

    def get_region_data(self, *args) -> Dict[str, Any]:
        """
        Retrieves the data for a specific region.

        Parameters
        ----------
            *args: Either a region name or chrom, start, end values.

        Returns
        -------
            region_data (dict): A dictionary containing the data for each dataset.

        Raises
        ------
            ValueError: If args is not a region name or chrom, start, end values.

        """
        if len(args) == 1:
            chrom, start, end = parse_region_name(args[0])
        elif len(args) == 3:
            chrom, start, end = args
        else:
            raise ValueError("args must be a region name or chrom, start, end")

        region_data = {}
        for name, dataset in self.datasets.items():
            _data = dataset.get_region_data(chrom=chrom, start=start, end=end)
            if isinstance(_data, dict):
                region_data.update(_data)
            else:
                region_data[name] = _data
        return region_data

    def get_regions_data(
        self, query_chunk_size: int = 5000, region_index: pd.Index = None
    ) -> Dict[str, Any]:
        """
        Retrieves the data for multiple regions.

        Parameters
        ----------
            chunk_size (int): The size of each chunk of regions during parallel retrieval. Default is 5000.
            region_index (pd.Index): The index of regions to retrieve data for. Default is None.


        Returns
        -------
            regions_data (dict): A dictionary containing the data for each dataset.

        """
        data_collections = {}
        for region_name, dataset_name in self._region_dataset_query_pairs:
            region_name = region_name.replace("|", "_")
            dataset_name = dataset_name.replace("|", "_")

            regions = self.regions[region_name]
            if region_index is not None:
                regions = regions.iloc[region_index, :].copy()

            dataset = self.datasets[dataset_name]
            regions_data = dataset.get_regions_data(
                regions=regions, chunk_size=query_chunk_size
            )
            if isinstance(regions_data, dict):
                for _ds_name, _data in regions_data.items():
                    _final_name = f"{region_name}|{_ds_name}"
                    data_collections[_final_name] = _data
            else:
                _final_name = f"{region_name}|{dataset_name}"
                data_collections[_final_name] = regions_data
            data_collections["region_ids"] = (
                regions["Chromosome"].astype(str)
                + ":"
                + regions["Start"].astype(str)
                + "-"
                + regions["End"].astype(str)
            ).values
        return data_collections

    def _get_ray_dataset(self, n_regions, collate_fn_dict=None, region_index=None):
        """
        Internal method to get a Ray dataset.

        Parameters
        ----------
            regions: The regions for which to retrieve the data.
            collate_fn_dict (dict): A dictionary of collate functions for each dataset. The keys can be the dataset name or the region name or their combination.
            Each collate function should take a numpy array as input and return a summary statistic.
            region_index (pd.Index): The index of regions to retrieve data for.

        Returns
        -------
            ds: The Ray dataset.

        """
        data_collections = self.get_regions_data(
            query_chunk_size=5000, region_index=region_index
        )

        # calculate summary stats
        summary_stats_collections = {}
        if collate_fn_dict:
            for _final_name, _data in data_collections.items():
                try:
                    _region_name, _ds_name = _final_name.split("|")
                except ValueError:
                    _region_name, _ds_name = "", ""
                if _final_name in collate_fn_dict:
                    _funcs = collate_fn_dict[_final_name]
                elif _ds_name in collate_fn_dict:
                    _funcs = collate_fn_dict[_ds_name]
                elif _region_name in collate_fn_dict:
                    _funcs = collate_fn_dict[_region_name]
                else:
                    _funcs = None
                if _funcs:
                    summary_stats_collections[_final_name] = _funcs(_data)

        # reorganize data into items
        item_dicts = []
        for idx in range(n_regions):
            item_dict = {}
            for name, regions_data in data_collections.items():
                item_dict[name] = regions_data[idx]
            item_dicts.append(item_dict)

        ds = ray.data.from_items(item_dicts)
        return ds, summary_stats_collections

    @classmethod
    def subset_regions(cls, dataset, region_sel):
        """
        Subsets the regions in the ensemble.

        Parameters
        ----------
            region_sel: The regions to select.

        Returns
        -------
            ensemble: The ensemble with the selected regions.

        """
        ensemble = cls(dataset.genome)
        ensemble.datasets = dataset.datasets
        ensemble.regions = {
            k: v.iloc[region_sel, :].copy() for k, v in dataset.regions.items()
        }
        ensemble.region_sizes = dataset.region_sizes.copy()
        ensemble._n_regions = len(ensemble.regions)
        ensemble._region_dataset_query_pairs = dataset._region_dataset_query_pairs
        return ensemble

    def prepare_ray_dataset(
        self,
        output_dir: str,
        dataset_size: int = 500000,
        collate_fn_dict: dict = None,
        region_index: pd.Index = None,
    ) -> None:
        """
        Prepares a Ray dataset for the given regions.

        Parameters
        ----------
            output_dir (str): The directory path to save the dataset.
            dataset_size (int): The maximum size of each dataset.
            collate_fn_dict (dict): A dictionary of collate functions for each dataset.
                The keys can be the dataset name or the region name or their combination.
                Each collate function should take a numpy array as input and return a summary statistic.
                Data will be saved by joblib.dump.
            region_index (pd.Index): The index of regions to retrieve data for.

        """
        if region_index is None:
            n_regions = self._n_regions
        else:
            n_regions = region_index.size

        dataset_path = f"{output_dir}/dataset/"
        stats_path = f"{output_dir}/stats/"
        success_flag_path = f"{output_dir}/success.flag"

        # save the dataset
        if n_regions <= dataset_size:
            from pyarrow import ArrowInvalid
            from pyarrow.fs import FileSystem, LocalFileSystem

            try:
                fs, path = FileSystem.from_uri(dataset_path)
            except ArrowInvalid:
                # assume local filesystem
                dataset_path = str(pathlib.Path(dataset_path).absolute().resolve())
                fs, path = FileSystem.from_uri(dataset_path)

            # check if success flag exists
            success = False
            if isinstance(fs, LocalFileSystem):
                if pathlib.Path(success_flag_path).exists():
                    success = True
            else:
                file_type = fs.get_file_info(success_flag_path).type
                if file_type:
                    success = True
            if success:
                print(f"Dataset already exists at {dataset_path}.")
                return

            ds, summary_stats_collections = self._get_ray_dataset(
                n_regions=n_regions,
                collate_fn_dict=collate_fn_dict,
                region_index=region_index,
            )
            ds.write_parquet(path, filesystem=fs)

            # save summary stats
            summary_stats_path = f"{stats_path}/summary_stats.npz"
            # determine if fs is local filesystem
            if isinstance(fs, LocalFileSystem):
                stats_path = pathlib.Path(stats_path)
                stats_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(summary_stats_path, **summary_stats_collections)
            else:
                with fs.open_output_stream(summary_stats_path) as f:
                    np.savez_compressed(f, **summary_stats_collections)

            # create success flag
            if isinstance(fs, LocalFileSystem):
                with open(success_flag_path, "w") as f:
                    f.write("Success")
            else:
                with fs.open_output_stream(success_flag_path) as f:
                    f.write("Success")
            return

        # split regions into chunks and save each chunk as a separate dataset
        else:
            starts = list(range(0, n_regions, dataset_size))
            for chunk_start in tqdm(
                starts, desc=f"Preparing dataset chunks of size {dataset_size} "
            ):
                chunk_end = min(chunk_start + dataset_size, n_regions)
                chunk_slice = slice(chunk_start, chunk_end)
                if region_index is None:
                    chunk_region_index = pd.Index(range(n_regions))[chunk_slice].copy()
                else:
                    chunk_region_index = region_index[chunk_slice].copy()

                chunk_output_path = f"{output_dir}/chunk_{chunk_start}_{chunk_end}"
                self.prepare_ray_dataset(
                    output_dir=chunk_output_path,
                    dataset_size=dataset_size,
                    collate_fn_dict=collate_fn_dict,
                    region_index=chunk_region_index,
                )
        return


def prepare_chromosome_dataset(
    genome: str,
    output_dir: str,
    regions_config: Union[str, Dict[str, str], List[str]],
    bigwig_config: Optional[Union[str, Dict[str, str], List[str]]] = None,
    zarr_config: Optional[Union[str, Dict[str, str], List[str]]] = None,
    collate_fn_dict: Optional[Dict[str, Callable]] = None,
) -> None:
    """
    Prepare chromosome datasets for a given genome.

    Parameters
    ----------
    genome : str
        The genome associated with the dataset.
    output_dir : str
        The directory to save the prepared datasets.
    regions_config : Union[str, Dict[str, str], List[str]]
        The configuration for the regions. It can be a single path string, a dictionary
        mapping region names to paths, or a list of paths.
    bigwig_config : Optional[Union[str, Dict[str, str], List[str]]], optional
        The configuration for the BigWig files, by default None. It can be a single path
        string, a dictionary mapping dataset names to paths, or a list of paths.
    zarr_config : Optional[Union[str, Dict[str, str], List[str]]], optional
        The configuration for the Zarr files, by default None. It can be a single path
        string, a dictionary mapping dataset names to paths, or a list of paths.
    collate_fn_dict : Optional[Dict[str, Callable]], optional
        A dictionary of collate functions for each dataset. The keys can be the dataset name
        or the region name or their combination. Each collate function should take a numpy
        array as input and return a summary statistic. Data will be saved by joblib.dump.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an invalid path type is provided.

    Example
    -------
    >>> genome = "mm10"
    >>> output_dir = "/mnt/datasets/chromosomes"
    >>> regions_config = {
    ...     "enhancers": "/mnt/data/regions/enhancers.bed",
    ...     "promoters": "/mnt/data/regions/promoters.bed",
    ... }
    >>> bigwig_config = {
    ...     "sample1": "/mnt/data/bigwig/sample1.bw",
    ...     "sample2": "/mnt/data/bigwig/sample2.bw",
    ... }
    >>> zarr_config = {
    ...     "methylation": "/mnt/data/zarr/methylation.zarr",
    ...     "histone_modifications": "/mnt/data/zarr/histone_modifications.zarr",
    ... }
    >>> prepare_chromosome_dataset(
    ...     genome, output_dir, regions_config, bigwig_config, zarr_config
    ... )
    """

    def _str_path_to_dict(p: Union[str, Dict[str, str], List[str]]) -> Dict[str, str]:
        if p is None:
            return {}
        if isinstance(p, dict):
            p = {k: str(pathlib.Path(v).absolute().resolve()) for k, v in p.items()}
            return p
        elif isinstance(p, (str, pathlib.Path)):
            ps = [pathlib.Path(p).absolute().resolve()]
        elif isinstance(p, list):
            ps = [pathlib.Path(pp).absolute().resolve() for pp in p]
        else:
            raise ValueError("Invalid path type.")
        p_dict = {pp.name: str(pp) for pp in ps}
        return p_dict

    # standardize paths
    regions_config = _str_path_to_dict(regions_config)
    bigwig_config = _str_path_to_dict(bigwig_config)
    zarr_config = _str_path_to_dict(zarr_config)

    # group each region by chrom and prepare the region configs
    chrom_region_configs = defaultdict(dict)
    for region_name, region_path in regions_config.items():
        region_bed = pr.read_bed(region_path, as_df=True)
        for chrom, chrom_region_bed in region_bed.groupby("Chromosome"):
            chrom_region_configs[chrom][region_name] = chrom_region_bed

    # prepare the dataset for each chromosome
    bar = tqdm(
        chrom_region_configs.items(),
        desc="Preparing chromosome datasets",
        total=len(chrom_region_configs),
    )
    if isinstance(genome, str):
        genome = Genome(genome)
    for chrom, chrom_region_config in bar:
        ensemble = GenomeEnsembleDataset(genome)

        if bigwig_config:
            ensemble.add_bigwig(**bigwig_config)

        if zarr_config:
            for name, zarr_path in zarr_config.items():
                ensemble.add_position_zarr(zarr_path=zarr_path, name=name)

        for n, p in chrom_region_config.items():
            ensemble.add_regions(name=n, regions=p, query_datasets="all")

        output_path = f"{output_dir}/{chrom}"
        ensemble.prepare_ray_dataset(
            output_dir=output_path, dataset_size=500000, collate_fn_dict=collate_fn_dict
        )
    return
