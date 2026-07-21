"""
Scripts adapted from scPrinter:
https://github.com/buenrostrolab/scPrinter/blob/main/LICENSE
"""

import itertools
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Literal

try:
    import MOODS
    import MOODS.parsers
    import MOODS.scan
    import MOODS.tools
except ModuleNotFoundError as e:  # pragma: no cover - optional git-only backend
    raise ModuleNotFoundError(
        "bolero.tl.motif.scan requires the 'MOODS' motif scanner. Install our fork with "
        "`pip install git+https://github.com/liuhlab/MOODS_pixi.git`, or use the pixi "
        "environment (which declares it). See docs/installation.md."
    ) from e
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm.auto import trange


def consecutive(data, stepsize=1):
    """
    Find consecutive numbers in a list, used to merge overlapping hits.

    Parameters
    ----------
    data : list
        A list of numbers.
    stepsize : int, optional
        The difference between consecutive numbers. Default is 1.

    Returns
    -------
    list
        A list of consecutive numbers. Each element in the list is a numpy array of consecutive numbers.
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


# This function is necessary for the MOODS package to work under multiprocessing
def _thresholds_from_p(m, bg, pvalue):
    return MOODS.tools.threshold_from_p(m, bg, pvalue)


class PFM:
    """
    A simple wrapper for PFM matrices to make it compatible with Bio.motifs
    """

    def __init__(self, name, counts):
        self.name = name
        self.counts = counts
        self.length = len(counts["A"])


def parse_jaspar(file_path):
    """
    Parse Jaspar-like format motifs.

    Parameters
    ----------
    file_path : str
        The path to the file containing the motifs in Jaspar format.

    Returns
    -------
    list of PFM objects
        A list of PFM (Position Frequency Matrix) objects representing the parsed motifs.

    Notes
    -----
    The Jaspar format is a simple text format used to store position frequency matrices (PFMs)
    representing transcription factor binding sites. Each motif is represented by a name and
    a set of weights for each nucleotide (A, C, G, T) at each position.

    The format of the input file is as follows:
    - Each motif is represented by a line starting with ">".
    - The name of the motif follows the ">" symbol.
    - Each line representing the weights for a nucleotide starts with the nucleotide symbol.
    - The weights for each position are separated by spaces.

    Example
    -------
    The following is an example of a Jaspar format file:

    >MA0004.1_Drosophila_melanogaster
    A [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
    C [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
    G [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
    T [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]

    """
    # Initialize variables
    records = []
    record = None

    # Open the file for reading
    with open(file_path) as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line.startswith(">"):  # Record name line
                # Save the previous record if it exists
                if record:
                    records.append(PFM(record["name"], record["weights"]))
                # Start a new record
                record = {
                    "name": line[1:].split(" ")[-1],
                    "weights": {"A": [], "C": [], "G": [], "T": []},
                }
            elif len(line) > 0:
                # Parse the weight line
                nucleotide, values_str = line.split(" ", 1)
                values = list(map(float, values_str.strip(" ").strip("[]").split()))
                if record:
                    record["weights"][nucleotide] = values

        # Don't forget to add the last record after exiting the loop
        if record:
            records.append(PFM(record["name"], record["weights"]))

    return records


def _jaspar_to_moods_matrix(jaspar_motif, bg, pseudocount, mode="motifmatchr"):
    """
    Convert a JASPAR motif to a MOODS matrix.

    Parameters
    ----------
    jaspar_motif : list of PFM matrices
        The JASPAR motif to be converted.

    bg : tuple of float
        The background nucleotide frequency.

    pseudocount : float
        The pseudocount for the motif.

    mode : str, optional
        The mode of the motif matching. By default, it is set to "motifmatchr".

    Returns
    -------
    m : list of tuples
        The converted MOODS matrix. Each tuple represents the log odds ratio for each nucleotide.
    """
    with tempfile.NamedTemporaryFile() as fn:
        f = open(fn.name, "w")
        for base in "ACGT":
            line = " ".join(str(x) for x in jaspar_motif.counts[base])
            f.write(line + "\n")

        f.close()
        if mode != "motifmatchr":
            m = MOODS.parsers.pfm_to_log_odds(fn.name, bg, pseudocount, 2)
        else:
            # This is to be consistent with motifmatchr
            even = [0.25, 0.25, 0.25, 0.25]
            m = MOODS.parsers.pfm_to_log_odds(fn.name, even, pseudocount, 2)
            m = [tuple(m[i] - (np.log2(0.25) - np.log2(bg[i]))) for i in range(len(bg))]
        return m


def scan_func(
    seq,
    motifs_length,
    motifs_name,
    chr,
    start,
    end,
    peak_idx,
    tfs,
    clean,
    split_tfs,
    strand,
):
    """
    Scan a sequence for motifs. This function is used as a global function for multiprocessing.
    This is basically the _parse_scan_results_all function, but with the scanner scan function within each child process.

    Parameters
    ----------
    seq : str
        String of "ACGT" representing the sequence to scan for motifs.
    motifs_length : list | np.ndarray
        Length of the motifs. This is necessary to infer the end positons of the motif matches.
    motifs_name : list | np.ndarray
        Name of the motifs.
    chr : str
        Chromosome name.
    start : int
        Start position of the sequence.
    end : int
        End position of the sequence.
    peak_idx : int
        The index of the peak (to associate results back to the peak).
    tfs : set
        Set of TFs to keep. (For cistarget motif collection, if one motif keeps 20 TFs, you can use this to keep only the TF you care about).
    clean : bool
        Whether to clean the results (merge overlapping hits).
    split_tfs : bool
        Whether to split the TFs by comma (for cistarget motif collection).
    strand : bool
        Whether to keep the strand information.

    Returns
    -------
    list
        List of parsed scan results. Each result is a list containing the motif start position, motif end position, score, and other relevant information.
    """
    # fetch the MOODS scanner and scan the sequence
    global scanner
    scan_res = scanner.scan(str(seq))
    # parse the results, keep only the position and score information
    results = [[[xxx.pos, xxx.score] for xxx in xx] for xx in scan_res]

    # parse the scan results.
    return _parse_scan_results_all(
        results,
        motifs_length,
        motifs_name,
        {"chr": chr, "start": start, "end": end, "index": peak_idx},
        tfs,
        clean,
        split_tfs,
        strand,
    )


def _parse_scan_results_all(
    moods_scan_res: list,
    motifs_length: list | np.ndarray,
    motifs_name: list | np.ndarray,
    bed_coord: dict,
    tfs: set,
    clean=False,
    split_tf=True,
    strand_spec=True,
):
    """
    Parse results of MOODS.scan.scan_dna and return

    Parameters
    ----------
    moods_scan_res: list
        list of scan results from MOODS length of motifs x 2 (first forward then reverse strand)
        each item in the list is another list of scanned results [position, scores]
    motifs_length: list | np.ndarray
        length of the motifs
    motifs_name: list | np.ndarray
        name of the motifs
    bed_coord: dict
        coordinates of the peak
    tfs: set
        set of TFs to keep. (for cistarget motif collection, if one motif keeps 20 TFs, you can use this to keep only the TF you care)
    clean: bool
        clean the results (merge overlapping hits)
    split_tf: bool
        split the TFs by comma (for cistarget motif collection)
    strand_spec:
        whether to keep the strand information
    """
    all_hits = []  # final return, lists of hits in this region.

    # First group by motif.name
    tf2results = {}
    num_motifs = len(motifs_name)

    for motif_idx, hits in enumerate(moods_scan_res):
        # The scan results from MOODS is organized in a way that the first half is the forward strand,
        # the second half is the reverse strand, hence idx % len(motifs_names), gives the correct motif name
        # get information of motif_name, length and stand information (which is * if strand_spec=False)
        motif_name = motifs_name[motif_idx % num_motifs]
        motif_name = motif_name.split(",") if split_tf else [motif_name]
        motif_length = motifs_length[motif_idx % num_motifs]
        strand = -1 if motif_idx >= len(motifs_name) else 1
        if not strand_spec:
            strand = "*"

        for name in motif_name:
            if name not in tfs:
                continue

            # Maintain a dictionary of tf name, strand to results.
            if (name, strand) not in tf2results:
                tf2results[(name, strand)] = []

            v = []
            for h in hits:
                # get the motif start position
                motif_start = int(h[0])
                motif_end = int(h[0]) + motif_length
                score = round(h[1], 4)
                # If direct output, then organize the record directly without merge overlapping hits
                if not clean:
                    record = [
                        bed_coord["chr"],
                        bed_coord["start"],
                        bed_coord["end"],
                        bed_coord["index"],
                        name,
                        score,
                        strand,
                        motif_start,
                        motif_end,
                    ]
                    all_hits.append(record)
                else:
                    # If wants to clean, then stack them directly then clean later.
                    motif_start = int(h[0])
                    motif_end = int(h[0]) + motif_length
                    v.append([motif_start, motif_end])
            if clean:
                tf2results[(name, strand)] += v
    if not clean:
        return all_hits

    for key in tf2results:
        name, strand = key
        v = tf2results[(name, strand)]
        if len(v) == 0:
            continue
        v = [np.arange(s, e) for s, e in v]
        v = np.unique(np.concatenate(v).reshape(-1))
        v = np.sort(v)
        v = consecutive(v)
        for h in v:
            motif_start = np.min(h)
            motif_end = np.max(h)
            record = [
                bed_coord["chr"],
                bed_coord["start"],
                bed_coord["end"],
                bed_coord["index"],
                name,
                0,
                strand,
                motif_start,
                motif_end,
            ]

            all_hits.append(record)
    return all_hits


class MutationManager:
    def __init__(
        self, mutation_table: pd.DataFrame, use_genotype: Literal["ref", "alt"] = "ref"
    ):
        """
        Manager for mutation information.

        Parameters
        ----------
        mutation_table: pd.DataFrame
            Table containing mutation information. Position is 1-based.
            Columns: ["Chromosome", "Position", "Ref", "Alt"]
        """
        assert (
            mutation_table.shape[1] == 4
        ), "Mutation table must have 4 columns: Chromosome, Position, Ref, Alt"
        # select SNV only
        snv_sel = (mutation_table["Ref"].str.len() == 1) & (
            mutation_table["Alt"].str.len() == 1
        )
        if not snv_sel.all():
            print(
                f"Warning: {snv_sel.sum()} non-SNV mutations found in mutation table, only SNV mutations are supported."
            )
        mutation_table = mutation_table[snv_sel].copy()
        mutation_table.iloc[:, 1] = mutation_table.iloc[:, 1] - 1
        mutation_table.columns = ["Chromosome", "Position0Based", "Ref", "Alt"]
        mutation_table["Ref"] = mutation_table["Ref"].str.upper()
        mutation_table["Alt"] = mutation_table["Alt"].str.upper()

        self.mutation_table = mutation_table
        self.use_genotype = use_genotype
        return

    def mutate_dna(self, idx: int, start: int, dna_seq: str):
        """
        Mutate the DNA sequence at the given position.
        """
        chrom, pos, ref, alt = self.mutation_table.iloc[idx]
        rel_pos = pos - start

        dna_seq = list(dna_seq)
        # if dna_seq[rel_pos] != ref:
        #     print(
        #         f"Warning: Mutation reference sequence does not match the DNA sequence at the given position: "
        #         f"mutataion:{chrom}_{pos}_{ref}_{alt}, dna_seq@rel_pos:{dna_seq[rel_pos]}, rel_pos:{rel_pos}"
        #     )
        if self.use_genotype == "ref":
            dna_seq[rel_pos] = ref
        else:
            dna_seq[rel_pos] = alt
        return "".join(dna_seq)


class Motifs:
    """
    A class for motif matching based on MOODS

    Parameters
    ----------
    ref_path_motif : str | Path
        Path to the motif file, in JASPAR format
    ref_path_fa : str | Path
        Path to the reference genome fasta file. An eazy way would be passing, the genome.fetch_fa() function
    bg : Literal['even', 'genome'] | tuple, optional
        Background nucleotide frequency, by default 'even' stands for equal frequency of A, C, G, T
        'genome' stands for the frequency of A, C, G, T in the reference genome
        When passing a tuple of size 4, they would be used as background frequency
    pseudocount : float, optional
        Pseudocount for the motif, by default 0.8 (same as motifmatchR)
    pvalue : float, optional
        P-value threshold for motif matching, by default 5e-5
    n_jobs : int, optional
        Number of cores to use, by default 32
    split_tf: bool, optional
        Whether to split the TFs by comma (for cistarget motif collection), by default True
    mode: Literal['motifmatchr', 'moods'], optional
        The mode of the motif matching, by default 'motifmatch
    """

    def __init__(
        self,
        ref_path_motif: str | Path,
        ref_path_fa: str | Path,
        bg: Literal["even", "genome"] | tuple = "even",
        pseudocount: float = 0.8,
        pvalue: float = 5e-5,
        n_jobs: int = 32,
        split_tf: bool = True,
        mode: Literal["motifmatchr", "moods"] = "motifmatchr",
        motif_name_func=None,
    ):
        """
        Motifs class for motif matching based on MOODS

        Parameters
        ----------
        ref_path_motif: str | Path
            Path to the motif file, in JASPAR format
        ref_path_fa: str | Path
            Path to the reference genome fasta file.
        bg: Literal['even', 'genome'] | tuple, optional
            Background nucleotide frequency, by default 'even' stands for equal frequency of A, C, G, T
            'genome' stands for the frequency of A, C, G, T in the reference genome
        pseudocount: float, optional
            Pseudocount for the motif, by default 0.8 (same as motifmatchR)
        pvalue: float, optional
            P-value threshold for motif matching, by default 5e-5
        n_jobs: int, optional
            Number of cores to use, by default 32
        split_tf: bool, optional
            Whether to split the TFs by comma (for cistarget motif collection), by default True
        mode: Literal['motifmatchr', 'moods'], optional
            The mode of the motif matching, by default 'motifmatchr
        motif_name_func: callable, optional
            A function to convert motif names to the desired format
        """
        self.n_jobs = n_jobs
        self.mode = mode
        self.all_motifs = parse_jaspar(ref_path_motif)  # now it's a list of PFM objects
        self.names = [
            set(motif.name.split(",")) if split_tf else {motif.name}
            for motif in self.all_motifs
        ]  # split the TFs by comma (for cistarget motif collection)

        # If there is a motif_name_func, use it to convert motif names to the desired format (remove \t, _ etc.)
        self.names = (
            [{motif_name_func(m) for m in motif} for motif in self.names]
            if motif_name_func is not None
            else self.names
        )
        if motif_name_func is not None:
            for m in self.all_motifs:
                m.name = motif_name_func(m.name)
        # a non-duplicated set of motif names by TF

        self.tfs = set().union(*self.names)
        self.genome_seq = Fasta(ref_path_fa)
        if bg == "even":
            self.bg = [0.25, 0.25, 0.25, 0.25]
        elif bg == "genome":
            b = ""
            for chrom in self.genome_seq.keys():
                b += str(self.genome_seq[chrom])

            self.bg = MOODS.tools.bg_from_sequence_dna(str(b), 1)
        else:
            self.bg = bg
        (
            self.pre_matrices_p,
            self.pre_thresholds_p,
            self.pre_matrices_m,
            self.pre_thresholds_m,
        ) = self._prepare_moods_settings(self.all_motifs, self.bg, pseudocount, pvalue)
        self.pseudocount = pseudocount

        self.pvalue = pvalue
        return

    def prep_scanner(
        self,
        tf_genes: list[str] | None = None,
        pseudocount: float = 0.8,
        pvalue: float = 5e-5,
        window: int = 7,
    ):
        """
        Prepare the MOODS scanner for motif matching

        Parameters
        ----------
        tf_genes: list[str] | None
            List of TFs to be used for motif matching, by default None, which means all TFs. Note that if you pass motif_name_func and/or split_tf,
            you should specify TFs by their names after being processed by those (so likely the true TF names).
        pseudocount: float
            Pseudocount for the motif, by default 0.8 (same as motifmatchR)
        pvalue: float
            P-value threshold for motif matching, by default 5e-5
        window: int
            Window size for motif matching, by default 7 (same as motifmatchR), passed to MOODS

        Returns
        -------
        scanner: MOODS.scan.Scanner

        """
        global scanner
        if tf_genes is None:
            tf_genes = self.tfs
        tf_genes_set = set(tf_genes)
        select = (
            slice(None)
            if tf_genes is None
            else [len(name & tf_genes_set) > 0 for name in self.names]
        )
        self.select = select

        if pseudocount != self.pseudocount or pvalue != self.pvalue:
            # motif = self.all_motifs[select]
            motif = [
                motif
                for motif, keep in zip(self.all_motifs, select, strict=False)
                if keep
            ]
            # Each TF gets a matrix for the + and for the - strand, and a corresponding threshold
            matrices_p, threshold_p, matrices_m, threshold_m = (
                self._prepare_moods_settings(motif, self.bg, pseudocount, pvalue)
            )
        else:
            matrices_p, threshold_p, matrices_m, threshold_m = (
                self.pre_matrices_p[select],
                self.pre_thresholds_p[select],
                self.pre_matrices_m[select],
                self.pre_thresholds_m[select],
            )

        matrices = np.concatenate([matrices_p, matrices_m])
        thresholds = np.concatenate([threshold_p, threshold_m])

        scanner_ = MOODS.scan.Scanner(window)  # parameter is the window size
        scanner_.set_motifs(matrices, self.bg, thresholds)
        self.scanner = scanner_
        self.tfs = set(tf_genes)
        return

    def scan(
        self,
        region_bed: pd.DataFrame,
        verbose: bool = True,
        mutation_table: pd.DataFrame = None,
        use_genotype: Literal["ref", "alt"] = "ref",
    ):
        """
        Perform motif scanning on the given peaks and store the results in the AnnData object.

        Parameters
        ----------
        region_bed : pd.DataFrame
            The region bed file containing the region information.
        verbose : bool, optional
            Whether to display a progress bar during the motif scanning process. Default is True.
        mutation_table : pd.DataFrame, optional
            Table containing mutation information. Position is 1-based.
            Columns: ["Chromosome", "Position", "Ref", "Alt"]
            Currently only supports SNV mutations. Mutation table index is the same as the region bed index.
            If provided, the DNA sequence will be mutated in the SNVs before motif matching.
        use_genotype: Literal["ref", "alt"], optional
            Whether to use the reference or alternative genotype for the mutation. Default is "ref".

        Returns
        -------
        pd.DataFrame
            A dataframe containing the motif match information for each region.
            The index is the region names, the columns are the motif names.
        """
        assert "Name" in region_bed.columns, "Region bed file must have a 'Name' column"
        region_names = region_bed["Name"].values
        peaks = region_bed.iloc[:, :3].values
        res = self.scan_motif(
            peaks,
            verbose=verbose,
            mutation_table=mutation_table,
            use_genotype=use_genotype,
        )
        res_dfs = {}
        dtypes = {
            "hit": bool,
            "score": "float32",
            "strand": "int8",
            "motif_start": "uint32",
            "motif_end": "uint32",
        }
        for (key, _), values in res.items():
            values = np.array(values).astype(dtypes[key])
            res_dfs[key] = pd.DataFrame(
                values, index=region_names, columns=[m.name for m in self.all_motifs]
            )
        return res_dfs

    def collect_child_process(
        self,
        p_list,
        verbose,
        bar,
        motifs_length,
        name2id,
        maps,
        break_on_min_jobs,
    ):
        """
        Collect completed processes from the process list and process their motif mapping results.

        Parameters
        ----------
        p_list : list
            A list of Process objects representing the child processes.
        verbose : bool
            A flag indicating whether to display a progress bar.
        bar : tqdm.tqdm
            A progress bar object for displaying the progress.
        motifs_length : list
            A list containing the lengths of the motifs.
        name2id : dict
            A dictionary mapping motif names to their indices.
        maps : dict
            A dict to store the results of the motif scanning process.
        break_on_min_jobs : bool
            A flag indicating whether to break the loop when the number of remaining processes is less than or equal to the number of workers.

        Returns
        -------
        None
            The function modifies the `maps` list in-place by adding the results of the completed processes.
        """
        for p in as_completed(p_list):
            if verbose:
                bar.update(1)
            all_hits = p.result()
            if len(all_hits) > 0:
                idx = all_hits[0][3]
                for key in maps.keys():
                    value_name, value_idx = key
                    _values = np.zeros((1, len(motifs_length)))
                    for hit in all_hits:
                        loc = name2id[hit[4]]
                        _values[0, loc] = 1 if value_name == "hit" else hit[value_idx]
                    maps[key][idx] = _values
            p_list.remove(p)
            del p

            if break_on_min_jobs and len(p_list) <= self.n_jobs:
                break

    def scan_motif(
        self,
        peaks_iter,
        clean: bool = False,
        verbose: bool = False,
        split_tfs: bool = True,
        strand: bool = True,
        mutation_table: pd.DataFrame = None,
        use_genotype: Literal["ref", "alt"] = "ref",
    ):
        """
        Scan motifs in the peaks iterator.

        Parameters
        ----------
        peaks_iter : iterable
            An iterable yielding a list [chrom, start, end], e.g., a pandas dataframe with three columns.
            This function currently only supports a specific format of regions.

        clean : bool, optional
            Whether to clean the output. If True, overlapping motif hits of the same TF will be merged into one hit.
            Default is False.

        verbose : bool, optional
            Whether to display a progress bar. Default is False.

        split_tfs : bool, optional
            Whether to split the output by transcription factors. Default is True.

        strand : bool, optional
            Whether to consider the strand information. Default is True.

        mutation_table: pd.DataFrame = None,
            Table containing mutation information. Position is 1-based.
            Columns: ["Chromosome", "Position", "Ref", "Alt"]
            Currently only supports SNV mutations.
            If provided, the DNA sequence will be mutated in the SNVs before motif matching.
        use_genotype: Literal["ref", "alt"] = "ref",
            Whether to use the reference or alternative genotype for the mutation. Default is "ref".

        Returns
        -------
        list or numpy.ndarray
            The output of the motif scanning process. The format depends on the values of the `clean`, `split_tfs`, and `strand` parameters.
        """
        if mutation_table is not None:
            mutation_manager = MutationManager(
                mutation_table, use_genotype=use_genotype
            )
        else:
            mutation_manager = None

        global scanner
        scanner = self.scanner

        keys_and_indices = [
            ("hit", 4),
            ("score", 5),
            ("strand", 6),
            ("motif_start", 7),
            ("motif_end", 8),
        ]
        maps = {k: [[] for _ in range(len(peaks_iter))] for k in keys_and_indices}
        motif = [
            motif
            for motif, keep in zip(self.all_motifs, self.select, strict=False)
            if keep
        ]
        motifs_length = [m.length for m in motif]
        motifs_name = [m.name for m in motif]
        name2id = {name: i for i, name in enumerate(motifs_name)}

        peaks_iter = np.array(peaks_iter)
        bar = None
        if verbose:
            bar = trange(len(peaks_iter) * 2)

        pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        p_list = []
        for peak_idx, peak in enumerate(peaks_iter):
            chr = peak[0]
            start = int(peak[1])
            end = int(peak[2])
            seq = self.genome_seq[chr][start:end].seq.upper()
            if mutation_manager is not None:
                seq = mutation_manager.mutate_dna(
                    idx=peak_idx, start=start, dna_seq=seq
                )

            if len(peaks_iter) == 1:
                # When we only need to scan once
                return scan_func(
                    seq,
                    motifs_length,
                    motifs_name,
                    chr,
                    start,
                    end,
                    peak_idx,
                    self.tfs,
                    clean,
                    split_tfs,
                    strand,
                )

            p_list.append(
                pool.submit(
                    scan_func,
                    seq,
                    motifs_length,
                    motifs_name,
                    chr,
                    start,
                    end,
                    peak_idx,
                    self.tfs,
                    clean,
                    split_tfs,
                    strand,
                )
            )

            if len(p_list) >= (self.n_jobs * 10):
                self.collect_child_process(
                    p_list,
                    verbose,
                    bar,
                    motifs_length,
                    name2id,
                    maps,
                    break_on_min_jobs=True,
                )

            if verbose:
                bar.update(1)

        self.collect_child_process(
            p_list,
            verbose,
            bar,
            motifs_length,
            name2id,
            maps,
            break_on_min_jobs=False,
        )
        for k in maps.keys():
            for i in range(len(maps[k])):
                if len(maps[k][i]) == 0:
                    maps[k][i] = np.zeros((1, len(motifs_length)))
        pool.shutdown(wait=True)

        if verbose:
            bar.close()

        res = {k: list(itertools.chain.from_iterable(v)) for k, v in maps.items()}
        return res

    def _prepare_moods_settings(self, jaspar_motifs, bg, pseduocount, pvalue=1e-5):
        """Find hits of list of jaspar_motifs in pyfasta object fasta, using the background distribution bg and
        pseudocount, significant to the give pvalue
        """
        pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        matrices_p = list(
            pool.map(
                partial(
                    _jaspar_to_moods_matrix,
                    bg=bg,
                    pseudocount=pseduocount,
                    mode=self.mode,
                ),
                jaspar_motifs,
                chunksize=100,
            )
        )
        matrices_m = list(
            pool.map(MOODS.tools.reverse_complement, matrices_p, chunksize=100)
        )

        thresholds_p = list(
            pool.map(
                partial(_thresholds_from_p, bg=bg, pvalue=pvalue),
                matrices_p,
                chunksize=100,
            )
        )
        thresholds_m = (
            thresholds_p
            if self.mode == "motifmatchr"
            else list(
                pool.map(
                    partial(_thresholds_from_p, bg=bg, pvalue=pvalue),
                    matrices_m,
                    chunksize=100,
                )
            )
        )
        pool.shutdown(wait=True)

        return (
            np.array(matrices_p, dtype=object),
            np.array(thresholds_p, dtype=object),
            np.array(matrices_m, dtype=object),
            np.array(thresholds_m, dtype=object),
        )
