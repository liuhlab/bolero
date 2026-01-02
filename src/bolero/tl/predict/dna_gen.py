import numpy as np
import pandas as pd
import pyfaidx
import torch
from einops import rearrange
from tangermeme.ersatz import dinucleotide_shuffle, shuffle
from tangermeme.utils import characters, random_one_hot

from bolero.pp.seq import one_hot_encoding_torch


class _GenomeSequenceHandler:
    """Wrapper class to provide a unified interface for FASTA and 2bit files."""

    def __init__(self, handler, file_type):
        self.handler = handler
        self.file_type = file_type

    @classmethod
    def from_file(cls, file_path: str):
        """
        Create a handler for FASTA or 2bit files, abstracting away file format details.

        This class method automatically detects the file format based on the file extension
        and returns a handler with a unified interface.

        Parameters
        ----------
        file_path: str
            Path to the FASTA (.fa, .fasta, .fa.gz, .fasta.gz) or 2bit (.2bit) file

        Returns
        -------
        _GenomeSequenceHandler
            Handler object with a unified interface for accessing sequences

        Raises
        ------
        ImportError
            If py2bit is not installed when trying to open a 2bit file
        ValueError
            If the file format is not supported
        """
        file_path_lower = str(file_path).lower()

        # Check if it's a 2bit file
        if file_path_lower.endswith(".2bit"):
            try:
                import py2bit
            except ImportError:
                raise ImportError(
                    "py2bit is required to open 2bit files. "
                    "Install it with: pip install py2bit"
                ) from None
            handler = py2bit.open(file_path)
            return cls(handler, "2bit")

        # Check if it's a FASTA file (supports .fa, .fasta, and their gzipped versions)
        elif (
            file_path_lower.endswith(".fa")
            or file_path_lower.endswith(".fasta")
            or file_path_lower.endswith(".fa.gz")
            or file_path_lower.endswith(".fasta.gz")
        ):
            handler = pyfaidx.Fasta(file_path)
            return cls(handler, "fasta")

        else:
            raise ValueError(
                f"Unsupported file format for {file_path}. "
                "Supported formats: FASTA (.fa, .fasta, .fa.gz, .fasta.gz) and 2bit (.2bit)"
            )

    def get_sequence(self, chrom: str, start: int, end: int) -> str:
        """
        Get DNA sequence from a genomic region.

        Parameters
        ----------
        chrom: str
            Chromosome name
        start: int
            Start position (0-based)
        end: int
            End position (0-based, exclusive)

        Returns
        -------
        str
            DNA sequence in uppercase
        """
        if self.file_type == "fasta":
            return self.handler[chrom][start:end].seq.upper()
        elif self.file_type == "2bit":
            return self.handler.sequence(chrom, start, end).upper()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")


class DNASynthesisFactory:
    def __init__(
        self,
        genome_fastas: dict[str, str] | None = None,
        mode: str = "genome",
        fix_seq_length_mode: str = "center",
    ):
        """
        DNA Synthesis Factory to synthesize DNA sequences

        Within the factory, DNA sequence is always represented as a torch.Tensor of shape (n_regions, 4, seq_len).
        Channel order is always ACGT.

        All coordinates should be 0-based, including variant positions.

        Step 1: Get background DNA sequence from a region in fasta/2bit file or from random generated sequences.
          - a. Get sequence from fasta or 2bit file
          - b. Get random sequence
        Step 2: Perform different kinds of DNA sequence modifications.
        Step 3: Return/yield the synthesized DNA sequence in different options.

        Parameters
        ----------
        genome_fastas: dict[str, str]
            Dictionary of genome names and their file paths.
            Supports FASTA files (.fa, .fasta, .fa.gz, .fasta.gz) and 2bit files (.2bit).
            For 2bit files, py2bit package must be installed.

        Returns
        -------
        None
        """
        self.genome_fastas = genome_fastas or {}
        self._fasta_handlers = {}
        self.default_genome = (
            None if len(self.genome_fastas) == 0 else list(self.genome_fastas.keys())[0]
        )
        self.default_mode = mode
        assert self.default_mode in ["genome", "mutation"], f"Invalid mode: {mode}"
        self.fix_seq_length_mode = fix_seq_length_mode
        assert self.fix_seq_length_mode in [
            "center",
            "left",
            "right",
        ], f"Invalid fix_seq_length_mode: {fix_seq_length_mode}"

        for genome, file_path in self.genome_fastas.items():
            self._fasta_handlers[genome] = _GenomeSequenceHandler.from_file(file_path)

    def get_fasta_region_sequence(
        self, chrom: str, start: int, end: int, genome: str = None
    ) -> str:
        """
        Get the DNA sequence from the fasta or 2bit file for a given region.
        """
        genome = genome or self.default_genome
        assert genome is not None, "No genome provided and no default genome set"

        handler = self._fasta_handlers[genome]
        sequence = handler.get_sequence(chrom, start, end)
        sequence = one_hot_encoding_torch(sequence, batch_dim=True)
        return sequence

    def get_random_sequence(
        self,
        n_regions: int,
        seq_len: int,
        probs: tuple[float, float, float, float] = (0.29, 0.21, 0.21, 0.29),
        random_state=None,
    ) -> torch.Tensor:
        """
        Get random DNA sequence from the factory.

        Parameters
        ----------
        n_regions: int
            Number of regions to generate.
        seq_len: int
            Length of the sequence.
        probs: tuple[float, float, float, float]
            Probabilities of the four bases.
        """
        probs = np.array(probs)
        one_hot = random_one_hot(
            shape=(n_regions, 4, seq_len), probs=probs, random_state=random_state
        )
        return one_hot

    def substitute_sequence(
        self, input_one_hot: torch.Tensor, pos_0base: int, seq: str
    ) -> torch.Tensor:
        """
        Substitute the DNA sequence at the given position.
        """
        seq_one_hot = one_hot_encoding_torch(seq, batch_dim=True)
        seq_len = seq_one_hot.shape[-1]
        input_one_hot[..., pos_0base : pos_0base + seq_len] = seq_one_hot
        return input_one_hot

    def _validate_ref(
        self, input_one_hot: torch.Tensor, pos_0base: int, ref: str
    ) -> bool:
        """
        Validate the reference sequence at the given position.
        """
        seq_one_hot = one_hot_encoding_torch(ref, batch_dim=True)
        seq_len = seq_one_hot.shape[-1]
        if not np.allclose(
            input_one_hot[..., pos_0base : pos_0base + seq_len], seq_one_hot
        ):
            return False
        return True

    def _fix_seq_length(
        self, input_one_hot: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """
        Fix the sequence length change caused by mutation.
        """
        mode = self.fix_seq_length_mode

        cur_len = input_one_hot.shape[-1]
        if cur_len == target_length:
            return input_one_hot
        elif cur_len > target_length:
            delta = cur_len - target_length
            if mode == "center":
                ldelta = delta // 2
                rdelta = delta - ldelta
                return input_one_hot[..., ldelta:-rdelta]
            elif mode == "left":
                return input_one_hot[..., delta:]
            elif mode == "right":
                return input_one_hot[..., :-delta]
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            delta = target_length - cur_len
            rand_one_hot = self.get_random_sequence(1, delta)
            if mode == "center":
                ldelta = delta // 2
                rdelta = delta - ldelta
                return torch.cat(
                    [
                        rand_one_hot[..., :ldelta],
                        input_one_hot,
                        rand_one_hot[..., -rdelta:],
                    ],
                    dim=-1,
                )
            elif mode == "left":
                return torch.cat([rand_one_hot, input_one_hot], dim=-1)
            elif mode == "right":
                return torch.cat([input_one_hot, rand_one_hot], dim=-1)
            else:
                raise ValueError(f"Invalid mode: {mode}")

    def mutate_single_sequence(
        self,
        input_one_hot: torch.Tensor,
        pos_0base: int,
        ref: str,
        alt: str,
        validate_ref: bool = True,
    ) -> torch.Tensor:
        """
        Mutate the DNA sequence at the given position.

        Parameters
        ----------
        input_one_hot: torch.Tensor
            Input DNA sequence one-hot encoding.
        pos_0base: int
            Position to mutate. Position is 0-based.
        ref: str
            Reference sequence.
        alt: str
            Alternative sequence.
        validate_ref: bool
            Whether to validate the reference sequence at the given position.
        """
        if validate_ref:
            if not self._validate_ref(input_one_hot, pos_0base, ref):
                raise ValueError(
                    f"Reference sequence at position {pos_0base} does not match the input one-hot encoding."
                )

        n_ref = len(ref)
        n_alt = len(alt)
        if n_ref == n_alt:
            return self.substitute_sequence(input_one_hot, pos_0base, alt)

        input_len = input_one_hot.shape[-1]
        alt = one_hot_encoding_torch(alt, batch_dim=True)
        before = input_one_hot[..., :pos_0base]
        after = input_one_hot[..., pos_0base + n_ref :]
        mutated = torch.cat([before, alt, after], dim=-1)
        mutated = self._fix_seq_length(mutated, input_len)
        return mutated

    def shuffle_sequence(
        self,
        input_one_hot: torch.Tensor,
        start: int,
        end: int,
        n_shuffle: int = 1,
        random_state: int = None,
        use_dinucleotide_shuffle: bool = False,
    ) -> torch.Tensor:
        """
        Shuffle the DNA sequence at the given position.

        input_one_hot shape should be (n_regions, 4, seq_len).
        output shape should be (n_regions * n_shuffle, 4, seq_len).

        Parameters
        ----------
        input_one_hot: torch.Tensor
            Input DNA sequence one-hot encoding.
        start: int
            Start position of the sequence.
        end: int
            End position of the sequence.
        n_shuffle: int
            Number of shuffle to perform.
        random_state: int
            Random state.
        use_dinucleotide_shuffle: bool
            Whether to perform dinucleotide shuffle (True) or simple shuffle (False).

        Returns
        -------
        torch.Tensor
            Shuffled sequence batch. Shape: (n_regions * n_shuffle, 4, seq_len).
        """
        if use_dinucleotide_shuffle:
            X_shuf = dinucleotide_shuffle(
                input_one_hot,
                start=start,
                end=end,
                n=n_shuffle,
                random_state=random_state,
            )
        else:
            X_shuf = shuffle(
                input_one_hot,
                start=start,
                end=end,
                n=n_shuffle,
                random_state=random_state,
            )
        # Tangermeme returns shape (n_regions, n_shuffle, 4, seq_len), we reshape to (n_regions * n_shuffle, 4, seq_len)
        # each region is repeated n_shuffle times with shuffled sequences.
        X_shuf = rearrange(X_shuf, "n s c l -> (n s) c l")
        return X_shuf

    def _check_genome(self, table: pd.DataFrame, genome: str = None) -> pd.DataFrame:
        """
        Check the genome of the table.
        """
        table = table.copy()
        if "genome" in table.columns:
            return table
        if genome is None:
            genome = self.default_genome
        assert genome is not None, "No genome provided and no default genome set"
        table["genome"] = genome
        return table

    def decode(self, one_hot: torch.Tensor) -> list[str]:
        """
        Decode the one-hot encoding to a DNA sequence.
        """
        return [characters(one_hot[i]) for i in range(one_hot.shape[0])]

    def genome_mutation_task(
        self,
        region_and_mutation: pd.DataFrame,
        genome: str = None,
        validate_ref: bool = False,
        decode: bool = False,
    ) -> torch.Tensor | list[str]:
        """
        Generate sequence batch using a table of region and mutation information.

        Parameters
        ----------
        region_and_mutation: pd.DataFrame
            Table of region and mutation information.
            Columns: chromosome, start, end, mut_pos_0base, ref, alt, genome (optional if each region has a different genome)
        genome: str
            Genome name. If not provided, the default genome will be used.
        validate_ref: bool
            Whether to validate the reference sequence at the given position.

        Returns
        -------
        torch.Tensor
            Sequence batch. Shape: (n_regions, 4, seq_len).
        """
        region_and_mutation = self._check_genome(region_and_mutation, genome)

        sequences = []
        for _, row in region_and_mutation.iterrows():
            chromosome, start, end, mut_pos_0base, ref, alt, genome = row
            seq = self.get_fasta_region_sequence(chromosome, start, end, genome)
            rel_pos = mut_pos_0base - start
            seq = self.mutate_single_sequence(
                seq, rel_pos, ref, alt, validate_ref=validate_ref
            )
            sequences.append(seq)
        sequences = torch.cat(sequences, dim=0)
        if decode:
            sequences = self.decode(sequences)
        return sequences

    def genome_task(
        self, region: pd.DataFrame, genome: str = None, decode: bool = False
    ) -> torch.Tensor | list[str]:
        """
        Generate sequence batch using a table of region information.

        Parameters
        ----------
        region: pd.DataFrame
            Table of region information.
        genome: str
            Genome name. If not provided, the default genome will be used.
        decode: bool
            Whether to decode the sequence to a DNA sequence.

        Returns
        -------
        torch.Tensor or list[str]
            Sequence batch. Shape: (n_regions, 4, seq_len). If decode is True, return list[str] of DNA sequences.
        """
        region = self._check_genome(region, genome)
        sequences = []
        for _, row in region.iterrows():
            chromosome, start, end, genome = row
            seq = self.get_fasta_region_sequence(chromosome, start, end, genome)
            sequences.append(seq)
        sequences = torch.cat(sequences, dim=0)
        if decode:
            sequences = self.decode(sequences)
        return sequences

    def region_string_to_dataframe(
        self, regions: list[str], region_names: list[str], mode: str = None
    ):
        """
        Convert a list of regions and region names to a dataframe.

        Parameters
        ----------
        regions: list[str]
            List of regions.
        region_names: list[str]
            List of region names.
            Region names should be in the format "region_name__additional_info".
            "additional_info" can be empty, or a string of additional information with fields separated by ":".
        mode: str
            Mode of the task.
            If "genome", the dataframe will have columns: chromosome, start, end, and optionally genome.
            If "mutation", the dataframe will have columns: chromosome, start, end,
            mut_pos_0base, ref, alt, and optionally genome.
        """
        mode = mode or self.default_mode
        assert mode is not None, "Mode is not provided and no default mode set"
        assert (
            len(regions) == len(region_names)
        ), f"Regions and region names must have the same length: {len(regions)} != {len(region_names)}"

        records = []
        for idx in range(len(regions)):
            region = regions[idx]
            chrom, coords = region.split(":")
            start, end = map(int, coords.split("-"))
            name = region_names[idx]
            name, *mut_info = name.split("__")
            if len(mut_info) == 1:
                mut_info = mut_info[0].split(":")
            else:
                mut_info = []
            records.append([chrom, start, end, *mut_info])

        # all regions should have the same length
        assert (
            len({len(rec) for rec in records}) == 1
        ), "All regions should have the same length"
        n_cols = len(records[0])
        if mode == "genome":
            _columns = ["chromosome", "start", "end"]
            if n_cols == 3:
                columns = _columns
            elif n_cols == 4:
                columns = _columns + ["genome"]
            else:
                raise ValueError(f"Invalid number of columns: {n_cols} for genome mode")
        elif mode == "mutation":
            _columns = ["chromosome", "start", "end", "mut_pos_0base", "ref", "alt"]
            if n_cols == 6:
                columns = _columns
            elif n_cols == 7:
                columns = _columns + ["genome"]
            else:
                raise ValueError(
                    f"Invalid number of columns: {n_cols} for mutation mode"
                )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        df = pd.DataFrame(records, columns=columns)
        if "mut_pos_0base" in df.columns:
            df["mut_pos_0base"] = df["mut_pos_0base"].astype(int)
        return df

    def get_regions_onehot(
        self, regions: list[str], region_names: list[str]
    ) -> torch.Tensor:
        """
        Get the one-hot encoding for the regions.
        """
        df = self.region_string_to_dataframe(regions, region_names)
        if self.default_mode == "mutation":
            onehot = self.genome_mutation_task(df, decode=False)
        elif self.default_mode == "genome":
            onehot = self.genome_task(df, decode=False)
        else:
            raise ValueError(f"Invalid mode: {self.default_mode}")
        return onehot
