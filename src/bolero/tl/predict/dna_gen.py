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


def random_mutations(
    seq_one_hot: torch.Tensor,
    n_mut_pos: int,
    start: int | None = None,
    end: int | None = None,
) -> torch.Tensor:
    """
    Create random mutations on the given DNA one-hot encoding.
    Each mutation is guaranteed to change the base (no silent same-base "mutations").

    Parameters
    ----------
    seq_one_hot : torch.Tensor
        Input DNA sequence one-hot encoding. Shape: (4, seq_len).
    n_mut_pos : int
        Number of mutation positions to create.
    start : int, optional
        Start (inclusive) of the window in which to mutate. Default 0.
    end : int, optional
        End (exclusive) of the window in which to mutate. Default seq_len.

    Returns
    -------
    torch.Tensor
        Cloned sequence with same shape (4, seq_len) and random mutations at
        n_mut_pos (or fewer if window is smaller) random positions within
        [start, end).
    """
    out = seq_one_hot.clone()
    _, seq_len = seq_one_hot.shape
    start = 0 if start is None else start
    end = seq_len if end is None else end
    window_len = end - start
    n_mut_pos = min(max(0, n_mut_pos), window_len)
    if n_mut_pos == 0:
        return out

    indices_in_window = torch.randperm(window_len, device=seq_one_hot.device)[
        :n_mut_pos
    ]
    mut_positions = start + indices_in_window
    current_bases = seq_one_hot.argmax(dim=0)[mut_positions]
    offset = torch.randint(1, 4, (n_mut_pos,), device=seq_one_hot.device)
    new_bases = (current_bases + offset) % 4

    out[:, mut_positions] = 0
    out[new_bases, mut_positions] = 1
    return out


def back_mutate_sequence(
    mutated_one_hot: torch.Tensor,
    reference_one_hot: torch.Tensor,
    back_mutation_rate: float = 0.2,
    min_diff_pos: int = 3,
) -> torch.Tensor:
    """
    Examine position differences between mutated and reference one-hot encodings.
    At each differing position, with probability back_mutation_rate, revert the
    mutated base to the reference base.

    Parameters
    ----------
    mutated_one_hot : torch.Tensor
        Mutated one-hot encoding. Shape: (4, seq_len).
    reference_one_hot : torch.Tensor
        Reference one-hot encoding. Shape: (4, seq_len).
    back_mutation_rate : float
        Probability of reverting a differing position to the reference base.

    Returns
    -------
    torch.Tensor
        The same tensor as mutated_one_hot, modified in-place. Shape: (4, seq_len).
    """
    differs = (mutated_one_hot != reference_one_hot).any(dim=0)
    diff_indices = differs.nonzero(as_tuple=True)[0]
    if diff_indices.numel() < min_diff_pos:
        return mutated_one_hot
    n_diff = diff_indices.numel()
    do_revert = torch.rand(n_diff, device=mutated_one_hot.device) < back_mutation_rate
    for i, pos in enumerate(diff_indices):
        if do_revert[i]:
            mutated_one_hot[:, pos] = reference_one_hot[:, pos]
    return mutated_one_hot


class DNAEvolutionFactory:
    """
    Beam-search DNA evolution: maintain a pool of parent sequences and produce
    mutated batches for scoring, then update the pool with the top-scoring
    sequences.

    Workflow
    --------
    1. Initialize with initial one-hot sequence(s) and evolution parameters.
    2. Call :meth:`get_regions_onehot` to get a batch of mutated sequences
       (batch_size, 4, seq_len) for the predictor to score.
    3. Select the top sequences by your criterion and call
       :meth:`update_current_one_hot` with that tensor.
    4. Repeat from step 2 for the next evolution round.

    The number of mutations per sequence is derived from the evolution window
    length and mutation_rate. Mutations are applied only within the evolution
    window [evolution_window_start, evolution_window_end).

    Parameters
    ----------
    input_one_hot : torch.Tensor
        Initial DNA one-hot encoding. Shape ``(4, seq_len)`` or
        ``(batch, 4, seq_len)``. If 2D, treated as a single parent and
        converted to ``(1, 4, seq_len)``.
    evolution_window_start : int
        Start (inclusive) of the evolution window. Used only to compute
        the number of mutations per sequence.
    evolution_window_end : int
        End (exclusive) of the evolution window. Must be > evolution_window_start.
    mutation_rate : float, optional
        Fraction of the window length to mutate per sequence, in (0, 1).
        Number of mutations = max(1, mutation_rate * (eend - estart)).
        Default is 0.01.
    back_mutation_rate : float, optional
        Probability of reverting a differing position to the reference base.
        Default is 0.5.

    Attributes
    ----------
    seq_len : int
        Sequence length.
    n_mutations : int
        Number of random mutation positions applied per sequence per round.
    """

    def __init__(
        self,
        input_sequence: str,
        evolution_window_start: int,
        evolution_window_end: int,
        mutation_rate: float = 0.01,
        device: str = "cuda",
        back_mutation_rate: float = 0.2,
        _min_diff_pos_when_back_mutate: int = 3,
        **kwargs,
    ):
        if isinstance(input_sequence, str):
            input_one_hot = one_hot_encoding_torch(input_sequence, batch_dim=True)
        else:
            input_one_hot = input_sequence
        input_one_hot = input_one_hot.to(device)

        if input_one_hot.ndim == 2:
            input_one_hot = input_one_hot.unsqueeze(0)
        assert input_one_hot.ndim == 3, (
            f"Input must be 2D (4, seq_len) or 3D (batch, 4, seq_len), "
            f"got {input_one_hot.ndim}D"
        )
        assert (
            input_one_hot.shape[1] == 4
        ), f"One-hot must have 4 channels (ACGT), got {input_one_hot.shape[1]}"

        self.seq_len = int(input_one_hot.shape[2])
        self.estart = int(evolution_window_start)
        self.eend = int(evolution_window_end)
        assert self.eend > self.estart, (
            f"evolution_window_end must be > evolution_window_start, "
            f"got {self.eend} <= {self.estart}"
        )
        self.mutation_rate = float(mutation_rate)
        self.back_mutation_rate = float(back_mutation_rate)
        self._min_diff_pos_when_back_mutate = int(_min_diff_pos_when_back_mutate)
        assert (
            0 < self.mutation_rate < 1
        ), f"mutation_rate must be in (0, 1), got {self.mutation_rate}"
        window_len = self.eend - self.estart
        self.n_mutations = max(1, int(self.mutation_rate * window_len))

        self.input_one_hot = input_one_hot.clone()
        self._current_one_hot = input_one_hot.clone()
        self._mutation_cache = set()
        return

    def get_regions_onehot(
        self, batch_size: int = 32, add_ref: bool = True
    ) -> torch.Tensor:
        """
        Generate a batch of mutated sequences from the current parent pool.

        Each of the batch_size sequences is produced by taking a parent from
        the pool (cycled by index) and applying n_mutations random mutations.
        Parents are selected in round-robin when batch_size > pool size.

        Parameters
        ----------
        batch_size: int
            Number of mutated sequences to return per call.
        add_ref: bool
            Whether to add the current sequence to the batch.

        Returns
        -------
        new_batch: torch.Tensor
            One-hot batch of shape (batch_size, 4, seq_len), same device/dtype
            as the current pool.
        """
        device = self._current_one_hot.device
        n_parents = self._current_one_hot.shape[0]
        new_batch = []
        # first, add current sequence to new batch,
        # to allow the model to score the current sequence
        # and able to discard the mutation if there is no improvement
        if add_ref:
            for _cur_seq in self._current_one_hot:
                new_batch.append(_cur_seq)

        # second, add additional mutated sequences to new batch
        i = 0
        while len(new_batch) < batch_size:
            # create random mutations
            parent_idx = i % n_parents
            parent = self._current_one_hot[parent_idx]
            mutated = random_mutations(
                parent, self.n_mutations, start=self.estart, end=self.eend
            )

            # random back mutation
            input_idx = (
                i % self.input_one_hot.shape[0]
            )  # always be 0 if there is only one input sequence
            mutated = back_mutate_sequence(
                mutated_one_hot=mutated,
                reference_one_hot=self.input_one_hot[input_idx],
                back_mutation_rate=self.back_mutation_rate,
                min_diff_pos=self._min_diff_pos_when_back_mutate,
            )

            # check if the mutation is already in the cache
            mutated_seq = characters(mutated)
            if mutated_seq in self._mutation_cache:
                continue
            self._mutation_cache.add(mutated_seq)
            new_batch.append(mutated)
            i += 1
        new_batch = torch.stack(new_batch, dim=0)
        return new_batch.to(device)

    def get_evolution_sequence(self, new_batch: torch.Tensor) -> list[str]:
        """
        Get the DNA sequence at the evolution window positions for the given batch.

        Parameters
        ----------
        new_batch: torch.Tensor
            One-hot batch of shape (batch_size, 4, seq_len), same device/dtype
            as the current pool.

        Returns
        -------
        current_evolution_sequence: list[str]
            List of DNA sequences at the evolution window positions for the given batch.
            Shape: (batch_size,).
        """
        return [
            characters(mutated[:, self.estart : self.eend]) for mutated in new_batch
        ]

    def update_current_one_hot(self, one_hot: torch.Tensor) -> None:
        """
        Set the current parent pool to the given sequences (e.g. top-k from
        the last batch after scoring).

        Parameters
        ----------
        one_hot : torch.Tensor
            One-hot sequences to use as parents for the next round. Shape
            (n, 4, seq_len) with n typically equal to n_top_regions. seq_len
            must match self.seq_len.
        """
        if one_hot.ndim == 2:
            one_hot = one_hot.unsqueeze(0)
        assert (
            one_hot.ndim == 3
            and one_hot.shape[1] == 4
            and one_hot.shape[2] == self.seq_len
        ), f"one_hot must have shape (n, 4, {self.seq_len}), got {one_hot.shape}"
        self._current_one_hot = one_hot.clone()
        return

    def reset_current_one_hot(self) -> None:
        """
        Reset the current one-hot to the input one-hot.
        """
        self._current_one_hot = self.input_one_hot.clone()
        self._mutation_cache = set()
        print("Reset current one-hot and mutation cache")
        return
