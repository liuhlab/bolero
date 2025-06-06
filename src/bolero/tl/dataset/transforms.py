"""
Transform classes for ray.data.Dataset objects.

Each transform is a function that dynamically creates a transform function for manipulating row or batches in a ray.data.Dataset object.

If perform row-wise transformations, used ray.data.Dataset.map() method.
If perform batch-wise transformations, use ray.data.Dataset.map_batches() method.
These transform classes take a data dictionary and returns a modified data dictionary.

If perform flat transformations (from one row to many rows), use ray.data.Dataset.flat_map() method.
These transform classes take a data dictionary and returns a list of modified data dictionaries.
"""

from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd

from bolero.pp.genome import FastaOneHotNoParallel
from bolero.tl.topic.region_embedding import RegionEmbedder


class CropRegionsWithJitter:
    """Crop regions from the input data batch."""

    def __init__(
        self,
        key: Union[str, list[str]],
        final_length: int,
        max_jitter: int = 0,
        crop_axis=0,
    ):
        """
        Crop regions from the input data batch.

        Args:
            key (Union[str, list[str]]): The key(s) of the data to be cropped.
            final_length (int): The desired length of the cropped regions.
            max_jitter (int, optional): The maximum amount of jitter to apply to the cropping position.
                Defaults to 0.
            crop_axis (int, optional): The axis to crop the regions. Defaults to 0.
        """
        if isinstance(key, str):
            key = [key]
        self.key = key
        if isinstance(final_length, int):
            final_length = [final_length] * len(key)
        else:
            assert len(final_length) == len(
                key
            ), "final_length must have the same length as key"
        self.final_length = final_length
        self.max_jitter = max_jitter
        self.crop_axis = crop_axis

    def __call__(self, data: dict) -> dict:
        """
        Crop regions from the input data batch.

        Args:
            data (dict): The input data batch.

        Returns
        -------
            dict: The cropped data batch.
        """
        if self.max_jitter > 0:
            jitter = (
                np.random.default_rng().integers(self.max_jitter * 2) - self.max_jitter
            )
        else:
            jitter = 0

        for k, length in zip(self.key, self.final_length):
            _input = data.pop(k)

            _input_length = _input.shape[self.crop_axis]
            _input_center = _input_length // 2
            _output_radius = length // 2
            _start = _input_center - _output_radius + jitter
            _end = _start + length
            sel = slice(_start, _end)
            idx = tuple(
                sel if i == self.crop_axis else slice(None) for i in range(_input.ndim)
            )
            data[k] = _input[idx]

        # data["jitter"] = np.array([jitter])
        return data


class CropLastAxisWithJitter:
    """Crop regions from the input data batch."""

    def __init__(
        self,
        key: Union[str, list[str]],
        final_length: int,
        max_jitter: int = 0,
    ):
        """
        Crop regions from the input data batch.

        Args:
            key (Union[str, list[str]]): The key(s) of the data to be cropped.
            final_length (int): The desired length of the cropped regions.
            max_jitter (int, optional): The maximum amount of jitter to apply to the cropping position.
                Defaults to 0.
            crop_axis (int, optional): The axis to crop the regions. Defaults to 0.
        """
        if isinstance(key, str):
            key = [key]
        self.key = key
        if isinstance(final_length, int):
            final_length = [final_length] * len(key)
        else:
            assert len(final_length) == len(
                key
            ), "final_length must have the same length as key"
        self.final_length = final_length
        self.max_jitter = max_jitter

    def __call__(self, data: dict) -> dict:
        """
        Crop regions from the input data batch.

        Args:
            data (dict): The input data batch.

        Returns
        -------
            dict: The cropped data batch.
        """
        if self.max_jitter > 0:
            jitter = (
                np.random.default_rng().integers(self.max_jitter * 2) - self.max_jitter
            )
        else:
            jitter = 0

        for k, length in zip(self.key, self.final_length):
            _input = data.pop(k)

            _input_length = _input.shape[-1]
            _input_center = _input_length // 2
            _output_radius = length // 2
            _start = _input_center - _output_radius + jitter
            _end = _start + length
            data[k] = _input[..., _start:_end].copy()

        data["jitter"] = np.full((data[self.key[-1]].shape[0], 1), jitter).astype(
            "float32"
        )
        return data


class ReverseComplement:
    """Reverse complements DNA sequences and signals in a batch."""

    def __init__(
        self,
        dna_key: Union[str, list[str]],
        signal_key: Union[str, list[str]],
        prob=0.5,
    ):
        """
        Reverses and complements DNA sequences and signals in a batch.

        Args:
            dna_key (str): The key to access the DNA sequence in the data dictionary.
            signal_key (str or List[str]): The key(s) to access the signal(s) in the data dictionary.
                If a single string is provided, it will be converted to a list.
            prob (float, optional): The probability of applying the transformation. Defaults to 0.5.
        """
        if isinstance(dna_key, str):
            dna_key = [dna_key]
        self.dna_key = dna_key

        if isinstance(signal_key, str):
            signal_key = [signal_key]
        self.signal_key = signal_key

        self.prob = prob

        self.flip_dna_axis = (-1, -2)
        self.flip_signal_axis = -1
        return

    def __call__(self, data: dict) -> dict:
        """
        Reverse complements the DNA sequence and reverses the signal(s) in the data dictionary.

        Args:
            data (dict): The input data dictionary.

        Returns
        -------
            dict: The modified data dictionary with the DNA sequence and signal(s) reversed and complemented.

        """
        try:
            bs = data[self.dna_key[0]].shape[0]
            if np.random.default_rng().random() > self.prob:
                # reverse complement DNA
                for k in self.dna_key:
                    data[k] = np.flip(data[k], axis=self.flip_dna_axis)

                # reverse signal
                for k in self.signal_key:
                    data[k] = np.flip(data[k], axis=self.flip_signal_axis)
                data["is_reverse_comp"] = np.ones(bs, dtype=np.int32)
            else:
                data["is_reverse_comp"] = np.zeros(bs, dtype=np.int32)

        except np.exceptions.AxisError as e:
            print("Error in ReverseComplement, the data causing the error is:")
            for k, v in data.items():
                print(k, v.shape)
            raise e
        return data


class ReverseComplmentMinusStrand:
    def __init__(
        self, dna_key, signal_key, strand_key="Strand", pos_strand=1, neg_strand=0
    ):
        if isinstance(dna_key, str):
            dna_key = [dna_key]
        if isinstance(signal_key, str):
            signal_key = [signal_key]
        self.dna_key = dna_key
        self.signal_key = signal_key
        self.strand_key = strand_key
        self.pos_strand = pos_strand
        self.neg_strand = neg_strand

        self.flip_dna_axis = (-1, -2)
        self.flip_signal_axis = -1
        return

    def __call__(self, data: dict) -> dict:
        """Reverse complement when strand is negative."""
        new_data = defaultdict(list)

        bs = data[self.dna_key[0]].shape[0]
        other_key = [k for k in data if k not in (self.dna_key + self.signal_key)]
        for i in range(bs):
            strand = data[self.strand_key][i]
            if strand == self.neg_strand:
                for k in self.dna_key:
                    new_data[k].append(np.flip(data[k][i], axis=self.flip_dna_axis))
                for k in self.signal_key:
                    new_data[k].append(np.flip(data[k][i], axis=self.flip_signal_axis))
                for k in other_key:
                    new_data[k].append(data[k][i])
                new_data["is_reverse_comp"].append(np.ones(1, dtype=np.int32))
            elif strand == self.pos_strand:
                for k in data.keys():
                    new_data[k].append(data[k][i])
                new_data["is_reverse_comp"].append(np.zeros(1, dtype=np.int32))
            else:
                raise ValueError(
                    f"Invalid strand value: {strand}, "
                    f"must be {self.pos_strand} or {self.neg_strand}"
                )
        new_data = {k: np.stack(v) for k, v in new_data.items()}
        return new_data


class BatchRegionEmbedding:
    """Embed the region information in the data dictionary."""

    def __init__(
        self,
        embedding: np.ndarray,
        region_key: str = "region",
    ) -> None:
        """
        Initialize the BatchRegionEmbedding transform.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding array.
        region_key : str, optional
            The key to access the region information in the data dictionary. Defaults to "region".
        pop_region_key : bool, optional
            Whether to remove the region key from the data dictionary after embedding. Defaults to True.
        """
        embedding = embedding.copy().astype(np.float32)
        self.embedder = RegionEmbedder()
        self.embedder.add_predefined_embedding(embedding)
        self.region_key = region_key

    def __call__(self, data_dict: dict) -> dict:
        """
        Apply the BatchRegionEmbedding transform to the input data.

        Parameters
        ----------
        data_dict : dict
            The input data dictionary.

        Returns
        -------
        dict
            The modified data dictionary with the region embedding added.
        """
        regions = data_dict[self.region_key]
        if isinstance(regions, str):
            regions = pd.Index([regions])
        data_dict["region_embedding"] = np.array(
            self.embedder(regions, predefined=True)
        )
        return data_dict


class AddChannels:
    """Add channel dimension to the input data.

    Parameters
    ----------
    key : Union[str, list[str]]
        The key(s) of the data to add channel dimension to.
    channel_func : callable, optional
        The function to add channel dimension. Defaults to None.
    channel_dim : int, optional
        The dimension to add the channel. Defaults to 1.

    Returns
    -------
    dict
        The modified data dictionary with the added channel dimension.

    """

    def __init__(
        self,
        key: Union[str, list[str]],
        channel_func: callable = None,
        channel_dim: int = 1,
    ):
        """
        Add channel dimension to the input data.

        Parameters
        ----------
        key : Union[str, list[str]]
            The key(s) of the data to add channel dimension to.
        channel_func : callable, optional
            The function to add channel dimension from the original data.
            If None, it will add the channel dimension using the unsqueeze(channel_dim).
            Defaults to None.
        channel_dim : int, optional
            The dimension to add the channel. Defaults to 1.

        """
        if isinstance(key, str):
            key = [key]
        self.keys = key

        if channel_func is None:
            channel_func = lambda x: np.expand_dims(x, channel_dim)
        self.channel_func = channel_func

    def __call__(self, data: dict) -> dict:
        """
        Add channel dimension to the input data.
        """
        for k in self.keys:
            data[k] = self.channel_func(data[k])
        return data


class FetchRegionOneHot:
    """Fetch the one-hot encoded DNA sequence from the genome."""

    def __init__(
        self,
        fasta_path: str,
        region_key: str = "region",
        output_key: str = "dna_one_hot",
        dtype: str = "float32",
        random_shift: bool = 0,
    ) -> None:
        """
        Initialize the FetchRegionOneHot transform.

        Parameters
        ----------
        region_key : str, optional
            The key to access the region name in the data dictionary. Defaults to "Name".
        output_key : str, optional
            The key to store the one-hot encoded DNA in the data dictionary. Defaults to "dna_one_hot".
        dtype : str, optional
            The data type of the one-hot encoded DNA. Defaults to "float32".
        random_shift : bool, optional
            Whether to randomly shift the DNA sequence. Defaults to False.
            Borzoi model uses this to randomly shift the DNA sequence. Other models should allways set to 0

        """
        self.region_key = region_key
        self.output_key = output_key
        self.dtype = dtype
        self.random_shift = random_shift
        self.onehot_encoder = FastaOneHotNoParallel(fasta_path=fasta_path)

    def __call__(self, data: dict, key_suffix=None) -> dict:
        """
        Apply the FetchRegionOneHot transform to the input data.

        Parameters
        ----------
        data : dict
            The input data dictionary.
        remote_genome_one_hot : ray.remote
            The remote object to fetch the genome one-hot encoder.
        key_suffix : str, optional
            The suffix to take region and add to the output key. Defaults to None.

        Returns
        -------
        dict
            The modified data dictionary with the one-hot encoded DNA.
            DNA one-hot shape: (batch, channel, length)
        """
        # shape: (batch, length, channel)

        if key_suffix is None:
            key_suffix = [""]

        for suffix in key_suffix:
            regions = data[self.region_key + suffix]

            if self.random_shift > 0:
                new_regions = []
                for region in regions:
                    shift = np.random.randint(-self.random_shift, self.random_shift + 1)
                    chrom, coords = region.split(":")
                    start, end = map(int, coords.split("-"))
                    start += shift
                    end += shift
                    region = f"{chrom}:{start}-{end}"
                    new_regions.append(region)
                regions = new_regions

            one_hot = (
                self.onehot_encoder.get_regions_one_hot(regions)
                .permute(0, 2, 1)
                .numpy()
            )
            # (batch, length, channel)
            data[self.output_key + suffix] = one_hot.astype(self.dtype)
        return data
