"""
Transform classes for ray.data.Dataset objects.

Each transform is a function that dynamically creates a transform function for manipulating row or batches in a ray.data.Dataset object.

If perform row-wise transformations, used ray.data.Dataset.map() method.
If perform batch-wise transformations, use ray.data.Dataset.map_batches() method.
These transform classes take a data dictionary and returns a modified data dictionary.

If perform flat transformations (from one row to many rows), use ray.data.Dataset.flat_map() method.
These transform classes take a data dictionary and returns a list of modified data dictionaries.
"""

from typing import Union

import numpy as np
import pandas as pd
import ray

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

        # data["jitter"] = np.array([jitter])
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
            input_type (str, optional): The input type of the data, choose from 'row' or 'batch'. Defaults to 'row'.
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
            if np.random.default_rng().random() > self.prob:
                # reverse complement DNA
                for k in self.dna_key:
                    data[k] = np.flip(data[k], axis=self.flip_dna_axis)

                # reverse signal
                for k in self.signal_key:
                    data[k] = np.flip(data[k], axis=self.flip_signal_axis)
        except np.exceptions.AxisError as e:
            print("Error in ReverseComplement, the data causing the error is:")
            for k, v in data.items():
                print(k, v.shape)
            raise e
        return data


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
        region_key: str = "region",
        output_key: str = "dna_one_hot",
        dtype: str = "float32",
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

        """
        self.region_key = region_key
        self.output_key = output_key
        self.dtype = dtype

    def __call__(self, data: dict, remote_genome_one_hot) -> dict:
        """
        Apply the FetchRegionOneHot transform to the input data.

        Parameters
        ----------
        data : dict
            The input data dictionary.

        Returns
        -------
        dict
            The modified data dictionary with the one-hot encoded DNA.
        """
        genome_one_hot = ray.get(remote_genome_one_hot)
        # shape: (batch, length, channel)
        one_hot = genome_one_hot.get_regions_one_hot(data[self.region_key])
        # change to (batch, channel, length)
        data[self.output_key] = np.moveaxis(one_hot.astype(self.dtype), -2, -1)
        return data
