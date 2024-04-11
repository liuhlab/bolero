"""
Transform classes for ray.data.Dataset objects.

Each transform is a function that dynamically creates a transform function for manipulating row or batches in a ray.data.Dataset object.

Row* classes are for row-wise transformations, aim to be used in ray.data.Dataset.map() method.
Batch* classes are for batch-wise transformations, aim to be used in ray.data.Dataset.map_batches() method.
These transform classes take a data dictionary and returns a modified data dictionary.

Flat* classes are for flat transformations, which will create new rows form the original row, aim to be used in ray.data.Dataset.flat_map() method.
These transform classes take a data dictionary and returns a list of modified data dictionaries.
"""

from typing import Union

import numpy as np

from bolero.tl.footprint.footprint import FootPrintModel


class BatchCropRegions:
    """Crop regions from the input data batch."""

    def __init__(
        self, key: Union[str, list[str]], final_length: int, max_jitter: int = 0
    ):
        """
        Crop regions from the input data batch.

        Args:
            key (Union[str, list[str]]): The key(s) of the data to be cropped.
            final_length (int): The desired length of the cropped regions.
            max_jitter (int, optional): The maximum amount of jitter to apply to the cropping position.
                Defaults to 0.
        """
        if isinstance(key, str):
            key = [key]
        self.key = key
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
        _batch_size, _length = data[self.key].shape[:2]

        if self.max_jitter > 0:
            _max = self.max_jitter * 2
            jitter = np.array(
                [np.random.default_rng().integers(_max) for _ in range(_batch_size)]
            )
        else:
            jitter = np.zeros(_batch_size)

        for k in self.key:
            _cropped = data[k]
            # second dim is region base pair
            _length = _cropped.shape[1]
            max_range = self.final_length + self.max_jitter * 2
            max_radius = max_range // 2
            left = _length // 2 - max_radius + jitter
            right = left + self.final_length

            assert left >= 0, f"left={left} must be >= 0"
            assert right <= _length, f"right={right} must be <= {_length}"

            _cropped = _cropped[:, left:right].copy()
            data[k] = _cropped
        return data


class BatchReverseComplement:
    """Reverse complements DNA sequences and signals in a batch."""

    def __init__(self, dna_key: str, signal_key: Union[str, list[str]]):
        """
        Reverses and complements DNA sequences and signals in a batch.

        Args:
            dna_key (str): The key to access the DNA sequence in the data dictionary.
            signal_key (str or List[str]): The key(s) to access the signal(s) in the data dictionary.
                If a single string is provided, it will be converted to a list.

        """
        self.dna_key = dna_key

        if isinstance(signal_key, str):
            signal_key = [signal_key]
        self.signal_key = signal_key

    def __call__(self, data: dict) -> dict:
        """
        Reverse complements the DNA sequence and reverses the signal(s) in the data dictionary.

        Args:
            data (dict): The input data dictionary.

        Returns
        -------
            dict: The modified data dictionary with the DNA sequence and signal(s) reversed and complemented.

        """
        if np.random.default_rng().random() > 0.5:
            # reverse complement DNA
            # second dim is region base pair, third dim is one hot encoding
            data[self.dna_key] = data[self.dna_key].flip(dims=(1, 2))

            # reverse signal
            # second dim is region base pair
            for k in self.signal_key:
                data[k] = np.flip(data[k], axis=1)
        return data


class BatchFootprint(FootPrintModel):
    """Apply footprint transformation to the given data batch."""

    def __init__(
        self,
        atac_key: Union[str, list[str]],
        bias_key: str,
        modes: np.ndarray = None,
        clip_min: float = -10,
        clip_max: float = 10,
    ):
        """
        Apply footprint transformation to the given data dictionary.

        Args:
            atac_key (Union[str, List[str]]): Key(s) for the ATAC data in the data dictionary.
            bias_key (str): Key for the bias data in the data dictionary.
            modes (np.ndarray): Modes for the footprint transformation.
            clip_min (float, optional): Minimum value for clipping. Defaults to -10.
            clip_max (float, optional): Maximum value for clipping. Defaults to 10.
        """
        if modes is None:
            modes = np.arange(2, 101, 1)
        else:
            modes = np.array(modes)
        super().__init__(bias_bw_path=None, dispmodels=None, modes=modes, device=None)
        if isinstance(atac_key, str):
            atac_key = [atac_key]
        self.atac_key = atac_key
        self.bias_key = bias_key
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data: dict) -> dict:
        """
        Apply the footprint transformation to the given data.

        Args:
            data (dict): Input data dictionary.

        Returns
        -------
            dict: Transformed data dictionary.
        """
        bias_data = data[self.bias_key]
        for atac in self.atac_key:
            atac_data = data[atac]
            fp = self.footprint_from_data(
                atac_data=atac_data,
                bias_data=bias_data,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
            data[f"{atac}_footprint"] = fp
        return data


# class FlatRowMutagenesis:
#     def __init__():
#         pass

#     def __call__(self, data):
#         pass
