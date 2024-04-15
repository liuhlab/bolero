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


class CropRegionsWithJitter:
    """Crop regions from the input data batch."""

    def __init__(
        self,
        key: Union[str, list[str]],
        final_length: int,
        max_jitter: int = 0,
        input_type="row",
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
        if isinstance(final_length, int):
            final_length = [final_length] * len(key)
        else:
            assert len(final_length) == len(
                key
            ), "final_length must have the same length as key"
        self.final_length = final_length
        self.max_jitter = max_jitter

        if input_type == "batch":
            self.crop_axis = 1
        elif input_type == "row":
            self.crop_axis = 0
        else:
            raise ValueError(f"input_type must be 'row' or 'batch', got {input_type}")

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
            jitter = np.random.default_rng().integers(self.max_jitter * 2) - self.max_jitter
        else:
            jitter = 0

        for k, length in zip(self.key, self.final_length):
            _input = data[k]

            _input_length = _input.shape[self.crop_axis]
            _input_center = _input_length // 2
            _output_radius = length // 2
            _start = _input_center - _output_radius + jitter
            _end = _start + length

            if self.crop_axis == 0:
                data[k] = _input[_start:_end]
            else:
                data[k] = _input[:, _start:_end]
        return data


class ReverseComplement:
    """Reverse complements DNA sequences and signals in a batch."""

    def __init__(
        self,
        dna_key: Union[str, list[str]],
        signal_key: Union[str, list[str]],
        input_type="row",
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

        if input_type == "batch":
            self.flip_dna_axis = (1, 2)
            self.flip_signal_axis = 1
        elif input_type == "row":
            self.flip_dna_axis = (0, 1)
            self.flip_signal_axis = 0
        else:
            raise ValueError(f"input_type must be 'row' or 'batch', got {input_type}")
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

        if np.random.default_rng().random() > self.prob:
            # reverse complement DNA
            for k in self.dna_key:
                data[k] = np.flip(data[k], axis=self.flip_dna_axis)

            # reverse signal
            for k in self.signal_key:
                data[k] = np.flip(data[k], axis=self.flip_signal_axis)
        return data


class BatchFootPrint(FootPrintModel):
    """Apply footprint transformation to the given data batch."""

    def __init__(
        self,
        atac_key: Union[str, list[str]],
        bias_key: str,
        modes: np.ndarray = None,
        clip_min: float = -10,
        clip_max: float = 10,
        return_pval: bool = False,
        smooth_radius: int = None,
        numpy=True,
        device=None,
    ):
        """
        Apply footprint transformation to the given data dictionary.

        Args:
            atac_key (Union[str, List[str]]): Key(s) for the ATAC data in the data dictionary.
            bias_key (str): Key for the bias data in the data dictionary.
            modes (np.ndarray): Modes for the footprint transformation.
            clip_min (float, optional): Minimum value for clipping. Defaults to -10.
            clip_max (float, optional): Maximum value for clipping. Defaults to 10.
            return_pval (bool, optional): Whether to return p-values. Defaults to False.
            smooth_radius (int, optional): Radius for smoothing. Defaults to None.
            numpy (bool, optional): Whether to use numpy. Defaults to True.
        """
        if modes is None:
            modes = np.arange(2, 101, 1)
        else:
            modes = np.array(modes)
        super().__init__(bias_bw_path=None, dispmodels=None, modes=modes, device=device)

        # get the device from the parameters
        self.device = next(self.parameters()).device

        if isinstance(atac_key, str):
            atac_key = [atac_key]
        self.atac_key = atac_key
        self.bias_key = bias_key
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.return_pval = return_pval
        self.smooth_radius = smooth_radius
        self.numpy = numpy

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
                return_pval=self.return_pval,
                smooth_radius=self.smooth_radius,
                numpy=self.numpy,
            )
            data[f"{atac}_footprint"] = fp
        return data


class BatchToFloat:
    """
    Convert the specified key(s) in the data dictionary to float type.

    Parameters
    ----------
    key : Union[str, list[str]]
        The key(s) of the data to be converted to float type.

    Returns
    -------
    dict
        The modified data dictionary with the specified key(s) converted to float type.
    """

    def __init__(self, key: Union[str, list[str]]):
        if isinstance(key, str):
            key = [key]
        self.key = key

    def __call__(self, data: dict) -> dict:
        """
        Convert the specified key(s) in the data dictionary to float type.

        Parameters
        ----------
        data : dict
            The input data dictionary.

        Returns
        -------
        dict
            The modified data dictionary with the specified key(s) converted to float type.
        """
        for k in self.key:
            if isinstance(data[k], np.ndarray):
                data[k] = data[k].astype(np.float32)
            else:
                data[k] = data[k].float()
        return data


# class FlatRowMutagenesis:
#     def __init__():
#         pass

#     def __call__(self, data):
#         pass
