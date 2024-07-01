from copy import deepcopy

from bolero.utils import validate_config


class GenericDataset:
    """
    Generic dataset class.
    """

    default_config: dict = {}

    _dataset_mode: str = None

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Get the default configuration for the dataset.

        Returns
        -------
            dict: The default configuration.
        """
        return deepcopy(cls.default_config)

    @classmethod
    def create_from_config(cls, config):
        """Create the dataset from a configuration dictionary."""
        config = {k: v for k, v in config.items() if k in cls.default_config}
        validate_config(config, cls.default_config)
        print(f"Create dataset with config: {config}")
        return cls(**config)

    @property
    def dataset_mode(self) -> str:
        """
        Get the dataset mode.

        Returns
        -------
            str: The dataset mode.
        """
        if self._dataset_mode is None:
            raise ValueError("Dataset mode not set.")
        return self._dataset_mode

    def train(self) -> None:
        """
        Set the dataset mode to "train".

        Returns
        -------
        None
        """
        self._dataset_mode = "train"
        return

    def eval(self) -> None:
        """
        Set the dataset mode to "eval".

        Returns
        -------
        None
        """
        self._dataset_mode = "eval"
        return

    def get_processed_dataset(self, chroms, region_bed_path) -> None:
        """
        Get the processed dataset with many oprators applied.

        Parameters
        ----------
            chroms (list): The list of chromosomes.
            region_bed_path (str): The path to the region bed file.

        Returns
        -------
            None
        """
        raise NotImplementedError

    def get_dataloader(self, chroms, region_bed_path, n_batches):
        """
        Get the dataloader for the dataset.

        Parameters
        ----------
            chroms (list): The list of chromosomes.
            region_bed_path (str): The path to the region bed file.
            n_batches (int): The number of batches.

        Returns
        -------
            DataLoader: The dataloader.
        """
        raise NotImplementedError
