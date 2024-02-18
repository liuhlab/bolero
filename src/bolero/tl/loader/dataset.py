import pathlib

import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, random_split


class CTDataset(Dataset):
    def __init__(
        self, dataset_path, features="X_dna_one_hot", binary_labels="y", sample_dim='region', load=True
    ):
        """
        features: Tensor of shape (n_seqs, seq_length, bases)
        cell_type_embedding: Tensor of shape (n_category, 1024)
        cell_type_tokens: Dict with keys input_ids and attention mask, each is Tensor of shape (n_category, max_len)
        binary_labels: Tensor of shape (n_seqs, n_category)
        """
        dataset_path = pathlib.Path(dataset_path).absolute()

        total_ds = []
        for p in dataset_path.glob("*.zarr"):
            _ds = xr.open_zarr(p)
            total_ds.append(_ds)
        total_ds = xr.merge(total_ds)

        if load:
            total_ds.load()

        self._ds = total_ds
        assert features in total_ds, f"{features} not in {total_ds}"
        self.features = features
        assert binary_labels in total_ds, f"{binary_labels} not in {total_ds}"
        self.binary_labels = binary_labels

        self.sample_dim = sample_dim

    def __len__(self):
        return self._ds.sizes[self.sample_dim]
    
    def _isel_region(self, idx):
        return self._ds.isel({self.sample_dim: idx}).load()

    def __getitem__(self, idx):
        # new input
        data = self._isel_region(idx)
        feature_vector = torch.FloatTensor(data[self.features].values)
        label = torch.FloatTensor(data[self.binary_labels].values)
        return feature_vector, label
