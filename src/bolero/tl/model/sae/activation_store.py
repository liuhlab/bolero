"""
This code is adapted from the following repository:
https://github.com/bartbussmann/BatchTopK/tree/main

Citation:
https://arxiv.org/abs/2412.06410
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


class ActivationsStore:
    def __init__(
        self,
        activation_iter,
        batch_size: int = 4096,
        num_batches_in_buffer: int = 100,
    ):
        """
        Store activations of a model on a dataset and provide a way to sample

        Args:
        activation_iter: iter
            Iterator of the activations of the model on the dataset
            Each iteration should return a tensor of shape (n_token, act_dim)
            n_token may not be constant across iterations, but act_dim should be
        batch_size: int
            Batch size to use for sampling activations from the buffer
        num_batches_in_buffer: int
            Number of batches to store in the buffer
        """
        self.activation_iter = activation_iter
        self.num_batches_in_buffer = num_batches_in_buffer
        self.batch_size = batch_size

    def get_activations(self):
        """Get activations of the model on the next batch of the dataset"""
        with torch.no_grad():
            batch = next(self.activation_iter)
        return batch

    def _fill_buffer(self):
        all_activations = []
        cum_count = 0
        total_buffer = self.num_batches_in_buffer * self.batch_size
        while cum_count < total_buffer:
            activations = self.get_activations()
            all_activations.append(activations)
            cum_count += activations.size(0)
        all_activations = torch.cat(all_activations, dim=0)
        all_activations = torch.relu(all_activations)
        return all_activations

    def _get_dataloader(self):
        return DataLoader(
            TensorDataset(self.activation_buffer),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def next_batch(self):
        """Sample a batch of activations from the buffer"""
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            torch.cuda.empty_cache()
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]
