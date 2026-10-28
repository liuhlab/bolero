"""The predictor's background batch loader, without a GPU, Ray or data on disk."""

import threading

import numpy as np
import pytest
import torch

from bolero.tl.predict.datamanager import BackgroundGenerator, GenericGenomeDataManager

# A regression makes the consumer block forever; fail the test instead.
JOIN_TIMEOUT = 10


def _consume(loader):
    """Iterate the loader in a helper thread and return what it produced.

    Returns the list of batches and the exception that stopped iteration (or
    None). Fails the test if iteration does not finish within JOIN_TIMEOUT.
    """
    result = {"batches": [], "error": None}

    def run():
        try:
            for batch in loader:
                result["batches"].append(batch)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(JOIN_TIMEOUT)
    if thread.is_alive():
        pytest.fail(f"loader still blocked after {JOIN_TIMEOUT} s")
    return result["batches"], result["error"]


def test_all_batches_arrive_then_iteration_stops():
    def gen():
        for i in range(3):
            yield {"x": np.array([i])}

    batches, error = _consume(BackgroundGenerator(gen(), as_torch=False))

    assert error is None
    assert [b["x"].item() for b in batches] == [0, 1, 2]


def test_generator_error_reaches_consumer_after_earlier_batches():
    def gen():
        yield {"x": np.array([0])}
        yield {"x": np.array([1])}
        raise FileNotFoundError("missing pseudobulk file")

    batches, error = _consume(BackgroundGenerator(gen(), as_torch=False))

    assert [b["x"].item() for b in batches] == [0, 1]
    assert isinstance(error, FileNotFoundError)
    assert str(error) == "missing pseudobulk file"


class _StubOneHotEncoder:
    """Returns all-zero one-hot of shape (n_regions, seq_len, 4)."""

    def get_regions_onehot(self, regions):
        return torch.zeros(len(regions), 8, 4)


def test_dataloader_without_region_names(monkeypatch):
    monkeypatch.setattr(
        GenericGenomeDataManager,
        "_create_one_hot_encoder",
        lambda self: _StubOneHotEncoder(),
    )
    manager = GenericGenomeDataManager(genome=object(), device="cpu")
    regions = ["chr1:0-8", "chr1:8-16", "chr1:16-24"]

    loader = manager.get_dataloader(
        regions=regions, batch_size=2, add_data=False, as_torch=False
    )
    batches, error = _consume(loader)

    assert error is None
    assert [list(b["region"]) for b in batches] == [regions[:2], regions[2:]]
    assert all("region_name" not in b for b in batches)
    assert [tuple(b["dna"].shape) for b in batches] == [(2, 4, 8), (1, 4, 8)]
