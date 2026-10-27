"""DatasetRecordManager as a predictor uses it: without the datasets' data folders."""

import joblib
import numpy as np
import pandas as pd

from bolero.tl.model.borzoi.dataset_multi import DatasetRecordManager


def _write_pseudobulk_file(path):
    """Write a minimal ensemble-pseudobulk records file.

    It has 2 pseudobulks over 3 meta cells, a 4-dim meta-cell embedding, and two
    condition terms: Age (continuous, width 1) and Treatment (2 categories).
    """
    meta_cells = ["mc0", "mc1", "mc2"]
    info = {
        "pseudobulk_records": {
            pid: {
                "embedding": np.zeros(8, dtype="float32"),
                "cluster_ids": {"prefix": ["mc0", "mc1"]},
                "n_frags": {"prefix": 100},
                "cov_scale": {"prefix": 0.0},
            }
            for pid in ["pb0", "pb1"]
        },
        "meta_cell_emb": pd.DataFrame(np.zeros((3, 4)), index=meta_cells),
        "meta_cell_n_frags": pd.Series(100, index=meta_cells),
        "target_cov": 5_000_000,
        "condition_to_related_pseudobulk": {"Ctrl": ["pb0"], "Stim": ["pb1"]},
        "condition_emb": {"Age": "CONTINUOUS", "Treatment": ["Ctrl", "Stim"]},
    }
    joblib.dump(info, path)
    return str(path)


def test_inference_records_load_without_data_folders(tmp_path):
    # The data_path folders (metacell parquet datasets) and the shared score
    # file do not exist, as for a user who downloaded only the model.
    # Genome names are not real UCSC assemblies: building a Genome would fail.
    dataset_records = {
        "DatasetA": {
            "pseudobulk_path": _write_pseudobulk_file(tmp_path / "a.joblib"),
            "data_path": str(tmp_path / "absent/DatasetA-MetaCell-32bp"),
            "genome": "genomeA",
        },
        "DatasetB": {
            "pseudobulk_path": _write_pseudobulk_file(tmp_path / "b.joblib"),
            "data_path": str(tmp_path / "absent/DatasetB-MetaCell-32bp"),
            "genome": "genomeB",
        },
    }

    manager = DatasetRecordManager(
        dataset_records=dataset_records,
        pseudobulker_cls="ensemble",
        shared_data_paths=str(tmp_path / "absent/AP1.chromvar_score.joblib.gz"),
        inference_only=True,
        shared_data_dim=1,
    )

    assert manager.make_cond_module_kwargs() == {
        "dataset_order": ["DatasetA", "DatasetB"],
        "dataset_specific_dims": {
            "DatasetA": {"Age": 1, "Treatment": 2, "cell": 8},
            "DatasetB": {"Age": 1, "Treatment": 2, "cell": 8},
        },
        "dataset_shared_dims": {"__genome__": 2, "__shared_data__": 1},
    }
    np.testing.assert_array_equal(manager.dataset_idx_to_genome_emb[0], [[1, 0]])
    np.testing.assert_array_equal(manager.dataset_idx_to_genome_emb[1], [[0, 1]])
