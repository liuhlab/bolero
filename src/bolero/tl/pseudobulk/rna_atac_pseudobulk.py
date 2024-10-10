import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from bolero.utils import validate_config

from .generator import EmbeddingScaler


# vq_records schema:
# vq_records[vq] = {
#     "atac_cluster_ids": cluster_ids, np.array
#     "vq_ind": vq_ind, np.array
#     "vq_emb": vq_emb, np.array
#     "n_frags": n_frags, int
#     "cov_scale": n_frags_log2fc, float, log2 fold change of psedobulk coverage to target coverage
# }
class RNAVQPseudobulker:
    default_config = {
        "vq_records": "REQUIRED",
        "target_cov": 8e6,
        "use_vq_emb": True,
        "prefix_name": "pseudobulk",
        "downsample_vq": None,
    }

    @classmethod
    def create_from_config(cls, **config):
        """Create the pseudobulk generator from configuration."""
        config = {k: v for k, v in config.items() if k in cls.default_config}
        validate_config(config, cls.default_config)
        pseudobulker = cls(**config)

        pseudobulker.prepare_scaler()
        return pseudobulker

    def __init__(
        self,
        vq_records,
        target_cov=8e6,
        use_vq_emb=True,
        prefix_name="pseudobulk",
        downsample_vq=None,
    ):
        """
        Load VQ records and prepare pseudobulks for RNA data.
        """
        self.target_cov = target_cov
        self.use_vq_emb = use_vq_emb

        emb_key = "vq_emb" if use_vq_emb else "vq_ind"
        cov_key = "n_frags"

        if isinstance(vq_records, str):
            vq_records = joblib.load(vq_records)

        if downsample_vq is not None:
            use_vq_id = np.random.choice(
                list(vq_records.keys()), downsample_vq, replace=False
            )
            vq_records = {k: v for k, v in vq_records.items() if k in use_vq_id}
            print(f"Downsampled to {len(vq_records)} VQs")

        self.predefined_pseudobulks = {}
        self.pseudobulk_ids = pd.Index(vq_records.keys())
        self.n_pids = self.pseudobulk_ids.size
        for idx, (vq, data) in enumerate(vq_records.items()):
            rows = data["cluster_ids"]
            prefix_to_rows = {prefix_name: data["cluster_ids"]}
            emb_data = data[emb_key]
            cov_value = np.array([np.log2(data[cov_key] / target_cov)])
            # last item of emb_data is cov_value
            emb_data = np.concatenate([emb_data, cov_value])
            self.predefined_pseudobulks[vq] = [rows, prefix_to_rows, emb_data, idx]

        self.vq_emb_dims = emb_data.size - 1

        self.random_pid = np.random.choice(
            self.pseudobulk_ids, self.n_pids, replace=False
        )

        self.scaler = EmbeddingScaler()
        return

    def _scale_without_vq_ind(self, emb_data):
        if self.use_vq_emb:
            emb_data = self.scaler.transform(emb_data)
        else:
            # using vq_ind only
            # only transform the non-ind part
            vq_ind = emb_data[: self.vq_emb_dims]
            emb_data = emb_data[self.vq_emb_dims :]
            emb_data = self.scaler.transform(emb_data)
            emb_data = np.concatenate([vq_ind, emb_data])
        return emb_data

    def take_by_name(self, name):
        """Take a VQ pseudobulk by name."""
        data = self.predefined_pseudobulks[name]

        # emb_data is data[2]
        data[2] = self._scale_without_vq_ind(data[2])
        return data

    def take(self, n):
        """Take n pseudobulks from the random pool."""
        while self.random_pid.size < n:
            _random_pid = np.random.choice(
                self.pseudobulk_ids, self.n_pids, replace=False
            )
            self.random_pid = np.concatenate([self.random_pid, _random_pid])
        use_pids = self.random_pid[:n]
        self.random_pid = self.random_pid[n:].copy()

        pseudobulks = [self.predefined_pseudobulks[pid] for pid in use_pids]

        # transform embeddings
        for data in pseudobulks:
            data[2] = self._scale_without_vq_ind(data[2])

        # each item in pseudobulks is
        # [
        #     rows: np.array,
        #     prefix_to_rows: dict[str, np.array],
        #     emb_data: np.array,
        #     idx: int
        # ]
        return pseudobulks

    def prepare_scaler(self):
        """
        Fit the scaler using predefined pseudobulks.
        """
        rows = []
        vqs = []
        for vq, data in self.predefined_pseudobulks.items():
            vqs.append(vq)
            embedding = data[2]
            if not self.use_vq_emb:
                # if not using vq_emb, we need to remove the vq_emb part which is just vq_ind
                embedding = embedding[self.vq_emb_dims :]
            rows.append(embedding)

        example_embedding = pd.DataFrame(rows, index=vqs)
        self.scaler.fit(example_embedding)
        return

    def save_scaler(self, path):
        """
        Save the scaler to path.

        Parameters
        ----------
        path (str): The path to save the scaler.
        """
        if not self.scaler.fitted:
            raise NotFittedError("Scaler is not fitted yet.")

        joblib.dump(self.scaler, path)
        return
