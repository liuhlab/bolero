import numpy as np
from sklearn.neighbors import NearestNeighbors


class VQKNN:
    def __init__(self, vq):
        """Wrapper for a VQ model that allows for fast nearest neighbor queries."""
        self.vq = vq.to("cpu")
        self.emb_vectors = vq._codebook.embed.detach().cpu().numpy()

        if self.emb_vectors.ndim == 2:
            # add the head dimension
            self.emb_vectors = self.emb_vectors[None, :, :]
        self.n_heads, self.n_codes, self.n_dim = self.emb_vectors.shape

        # each head is considered indipendently, just like multihead vq
        self.nn = []
        self.dist = []
        for head_emb in self.emb_vectors:
            _nn = NearestNeighbors(
                n_neighbors=head_emb.shape[0], algorithm="auto", metric="euclidean"
            )
            _nn.fit(head_emb)
            dist, nn = _nn.kneighbors(head_emb)
            self.nn.append(nn)
            self.dist.append(dist)
        self.nn = np.array(self.nn)
        self.dist = np.array(self.dist)

    def kneighbors(self, ind, k=5) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a batch of indices, return the k nearest neighbors for each index

        Parameters
        ----------
        ind : np.ndarray
            The indices to query, shape (batch_size, n_heads)
        k : int
            The number of neighbors to return

        Returns
        -------
        dist, nn : np.ndarray
            The distances and indices of the nearest neighbors, shape (batch_size, n_heads, k)
            k=0 is the index itself
        """
        assert 0 < k < self.n_codes
        if ind.ndim == 1:
            ind = ind[:, None]
        assert ind.shape[1] == self.n_heads

        head_idx = np.arange(self.n_heads)[None, :]
        nn = self.nn[head_idx, ind, :k]
        dist = self.dist[head_idx, ind, :k]
        return dist, nn
