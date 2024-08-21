import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


class ScanpyGeneScaler:
    def __init__(self, clip=3):
        self.scaler = StandardScaler()
        self.clip = clip
        self.fitted = False

    def fit(self, X):
        """
        Fit the scaler using the given gene adata.
        """
        if self.fitted:
            raise ValueError("This ScanpyGeneScaler has already been fitted.")

        X = self.scaler.fit_transform(X)
        self.fitted = True
        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the given gene data using the fitted scaler.
        """
        X = np.clip(self.scaler.transform(X), -self.clip, self.clip)
        return X


class IdentityScaler:
    def __init__(self):
        self.fitted = True

    def fit(self, *args, **kwargs):
        """Do nothing."""
        return

    def transform(self, X):
        """Do nothing."""
        return X


class CombinedScaler:
    """
    A scaler for gene data.
    It first applies a robust scaler to clip the data to a certain range,
    then applies a standard scaler to normalize the data.
    """

    def __init__(self):
        self.scaler1 = RobustScaler(quantile_range=(5, 95))
        self.scaler2 = StandardScaler()
        self.fitted = False

    def fit(self, X):
        """
        Fit the scaler using the given gene adata.
        """
        if self.fitted:
            raise ValueError("This CombinedScaler has already been fitted.")

        X = np.clip(self.scaler1.fit_transform(X), -1, 1)
        X = self.scaler2.fit_transform(X.reshape((-1, 1))).reshape(X.shape)
        self.fitted = True
        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the given gene data using the fitted scaler.
        """
        X = np.clip(self.scaler1.transform(X), -1, 1)
        X = self.scaler2.transform(X.reshape((-1, 1))).reshape(X.shape)
        return X


class GeneDataScaler:
    def __init__(self, pca=True, n_components=512, scale_pc=True, scale_gene=True):
        self.fitted = False
        self._signature = (pca, n_components, scale_pc, scale_gene)

        self.input_dim = None
        self.output_dim = None
        self.gene_order: pd.Index = None

        self.gene_scaler = ScanpyGeneScaler() if scale_gene else IdentityScaler()
        self.pca = PCA(n_components=n_components) if pca else IdentityScaler()
        # only scale pc if pca is enabled and scale_pc is True
        self.pc_scaler = CombinedScaler() if scale_pc and pca else IdentityScaler()

    def validate(self, pca, n_components, scale_pc, scale_gene, gene_index):
        """Validate if the given parameters match the fitted parameters."""
        same_sig = self._signature == (pca, n_components, scale_pc, scale_gene)
        if gene_index is None:
            same_gene = True
        else:
            same_gene = self.gene_order.equals(gene_index)
        good = self.fitted and same_sig and same_gene
        return good

    def fit(self, adata):
        """
        Fit the PCA and scaler using the given gene adata.
        """
        if self.fitted:
            raise ValueError("This GeneDataScaler has already been fitted.")

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        # save metadata for later checks
        self.input_dim = X.shape[1]
        self.output_dim = (
            self.pca.n_components if isinstance(self.pca, PCA) else self.input_dim
        )
        self.gene_order = adata.var_names.copy()

        self.gene_scaler.fit(X)
        X = self.gene_scaler.transform(X)

        self.pca.fit(X)
        X = self.pca.transform(X)

        self.pc_scaler.fit(X)
        self.fitted = True
        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the given gene data using the fitted PCA and scaler.
        """
        if not self.fitted:
            raise ValueError("This GeneDataScaler has not been fitted yet.")

        X = self.gene_scaler.transform(X)
        X = self.pca.transform(X)
        X = self.pc_scaler.transform(X)
        return X

    def dump(self, path):
        """Save the scaler to a file."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load the scaler from a file."""
        scaler = joblib.load(path)
        assert isinstance(scaler, cls)
        return scaler
