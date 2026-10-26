"""
Dimensionality reduction routines (PCA) for portfolio visual projection.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(
    X_scaled: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, PCA, np.ndarray, pd.DataFrame]:
    """Fits PCA and reduces dimensionality to specified number of components."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    comp_df = pd.DataFrame(
        pca.components_,
        columns=[f"feat_{i}" for i in range(X_scaled.shape[1])],
        index=[f"PC{i+1}" for i in range(n_components)]
    )
    return X_pca, pca, var_ratio, comp_df
