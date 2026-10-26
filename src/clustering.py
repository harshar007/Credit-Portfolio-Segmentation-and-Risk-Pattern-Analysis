"""
Clustering algorithms (K-Means, DBSCAN) for borrower segmentation.
"""
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, float]:
    """Fits K-Means clustering algorithm and returns model, cluster labels, and inertia."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels, float(km.inertia_)


def fit_dbscan(
    X_scaled: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 25
) -> Tuple[DBSCAN, np.ndarray, Dict[str, Any]]:
    """Fits DBSCAN clustering and calculates basic cluster and noise statistics."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    stats = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / len(labels))
    }
    return db, labels, stats
