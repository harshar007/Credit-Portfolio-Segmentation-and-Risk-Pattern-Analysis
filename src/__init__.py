"""
Credit Portfolio Segmentation and Risk Pattern Analysis Source Package.
"""
from src.data_loader import load_processed_data
from src.preprocessing import prepare_clustering_features, transform_new_applicant
from src.dimensionality_reduction import apply_pca
from src.clustering import fit_kmeans, fit_dbscan
from src.risk_analysis import (
    profile_borrower_segments,
    analyze_anomalies,
    generate_business_recommendations,
    SEGMENT_LABEL_MAPPING,
)
from src.pipeline import run_pipeline

__all__ = [
    "load_processed_data",
    "prepare_clustering_features",
    "transform_new_applicant",
    "apply_pca",
    "fit_kmeans",
    "fit_dbscan",
    "profile_borrower_segments",
    "analyze_anomalies",
    "generate_business_recommendations",
    "SEGMENT_LABEL_MAPPING",
    "run_pipeline",
]
