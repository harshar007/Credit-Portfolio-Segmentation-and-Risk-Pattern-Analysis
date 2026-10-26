"""
Unit and integration tests for Credit Portfolio Segmentation pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_processed_data
from src.preprocessing import prepare_clustering_features, transform_new_applicant
from src.clustering import fit_kmeans, fit_dbscan
from src.dimensionality_reduction import apply_pca
from src.risk_analysis import (
    profile_borrower_segments,
    analyze_anomalies,
    generate_business_recommendations,
    SEGMENT_LABEL_MAPPING,
)


@pytest.fixture
def sample_data():
    """Provides a small synthetic borrower dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "annual_income": np.random.uniform(30000, 150000, n),
        "dti": np.random.uniform(5, 35, n),
        "loan_amount": np.random.uniform(5000, 35000, n),
        "fico_score": np.random.uniform(620, 800, n),
        "emp_length_years": np.random.uniform(0, 10, n),
        "Default": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    return df


def test_prepare_clustering_features(sample_data):
    X_scaled, features, scaler = prepare_clustering_features(sample_data)
    assert X_scaled.shape == (100, 5)
    assert len(features) == 5
    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-5)
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-5)


def test_fit_kmeans(sample_data):
    X_scaled, _, _ = prepare_clustering_features(sample_data)
    km_model, labels, inertia = fit_kmeans(X_scaled, n_clusters=4, random_state=42)
    assert len(labels) == 100
    assert len(set(labels)) <= 4
    assert inertia > 0


def test_fit_dbscan(sample_data):
    X_scaled, _, _ = prepare_clustering_features(sample_data)
    db_model, labels, stats = fit_dbscan(X_scaled, eps=1.5, min_samples=5)
    assert len(labels) == 100
    assert "n_clusters" in stats
    assert "n_noise" in stats


def test_apply_pca(sample_data):
    X_scaled, _, _ = prepare_clustering_features(sample_data)
    X_pca, pca, var_ratio, comp_df = apply_pca(X_scaled, n_components=2)
    assert X_pca.shape == (100, 2)
    assert len(var_ratio) == 2
    assert comp_df.shape == (2, 5)


def test_profile_borrower_segments(sample_data):
    X_scaled, _, _ = prepare_clustering_features(sample_data)
    _, labels, _ = fit_kmeans(X_scaled, n_clusters=4, random_state=42)
    profile_df = profile_borrower_segments(sample_data, labels)
    assert "Cluster" in profile_df.columns
    assert "default_rate_pct" in profile_df.columns
    assert "relative_risk_ratio" in profile_df.columns
    assert len(profile_df) == len(set(labels))


def test_transform_new_applicant(sample_data):
    X_scaled, features, scaler = prepare_clustering_features(sample_data)
    applicant = {
        "annual_income": 75000.0,
        "dti": 18.5,
        "loan_amount": 15000.0,
        "fico_score": 720.0,
        "emp_length_years": 5.0,
    }
    scaled = transform_new_applicant(applicant, scaler, features)
    assert scaled.shape == (1, 5)


def test_generate_business_recommendations(sample_data):
    X_scaled, _, _ = prepare_clustering_features(sample_data)
    _, labels, _ = fit_kmeans(X_scaled, n_clusters=4, random_state=42)
    profile_df = profile_borrower_segments(sample_data, labels)
    recs = generate_business_recommendations(profile_df)
    assert len(recs) == len(profile_df)
    for rec in recs:
        assert "segment" in rec
        assert "risk_level" in rec
        assert "action" in rec
        assert "strategy" in rec
