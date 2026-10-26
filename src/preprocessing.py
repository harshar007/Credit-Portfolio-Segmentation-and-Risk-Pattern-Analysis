"""
Preprocessing and feature scaling for portfolio clustering.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_clustering_features(
    df: pd.DataFrame,
    features: List[str] = None
) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """Standardizes selected numerical features using StandardScaler."""
    if features is None:
        features = ["annual_income", "dti", "loan_amount", "fico_score", "emp_length_years"]
    
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features, scaler


def transform_new_applicant(
    applicant_data: Dict[str, Any],
    scaler: StandardScaler,
    feature_names: List[str]
) -> np.ndarray:
    """Scales a single applicant's dictionary input using the pre-fitted scaler."""
    row_df = pd.DataFrame([[applicant_data[col] for col in feature_names]], columns=feature_names)
    return scaler.transform(row_df)
