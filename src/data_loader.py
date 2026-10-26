"""
Data loading utilities for Credit Portfolio Segmentation.
"""
from pathlib import Path
import pandas as pd


def load_processed_data(file_path: str) -> pd.DataFrame:
    """Loads processed loan portfolio data from a CSV file."""
    path = Path(file_path)
    if not path.exists():
        # Fallback to local relative path
        alt_path = Path("data/processed/cleaned_loans_data.csv")
        if alt_path.exists():
            path = alt_path
        else:
            alt_path2 = Path("data/processed/cleaned_loans_dataset.csv")
            if alt_path2.exists():
                path = alt_path2
    return pd.read_csv(path)
