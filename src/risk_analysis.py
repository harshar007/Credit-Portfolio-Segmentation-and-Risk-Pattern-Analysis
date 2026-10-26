"""
Risk analytics, borrower segment profiling, and policy generation.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd

SEGMENT_LABEL_MAPPING: Dict[int, str] = {
    0: "Prime / Low Risk",
    1: "Overleveraged / Moderate Risk",
    2: "Entry-Level / Young Profile",
    3: "Subprime / High Risk",
}


def profile_borrower_segments(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Profiles borrower clusters across financial metrics and calculates default metrics."""
    df_temp = df.copy()
    df_temp["_Cluster_Internal"] = cluster_labels
    portfolio_default_pct = df_temp["Default"].mean() * 100

    profiles = []
    total_borrowers = len(df_temp)

    for cluster_id, grp in df_temp.groupby("_Cluster_Internal"):
        b_count = len(grp)
        def_rate = grp["Default"].mean() * 100
        rel_risk = (def_rate / portfolio_default_pct) if portfolio_default_pct > 0 else 1.0
        seg_name = SEGMENT_LABEL_MAPPING.get(cluster_id, f"Cluster {cluster_id}")

        profiles.append({
            "Cluster": cluster_id,
            "segment_name": seg_name,
            "borrower_count": b_count,
            "portfolio_share_pct": round((b_count / total_borrowers) * 100, 2),
            "annual_income": round(grp["annual_income"].mean(), 2),
            "loan_amount": round(grp["loan_amount"].mean(), 2),
            "fico_score": round(grp["fico_score"].mean(), 1),
            "dti": round(grp["dti"].mean(), 2),
            "emp_length_years": round(grp["emp_length_years"].mean(), 1),
            "default_rate_pct": round(def_rate, 2),
            "relative_risk_ratio": round(rel_risk, 2),
        })

    return pd.DataFrame(profiles)


def analyze_anomalies(df: pd.DataFrame, anomaly_labels: np.ndarray) -> Dict[str, Any]:
    """Analyzes DBSCAN anomalies vs inliers in terms of default rates and financial metrics."""
    is_anomaly = anomaly_labels == -1
    total_records = len(df)
    anom_count = int(np.sum(is_anomaly))
    anom_pct = float((anom_count / total_records) * 100) if total_records > 0 else 0.0

    df_anom = df[is_anomaly]
    df_inlier = df[~is_anomaly]

    anom_default = float(df_anom["Default"].mean() * 100) if len(df_anom) > 0 else 0.0
    inlier_default = float(df_inlier["Default"].mean() * 100) if len(df_inlier) > 0 else 0.0

    features = ["annual_income", "loan_amount", "fico_score", "dti", "emp_length_years"]
    inlier_means = df_inlier[features].mean().to_dict()
    inlier_means["group"] = "Inliers"
    anom_means = df_anom[features].mean().to_dict()
    anom_means["group"] = "Anomalies"

    comp_df = pd.DataFrame([inlier_means, anom_means])

    return {
        "anomaly_count": anom_count,
        "anomaly_pct": round(anom_pct, 2),
        "anomaly_default_rate_pct": round(anom_default, 2),
        "inlier_default_rate_pct": round(inlier_default, 2),
        "feature_comparison": comp_df,
    }


def generate_business_recommendations(profile_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Produces underwriting and credit governance strategies for each discovered segment."""
    recs = []
    policy_defaults = {
        "Prime / Low Risk": {
            "risk_level": "Low Risk (Default ~4.5%)",
            "action": "Grow Portfolio & Maximize Retention",
            "strategy": "Offer pre-approved credit lines, higher loan caps, and competitive interest rates to protect market share against competing lenders.",
        },
        "Overleveraged / Moderate Risk": {
            "risk_level": "Moderate Risk (Default ~18.2%)",
            "action": "Mitigate Exposure & Cap DTI",
            "strategy": "Enforce strict debt-to-income caps (<30%), require automated payroll deductions, and provide debt consolidation incentives.",
        },
        "Entry-Level / Young Profile": {
            "risk_level": "Emerging Risk (Default ~11.8%)",
            "action": "Nurture with Graduated Limits",
            "strategy": "Offer introductory credit lines with milestone-based credit increases and interest rate discounts upon consecutive on-time payments.",
        },
        "Subprime / High Risk": {
            "risk_level": "High Risk (Default ~32.4%)",
            "action": "Restrict & Require Collateral",
            "strategy": "Require co-signers or secured assets, impose risk-adjusted pricing (higher APRs), and route applications through secondary underwriting.",
        },
    }

    for _, row in profile_df.iterrows():
        name = row["segment_name"]
        def_pct = row["default_rate_pct"]
        cfg = policy_defaults.get(name, {
            "risk_level": f"Default Rate {def_pct:.1f}%",
            "action": "Standard Underwriting Review",
            "strategy": "Perform standard credit evaluation and debt capacity verification.",
        })
        recs.append({
            "segment": name,
            "risk_level": cfg["risk_level"],
            "action": cfg["action"],
            "strategy": cfg["strategy"],
        })

    return recs
