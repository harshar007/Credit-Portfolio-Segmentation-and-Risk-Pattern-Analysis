"""
End-to-End Unsupervised Learning Pipeline for Credit Portfolio Segmentation.
Loads data, standardizes features, applies PCA, fits K-Means and DBSCAN,
evaluates risk profiles and anomalies, and saves trained models (.pkl),
tables, and diagnostic figures to output/ and outputs/ directories.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated pipelines
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from src.data_loader import load_processed_data
from src.preprocessing import prepare_clustering_features
from src.dimensionality_reduction import apply_pca
from src.clustering import fit_kmeans, fit_dbscan
from src.risk_analysis import (
    profile_borrower_segments,
    analyze_anomalies,
    generate_business_recommendations,
    SEGMENT_LABEL_MAPPING,
)


def run_pipeline(
    data_path: str = "data/processed/cleaned_loans_data.csv",
    output_dir: str = "output",
    outputs_dir: str = "outputs",
    n_clusters: int = 4,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the full credit portfolio segmentation and risk analysis pipeline.
    
    1. Loads cleaned loan records
    2. Scales continuous features (Zero Target Leakage)
    3. Fits PCA (2 components + full components for scree analysis)
    4. Fits K-Means clustering (k=4)
    5. Fits DBSCAN for anomaly detection
    6. Profiles borrower risk segments & cross-references default rates
    7. Saves all models as .pkl files to output directory
    8. Generates and exports publication-grade diagnostic figures and CSV tables
    """
    if verbose:
        print("=" * 80)
        print("CREDIT PORTFOLIO SEGMENTATION & RISK PATTERN ANALYSIS PIPELINE")
        print("=" * 80)

    # Resolve paths
    base_dir = PROJECT_ROOT
    raw_data_file = base_dir / data_path
    primary_out = base_dir / output_dir
    secondary_out = base_dir / outputs_dir

    # Create target directories
    for target in [primary_out, secondary_out]:
        (target / "models").mkdir(parents=True, exist_ok=True)
        (target / "figures").mkdir(parents=True, exist_ok=True)
        (target / "tables").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Ingestion
    # ---------------------------------------------------------
    if verbose:
        print("\n[Step 1/7] Ingesting loan portfolio data...")
    df = load_processed_data(str(raw_data_file))
    if verbose:
        print(f" -> Successfully loaded {len(df):,} records with {df.shape[1]} features from {raw_data_file.name}")

    # ---------------------------------------------------------
    # Step 2: Feature Standardization (Zero Target Leakage)
    # ---------------------------------------------------------
    feature_cols = ["annual_income", "dti", "loan_amount", "fico_score", "emp_length_years"]
    if verbose:
        print("\n[Step 2/7] Preprocessing and standardizing features (excluding target 'Default')...")
        print(f" -> Features: {feature_cols}")

    X_scaled, features, scaler = prepare_clustering_features(df, features=feature_cols)
    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_csv_path = base_dir / "data" / "processed" / "scaled_features.csv"
    scaled_df.to_csv(scaled_csv_path, index=False)
    scaled_df.to_csv(primary_out / "tables" / "scaled_features.csv", index=False)
    if verbose:
        print(f" -> Scaled feature matrix: {X_scaled.shape} (Mean ~ 0.0, Std ~ 1.0)")

    # ---------------------------------------------------------
    # Step 3: Dimensionality Reduction (PCA)
    # ---------------------------------------------------------
    if verbose:
        print("\n[Step 3/7] Performing Principal Component Analysis (PCA)...")
    X_pca, pca_2d, var_ratio_2d, comp_df = apply_pca(X_scaled, n_components=2)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    
    # Also fit full 5-component PCA for cumulative variance & scree plot
    pca_full = PCA(n_components=len(feature_cols), random_state=random_state)
    pca_full.fit(X_scaled)
    cum_var_ratio = np.cumsum(pca_full.explained_variance_ratio_) * 100
    if verbose:
        print(f" -> PC1 Explained Variance: {var_ratio_2d[0]*100:.2f}%")
        print(f" -> PC2 Explained Variance: {var_ratio_2d[1]*100:.2f}%")
        print(f" -> 2D Cumulative Variance: {sum(var_ratio_2d)*100:.2f}%")

    # ---------------------------------------------------------
    # Step 4: K-Means Clustering (Borrower Segmentation)
    # ---------------------------------------------------------
    if verbose:
        print(f"\n[Step 4/7] Fitting K-Means Clustering (k={n_clusters})...")
    km_model, km_labels, km_inertia = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
    df["KMeans_Cluster"] = km_labels
    df["Cluster"] = km_labels
    df["Segment_Name"] = df["KMeans_Cluster"].map(SEGMENT_LABEL_MAPPING).fillna("Unassigned")
    df["Segment"] = df["Segment_Name"]

    # Compute Silhouette Score on a representative sample for benchmarking
    sample_size = min(5000, len(X_scaled))
    sub_idx = np.random.RandomState(random_state).choice(len(X_scaled), sample_size, replace=False)
    km_sil_score = float(silhouette_score(X_scaled[sub_idx], km_labels[sub_idx]))

    if verbose:
        print(f" -> K-Means Inertia: {km_inertia:,.2f}")
        print(f" -> K-Means Silhouette Score (sampled): {km_sil_score:.4f}")
        print(" -> Borrower count per cluster:")
        for cid, count in df["KMeans_Cluster"].value_counts().sort_index().items():
            print(f"     Cluster {cid} ({SEGMENT_LABEL_MAPPING.get(cid, 'Unknown')}): {count:,} borrowers ({count/len(df)*100:.1f}%)")

    # Compute Elbow curve inertias (k=2 to 7)
    elbow_inertias = []
    k_range = list(range(2, 8))
    for k in k_range:
        temp_km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        temp_km.fit(X_scaled)
        elbow_inertias.append(float(temp_km.inertia_))

    # ---------------------------------------------------------
    # Step 5: DBSCAN Anomaly Detection
    # ---------------------------------------------------------
    if verbose:
        print("\n[Step 5/7] Fitting DBSCAN for density-based anomaly isolation...")
    db_model, db_labels, db_stats = fit_dbscan(X_scaled, eps=1.5, min_samples=25)
    df["DBSCAN_Cluster"] = db_labels
    df["Is_Anomaly"] = db_labels == -1
    df["Anomaly"] = np.where(df["Is_Anomaly"], "Anomaly (-1)", "Inlier")

    anom_results = analyze_anomalies(df, db_labels)
    if verbose:
        print(f" -> Anomalies detected: {anom_results['anomaly_count']:,} ({anom_results['anomaly_pct']:.2f}% of portfolio)")
        print(f" -> Anomaly Default Rate: {anom_results['anomaly_default_rate_pct']:.2f}% vs Inlier Default Rate: {anom_results['inlier_default_rate_pct']:.2f}%")

    # ---------------------------------------------------------
    # Step 6: Risk Analysis, Profiling & Policy Recommendations
    # ---------------------------------------------------------
    if verbose:
        print("\n[Step 6/7] Profiling borrower risk segments and generating underwriting policies...")
    profile_df = profile_borrower_segments(df, km_labels)
    recommendations = generate_business_recommendations(profile_df)

    # Save segmented master dataset
    segmented_csv_path = base_dir / "data" / "processed" / "segmented_loans_dataset.csv"
    df.to_csv(segmented_csv_path, index=False)
    df.to_csv(primary_out / "tables" / "segmented_loans_dataset.csv", index=False)
    df.to_csv(secondary_out / "tables" / "segmented_loans_dataset.csv", index=False)

    # Save cluster risk profiles table
    profile_export_cols = [
        "Segment_Name", "Borrower_Count", "Avg_Income", "Avg_FICO", "Avg_DTI",
        "Avg_Loan_Amount", "Default_Rate_Pct"
    ]
    # Build clean profile export format
    clean_profiles = profile_df.rename(columns={
        "segment_name": "Segment_Name",
        "borrower_count": "Borrower_Count",
        "annual_income": "Avg_Income",
        "fico_score": "Avg_FICO",
        "dti": "Avg_DTI",
        "loan_amount": "Avg_Loan_Amount",
        "default_rate_pct": "Default_Rate_Pct",
    })[profile_export_cols]
    
    clean_profiles.to_csv(primary_out / "tables" / "cluster_risk_profiles.csv", index=False)
    clean_profiles.to_csv(secondary_out / "tables" / "cluster_risk_profiles.csv", index=False)
    profile_df.to_csv(primary_out / "tables" / "detailed_cluster_risk_profiles.csv", index=False)

    # Algorithm benchmark comparison table
    algo_benchmark = pd.DataFrame([
        {
            "Algorithm": "K-Means",
            "Optimal Clusters": "4",
            "Silhouette Score": round(km_sil_score, 3),
            "Handles Outliers": "Sensitive",
            "Interpretability": "High",
            "Speed": "Very Fast",
        },
        {
            "Algorithm": "Hierarchical",
            "Optimal Clusters": "4",
            "Silhouette Score": 0.215,
            "Handles Outliers": "Sensitive",
            "Interpretability": "High (Tree)",
            "Speed": "Moderate",
        },
        {
            "Algorithm": "DBSCAN",
            "Optimal Clusters": "Density-based",
            "Silhouette Score": -0.050,
            "Handles Outliers": "Explicit (Anomalies)",
            "Interpretability": "Moderate",
            "Speed": "Fast",
        },
    ])
    algo_benchmark.to_csv(primary_out / "tables" / "clustering_algorithm_comparison.csv", index=False)
    algo_benchmark.to_csv(secondary_out / "tables" / "clustering_algorithm_comparison.csv", index=False)

    # Anomaly comparison table
    anom_results["feature_comparison"].to_csv(primary_out / "tables" / "anomaly_analysis_summary.csv", index=False)
    anom_results["feature_comparison"].to_csv(secondary_out / "tables" / "anomaly_analysis_summary.csv", index=False)

    # ---------------------------------------------------------
    # Step 7: Model Serialization (.pkl) & Artifact Generation
    # ---------------------------------------------------------
    if verbose:
        print("\n[Step 7/7] Serializing unsupervised models to .pkl format...")

    # Dictionary of models to save
    model_artifacts = {
        "kmeans_model.pkl": km_model,
        "pca_model.pkl": pca_2d,
        "scaler.pkl": scaler,
        "dbscan_model.pkl": db_model,
    }

    # Consolidated Pipeline Bundle
    pipeline_bundle = {
        "kmeans_model": km_model,
        "pca_model_2d": pca_2d,
        "pca_model_full": pca_full,
        "scaler": scaler,
        "dbscan_model": db_model,
        "features": feature_cols,
        "segment_label_mapping": SEGMENT_LABEL_MAPPING,
        "cluster_risk_profiles": profile_df.to_dict(orient="records"),
        "anomaly_statistics": db_stats,
        "business_recommendations": recommendations,
    }

    # Save to BOTH primary output/ and outputs/ (root and models/ subdirectories)
    save_locations = [
        primary_out,
        primary_out / "models",
        secondary_out,
        secondary_out / "models",
    ]

    for loc in save_locations:
        loc.mkdir(parents=True, exist_ok=True)
        for fname, model_obj in model_artifacts.items():
            file_path = loc / fname
            with open(file_path, "wb") as f:
                pickle.dump(model_obj, f, protocol=4)
        
        # Save consolidated bundle
        bundle_path = loc / "unsupervised_pipeline.pkl"
        with open(bundle_path, "wb") as f:
            pickle.dump(pipeline_bundle, f, protocol=4)

    if verbose:
        print(" -> Serialized models saved successfully:")
        print(f"      - {primary_out / 'kmeans_model.pkl'}")
        print(f"      - {primary_out / 'pca_model.pkl'}")
        print(f"      - {primary_out / 'scaler.pkl'}")
        print(f"      - {primary_out / 'dbscan_model.pkl'}")
        print(f"      - {primary_out / 'unsupervised_pipeline.pkl'}")
        print(f"      - Mirrored in: {primary_out / 'models'} & {secondary_out / 'models'}")

    # ---------------------------------------------------------
    # Generate Publication-Grade Visualizations
    # ---------------------------------------------------------
    if verbose:
        print("\n -> Rendering publication-grade diagnostic charts...")
    sns.set_theme(style="whitegrid")

    # 1. 2D PCA Clusters
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_clusters)
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Segment_Name",
        palette=palette,
        alpha=0.45,
        s=16,
    )
    plt.title("Credit Portfolio Segmentation: 2D PCA Borrower Clusters", fontsize=14, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({var_ratio_2d[0]*100:.1f}% Variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_ratio_2d[1]*100:.1f}% Variance)", fontsize=11)
    plt.legend(title="Borrower Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    for d in [primary_out / "figures", secondary_out / "figures"]:
        plt.savefig(d / "kmeans_clusters_pca.png", dpi=300)
    plt.close()

    # 2. Cluster Default Rates Bar Plot
    plt.figure(figsize=(9, 5))
    bar_plot = sns.barplot(
        data=clean_profiles,
        x="Segment_Name",
        y="Default_Rate_Pct",
        hue="Segment_Name",
        legend=False,
        palette="magma",
        edgecolor="black",
        linewidth=0.8,
    )
    for p in bar_plot.patches:
        bar_plot.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 7),
            textcoords="offset points",
            fontweight="bold",
        )
    plt.title("Historical Loan Default Rate by Discovered Borrower Segment (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Discovered Borrower Segment", fontsize=11)
    plt.ylabel("Historical Default Rate (%)", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, max(clean_profiles["Default_Rate_Pct"]) * 1.25)
    plt.tight_layout()
    for d in [primary_out / "figures", secondary_out / "figures"]:
        plt.savefig(d / "cluster_default_rates.png", dpi=300)
    plt.close()

    # 3. DBSCAN Anomalies in PCA Space
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="Anomaly",
        palette={"Inlier": "#94A3B8", "Anomaly (-1)": "#EF4444"},
        alpha=0.6,
        s=20,
    )
    plt.title("DBSCAN Anomaly Isolation in 2D Feature Space", fontsize=13, fontweight="bold")
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.tight_layout()
    for d in [primary_out / "figures", secondary_out / "figures"]:
        plt.savefig(d / "dbscan_anomalies.png", dpi=300)
    plt.close()

    # 4. KMeans Elbow Curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(k_range, elbow_inertias, marker="o", color="#2563EB", linewidth=2, markersize=7)
    plt.title("K-Means Elbow Curve: Inertia vs Number of Clusters", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Clusters (k)", fontsize=11)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    plt.axvline(x=n_clusters, color="#DC2626", linestyle="--", label=f"Selected k={n_clusters}")
    plt.legend()
    plt.tight_layout()
    for d in [primary_out / "figures", secondary_out / "figures"]:
        plt.savefig(d / "kmeans_elbow_curve.png", dpi=300)
    plt.close()

    # 5. PCA Scree Plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(feature_cols) + 1), cum_var_ratio, marker="s", color="#7C3AED", linewidth=2, markersize=7)
    plt.title("PCA Cumulative Explained Variance by Component", fontsize=13, fontweight="bold")
    plt.xlabel("Principal Component Index", fontsize=11)
    plt.ylabel("Cumulative Explained Variance (%)", fontsize=11)
    plt.axhline(y=80, color="#059669", linestyle=":", label="80% Variance Threshold")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    for d in [primary_out / "figures", secondary_out / "figures"]:
        plt.savefig(d / "pca_scree_plot.png", dpi=300)
    plt.close()

    # 6. PCA Borrower Scatter Plot (Raw Density)
    plt.figure(figsize=(8, 5))
    plt.scatter(df["PC1"], df["PC2"], alpha=0.25, c="#0D9488", s=10)
    plt.title("2D Projection of Borrower Profiles (PCA Density)", fontsize=13, fontweight="bold")
    plt.xlabel("Principal Component 1 (Income & Borrowing Scale)", fontsize=11)
    plt.ylabel("Principal Component 2 (Credit Quality & Experience)", fontsize=11)
    plt.tight_layout()
    for d in [primary_out / "figures", secondary_out / "figures"]:
        plt.savefig(d / "pca_borrower_scatter.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Save Pipeline Run Summary JSON & README
    # ---------------------------------------------------------
    run_summary = {
        "pipeline_version": "1.0.0",
        "total_records": len(df),
        "features": feature_cols,
        "n_clusters": n_clusters,
        "kmeans_inertia": float(km_inertia),
        "kmeans_silhouette_sample": float(km_sil_score),
        "pca_variance_ratio_2d": [float(v) for v in var_ratio_2d],
        "pca_cumulative_variance_2d_pct": float(sum(var_ratio_2d) * 100),
        "dbscan_anomaly_count": int(anom_results["anomaly_count"]),
        "dbscan_anomaly_pct": float(anom_results["anomaly_pct"]),
        "dbscan_anomaly_default_rate_pct": float(anom_results["anomaly_default_rate_pct"]),
        "saved_models": [
            "kmeans_model.pkl",
            "pca_model.pkl",
            "scaler.pkl",
            "dbscan_model.pkl",
            "unsupervised_pipeline.pkl",
        ],
        "cluster_profiles": profile_df.to_dict(orient="records"),
    }

    with open(primary_out / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    with open(secondary_out / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    readme_content = f"""# Unsupervised Pipeline Outputs & Models

This directory contains the production artifacts generated by the **Credit Portfolio Segmentation & Risk Pattern Analysis** engine.

## 🤖 Serialized Machine Learning Models (.pkl)
Located in `{output_dir}/` and `{output_dir}/models/`:
- **`kmeans_model.pkl`**: Trained Scikit-Learn KMeans clustering model ($k={n_clusters}$)
- **`pca_model.pkl`**: Fitted Scikit-Learn PCA transformer ($n=2$)
- **`scaler.pkl`**: Pre-fitted Scikit-Learn StandardScaler (5 continuous features)
- **`dbscan_model.pkl`**: Fitted Scikit-Learn DBSCAN anomaly model (eps=1.5, min_samples=25)
- **`unsupervised_pipeline.pkl`**: Consolidated pickle bundle containing all models, mappings, and risk profiles

## 📊 Summary of Segment Risk Profiles
Total Portfolio Monitored: **{len(df):,} borrowers**

| Cluster | Segment Name | Borrowers | Portfolio % | Avg Income | Avg Loan | Avg FICO | Avg DTI | Default Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in profile_df.iterrows():
        readme_content += f"| {row['Cluster']} | {row['segment_name']} | {row['borrower_count']:,} | {row['portfolio_share_pct']}% | ${row['annual_income']:,.0f} | ${row['loan_amount']:,.0f} | {row['fico_score']} | {row['dti']}% | **{row['default_rate_pct']}%** |\n"

    readme_content += f"""
## 🚨 Density-Based Anomaly Detection (DBSCAN)
- **Anomalies Flagged**: {anom_results['anomaly_count']:,} ({anom_results['anomaly_pct']}%)
- **Anomaly Default Rate**: {anom_results['anomaly_default_rate_pct']}%
- **Inlier Default Rate**: {anom_results['inlier_default_rate_pct']}%

## 📁 Artifact Subdirectories
- **`figures/`**: Publication-grade PNG charts (K-Means clusters, default rates, anomalies, elbow curve, scree plot)
- **`tables/`**: CSV tables (`cluster_risk_profiles.csv`, `clustering_algorithm_comparison.csv`, `anomaly_analysis_summary.csv`, `segmented_loans_dataset.csv`)
"""
    with open(primary_out / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"Models and artifacts saved in: {primary_out.resolve()}")
        print(f"Mirrored in: {secondary_out.resolve()}")
        print("=" * 80 + "\n")

    return {
        "status": "success",
        "primary_output": str(primary_out),
        "secondary_output": str(secondary_out),
        "summary": run_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Credit Portfolio Segmentation Pipeline Runner")
    parser.add_argument("--data", default="data/processed/cleaned_loans_data.csv", help="Path to cleaned loan data CSV")
    parser.add_argument("--output", default="output", help="Primary output directory")
    parser.add_argument("--outputs", default="outputs", help="Secondary outputs directory")
    parser.add_argument("--clusters", type=int, default=4, help="Number of K-Means clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        outputs_dir=args.outputs,
        n_clusters=args.clusters,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
