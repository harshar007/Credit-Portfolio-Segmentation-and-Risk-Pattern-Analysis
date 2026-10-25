"""
Interactive Streamlit Dashboard: Credit Portfolio Segmentation & Risk Pattern Analysis.
Provides executive portfolio KPI tracking, unsupervised borrower cluster exploration (PCA/UMAP),
default pattern validation, density-based anomaly detection, and a What-If underwriting simulator.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

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

# Page Configuration
st.set_page_config(
    page_title="Credit Portfolio Risk & Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_portfolio_data():
    """Loads and caches processed loan portfolio data."""
    csv_path = PROJECT_ROOT / "data" / "processed" / "cleaned_loans_data.csv"
    df = load_processed_data(str(csv_path))
    
    features = ["annual_income", "dti", "loan_amount", "fico_score", "emp_length_years"]
    X_scaled, feat_names, scaler = prepare_clustering_features(df, features=features)
    
    # Precompute K-Means (k=4)
    km_model, km_labels, _ = fit_kmeans(X_scaled, n_clusters=4, random_state=42)
    df["Cluster"] = km_labels
    df["Segment"] = df["Cluster"].map(SEGMENT_LABEL_MAPPING).fillna("Other")
    
    # 2D PCA for visualization
    X_pca, pca_model, var_ratio, _ = apply_pca(X_scaled, n_components=2)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]
    
    # Precompute DBSCAN anomalies
    _, db_labels, db_stats = fit_dbscan(X_scaled, eps=1.5, min_samples=25)
    df["Anomaly"] = np.where(db_labels == -1, "Anomaly (-1)", "Inlier")
    
    return df, scaler, feat_names, km_model, var_ratio, db_stats


# Load Cached Data
try:
    df, scaler, feature_names, km_model, pca_var_ratio, db_stats = get_cached_portfolio_data()
except Exception as e:
    st.error(f"Error loading portfolio data: {e}")
    st.stop()


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-cards.png", width=70)
st.sidebar.title("Credit Risk Analytics")
st.sidebar.markdown("**Portfolio Segmentation & Risk Hub**")

nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Portfolio KPI Overview",
        "🔍 Unsupervised Segmentation",
        "⚠️ Default Pattern Analysis",
        "🚨 Anomaly Detection (DBSCAN)",
        "🎛️ What-If Borrower Qualifier",
        "📋 Strategic Credit Policy",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Architecture Note:** The historical `Default` target was strictly excluded from clustering inputs to maintain genuine unsupervised learning."
)


# ---------------- PAGE 1: PORTFOLIO OVERVIEW ----------------
if nav_selection == "📊 Portfolio KPI Overview":
    st.markdown('<div class="main-header">Credit Portfolio Executive Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Holistic monitoring of loan portfolio volume, debt capacity, FICO credit distribution, and repayment outcomes.</div>',
        unsafe_allow_html=True,
    )

    # Key KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total_loans = len(df)
    avg_loan = df["loan_amount"].mean()
    avg_income = df["annual_income"].mean()
    avg_fico = df["fico_score"].mean()
    default_rate = df["Default"].mean() * 100

    col1.metric("Total Loans Monitored", f"{total_loans:,}")
    col2.metric("Avg Loan Amount", f"${avg_loan:,.0f}")
    col3.metric("Avg Annual Income", f"${avg_income:,.0f}")
    col4.metric("Average FICO Score", f"{avg_fico:.1f}")
    col5.metric("Portfolio Default Rate", f"{default_rate:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Distribution of Loan Purpose")
        purpose_counts = df["purpose"].value_counts().head(8).reset_index()
        purpose_counts.columns = ["Purpose", "Count"]
        fig_purpose = px.bar(
            purpose_counts,
            x="Count",
            y="Purpose",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            title="Top Loan Purposes by Application Volume",
        )
        fig_purpose.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_purpose, use_container_width=True)

    with col_right:
        st.subheader("Home Ownership Distribution")
        home_counts = df["home_ownership"].value_counts().reset_index()
        home_counts.columns = ["Ownership", "Count"]
        fig_home = px.pie(
            home_counts,
            values="Count",
            names="Ownership",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Borrower Housing Status Breakdown",
        )
        st.plotly_chart(fig_home, use_container_width=True)

    st.subheader("Borrower Annual Income vs Loan Amount")
    sample_scatter = df.sample(min(3000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample_scatter,
        x="annual_income",
        y="loan_amount",
        color="Segment",
        opacity=0.6,
        labels={"annual_income": "Annual Income ($)", "loan_amount": "Loan Amount ($)"},
        title="Income vs Loan Size Colored by Discovered Borrower Segment",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------- PAGE 2: UNSUPERVISED SEGMENTATION ----------------
elif nav_selection == "🔍 Unsupervised Segmentation":
    st.markdown('<div class="main-header">Unsupervised Borrower Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-dimensional borrower clustering with K-Means (k=4) mapped across Principal Component projections.</div>',
        unsafe_allow_html=True,
    )

    profile_df = profile_borrower_segments(df, df["Cluster"].values)

    st.subheader("Discovered Borrower Segment Profiles")
    display_profile = profile_df.copy()
    display_profile = display_profile[
        [
            "Cluster",
            "segment_name",
            "borrower_count",
            "portfolio_share_pct",
            "annual_income",
            "loan_amount",
            "fico_score",
            "dti",
            "emp_length_years",
            "default_rate_pct",
        ]
    ]
    display_profile.columns = [
        "ID",
        "Segment Name",
        "Borrowers",
        "Share (%)",
        "Avg Income ($)",
        "Avg Loan ($)",
        "Avg FICO",
        "Avg DTI (%)",
        "Avg Tenure (Yrs)",
        "Default Rate (%)",
    ]
    st.dataframe(display_profile.style.highlight_max(subset=["Default Rate (%)"], color="#FEE2E2"), use_container_width=True)

    st.markdown("---")

    col_pca, col_radar = st.columns([1.2, 1])

    with col_pca:
        st.subheader("2D Principal Component Space (PCA)")
        st.caption(f"PC1 explains {pca_var_ratio[0]*100:.1f}% variance | PC2 explains {pca_var_ratio[1]*100:.1f}% variance")
        pca_sample = df.sample(min(4000, len(df)), random_state=42)
        fig_pca = px.scatter(
            pca_sample,
            x="PCA1",
            y="PCA2",
            color="Segment",
            hover_data=["annual_income", "loan_amount", "fico_score", "dti"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7,
            title="Cluster Separation in 2D Latent Space",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_radar:
        st.subheader("Segment Trait Comparison")
        categories = ["Income", "Loan Size", "FICO Score", "Debt Burden (DTI)", "Job Tenure"]
        
        # Normalized metrics for radar chart
        radar_fig = go.Figure()
        seg_groups = df.groupby("Segment")
        
        for name, group in seg_groups:
            r_vals = [
                group["annual_income"].mean() / df["annual_income"].mean(),
                group["loan_amount"].mean() / df["loan_amount"].mean(),
                group["fico_score"].mean() / df["fico_score"].mean(),
                group["dti"].mean() / df["dti"].mean(),
                (group["emp_length_years"].mean() + 0.1) / (df["emp_length_years"].mean() + 0.1),
            ]
            r_vals.append(r_vals[0])
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=r_vals,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=name,
                    opacity=0.4,
                )
            )

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.8])),
            showlegend=True,
            title="Relative Segment Feature Ratios (1.0 = Portfolio Average)",
        )
        st.plotly_chart(radar_fig, use_container_width=True)


# ---------------- PAGE 3: DEFAULT PATTERN ANALYSIS ----------------
elif nav_selection == "⚠️ Default Pattern Analysis":
    st.markdown('<div class="main-header">Historical Default Pattern Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Post-clustering validation: Uncovering repayment performance across unsupervised borrower segments.</div>',
        unsafe_allow_html=True,
    )

    profile_df = profile_borrower_segments(df, df["Cluster"].values)
    port_default = df["Default"].mean() * 100

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historical Default Rate by Discovered Segment")
        fig_def = px.bar(
            profile_df,
            x="segment_name",
            y="default_rate_pct",
            color="default_rate_pct",
            color_continuous_scale="Reds",
            text="default_rate_pct",
            labels={"segment_name": "Borrower Segment", "default_rate_pct": "Default Rate (%)"},
            title="Segment Default Rate vs Overall Portfolio Benchmark",
        )
        fig_def.add_hline(
            y=port_default,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Portfolio Benchmark ({port_default:.1f}%)",
            annotation_position="bottom right",
        )
        fig_def.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_def, use_container_width=True)

    with col2:
        st.subheader("Relative Risk Ratio Index")
        fig_rr = px.bar(
            profile_df,
            x="segment_name",
            y="relative_risk_ratio",
            color="relative_risk_ratio",
            color_continuous_scale="RdYlGn_r",
            text="relative_risk_ratio",
            labels={"segment_name": "Borrower Segment", "relative_risk_ratio": "Relative Risk Index"},
            title="Relative Risk (Default Rate / Portfolio Mean)",
        )
        fig_rr.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Baseline (1.0)")
        fig_rr.update_traces(texttemplate="%{text:.2f}x", textposition="outside")
        st.plotly_chart(fig_rr, use_container_width=True)

    st.markdown("---")
    st.subheader("Default Breakdown by Segment and Loan Purpose")
    cross_tab = (
        pd.crosstab(
            df["purpose"],
            df["Segment"],
            values=df["Default"],
            aggfunc="mean",
        )
        * 100
    ).round(1)

    st.dataframe(cross_tab.style.background_gradient(cmap="Reds", axis=None), use_container_width=True)


# ---------------- PAGE 4: ANOMALY DETECTION (DBSCAN) ----------------
elif nav_selection == "🚨 Anomaly Detection (DBSCAN)":
    st.markdown('<div class="main-header">Density-Based Anomaly Detection (DBSCAN)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Screening irregular borrower profiles and out-of-distribution credit applications with DBSCAN density clustering.</div>',
        unsafe_allow_html=True,
    )

    anom_results = analyze_anomalies(df, np.where(df["Anomaly"] == "Anomaly (-1)", -1, 0))

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomalies Detected", f"{anom_results['anomaly_count']:,}")
    col2.metric("Anomaly Share", f"{anom_results['anomaly_pct']:.2f}%")
    col3.metric(
        "Anomaly Default Rate",
        f"{anom_results.get('anomaly_default_rate_pct', 0):.2f}%",
        f"{anom_results.get('anomaly_default_rate_pct', 0) - anom_results.get('inlier_default_rate_pct', 0):+.2f}% vs Inliers",
    )

    st.markdown("---")
    col_plot, col_comp = st.columns([1.2, 1])

    with col_plot:
        st.subheader("PCA Visualization of Flagged Anomalies")
        sample_anom = df.sample(min(4000, len(df)), random_state=42)
        fig_anom = px.scatter(
            sample_anom,
            x="PCA1",
            y="PCA2",
            color="Anomaly",
            color_discrete_map={"Inlier": "#CBD5E1", "Anomaly (-1)": "#EF4444"},
            opacity=0.7,
            title="DBSCAN Isolated Outliers in Feature Space",
        )
        st.plotly_chart(fig_anom, use_container_width=True)

    with col_comp:
        st.subheader("Financial Comparison: Inliers vs Anomalies")
        comp_df = anom_results["feature_comparison"].copy()
        st.dataframe(comp_df[["group", "annual_income", "loan_amount", "fico_score", "dti", "emp_length_years"]], use_container_width=True)
        st.info(
            "🔎 **Key Insight:** Anomalies frequently demonstrate disproportionately high debt burdens (DTI > 35%) relative to modest incomes or unusually large loan requests."
        )

    st.subheader("Anomaly Audit Sample (Top Records)")
    anomaly_records = df[df["Anomaly"] == "Anomaly (-1)"][
        ["annual_income", "loan_amount", "dti", "fico_score", "emp_length_years", "purpose", "Default"]
    ].head(10)
    st.dataframe(anomaly_records, use_container_width=True)


# ---------------- PAGE 5: WHAT-IF BORROWER QUALIFIER ----------------
elif nav_selection == "🎛️ What-If Borrower Qualifier":
    st.markdown('<div class="main-header">Underwriting What-If Borrower Qualifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive real-time scoring: Simulate new applicant financials, predict assigned segment, and evaluate default risk.</div>',
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.subheader("Applicant Financial Profile")
        input_income = st.slider("Annual Income ($)", min_value=10000, max_value=250000, value=75000, step=2500)
        input_loan = st.slider("Requested Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=500)
        input_fico = st.slider("FICO Credit Score", min_value=600, max_value=850, value=710, step=5)
        input_dti = st.slider("Debt-to-Income Ratio (DTI %)", min_value=0.0, max_value=45.0, value=15.0, step=0.5)
        input_emp = st.slider("Employment Tenure (Years)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)

    with col_result:
        st.subheader("Real-Time Underwriting Assessment")
        
        applicant_data = {
            "annual_income": float(input_income),
            "dti": float(input_dti),
            "loan_amount": float(input_loan),
            "fico_score": float(input_fico),
            "emp_length_years": float(input_emp),
        }
        
        scaled_input = transform_new_applicant(applicant_data, scaler, feature_names)
        pred_cluster = int(km_model.predict(scaled_input)[0])
        seg_name = SEGMENT_LABEL_MAPPING.get(pred_cluster, f"Cluster {pred_cluster}")

        # Segment benchmark stats
        seg_stats = df[df["Cluster"] == pred_cluster]
        seg_default = seg_stats["Default"].mean() * 100

        if seg_default < 12.0:
            tier_badge = "🟢 LOW RISK TIER"
            tier_color = "#10B981"
            action_text = "AUTO-APPROVE: Prime profile, qualifying for preferred low-interest credit tiers."
        elif seg_default < 21.0:
            tier_badge = "🟡 MODERATE RISK TIER"
            tier_color = "#F59E0B"
            action_text = "CONDITIONAL APPROVAL: Standard underwriting terms with income verification."
        else:
            tier_badge = "🔴 HIGH RISK TIER"
            tier_color = "#EF4444"
            action_text = "MANUAL REVIEW REQUIRED: Stringent debt verification and co-signer recommended."

        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; padding: 20px; border-radius: 8px; border-left: 6px solid {tier_color};">
                <h3 style="margin-top:0; color: #1E293B;">Assigned Segment: <b>{seg_name}</b></h3>
                <h4 style="color: {tier_color}; margin: 5px 0;">{tier_badge}</h4>
                <p><b>Historical Segment Default Rate:</b> {seg_default:.1f}%</p>
                <p><b>Underwriter Recommendation:</b> {action_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Comparison to Segment Average")
        comp_df = pd.DataFrame(
            {
                "Metric": ["Annual Income ($)", "Loan Amount ($)", "FICO Score", "DTI (%)", "Tenure (Yrs)"],
                "Applicant": [f"${input_income:,.0f}", f"${input_loan:,.0f}", input_fico, f"{input_dti:.1f}%", f"{input_emp:.1f}"],
                f"Avg ({seg_name})": [
                    f"${seg_stats['annual_income'].mean():,.0f}",
                    f"${seg_stats['loan_amount'].mean():,.0f}",
                    f"{seg_stats['fico_score'].mean():.1f}",
                    f"{seg_stats['dti'].mean():.1f}%",
                    f"{seg_stats['emp_length_years'].mean():.1f}",
                ],
            }
        )
        st.table(comp_df)


# ---------------- PAGE 6: STRATEGIC CREDIT POLICY ----------------
elif nav_selection == "📋 Strategic Credit Policy":
    st.markdown('<div class="main-header">Strategic Credit Policy & Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Actionable lending strategies, capital allocation optimizations, and credit governance policies.</div>',
        unsafe_allow_html=True,
    )

    recs = generate_business_recommendations(profile_borrower_segments(df, df["Cluster"].values))

    for rec in recs:
        with st.expander(f"📌 {rec['segment']} — {rec['risk_level']}", expanded=True):
            st.markdown(f"**Recommended Underwriting Action:** `{rec['action']}`")
            st.markdown(f"**Strategic Implementation:** {rec['strategy']}")

    st.markdown("---")
    st.subheader("Key Takeaways for Risk Management")
    st.markdown(
        """
        1. **Model Governance & Compliance:** Strictly maintaining separation between clustering features and historical repayment outcomes (`Default`) ensures non-discriminatory, data-driven segmentation.
        2. **Multi-Factor Risk Isolation:** FICO score alone is insufficient; applicants with FICO ~700 in the *Affluent High-Balance* cluster still suffer a 20.9% default rate due to excessive loan leverage.
        3. **Automated Anomaly Interception:** Integrating DBSCAN-based anomaly detection directly into the origination funnel prevents abnormal debt-to-income spikes from slipping through standard rules.
        """
    )
