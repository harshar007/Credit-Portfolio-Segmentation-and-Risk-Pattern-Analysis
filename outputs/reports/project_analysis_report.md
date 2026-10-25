# Credit Portfolio Segmentation & Risk Pattern Analysis: Technical Report

## 1. Executive Summary
This project analyzes a credit portfolio consisting of personal loans using unsupervised machine learning techniques. The objective is to identify distinct borrower profiles, examine historical default patterns across these behavioral segments without data leakage, and detect high-risk anomalies.

## 2. Dataset & Quality Checks
- **Source**: Lending Club loan granting dataset (`loans_granting_model_dataset.csv`).
- **Sample Size**: 25,000 borrower records.
- **Key Features**:
  - `annual_income`: Annual stated income ($).
  - `dti`: Debt-to-Income ratio (%).
  - `loan_amount`: Funded loan amount ($).
  - `fico_score`: Credit score.
  - `emp_length_years`: Employment tenure in years.
  - `Default`: Ground-truth loan performance (0 = Fully Paid, 1 = Defaulted).
- **Data Quality**:
  - Imputed missing numerical values using the median to resist extreme outlier influence.
  - Filtered extreme outliers (income < $5k or > $1M; DTI < 0% or > 100%).

## 3. Preprocessing & Dimensionality Reduction
- **Standardization**: Applied `StandardScaler` to align features to mean 0, variance 1. Crucially, the target column `Default` was excluded during training to preserve the unsupervised learning design.
- **PCA**: Compressed the 5 financial features into 2 orthogonal components (PC1, PC2) capturing >50% of total variance, enabling 2D visual projection.

## 4. Clustering & Anomaly Detection Analysis
We implemented and evaluated three distinct algorithms:
1. **K-Means Clustering**:
   - The Elbow Method and Silhouette Analysis identified $k=4$ as the optimal number of clusters.
   - Provides clear, actionable centroids and rapid execution.
2. **Agglomerative Hierarchical Clustering**:
   - Built a Ward-linkage dendrogram confirming the 4 primary clusters.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
   - Configured with $\epsilon = 1.5$ and `min_samples = 15`.
   - Identified 1.8% of borrowers as severe anomalies (-1).

### Clustering Comparison Matrix

| Algorithm | Optimal Clusters | Silhouette Score | Outlier Handling | Business Usability |
| :--- | :---: | :---: | :--- | :--- |
| **K-Means** | 4 | 0.228 | Sensitive | High (Operational Standard) |
| **Hierarchical** | 4 | 0.215 | Sensitive | High (Auditing & Hierarchy) |
| **DBSCAN** | Density-based | -0.05 | Explicit (Anomalies) | High (Fraud & Anomaly Flagging) |

## 5. Segment Risk Profiles & Default Validation

| Segment Persona | Avg Income | Avg FICO | Avg DTI | Avg Loan | Historical Default Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Prime / Low Risk (0)** | $98,400 | 735 | 12.4% | $18,200 | **4.5%** |
| **Overleveraged Borrowers (1)** | $62,100 | 672 | 26.8% | $14,500 | **18.2%** |
| **Entry-Level / Young Profile (2)** | $44,500 | 684 | 15.1% | $7,200 | **11.8%** |
| **Subprime / High Risk (3)** | $51,200 | 618 | 22.5% | $16,800 | **32.4%** |
| **DBSCAN Risk Anomalies (-1)** | Variable | Variable | Extreme | Variable | **38.6%** |

## 6. Strategic Business Recommendations
1. **Prime Segment**: Expand credit limits, offer pre-approved auto/home loan bundles, and lower APRs to maintain retention.
2. **Overleveraged Segment**: Mandate debt consolidation, cap additional revolving credit exposure, and verify total debt obligations.
3. **Entry-Level Segment**: Offer structured starter loans with progressive rate reductions tied to on-time repayment milestones.
4. **Subprime Segment**: Enforce strict underwriting caps, require co-signers or collateral, and price at higher risk-adjusted margins.
5. **DBSCAN Anomalies**: Implement automated underwriting stops to route irregular profiles to senior risk officers for fraud and solvency review.

## 7. Interactive Streamlit Dashboard
The accompanying Streamlit application (`dashboard/app.py`) provides an interactive interface for credit underwriters, featuring portfolio KPIs, dynamic Plotly cluster charts, and a real-time risk assessment simulator for new loan applicants.
