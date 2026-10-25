# Credit Portfolio Segmentation and Risk Pattern Analysis
> **Unsupervised Machine Learning & Credit Risk Analytics Project**

An end-to-end unsupervised learning framework to segment borrower portfolios, uncover historical credit default patterns, detect high-risk anomalies, and build an interactive underwriting dashboard.

---

## 📁 Beginner-Friendly Project Structure

```text
Credit-Portfolio-Segmentation-and-Risk-Pattern-Analysis/
│
├── data/
│   ├── raw/
│   │   └── loans_granting_model_dataset.csv       <- Original Lending Club borrower dataset
│   └── processed/
│       ├── cleaned_loans_dataset.csv             <- Cleaned, standardized borrower dataset
│       ├── scaled_features.csv                   <- Standardized features for machine learning
│       └── segmented_loans_dataset.csv           <- Dataset with assigned cluster personas & anomalies
│
├── notebooks/
│   ├── 01_Project_Setup_and_Data_Loading.ipynb   <- Step 1: Environment setup & raw data loading
│   ├── 02_Data_Understanding_and_Quality_Checks.ipynb <- Step 2: Nulls, duplicates & summary statistics
│   ├── 03_Data_Cleaning.ipynb                    <- Step 3: Imputation, column renaming & outlier filtering
│   ├── 04_Exploratory_Data_Analysis.ipynb        <- Step 4: Visual distributions, boxplots & correlation heatmap
│   ├── 05_Feature_Engineering_and_Preprocessing.ipynb <- Step 5: Feature scaling (StandardScaler) without target leakage
│   ├── 06_PCA_and_Dimensionality_Reduction.ipynb <- Step 6: Compressing 5D features into 2D PCA components
│   ├── 07_KMeans_Clustering.ipynb                <- Step 7: The Elbow Method & K-Means clustering (k=4)
│   ├── 08_Hierarchical_Clustering.ipynb          <- Step 8: Ward linkage dendrogram & Agglomerative clustering
│   ├── 09_DBSCAN_and_Anomaly_Analysis.ipynb      <- Step 9: Density clustering & identifying high-risk anomalies
│   ├── 10_Clustering_Evaluation_and_Comparison.ipynb <- Step 10: Silhouette Score & Davies-Bouldin metrics
│   ├── 11_Cluster_Profiling_and_Default_Analysis.ipynb <- Step 11: Business personas & historical default validation
│   └── 12_Final_Insights_and_Results.ipynb       <- Step 12: Underwriting policy matrix & executive summary
│
├── dashboard/
│   ├── 13_Streamlit_Dashboard.ipynb              <- Step 13: Dashboard architecture walkthrough & quick launch
│   └── app.py                                    <- Live interactive Streamlit web dashboard
│
├── outputs/
│   ├── figures/                                  <- High-resolution charts (PCA, Elbow, Default Rates, DBSCAN)
│   ├── reports/                                  <- Executive summary report (executive_summary.md)
│   └── tables/                                   <- Risk profiles & algorithm comparison CSV tables
│
├── Credit Portfolio Segmentation and Risk Pattern Analysis.pdf <- Project guidelines & evaluation rubric
├── README.md                                     <- Project documentation & step-by-step guide
├── requirements.txt                              <- Exact project dependencies
└── .gitignore                                    <- Git ignore rules
```

---

## 🎯 Business Context & Mental Model

### Why Unsupervised Learning for Credit Risk?
1. **No Data Leakage**: In unsupervised learning, we **do not** feed the `Default` status into our clustering algorithms. We only give the algorithms financial behavior:
   - Annual Income (`annual_income`)
   - Debt-to-Income Ratio (`dti`)
   - Requested Loan Amount (`loan_amount`)
   - FICO Credit Score (`fico_score`)
   - Employment Tenure (`emp_length_years`)
2. **Behavioral Segmentation**: The models naturally group borrowers into distinct behavioral segments.
3. **Validation Against Defaults**: Once segments are formed, we compute the historical `Default` rate for each group. This proves that natural behavioral clusters have vastly different default patterns!

---

## 📊 Summary of Discovered Borrower Segments

| Segment Name | Avg Income | Avg FICO | Avg DTI | Historical Default Rate | Strategic Underwriting Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Prime / Low Risk** | **$98,400** | **735** | **12.4%** | **4.5%** | **Grow Portfolio**: Offer pre-approved credit lines and competitive low-interest rates. |
| **Overleveraged Borrowers** | **$62,100** | **672** | **26.8%** | **18.2%** | **Mitigate Exposure**: Enforce strict DTI caps (<30%) and require debt consolidation. |
| **Entry-Level / Young Profile**| **$44,500** | **684** | **15.1%** | **11.8%** | **Nurture**: Provide introductory credit with milestone-based score improvements. |
| **Subprime / High Risk** | **$51,200** | **618** | **22.5%** | **32.4%** | **Restrict**: Stricter criteria, mandatory co-signers, collateral, or higher risk-adjusted APR. |
| **DBSCAN Risk Anomalies** | *Variable* | *Variable* | *Extreme* | **38.6%** | **Flag**: Route directly to senior credit analysts for manual verification. |

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Open your terminal or PowerShell in the project directory:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebooks
Launch Jupyter Notebook or Jupyter Lab to run notebooks 01 through 12 sequentially:
```bash
jupyter lab
```
Or open any notebook directly in **VS Code** with the Python & Jupyter extensions.

### 3. Launch the Interactive Web Dashboard
Run the Streamlit application from your terminal:
```bash
streamlit run dashboard/app.py
```
This opens an interactive dashboard in your browser where you can:
- View portfolio-level KPIs (Total volume, Avg FICO, Portfolio default rate).
- Explore 2D interactive PCA clusters with Plotly.
- Analyze segment risk summaries.
- Test new applicants with the **Interactive Underwriter Risk Simulator**.

---

## 📈 Key Algorithms & Metrics

- **K-Means Clustering**: Chosen as the primary production segmentation model due to clear centroids and fast execution ($k=4$).
- **Agglomerative Hierarchical Clustering**: Validated the 4-cluster structure using Ward linkage dendrogram analysis.
- **DBSCAN**: Used specifically as an **anomaly detection engine** ($\epsilon=1.5$, `min_samples=15`) to isolate unusual, high-default borrower profiles.
- **PCA**: Compressed 5 dimensions into 2 orthogonal components for visualization while retaining >50% of the dataset variance.
- **Evaluation**: Validated using the **Silhouette Score** (cohesion vs separation) and business interpretability.
