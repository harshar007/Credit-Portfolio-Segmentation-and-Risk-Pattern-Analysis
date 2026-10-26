# Credit Portfolio Segmentation and Risk Pattern Analysis
> **Unsupervised Machine Learning & Credit Risk Analytics**

An end-to-end unsupervised learning framework designed to segment credit portfolios, analyze historical loan default patterns, identify high-risk anomalies, and deliver an interactive underwriting risk simulator.

---

## 📁 Project Architecture & Workflow

```text
Credit-Portfolio-Segmentation-and-Risk-Pattern-Analysis/
│
├── data/
│   ├── raw/
│   │   ├── loans_granting_model_dataset.csv       <- Full Lending Club dataset (stored locally)
│   │   └── README.md                             <- Data directory instructions
│   └── processed/
│       ├── cleaned_loans_dataset.csv             <- Cleaned, standardized 25,000-borrower dataset
│       ├── scaled_features.csv                   <- Standardized features for machine learning
│       └── segmented_loans_dataset.csv           <- Portfolio with assigned clusters and anomaly flags
│
├── notebooks/
│   ├── 01_Project_Setup_and_Data_Loading.ipynb   <- Environment setup, path validation & initial data load
│   ├── 02_Data_Understanding_and_Quality_Checks.ipynb <- Data health, missing values & summary statistics
│   ├── 03_Data_Cleaning.ipynb                    <- Median imputation, column renaming & outlier filtering
│   ├── 04_Exploratory_Data_Analysis.ipynb        <- Visual distributions, boxplots & correlation heatmap
│   ├── 05_Feature_Engineering_and_Preprocessing.ipynb <- Feature standardization (StandardScaler) without target leakage
│   ├── 06_PCA_and_Dimensionality_Reduction.ipynb <- Dimensionality reduction (PCA) for 2D visual projection
│   ├── 07_KMeans_Clustering.ipynb                <- The Elbow Method & K-Means clustering (k=4)
│   ├── 08_Hierarchical_Clustering.ipynb          <- Ward-linkage dendrogram & Agglomerative clustering
│   ├── 09_DBSCAN_and_Anomaly_Analysis.ipynb      <- Density clustering & high-risk anomaly identification
│   ├── 10_Clustering_Evaluation_and_Comparison.ipynb <- Silhouette Score & Davies-Bouldin benchmarking
│   ├── 11_Cluster_Profiling_and_Default_Analysis.ipynb <- Risk persona mapping & historical default validation
│   └── 12_Final_Insights_and_Results.ipynb       <- Underwriting policy matrix & executive summary
│
├── dashboard/
│   └── app.py                                    <- Interactive Streamlit web application
│
├── outputs/
│   ├── figures/                                  <- High-resolution plots (PCA, Elbow, Default Rates, DBSCAN)
│   ├── reports/                                  <- Technical report & executive summary
│   └── tables/                                   <- Segment risk profiles & algorithm comparison tables
│
├── Credit Portfolio Segmentation and Risk Pattern Analysis.pdf <- Project specifications & evaluation rubric
├── README.md                                     <- Project documentation & setup instructions
├── requirements.txt                              <- Exact project dependencies
└── .gitignore                                    <- Git ignore rules
```

---

## 🎯 Business Context & Methodology

### Why Unsupervised Learning for Credit Portfolios?
1. **Behavioral Grouping Without Bias**: In unsupervised learning, the loan performance flag (`Default`) is withheld during model training. The algorithms group borrowers based strictly on financial behavior:
   - Annual Income (`annual_income`)
   - Debt-to-Income Ratio (`dti`)
   - Requested Loan Amount (`loan_amount`)
   - Credit Score (`fico_score`)
   - Employment Tenure (`emp_length_years`)
2. **Ground-Truth Risk Validation**: After clusters are formed, historical default rates are calculated across segments. This validates whether natural behavioral groupings correlate with true credit risk.

---

## 📊 Discovered Borrower Segments & Strategic Policies

| Segment Persona | Avg Income | Avg FICO | Avg DTI | Historical Default Rate | Recommended Underwriting Policy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Prime / Low Risk** | **$98,400** | **735** | **12.4%** | **4.5%** | **Grow Portfolio**: Offer pre-approved credit lines and competitive rates to boost customer retention. |
| **Overleveraged Borrowers** | **$62,100** | **672** | **26.8%** | **18.2%** | **Mitigate Exposure**: Enforce strict debt-to-income caps (<30%) and require debt consolidation. |
| **Entry-Level / Young Profile**| **$44,500** | **684** | **15.1%** | **11.8%** | **Nurture**: Provide starter loan products with milestone-based rate reductions as credit improves. |
| **Subprime / High Risk** | **$51,200** | **618** | **22.5%** | **32.4%** | **Restrict**: Stricter criteria, mandatory co-signers, collateral, or higher risk-adjusted pricing. |
| **DBSCAN Risk Anomalies** | *Variable* | *Variable* | *Extreme* | **38.6%** | **Flag**: Route directly to senior credit analysts for manual verification before approval. |

---

## 🚀 Getting Started

### 1. Install Dependencies
Open your terminal or command prompt in the project root:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebooks
Launch Jupyter Lab or open the notebooks directly in your preferred editor (e.g., VS Code):
```bash
jupyter lab
```
Run notebooks `01` through `12` sequentially to follow the complete data science lifecycle.

### 3. Launch the Interactive Web Dashboard
Run the Streamlit application:
```bash
streamlit run dashboard/app.py
```
This launches a browser interface featuring:
- High-level portfolio KPIs (Total Volume, Average FICO, Overall Default Rate).
- Interactive 2D PCA cluster visualizations with Plotly.
- Risk profile breakdowns for each borrower segment.
- An interactive **Underwriter Risk Simulator** to evaluate prospective loan applicants in real time.

---

## 🐳 Docker Deployment

The project includes full containerization for both the **Streamlit Dashboard** and **JupyterLab** notebook environment.

### Run with Docker Compose (Recommended)

1. **Launch the Streamlit Dashboard (Port 8501):**
   ```bash
   docker compose up dashboard
   ```
   Open your browser at **http://localhost:8501**.

2. **Launch JupyterLab for Notebook Exploration (Port 8888):**
   ```bash
   docker compose up jupyter
   ```
   Open your browser at **http://localhost:8888**.

3. **Run Both Services Concurrently:**
   ```bash
   docker compose up
   ```

4. **Run Unit Tests Inside Docker:**
   ```bash
   docker compose run --rm dashboard pytest tests
   ```

5. **Stop Containers:**
   ```bash
   docker compose down
   ```

### Run with Standalone Docker

```bash
# Build Docker image
docker build -t credit-risk-analytics .

# Run Streamlit Dashboard
docker run -d -p 8501:8501 --name credit-dashboard credit-risk-analytics

# Run JupyterLab
docker run -d -p 8888:8888 --name credit-jupyter credit-risk-analytics jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```
