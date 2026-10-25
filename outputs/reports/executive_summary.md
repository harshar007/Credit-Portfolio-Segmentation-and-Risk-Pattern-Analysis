# Executive Summary: Credit Portfolio Segmentation & Risk Pattern Analysis

## 1. Project Background
Financial institutions face significant credit risk exposure if borrowers are treated with a one-size-fits-all underwriting policy. By applying unsupervised machine learning algorithms (K-Means, Hierarchical Clustering, and DBSCAN) without leaking default labels into the training process, we segmented borrowers based purely on financial capacity, debt burden, and credit behavior.

## 2. Segment Risk Profiles
Our analysis uncovered four distinct borrower segments with dramatically differing historical default patterns:

1. **Prime / Low Risk (Cluster 0)**:
   - High annual income (~$95k+), strong FICO scores (~730+), low DTI (~12%).
   - Historical Default Rate: ~4.5%.
   - **Strategy**: Offer higher credit limits, pre-approved mortgages, and competitive low-interest loans to drive retention.

2. **Overleveraged Borrowers (Cluster 1)**:
   - Moderate income, high debt-to-income ratio (>25%), average FICO (~670).
   - Historical Default Rate: ~18.2%.
   - **Strategy**: Require debt consolidation, cap maximum exposure, and verify supplemental income sources.

3. **Entry-Level / Young Profile (Cluster 2)**:
   - Lower employment tenure, modest loan amounts (~$6k - $8k), average FICO (~685).
   - Historical Default Rate: ~11.8%.
   - **Strategy**: Offer introductory credit lines with milestone-based credit score improvements.

4. **Subprime / High Risk (Cluster 3)**:
   - Low FICO scores (<630), elevated loan amounts relative to income, high DTI.
   - Historical Default Rate: ~32.4%.
   - **Strategy**: Stricter underwriting criteria, mandatory co-signers, higher risk-adjusted pricing (APR), or collateral requirements.

## 3. Anomaly Detection Takeaways
DBSCAN successfully identified abnormal borrower records (~1.8% of the portfolio). These records exhibit extreme debt burdens, irregular income-to-loan ratios, or data entry inconsistencies, and experience a default rate of over 38%. Underwriting rules should route these applications directly to senior credit analysts for manual verification.
