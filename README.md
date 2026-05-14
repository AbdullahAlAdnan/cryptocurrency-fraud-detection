# Cryptocurrency Fraud Detection

## Motivation
This project empirically implements the AI-driven forensic 
framework proposed in my published Springer paper:

**"Forensic Analysis of Cryptocurrency Transactions: 
Leveraging Blockchain for Fraud Detection and Regulatory 
Compliance"** — ICTIS 2025, Springer.
DOI: 10.1007/978-981-95-1357-4_38

## Datasets

### Primary Dataset — Elliptic Bitcoin Dataset
- 203,769 real Bitcoin transactions
- 166 features per transaction
- Labels: illicit (fraud), licit (normal), unknown
- Source: Elliptic blockchain analytics firm
- Direct connection to cryptocurrency forensics research

### Preliminary Dataset — IEEE-CIS Fraud Detection
- 590,540 e-commerce transactions
- Used for baseline methodology validation
- Source: Kaggle

## Research Questions
1. Which transaction features best distinguish 
   illicit from licit Bitcoin transactions?
2. How do network/graph features compare to 
   local transaction features for fraud detection?
3. Which ML model performs best under class imbalance?
4. How do fraud patterns evolve across time steps?
5. Can ML provide explainable predictions that satisfy
   regulatory compliance requirements (FATF, EU MiCA)?

## Key Findings So Far

### Phase 1 — EDA Findings (Elliptic Bitcoin Dataset)
- **Finding 1:** Bitcoin fraud rate = 9.76% 
  (4x higher than e-commerce at 2.65%)
- **Finding 2:** Fraud increases 3x over time 
  (17 illicit at step 1 → 56 at step 49)
- **Finding 3:** All 165 features differ between 
  illicit and licit transactions
- **Finding 4:** Network/aggregated features are 
  strongest fraud signals (feature_53 diff = 1.263)
- **Finding 5:** Illicit transactions show lower 
  network connectivity — matches laundering behavior
- **Finding 6:** Zero missing values — clean dataset
- **Finding 7:** 9.2:1 class imbalance — standard 
  accuracy is misleading, AUC-ROC required

### Baseline Finding (IEEE-CIS Dataset)
- Simple threshold catches only 5.3% of fraud
- Confirms multi-feature ML approaches are essential

## Project Phases
- [x] Phase 1 — Baseline analysis (IEEE-CIS dataset)
- [x] Phase 1 — EDA on Elliptic Bitcoin Dataset
- [ ] Phase 2 — Visualization and pattern analysis
- [ ] Phase 3 — ML Models (Logistic Regression, 
               Random Forest, XGBoost)
- [ ] Phase 4 — SHAP explainability analysis
- [ ] Phase 5 — Results write-up and paper draft

## Tools
Python, NumPy, Pandas, Matplotlib, Seaborn, 
Scikit-learn, SHAP

## Author
Abdullah Al Adnan
MS Information Technology — Emporia State University
[GitHub](https://github.com/AbdullahAlAdnan)

## Related Publication
This project extends:
Nafiz Eashrak, Mohammad Ikbal Hossain, 
Md. Omum Siddique Auyon, Abdullah Al Adnan. (2026).
Forensic Analysis of Cryptocurrency Transactions: 
Leveraging Blockchain for Fraud Detection and 
Regulatory Compliance. Springer ICTIS 2025.
DOI: 10.1007/978-981-95-1357-4_38
