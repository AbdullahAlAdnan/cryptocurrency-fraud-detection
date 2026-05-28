# Cryptocurrency Fraud Detection

## Motivation

This project extends and experimentally explores 
concepts introduced in my published Springer paper:

"Forensic Analysis of Cryptocurrency Transactions: 
Leveraging Blockchain for Fraud Detection and 
Regulatory Compliance" - ICTIS 2025, Springer.
DOI: 10.1007/978-981-95-1357-4_38

The goal is to empirically validate whether machine 
learning can automatically detect illicit Bitcoin 
transactions, reducing the need for manual forensic 
investigation.

## Datasets

### Primary Dataset - Elliptic Bitcoin Dataset
- 203,769 real Bitcoin transactions
- 166 features per transaction
- Labels: illicit (fraud), licit (normal), unknown
- Source: Elliptic blockchain analytics firm
- Directly connected to cryptocurrency forensics 
  research

### Preliminary Dataset - IEEE-CIS Fraud Detection
- 590,540 e-commerce transactions
- Used for methodology validation only
- Different domain from cryptocurrency
- Source: Kaggle

## Research Questions

1. Which transaction features best distinguish 
   illicit from licit Bitcoin transactions?

2. How do network and graph features compare to 
   local transaction features for fraud detection?

3. Which ML model performs best under class 
   imbalance conditions?

4. How do fraud patterns evolve across time steps?

5. Can ML provide explainable predictions that 
   satisfy regulatory compliance requirements 
   such as FATF and EU MiCA?

## Key Findings

### Phase 1 - Exploratory Data Analysis

Finding 1: Fraud rate is 9.76% among labeled 
transactions (4,545 illicit out of 46,564 labeled).
This reflects fraud prevalence in transactions 
investigated by Elliptic analysts.

Finding 2: Normalized fraud rate increases 14.9x 
over time, from 0.79% at time step 1 to 11.76% 
at time step 49. This confirms that fraud patterns 
escalate dramatically and models need continuous 
retraining.

Finding 3: 141 out of 165 features show 
statistically significant differences between 
illicit and licit transactions (p less than 0.05), 
with effect sizes ranging from negligible to 
very large.

Finding 4: feature_53 is the strongest discriminator 
with Cohen's d of 1.162 and p less than 0.001, 
indicating a very large practical effect size.

Finding 5: All top 10 discriminative features show 
negative mean values for illicit transactions and 
positive mean values for licit transactions, 
consistent with fraudulent wallet isolation 
behavior in the Bitcoin network.

Finding 6: The dataset contains zero missing values 
and is clean and ready for machine learning without 
additional preprocessing.

Finding 7: The dataset exhibits a 9.2 to 1 class 
imbalance. A naive classifier predicting all 
transactions as licit achieves 90.24% accuracy 
while detecting zero fraud cases, confirming that 
AUC-ROC and F1 score are required evaluation 
metrics.

### Phase 1 Extension - Statistical Validation

Mann-Whitney U tests confirm statistically 
significant differences (p less than 0.001) for 
all top discriminative features. Cohen's d effect 
size analysis was performed on all 165 features. 
All 7 findings were independently verified using 
the complete dataset.

## Project Phases

Phase 1 - Baseline analysis on IEEE-CIS dataset 
(completed)

Phase 1 - Exploratory data analysis on Elliptic 
Bitcoin dataset (completed)

Phase 1 Extension - Statistical validation with 
Mann-Whitney U test and Cohen's d (completed)

Phase 2 - Visualization and pattern analysis 
(in progress)

Phase 3 - Machine learning models including 
Logistic Regression, Random Forest, and XGBoost 
(pending)

Phase 4 - SHAP explainability analysis (pending)

Phase 5 - Results write-up and paper draft 
(pending)

## Tools

Python, NumPy, Pandas, Matplotlib, Seaborn, 
SciPy, Scikit-learn, SHAP

## Limitations and Future Work

This project applies tabular machine learning 
to graph-structured blockchain data. Future work 
will explore Graph Neural Networks including 
GraphSAGE and GCN on the transaction network 
structure. Statistical significance testing 
will be extended to include mutual information 
and permutation importance in Phase 3.

## Author

Abdullah Al Adnan
MS Information Technology - Emporia State University
Email: abdullah.adnan112@gmail.com
GitHub: github.com/AbdullahAlAdnan

## Related Publication

This project extends:

Nafiz Eashrak, Mohammad Ikbal Hossain, 
Md. Omum Siddique Auyon, Abdullah Al Adnan. (2026).
Forensic Analysis of Cryptocurrency Transactions: 
Leveraging Blockchain for Fraud Detection and 
Regulatory Compliance. Springer ICTIS 2025.
DOI: 10.1007/978-981-95-1357-4_38

## License

This project is open source and available 
under the MIT License.
