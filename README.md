# Cryptocurrency Fraud Detection

## Motivation
This project empirically implements the AI-driven forensic 
framework proposed in my published Springer paper:

**"Forensic Analysis of Cryptocurrency Transactions: 
Leveraging Blockchain for Fraud Detection and Regulatory 
Compliance"** — ICTIS 2025, Springer.
DOI: 10.1007/978-981-95-1357-4_38

## Dataset
IEEE-CIS Fraud Detection Dataset (Kaggle)
- 590,540 transactions
- 394 features
- Real-world e-commerce transaction data

## Research Questions
1. Which features best distinguish fraud from normal?
2. How does class imbalance affect detection?
3. Which ML model performs best for fraud detection?

## Key Finding So Far
Simple statistical threshold (2*std) catches only 
5.3% of fraud cases — demonstrating that amount alone 
is insufficient for fraud detection and confirming the 
need for multi-feature ML approaches.

## Project Phases
- [x] Phase 1 — NumPy baseline (IEEE-CIS dataset)
- [x] Phase 1 — EDA on Elliptic Bitcoin Dataset
      - 7 key research findings documented
      - Fraud rate: 9.76%
      - Network features identified as strongest signals
      - Class imbalance confirmed (9.2:1 ratio)
- [ ] Phase 2 — Visualization
- [ ] Phase 3 — ML Models
- [ ] Phase 4 — Results write-up

## Tools
Python, NumPy, Pandas, Matplotlib, Scikit-learn

## Author
Abdullah Al Adnan
MS Information Technology — Emporia State University
[GitHub](https://github.com/AbdullahAlAdnan)
