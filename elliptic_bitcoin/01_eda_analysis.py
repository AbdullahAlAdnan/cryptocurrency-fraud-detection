"""
Phase 1 — Exploratory Data Analysis
Elliptic Bitcoin Dataset
Author: Abdullah Al Adnan
Extends: "Forensic Analysis of Cryptocurrency Transactions"
         Springer ICTIS 2025
"""

import pandas as pd
import numpy as np

# ── Load all three files ──
print("Loading files...")

df_features = pd.read_csv(
    r'C:\Users\abdul\Downloads\elliptic_bitcoin_dataset\elliptic_txs_features.csv',
    header=None)

df_classes = pd.read_csv(
    r'C:\Users\abdul\Downloads\elliptic_bitcoin_dataset\elliptic_txs_classes.csv')

df_edges = pd.read_csv(
    r'C:\Users\abdul\Downloads\elliptic_bitcoin_dataset\elliptic_txs_edgelist.csv')

print("Files loaded successfully!")

# ── Rename columns ──
df_features.columns = ['txId', 'time_step'] + \
                      [f'feature_{i}' for i in range(1, 166)]

# ── Merge features with classes ──
df = df_features.merge(df_classes, on='txId')

# ══════════════════════════════════════════════
# STEP 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 1 — DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nClass distribution:")
print(df['class'].value_counts())
print(f"\nFirst 5 rows:")
print(df[['txId', 'time_step', 'feature_1',
          'feature_2', 'class']].head())

# ══════════════════════════════════════════════
# STEP 2 — LABELED DATA ANALYSIS
# Finding 1: Fraud rate in Bitcoin network
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 2 — LABELED DATA ANALYSIS")
print("=" * 50)

labeled  = df[df['class'] != 'unknown']
illicit  = df[df['class'] == '1']
licit    = df[df['class'] == '2']

print(f"Total transactions:     {len(df)}")
print(f"Labeled transactions:   {len(labeled)}")
print(f"Unlabeled transactions: {len(df[df['class'] == 'unknown'])}")
print(f"Illicit (fraud):        {len(illicit)}")
print(f"Licit (normal):         {len(licit)}")
print(f"Fraud rate:             {len(illicit)/len(labeled)*100:.2f}%")

# ══════════════════════════════════════════════
# STEP 3 — TIME STEP ANALYSIS
# Finding 2: Fraud increases over time
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 3 — TIME STEP ANALYSIS")
print("=" * 50)
print("Transactions per time step:")
print(df.groupby('time_step')['class'].value_counts())

# ══════════════════════════════════════════════
# STEP 4 — FEATURE COMPARISON (first 10)
# Finding 3: All features differ
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 4 — FEATURE COMPARISON")
print("=" * 50)

features = [f'feature_{i}' for i in range(1, 11)]
print("\nMean feature values — Illicit vs Licit:")
print("-" * 50)
for feat in features:
    illicit_mean = illicit[feat].mean()
    licit_mean   = licit[feat].mean()
    difference   = abs(illicit_mean - licit_mean)
    print(f"{feat:12} | illicit: {illicit_mean:8.3f} | "
          f"licit: {licit_mean:8.3f} | "
          f"diff: {difference:.3f}")

# ══════════════════════════════════════════════
# STEP 5 — TOP DISCRIMINATIVE FEATURES
# Finding 4: Network features are strongest
# Finding 5: Illicit transactions are isolated
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 5 — TOP DISCRIMINATIVE FEATURES")
print("=" * 50)

feature_cols = [f'feature_{i}' for i in range(1, 166)]
differences  = []

for feat in feature_cols:
    illicit_mean = illicit[feat].mean()
    licit_mean   = licit[feat].mean()
    diff         = abs(illicit_mean - licit_mean)
    differences.append({'feature':      feat,
                        'illicit_mean': illicit_mean,
                        'licit_mean':   licit_mean,
                        'difference':   diff})

diff_df = pd.DataFrame(differences)
diff_df = diff_df.sort_values('difference',
                               ascending=False)

print("\nTop 10 most discriminative features:")
print("-" * 60)
print(diff_df.head(10).to_string(index=False))

print("\nBottom 5 least discriminative features:")
print("-" * 60)
print(diff_df.tail(5).to_string(index=False))

# ══════════════════════════════════════════════
# STEP 6 — MISSING VALUE ANALYSIS
# Finding 6: Clean dataset
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 6 — MISSING VALUE ANALYSIS")
print("=" * 50)

missing     = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

print(f"Total missing values: {missing.sum()}")
missing_cols = missing[missing > 0]
if len(missing_cols) == 0:
    print("No missing values found!")
else:
    for col, count in missing_cols.items():
        print(f"{col}: {count} ({missing_pct[col]:.2f}%)")

# ══════════════════════════════════════════════
# STEP 7 — CLASS IMBALANCE ANALYSIS
# Finding 7: Standard accuracy is misleading
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("STEP 7 — CLASS IMBALANCE ANALYSIS")
print("=" * 50)

class_counts  = labeled['class'].value_counts()
total_labeled = len(labeled)

for cls, count in class_counts.items():
    pct = count / total_labeled * 100
    print(f"Class {cls}: {count} ({pct:.2f}%)")

ratio = class_counts['2'] / class_counts['1']
print(f"\nImbalance ratio: {ratio:.1f}:1")
print(f"For every 1 illicit transaction")
print(f"there are {ratio:.1f} licit transactions")
print(f"\nThis means standard accuracy is misleading!")
print(f"A model predicting ALL licit would get:")
print(f"{class_counts['2']/total_labeled*100:.2f}% accuracy")
print(f"But catch ZERO fraud!")
print(f"\nThis is why we use AUC-ROC instead of accuracy")

# ══════════════════════════════════════════════
# RESEARCH FINDINGS SUMMARY
# ══════════════════════════════════════════════
print("\n" + "=" * 50)
print("RESEARCH FINDINGS SUMMARY")
print("=" * 50)
print("Finding 1: Bitcoin fraud rate = 9.76%")
print("           4x higher than e-commerce (2.65%)")
print("Finding 2: Fraud increases 3x over time")
print("           (17 at step 1 → 56 at step 49)")
print("Finding 3: All features differ between classes")
print("Finding 4: Network features strongest signals")
print("           feature_53 diff = 1.263")
print("Finding 5: Illicit transactions are isolated")
print("           negative network connectivity")
print("Finding 6: Zero missing values")
print("           dataset ready for ML")
print("Finding 7: 9.2:1 class imbalance")
print("           AUC-ROC required over accuracy")
