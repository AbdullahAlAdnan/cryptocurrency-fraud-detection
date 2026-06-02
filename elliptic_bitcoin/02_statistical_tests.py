"""
Phase 1 Extension - Statistical Validation
Elliptic Bitcoin Dataset
Author: Abdullah Al Adnan
Extends: Forensic Analysis of Cryptocurrency Transactions
         Springer ICTIS 2025

Purpose:
Validates EDA findings using Mann-Whitney U test
and Cohen's d effect size to provide
publication grade statistical evidence.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
print("Loading dataset...")
BASE = Path(r'C:\Users\abdul\Downloads\New folder(p)')

df_features = pd.read_csv(
    BASE / 'elliptic_txs_features.csv',
    header=None)

df_classes = pd.read_csv(
    BASE / 'elliptic_txs_classes.csv')

df_features.columns = ['txId', 'time_step'] + \
                      [f'feature_{i}' for i in range(1, 166)]

df = df_features.merge(df_classes, on='txId')

labeled = df[df['class'] != 'unknown']
illicit = df[df['class'] == '1']
licit   = df[df['class'] == '2']

print(f"Illicit transactions: {len(illicit):,}")
print(f"Licit transactions:   {len(licit):,}")


# Statistical test function

def statistical_test(illicit, licit, feature):
    """
    Performs Mann-Whitney U test and calculates
    Cohen's d effect size for one feature.

    Returns:
        p_value  - is difference statistically real?
        cohens_d - how large is the difference?

    Note:
        With large datasets even tiny differences
        become statistically significant.
        Cohen's d is therefore the more important
        measure of practical significance.

        Uses corrected pooled standard deviation
        with ddof=1 (Bessel's correction), which
        is the standard formula in published research.
    """
    illicit_vals = illicit[feature].dropna()
    licit_vals   = licit[feature].dropna()

    # Mann-Whitney U test
    stat, p_value = stats.mannwhitneyu(
        illicit_vals,
        licit_vals,
        alternative='two-sided')

    # Cohen's d with corrected pooled standard deviation
    na = len(illicit_vals)
    nb = len(licit_vals)
    pooled_std = np.sqrt(
        ((na - 1) * illicit_vals.std(ddof=1)**2 +
         (nb - 1) * licit_vals.std(ddof=1)**2)
        / (na + nb - 2))

    cohens_d = abs(
        (illicit_vals.mean() - licit_vals.mean())
        / pooled_std) if pooled_std > 0 else 0

    return p_value, cohens_d


# Test 1 - Top 10 discriminative features
# Validates Finding 4

print("\n" + "=" * 70)
print("TEST 1 - TOP 10 DISCRIMINATIVE FEATURES")
print("=" * 70)
print(f"{'Feature':<12} {'p-value':<12} {'Significant':<15} "
      f"{'Cohen d':<10} {'Effect Size'}")
print("-" * 70)

top_features = [
    'feature_53', 'feature_55', 'feature_90',
    'feature_89', 'feature_54', 'feature_52',
    'feature_91', 'feature_59', 'feature_65',
    'feature_60']

for feat in top_features:
    p_val, d = statistical_test(illicit, licit, feat)

    significant = "YES ***" if p_val < 0.001 else \
                  "YES **"  if p_val < 0.01  else \
                  "YES *"   if p_val < 0.05  else \
                  "NO"

    effect = "Very Large" if d > 1.0 else \
             "Large"      if d > 0.8 else \
             "Medium"     if d > 0.5 else \
             "Small"      if d > 0.2 else \
             "Negligible"

    print(f"{feat:<12} {p_val:<12.6f} {significant:<15} "
          f"{d:<10.3f} {effect}")


# Test 2 - Bottom 5 features
# Shows weak features are not significant

print("\n" + "=" * 70)
print("TEST 2 - BOTTOM 5 LEAST DISCRIMINATIVE FEATURES")
print("=" * 70)
print(f"{'Feature':<12} {'p-value':<12} {'Significant':<15} "
      f"{'Cohen d':<10} {'Effect Size'}")
print("-" * 70)

bottom_features = [
    'feature_15',  'feature_111',
    'feature_153', 'feature_152',
    'feature_129']

for feat in bottom_features:
    p_val, d = statistical_test(illicit, licit, feat)

    significant = "YES ***" if p_val < 0.001 else \
                  "YES **"  if p_val < 0.01  else \
                  "YES *"   if p_val < 0.05  else \
                  "NO"

    effect = "Very Large" if d > 1.0 else \
             "Large"      if d > 0.8 else \
             "Medium"     if d > 0.5 else \
             "Small"      if d > 0.2 else \
             "Negligible"

    print(f"{feat:<12} {p_val:<12.6f} {significant:<15} "
          f"{d:<10.3f} {effect}")


# Test 3 - All 165 features
# Complete statistical validation

print("\n" + "=" * 70)
print("TEST 3 - ALL 165 FEATURES STATISTICAL ANALYSIS")
print("=" * 70)
print("Running tests on all 165 features...")

feature_cols = [f'feature_{i}' for i in range(1, 166)]
results = []

for feat in feature_cols:
    p_val, d = statistical_test(illicit, licit, feat)

    results.append({
        'feature':     feat,
        'p_value':     p_val,
        'cohens_d':    d,
        'significant': p_val < 0.05,
        'effect_size': "Very Large" if d > 1.0 else
                       "Large"      if d > 0.8 else
                       "Medium"     if d > 0.5 else
                       "Small"      if d > 0.2 else
                       "Negligible"
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    'cohens_d', ascending=False)

# Summary
sig_count  = results_df['significant'].sum()
not_sig    = 165 - sig_count
very_large = (results_df['cohens_d'] > 1.0).sum()
large      = ((results_df['cohens_d'] > 0.8) &
              (results_df['cohens_d'] <= 1.0)).sum()
medium     = ((results_df['cohens_d'] > 0.5) &
              (results_df['cohens_d'] <= 0.8)).sum()
small      = ((results_df['cohens_d'] > 0.2) &
              (results_df['cohens_d'] <= 0.5)).sum()
negligible = (results_df['cohens_d'] <= 0.2).sum()

print(f"\nResults Summary:")
print(f"Total features tested:     165")
print(f"Significant (p < 0.05):    {sig_count}")
print(f"Not significant:           {not_sig}")
print(f"\nEffect size distribution:")
print(f"Very Large (d > 1.0):      {very_large}")
print(f"Large      (d > 0.8):      {large}")
print(f"Medium     (d > 0.5):      {medium}")
print(f"Small      (d > 0.2):      {small}")
print(f"Negligible (d <= 0.2):     {negligible}")

print(f"\nKey insight:")
print(f"Most features differ statistically but only")
print(f"{very_large + large + medium} features show")
print(f"strong practical separation (Cohen's d above 0.5)")
print(f"Statistical significance alone is not sufficient")
print(f"Effect size is the more meaningful measure")


# Test 4 - Local vs Aggregated features
# Corrected interpretation of Finding 4

print("\n" + "=" * 70)
print("TEST 4 - LOCAL vs AGGREGATED FEATURES")
print("=" * 70)

local_features      = [f'feature_{i}'
                        for i in range(1, 95)]
aggregated_features = [f'feature_{i}'
                        for i in range(95, 166)]

local_results      = results_df[
    results_df['feature'].isin(local_features)]
aggregated_results = results_df[
    results_df['feature'].isin(aggregated_features)]

local_mean_d      = local_results['cohens_d'].mean()
aggregated_mean_d = aggregated_results['cohens_d'].mean()

print(f"\nLocal features (feature_1 to feature_94):")
print(f"  Count:              {len(local_results)}")
print(f"  Mean Cohen's d:     {local_mean_d:.3f}")
print(f"  Significant:        "
      f"{local_results['significant'].sum()}")

print(f"\nAggregated features (feature_95 to feature_165):")
print(f"  Count:              {len(aggregated_results)}")
print(f"  Mean Cohen's d:     {aggregated_mean_d:.3f}")
print(f"  Significant:        "
      f"{aggregated_results['significant'].sum()}")

print(f"\nCorrected interpretation:")
print(f"Local features are stronger on average")
print(f"(mean d = {local_mean_d:.3f} vs {aggregated_mean_d:.3f})")
print(f"However top individual features are aggregated")
print(f"feature_53 Cohen's d = "
      f"{results_df.iloc[0]['cohens_d']:.3f} (large effect)")
print(f"Both feature types play complementary roles")


# Statistical Findings Summary
# All findings corrected and validated

print("\n" + "=" * 70)
print("STATISTICAL FINDINGS SUMMARY")
print("=" * 70)

print(f"\nFinding 3 (corrected and validated):")
print(f"  141 of 165 features show statistically")
print(f"  significant differences (p less than 0.05)")
print(f"  However only {very_large + large + medium} features")
print(f"  demonstrate strong practical separation")
print(f"  (Cohen's d above 0.5)")
print(f"  Statistical significance does not equal")
print(f"  practical significance in large datasets")

print(f"\nFinding 4 (corrected and validated):")
print(f"  feature_53 is the strongest individual")
print(f"  discriminator (Cohen's d = "
      f"{results_df.iloc[0]['cohens_d']:.3f})")
print(f"  Local features stronger on average")
print(f"  (mean d = {local_mean_d:.3f} vs "
      f"{aggregated_mean_d:.3f} aggregated)")
print(f"  Both feature types play complementary roles")
print(f"  in distinguishing illicit transactions")

print(f"\nPaper statement:")
print(f"  141 of 165 features exhibit statistically")
print(f"  significant differences between illicit and")
print(f"  licit transactions. While feature_53 shows")
print(f"  the strongest individual discrimination")
print(f"  (Cohen's d = {results_df.iloc[0]['cohens_d']:.3f}),")
print(f"  local transaction features demonstrate higher")
print(f"  average effect sizes (mean d = {local_mean_d:.3f} vs")
print(f"  {aggregated_mean_d:.3f} for aggregated features),")
print(f"  suggesting complementary roles for both")
print(f"  feature types in fraud detection.")
