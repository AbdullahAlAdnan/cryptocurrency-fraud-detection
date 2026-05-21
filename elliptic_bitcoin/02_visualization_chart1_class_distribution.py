"""
Phase 2 — Visualization | Chart 1: Class Distribution
Elliptic Bitcoin Dataset
Author: Abdullah Al Adnan
Extends: "Forensic Analysis of Cryptocurrency Transactions"
         Springer ICTIS 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# output folder
OUTPUT_DIR = Path(r'C:\Users\abdul\Downloads\New folder(p)\PythonProject\figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# consistent academic style across all charts
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})

# ── load data ──
print("Loading dataset...")

BASE = Path(r'C:\Users\abdul\Downloads\New folder(p)')

df_features = pd.read_csv(BASE / 'elliptic_txs_features.csv', header=None)
df_classes  = pd.read_csv(BASE / 'elliptic_txs_classes.csv')

df_features.columns = ['txId', 'time_step'] + \
                      [f'feature_{i}' for i in range(1, 166)]

df = df_features.merge(df_classes, on='txId')

labeled = df[df['class'] != 'unknown']
illicit = df[df['class'] == '1']
licit   = df[df['class'] == '2']
unknown = df[df['class'] == 'unknown']

print(f"Loaded {len(df):,} transactions")

# ══════════════════════════════════════════════
# CHART 1 — Class Distribution
# Finding 1: Fraud rate = 9.76% among labeled
# Finding 7: 9.2:1 class imbalance
# ══════════════════════════════════════════════

labels      = ['Illicit\n(Fraud)', 'Licit\n(Normal)', 'Unlabeled']
values      = [len(illicit), len(licit), len(unknown)]
colors      = ['#c0392b', '#2980b9', '#95a5a6']

fraud_rate  = len(illicit) / len(labeled) * 100
illicit_pct = len(illicit) / len(df) * 100
licit_pct   = len(licit)   / len(df) * 100
unknown_pct = len(unknown) / len(df) * 100
percentages = [illicit_pct, licit_pct, unknown_pct]

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.bar(labels, values, color=colors,
              width=0.5, edgecolor='white', linewidth=0.8)

# value + percentage label above each bar
for bar, pct in zip(bars, percentages):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h + 1500,
            f'{h:,}\n({pct:.1f}%)',
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', linespacing=1.4)

# fraud rate — plain text box on the illicit bar, no arrow crossing other bars
ax.text(0, max(values) * 0.60,
        f'Fraud rate: {fraud_rate:.2f}%\namong Elliptic\ninvestigated transactions',
        ha='center', va='bottom',
        fontsize=9.2, color='#c0392b',
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.45',
                  fc='#fdf2f2', ec='#c0392b', alpha=0.95))

ax.set_title('Transaction Class Distribution in the Elliptic Bitcoin Dataset')
ax.set_ylabel('Number of Transactions')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.set_ylim(0, max(values) * 1.22)

fig.text(0.5, -0.02,
         'Figure 1. Class distribution across 203,769 transactions. '
         'Labeled data shows a 9.2:1 licit-to-illicit imbalance.',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()

save_path = OUTPUT_DIR / 'chart1_class_distribution.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()
plt.close()

print(f"Chart 1 saved → {save_path}")
