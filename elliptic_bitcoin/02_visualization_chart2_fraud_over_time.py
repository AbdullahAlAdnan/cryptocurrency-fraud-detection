"""
Phase 2 — Visualization | Chart 2: Fraud Over Time
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

df      = df_features.merge(df_classes, on='txId')
illicit = df[df['class'] == '1']
licit   = df[df['class'] == '2']

print(f"Loaded {len(df):,} transactions")

# ══════════════════════════════════════════════
# CHART 2 — Fraud Over Time
# Finding 2: Fraud increases over time
# with high volatility and periodic spikes
# ══════════════════════════════════════════════

# aggregate counts per time step
illicit_ts = illicit.groupby('time_step').size().reset_index(name='illicit')
licit_ts   = licit.groupby('time_step').size().reset_index(name='licit')
time_df    = illicit_ts.merge(licit_ts, on='time_step', how='outer').fillna(0)
time_df    = time_df.sort_values('time_step').reset_index(drop=True)

# key values for annotations
first_step = time_df['time_step'].min()
last_step  = time_df['time_step'].max()
first_val  = int(time_df.loc[time_df['time_step'] == first_step, 'illicit'].values[0])
last_val   = int(time_df.loc[time_df['time_step'] == last_step,  'illicit'].values[0])
peak_val   = int(time_df['illicit'].max())
peak_step  = int(time_df.loc[time_df['illicit'].idxmax(), 'time_step'])

fig, ax1 = plt.subplots(figsize=(10, 5))

# ── illicit line on left axis ──
ax1.plot(time_df['time_step'], time_df['illicit'],
         color='#c0392b', linewidth=2,
         marker='o', markersize=3.5, label='Illicit (fraud)')
ax1.fill_between(time_df['time_step'], time_df['illicit'],
                 color='#c0392b', alpha=0.10)
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Illicit Transactions', color='#c0392b')
ax1.tick_params(axis='y', labelcolor='#c0392b', direction='out')
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# ── licit line on right axis ──
ax2 = ax1.twinx()
ax2.plot(time_df['time_step'], time_df['licit'],
         color='#2980b9', linewidth=2,
         marker='s', markersize=3.5,
         linestyle='--', label='Licit (normal)')
ax2.set_ylabel('Licit Transactions', color='#2980b9')
ax2.tick_params(axis='y', labelcolor='#2980b9', direction='out')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(True)
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# ── annotate peak illicit step ──
ax1.annotate(f'Peak: {peak_val} illicit\n(step {peak_step})',
             xy=(peak_step, peak_val),
             xytext=(peak_step + 3, peak_val - 60),
             fontsize=8.5, color='#c0392b',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.0),
             bbox=dict(boxstyle='round,pad=0.3',
                       fc='#fdf2f2', ec='#c0392b', alpha=0.85))

# ── annotate start vs end ──
ax1.text(2, ax1.get_ylim()[1] * 0.88,
         f'Step {first_step}: {first_val} illicit  →  '
         f'Step {last_step}: {last_val} illicit',
         fontsize=8.5, color='#c0392b',
         bbox=dict(boxstyle='round,pad=0.3',
                   fc='#fdf2f2', ec='#c0392b', alpha=0.85))

# ── combined legend ──
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper right', fontsize=9, framealpha=0.7)

ax1.set_title('Illicit vs Licit Transaction Volume Across 49 Time Steps')

fig.text(0.5, -0.03,
         f'Figure 2. Illicit transactions show high volatility with a peak of '
         f'{peak_val} at step {peak_step}. '
         f'Dual axis used due to scale difference between classes.',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()

save_path = OUTPUT_DIR / 'chart2_fraud_over_time.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()
plt.close()

print(f"Chart 2 saved → {save_path}")
