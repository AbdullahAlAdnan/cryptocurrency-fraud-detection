"""
Phase 2: Visualization - Elliptic Bitcoin Dataset
Author: Abdullah Al Adnan
GitHub: github.com/AbdullahAlAdnan/cryptocurrency-fraud-detection
Paper:  DOI 10.1007/978-981-95-1357-4_38

Outputs saved to elliptic_bitcoin/figures/
    fig1_class_distribution.png   (Findings 1 & 7)
    fig2_fraud_over_time.png      (Finding 2)
    fig3_top_features.png         (Findings 3 & 4)
    fig4_kde_feature53.png        (Finding 5)
    fig5_correlation_heatmap.png  (Finding 3)
    fig0_overview.png             (all 5 combined, for README)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib import rcParams

# data location
BASE = r'C:\Users\abdul\Downloads\New folder(p)'

features_file = os.path.join(BASE, 'elliptic_txs_features.csv')
classes_file  = os.path.join(BASE, 'elliptic_txs_classes.csv')

script_dir  = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# publication style: serif font, minimal spines
rcParams['font.family']       = 'serif'
rcParams['font.serif']        = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['axes.spines.top']   = False
rcParams['axes.spines.right'] = False
rcParams['axes.linewidth']    = 0.9
rcParams['xtick.major.size']  = 4
rcParams['ytick.major.size']  = 4
rcParams['xtick.labelsize']   = 10
rcParams['ytick.labelsize']   = 10
rcParams['axes.labelsize']    = 11

# load classes
print("Loading data ...")
classes_raw = pd.read_csv(classes_file, header=None, names=['txId', 'class'])
classes_raw["txId"] = pd.to_numeric(classes_raw["txId"], errors="coerce")
# drop the header row if the CSV accidentally included it (txId would be NaN after coerce)
classes_raw = classes_raw.dropna(subset=["txId"])
classes_raw["txId"] = classes_raw["txId"].astype("int64")
classes_raw['label'] = classes_raw['class'].astype(str).map(
    {'1': 'Illicit', '2': 'Licit', 'unknown': 'Unlabeled'}
)

# load features; col0 = txId, col1 = time_step, remaining = features
feat_raw = pd.read_csv(features_file, header=None)
n_feats  = feat_raw.shape[1] - 2
feat_cols = [f'feature_{i}' for i in range(1, n_feats + 1)]
feat_raw.columns = ['txId', 'time_step'] + feat_cols

labeled      = classes_raw[classes_raw['label'].isin(['Illicit', 'Licit'])]
feat_labeled = feat_raw.merge(labeled[['txId', 'label']], on='txId', how='inner')

illicit_feat = feat_labeled[feat_labeled['label'] == 'Illicit']
licit_feat   = feat_labeled[feat_labeled['label'] == 'Licit']

print(f"  Total transactions : {len(classes_raw):,}")
print(f"  Illicit            : {len(illicit_feat):,}")
print(f"  Licit              : {len(licit_feat):,}")
print(f"  Feature columns    : {len(feat_cols)}")


def cohens_d(a, b):
    """Pooled Cohen's d."""
    na, nb = len(a), len(b)
    pooled_s = np.sqrt(
        ((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2)
        / (na + nb - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled_s if pooled_s > 0 else 0.0


def feature_type(name):
    # local = feature_1 to feature_94, aggregated = feature_95 to feature_165
    # boundary confirmed in Phase 1 statistical analysis
    return 'Aggregated' if int(name.split('_')[1]) >= 95 else 'Local'


def save_fig(fig, fname):
    path = os.path.join(figures_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# chart 1: class distribution (Findings 1 & 7)
# dual panel: left shows all 3 classes, right zooms into labeled only
# so the illicit bar is readable despite the scale difference
def chart1_class_distribution():
    counts    = classes_raw['label'].value_counts()
    illicit   = int(counts.get('Illicit',   0))
    licit     = int(counts.get('Licit',     0))
    unlabeled = int(counts.get('Unlabeled', 0))
    total     = illicit + licit + unlabeled
    labeled   = illicit + licit
    fraud_rate = illicit / labeled * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2),
                                   gridspec_kw={'width_ratios': [1.4, 1]})

    # left panel: all three classes
    cats_all   = ['Illicit', 'Licit', 'Unlabeled']
    vals_all   = [illicit, licit, unlabeled]
    colors_all = ['#C0392B', '#27AE60', '#95A5A6']
    pcts_all   = [v / total * 100 for v in vals_all]

    bars1 = ax1.bar(cats_all, vals_all, color=colors_all,
                    width=0.52, edgecolor='white', linewidth=1.2, zorder=3)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.45, color='#CCCCCC', zorder=0)
    ax1.set_axisbelow(True)

    for bar, count, pct in zip(bars1, vals_all, pcts_all):
        x = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        ax1.text(x, h + total * 0.007, f'{count:,}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        if h > total * 0.02:
            ax1.text(x, h * 0.55, f'{pct:.1f}%',
                     ha='center', va='center', fontsize=10,
                     color='white', fontweight='bold')

    ax1.set_ylabel('Number of Transactions')
    ax1.set_title('All Transactions', fontsize=11, pad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.set_ylim(0, unlabeled * 1.22)

    # right panel: labeled only (illicit vs licit) — shows the 9.2:1 imbalance clearly
    cats_lab   = ['Illicit', 'Licit']
    vals_lab   = [illicit, licit]
    colors_lab = ['#C0392B', '#27AE60']
    pcts_lab   = [v / labeled * 100 for v in vals_lab]

    bars2 = ax2.bar(cats_lab, vals_lab, color=colors_lab,
                    width=0.45, edgecolor='white', linewidth=1.2, zorder=3)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.45, color='#CCCCCC', zorder=0)
    ax2.set_axisbelow(True)

    for bar, count, pct in zip(bars2, vals_lab, pcts_lab):
        x = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        ax2.text(x, h + labeled * 0.012, f'{count:,}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.text(x, h * 0.5, f'{pct:.1f}%',
                 ha='center', va='center', fontsize=11,
                 color='white', fontweight='bold')

    ax2.set_title('Labeled Transactions Only', fontsize=11, pad=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax2.set_ylim(0, licit * 1.28)

    ax2.text(0.97, 0.97,
             f'Fraud rate: {fraud_rate:.2f}%\nImbalance: {licit/illicit:.1f} : 1',
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=9, color='#555555')

    fig.suptitle('Transaction Class Distribution - Elliptic Bitcoin Dataset',
                 fontsize=13, fontweight='bold', y=1.01)

    fig.text(0.5, -0.03,
             'Figure 1. (Left) Full dataset: 203,769 transactions across three classes. '
             '(Right) Labeled subset only: illicit transactions represent 9.76% of '
             'labeled data (Finding 1). The 9.2 : 1 licit-to-illicit imbalance '
             'makes AUC-ROC and F1 the necessary evaluation metrics (Finding 7).',
             ha='center', va='top', fontsize=8.5, style='italic',
             color='#444444', transform=fig.transFigure)
    plt.tight_layout()
    return save_fig(fig, 'fig1_class_distribution.png')


# chart 2: fraud rate over time (Finding 2)
def chart2_fraud_over_time():
    ts = (feat_labeled
          .groupby(['time_step', 'label'])
          .size()
          .unstack(fill_value=0)
          .rename_axis(None, axis=1)
          .reset_index())

    for col in ('Illicit', 'Licit'):
        if col not in ts.columns:
            ts[col] = 0

    ts['total']     = ts['Illicit'] + ts['Licit']
    ts['fraud_pct'] = (ts['Illicit'] / ts['total'] * 100).fillna(0)
    steps = ts['time_step'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5),
                                   sharex=True,
                                   gridspec_kw={'hspace': 0.12})

    ax1.fill_between(steps, ts['Illicit'], alpha=0.18, color='#C0392B')
    ax1.plot(steps, ts['Illicit'], color='#C0392B', lw=2.0, label='Illicit')
    ax1.fill_between(steps, ts['Licit'], alpha=0.10, color='#27AE60')
    ax1.plot(steps, ts['Licit'], color='#27AE60', lw=2.0, label='Licit')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
    ax1.set_axisbelow(True)
    ax1.set_ylabel('Transaction Count')
    ax1.legend(fontsize=9.5, framealpha=0.7, loc='upper left')
    ax1.set_xlim(1, 49)
    ax1.set_title('Fraud Patterns Over Time - Elliptic Bitcoin Dataset',
                  fontsize=13, fontweight='bold', pad=12)

    # raw data as faint line, rolling average as the primary trend line
    ts['rolling'] = ts['fraud_pct'].rolling(window=5, center=True, min_periods=1).mean()
    ax2.plot(steps, ts['fraud_pct'], color='#C0392B', lw=1.0, alpha=0.35,
             linestyle='-', label='Per-step rate')
    ax2.plot(steps, ts['rolling'], color='#C0392B', lw=2.4,
             label='5-step rolling average')

    # horizontal mean line
    ax2.axhline(11.35, color='#555555', lw=1.2, linestyle='--')
    ax2.text(0.01, 11.35 + 0.6, 'Mean: 11.35%',
             transform=ax2.get_yaxis_transform(), fontsize=8.5, color='#555555')

    # peak annotation
    peak_step = int(ts.loc[ts['fraud_pct'].idxmax(), 'time_step'])
    ax2.annotate('Peak: 35.97% (step 13)',
                 xy=(peak_step, 35.97),
                 xytext=(peak_step + 5, 35.97 - 4),
                 fontsize=8.5, color='#7B241C',
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))

    ax2.yaxis.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
    ax2.set_axisbelow(True)
    ax2.set_ylabel('Fraud Rate (%)')
    ax2.set_xlabel('Time Step')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax2.set_xlim(1, 49)

    fig.text(0.5, -0.02,
             'Figure 2. Fraud rate per time step shows high volatility ranging from '
             '0.28% to 35.97% with mean 11.35% and std 9.49%. The absence of a stable '
             'baseline confirms that fraud patterns are unpredictable over time, '
             'supporting the need for continuous model retraining (Finding 2).',
             ha='center', va='top', fontsize=8.5, style='italic', color='#444444')
    fig.tight_layout()
    return save_fig(fig, 'fig2_fraud_over_time.png')


# chart 3: top discriminative features by Cohen's d (Findings 3 & 4)
def chart3_top_features():
    print("  Computing Cohen's d for all features ...")
    records = []
    for col in feat_cols:
        a = illicit_feat[col].dropna().values
        b = licit_feat[col].dropna().values
        records.append({
            'feature'  : col,
            'cohens_d' : abs(cohens_d(a, b)),
            'type'     : feature_type(col)
        })

    effect_df = (pd.DataFrame(records)
                 .sort_values('cohens_d', ascending=False)
                 .head(15)
                 .reset_index(drop=True))

    # reverse order so the highest bar appears at the top
    plot_df = effect_df.iloc[::-1].reset_index(drop=True)
    colors  = ['#2980B9' if t == 'Local' else '#E67E22' for t in plot_df['type']]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.barh(plot_df['feature'], plot_df['cohens_d'],
                   color=colors, edgecolor='white', linewidth=0.8, height=0.6)

    for bar, val in zip(bars, plot_df['cohens_d']):
        ax.text(bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8.5)

    ax.axvline(0.5, color='#888888', linestyle='--', lw=1.0)
    ax.text(0.51, len(plot_df) - 0.1,
            'd = 0.5 threshold  (9 of 165 features exceed this)',
            fontsize=7.5, color='#666666', va='top')

    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.set_xlabel("Cohen's d  (effect size)")
    ax.set_title("Top 15 Discriminative Features - Cohen's d Effect Size",
                 fontsize=13, fontweight='bold', pad=12)

    legend_handles = [
        mpatches.Patch(color='#2980B9', label='Local features (1-94)'),
        mpatches.Patch(color='#E67E22', label='Aggregated features (95-165)'),
        plt.Line2D([0], [0], color='#888888', linestyle='--',
                   lw=1.0, label='d = 0.5 threshold'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right', framealpha=0.8)

    fig.text(0.5, -0.03,
             "Figure 3. Top 15 features ranked by Cohen's d. "
             'feature_53 is the strongest discriminator (d = 0.913, Finding 4). '
             'Only 9 of 165 features demonstrate strong practical separation '
             '(d above 0.5). Local features (blue) show higher average effect '
             'sizes than aggregated features (orange); both contribute '
             'complementary signal (Findings 3 & 4). '
             'Features ranked 10 to 15 are included for context but fall below '
             'the d = 0.5 threshold and do not constitute strong practical discriminators.',
             ha='center', va='top', fontsize=8.5, style='italic', color='#444444')
    plt.tight_layout()
    save_fig(fig, 'fig3_top_features.png')
    return effect_df


# chart 4: KDE of feature_53, illicit vs licit (Finding 5)
def chart4_kde_feature53():
    target = 'feature_53'
    if target not in feat_labeled.columns:
        target = feat_cols[52] if len(feat_cols) > 52 else feat_cols[0]
        print(f'  Note: fell back to {target}')

    ill_vals = illicit_feat[target].dropna().values
    lic_vals = licit_feat[target].dropna().values
    d_val    = abs(cohens_d(ill_vals, lic_vals))

    # clip x to the region of interest; the long right tail of licit adds no signal
    x_min   = max(min(ill_vals.min(), lic_vals.min()), -1.5)
    x_max   = min(max(ill_vals.max(), lic_vals.max()),  4.0)
    x_range = np.linspace(x_min, x_max, 600)
    kde_ill = gaussian_kde(ill_vals)
    kde_lic = gaussian_kde(lic_vals)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.fill_between(x_range, kde_ill(x_range), alpha=0.18, color='#C0392B')
    ax.fill_between(x_range, kde_lic(x_range), alpha=0.14, color='#27AE60')
    ax.plot(x_range, kde_ill(x_range), color='#C0392B', lw=2.2,
            label='Illicit transactions')
    ax.plot(x_range, kde_lic(x_range), color='#27AE60', lw=2.2,
            label='Licit transactions')

    ill_mean = ill_vals.mean()
    lic_mean = lic_vals.mean()
    ax.axvline(ill_mean, color='#C0392B', lw=1.2, linestyle='--',
               label=f'Illicit mean = {ill_mean:.3f}')
    ax.axvline(lic_mean, color='#27AE60', lw=1.2, linestyle='--',
               label=f'Licit mean = {lic_mean:.3f}')
    ax.axvline(0, color='#AAAAAA', lw=0.7, linestyle=':')

    ax.text(0.97, 0.96, f"Cohen's d = {d_val:.3f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            color='#2980B9')

    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.set_xlabel(f'Value of {target}')
    ax.set_ylabel('Density')
    ax.set_title(f'Feature Distribution Comparison - {target}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9.5, framealpha=0.8)

    # let both curves show fully within the clipped x window
    ax.set_xlim(x_min, x_max)

    fig.text(0.5, -0.03,
             f'Figure 4. Kernel density estimate of {target} for illicit vs licit '
             f'transactions. Illicit values concentrate at negative values while '
             f'licit values concentrate at positive values, consistent with '
             f'fraudulent wallet isolation behavior (Finding 5). '
             f"Cohen's d = {d_val:.3f}.",
             ha='center', va='top', fontsize=8.5, style='italic', color='#444444')
    plt.tight_layout()
    return save_fig(fig, 'fig4_kde_feature53.png')


# chart 5: correlation heatmap of top 15 features (Finding 3)
def chart5_correlation_heatmap(effect_df):
    top15 = effect_df.sort_values('cohens_d', ascending=False).head(15)
    cols  = top15['feature'].tolist()
    corr  = feat_labeled[cols].corr()

    # mask upper triangle to avoid redundant pairs
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    fig, ax = plt.subplots(figsize=(9.5, 8))

    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f', annot_kws={'size': 7.5},
        linewidths=0.4, linecolor='#DDDDDD',
        square=True,
        cbar_kws={'label': 'Pearson r', 'shrink': 0.72}
    )

    ax.set_title('Pairwise Correlations - Top 15 Discriminative Features',
                 fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(axis='x', labelsize=8.5, rotation=45)
    ax.tick_params(axis='y', labelsize=8.5, rotation=0)

    fig.text(0.5, -0.02,
             'Figure 5. Pearson correlation matrix for the top 15 features by '
             "Cohen's d. Most feature pairs show low correlation, confirming "
             'complementary and non-redundant signal for ML models (Finding 3). '
             'Notable exception: feature_83 and feature_85 are highly correlated '
             '(r = 0.97) and are candidates for removal during Phase 3 feature selection.',
             ha='center', va='top', fontsize=8.5, style='italic', color='#444444')
    plt.tight_layout()
    return save_fig(fig, 'fig5_correlation_heatmap.png')


# overview: all 5 charts as subplots for the GitHub README
def chart0_overview():
    panels = [
        ('fig1_class_distribution.png',  'Finding 1 & 7\nClass Distribution'),
        ('fig2_fraud_over_time.png',      'Finding 2\nFraud Rate Over Time'),
        ('fig3_top_features.png',         "Finding 3 & 4\nTop Features (Cohen's d)"),
        ('fig4_kde_feature53.png',        'Finding 5\nKDE - feature_53'),
        ('fig5_correlation_heatmap.png',  'Finding 3\nCorrelation Heatmap'),
    ]

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.98,
             'Elliptic Bitcoin Dataset - Phase 2 Visualization Summary',
             ha='center', va='top', fontsize=17, fontweight='bold', color='#1A252F')
    fig.text(0.5, 0.955,
             'Cryptocurrency Fraud Detection  |  Abdullah Al Adnan  |  '
             'DOI: 10.1007/978-981-95-1357-4_38',
             ha='center', va='top', fontsize=10, color='#555555', style='italic')

    # use add_axes for precise row spacing rather than subplots_adjust
    col_w = 0.295
    col_x = [0.02, 0.34, 0.66]
    row_y = [0.48, 0.03]
    row_h = 0.42

    axes = []
    for row in range(2):
        for col in range(3):
            axes.append(fig.add_axes([col_x[col], row_y[row], col_w, row_h]))

    for ax, (fname, label) in zip(axes[:5], panels):
        ax.imshow(mpimg.imread(os.path.join(figures_dir, fname)), aspect='auto')
        ax.axis('off')

    axes[5].axis('off')
    summary = (
        'Key Findings\n'
        '────────────────────────────\n'
        'F1  Fraud rate: 9.76% of labeled txs\n'
        '      4,545 illicit / 46,564 labeled\n\n'
        'F2  Fraud rate highly volatile\n'
        '      0.28% to 35.97%, mean 11.35%\n\n'
        'F3  141 / 165 features significant\n'
        '      (p < 0.05); only 9 with d > 0.5\n\n'
        'F4  feature_53 strongest (d = 0.913)\n'
        '      Local mean d = 0.227\n'
        '      Aggregated mean d = 0.159\n\n'
        'F5  Top-10 features: negative illicit\n'
        '      means -> wallet isolation behavior\n\n'
        'F6  Zero missing values\n\n'
        'F7  9.2 : 1 class imbalance\n'
        '      -> AUC-ROC & F1 required'
    )
    axes[5].text(0.04, 0.97, summary,
                 transform=axes[5].transAxes, va='top', ha='left',
                 fontsize=9.2, fontfamily='monospace', color='#1A252F',
                 linespacing=1.55)

    return save_fig(fig, 'fig0_overview.png')


if __name__ == '__main__':
    print('\nChart 1: Class Distribution')
    chart1_class_distribution()

    print('\nChart 2: Fraud Rate Over Time')
    chart2_fraud_over_time()

    print('\nChart 3: Top Discriminative Features')
    effect_df = chart3_top_features()

    print('\nChart 4: KDE - feature_53')
    chart4_kde_feature53()

    print('\nChart 5: Correlation Heatmap')
    chart5_correlation_heatmap(effect_df)

    print('\nOverview: All 5 combined')
    chart0_overview()

    print(f'\nAll figures saved -> {figures_dir}')