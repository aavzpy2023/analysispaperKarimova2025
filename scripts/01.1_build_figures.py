import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import central configuration paths
from paths_config import *

# Define path for t-tests (not explicitly named in paths_config, but located in RESULTS_DIR)
ttests_path = os.path.join(RESULTS_DIR, 'nested_cv_final_results_ttests.csv')

# Configure styling for scientific publications
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

# Define global mapping to standardize labels across figures
mode_labels = {
    'morgan': 'Morgan FP',
    'rdkit2d': 'RDKit 2D',
    'rdkit2d_fp': 'RDKit 2D FP',
    'rdkit2d3d_fp': 'RDKit 2D+3D FP'
}

# Load datasets using paths from paths_config
df_checkpoint = pd.read_csv(CHECKPOINT_FILE)
df_checkpoint['Mode_Clean'] = df_checkpoint['Mode'].map(mode_labels)
df_ttests = pd.read_csv(ttests_path)

# ===================================================================
# FIGURE 1: Model Stability (Boxplot + Swarmplot)
# ===================================================================
plt.figure(figsize=(9, 6))
ax1 = sns.boxplot(data=df_checkpoint, x='Mode_Clean', y='R2_outer',
                  hue='Mode_Clean', palette='pastel', legend=False,
                  showfliers=False, width=0.6)

# Use solid black for higher contrast
sns.stripplot(data=df_checkpoint, x='Mode_Clean', y='R2_outer',
              color='black', alpha=0.7, size=6, jitter=True, ax=ax1)

plt.title('Performance Distribution across Feature Spaces\n(Outer Cross-Validation)', pad=15)
plt.ylabel('$R^2$ (Outer Loop)')
plt.xlabel('Molecular Representation')
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Subtle background grid
plt.tight_layout()

fig1_path = os.path.join(FIGURES_DIR, '01_nested_cv_stability_boxplot.png')
plt.savefig(fig1_path, dpi=300)
plt.close()

# ===================================================================
# FIGURE 2: Performance vs. Literature Benchmarks
# ===================================================================
df_ttest_morgan = df_ttests[df_ttests['Mode'] == 'morgan'].copy()

plt.figure(figsize=(10, 6))
x = range(len(df_ttest_morgan))
width = 0.35

plt.bar([pos - width/2 for pos in x], df_ttest_morgan['Our_R2_mean'], width,
        label='Our model (CV Mean)', color='#4C72B0', edgecolor='black', zorder=3)
plt.bar([pos + width/2 for pos in x], df_ttest_morgan['Paper_R2'], width,
        label='Literature benchmark', color='#DD8452', edgecolor='black', zorder=3)

# Significance asterisks
for i, row in enumerate(df_ttest_morgan.itertuples()):
    if getattr(row, 'Significant_p<0.05', False) or getattr(row, '_9', False):
        # Place asterisk slightly above the taller bar
        y_max = max(row.Our_R2_mean, row.Paper_R2)
        plt.text(i, y_max + 0.01, '*', ha='center', va='bottom', fontsize=20, fontweight='bold', color='black')

plt.xticks(x, df_ttest_morgan['Paper_reference_label'])
plt.ylabel('Performance ($R^2$)')
plt.title('Performance comparison: Our model vs. Literature benchmark', pad=15)
plt.legend(loc='lower right')
plt.ylim(0.6, 0.9)  # Crop Y-axis to highlight differences (adjust based on data)
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)  # Add grid to compare bar heights
plt.tight_layout()

fig2_path = os.path.join(FIGURES_DIR, '01_nested_cv_benchmark_comparison.png')
plt.savefig(fig2_path, dpi=300)
plt.close()

# ===================================================================
# FIGURE 3: Model Selection Frequency (Heatmap)
# ===================================================================
# Calculate selection percentages
selection_counts = df_checkpoint.groupby(['Mode', 'Selected_Model']).size().unstack(fill_value=0)
selection_pct = selection_counts.div(selection_counts.sum(axis=1), axis=0) * 100

# Remove models that were never selected
selection_pct = selection_pct.loc[:, (selection_pct != 0).any(axis=0)]

# ESTHETIC IMPROVEMENT 1: Rename index to formal labels for paper presentation
selection_pct = selection_pct.rename(index=mode_labels)

# ESTHETIC IMPROVEMENT 2: Use np.nan instead of zeros.
# This visually hides zero values while allowing Seaborn to automatically
# determine optimal font color (white or black) based on cell luminance.
selection_pct_masked = selection_pct.replace(0, np.nan)

plt.figure(figsize=(14, 6))

# Draw heatmap
sns.heatmap(selection_pct_masked,
            annot=True,         # Native annotation enabled
            fmt=".1f",          # One decimal place format
            cmap="Blues",
            linewidths=0.5,
            linecolor='lightgray',
            cbar_kws={'label': 'Selection Frequency (%)'})

plt.title('Ensemble Selection Frequency in Inner Loop', pad=20, fontsize=14, fontweight='bold')
plt.ylabel('Feature Space', fontsize=12, fontweight='bold')
plt.xlabel('Selected Model/Ensemble', fontsize=12, fontweight='bold')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()

fig3_path = os.path.join(FIGURES_DIR, '01_nested_cv_model_selection_heatmap.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()

# ===================================================================
# FIGURE 4: Time vs. Performance Trade-off
# ===================================================================
plt.figure(figsize=(10, 6))

# Use cleaned label column for legend
sns.scatterplot(data=df_checkpoint, x='Time_s', y='R2_outer',
                hue='Mode_Clean', style='Mode_Clean', s=100, alpha=0.7, palette='deep')

# Optional: Add group centroid markers
means = df_checkpoint.groupby('Mode_Clean')[['Time_s', 'R2_outer']].mean().reset_index()
sns.scatterplot(data=means, x='Time_s', y='R2_outer',
                hue='Mode_Clean', marker='X', s=300, edgecolor='black',
                linewidth=1.5, legend=False, palette='deep')

plt.title('Computational Trade-off: Time vs. Predictive Accuracy', pad=15)
plt.xlabel('Execution Time per Fold (seconds)')
plt.ylabel('$R^2$ (Outer Loop)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Representation')
plt.tight_layout()

fig4_path = os.path.join(FIGURES_DIR, '01_nested_cv_time_vs_performance.png')
plt.savefig(fig4_path, dpi=300)
plt.close()

print("All figures has been generated. Check the figures folder (./figures)...")
