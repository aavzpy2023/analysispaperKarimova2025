import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Link root directory to import paths_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import RESULTS_DIR, FIGURES_DIR

# =========================================================
# CONFIGURATION
# =========================================================
# Input and output paths
CSV_FILE = os.path.join(RESULTS_DIR, 'FDA_Candidates_For_Docking.csv')

# Global style configuration for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

def generate_plots():
    # Load data
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] File not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    print("Generating plots...")

    # Ensure output directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =========================================================
    # PLOT 1: Bar Chart (pIC50 Comparison)
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    # Sort data from lowest to highest so the best is at the top
    df_sorted = df.sort_values('pIC50_pred', ascending=True)

    # Assign colors based on type
    colors = df_sorted['Type'].map({'[TOP_ML]': '#2b5c8f', '[REFERENCE]': '#d95f02'})

    bars = ax1.barh(df_sorted['Name'], df_sorted['pIC50_pred'], color=colors, edgecolor='black', alpha=0.85)

    ax1.set_xlabel('Predicted $pIC_{50}$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Compound Name', fontsize=12, fontweight='bold')
    ax1.set_title('Virtual Screening Candidates vs Reference Controls ($pIC_{50}$)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(4.5, 8.0)

    # Add numerical values at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                va='center', ha='left', fontsize=9, fontweight='semibold')

    # Custom legend
    legend_elements = [
        Patch(facecolor='#2b5c8f', edgecolor='black', label='Top ML Candidates'),
        Patch(facecolor='#d95f02', edgecolor='black', label='Reference Controls')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    out_file1 = os.path.join(FIGURES_DIR, '05_pIC50_candidates_comparison.png')
    plt.savefig(out_file1, dpi=300)
    plt.close(fig1)
    print(f"[SUCCESS] Saved: {out_file1}")

    # =========================================================
    # PLOT 2: Scatter Plot (pIC50 vs LE)
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x='LE',
        y='pIC50_pred',
        hue='Type',
        palette={'[TOP_ML]': '#2b5c8f', '[REFERENCE]': '#d95f02'},
        s=120,
        style='Type',
        markers={'[TOP_ML]': 'o', '[REFERENCE]': 's'},
        ax=ax2
    )

    # Label key compounds to avoid cluttering the plot
    key_compounds = ['Methotrexate', 'Pyrimethamine', 'Methylene Blue', 'Triamterene', 'Pralatrexate', 'Trimethoprim']
    for i, row in df.iterrows():
        if row['Name'] in key_compounds:
            ax2.annotate(
                row['Name'],
                (row['LE'], row['pIC50_pred']),
                textcoords="offset points",
                xytext=(5, 5),
                ha='left',
                fontsize=9,
                fontweight='bold' if row['Type'] == '[TOP_ML]' else 'normal'
            )

    ax2.set_xlabel('Ligand Efficiency (LE)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted $pIC_{50}$', fontsize=12, fontweight='bold')
    ax2.set_title('Predictive Affinity vs. Ligand Efficiency (LE)', fontsize=14, fontweight='bold', pad=15)

    # Threshold lines
    ax2.axhline(6.0, color='gray', linestyle='--', alpha=0.7, label='pIC50 threshold = 6.0')
    ax2.axvline(0.3, color='green', linestyle=':', alpha=0.7, label='Optimal LE threshold = 0.30')

    # LEYENDA AJUSTADA AQUÍ (texto más pequeño y marcadores reducidos)
    ax2.legend(loc='upper right', frameon=True, fontsize=8, markerscale=0.8)

    plt.tight_layout()
    out_file2 = os.path.join(FIGURES_DIR, '05_pIC50_vs_LE_scatter.png')
    plt.savefig(out_file2, dpi=300)
    plt.close(fig2)
    print(f"[SUCCESS] Saved: {out_file2}")

    # =========================================================
    # PLOT 3: Heatmap (Efficiency Metrics)
    # =========================================================
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    # Prepare dataframe for heatmap (only relevant numerical columns indexed by name)
    metrics_df = df.set_index('Name')[['pIC50_pred', 'LE', 'BEI', 'LLE', 'SEI']]

    sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Metric Value'}, ax=ax3)

    ax3.set_title('Multi-Metric Profile of Screening Candidates and References', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel('Compound Name', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_file3 = os.path.join(FIGURES_DIR, '05_candidates_metrics_heatmap.png')
    plt.savefig(out_file3, dpi=300)
    plt.close(fig3)
    print(f"[SUCCESS] Saved: {out_file3}")

if __name__ == "__main__":
    generate_plots()
