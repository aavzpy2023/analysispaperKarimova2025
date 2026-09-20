import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from paths_config import DOCKING_RESULTS_CSV, FIGURES_DIR
from adjustText import adjust_text

def plot_docking_validation():
    print(f"[-] Reading data from: {DOCKING_RESULTS_CSV}")

    if not os.path.exists(DOCKING_RESULTS_CSV):
        print(f"[ERROR] CSV file not found: {DOCKING_RESULTS_CSV}")
        return

    df = pd.read_csv(DOCKING_RESULTS_CSV)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.set_theme(style="whitegrid", context="paper")

    # Semantic color mapping for experimental validation state
    custom_palette = {
        'VALIDATED (+)': '#2ca02c',
        'ML FALSE POSITIVE': '#d62728',
        'VALIDATED (-)': '#7f7f7f',
        'ML FALSE NEGATIVE': '#ff7f0e',
        'Neutral': '#1f77b4'
    }

    ml_threshold = 6.0
    dock_threshold = -7.5

    # Set X-axis boundaries with data padding and reverse direction (lower score on right)
    x_max = df['Docking_Score'].max() + 0.5
    x_min = df['Docking_Score'].min() - 0.5
    ax.set_xlim(x_max, x_min)

    # Set Y-axis view window
    ylims = (5.0, 7.8)
    ax.set_ylim(ylims)

    # Shade the consensus active region (upper-right quadrant)
    ax.fill_between([dock_threshold, x_min], ml_threshold, ylims[1],
                    color='#2ca02c', alpha=0.08, zorder=0, label='Consensus Active Zone')

    # Main scatter plot layer
    sns.scatterplot(
        data=df,
        x='Docking_Score',
        y='ML_pIC50',
        hue='Agreement',
        style='Type',
        palette=custom_palette,
        s=110,
        alpha=0.9,
        edgecolor='white',
        linewidth=0.8,
        ax=ax,
        zorder=3
    )

    # Linear regression model line without confidence band
    sns.regplot(
        data=df,
        x='Docking_Score',
        y='ML_pIC50',
        scatter=False,
        ax=ax,
        color='#555555',
        ci=None,
        line_kws={'linestyle': '--', 'alpha': 0.6, 'linewidth': 1.2}
    )

    # Activity classification threshold guidelines
    ax.axhline(ml_threshold, color='black', linestyle=':', alpha=0.5, linewidth=1.2)
    ax.axvline(dock_threshold, color='black', linestyle=':', alpha=0.5, linewidth=1.2)

    # Filter and instantiate annotation text elements
    texts = []
    seen_base_words = set()
    seen_coords = set()

    for _, row in df.iterrows():
        if row['Agreement'] in ['VALIDATED (+)', 'ML FALSE POSITIVE'] or row['Type'] == '[REFERENCE]':

            full_name = str(row['Name']).strip()

            # Deduplicate entries by base compound root
            first_word = full_name.lower().split()[0]
            if first_word in seen_base_words:
                continue

            # Deduplicate entries by spatial coordinates
            coord_key = (round(row['Docking_Score'], 2), round(row['ML_pIC50'], 2))
            if coord_key in seen_coords:
                continue

            seen_base_words.add(first_word)
            seen_coords.add(coord_key)

            # Add initial positional offset to clear marker centers
            texts.append(ax.text(
                row['Docking_Score'] + 0.06,
                row['ML_pIC50'] + 0.06,
                full_name,
                ha='left',
                va='bottom',
                fontsize=7,
                fontweight='bold' if row['Agreement'] == 'VALIDATED (+)' else 'normal',
                alpha=0.9,
                zorder=10
            ))

    # Resolve text overlaps against dataset coordinates
    adjust_text(
        texts,
        x=df['Docking_Score'].tolist(),
        y=df['ML_pIC50'].tolist(),
        force_points=(2.5, 2.0),
        force_text=(1.5, 1.0),
        expand_points=(2.5, 2.5),
        arrowprops=dict(arrowstyle="-", color='#999999', lw=0.6, alpha=0.7)
    )

    # Quadrant indicator annotation box
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85)
    ax.text(dock_threshold - 0.1, ylims[1] - 0.15, 'CONSENSUS ZONE\n(High ML + High Docking)',
            color='#2ca02c', fontsize=8.5, fontweight='bold', bbox=bbox_props, ha='left')

    # Axis titles and labels
    ax.set_title('Consensus Bioactivity: ML pIC50 vs. AutoDock Vina Score', pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel('AutoDock Vina Score (kcal/mol) → [Better Binding]', fontsize=10, fontweight='medium')
    ax.set_ylabel('Predicted ML Bioactivity (pIC50) →', fontsize=10, fontweight='medium')

    # External legend layout configuration
    ax.legend(
        bbox_to_anchor=(1.04, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=8.5,           # Stronger push away from points and crosses
        title_fontsize=9.5,     # Push texts away from each other
        markerscale=0.8,        # Creates a larger invisible boundary to protect the markers
        edgecolor='lightgray'
    )

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, '07_docking_scatter.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[+] Enhanced figure saved successfully at: {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_docking_validation()
