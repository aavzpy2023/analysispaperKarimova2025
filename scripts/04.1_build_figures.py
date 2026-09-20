import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Link root directory to import paths_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import RESULTS_DIR, FIGURES_DIR, CSV_04_OUTPUT_FILE

# =========================================================
# CONFIGURATION
# =========================================================
CSV_FILE = os.path.join(RESULTS_DIR, CSV_04_OUTPUT_FILE)
FIGURE_FILE = os.path.join(FIGURES_DIR, "04_augment_r2_comparison.png")

PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP, no feature selection', 0.75),
    'PaperSelected': ('After Permutation Importance selection', 0.82),
    'PaperFinal': ('Data augmentation + DNN ensemble', 0.85),
}

# =========================================================
# FIGURE GENERATION
# =========================================================
def generate_figure():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: File not found: {CSV_FILE}. Please run 04_augmentation_training.py first.")
        return

    df = pd.read_csv(CSV_FILE)
    df = df.set_index('Experiment')

    labels = ['Exp A\n(ML baseline)', 'Exp B\n(+feat. select.)', 'Exp C\n(+augmentation)']

    means = [df.loc['Exp_A', 'R2'], df.loc['Exp_B', 'R2'], df.loc['Exp_C', 'R2']]
    ci_lo = [df.loc['Exp_A', 'CI_Lower'], df.loc['Exp_B', 'CI_Lower'], df.loc['Exp_C', 'CI_Lower']]
    ci_hi = [df.loc['Exp_A', 'CI_Upper'], df.loc['Exp_B', 'CI_Upper'], df.loc['Exp_C', 'CI_Upper']]

    errs = [[m - l for m, l in zip(means, ci_lo)],
            [h - m for m, h in zip(means, ci_hi)]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=6, color=['#4C72B0', '#55A868', '#C44E52'], alpha=0.85)

    for label, (desc, val) in PAPER_R2.items():
        ls = '--' if 'Final' not in label else '-'
        ax.axhline(val, linestyle=ls, linewidth=1.2, alpha=0.7, label=f"{label} ({val})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² (bootstrap mean ± 95% CI)")
    ax.set_title("Classical ML: Effect of Feature Selection and Data Augmentation")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(FIGURE_FILE), exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.close(fig)
    print(f"✅ Figure successfully saved to: {FIGURE_FILE}")

if __name__ == "__main__":
    generate_figure()
