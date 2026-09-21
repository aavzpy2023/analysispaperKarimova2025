import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import RESULTS_DIR, FIGURES_DIR, CSV_04_OUTPUT_FILE

CSV_FILE = os.path.join(RESULTS_DIR, CSV_04_OUTPUT_FILE)
FIG_BASE = os.path.join(FIGURES_DIR, "04_augment_r2_comparison")

# Reference values reported in the original paper (DNN pipeline)
PAPER_R2 = {
    'Paper baseline': 0.75,
    'Paper + feat. sel.': 0.82,
    'Paper final (DNN)': 0.85,
}
REF_STYLE = [(':', '#7F7F7F'), ('--', '#7F7F7F'), ('-', '#333333')]

# Colour-blind-safe palette (Okabe-Ito), one colour per experiment
COLORS = ['#0072B2', '#009E73', '#E69F00']

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans'})


def generate_figure():
    if not os.path.exists(CSV_FILE):
        print(f"Error: file not found: {CSV_FILE}. Run 04_augmentation_training.py first.")
        return
    df = pd.read_csv(CSV_FILE).set_index('Experiment')
    keys = ['Exp_A', 'Exp_B', 'Exp_C']
    r2 = df.loc[keys, 'R2'].values
    lo = df.loc[keys, 'CI_Lower'].values
    hi = df.loc[keys, 'CI_Upper'].values
    mae = df.loc[keys, 'MAE'].values
    nf = df.loc[keys, 'N_Features'].astype(int).values
    ntr = df.loc[keys, 'Aug_Size'].astype(int).values

    names = ['A: Baseline', 'B: + Feature\nselection', 'C: + Gaussian\naugmentation']
    xt = [f"{n}\n{f:,} feat. | n$_{{train}}$={t:,}" for n, f, t in zip(names, nf, ntr)]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(3)

    # Reference lines (paper) with direct labels instead of a legend
    for (lab, val), (ls, c) in zip(PAPER_R2.items(), REF_STYLE):
        ax.axhline(val, ls=ls, lw=1.2, color=c, zorder=1)
        ax.text(2.56, val, f"{lab} ({val:.2f})", va='center', ha='left',
                fontsize=8, color=c)

    # Point estimate + 95% CI (no truncated bars)
    for i in range(3):
        ax.errorbar(x[i], r2[i], yerr=[[r2[i] - lo[i]], [hi[i] - r2[i]]],
                    fmt='o', ms=9, capsize=6, lw=2, color=COLORS[i],
                    mec='white', mew=1, zorder=3)
        ax.text(x[i] + 0.09, r2[i], f"R² = {r2[i]:.3f}\nMAE = {mae[i]:.3f}",
                va='center', ha='left', fontsize=8.5, color=COLORS[i], zorder=4)
        ax.text(x[i], lo[i] - 0.012, f"[{lo[i]:.3f}, {hi[i]:.3f}]",
                va='top', ha='center', fontsize=7.5, color='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=8.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.60, 0.92)
    ax.set_ylabel("R² on held-out test set (95% bootstrap CI)")
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    # widen right margin for reference labels
    fig.subplots_adjust(left=0.11, right=0.80, bottom=0.2, top=0.96)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f"{FIG_BASE}.{ext}", dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {FIG_BASE}.png / .pdf")


if __name__ == "__main__":
    generate_figure()
