import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from paths_config import GNN_RESULTS_CSV, GNN_OOF_PREDICTIONS_CSV, FIGURES_DIR, CLASSICAL_R2, CLASSICAL_R2_STD, PAPER_R2

def plot_gnn_benchmark():
    print(f"[-] Reading GNN benchmark results from: {GNN_RESULTS_CSV}")

    if not os.path.exists(GNN_RESULTS_CSV):
        print(f"[ERROR] Results file not found at {GNN_RESULTS_CSV}")
        return

    if not os.path.exists(GNN_OOF_PREDICTIONS_CSV):
        print(f"[ERROR] Out-of-fold predictions file not found at {GNN_OOF_PREDICTIONS_CSV}. "
              f"Re-run 09_gnn_baseline.py (it now exports this file) before generating this figure.")
        return

    df_gnn = pd.read_csv(GNN_RESULTS_CSV)
    df_oof = pd.read_csv(GNN_OOF_PREDICTIONS_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sns.set_theme(style="whitegrid", context="paper")

    # ---------------------------------------------------------
    # PANEL A: Model Comparison (Bar Plot)
    # ---------------------------------------------------------
    labels = list(CLASSICAL_R2.keys())
    means = list(CLASSICAL_R2.values())
    # Real per-mode std from the nested CV results (nested_cv_final_results.csv), not a placeholder.
    errs = [CLASSICAL_R2_STD.get(label, 0.0) for label in labels]

    for _, row in df_gnn.iterrows():
        labels.append(row['Model'])
        means.append(row['R2'])
        errs.append((row['CI_High'] - row['CI_Low']) / 2)

    colors = ['#4C72B0'] * len(CLASSICAL_R2) + ['#C44E52'] * len(df_gnn)

    bars = axes[0].bar(
        np.arange(len(labels)),
        means,
        yerr=errs,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.8,
        error_kw=dict(lw=1.2, capthick=1.2, capsize=5),
        zorder=3
    )

    line_styles = ['--', ':', '-.']
    for idx, (paper_label, (_, val)) in enumerate(PAPER_R2.items()):
        axes[0].axhline(
            val,
            linestyle=line_styles[idx % len(line_styles)],
            color='#333333',
            linewidth=1.2,
            alpha=0.7,
            label=f"Lit: {paper_label} ($R^2={val:.2f}$)",
            zorder=2
        )

    # Annotate numeric values above bars with white background masking
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.012,
            f"{mean:.3f}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=4
        )

    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=15, ha='right', fontsize=9.5)
    axes[0].set_ylabel("$R^2$ Score", fontsize=10, fontweight='medium')
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_title("(A) Predictive Performance ($R^2$) Across Architecture Types", fontsize=11, fontweight='bold', loc='left')
    axes[0].legend(loc='upper left', fontsize=8.5, frameon=True, edgecolor='lightgray', borderpad=0.6)

    # ---------------------------------------------------------
    # PANEL B: ChemProp Scatter Plot (Experimental vs Predicted)
    # ---------------------------------------------------------
    best_gnn = df_gnn.sort_values(by='R2', ascending=False).iloc[0]

    # Real out-of-fold predictions for the best GNN model (from 09_gnn_baseline.py's export).
    y_test = df_oof['y_test'].values
    y_pred = df_oof['y_pred'].values

    axes[1].scatter(
        y_test,
        y_pred,
        alpha=0.75,
        s=38,
        color='#C44E52',
        edgecolor='white',
        linewidth=0.6,
        zorder=3
    )

    mn, mx = min(min(y_test), min(y_pred)) - 0.5, max(max(y_test), max(y_pred)) + 0.5
    axes[1].plot([mn, mx], [mn, mx], 'k--', linewidth=1.2, alpha=0.6, zorder=2)

    axes[1].set_xlim(mn, mx)
    axes[1].set_ylim(mn, mx)
    axes[1].set_aspect('equal', adjustable='box')

    axes[1].set_xlabel("Experimental $\\text{pIC}_{50}$", fontsize=10, fontweight='medium')
    axes[1].set_ylabel("Predicted $\\text{pIC}_{50}$", fontsize=10, fontweight='medium')
    axes[1].set_title(f"(B) {best_gnn['Model']} Goodness-of-Fit", fontsize=11, fontweight='bold', loc='left')

    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="lightgray", lw=1, alpha=0.95)
    stats_text = f"$R^2 = {best_gnn['R2']:.3f}$\n$\\text{{MAE}} = {best_gnn['MAE']:.3f}$"
    axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, fontsize=9.5, verticalalignment='top', bbox=bbox_props, zorder=5)

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, '09_gnn_benchmark_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[+] Enhanced GNN benchmark figure saved at: {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_gnn_benchmark()
