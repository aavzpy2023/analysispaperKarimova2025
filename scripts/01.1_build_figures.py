"""
Figures for publication - v4
Narrative: "When Simple Wins" — Classical ML vs GNNs in low-data QSAR

Fig 1 — R2 distribution per molecular representation (boxplot + reference lines)
Fig 2 — Our model vs Karimova progressive pipeline (horizontal bars)
Fig 3 — Classical ML algorithms: who was selected and how often (NEW)
Fig 4 — Performance summary: R2 and MAE across representations
Fig 5 — Accuracy vs computational cost (scatter)
S1    — Exact ensemble selection per fold (supplementary heatmap)
"""

import os
import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
try:
    from paths_config import *          # noqa: F401,F403
except ImportError:
    RESULTS_DIR    = os.environ.get("RESULTS_DIR",   ".")
    FIGURES_DIR    = os.environ.get("FIGURES_DIR",   "./figures")
    CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "nested_cv_checkpoint.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
TTESTS_PATH = os.path.join(RESULTS_DIR, "nested_cv_final_results_ttests.csv")

# ------------------------------------------------------------------
# Global style
# ------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.15)
mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize":   12,
    "axes.titleweight": "semibold",
    "axes.labelsize":   11,
    "axes.linewidth":   0.9,
    "axes.edgecolor":   "#444444",
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "grid.color":       "#DDDDDD",
    "grid.linewidth":   0.7,
    "legend.frameon":   False,
    "savefig.bbox":     "tight",
    "savefig.dpi":      400,
})

# Representation order (descending mean R2)
MODE_ORDER  = ["morgan", "rdkit2d3d_fp", "rdkit2d_fp", "rdkit2d"]
MODE_LABELS = {
    "morgan":       "Morgan FP",
    "rdkit2d":      "RDKit 2D",
    "rdkit2d_fp":   "RDKit 2D+FP",
    "rdkit2d3d_fp": "RDKit 2D+3D+FP",
}
LABEL_ORDER = [MODE_LABELS[m] for m in MODE_ORDER]

PALETTE = {
    "Morgan FP":      "#3B6BA5",
    "RDKit 2D+3D+FP": "#C4604E",
    "RDKit 2D+FP":    "#4E9B74",
    "RDKit 2D":       "#D98A45",
}

# Karimova (2025) reference stages
STAGE_LABELS = {
    "PaperBaseline": "Karimova et al.\nBaseline",
    "PaperSelected": "Karimova et al.\n+ Feature selection",
    "PaperFinal":    "Karimova et al.\n+ Aug. & DNN ensemble",
}
STAGE_COLORS = {
    "PaperBaseline": "#888888",
    "PaperSelected": "#555555",
    "PaperFinal":    "#222222",
}
LS_MAP = {
    "PaperBaseline": (0, (4, 3)),
    "PaperSelected": "--",
    "PaperFinal":    "-",
}
STAGE_ORDER = ["PaperBaseline", "PaperSelected", "PaperFinal"]

# D-MPNN baseline (reported in this study)
DMPNN_R2  = 0.6866
DMPNN_STD = 0.0573
DMPNN_MAE = 0.5328

# Base learner families (for grouping in Fig 3)
ALGO_FAMILY = {
    "RF":   "Tree ensemble",
    "ET":   "Tree ensemble",
    "XGB":  "Boosting",
    "LGBM": "Boosting",
    "SVM":  "Kernel / lazy",
    "kNN":  "Kernel / lazy",
}
FAMILY_COLORS = {
    "Tree ensemble": "#4E9B74",
    "Boosting":      "#D98A45",
    "Kernel / lazy": "#3B6BA5",
}

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def ci95(values):
    n  = len(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    return stats.t.ppf(0.975, n - 1) * se

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    plt.close(fig)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df = pd.read_csv(CHECKPOINT_FILE)
df["Mode_Clean"] = pd.Categorical(
    df["Mode"].map(MODE_LABELS), categories=LABEL_ORDER, ordered=True)
df["Time_min"]       = df["Time_s"] / 60
df["Ensemble_Size"]  = df["Selected_Model"].str.count(r"\+") + 1

tt = pd.read_csv(TTESTS_PATH)
tt["Mode_Clean"] = pd.Categorical(
    tt["Mode"].map(MODE_LABELS), categories=LABEL_ORDER, ordered=True)

paper_r2 = tt.groupby("Paper_reference_label")["Paper_R2"].first()

# ==================================================================
# FIGURE 1 — R2 distribution per molecular representation
#   Message: Morgan FP matches Karimova baseline without 3D descriptors
# ==================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.8))

sns.boxplot(data=df, x="Mode_Clean", y="R2_outer", order=LABEL_ORDER,
            hue="Mode_Clean", palette=PALETTE, legend=False, dodge=False,
            showfliers=False, width=0.52, linewidth=1.0,
            boxprops=dict(alpha=0.40),
            medianprops=dict(color="#111111", lw=2.0),
            whiskerprops=dict(color="#666666", lw=0.9),
            capprops=dict(color="#666666", lw=0.9),
            ax=ax)

sns.stripplot(data=df, x="Mode_Clean", y="R2_outer", order=LABEL_ORDER,
              hue="Mode_Clean", palette=PALETTE, legend=False,
              alpha=0.80, size=4.5, jitter=0.15,
              linewidth=0.4, edgecolor="white", ax=ax)

means = (df.groupby("Mode_Clean", observed=True)["R2_outer"]
           .mean().reindex(LABEL_ORDER))
ax.scatter(range(len(LABEL_ORDER)), means.values,
           marker="D", s=45, facecolor="white",
           edgecolor="#222222", zorder=6, lw=1.3, label="Mean")

for st in STAGE_ORDER:
    ax.axhline(paper_r2[st], ls=LS_MAP[st],
               color=STAGE_COLORS[st], lw=1.3, zorder=1, alpha=0.85)
    ax.text(len(LABEL_ORDER) - 0.45, paper_r2[st] + 0.005,
            f"{STAGE_LABELS[st].splitlines()[0]}  R²={paper_r2[st]:.2f}",
            fontsize=8.2, color=STAGE_COLORS[st], ha="right", va="bottom")

ax.axhspan(DMPNN_R2 - DMPNN_STD, DMPNN_R2 + DMPNN_STD,
           alpha=0.08, color="#AA3333", zorder=0)
ax.axhline(DMPNN_R2, ls=":", color="#AA3333", lw=1.2, zorder=1)
ax.text(-0.48, DMPNN_R2 - 0.007,
        f"D-MPNN  R²={DMPNN_R2:.3f}",
        fontsize=8.2, color="#AA3333", ha="left", va="top")

ax.set_ylabel(r"$R^2$ (outer fold)")
ax.set_xlabel("")
ax.set_title("Nested CV performance across molecular representations", pad=10)
ax.yaxis.grid(True, linestyle=":", alpha=0.7)
ax.set_axisbelow(True)

sds = (df.groupby("Mode_Clean", observed=True)["R2_outer"]
         .std().reindex(LABEL_ORDER))
xlabels = [f"{lab}\n{means[lab]:.3f} ± {sds[lab]:.3f}"
           for lab in LABEL_ORDER]
ax.set_xticks(range(len(LABEL_ORDER)))
ax.set_xticklabels(xlabels, fontsize=9.2)

h_mean   = mlines.Line2D([], [], marker="D", color="w",
                          markerfacecolor="w", markeredgecolor="#222222",
                          ms=7, label="Mean")
h_median = mlines.Line2D([], [], color="#111111", lw=2, label="Median")
ax.legend(handles=[h_mean, h_median], loc="lower left", fontsize=9)
ax.text(0.99, 0.02, "n = 25 outer folds per representation",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.2, color="#777777")
sns.despine(ax=ax)
save(fig, "01_nested_cv_stability_boxplot")


# ==================================================================
# FIGURE 2 — Our model vs Karimova progressive pipeline
#   Message: simple ensemble matches baseline without augmentation
# ==================================================================
morgan_mean = df[df["Mode"] == "morgan"]["R2_outer"].mean()
morgan_ci   = ci95(df[df["Mode"] == "morgan"]["R2_outer"].values)

models = [
    ("D-MPNN\n(ChemProp, this study)",   DMPNN_R2,    DMPNN_STD / np.sqrt(25), "#C0392B", False),
    ("Our model\n(Morgan FP ensemble)",  morgan_mean, morgan_ci,                "#3B6BA5", True),
    ("Karimova et al.\nBaseline",        0.75,        None,                     "#AAAAAA", False),
    ("Karimova et al.\n+ Feature sel.",  0.82,        None,                     "#777777", False),
    ("Karimova et al.\n+ Aug. & DNN",    0.85,        None,                     "#444444", False),
]

fig, ax = plt.subplots(figsize=(8.0, 4.6))

for i, (label, r2, err, color, highlight) in enumerate(models):
    lw  = 2.0 if highlight else 1.2
    alf = 1.0 if highlight else 0.85

    if err is not None:
        ax.barh(i, r2, height=0.36, color=color, alpha=alf,
                edgecolor="white" if highlight else color,
                linewidth=lw, zorder=3)
        ax.errorbar(r2, i, xerr=err, fmt="none",
                    color="#222222", capsize=4, lw=1.5, zorder=4)
        if label.startswith("Our"):
            row = tt[(tt["Mode"] == "morgan") &
                     (tt["Paper_reference_label"] == "PaperBaseline")].iloc[0]
            s = stars(row["p_value"])
            ax.text(r2 + err + 0.004, i, s,
                    va="center", ha="left", fontsize=10, color="#333333")
    else:
        ax.barh(i, r2, height=0.36, color=color, alpha=alf,
                edgecolor=color, linewidth=lw, zorder=3,
                linestyle="--")

    ax.text(r2 - 0.003, i, f"R²={r2:.3f}",
            va="center", ha="right", fontsize=9.0,
            color="white" if r2 > 0.72 else "#333333",
            fontweight="bold" if highlight else "normal")

ax.set_yticks(range(len(models)))
ax.set_yticklabels([m[0] for m in models], fontsize=9.5)
ax.axhline(1.5, color="#CCCCCC", lw=1.0, ls="--")
ax.text(0.602, 1.52, "Reference study (Karimova et al., 2025)",
        fontsize=8.2, color="#888888", va="bottom")
ax.set_xlabel(r"Mean $R^2$ (cross-validated, 25 outer folds)")
ax.set_title("Predictive performance: this study vs. reference pipeline", pad=10)
ax.set_xlim(0.60, 0.91)
ax.xaxis.grid(True, linestyle=":", alpha=0.7, zorder=0)
ax.set_axisbelow(True)
sns.despine(ax=ax, left=True)
ax.tick_params(axis="y", length=0)
fig.text(0.5, -0.03,
         "Error bars: 95% CI  |  n.s. p≥0.05, * p<0.05, ** p<0.01, *** p<0.001"
         "  (one-sample t-test, Nadeau-Bengio correction, vs. Karimova et al. baseline)",
         ha="center", va="top", fontsize=7.5, color="#888888")
save(fig, "01_nested_cv_benchmark_comparison")


# ==================================================================
# FIGURE 3 — Classical ML algorithms: selection frequency and R2
#   Message: SVM + tree ensembles (ET, LGBM) are the workhorses
#   Two panels:
#     A — base-learner participation (% folds) per representation
#     B — ensemble size distribution
# ==================================================================
ALGOS = ["RF", "ET", "XGB", "LGBM", "SVM", "kNN"]

# Derive present algorithms from data
_found     = {a for s in df["Selected_Model"] for a in s.split("+")}
ALGOS_USED = [a for a in ALGOS if a in _found] + sorted(_found - set(ALGOS))

# Participation matrix (% outer folds)
part = pd.DataFrame(0.0, index=LABEL_ORDER, columns=ALGOS_USED)
for lab, g in df.groupby("Mode_Clean", observed=True):
    c = collections.Counter(a for s in g["Selected_Model"]
                            for a in s.split("+"))
    for a in ALGOS_USED:
        part.loc[lab, a] = 100 * c.get(a, 0) / len(g)

# Ensemble size distribution (%)
size_ct  = (df.groupby(["Mode_Clean", "Ensemble_Size"], observed=True)
              .size().unstack(fill_value=0).reindex(LABEL_ORDER))
size_pct = size_ct.div(size_ct.sum(axis=1), axis=0) * 100

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.2),
                                gridspec_kw={"width_ratios": [1.6, 1],
                                             "wspace": 0.42})

# --- Panel A: heatmap with family color bands on top ---
algo_colors = [FAMILY_COLORS[ALGO_FAMILY.get(a, "Kernel / lazy")]
               for a in ALGOS_USED]

sns.heatmap(part, annot=True, fmt=".0f", cmap="Blues",
            vmin=0, vmax=100,
            linewidths=1.2, linecolor="white",
            annot_kws={"fontsize": 10},
            cbar_kws={"label": "% of outer folds selected",
                      "pad": 0.025, "shrink": 0.85,
                      "ticks": [0, 25, 50, 75, 100]},
            ax=axA)

# Color band above each column (algorithm family)
for j, (a, c) in enumerate(zip(ALGOS_USED, algo_colors)):
    axA.add_patch(plt.Rectangle((j, len(LABEL_ORDER)), 1, 0.45,
                                 color=c, alpha=0.55,
                                 transform=axA.transData, clip_on=False))

# Family legend
for fam, fc in FAMILY_COLORS.items():
    axA.bar(0, 0, color=fc, alpha=0.55, label=fam)
axA.legend(loc="upper left", bbox_to_anchor=(0, -0.22),
           ncol=3, fontsize=8.8, title="Algorithm family", title_fontsize=9)

axA.set_title("A  Base-learner participation in selected ensemble",
              loc="left", pad=10)
axA.set_xlabel("")
axA.set_ylabel("")
axA.tick_params(axis="y", rotation=0, length=0)
axA.tick_params(axis="x", length=0)

# --- Panel B: stacked bar — ensemble size ---
blues = ["#D7E2EE", "#8FAFCE", "#3B6BA5"]
left  = np.zeros(len(size_pct))
for j, col in enumerate(sorted(size_pct.columns)):
    vals = size_pct[col].values
    axB.barh(range(len(size_pct)), vals, left=left,
             height=0.58, color=blues[j % len(blues)],
             edgecolor="white", linewidth=1.2,
             label=f"{col} model{'s' if col > 1 else ''}")
    for i, v in enumerate(vals):
        if v > 6:
            axB.text(left[i] + v / 2, i, f"{v:.0f}%",
                     ha="center", va="center", fontsize=9,
                     color="white" if j == 2 else "#333333")
    left += vals

axB.set_yticks(range(len(size_pct)))
axB.set_yticklabels(size_pct.index, fontsize=9.5)
axB.invert_yaxis()
axB.set_xlim(0, 100)
axB.set_xlabel("% of outer folds")
axB.set_title("B  Ensemble size", loc="left", pad=10)
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, fontsize=9)
sns.despine(ax=axB, left=True)
axB.tick_params(axis="y", length=0)

save(fig, "01_nested_cv_algorithm_selection")


# ==================================================================
# FIGURE 4 — Performance summary: R2 and MAE
# ==================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.6),
                                gridspec_kw={"width_ratios": [1.3, 1]})

for i, lab in enumerate(LABEL_ORDER):
    g = df[df["Mode_Clean"] == lab]
    jitter = np.random.default_rng(42).uniform(-0.13, 0.13, len(g))
    axA.scatter(g["R2_outer"], [i] * len(g) + jitter,
                color=PALETTE[lab], s=22, alpha=0.55, zorder=3,
                edgecolor="white", lw=0.3)
    m  = g["R2_outer"].mean()
    ci = ci95(g["R2_outer"].values)
    axA.errorbar(m, i, xerr=ci, fmt="o", ms=9,
                 color=PALETTE[lab], mec="white", mew=1.4,
                 lw=2.0, capsize=4, zorder=5)

for st in STAGE_ORDER:
    axA.axvline(paper_r2[st], ls=LS_MAP[st],
                color=STAGE_COLORS[st], lw=1.1, alpha=0.7)
axA.axvline(DMPNN_R2, ls=":", color="#AA3333", lw=1.1, alpha=0.7)
axA.text(DMPNN_R2 + 0.002, 3.42, "D-MPNN",
         fontsize=7.8, color="#AA3333", va="top")
axA.text(paper_r2["PaperBaseline"] + 0.002, -0.45,
         "Karimova\nbaseline", fontsize=7.5,
         color=STAGE_COLORS["PaperBaseline"], va="bottom")

axA.set_yticks(range(len(LABEL_ORDER)))
axA.set_yticklabels(LABEL_ORDER, fontsize=9.5)
axA.set_xlabel(r"Mean $R^2$ (outer fold)")
axA.set_title("A  Per-fold R² (mean ± 95% CI)", loc="left", pad=8)
axA.xaxis.grid(True, linestyle=":", alpha=0.7)
axA.set_axisbelow(True)
sns.despine(ax=axA, left=True)
axA.tick_params(axis="y", length=0)

for i, lab in enumerate(LABEL_ORDER):
    g = df[df["Mode_Clean"] == lab]
    m  = g["MAE_outer"].mean()
    ci_mae = ci95(g["MAE_outer"].values)
    jitter = np.random.default_rng(99).uniform(-0.13, 0.13, len(g))
    axB.scatter([i] * len(g) + jitter, g["MAE_outer"],
                color=PALETTE[lab], s=22, alpha=0.55, zorder=3,
                edgecolor="white", lw=0.3)
    axB.errorbar(i, m, yerr=ci_mae, fmt="o", ms=9,
                 color=PALETTE[lab], mec="white", mew=1.4,
                 lw=2.0, capsize=4, zorder=5)
    axB.text(i, m + ci_mae + 0.006, f"{m:.3f}",
             ha="center", va="bottom", fontsize=8.5,
             color=PALETTE[lab], fontweight="semibold")

axB.axhline(DMPNN_MAE, ls=":", color="#AA3333", lw=1.2)
axB.text(len(LABEL_ORDER) - 0.5, DMPNN_MAE + 0.005,
         "D-MPNN MAE", fontsize=7.8, color="#AA3333", ha="right")
axB.set_xticks(range(len(LABEL_ORDER)))
axB.set_xticklabels([l.replace(" ", "\n") for l in LABEL_ORDER], fontsize=8.8)
axB.set_ylabel(r"MAE (outer fold)  [pIC$_{50}$ units]")
axB.set_title("B  Mean absolute error (mean ± 95% CI)", loc="left", pad=8)
axB.yaxis.grid(True, linestyle=":", alpha=0.7)
axB.set_axisbelow(True)
axB.set_ylim(0.42, 0.60)
sns.despine(ax=axB)

fig.suptitle("Performance summary across representations and baselines",
             y=1.02, fontsize=12, fontweight="semibold")
save(fig, "01_nested_cv_performance_summary")


# ==================================================================
# FIGURE 5 — Accuracy vs computational cost
#   Message: Morgan FP is on the Pareto front — best trade-off
# ==================================================================
LABEL_OFFSETS = {
    "Morgan FP":      (0,   26, "center"),
    "RDKit 2D":       (0,   26, "center"),
    "RDKit 2D+FP":    (-12, -34, "right"),
    "RDKit 2D+3D+FP": (14,  26, "left"),
}

fig, ax = plt.subplots(figsize=(7.8, 4.8))

for lab in LABEL_ORDER:
    g = df[df["Mode_Clean"] == lab]
    ax.scatter(g["Time_min"], g["R2_outer"],
               s=20, alpha=0.22, color=PALETTE[lab],
               linewidth=0, zorder=2)
    mx, my = g["Time_min"].mean(), g["R2_outer"].mean()
    sx, sy = g["Time_min"].std(),  g["R2_outer"].std()
    ax.errorbar(mx, my, xerr=sx, yerr=sy,
                fmt="o", ms=11, color=PALETTE[lab],
                mec="white", mew=1.6, lw=1.8, capsize=3, zorder=5)
    dx, dy, ha = LABEL_OFFSETS[lab]
    ax.annotate(lab, (mx, my),
                textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=9.5, fontweight="semibold",
                color=PALETTE[lab], zorder=6)

# Pareto front
cent = (df.groupby("Mode_Clean", observed=True)
          .agg(t=("Time_min", "mean"), r=("R2_outer", "mean"))
          .sort_values("t"))
front, best = [], -np.inf
for lab, row in cent.iterrows():
    if row["r"] > best:
        front.append((row["t"], row["r"]))
        best = row["r"]
if len(front) > 1:
    ax.plot(*zip(*front), ls="--", lw=1.2, color="#AAAAAA", zorder=1)
    mx_f = (front[0][0] + front[-1][0]) / 2
    my_f = (front[0][1] + front[-1][1]) / 2
    ax.text(mx_f, my_f - 0.025, "Pareto front",
            fontsize=8.2, color="#AAAAAA", ha="center", style="italic")

ax.axhline(DMPNN_R2, ls=":", color="#AA3333", lw=1.2, alpha=0.8)
ax.text(5.5, DMPNN_R2 - 0.012,
        f"D-MPNN  R²={DMPNN_R2:.3f}",
        fontsize=8.2, color="#AA3333", ha="left", va="top")

ax.set_xlabel("Runtime per outer fold (min)")
ax.set_ylabel(r"$R^2$ (outer fold)")
ax.set_title("Accuracy vs. computational cost  (mean ± SD)", pad=10)
ax.grid(True, linestyle=":", alpha=0.7)
ax.set_axisbelow(True)
sns.despine(ax=ax)
save(fig, "01_nested_cv_time_vs_performance")


# ==================================================================
# S1 — Exact ensemble selection per fold (supplementary)
# ==================================================================
sel = (df.groupby(["Mode_Clean", "Selected_Model"], observed=True)
         .size().unstack(fill_value=0).reindex(LABEL_ORDER))
sel = sel.loc[:, sel.sum().sort_values(ascending=False).index]
sel_masked = sel.replace(0, np.nan)

fig, ax = plt.subplots(figsize=(14, 3.6))
sns.heatmap(sel_masked, annot=True, fmt=".0f", cmap="Blues",
            linewidths=1.2, linecolor="white",
            annot_kws={"fontsize": 9},
            cbar_kws={"label": f"Folds selected (out of {int(sel.sum(axis=1).max())})",
                      "pad": 0.01, "shrink": 0.85}, ax=ax)
ax.set_title("Exact ensemble selected per outer fold — supplementary",
             loc="left", pad=10)
ax.set_xlabel("Selected ensemble")
ax.set_ylabel("")
ax.tick_params(axis="y", rotation=0, length=0)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
save(fig, "S1_nested_cv_selection_full_heatmap")

print("All figures generated in:", FIGURES_DIR)
