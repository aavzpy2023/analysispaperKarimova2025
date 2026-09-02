import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import warnings
from paths_config import *
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, RegressorMixin, clone
import lightgbm as lgb
from xgboost import XGBRegressor

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
TRAIN_FILE       = TRAIN_FILE
RANDOM_STATE     = 42
N_JOBS           = 44
N_ESTIMATORS     = 200
TEST_SIZE        = 0.15          # identical to paper
N_ENSEMBLE_RUNS  = 5             # identical to paper (5 runs, average predictions)
N_BOOTSTRAP      = 2000
LATEX_FILE       = LATEX_AUGMENT
FIGURE_FILE      = FIGURE_AUGMENT

# Gaussian noise levels — identical to paper (Section 3.5)
NOISE_LEVELS     = [0.01, 0.001]

# Paper reference values
PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP, no feature selection',              0.75),
    'PaperSelected': ('After Permutation Importance feature selection', 0.82),
    'PaperFinal':    ('Data augmentation + DNN ensemble',             0.85),
}

# Winner from 0STACK_nested_cv_v3.py — update if your nested CV produced
# a different winner. This is the architecture used for all experiments here.
# Format: list of (name, estimator) tuples for StackingRegressor.
# If the winner was a single model (e.g. 'SVM'), set STACK_MEMBERS = None
# and SINGLE_MODEL to the corresponding estimator.
STACK_MEMBERS = [
    ('rf',   RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1)),
    ('xgb',  XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1, verbosity=0)),
    ('svm',  SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)),
]
# ^^^ EDIT THIS to reflect your actual nested CV winner before running.


# =========================================================
# MOLECULAR REPRESENTATION  (Morgan FP — best mode from nested CV)
# =========================================================
_MORGAN_GEN = None

def _get_morgan_generator():
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _MORGAN_GEN

def get_morgan_fp(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return np.zeros((2048,), dtype=np.int8)
        return _get_morgan_generator().GetFingerprintAsNumPy(m).astype(np.int8)
    except Exception:
        return np.zeros((2048,), dtype=np.int8)

def build_X(smiles_list):
    return np.array([get_morgan_fp(s) for s in smiles_list], dtype=float)


# =========================================================
# MODEL BUILDER
# =========================================================
def build_model():
    """Returns a fresh clone of the winning architecture."""
    estimators = [(n, clone(m)) for n, m in STACK_MEMBERS]
    if len(estimators) == 1:
        return clone(estimators[0][1])
    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=1,
    )


# =========================================================
# PERMUTATION IMPORTANCE FEATURE SELECTION
# Mirrors the paper's Section 3.4 exactly:
#   - shuffle each feature, measure drop in neg-MSE
#   - n_repeats=5
#   - keep only features with mean importance > 0
# =========================================================
def select_features_by_permutation_importance(X, y, model, n_repeats=5):
    """
    Trains model on (X, y), computes Permutation Importance,
    returns the boolean mask of features with mean importance > 0.
    Identical protocol to the paper (Section 3.4).
    """
    print("  [PI] Fitting model for Permutation Importance calculation...")
    model.fit(X, y)
    print("  [PI] Computing Permutation Importance (n_repeats={})...".format(n_repeats))
    result = permutation_importance(
        model, X, y,
        scoring='neg_mean_squared_error',
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    mask = result.importances_mean > 0
    n_selected = mask.sum()
    print(f"  [PI] Features selected (importance > 0): {n_selected}/{X.shape[1]}")
    return mask, result


# =========================================================
# GAUSSIAN AUGMENTATION
# Mirrors the paper's Section 3.5 exactly:
#   - noise added to FEATURES only, not to target y
#   - two noise levels (sigma=0.01 and sigma=0.001)
#   - creates 2 augmented copies per data point → 3x dataset
#   - test set is held out BEFORE augmentation (real data only)
# =========================================================
def augment_data(X_train, y_train, noise_levels=None, seed=RANDOM_STATE):
    """
    Returns augmented (X_aug, y_aug) where each original sample
    appears once unperturbed plus once per noise level.
    The test set must already be separated before calling this.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    rng = np.random.RandomState(seed)
    X_parts = [X_train]
    y_parts = [y_train]
    for sigma in noise_levels:
        noise = rng.normal(0, sigma, X_train.shape)
        X_parts.append(X_train + noise)
        y_parts.append(y_train)
    X_aug = np.vstack(X_parts)
    y_aug = np.concatenate(y_parts)
    print(f"  [AUG] Dataset size: {len(y_train)} -> {len(y_aug)} "
          f"(noise levels: {noise_levels})")
    return X_aug, y_aug


# =========================================================
# BOOTSTRAP CI
# =========================================================
def bootstrap_r2_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=0.95):
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    boots = [r2_score(y_true[idx := rng.randint(0, n, n)],
                      y_pred[idx]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(1-ci)/2*100, (1+ci)/2*100])
    return float(np.mean(boots)), float(lo), float(hi)


# =========================================================
# ENSEMBLE PREDICTION (5 runs, average — mirrors paper Section 3.3)
# =========================================================
def ensemble_predict(X_train, y_train, X_test, n_runs=N_ENSEMBLE_RUNS):
    """Train n_runs independent models, return mean prediction."""
    all_preds = []
    for run in range(n_runs):
        m = build_model()
        # vary random state per run for diversity, same as paper's stochastic ensemble
        if hasattr(m, 'random_state'):
            m.set_params(random_state=RANDOM_STATE + run)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        if pred.ndim > 1:
            pred = pred.flatten()
        all_preds.append(pred)
    return np.mean(all_preds, axis=0)


# =========================================================
# LATEX EXPORT
# =========================================================
def newcommand(f, name, value):
    f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")

def export_latex(results_dict):
    with open(LATEX_FILE, 'w', encoding='utf-8') as f:
        f.write("% =====================================================\n")
        f.write("% Variables auto-generated by 1AUGMENT.py\n")
        f.write("% Include with: \\input{augment_variables.tex}\n")
        f.write("% =====================================================\n\n")

        f.write("% --- Experiment A: no feature selection, no augmentation ---\n")
        a = results_dict['A']
        newcommand(f, "ExpARTwoMean",   f"{a['r2']:.4f}")
        newcommand(f, "ExpAMae",        f"{a['mae']:.4f}")
        newcommand(f, "ExpARTwoCILow",  f"{a['ci_lo']:.4f}")
        newcommand(f, "ExpARTwoCIHigh", f"{a['ci_hi']:.4f}")
        f.write("\n")

        f.write("% --- Experiment B: Permutation Importance feature selection ---\n")
        b = results_dict['B']
        newcommand(f, "ExpBRTwoMean",        f"{b['r2']:.4f}")
        newcommand(f, "ExpBMae",             f"{b['mae']:.4f}")
        newcommand(f, "ExpBRTwoCILow",       f"{b['ci_lo']:.4f}")
        newcommand(f, "ExpBRTwoCIHigh",      f"{b['ci_hi']:.4f}")
        newcommand(f, "ExpBNFeaturesTotal",  str(b['n_features_total']))
        newcommand(f, "ExpBNFeaturesSelected", str(b['n_features_selected']))
        f.write("\n")

        f.write("% --- Experiment C: feature selection + Gaussian augmentation ---\n")
        c = results_dict['C']
        newcommand(f, "ExpCRTwoMean",     f"{c['r2']:.4f}")
        newcommand(f, "ExpCMae",          f"{c['mae']:.4f}")
        newcommand(f, "ExpCRTwoCILow",    f"{c['ci_lo']:.4f}")
        newcommand(f, "ExpCRTwoCIHigh",   f"{c['ci_hi']:.4f}")
        newcommand(f, "ExpCAugFactor",    str(1 + len(NOISE_LEVELS)))
        newcommand(f, "ExpCAugSizeTrain", str(c['aug_size']))
        newcommand(f, "ExpCNoiseLevels",  str(NOISE_LEVELS).replace('[','').replace(']',''))
        f.write("\n")

        f.write("% --- Paper references ---\n")
        for label, (_, val) in PAPER_R2.items():
            newcommand(f, f"{label}RTwo", f"{val:.2f}")
        f.write("\n")

        f.write("% --- Statistical tests (one-sample t vs paper) ---\n")
        for exp_key in ('A', 'B', 'C'):
            exp = results_dict[exp_key]
            for paper_key, (_, paper_val) in PAPER_R2.items():
                # one-sample t-test: compare bootstrap distribution vs paper value
                t, p = stats.ttest_1samp(exp['boot_r2s'], paper_val)
                newcommand(f, f"Exp{exp_key}Vs{paper_key}T",   f"{t:.3f}")
                newcommand(f, f"Exp{exp_key}Vs{paper_key}P",   f"{p:.4f}")
                sig = "true" if p < 0.05 else "false"
                newcommand(f, f"Exp{exp_key}Vs{paper_key}Sig", sig)

    print(f"\n[LATEX] Variables exported to {LATEX_FILE}")


# =========================================================
# FIGURE
# =========================================================
def export_figure(results_dict):
    labels = ['Exp A\n(ML baseline)', 'Exp B\n(+feat. select.)', 'Exp C\n(+augmentation)']
    means  = [results_dict[k]['r2'] for k in ('A', 'B', 'C')]
    ci_lo  = [results_dict[k]['ci_lo'] for k in ('A', 'B', 'C')]
    ci_hi  = [results_dict[k]['ci_hi'] for k in ('A', 'B', 'C')]
    errs   = [[m - l for m, l in zip(means, ci_lo)],
              [h - m for m, h in zip(means, ci_hi)]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=6, color=['#4C72B0','#55A868','#C44E52'], alpha=0.85)

    for label, (desc, val) in PAPER_R2.items():
        ls = '--' if 'Final' not in label else '-'
        ax.axhline(val, linestyle=ls, linewidth=1.2, alpha=0.7, label=f"{label} ({val})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² (bootstrap mean ± 95% CI)")
    ax.set_title("Classical ML: effect of feature selection and data augmentation")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.close(fig)
    print(f"[FIGURE] Saved {FIGURE_FILE}")


# =========================================================
# MAIN
# =========================================================
def run():
    print("=" * 100)
    print("1AUGMENT.py — Feature Selection + Gaussian Augmentation Benchmark")
    print("Mirrors the paper pipeline (Sections 3.4 and 3.5) for classical ML")
    print("=" * 100)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    print(f"\n[DATA] {len(df)} compounds loaded.")

    X_all = build_X(df['Smiles'].tolist())
    y_all = df['pIC50 Value'].values

    # ── Split: 15% test held-out BEFORE any augmentation (identical to paper) ──
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[SPLIT] Dev={len(y_dev)} | Test held-out={len(y_test)} (never used for selection)")

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A — ML baseline (no feature selection, no augmentation)
    # Reproduces the single-split result from 0STACK with the winner model
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 100)
    print("EXPERIMENT A: ML stacking baseline (no feature selection, no augmentation)")
    print("─" * 100)
    t0 = time.time()
    y_pred_a = ensemble_predict(X_dev, y_dev, X_test)
    r2_a  = r2_score(y_test, y_pred_a)
    mae_a = mean_absolute_error(y_test, y_pred_a)
    boot_mean_a, ci_lo_a, ci_hi_a = bootstrap_r2_ci(y_test, y_pred_a)

    # Store bootstrap samples for t-test
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_test)
    boot_r2s_a = [r2_score(y_test[idx := rng.randint(0,n,n)], y_pred_a[idx])
                  for _ in range(N_BOOTSTRAP)]

    results['A'] = dict(r2=r2_a, mae=mae_a, ci_lo=ci_lo_a, ci_hi=ci_hi_a,
                        boot_r2s=boot_r2s_a, n_features_total=X_all.shape[1],
                        n_features_selected=X_all.shape[1], aug_size=len(y_dev))
    print(f"  R2={r2_a:.4f} | MAE={mae_a:.4f} | 95%CI [{ci_lo_a:.4f}, {ci_hi_a:.4f}] | "
          f"{time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B — Feature selection via Permutation Importance
    # Mirrors paper Section 3.4: keep features with mean PI > 0
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 100)
    print("EXPERIMENT B: + Permutation Importance feature selection (mirrors paper Section 3.4)")
    print("─" * 100)
    t0 = time.time()
    pi_model = build_model()
    mask_b, pi_result = select_features_by_permutation_importance(X_dev, y_dev, pi_model)

    X_dev_b  = X_dev[:, mask_b]
    X_test_b = X_test[:, mask_b]

    y_pred_b = ensemble_predict(X_dev_b, y_dev, X_test_b)
    r2_b  = r2_score(y_test, y_pred_b)
    mae_b = mean_absolute_error(y_test, y_pred_b)
    boot_mean_b, ci_lo_b, ci_hi_b = bootstrap_r2_ci(y_test, y_pred_b)

    rng = np.random.RandomState(RANDOM_STATE)
    boot_r2s_b = [r2_score(y_test[idx := rng.randint(0,n,n)], y_pred_b[idx])
                  for _ in range(N_BOOTSTRAP)]

    results['B'] = dict(r2=r2_b, mae=mae_b, ci_lo=ci_lo_b, ci_hi=ci_hi_b,
                        boot_r2s=boot_r2s_b, n_features_total=X_all.shape[1],
                        n_features_selected=int(mask_b.sum()), aug_size=len(y_dev))
    print(f"  R2={r2_b:.4f} | MAE={mae_b:.4f} | 95%CI [{ci_lo_b:.4f}, {ci_hi_b:.4f}] | "
          f"{time.time()-t0:.1f}s")
    print(f"  Paper analogue: R2=0.82 after PI selection")

    # Save selected feature mask for downstream use (2FDA.py needs it)
    np.save(MASK_FILE, mask_b)
    print("  [SAVED] selected_features_mask.npy")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT C — Feature selection + Gaussian augmentation
    # Mirrors paper Section 3.5 exactly:
    #   - holdout 15% test BEFORE augmentation (already done above)
    #   - augment only the dev/train split
    #   - evaluate on REAL test set only
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 100)
    print("EXPERIMENT C: + Gaussian augmentation (mirrors paper Section 3.5)")
    print("─" * 100)
    t0 = time.time()

    X_aug, y_aug = augment_data(X_dev_b, y_dev)

    y_pred_c = ensemble_predict(X_aug, y_aug, X_test_b)
    r2_c  = r2_score(y_test, y_pred_c)
    mae_c = mean_absolute_error(y_test, y_pred_c)
    boot_mean_c, ci_lo_c, ci_hi_c = bootstrap_r2_ci(y_test, y_pred_c)

    rng = np.random.RandomState(RANDOM_STATE)
    boot_r2s_c = [r2_score(y_test[idx := rng.randint(0,n,n)], y_pred_c[idx])
                  for _ in range(N_BOOTSTRAP)]

    results['C'] = dict(r2=r2_c, mae=mae_c, ci_lo=ci_lo_c, ci_hi=ci_hi_c,
                        boot_r2s=boot_r2s_c, n_features_total=X_all.shape[1],
                        n_features_selected=int(mask_b.sum()), aug_size=len(y_aug))
    print(f"  R2={r2_c:.4f} | MAE={mae_c:.4f} | 95%CI [{ci_lo_c:.4f}, {ci_hi_c:.4f}] | "
          f"{time.time()-t0:.1f}s")
    print(f"  Paper analogue: R2=0.85 after augmentation + DNN ensemble")

    # ══════════════════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY — read top to bottom to draft Results/Discussion")
    print("=" * 100)

    for exp_key, exp_label in [('A','Exp A (baseline)'), ('B','Exp B (+feat.sel.)'), ('C','Exp C (+augment.)')]:
        exp = results[exp_key]
        print(f"\n{exp_label}: R2={exp['r2']:.4f} | MAE={exp['mae']:.4f} | "
              f"95%CI [{exp['ci_lo']:.4f}, {exp['ci_hi']:.4f}]")
        for paper_key, (desc, paper_val) in PAPER_R2.items():
            t, p = stats.ttest_1samp(exp['boot_r2s'], paper_val)
            sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "NOT significant (p>=0.05)"
            direction = "above" if exp['r2'] > paper_val else "below"
            print(f"  vs {desc} (R2={paper_val}): {direction} by {abs(exp['r2']-paper_val):.4f} "
                  f"| t={t:.3f}, p={p:.4f} -> {sig}")

    print("\n" + "─" * 100)
    print("KEY FINDING FOR THE DISCUSSION:")
    c = results['C']
    a = results['A']

    # Compare A vs C
    delta_ac = c['r2'] - a['r2']
    t_ac, p_ac = stats.ttest_rel(c['boot_r2s'], a['boot_r2s'])
    print(f"  Augmentation gain (Exp A -> Exp C): delta R2={delta_ac:+.4f} "
          f"(paired t={t_ac:.3f}, p={p_ac:.4f})")

    # Compare C vs paper final
    t_cp, p_cp = stats.ttest_1samp(c['boot_r2s'], 0.85)
    sig_cp = "NOT significantly different from" if p_cp >= 0.05 else "significantly different from"
    print(f"  Exp C vs paper final (R2=0.85): {sig_cp} (t={t_cp:.3f}, p={p_cp:.4f})")

    if p_cp >= 0.05:
        print("\n  CANDIDATE SENTENCE (Discussion):")
        print(f'  "Applying identical Gaussian noise augmentation (sigma=0.01/0.001) to the '
              f'classical stacking ensemble after Permutation Importance feature selection '
              f'yielded R2={c["r2"]:.3f} (95%CI [{c["ci_lo"]:.3f}, {c["ci_hi"]:.3f}]), '
              f'statistically indistinguishable from the DNN ensemble reported in the '
              f'original study (R2=0.85, p={p_cp:.3f}). This demonstrates that the '
              f'performance gain is attributable to the data augmentation strategy rather '
              f'than the deep learning architecture."')
    else:
        print("\n  CANDIDATE SENTENCE (Discussion):")
        print(f'  "Despite applying identical augmentation, the classical stacking ensemble '
              f'reached R2={c["r2"]:.3f} vs the DNN\'s R2=0.85 (p={p_cp:.4f}), suggesting '
              f'that architectural factors beyond data volume contribute to the DNN advantage '
              f'for this dataset size."')

    print("=" * 100)

    # ── Save best model + mask for 2FDA.py ────────────────────────────────────
    print("\n[SAVE] Fitting final model on full dev set (augmented) for 2FDA.py...")
    final_model = build_model()
    final_model.fit(X_aug, y_aug)
    import joblib
    joblib.dump(final_model, MODEL_FILE)
    print("[SAVE] best_model.joblib saved (used by 2FDA.py)")

    export_latex(results)
    export_figure(results)
    print("\n[DONE] 1AUGMENT.py complete.")


if __name__ == "__main__":
    run()