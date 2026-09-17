import os
# =========================================================
# PARALELISMO DESBLOQUEADO
# Las restricciones de un solo hilo han sido comentadas para
# permitir que las librerías matemáticas usen múltiples núcleos.
# =========================================================
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.base import clone
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# Project internal paths
from paths_config import (
    TRAIN_FILE, MODEL_FILE, MASK_FILE,
    LATEX_AUGMENT, FIGURE_AUGMENT
)

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION & PARAMETERS
# =========================================================
RANDOM_STATE     = 42
N_JOBS           = 44          # Utiliza 44 de los 48 núcleos disponibles para paralelismo
N_ESTIMATORS     = 200
TEST_SIZE        = 0.15          # 15% holdout test set (identical to paper)
N_ENSEMBLE_RUNS  = 5             # 5 stochastic ensemble runs (averaged)
N_BOOTSTRAP      = 2000          # Resampling iterations for non-parametric CIs
LATEX_FILE       = LATEX_AUGMENT
FIGURE_FILE      = FIGURE_AUGMENT

# Gaussian noise perturbation levels (Section 3.5)
NOISE_LEVELS     = [0.01, 0.001]

# Reference benchmarks reported in literature
PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP, no feature selection',              0.75),
    'PaperSelected': ('After Permutation Importance feature selection', 0.82),
    'PaperFinal':    ('Data augmentation + DNN ensemble',             0.85),
}

# =========================================================
# MODEL BASE ESTIMATORS (MULTITHREADING ENABLED)
# =========================================================
# Base estimators for the Stacking Regressor (winning topology from nested CV)
# N_JOBS is injected into Random Forest and XGBoost. SVR runs sequentially by nature.
STACK_MEMBERS = [
    ('rf',   RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
    ('xgb',  XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)),
    ('svm',  SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)),
]


# =========================================================
# MOLECULAR REPRESENTATIONS
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
    """Returns a fresh instance of the winning Stacking Regressor architecture."""
    estimators = [(n, clone(m)) for n, m in STACK_MEMBERS]
    if len(estimators) == 1:
        return clone(estimators[0][1])

    # CRÍTICO: El meta-modelo (StackingRegressor) MANTIENE n_jobs=1.
    # Si se pone N_JOBS aquí, multiplicará hilos (N_JOBS * N_JOBS) y congelará el servidor.
    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=1,
    )


# =========================================================
# PERMUTATION IMPORTANCE FEATURE SELECTION
# =========================================================
def select_features_by_permutation_importance(X, y, model, n_repeats=5):
    """
    Computes permutation feature importance on training split.
    Retains features with mean importance > 0 (Section 3.4 protocol).
    """
    print("  [PI] Fitting model for Permutation Importance calculation...")
    model.fit(X, y)
    print(f"  [PI] Computing Permutation Importance (n_repeats={n_repeats}, n_jobs={N_JOBS})...")

    # Aceleración principal: Se inyecta N_JOBS al cálculo de permutación
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
# =========================================================
def augment_data(X_train, y_train, noise_levels=None, seed=RANDOM_STATE):
    """
    Applies Gaussian noise to training feature vectors (Section 3.5).
    Original sample + 1 copy per noise level -> 3x dataset expansion.
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
    print(f"  [AUG] Dataset expansion: {len(y_train)} -> {len(y_aug)} "
          f"(noise levels: {noise_levels})")
    return X_aug, y_aug


# =========================================================
# BOOTSTRAP STATISTICS & EMPIRICAL P-VALUES
# =========================================================
def bootstrap_r2_distribution(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Generates non-parametric bootstrap sampling distribution of R2 scores."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boots.append(r2_score(y_true[idx], y_pred[idx]))
    return np.array(boots)

def compute_bootstrap_metrics(boot_array, ci=0.95):
    """Calculates bootstrap mean and percentile-based confidence intervals."""
    mean_val = float(np.mean(boot_array))
    lo, hi = np.percentile(boot_array, [(1-ci)/2*100, (1+ci)/2*100])
    return mean_val, float(lo), float(hi)

def calculate_empirical_bootstrap_pvalue(boot_array, reference_val):
    """
    Calculates two-tailed empirical bootstrap p-value against a point reference value.
    Avoids parametric t-test pseudoreplication artifacts on bootstrap draws.
    """
    prop_below = np.mean(boot_array <= reference_val)
    prop_above = np.mean(boot_array >= reference_val)
    p_emp = 2.0 * min(prop_below, prop_above)
    return float(min(1.0, p_emp))


# =========================================================
# ENSEMBLE PREDICTION
# =========================================================
def ensemble_predict(X_train, y_train, X_test, n_runs=N_ENSEMBLE_RUNS):
    """Trains n_runs independent stochastic models and returns averaged predictions."""
    all_preds = []
    for run in range(n_runs):
        m = build_model()
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
        f.write("% Variables auto-generated by 04_augmentation_training.py\n")
        f.write("% Includes rigorous bootstrap CIs and non-parametric p-values\n")
        f.write("% =====================================================\n\n")

        f.write("% --- Experiment A: Baseline Stacking (No Feat Sel, No Aug) ---\n")
        a = results_dict['A']
        newcommand(f, "ExpARTwoMean",   f"{a['r2']:.4f}")
        newcommand(f, "ExpAMae",        f"{a['mae']:.4f}")
        newcommand(f, "ExpARTwoCILow",  f"{a['ci_lo']:.4f}")
        newcommand(f, "ExpARTwoCIHigh", f"{a['ci_hi']:.4f}")
        f.write("\n")

        f.write("% --- Experiment B: Permutation Importance Selection ---\n")
        b = results_dict['B']
        newcommand(f, "ExpBRTwoMean",        f"{b['r2']:.4f}")
        newcommand(f, "ExpBMae",             f"{b['mae']:.4f}")
        newcommand(f, "ExpBRTwoCILow",       f"{b['ci_lo']:.4f}")
        newcommand(f, "ExpBRTwoCIHigh",      f"{b['ci_hi']:.4f}")
        newcommand(f, "ExpBNFeaturesTotal",  str(b['n_features_total']))
        newcommand(f, "ExpBNFeaturesSelected", str(b['n_features_selected']))
        f.write("\n")

        f.write("% --- Experiment C: Feature Selection + Gaussian Augmentation ---\n")
        c = results_dict['C']
        newcommand(f, "ExpCRTwoMean",     f"{c['r2']:.4f}")
        newcommand(f, "ExpCMae",          f"{c['mae']:.4f}")
        newcommand(f, "ExpCRTwoCILow",    f"{c['ci_lo']:.4f}")
        newcommand(f, "ExpCRTwoCIHigh",   f"{c['ci_hi']:.4f}")
        newcommand(f, "ExpCAugFactor",    str(1 + len(NOISE_LEVELS)))
        newcommand(f, "ExpCAugSizeTrain", str(c['aug_size']))
        newcommand(f, "ExpCNoiseLevels",  str(NOISE_LEVELS).replace('[','').replace(']',''))
        f.write("\n")

        f.write("% --- Reference Benchmarks ---\n")
        for label, (_, val) in PAPER_R2.items():
            f.write(f"\\providecommand{{\\{label}RTwo}}{{{val:.2f}}}\n")
        f.write("\n")

        f.write("% --- Rigorous Statistical Evaluations vs Benchmarks ---\n")
        for exp_key in ('A', 'B', 'C'):
            exp = results_dict[exp_key]
            for paper_key, (_, paper_val) in PAPER_R2.items():
                is_within_ci = exp['ci_lo'] <= paper_val <= exp['ci_hi']
                p_boot = calculate_empirical_bootstrap_pvalue(exp['boot_r2s'], paper_val)
                sig = "false" if is_within_ci else "true"

                newcommand(f, f"Exp{exp_key}Vs{paper_key}InCI", "true" if is_within_ci else "false")
                newcommand(f, f"Exp{exp_key}Vs{paper_key}PBoot", f"{p_boot:.4f}")
                newcommand(f, f"Exp{exp_key}Vs{paper_key}Sig",   sig)

    print(f"\n[LATEX] Variables successfully exported to {LATEX_FILE}")


# =========================================================
# FIGURE GENERATION
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
    ax.set_title("Classical ML: Effect of Feature Selection and Data Augmentation")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.close(fig)
    print(f"[FIGURE] Saved figure to {FIGURE_FILE}")


# =========================================================
# MAIN EXECUTION PIPELINE
# =========================================================
def run():
    print("=" * 100)
    print("04_augmentation_training.py — Feature Selection & Gaussian Augmentation Benchmark")
    print("Evaluates classical ML pipeline parity against reported DNN benchmarks")
    print("=" * 100)

    # 1. Load Data
    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    print(f"\n[DATA] {len(df)} compounds successfully loaded.")

    X_all = build_X(df['Smiles'].tolist())
    y_all = df['pIC50 Value'].values

    # 2. Train / Test Split (15% test held-out BEFORE selection/augmentation)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[SPLIT] Dev={len(y_dev)} | Test held-out={len(y_test)} (isolated from tuning/augmentation)")

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT A: Baseline Stacking Ensemble (No Feature Selection, No Augmentation)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("EXPERIMENT A: ML Stacking Baseline")
    print("─" * 100)
    t0 = time.time()
    y_pred_a = ensemble_predict(X_dev, y_dev, X_test)
    r2_a  = r2_score(y_test, y_pred_a)
    mae_a = mean_absolute_error(y_test, y_pred_a)

    boot_r2s_a = bootstrap_r2_distribution(y_test, y_pred_a)
    boot_mean_a, ci_lo_a, ci_hi_a = compute_bootstrap_metrics(boot_r2s_a)

    results['A'] = dict(r2=r2_a, mae=mae_a, ci_lo=ci_lo_a, ci_hi=ci_hi_a,
                        boot_r2s=boot_r2s_a, n_features_total=X_all.shape[1],
                        n_features_selected=X_all.shape[1], aug_size=len(y_dev))
    print(f"  R2={r2_a:.4f} | MAE={mae_a:.4f} | 95% CI [{ci_lo_a:.4f}, {ci_hi_a:.4f}] | "
          f"Time: {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT B: Permutation Importance Feature Selection
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("EXPERIMENT B: + Permutation Importance Feature Selection")
    print("─" * 100)
    t0 = time.time()
    pi_model = build_model()
    mask_b, pi_result = select_features_by_permutation_importance(X_dev, y_dev, pi_model)

    X_dev_b  = X_dev[:, mask_b]
    X_test_b = X_test[:, mask_b]

    y_pred_b = ensemble_predict(X_dev_b, y_dev, X_test_b)
    r2_b  = r2_score(y_test, y_pred_b)
    mae_b = mean_absolute_error(y_test, y_pred_b)

    boot_r2s_b = bootstrap_r2_distribution(y_test, y_pred_b)
    boot_mean_b, ci_lo_b, ci_hi_b = compute_bootstrap_metrics(boot_r2s_b)

    results['B'] = dict(r2=r2_b, mae=mae_b, ci_lo=ci_lo_b, ci_hi=ci_hi_b,
                        boot_r2s=boot_r2s_b, n_features_total=X_all.shape[1],
                        n_features_selected=int(mask_b.sum()), aug_size=len(y_dev))
    print(f"  R2={r2_b:.4f} | MAE={mae_b:.4f} | 95% CI [{ci_lo_b:.4f}, {ci_hi_b:.4f}] | "
          f"Time: {time.time()-t0:.1f}s")

    np.save(MASK_FILE, mask_b)
    print(f"  [SAVED] {MASK_FILE}")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT C: Feature Selection + Gaussian Augmentation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("EXPERIMENT C: + Gaussian Data Augmentation")
    print("─" * 100)
    t0 = time.time()

    X_aug, y_aug = augment_data(X_dev_b, y_dev)

    y_pred_c = ensemble_predict(X_aug, y_aug, X_test_b)
    r2_c  = r2_score(y_test, y_pred_c)
    mae_c = mean_absolute_error(y_test, y_pred_c)

    boot_r2s_c = bootstrap_r2_distribution(y_test, y_pred_c)
    boot_mean_c, ci_lo_c, ci_hi_c = compute_bootstrap_metrics(boot_r2s_c)

    results['C'] = dict(r2=r2_c, mae=mae_c, ci_lo=ci_lo_c, ci_hi=ci_hi_c,
                        boot_r2s=boot_r2s_c, n_features_total=X_all.shape[1],
                        n_features_selected=int(mask_b.sum()), aug_size=len(y_aug))
    print(f"  R2={r2_c:.4f} | MAE={mae_c:.4f} | 95% CI [{ci_lo_c:.4f}, {ci_hi_c:.4f}] | "
          f"Time: {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # EXECUTIVE SUMMARY & RIGOROUS HYPOTHESIS TESTING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY — RIGOROUS STATISTICAL AUDIT")
    print("=" * 100)

    for exp_key, exp_label in [('A','Exp A (baseline)'), ('B','Exp B (+feat.sel.)'), ('C','Exp C (+augment.)')]:
        exp = results[exp_key]
        print(f"\n{exp_label}: R2={exp['r2']:.4f} | MAE={exp['mae']:.4f} | "
              f"95% CI [{exp['ci_lo']:.4f}, {exp['ci_hi']:.4f}]")
        for paper_key, (desc, paper_val) in PAPER_R2.items():
            in_ci = exp['ci_lo'] <= paper_val <= exp['ci_hi']
            p_boot = calculate_empirical_bootstrap_pvalue(exp['boot_r2s'], paper_val)
            sig_text = "NOT significantly different (within 95% CI)" if in_ci else "SIGNIFICANTLY different"
            direction = "above" if exp['r2'] > paper_val else "below"
            print(f"  vs {desc} (R2={paper_val}): {direction} by {abs(exp['r2']-paper_val):.4f} "
                  f"| p_boot={p_boot:.4f} -> {sig_text}")

    # Paired Bootstrap Difference Test: Exp A vs Exp C
    diff_ca_boots = results['C']['boot_r2s'] - results['A']['boot_r2s']
    diff_mean, diff_lo, diff_hi = compute_bootstrap_metrics(diff_ca_boots)
    p_diff = calculate_empirical_bootstrap_pvalue(diff_ca_boots, 0.0)

    print("\n" + "─" * 100)
    print("EVALUATION OF DATA AUGMENTATION EFFECT (Exp C vs Exp A):")
    print(f"  Delta R2 (C - A): {diff_mean:+.4f} (95% CI [{diff_lo:+.4f}, {diff_hi:+.4f}], p_boot={p_diff:.4f})")

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL PERSISTENCE FOR VIRTUAL SCREENING (05_virtual_screening.py)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("PERSISTENCE AUDIT — Matching Section 2.5 Protocol")
    print("─" * 100)

    # Clean model (unaugmented) as required by paper Section 2.5 for Virtual Screening
    print("[SAVE] Fitting clean baseline model (NO augmentation) on dev set for Virtual Screening...")
    final_model_clean = build_model()
    final_model_clean.fit(X_dev_b, y_dev)
    joblib.dump(final_model_clean, MODEL_FILE)
    print(f"[SAVED] {MODEL_FILE} (Unaugmented clean baseline model)")

    # Optional: Save augmented model under separate filename if required for downstream auditing
    aug_model_path = MODEL_FILE.replace('.joblib', '_augmented.joblib')
    final_model_aug = build_model()
    final_model_aug.fit(X_aug, y_aug)
    joblib.dump(final_model_aug, aug_model_path)
    print(f"[SAVED] {aug_model_path} (Augmented model saved separately)")

    # Export artifacts
    export_latex(results)
    export_figure(results)
    print("\n[DONE] 04_augmentation_training.py execution complete with Q1 statistical rigor.")


if __name__ == "__main__":
    run()
