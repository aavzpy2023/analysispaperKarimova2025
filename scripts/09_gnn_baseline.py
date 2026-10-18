import logging
import os
import math
import re
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from splits import outer_splits
from nb_stats import compare_to_classical
from sklearn.metrics import r2_score, mean_absolute_error

# Link root directory to import paths_config and logger_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import *
from logger_utils import setup_logger, explain

# Filter PyTorch Lightning tips
class TipFilter(logging.Filter):
    def filter(self, record):
        return " Tip" not in record.getMessage()

logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(TipFilter())
warnings.filterwarnings("ignore")

# Silence RDKit verbose warnings
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

# =========================================================
# LOGGING SETUP
# =========================================================
log_file_path = os.path.join(LOGS_DIR, "09_gnn_baseline.log")
setup_logger(log_file_path)

# =========================================================
# MODEL 1: ChemProp 5x5 Repeated K-Fold CV
# =========================================================
def run_chemprop_cv(smiles, y, n_splits=5, n_repeats=5, random_state=42):
    """
    Executes a 5x5 Repeated K-Fold Cross-Validation using ChemProp (D-MPNN).
    """
    try:
        import chemprop
        import torch
        from lightning import pytorch as pl
        from chemprop import data as cpdata, models, nn as cpnn
        torch.set_num_threads(max(1, min(24, N_JOBS)))
    except ImportError as e:
        if os.environ.get("MLGNN_STRICT") == "1":
            raise
        print(f"  [ChemProp] Import error: {e}", flush=True)
        return None

    total_folds = n_splits * n_repeats
    print(f"\n  [ChemProp] Starting {n_splits}x{n_repeats} CV (N={len(smiles)} compounds)...", flush=True)

    folds = list(outer_splits(smiles, n_splits, n_repeats, random_state, SPLIT_KIND))
    oof_predictions = np.zeros((len(y), n_repeats))

    fold_r2s, fold_maes = [], []
    smiles_arr = np.array(smiles)

    fold_times, fold_sizes = [], []
    for fold, (train_idx, val_idx) in enumerate(folds):
        t_fold = time.time()
        pl.seed_everything(random_state + fold, workers=True)   # reproducible init/shuffling per fold
        print(f"  --> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training Fold {fold+1}/{total_folds}...", flush=True)

        smiles_train, y_train = smiles_arr[train_idx].tolist(), y[train_idx].reshape(-1, 1).tolist()
        smiles_val, y_val_vals = smiles_arr[val_idx].tolist(), y[val_idx].reshape(-1, 1).tolist()

        train_data = [cpdata.MoleculeDatapoint.from_smi(s, t) for s, t in zip(smiles_train, y_train) if cpdata.MoleculeDatapoint.from_smi(s, t) is not None]
        val_data   = [cpdata.MoleculeDatapoint.from_smi(s, t) for s, t in zip(smiles_val, y_val_vals) if cpdata.MoleculeDatapoint.from_smi(s, t) is not None]

        featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset, val_dset = cpdata.MoleculeDataset(train_data, featurizer), cpdata.MoleculeDataset(val_data, featurizer)

        train_loader = cpdata.build_dataloader(train_dset, shuffle=True, num_workers=N_WORKERS_GNN)
        val_loader   = cpdata.build_dataloader(val_dset, shuffle=False, num_workers=N_WORKERS_GNN)

        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        mp, agg, ffn = cpnn.BondMessagePassing(), cpnn.MeanAggregation(), cpnn.RegressionFFN()
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[cpnn.metrics.RMSE()])

        # enable_checkpointing=False: predictions use the in-memory model, so no .ckpt files are needed.
        # (Default checkpointing wrote 25 files/run into ./checkpoints and accumulated -vN duplicates.)
        trainer = pl.Trainer(max_epochs=100, enable_progress_bar=False, enable_model_summary=False,
                             enable_checkpointing=False, logger=False, accelerator='cpu')
        trainer.fit(mpnn, train_loader)

        preds_raw = trainer.predict(mpnn, val_loader)
        y_pred = scaler.inverse_transform(torch.cat(preds_raw).numpy().flatten().reshape(-1, 1)).flatten()

        f_r2, f_mae = r2_score(y[val_idx], y_pred), mean_absolute_error(y[val_idx], y_pred)
        fold_r2s.append(f_r2)
        fold_sizes.append((len(train_idx), len(val_idx)))
        fold_maes.append(f_mae)
        elapsed = time.time() - t_fold
        fold_times.append(elapsed)
        print(f"  --> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Fold {fold+1}/{total_folds} OK | R2={f_r2:.4f} | MAE={f_mae:.4f} "
              f"| {elapsed:.1f}s", flush=True)

        repeat_idx = fold // n_splits
        oof_predictions[val_idx, repeat_idx] = y_pred

    print("\n  [ChemProp] 5x5 CV Completed.", flush=True)

    mean_r2, std_r2 = float(np.mean(fold_r2s)), float(np.std(fold_r2s, ddof=1))
    mean_mae = float(np.mean(fold_maes))
    # Nadeau-Bengio corrected interval: repeated-CV folds share training data, so the naive standard error is too small.
    n_train_mean, n_test_mean = (float(np.mean(v)) for v in zip(*fold_sizes))
    half_width = stats.t.ppf(0.975, df=total_folds - 1) * std_r2 * np.sqrt(1.0 / total_folds + n_test_mean / n_train_mean)
    ci_lo, ci_hi = mean_r2 - half_width, mean_r2 + half_width

    print(f"  [ChemProp Result] R2={mean_r2:.4f} ± {std_r2:.4f} | MAE={mean_mae:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    final_oof = np.mean(oof_predictions, axis=1)

    return dict(model='ChemProp (D-MPNN Fold)', r2=mean_r2, std=std_r2, mae=mean_mae, ci_lo=ci_lo, ci_hi=ci_hi, boots=fold_r2s, fold_times=fold_times, fold_maes=fold_maes, fold_sizes=fold_sizes, n_folds=total_folds, y_test=y, y_pred=final_oof)

# =========================================================
# DATA & EXPORT UTILITIES
# =========================================================
def load_data():
    """
    Loads chemical dataset from TRAIN_FILE.
    """
    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    return df['Smiles'].tolist(), df['pIC50 Value'].values

def compare_with_classical(res):
    """Paired comparison of the D-MPNN with each classical representation on the shared folds (see nb_stats)."""
    gnn = pd.DataFrame({'Fold': range(res['n_folds']), 'R2': res['boots'], 'MAE': res['fold_maes'],
                        'N_train': [a for a, _ in res['fold_sizes']], 'N_test': [b for _, b in res['fold_sizes']]})
    gnn.to_csv(GNN_FOLD_SCORES_CSV, index=False)
    table, macros = compare_to_classical(gnn, pd.read_csv(CHECKPOINT_FILE), "GnnVs")
    table.to_csv(GNN_VS_CLASSICAL_CSV, index=False)
    return macros


def export_timing(res):
    """Wall-clock cost of one D-MPNN training versus one classical outer fold (inner search included), plus the cost an
    equivalent inner search would add to the D-MPNN. Classical times come from the step 01 checkpoint."""
    times = np.array(res['fold_times'])
    n_fits = sum(math.comb(6, k) for k in range(1, CFG['MAX_COMBO_SIZE'] + 1)) * CFG['INNER_N_SPLITS']
    values = {"GnnFoldSecMean": f"{times.mean():.0f}", "GnnFoldSecMin": f"{times.min():.0f}",
              "GnnFoldSecMax": f"{times.max():.0f}", "GnnTotalMin": f"{times.sum() / 60:.0f}",
              "GnnSearchFits": n_fits, "GnnSearchHoursPerFold": f"{n_fits * times.mean() / 3600:.0f}",
              "GnnSearchDaysTotal": f"{n_fits * times.sum() / 86400:.0f}"}
    if os.path.exists(CHECKPOINT_FILE):
        per_mode = pd.read_csv(CHECKPOINT_FILE).groupby('Mode')['Time_s'].mean()
        values.update({"ClassicalFoldSecMin": f"{per_mode.min():.0f}", "ClassicalFoldSecMax": f"{per_mode.max():.0f}",
                       "ClassicalMorganFoldSec": f"{per_mode.get('morgan', float('nan')):.0f}",
                       "ClassicalMorganTotalHours": f"{per_mode.get('morgan', float('nan')) * res['n_folds'] / 3600:.1f}"})
    os.makedirs(os.path.dirname(LATEX_TIMING), exist_ok=True)
    with open(LATEX_TIMING, 'w', encoding='utf-8') as f:
        f.write("% Auto-generated by 09_gnn_baseline.py\n")
        for name, value in values.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    print(f"[EXPORT] Timing variables saved to {LATEX_TIMING}", flush=True)


def export_latex(results, extra=None):
    """
    Exports GNN metrics to LaTeX command variables.
    """
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("\n  [WARNING] No valid results found. gnn_variables.tex was not modified.", flush=True)
        return

    os.makedirs(os.path.dirname(LATEX_GNN), exist_ok=True)
    with open(LATEX_GNN, 'w') as f:
        for res in valid_results:
            label = re.sub(r'[^A-Za-z]', '', res['model'])
            f.write(f"\\newcommand{{\\{label}RTwoMean}}{{{res['r2']:.4f}}}\n")
            f.write(f"\\newcommand{{\\{label}Mae}}{{{res['mae']:.4f}}}\n")
            if 'std' in res:
                f.write(f"\\newcommand{{\\{label}RTwoStd}}{{{res['std']:.4f}}}\n")
            if 'ci_lo' in res and 'ci_hi' in res:
                f.write(f"\\newcommand{{\\{label}CILow}}{{{res['ci_lo']:.4f}}}\n")
                f.write(f"\\newcommand{{\\{label}CIHigh}}{{{res['ci_hi']:.4f}}}\n")
        for name, value in (extra or {}).items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")

    print(f"[EXPORT] LaTeX variables saved to {LATEX_GNN}", flush=True)

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    t0 = time.time()
    print("=" * 105)
    print("09_gnn_baseline.py — Graph Neural Network (GNN) Benchmarking: ChemProp D-MPNN, 5x5 repeated CV")
    print(f"Log file: {log_file_path}")
    print("=" * 105)
    explain("What this step does", """
        Compares the classical machine-learning models (steps 01/04, which describe each molecule with fixed fingerprints or
        descriptors) against a Graph Neural Network. ChemProp's D-MPNN treats the molecule as a graph (atoms = nodes, bonds =
        edges) and learns its own representation by passing messages between neighbouring atoms, instead of using a
        hand-designed fingerprint.

        Same evaluation protocol as the classical models: 5 x 5 repeated cross-validation (data split into 5 parts, each part
        used once as unseen test data, whole procedure repeated 5 times with different splits). Out-of-fold predictions are
        collected so every molecule is predicted by a model that never saw it.

        Why it matters: with only a few hundred compounds, deep models often do not beat well-tuned classical models. If
        ChemProp's R2 is not clearly above the classical R2 (taking the 95% CI into account), the simpler models are preferable
        because they are faster, easier to interpret and not worse in accuracy. R2: 1 = perfect, 0 = no better than the mean.
        MAE is the average error in pIC50 units (0.5 = about 3-fold in IC50).
    """)

    print(f"[1] Loading dataset from {TRAIN_FILE}...")
    smiles, y = load_data()
    print(f"[1] Dataset loaded successfully: {len(smiles)} compounds.")

    print("\n[2] Evaluating ChemProp (D-MPNN 5x5 CV)...")
    res_chemprop = run_chemprop_cv(smiles, y, n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    results = [res_chemprop]
    valid = [r for r in results if r is not None]

    print("\n" + "=" * 105)
    print("GNN BENCHMARK SUMMARY")
    print("=" * 105)
    for res in valid:
        print(f"{res['model']:<35} | R2: {res['r2']:.4f} | MAE: {res['mae']:.4f} | 95% CI: [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")

    explain("Reading the benchmark", """
        'R2 ± std' is the mean and spread of R2 over the cross-validation folds; the 95% CI is the range in which the true
        performance is expected to fall. Compare it with the classical R2 values drawn as dashed lines in the figure. Large
        std or a wide CI means the result depends strongly on which molecules landed in the test fold (typical for small data).
        The scatter plot (right panel of the figure) shows predicted vs. experimental pIC50: points on the diagonal are perfect
        predictions; systematic curvature at the extremes means the model compresses very high/low potencies toward the mean.
    """)
    print("\n[3] Exporting artifacts...")
    try:
        test_macros = compare_with_classical(res_chemprop)
    except Exception as e:
        import traceback
        print(f"[WARNING] Paired comparison with the classical models failed: {e}\n{traceback.format_exc()}", flush=True)
        test_macros = {}
    export_latex(results, extra=test_macros)
    try:
        export_timing(res_chemprop)
    except Exception as e:
        import traceback
        print(f"[WARNING] Timing variables were not exported: {e}\n{traceback.format_exc()}", flush=True)

    if valid:
        df_res = pd.DataFrame([{'Model': r['model'], 'R2': r['r2'], 'MAE': r['mae'], 'CI_Low': r['ci_lo'], 'CI_High': r['ci_hi']} for r in valid])
        df_res.to_csv(GNN_RESULTS_CSV, index=False)
        print(f"[EXPORT] Summary CSV saved to {GNN_RESULTS_CSV}")

        # Export per-compound out-of-fold predictions (best model) for downstream figure
        # generation (09.1_build_figures.py). The summary CSV above only holds one row
        # per model, so the underlying y_test/y_pred arrays never survive into it.
        best = max(valid, key=lambda r: r['r2'])
        df_oof = pd.DataFrame({
            'Model': best['model'],
            'y_test': best['y_test'],
            'y_pred': best['y_pred'],
        })
        df_oof.to_csv(GNN_OOF_PREDICTIONS_CSV, index=False)
        print(f"[EXPORT] Out-of-fold predictions ({best['model']}) saved to {GNN_OOF_PREDICTIONS_CSV}")

    t_elapsed = time.time() - t0
    mins, secs = divmod(t_elapsed, 60)
    print(f"[TIME] Total execution: {int(mins)}m {int(secs)}s")
    print("[DONE] 09_gnn_baseline.py complete.")

if __name__ == "__main__":
    main()