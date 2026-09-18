import logging
import os
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
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Link root directory to import paths_config and logger_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import *
from logger_utils import setup_logger

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
        torch.set_num_threads(24)
    except ImportError as e:
        print(f"  [ChemProp] Import error: {e}", flush=True)
        return None

    total_folds = n_splits * n_repeats
    print(f"\n  [ChemProp] Starting {n_splits}x{n_repeats} CV (N={len(smiles)} compounds)...", flush=True)

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    oof_predictions = np.zeros((len(y), n_repeats))

    fold_r2s, fold_maes = [], []
    smiles_arr = np.array(smiles)

    for fold, (train_idx, val_idx) in enumerate(rkf.split(smiles_arr)):
        t_fold = time.time()
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

        trainer = pl.Trainer(max_epochs=100, enable_progress_bar=False, enable_model_summary=False, logger=False, accelerator='cpu')
        trainer.fit(mpnn, train_loader)

        preds_raw = trainer.predict(mpnn, val_loader)
        y_pred = scaler.inverse_transform(torch.cat(preds_raw).numpy().flatten().reshape(-1, 1)).flatten()

        f_r2, f_mae = r2_score(y[val_idx], y_pred), mean_absolute_error(y[val_idx], y_pred)
        fold_r2s.append(f_r2)
        fold_maes.append(f_mae)
        elapsed = time.time() - t_fold
        print(f"  --> [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Fold {fold+1}/{total_folds} OK | R2={f_r2:.4f} | MAE={f_mae:.4f} "
              f"| {elapsed:.1f}s", flush=True)

        repeat_idx = fold // n_splits
        oof_predictions[val_idx, repeat_idx] = y_pred

    print("\n  [ChemProp] 5x5 CV Completed.", flush=True)

    mean_r2, std_r2 = float(np.mean(fold_r2s)), float(np.std(fold_r2s))
    mean_mae = float(np.mean(fold_maes))
    ci_lo, ci_hi = mean_r2 - 1.96 * (std_r2 / np.sqrt(total_folds)), mean_r2 + 1.96 * (std_r2 / np.sqrt(total_folds))

    print(f"  [ChemProp Result] R2={mean_r2:.4f} ± {std_r2:.4f} | MAE={mean_mae:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    final_oof = np.mean(oof_predictions, axis=1)

    return dict(model='ChemProp (D-MPNN Fold)', r2=mean_r2, std=std_r2, mae=mean_mae, ci_lo=ci_lo, ci_hi=ci_hi, boots=fold_r2s, y_test=y, y_pred=final_oof)

# =========================================================
# MODEL 2: AttentiveFP (Single-Split Reference)
# =========================================================
def run_attentivefp(smiles, y, test_idx, train_idx):
    """
    Evaluates AttentiveFP on a single train/test split benchmark.
    """
    try:
        import torch
        import dgl
        from dgllife.model import AttentiveFPPredictor
        from dgllife.utils import (AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, mol_to_bigraph)
        from rdkit import Chem
        from torch.utils.data import DataLoader, Dataset
        torch.set_num_threads(24)
    except Exception as e:
        print(f"  [AttentiveFP] Could not initialize DGL ({e}), skipping.", flush=True)
        return None

    print("\n  [AttentiveFP] Starting model (WARNING: Single Split, not directly comparable with 5x5 CV)...", flush=True)
    atom_featurizer, bond_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv'), AttentiveFPBondFeaturizer(bond_data_field='he')

    def smiles_to_graph(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return mol_to_bigraph(mol, add_self_loop=True, node_featurizer=atom_featurizer, bond_featurizer=bond_featurizer)

    class MolDataset(Dataset):
        def __init__(self, smi_list, labels):
            valid = [(smiles_to_graph(s), l) for s, l in zip(smi_list, labels) if smiles_to_graph(s) is not None]
            self.graphs, self.labels = [v[0] for v in valid], torch.tensor([v[1] for v in valid], dtype=torch.float32).unsqueeze(1)
        def __len__(self): return len(self.graphs)
        def __getitem__(self, i): return self.graphs[i], self.labels[i]

    def collate(batch):
        gs, ls = zip(*batch)
        return dgl.batch(gs), torch.stack(ls)

    train_dl = DataLoader(MolDataset([smiles[i] for i in train_idx], y[train_idx]), batch_size=32, shuffle=True, collate_fn=collate, num_workers=N_WORKERS_GNN)
    test_dl  = DataLoader(MolDataset([smiles[i] for i in test_idx], y[test_idx]), batch_size=32, shuffle=False, collate_fn=collate, num_workers=N_WORKERS_GNN)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentiveFPPredictor(node_feat_size=atom_featurizer.feat_size('hv'), edge_feat_size=bond_featurizer.feat_size('he'), num_layers=2, num_timesteps=2, graph_feat_size=200, n_tasks=1, dropout=0.2).to(device)
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=1e-3), torch.nn.MSELoss()

    for _ in range(100):
        model.train()
        for g, lab in train_dl:
            g, lab = g.to(device), lab.to(device)
            pred = model(g, g.ndata['hv'], g.edata['he'])
            loss = loss_fn(pred, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for g, lab in test_dl:
            g = g.to(device)
            all_preds.append(model(g, g.ndata['hv'], g.edata['he']).cpu().numpy())
            all_labels.append(lab.numpy())

    y_pred_arr, y_test_arr = np.vstack(all_preds).flatten(), np.vstack(all_labels).flatten()
    r2, mae = r2_score(y_test_arr, y_pred_arr), mean_absolute_error(y_test_arr, y_pred_arr)

    rng = np.random.RandomState(RANDOM_STATE)
    boots = [r2_score(y_test_arr[idx := rng.randint(0, len(y_test_arr), len(y_test_arr))], y_pred_arr[idx]) for _ in range(N_BOOTSTRAP_GNN)]
    ci_lo, ci_hi, std_r2 = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(np.std(boots))

    print(f"  [AttentiveFP Result] R2={r2:.4f} | MAE={mae:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)
    return dict(model='AttentiveFP (Single-Split)', r2=r2, std=std_r2, mae=mae, ci_lo=ci_lo, ci_hi=ci_hi, boots=boots, y_test=y_test_arr, y_pred=y_pred_arr)

# =========================================================
# DATA & EXPORT UTILITIES
# =========================================================
def load_data():
    """
    Loads chemical dataset from TRAIN_FILE.
    """
    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    return df['Smiles'].tolist(), df['pIC50 Value'].values

def export_latex(results):
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

    print(f"[EXPORT] LaTeX variables saved to {LATEX_GNN}", flush=True)

def export_figure(results):
    """
    Generates comparison bar plot and prediction scatter plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels, means, errs = list(CLASSICAL_R2.keys()), list(CLASSICAL_R2.values()), [0.05]*len(CLASSICAL_R2)

    for res in results:
        if res is not None:
            labels.append(res['model'])
            means.append(res['r2'])
            errs.append((res['ci_hi'] - res['ci_lo']) / 2)

    colors = ['#4C72B0'] * len(CLASSICAL_R2) + ['#C44E52'] * sum(1 for r in results if r is not None)
    axes[0].bar(np.arange(len(labels)), means, yerr=errs, capsize=5, color=colors, alpha=0.85)
    for label, (_, val) in PAPER_R2.items():
        axes[0].axhline(val, linestyle='--', linewidth=1.2, alpha=0.7)

    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(0.5, 1.0)

    best_gnn = max([r for r in results if r is not None], key=lambda r: r['r2'], default=None)
    if best_gnn:
        axes[1].scatter(best_gnn['y_test'], best_gnn['y_pred'], alpha=0.6, s=30, color='#C44E52')
        mn, mx = min(best_gnn['y_test']), max(best_gnn['y_test'])
        axes[1].plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        axes[1].set_xlabel("Experimental pIC₅₀")
        axes[1].set_ylabel("Predicted pIC₅₀")
        axes[1].set_title(f"{best_gnn['model']} (R²={best_gnn['r2']:.3f})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIGURE_GNN), exist_ok=True)
    plt.savefig(FIGURE_GNN, dpi=300)
    plt.close(fig)
    print(f"[EXPORT] Comparison plot saved to {FIGURE_GNN}", flush=True)

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    t0 = time.time()
    print("=" * 105)
    print("09_gnn_baseline.py — Graph Neural Network (GNN) Benchmarking")
    print(f"Log file: {log_file_path}")
    print("=" * 105)

    print(f"[1] Loading dataset from {TRAIN_FILE}...")
    smiles, y = load_data()
    print(f"[1] Dataset loaded successfully: {len(smiles)} compounds.")

    train_idx, test_idx = train_test_split(
        np.arange(len(smiles)), test_size=TEST_SIZE_GNN, random_state=RANDOM_STATE
    )

    print("\n[2] Evaluating ChemProp (D-MPNN 5x5 CV)...")
    res_chemprop = run_chemprop_cv(smiles, y, n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    print("\n[3] Evaluating AttentiveFP (Single Split)...")
    res_attentive = run_attentivefp(smiles, y, test_idx, train_idx)

    results = [res_chemprop, res_attentive]
    valid = [r for r in results if r is not None]

    print("\n" + "=" * 105)
    print("GNN BENCHMARK SUMMARY")
    print("=" * 105)
    for res in valid:
        print(f"{res['model']:<35} | R2: {res['r2']:.4f} | MAE: {res['mae']:.4f} | 95% CI: [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")

    print("\n[4] Exporting artifacts...")
    export_latex(results)
    export_figure(results)

    if valid:
        df_res = pd.DataFrame([{'Model': r['model'], 'R2': r['r2'], 'MAE': r['mae'], 'CI_Low': r['ci_lo'], 'CI_High': r['ci_hi']} for r in valid])
        df_res.to_csv(GNN_RESULTS_CSV, index=False)
        print(f"[EXPORT] Summary CSV saved to {GNN_RESULTS_CSV}")

    t_elapsed = time.time() - t0
    mins, secs = divmod(t_elapsed, 60)
    print(f"[TIME] Total execution: {int(mins)}m {int(secs)}s")
    print("[DONE] 09_gnn_baseline.py complete.")

if __name__ == "__main__":
    main()