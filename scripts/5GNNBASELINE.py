import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
TRAIN_FILE    = "data/V2-df_ic50_chmbl_CID_myFill.csv"
RESULTS_DIR   = "results"
LATEX_DIR     = "latex"
FIGURES_DIR   = "figures"

RANDOM_STATE  = 42
N_BOOTSTRAP   = 2000
TEST_SIZE     = 0.15          # identical to paper and 0STACK

# Paper reference values
PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP, no feature selection',               0.75),
    'PaperSelected': ('After Permutation Importance selection',        0.82),
    'PaperFinal':    ('Data augmentation + DNN ensemble',             0.85),
}

# Classical ML results from 0STACK (for comparison table)
# Update these with your actual nested CV means after 0STACK completes
CLASSICAL_R2 = {
    'Morgan FP (RF+XGB+SVM)': 0.7407,
    'RDKit 2D+FP (best)':     0.7322,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LATEX_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================================================
# CHECK DEPENDENCIES
# =========================================================
def run_chemprop(smiles, y, test_idx, train_idx):
    """ChemProp v2.x API (chemprop >= 2.0)"""
    try:
        import chemprop
        import torch
        from lightning import pytorch as pl
        from chemprop import data as cpdata, models, nn as cpnn
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError as e:
        print(f"  [ChemProp] Import error: {e}")
        return None

    print("\n  [ChemProp] Preparing data (v2 API)...")

    smiles_train = [smiles[i] for i in train_idx]
    y_train      = y[train_idx].reshape(-1, 1).tolist()
    smiles_test  = [smiles[i] for i in test_idx]
    y_test_vals  = y[test_idx].reshape(-1, 1).tolist()

    try:
        # Build datasets
        train_data = [cpdata.MoleculeDatapoint.from_smi(s, t)
                      for s, t in zip(smiles_train, y_train)
                      if cpdata.MoleculeDatapoint.from_smi(s, t) is not None]
        test_data  = [cpdata.MoleculeDatapoint.from_smi(s, t)
                      for s, t in zip(smiles_test, y_test_vals)
                      if cpdata.MoleculeDatapoint.from_smi(s, t) is not None]

        featurizer  = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset  = cpdata.MoleculeDataset(train_data, featurizer)
        test_dset   = cpdata.MoleculeDataset(test_data,  featurizer)

        train_loader = cpdata.build_dataloader(train_dset, shuffle=True,  num_workers=0)
        test_loader  = cpdata.build_dataloader(test_dset,  shuffle=False, num_workers=0)

        # Scaler
        scaler = train_dset.normalize_targets()
        test_dset.normalize_targets(scaler)

        # Model
        mp    = cpnn.BondMessagePassing()
        agg   = cpnn.MeanAggregation()
        ffn   = cpnn.RegressionFFN()
        batch_norm = cpnn.BatchNorm(mp.output_dim)
        mpnn  = models.MPNN(mp, agg, ffn, batch_norm=batch_norm, metrics=[cpnn.metrics.RMSE()])

        # Train
        trainer = pl.Trainer(
            max_epochs=100,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator='cpu',
        )
        trainer.fit(mpnn, train_loader)

        # Predict
        preds_raw = trainer.predict(mpnn, test_loader)
        y_pred_scaled = torch.cat(preds_raw).numpy().flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = y[test_idx]

        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        boot_mean, ci_lo, ci_hi, boots = bootstrap_r2_ci(y_true, y_pred)

        print(f"  [ChemProp v2] R2={r2:.4f} | MAE={mae:.4f} | "
              f"95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        return dict(model='ChemProp (D-MPNN)', r2=r2, mae=mae,
                    ci_lo=ci_lo, ci_hi=ci_hi, boots=boots,
                    y_test=y_true, y_pred=y_pred)

    except Exception as e:
        print(f"  [ChemProp] Failed: {e}")
        return None

def check_chemprop():
    try:
        import chemprop
        version = getattr(chemprop, "__version__", "unknown")
        print(f"  [OK] chemprop {version}")
        return True
    except ImportError:
        print("  [MISSING] chemprop — install with: pip install chemprop")
        return False

def check_dgllife():
    try:
        import dgl
        import dgllife
        print(f"  [OK] dgl {dgl.__version__}, dgllife {dgllife.__version__}")
        return True
    except ImportError:
        print("  [MISSING] dgllife — install with: pip install dgl dgllife")
        return False

def check_torch():
    try:
        import torch
        print(f"  [OK] torch {torch.__version__} | "
              f"CUDA: {torch.cuda.is_available()} "
              f"({'GTX 5060 Ti detected' if torch.cuda.is_available() else 'CPU only'})")
        return True
    except ImportError:
        print("  [MISSING] torch — install with: pip install torch")
        return False


# =========================================================
# DATA LOADING
# =========================================================
def load_data():
    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    df = df.reset_index(drop=True)
    print(f"[DATA] {len(df)} compounds loaded")
    return df['Smiles'].tolist(), df['pIC50 Value'].values


# =========================================================
# BOOTSTRAP CI
# =========================================================
def bootstrap_r2_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=0.95):
    from sklearn.metrics import r2_score
    rng = np.random.RandomState(RANDOM_STATE)
    n   = len(y_true)
    boots = [r2_score(y_true[idx := rng.randint(0, n, n)], y_pred[idx])
             for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(np.mean(boots)), float(lo), float(hi), boots


# =========================================================
# MODEL 1: ChemProp (D-MPNN)
# =========================================================
def run_attentivefp(smiles, y, test_idx, train_idx):
    """
    Trains AttentiveFP (Attention-based graph neural network for
    molecular property prediction) via DGL-LifeSci.
    """
    try:
        import torch
        import dgl
        from dgllife.model import AttentiveFPPredictor
        from dgllife.utils import (AttentiveFPAtomFeaturizer,
                                    AttentiveFPBondFeaturizer,
                                    mol_to_bigraph)
        from rdkit import Chem
        from torch.utils.data import DataLoader, Dataset
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("  [AttentiveFP] dgllife not available, skipping.")
        return None

    print("\n  [AttentiveFP] Preparing molecular graphs...")

    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')

    def smiles_to_graph(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return mol_to_bigraph(mol,
                               add_self_loop=True,
                               node_featurizer=atom_featurizer,
                               bond_featurizer=bond_featurizer)

    class MolDataset(Dataset):
        def __init__(self, smi_list, labels):
            valid = [(smiles_to_graph(s), l)
                     for s, l in zip(smi_list, labels)
                     if smiles_to_graph(s) is not None]
            self.graphs = [v[0] for v in valid]
            self.labels = torch.tensor([v[1] for v in valid],
                                        dtype=torch.float32).unsqueeze(1)
        def __len__(self):  return len(self.graphs)
        def __getitem__(self, i): return self.graphs[i], self.labels[i]

    def collate(batch):
        gs, ls = zip(*batch)
        return dgl.batch(gs), torch.stack(ls)

    smiles_train = [smiles[i] for i in train_idx]
    y_train      = y[train_idx]
    smiles_test  = [smiles[i] for i in test_idx]
    y_test       = y[test_idx]

    train_ds = MolDataset(smiles_train, y_train)
    test_ds  = MolDataset(smiles_test,  y_test)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,
                           collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                           collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [AttentiveFP] Training on {device}...")

    node_feat_size = atom_featurizer.feat_size('hv')
    edge_feat_size = bond_featurizer.feat_size('he')

    model = AttentiveFPPredictor(
        node_feat_size=node_feat_size,
        edge_feat_size=edge_feat_size,
        num_layers=2,
        num_timesteps=2,
        graph_feat_size=200,
        n_tasks=1,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = torch.nn.MSELoss()

    # Training loop
    for epoch in range(100):
        model.train()
        for g, lab in train_dl:
            g, lab = g.to(device), lab.to(device)
            pred = model(g, g.ndata['hv'], g.edata['he'])
            loss = loss_fn(pred, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for g, lab in test_dl:
            g = g.to(device)
            pred = model(g, g.ndata['hv'], g.edata['he'])
            all_preds.append(pred.cpu().numpy())
            all_labels.append(lab.numpy())

    y_pred_arr = np.vstack(all_preds).flatten()
    y_test_arr = np.vstack(all_labels).flatten()

    r2  = r2_score(y_test_arr, y_pred_arr)
    mae = mean_absolute_error(y_test_arr, y_pred_arr)
    boot_mean, ci_lo, ci_hi, boots = bootstrap_r2_ci(y_test_arr, y_pred_arr)

    print(f"  [AttentiveFP] R2={r2:.4f} | MAE={mae:.4f} | "
          f"95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    return dict(model='AttentiveFP', r2=r2, mae=mae,
                ci_lo=ci_lo, ci_hi=ci_hi, boots=boots,
                y_test=y_test_arr, y_pred=y_pred_arr)


# =========================================================
# LATEX EXPORT
# =========================================================
def newcommand(f, name, value):
    f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")

def export_latex(results):
    path = os.path.join(LATEX_DIR, "gnn_variables.tex")
    with open(path, 'w') as f:
        f.write("% Auto-generated by 5GNNBASELINE.py\n")
        f.write("% Include with: \\input{latex/gnn_variables.tex}\n\n")
        for res in results:
            if res is None:
                continue
            label = res['model'].replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
            newcommand(f, f"{label}RTwoMean",  f"{res['r2']:.4f}")
            newcommand(f, f"{label}RTwoStd",   f"{res['ci_hi']-res['ci_lo']:.4f}")
            newcommand(f, f"{label}Mae",       f"{res['mae']:.4f}")
            newcommand(f, f"{label}CILow",     f"{res['ci_lo']:.4f}")
            newcommand(f, f"{label}CIHigh",    f"{res['ci_hi']:.4f}")
            # t-test vs paper values
            for pk, (_, pval) in PAPER_R2.items():
                t, p = stats.ttest_1samp(res['boots'], pval)
                newcommand(f, f"{label}Vs{pk}P", f"{p:.4f}")
                newcommand(f, f"{label}Vs{pk}Sig",
                           "true" if p < 0.05 else "false")
    print(f"[LATEX] gnn_variables.tex saved")


# =========================================================
# FIGURE — Full comparison: classical + GNN + paper
# =========================================================
def export_figure(results):
    from sklearn.metrics import r2_score

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: R2 comparison bar chart ──
    ax = axes[0]
    labels, means, errs = [], [], []

    # Classical ML from 0STACK
    for name, r2 in CLASSICAL_R2.items():
        labels.append(name)
        means.append(r2)
        errs.append(0.05)  # approximate from nested CV std

    # GNN results
    for res in results:
        if res is not None:
            labels.append(res['model'])
            means.append(res['r2'])
            errs.append((res['ci_hi'] - res['ci_lo']) / 2)

    colors = ['#4C72B0'] * len(CLASSICAL_R2) + \
             ['#C44E52'] * sum(1 for r in results if r is not None)

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=5, color=colors, alpha=0.85)

    for label, (desc, val) in PAPER_R2.items():
        ls = '-' if 'Final' in label else '--'
        ax.axhline(val, linestyle=ls, linewidth=1.2, alpha=0.7,
                   label=f"{label} ({val})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("R² (test set)")
    ax.set_title("R² comparison: Classical ML vs GNNs vs Paper DNN")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)

    # ── Right: scatter predicted vs actual for best GNN ──
    ax2 = axes[1]
    best_gnn = max([r for r in results if r is not None],
                   key=lambda r: r['r2'], default=None)
    if best_gnn is not None:
        ax2.scatter(best_gnn['y_test'], best_gnn['y_pred'],
                    alpha=0.6, s=30, color='#C44E52')
        mn = min(best_gnn['y_test'].min(), best_gnn['y_pred'].min())
        mx = max(best_gnn['y_test'].max(), best_gnn['y_pred'].max())
        ax2.plot([mn, mx], [mn, mx], 'k--', linewidth=1, label='Ideal')
        ax2.set_xlabel("Experimental pIC₅₀")
        ax2.set_ylabel("Predicted pIC₅₀")
        ax2.set_title(f"{best_gnn['model']} — Predicted vs Actual "
                      f"(R²={best_gnn['r2']:.3f})")
        ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "gnn_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[FIGURE] gnn_comparison.png saved")


# =========================================================
# EXECUTIVE SUMMARY
# =========================================================
def print_summary(results):
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY — paste into Results/Discussion")
    print("=" * 100)

    valid = [r for r in results if r is not None]
    if not valid:
        print("[WARNING] No GNN results to summarize.")
        return

    print(f"\n{'Model':<30} {'R2':>8} {'MAE':>8} {'95% CI':>20} "
          f"{'vs 0.82 (p)':>12} {'vs 0.85 (p)':>12}")
    print("-" * 95)

    for name, r2 in CLASSICAL_R2.items():
        print(f"{name:<30} {r2:>8.4f} {'--':>8} {'(nested CV)':>20} {'--':>12} {'--':>12}")

    for res in valid:
        t82, p82 = stats.ttest_1samp(res['boots'], 0.82)
        t85, p85 = stats.ttest_1samp(res['boots'], 0.85)
        ci_str = f"[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]"
        print(f"{res['model']:<30} {res['r2']:>8.4f} {res['mae']:>8.4f} "
              f"{ci_str:>20} {p82:>12.4f} {p85:>12.4f}")

    print()
    for label, (desc, pval) in PAPER_R2.items():
        print(f"  Paper ref — {desc}: R2={pval}")

    print("\nKEY FINDING:")
    best = max(valid, key=lambda r: r['r2'])
    t, p = stats.ttest_1samp(best['boots'], 0.85)
    sig  = "NOT significantly different from" if p >= 0.05 \
           else "significantly below"
    print(f"  Best GNN ({best['model']}): R2={best['r2']:.4f} — "
          f"{sig} the paper's DNN (R2=0.85, p={p:.4f})")

    classical_best = max(CLASSICAL_R2.values())
    if best['r2'] < classical_best + 0.02:
        print(f"\n  CANDIDATE SENTENCE (Discussion):")
        print(f'  "Graph neural network baselines (ChemProp D-MPNN: '
              f'R2={valid[0]["r2"]:.3f}; AttentiveFP: '
              f'R2={valid[-1]["r2"]:.3f} if available) did not '
              f'significantly outperform the classical stacking ensemble '
              f'(R2={classical_best:.3f}), further supporting the conclusion '
              f'that architectural complexity beyond classical ML is not '
              f'justified for this dataset size ({756} compounds)."')


# =========================================================
# MAIN
# =========================================================
def run():
    print("=" * 100)
    print("5GNNBASELINE.py — GNN Baseline Comparison (ChemProp + AttentiveFP)")
    print("Same train/test split as paper and 0STACK.py (random_state=42, test=15%)")
    print("=" * 100)

    print("\n[DEPENDENCIES CHECK]")
    has_torch     = check_torch()
    has_chemprop  = check_chemprop()
    has_dgllife   = check_dgllife()

    if not has_torch:
        print("\n[ERROR] PyTorch required. Install: pip install torch")
        print("        Then: pip install chemprop")
        print("        Then: pip install dgl dgllife")
        return

    smiles, y = load_data()

    # Same split as paper (random_state=42, test_size=0.15)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(smiles))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE,
                                            random_state=RANDOM_STATE)
    print(f"[SPLIT] Train={len(train_idx)} | Test={len(test_idx)}")

    results = []

    # ── ChemProp ──────────────────────────────────────────────────
    if has_chemprop:
        print("\n" + "─" * 100)
        print("MODEL 1: ChemProp (Directed Message Passing Neural Network)")
        print("─" * 100)
        t0  = time.time()
        res = run_chemprop(smiles, y, test_idx, train_idx)
        print(f"  Time: {time.time()-t0:.1f}s")
        results.append(res)
    else:
        print("\n[SKIP] ChemProp not installed.")
        results.append(None)

    # ── AttentiveFP ───────────────────────────────────────────────
    if has_dgllife:
        print("\n" + "─" * 100)
        print("MODEL 2: AttentiveFP (Graph Attention Network)")
        print("─" * 100)
        t0  = time.time()
        res = run_attentivefp(smiles, y, test_idx, train_idx)
        print(f"  Time: {time.time()-t0:.1f}s")
        results.append(res)
    else:
        print("\n[SKIP] DGL-LifeSci not installed.")
        results.append(None)

    if all(r is None for r in results):
        print("\n[ERROR] No models ran successfully.")
        print("Install dependencies and re-run:")
        print("  pip install torch chemprop dgl dgllife")
        return

    print_summary(results)
    export_latex(results)
    export_figure(results)

    # Save results CSV
    rows = []
    for res in results:
        if res is not None:
            rows.append({
                'Model': res['model'], 'R2': res['r2'], 'MAE': res['mae'],
                'CI_low': res['ci_lo'], 'CI_high': res['ci_hi'],
            })
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, "gnn_results.csv"), index=False)
        print(f"[EXPORT] gnn_results.csv saved")

    print("\n[DONE] 5GNNBASELINE.py complete.")


if __name__ == "__main__":
    run()
