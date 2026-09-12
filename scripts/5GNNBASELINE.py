import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
TRAIN_FILE    = "data/V2-df_ic50_chmbl_CID_myFill.csv"
RESULTS_DIR   = "results"
LATEX_DIR     = "latex"
FIGURES_DIR   = "figures"

RANDOM_STATE  = 42
N_BOOTSTRAP   = 2000
TEST_SIZE     = 0.15

PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP, no feature selection', 0.75),
    'PaperSelected': ('After Permutation Importance selection', 0.82),
    'PaperFinal':    ('Data augmentation + DNN ensemble', 0.85),
}

CLASSICAL_R2 = {
    'Morgan FP (RF+XGB+SVM)': 0.7407,
    'RDKit 2D+FP (best)':     0.7322,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LATEX_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================================================
# MODELO 1: ChemProp 15-Fold CV (Optimizado para CPU Multi-Core)
# =========================================================
def run_chemprop_15fold(smiles, y, n_splits=15, random_state=42):
    try:
        import chemprop
        import torch
        from lightning import pytorch as pl
        from chemprop import data as cpdata, models, nn as cpnn
        
        # Configurar PyTorch para aprovechar los núcleos del CPU
        torch.set_num_threads(24)
        
    except ImportError as e:
        print(f"  [ChemProp] Error de importación: {e}")
        return None

    print(f"\n  [ChemProp] Iniciando 15-Fold CV en N={len(smiles)} compuestos (Multi-Core CPU)...")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(len(y))
    fold_r2s = []
    fold_maes = []
    smiles_arr = np.array(smiles)

    for fold, (train_idx, val_idx) in enumerate(kf.split(smiles_arr)):
        print(f"  --> Entrenando Fold {fold+1}/{n_splits}...", end="\r")

        smiles_train = smiles_arr[train_idx].tolist()
        y_train      = y[train_idx].reshape(-1, 1).tolist()
        smiles_val   = smiles_arr[val_idx].tolist()
        y_val_vals   = y[val_idx].reshape(-1, 1).tolist()

        train_data = [cpdata.MoleculeDatapoint.from_smi(s, t) for s, t in zip(smiles_train, y_train) if cpdata.MoleculeDatapoint.from_smi(s, t) is not None]
        val_data   = [cpdata.MoleculeDatapoint.from_smi(s, t) for s, t in zip(smiles_val, y_val_vals) if cpdata.MoleculeDatapoint.from_smi(s, t) is not None]

        featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = cpdata.MoleculeDataset(train_data, featurizer)
        val_dset   = cpdata.MoleculeDataset(val_data, featurizer)

        # num_workers=8 para paralelizar la carga en CPU
        train_loader = cpdata.build_dataloader(train_dset, shuffle=True, num_workers=8)
        val_loader   = cpdata.build_dataloader(val_dset, shuffle=False, num_workers=8)

        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        mp   = cpnn.BondMessagePassing()
        agg  = cpnn.MeanAggregation()
        ffn  = cpnn.RegressionFFN()
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[cpnn.metrics.RMSE()])

        trainer = pl.Trainer(
            max_epochs=100,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator='cpu'
        )
        trainer.fit(mpnn, train_loader)

        preds_raw = trainer.predict(mpnn, val_loader)
        y_pred_scaled = torch.cat(preds_raw).numpy().flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        oof_predictions[val_idx] = y_pred
        
        f_r2  = r2_score(y[val_idx], y_pred)
        f_mae = mean_absolute_error(y[val_idx], y_pred)
        fold_r2s.append(f_r2)
        fold_maes.append(f_mae)

    print("\n  [ChemProp] 15-Fold CV Completado.")

    mean_r2  = float(np.mean(fold_r2s))
    std_r2   = float(np.std(fold_r2s))
    mean_mae = float(np.mean(fold_maes))
    
    ci_lo = mean_r2 - 1.96 * (std_r2 / np.sqrt(n_splits))
    ci_hi = mean_r2 + 1.96 * (std_r2 / np.sqrt(n_splits))

    print(f"  [ChemProp Result] R2={mean_r2:.4f} ± {std_r2:.4f} | MAE={mean_mae:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    return dict(model='ChemProp (D-MPNN 15-Fold)', r2=mean_r2, std=std_r2, mae=mean_mae, ci_lo=ci_lo, ci_hi=ci_hi, boots=fold_r2s, y_test=y, y_pred=oof_predictions)


# =========================================================
# MODELO 2: AttentiveFP
# =========================================================
def run_attentivefp(smiles, y, test_idx, train_idx):
    try:
        import torch
        import dgl
        from dgllife.model import AttentiveFPPredictor
        from dgllife.utils import (AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, mol_to_bigraph)
        from rdkit import Chem
        from torch.utils.data import DataLoader, Dataset
        
        torch.set_num_threads(24)
    except Exception as e:
        print(f"  [AttentiveFP] No se pudo inicializar DGL ({e}), omitiendo.")
        return None

    print("\n  [AttentiveFP] Preparing molecular graphs...")
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')

    def smiles_to_graph(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return mol_to_bigraph(mol, add_self_loop=True, node_featurizer=atom_featurizer, bond_featurizer=bond_featurizer)

    class MolDataset(Dataset):
        def __init__(self, smi_list, labels):
            valid = [(smiles_to_graph(s), l) for s, l in zip(smi_list, labels) if smiles_to_graph(s) is not None]
            self.graphs = [v[0] for v in valid]
            self.labels = torch.tensor([v[1] for v in valid], dtype=torch.float32).unsqueeze(1)
        def __len__(self): return len(self.graphs)
        def __getitem__(self, i): return self.graphs[i], self.labels[i]

    def collate(batch):
        gs, ls = zip(*batch)
        return dgl.batch(gs), torch.stack(ls)

    smiles_train = [smiles[i] for i in train_idx]
    y_train      = y[train_idx]
    smiles_test  = [smiles[i] for i in test_idx]
    y_test       = y[test_idx]

    train_ds = MolDataset(smiles_train, y_train)
    test_ds  = MolDataset(smiles_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=8)
    test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [AttentiveFP] Training on {device}...")

    model = AttentiveFPPredictor(node_feat_size=atom_featurizer.feat_size('hv'), edge_feat_size=bond_featurizer.feat_size('he'), num_layers=2, num_timesteps=2, graph_feat_size=200, n_tasks=1, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = torch.nn.MSELoss()

    for epoch in range(100):
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
            pred = model(g, g.ndata['hv'], g.edata['he'])
            all_preds.append(pred.cpu().numpy())
            all_labels.append(lab.numpy())

    y_pred_arr = np.vstack(all_preds).flatten()
    y_test_arr = np.vstack(all_labels).flatten()

    r2 = r2_score(y_test_arr, y_pred_arr)
    mae = mean_absolute_error(y_test_arr, y_pred_arr)
    
    # Simple bootstrap for single split models to get CI
    rng = np.random.RandomState(RANDOM_STATE)
    boots = [r2_score(y_test_arr[idx := rng.randint(0, len(y_test_arr), len(y_test_arr))], y_pred_arr[idx]) for _ in range(N_BOOTSTRAP)]
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    print(f"  [AttentiveFP] R2={r2:.4f} | MAE={mae:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    std_r2 = float(np.std(boots)) # Calculate std from bootstrap
    return dict(model='AttentiveFP', r2=r2, std=std_r2, mae=mae, ci_lo=ci_lo, ci_hi=ci_hi, boots=boots, y_test=y_test_arr, y_pred=y_pred_arr)


# =========================================================
# UTILIDADES
# =========================================================
def load_data():
    df = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    df = df.reset_index(drop=True)
    return df['Smiles'].tolist(), df['pIC50 Value'].values

def export_latex(results):
    path = os.path.join(LATEX_DIR, "gnn_variables.tex")
    with open(path, 'w') as f:
        for res in results:
            if res is None: continue

            label = re.sub(r'[^A-Za-z]', '', res['model'])

            # Export Mean R2 and MAE
            f.write(f"\\newcommand{{\\{label}RTwoMean}}{{{res['r2']:.4f}}}\n")
            f.write(f"\\newcommand{{\\{label}Mae}}{{{res['mae']:.4f}}}\n")

            # Export Standard Deviation and Confidence Intervals
            if 'std' in res:
                f.write(f"\\newcommand{{\\{label}RTwoStd}}{{{res['std']:.4f}}}\n")
            if 'ci_lo' in res and 'ci_hi' in res:
                f.write(f"\\newcommand{{\\{label}CILow}}{{{res['ci_lo']:.4f}}}\n")
                f.write(f"\\newcommand{{\\{label}CIHigh}}{{{res['ci_hi']:.4f}}}\n")

def export_figure(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    labels, means, errs = [], [], []

    for name, r2 in CLASSICAL_R2.items():
        labels.append(name)
        means.append(r2)
        errs.append(0.05)

    for res in results:
        if res is not None:
            labels.append(res['model'])
            means.append(res['r2'])
            errs.append((res['ci_hi'] - res['ci_lo']) / 2)

    colors = ['#4C72B0'] * len(CLASSICAL_R2) + ['#C44E52'] * sum(1 for r in results if r is not None)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=5, color=colors, alpha=0.85)

    for label, (desc, val) in PAPER_R2.items():
        ax.axhline(val, linestyle='--', linewidth=1.2, alpha=0.7, label=f"{label} ({val})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel("R²")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    best_gnn = max([r for r in results if r is not None], key=lambda r: r['r2'], default=None)
    if best_gnn is not None:
        ax2.scatter(best_gnn['y_test'], best_gnn['y_pred'], alpha=0.6, s=30, color='#C44E52')
        mn, mx = min(best_gnn['y_test']), max(best_gnn['y_test'])
        ax2.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        ax2.set_xlabel("Experimental pIC₅₀")
        ax2.set_ylabel("Predicted pIC₅₀")
        ax2.set_title(f"{best_gnn['model']} (R²={best_gnn['r2']:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "gnn_comparison.png"), dpi=300)
    plt.close(fig)

def print_summary(results):
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    valid = [r for r in results if r is not None]
    for res in valid:
        print(f"{res['model']:<30} R2: {res['r2']:.4f} | MAE: {res['mae']:.4f}")

# =========================================================
# MAIN
# =========================================================
def run():
    print("Iniciando 5GNNBASELINE.py con soporte Multi-Core (24 Núcleos)...")
    smiles, y = load_data()
    idx = np.arange(len(smiles))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    results = []

    print("\n─" * 80)
    t0 = time.time()
    res_chemprop = run_chemprop_15fold(smiles, y, n_splits=15, random_state=RANDOM_STATE)
    print(f"  Tiempo total: {time.time()-t0:.1f}s")
    results.append(res_chemprop)

    print("\n─" * 80)
    t0 = time.time()
    res_attentive = run_attentivefp(smiles, y, test_idx, train_idx)
    print(f"  Tiempo total: {time.time()-t0:.1f}s")
    results.append(res_attentive)

    print_summary(results)
    export_latex(results)
    export_figure(results)
    
    valid_rows = [{'Model': r['model'], 'R2': r['r2'], 'MAE': r['mae']} for r in results if r is not None]
    if valid_rows:
        pd.DataFrame(valid_rows).to_csv(os.path.join(RESULTS_DIR, "gnn_results.csv"), index=False)

if __name__ == "__main__":
    run()
