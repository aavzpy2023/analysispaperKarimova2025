import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
import time
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import IsolationForest

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
TRAIN_FILE       = "./V2-df_ic50_chmbl_CID_myFill.csv"
FDA_FILE         = "PubChem_FDA-approved_NoInorganics.csv"
MODEL_FILE       = "best_model.joblib"        # saved by 1AUGMENT.py
MASK_FILE        = "selected_features_mask.npy"  # saved by 1AUGMENT.py
OUTPUT_CSV       = "FDA_Candidates_For_Docking.csv"
RANDOM_STATE     = 42
ISO_CONTAMINATION = 0.05
MAX_MW           = 1000.0

# Allowed atom numbers — identical to original paper
ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

# Reference drugs with known experimental pIC50 (for accuracy validation)
KNOWN_VALUES = {
    'Pyrimethamine': 6.56,
    'Trimethoprim':  5.57,
}

# Paper top candidates (used as reference rows in output)
PAPER_TOP = ['Bisacodyl', 'Etodolac', 'Triamterene', 'Lorlatinib',
             'Finerenone', 'Methotrexate', 'Pyrimethamine', 'Trimethoprim']


# =========================================================
# MOLECULAR REPRESENTATION  (Morgan FP — same as 0STACK + 1AUGMENT)
# =========================================================
_MORGAN_GEN = None

def _get_generator():
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _MORGAN_GEN

def get_morgan_fp(mol):
    if mol is None:
        return np.zeros((2048,), dtype=np.float32)
    try:
        return _get_generator().GetFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        return np.zeros((2048,), dtype=np.float32)

def check_atoms(mol):
    if mol is None:
        return False
    return all(a.GetAtomicNum() in ALLOWED_ATOMS for a in mol.GetAtoms())


# =========================================================
# LIGAND EFFICIENCY METRICS  (identical to paper Equations 5-8)
# =========================================================
def ligand_efficiency(pic50, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(LE=np.nan, BEI=np.nan, LLE=np.nan, SEI=np.nan)
    hac  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    psa  = Descriptors.TPSA(mol)
    le   = 1.37 * pic50 / hac if hac > 0 else np.nan
    bei  = pic50 / mw * 1000  if mw  > 0 else np.nan
    lle  = pic50 - logp
    sei  = pic50 / psa * 100  if psa > 0 else np.nan
    return dict(LE=round(le,2), BEI=round(bei,2), LLE=round(lle,2), SEI=round(sei,2))


# =========================================================
# MAIN
# =========================================================
def run():
    print("=" * 100)
    print("2FDA.py — FDA Drug Screening with Winner Classical ML Model")
    print("=" * 100)

    # ── Load model and feature mask ───────────────────────────────────────────
    print(f"\n[1] Loading model from {MODEL_FILE}...")
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] {MODEL_FILE} not found. Run 1AUGMENT.py first.")
        return
    model = joblib.load(MODEL_FILE)

    mask = None
    if os.path.exists(MASK_FILE):
        mask = np.load(MASK_FILE)
        print(f"[1] Feature mask loaded: {mask.sum()} / {len(mask)} features selected")
    else:
        print("[1] No feature mask found — using all Morgan FP features")

    # ── Build training fingerprints for Isolation Forest ─────────────────────
    print("\n[2] Building training fingerprint matrix for outlier detection...")
    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    train_mols = [Chem.MolFromSmiles(s) for s in df_train['Smiles']]
    X_train_full = np.array([get_morgan_fp(m) for m in train_mols])
    X_train = X_train_full[:, mask] if mask is not None else X_train_full

    iso = IsolationForest(contamination=ISO_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train)
    print(f"[2] Isolation Forest fitted on {len(X_train)} training compounds")

    # ── Load and filter FDA dataset ───────────────────────────────────────────
    print(f"\n[3] Loading FDA dataset from {FDA_FILE}...")
    if not os.path.exists(FDA_FILE):
        print(f"[ERROR] {FDA_FILE} not found.")
        return
    df_fda = pd.read_csv(FDA_FILE)
    df_fda.columns = df_fda.columns.str.strip()
    print(f"[3] Raw FDA compounds: {len(df_fda)}")

    candidates = []
    for _, row in df_fda.iterrows():
        smi = row.get('isosmiles') or row.get('canonicalsmiles')
        if pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        mw = row.get('mw')
        mw = Descriptors.MolWt(mol) if (pd.isna(mw) or mw == '') else float(mw)
        if mw > MAX_MW:
            continue
        if not check_atoms(mol):
            continue
        fp = get_morgan_fp(mol)
        candidates.append({
            'CID': row.get('cid'), 'Name': row.get('cmpdname'),
            'SMILES': str(smi), 'FP': fp,
        })

    print(f"[3] After atom/MW filter: {len(candidates)} compounds")

    X_fda_full = np.array([c['FP'] for c in candidates])
    X_fda = X_fda_full[:, mask] if mask is not None else X_fda_full
    iso_labels = iso.predict(X_fda)
    final = [candidates[i] for i in range(len(candidates)) if iso_labels[i] == 1]
    X_final = np.array([c['FP'] for c in final])
    X_final = X_final[:, mask] if mask is not None else X_final
    print(f"[3] After Isolation Forest (contamination={ISO_CONTAMINATION}): {len(final)} compounds")

    # ── Predict pIC50 ─────────────────────────────────────────────────────────
    print("\n[4] Predicting pIC50 for all filtered FDA candidates...")
    t0 = time.time()
    preds = model.predict(X_final)
    if preds.ndim > 1:
        preds = preds.flatten()
    print(f"[4] Predictions done ({time.time()-t0:.1f}s)")

    # ── Build results dataframe ───────────────────────────────────────────────
    rows = []
    for i, c in enumerate(final):
        metrics = ligand_efficiency(preds[i], c['SMILES'])
        rows.append({
            'CID': c['CID'], 'Name': c['Name'], 'SMILES': c['SMILES'],
            'pIC50_pred': round(float(preds[i]), 4),
            **metrics,
        })
    df_res = pd.DataFrame(rows).sort_values('pIC50_pred', ascending=False).reset_index(drop=True)

    # ── Accuracy validation against known drugs ───────────────────────────────
    print("\n" + "=" * 80)
    print("REPORT 1 — Accuracy validation against known TgDHFR inhibitors")
    print("=" * 80)
    print(f"{'Drug':<18} | {'Experimental':>12} | {'Predicted':>9} | {'Abs Error':>9} | Status")
    print("-" * 70)
    for drug, real in KNOWN_VALUES.items():
        match = df_res[df_res['Name'].str.contains(drug, case=False, na=False)]
        if not match.empty:
            pred = match.iloc[0]['pIC50_pred']
            err = abs(real - pred)
            status = "EXCELLENT" if err < 0.5 else "ACCEPTABLE" if err < 1.0 else "ALERT"
            print(f"{drug:<18} | {real:>12.2f} | {pred:>9.4f} | {err:>9.4f} | {status}")
        else:
            print(f"{drug:<18} | Not found in filtered set")

    # ── Top 10 ranking ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("REPORT 2 — Top 10 predicted TgDHFR inhibitors (classical ML model)")
    print("=" * 100)
    print(f"{'Rank':<5} {'Name':<30} {'pIC50':>7} {'LE':>6} {'BEI':>7} {'LLE':>6} {'SEI':>6}")
    print("-" * 70)
    for rank, (_, r) in enumerate(df_res.head(10).iterrows(), 1):
        name = str(r['Name'])[:28]
        print(f"{rank:<5} {name:<30} {r['pIC50_pred']:>7.4f} {r['LE']:>6.2f} "
              f"{r['BEI']:>7.2f} {r['LLE']:>6.2f} {r['SEI']:>6.2f}")

    # ── Comparison with paper top candidates ──────────────────────────────────
    print("\n" + "=" * 100)
    print("REPORT 3 — Paper top candidates predicted by our classical ML model")
    print("(Do our top-10 agree with the paper's top-10?)")
    print("=" * 100)
    paper_top_mask = df_res['Name'].str.contains('|'.join(PAPER_TOP), case=False, na=False)
    df_paper_refs = df_res[paper_top_mask].copy()
    if not df_paper_refs.empty:
        print(f"{'Name':<30} {'pIC50 (ours)':>12} {'Paper rank':>10}")
        print("-" * 55)
        for _, r in df_paper_refs.head(10).iterrows():
            rank_ours = df_res.index[df_res['CID'] == r['CID']].tolist()
            rank_str = str(rank_ours[0]+1) if rank_ours else "?"
            print(f"{str(r['Name'])[:28]:<30} {r['pIC50_pred']:>12.4f} {rank_ours[0]+1:>10}")

    # ── Export for docking ────────────────────────────────────────────────────
    top10 = df_res.head(10).copy()
    top10['Type'] = '[TOP]'
    refs = df_res[paper_top_mask].copy()
    refs['Type'] = '[REF]'
    combined = pd.concat([top10, refs]).drop_duplicates('CID').sort_values('pIC50_pred', ascending=False)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[EXPORT] {len(combined)} candidates saved to {OUTPUT_CSV} (for 3DOCKING.py)")
    print("[DONE] 2FDA.py complete.")


if __name__ == "__main__":
    run()