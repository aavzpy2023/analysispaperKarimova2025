import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import warnings
import time
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import IsolationForest

# ── Configuration and Logging Integration ────────────────────────────────────
from paths_config import *
from logger_utils import setup_logger, explain

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

LOG_FILE = os.path.join(LOGS_DIR, "05_virtual_screening.log")
OUTPUT_CSV = FDA_CANDIDATES_CSV


# =========================================================
# MOLECULAR REPRESENTATION
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
# LIGAND EFFICIENCY METRICS
# =========================================================
def ligand_efficiency(pic50, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(LE=np.nan, BEI=np.nan, LLE=np.nan, SEI=np.nan)
    hac = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    psa = Descriptors.TPSA(mol)
    le = 1.37 * pic50 / hac if hac > 0 else np.nan
    bei = pic50 / mw * 1000 if mw > 0 else np.nan
    lle = pic50 - logp
    sei = pic50 / psa * 100 if psa > 0 else np.nan
    return dict(LE=round(le, 2), BEI=round(bei, 2), LLE=round(lle, 2), SEI=round(sei, 2))


def find_control(df_res, drug):
    """Exact (case-insensitive) name match first, so that 'Trimethoprim' is not resolved to a combination product."""
    exact = df_res[df_res['Name'].str.lower() == drug.lower()]
    return exact if not exact.empty else df_res[df_res['Name'].str.contains(drug, case=False, na=False)]


# =========================================================
# MAIN PIPELINE
# =========================================================
def run():
    # 1. Initialize the agnostic logger
    setup_logger(LOG_FILE)

    start_pipeline = time.time()
    print("=" * 100)
    print("VIRTUAL SCREENING PIPELINE — TgDHFR Inhibitor Search")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    explain("What this step does", """
        Uses the trained model from step 04 to predict how strongly each FDA-approved drug would inhibit the parasite enzyme
        TgDHFR (Toxoplasma gondii dihydrofolate reductase), without doing any lab work (virtual screening).

        1) Load the model. 2) Learn what the training molecules look like and flag drugs that are too different from them
        (Applicability Domain, via an Isolation Forest): the model is only trustworthy for molecules similar to those it was
        trained on, so unusual drugs are dropped instead of receiving a made-up prediction. 3) Drop drugs that are very large
        (MW above the limit) or contain atoms outside the allowed set (H, C, N, O, F, P, S, Cl, Br, I). 4) Predict pIC50.
        5) Report and export the ranking.

        pIC50 = -log10(IC50 in molar). Higher means more potent: 6 is IC50 = 1 uM, 7 is 100 nM, 8 is 10 nM.
        Candidates are later confirmed by docking (step 07) and filtered for drug-likeness/safety (step 08).
    """)

    # ── ML Architecture Loading ──────────────────────────────────────────────
    print(f"\n[1/5] ML ARCHITECTURE LOADING")
    print(f"  [-] Model path: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        print(f"  [ERROR] {MODEL_FILE} not found. Interrupting execution.")
        sys.exit(1)

    t0 = time.time()
    model = joblib.load(MODEL_FILE)
    print(f"  [+] Ensembled model loaded successfully ({time.time() - t0:.2f}s)")

    mask = None
    if os.path.exists(MASK_FILE):
        mask = np.load(MASK_FILE)
        n_selected = mask.sum()
        total_feats = len(mask)
        print(f"  [-] Mask path: {MASK_FILE}")
        print(
            f"  [+] Mask applied: {n_selected} / {total_feats} features ({(n_selected / total_feats) * 100:.2f}% retention)")
    else:
        print("  [!] No mask detected. Using full 2D Morgan topology (2048 dimensions).")

    # ── Applicability Domain Configuration ────────────────────────────────────
    print(f"\n[2/5] APPLICABILITY DOMAIN CONFIGURATION (ISOLATION FOREST)")
    print(f"  [-] Training source: {TRAIN_FILE}")
    t0 = time.time()

    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    train_mols = [Chem.MolFromSmiles(s) for s in df_train['Smiles']]
    X_train_full = np.array([get_morgan_fp(m) for m in train_mols])
    X_train = X_train_full[:, mask] if mask is not None else X_train_full

    iso = IsolationForest(contamination=ISO_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train)

    print(f"  [+] Training matrix generated: {X_train.shape}")
    print(f"  [+] Isolation Forest fitted (Contamination: {ISO_CONTAMINATION * 100}%) in {time.time() - t0:.2f}s")
    explain("Applicability Domain", f"""
        The Isolation Forest learns the 'normal' region of chemical space covered by the {len(df_train)} training molecules.
        By construction about {ISO_CONTAMINATION * 100:.0f}% of the training molecules themselves sit at the edge and are treated
        as outliers. Any FDA drug that falls outside this region is excluded below, because the model would be extrapolating.
    """)

    # ── FDA Library Ingestion and Structural Filtering ────────────────────────
    print(f"\n[3/5] FDA LIBRARY PROCESSING")
    print(f"  [-] Screening source: {FDA_FILE}")
    if not os.path.exists(FDA_FILE):
        print(f"  [ERROR] {FDA_FILE} not found.")
        sys.exit(1)

    df_fda = pd.read_csv(FDA_FILE)
    df_fda.columns = df_fda.columns.str.strip()
    initial_fda_count = len(df_fda)
    print(f"  [+] Raw compounds imported: {initial_fda_count}")

    t0 = time.time()
    candidates = []
    for _, row in df_fda.iterrows():
        smi = row.get('isosmiles') or row.get('canonicalsmiles')
        if pd.isna(smi): continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None: continue

        mw = row.get('mw')
        mw = Descriptors.MolWt(mol) if (pd.isna(mw) or mw == '') else float(mw)

        if mw > MAX_MW or not check_atoms(mol):
            continue

        fp = get_morgan_fp(mol)
        candidates.append({
            'CID': row.get('cid'), 'Name': row.get('cmpdname'),
            'SMILES': str(smi), 'FP': fp,
        })

    structural_retention = (len(candidates) / initial_fda_count) * 100
    print(
        f"  [+] Physical/Structural Filtering (MW < {MAX_MW}, Allowed Atoms): {len(candidates)} candidates retained ({structural_retention:.1f}%)")

    # ── In-Domain Filtering ───────────────────────────────────────────────────
    X_fda_full = np.array([c['FP'] for c in candidates])
    X_fda = X_fda_full[:, mask] if mask is not None else X_fda_full

    iso_labels = iso.predict(X_fda)
    final = [candidates[i] for i in range(len(candidates)) if iso_labels[i] == 1]
    X_final = np.array([c['FP'] for c in final])
    X_final = X_final[:, mask] if mask is not None else X_final

    ad_retention = (len(final) / len(candidates)) * 100
    print(f"  [+] In-Domain Filtering (Isolation Forest): {len(final)} compounds retained ({ad_retention:.1f}%)")
    print(f"  [+] Total filtering time: {time.time() - t0:.2f}s")
    explain("Reading the filter counts", f"""
        {initial_fda_count} drugs in the library -> {len(candidates)} after size/atom filters -> {len(final)} inside the model's
        domain. Drugs lost at each stage were never scored. A very low retention in the last stage would mean the FDA library
        is chemically far from the training set (then predictions are less reliable and more training data would help).
    """)

    # ── ML Inference ──────────────────────────────────────────────────────────
    print(f"\n[4/5] ML INFERENCE - BIOACTIVITY PREDICTION (pIC50)")
    t0 = time.time()
    preds = model.predict(X_final)
    if preds.ndim > 1:
        preds = preds.flatten()

    print(f"  [+] Inferences completed in {time.time() - t0:.3f}s")
    print(f"  [-] Prediction Statistics -> Mean: {preds.mean():.4f} | Max: {preds.max():.4f} | Min: {preds.min():.4f}")

    # Result dataframe construction
    rows = []
    for i, c in enumerate(final):
        metrics = ligand_efficiency(preds[i], c['SMILES'])
        rows.append({
            'CID': c['CID'], 'Name': c['Name'], 'SMILES': c['SMILES'],
            'pIC50_pred': round(float(preds[i]), 4),
            **metrics,
        })
    df_res = pd.DataFrame(rows).sort_values('pIC50_pred', ascending=False).reset_index(drop=True)

    # ── Reports and Export ────────────────────────────────────────────────────
    print(f"\n[5/5] PERFORMANCE REPORTS AND EXPORT")
    explain("Report A: sanity check against known drugs", """
        Pyrimethamine and trimethoprim are classic DHFR inhibitors with known experimental pIC50. If the model predicts them
        close to the lab value it is behaving sensibly. Status: absolute error < 0.5 = EXCELLENT, < 1.0 = ACCEPTABLE, else ALERT.
        Caution: if these drugs (or close analogues) are in the training set, a good match is partly memorisation and not a
        true test of generalisation.
    """)
    print("\n" + "-" * 80)
    print("REPORT A: Internal Validation vs Known TgDHFR Controls")
    print("-" * 80)
    print(f"{'Drug':<18} | {'pIC50 Exp.':>12} | {'pIC50 Pred.':>12} | {'Abs Error':>10} | {'Status'}")
    print("-" * 80)
    for drug, real in KNOWN_VALUES.items():
        match = find_control(df_res, drug)
        if not match.empty:
            pred = match.iloc[0]['pIC50_pred']
            err = abs(real - pred)
            status = "EXCELLENT" if err < 0.5 else "ACCEPTABLE" if err < 1.0 else "ALERT"
            print(f"{drug:<18} | {real:>12.2f} | {pred:>12.4f} | {err:>10.4f} | {status}")
        else:
            print(f"{drug:<18} | [Excluded in previous filters]")

    explain("Report B: ranking and efficiency metrics", """
        Besides predicted potency (pIC50), the table gives metrics that judge whether the potency is 'earned' efficiently:
        LE (ligand efficiency) = potency per heavy atom, >= 0.3 is generally considered good; BEI = potency per kDa of mass;
        LLE = pIC50 minus logP, >= 5 suggests potency is not just coming from greasiness (lipophilicity); SEI = potency per
        polar surface area. A high pIC50 with poor LE/LLE usually means a big or greasy molecule that is a weaker lead.
    """)
    print("\n" + "-" * 100)
    print("REPORT B: Top 10 FDA Candidates (Sorted by Predicted Affinity)")
    print("-" * 100)
    print(f"{'Rank':<5} {'Compound Name':<30} {'pIC50':>7} {'LE':>6} {'BEI':>7} {'LLE':>6} {'SEI':>6}")
    print("-" * 100)
    for rank, (_, r) in enumerate(df_res.head(10).iterrows(), 1):
        name = str(r['Name'])[:28]
        print(f"{rank:<5} {name:<30} {r['pIC50_pred']:>7.4f} {r['LE']:>6.2f} "
              f"{r['BEI']:>7.2f} {r['LLE']:>6.2f} {r['SEI']:>6.2f}")

    # Safe export
    paper_top_mask = df_res['Name'].str.contains('|'.join(PAPER_TOP), case=False, na=False)

    top10 = df_res.head(10).copy()
    top10['Type'] = '[TOP_ML]'

    refs = df_res[paper_top_mask].copy()
    refs['Type'] = '[REFERENCE]'

    combined = pd.concat([top10, refs]).drop_duplicates('CID').sort_values('pIC50_pred', ascending=False)

    # Write to agnostic CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    os.makedirs(os.path.dirname(LATEX_SCREENING), exist_ok=True)
    with open(LATEX_SCREENING, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by 05_virtual_screening.py\n")
        counts = {"ScreenNLibrary": initial_fda_count, "ScreenNStructural": len(candidates), "ScreenNDomain": len(final),
                  "ScreenNCandidates": len(combined), "ScreenNTopML": int((combined['Type'] == '[TOP_ML]').sum()),
                  "ScreenNReference": int((combined['Type'] == '[REFERENCE]').sum()),
                  "ScreenNTrainMols": len(df_train), "ScreenMaskBits": int(mask.sum()) if mask is not None else 2048,
                  "ScreenMaxMW": f"{MAX_MW:.0f}", "ScreenIsoContamination": f"{ISO_CONTAMINATION * 100:.0f}",
                  "ScreenTrainMaxMW": f"{max(Descriptors.MolWt(m) for m in train_mols):.0f}",
                  "ScreenTrainNOutsideElements": sum(1 for m in train_mols if not check_atoms(m))}
        for name, value in counts.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
        for cand in ("Methotrexate", "Trimetrexate", "Pralatrexate", "Triamterene", "Pyrimethamine"):
            row = find_control(df_res, cand)
            if not row.empty:
                r0 = row.iloc[0]
                f.write(f"\\newcommand{{\\Cand{cand}Pred}}{{{r0['pIC50_pred']:.2f}}}\n")
                f.write(f"\\newcommand{{\\Cand{cand}LE}}{{{r0['LE']:.2f}}}\n")
                f.write(f"\\newcommand{{\\Cand{cand}BEI}}{{{r0['BEI']:.1f}}}\n")
        antifolate_le = [find_control(df_res, n).iloc[0]['LE'] for n in ("Methotrexate", "Trimetrexate", "Pralatrexate")
                         if not find_control(df_res, n).empty]
        f.write(f"\\newcommand{{\\AntifolateLEMin}}{{{min(antifolate_le):.2f}}}\n")
        f.write(f"\\newcommand{{\\AntifolateLEMax}}{{{max(antifolate_le):.2f}}}\n")
        for drug, real in KNOWN_VALUES.items():
            match = find_control(df_res, drug)
            if not match.empty:
                f.write(f"\\newcommand{{\\Control{drug}Exp}}{{{real:.2f}}}\n")
                f.write(f"\\newcommand{{\\Control{drug}Pred}}{{{match.iloc[0]['pIC50_pred']:.2f}}}\n")
                f.write(f"\\newcommand{{\\Control{drug}Err}}{{{abs(real - match.iloc[0]['pIC50_pred']):.2f}}}\n")

    print("\n" + "=" * 100)
    print(f"[SUCCESS] {len(combined)} candidates safely exported to:")
    print(f"        -> {OUTPUT_CSV}")
    print(f"        Logs saved in: {LOG_FILE}")
    print(f"        Total execution time: {time.time() - start_pipeline:.2f}s")
    print("=" * 100)


if __name__ == "__main__":
    run()