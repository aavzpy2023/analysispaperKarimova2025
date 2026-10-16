import concurrent.futures
import os
import sys
import time
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tdc.single_pred import ADME, Tox

# Link root directory to import paths_config and logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import *
from logger_utils import setup_logger, explain

RDLogger.DisableLog('rdApp.*')

# =========================================================
# LOGGING SETUP
# =========================================================
log_file_path = os.path.join(LOGS_DIR, "08_admet_profiling.log")
# setup_logger() is called inside main(): a module-level call would be re-executed (and truncate the log)
# in every worker process when multiprocessing uses "spawn" (macOS / Windows).

# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = DOCKING_RESULTS_CSV
OUTPUT_PATH = FDA_ADMET_CANDIDATES_CSV


def parent_smiles(smiles):
    """Largest fragment of the input, so that counter-ions of salt forms (e.g. glucuronate) do not distort the
    Lipinski counts or the fingerprints; docking already used the largest fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(rdMolStandardize.LargestFragmentChooser().choose(mol))


def lipinski_pass(smiles):
    """Evaluates three Lipinski criteria (H-bond donors, acceptors, logP). Molecular weight is NOT evaluated here."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    return (
            Descriptors.NumHDonors(mol) <= LIPINSKI_MAX_HDONORS
            and Descriptors.NumHAcceptors(mol) <= LIPINSKI_MAX_HACCEPTORS
            and Descriptors.MolLogP(mol) <= LIPINSKI_MAX_LOGP
    )


def evaluate_smiles(row):
    """Wrapper function to process each row in parallel."""
    smiles = row["Parent_SMILES"]
    return smiles if lipinski_pass(smiles) else None


def smiles_to_fps(smiles_list, radius=2, n_bits=2048):
    """Generates Morgan fingerprints as Numpy arrays using the modern RDKit Generator."""
    fps = []
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            fps.append(fp)
        else:
            fps.append(np.zeros(n_bits))
    return np.array(fps)


def main():
    setup_logger(log_file_path)
    t0 = time.time()

    print("=" * 90)
    print("08_admet_profiling.py — ADMET Profiling & Toxicity Risk Assessment")
    print(f"Log file: {log_file_path}")
    print("=" * 90)
    explain("What this step does", f"""
        A molecule that binds the target is useless as a drug if it is toxic or cannot reach the bloodstream. This step
        screens the docked candidates for three drug-likeness properties, using fast models trained on public data (TDC):

        1) Lipinski rule of five: a rule of thumb for oral drugs (limits on hydrogen-bond donors <= {LIPINSKI_MAX_HDONORS},
        acceptors <= {LIPINSKI_MAX_HACCEPTORS}, logP <= {LIPINSKI_MAX_LOGP}, plus molecular weight). Molecules that break it
        tend to be poorly absorbed.
        2) hERG: probability that the drug blocks the hERG potassium channel in the heart, a main cause of dangerous heart
        rhythm side effects (QT prolongation). Probability < {HERG_THRESH} is treated as low risk. Candidates above it are removed.
        3) Caco-2: predicted permeability through a model of the intestinal wall (log cm/s, less negative = better absorbed).
        Above {CACO2_THRESH} = moderate/high oral absorption; below it the drug is labelled 'Parenteral Candidate' (better
        given by injection) but is NOT removed.

        Limitations: both models are Random Forests on molecular fingerprints trained on a few hundred to a few thousand
        compounds, so values are screening-level estimates, not experimental toxicity. Not modelled: liver metabolism,
        blood-brain barrier (important for Toxoplasma, which infects the brain), and other off-target toxicity.
    """)

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found at {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    df["Parent_SMILES"] = df["SMILES"].map(parent_smiles)
    print(f"[1] Total initial candidates from docking: {len(df)}")
    print(f"[1] Evaluating Lipinski rule of five on {CORES_ADMET} cores...")

    # 1. Lipinski Filtering / Tagging
    valid_smiles = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=CORES_ADMET) as executor:
        results = executor.map(evaluate_smiles, [row for _, row in df.iterrows()])
        for res in results:
            if res is not None:
                valid_smiles.add(res)

    df_filtered = df[df["Parent_SMILES"].isin(valid_smiles)].copy()
    print(f"[1] Candidates passing Lipinski: {len(df_filtered)}")

    if df_filtered.empty:
        print("[ERROR] No candidates passed Lipinski filter.")
        sys.exit(1)

    smiles_candidates = df_filtered["Parent_SMILES"].tolist()
    X_cand = smiles_to_fps(smiles_candidates)

    # 2. hERG Training and Prediction (Cardiotoxicity Classifier)
    herg_dataset = Tox(name="hERG", path=DATA_DIR).get_data()
    print(f"\n[2] Training hERG Cardiotoxicity Classifier (TDC Dataset: {len(herg_dataset)} samples)...")
    X_train_herg = smiles_to_fps(herg_dataset["Drug"].tolist())
    y_train_herg = herg_dataset["Y"].values

    rf_herg = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=CORES_ADMET)
    rf_herg.fit(X_train_herg, y_train_herg)
    df_filtered["hERG_Blocker_Prob"] = rf_herg.predict_proba(X_cand)[:, 1]

    # 3. Caco-2 Training and Prediction (Intestinal Permeability Regressor)
    caco_dataset = ADME(name="Caco2_Wang", path=DATA_DIR).get_data()
    print(f"\n[3] Training Caco-2 Permeability Regressor (TDC Dataset: {len(caco_dataset)} samples)...")
    X_train_caco = smiles_to_fps(caco_dataset["Drug"].tolist())
    y_train_caco = caco_dataset["Y"].values

    rf_caco = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=CORES_ADMET)
    rf_caco.fit(X_train_caco, y_train_caco)
    df_filtered["Caco2_Permeability"] = rf_caco.predict(X_cand)

    # 4. ADMET Annotation & Non-Destructive Filtering
    df_filtered["hERG_Pass"] = df_filtered["hERG_Blocker_Prob"] < HERG_THRESH
    df_filtered["Caco2_Pass"] = df_filtered["Caco2_Permeability"] > CACO2_THRESH

    # Categorize predicted route/profile
    conditions = [
        (df_filtered["hERG_Pass"] & df_filtered["Caco2_Pass"]),
        (df_filtered["hERG_Pass"] & ~df_filtered["Caco2_Pass"]),
        (~df_filtered["hERG_Pass"])
    ]
    choices = [
        "ADMET-Compliant (Oral)",
        "Parenteral Candidate (Low Caco-2)",
        "Flagged: High hERG Cardiotoxicity Risk"
    ]
    df_filtered["ADMET_Profile"] = np.select(conditions, choices, default="Unclassified")

    # SAVE 1: Full Dataset (Pre-Filter) for Analytics/Plotting
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    full_output_path = OUTPUT_PATH.replace(".csv", "_FULL_PRE_FILTER.csv")
    df_filtered.to_csv(full_output_path, index=False)
    print(f"\n[EXPORT] Full analytical dataset (for plotting) saved to {full_output_path}")

    # SAVE 2: Filtered Dataset (Only Safe Candidates)
    df_final = df_filtered[df_filtered["hERG_Pass"]].copy()

    # Calculate funnel stats
    flagged_herg = len(df_filtered) - len(df_final)
    print(f"\n[4] Flagged by hERG toxicity: {flagged_herg} compounds")

    df_final.to_csv(OUTPUT_PATH, index=False)

    def canonical(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None else None

    herg_set = {canonical(x) for x in herg_dataset["Drug"]}
    caco_set = {canonical(x) for x in caco_dataset["Drug"]}
    cand_canon = df_filtered["Parent_SMILES"].map(canonical)
    profile = df_filtered["ADMET_Profile"]
    admet_values = {
        "AdmetNDocked": len(df), "AdmetNLipinski": len(df_filtered), "AdmetNHergFlagged": flagged_herg,
        "AdmetNRemaining": len(df_final), "AdmetNOral": int((profile == choices[0]).sum()),
        "AdmetNParenteral": int((profile == choices[1]).sum()),
        "AdmetNHergTrain": len(herg_dataset), "AdmetNCacoTrain": len(caco_dataset),
        "AdmetOverlapHerg": int(cand_canon.isin(herg_set).sum()), "AdmetOverlapCaco": int(cand_canon.isin(caco_set).sum()),
        "AdmetHergThresh": HERG_THRESH, "AdmetCacoThresh": CACO2_THRESH,
        "AdmetLipinskiHBD": LIPINSKI_MAX_HDONORS, "AdmetLipinskiHBA": LIPINSKI_MAX_HACCEPTORS,
        "AdmetLipinskiLogP": f"{LIPINSKI_MAX_LOGP:g}",
    }
    consensus = df[df["Agreement"] == "VALIDATED (+)"]
    admet_values["AdmetNConsensus"] = len(consensus.drop_duplicates(["ML_pIC50", "Docking_Score"]))
    survivors = df_final[df_final["Agreement"] == "VALIDATED (+)"]
    admet_values["AdmetNConsensusSurvivors"] = len(survivors.drop_duplicates(["ML_pIC50", "Docking_Score"]))
    for cand in ("Pralatrexate", "Methotrexate", "Triamterene", "Pyrimethamine"):
        row = df_filtered[df_filtered["Name"] == cand]
        if not row.empty:
            admet_values[f"AdmetHerg{cand}"] = f"{row.iloc[0]['hERG_Blocker_Prob']:.2f}"
            admet_values[f"AdmetCaco{cand}"] = f"{row.iloc[0]['Caco2_Permeability']:.2f}"
        else:
            admet_values[f"Admet{cand}FailedLipinski"] = "true"
    os.makedirs(os.path.dirname(LATEX_ADMET), exist_ok=True)
    with open(LATEX_ADMET, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by 08_admet_profiling.py\n")
        for name, value in admet_values.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")

    explain("Reading the summary", f"""
        Of {len(df_filtered)} candidates that passed Lipinski, {flagged_herg} were dropped for hERG risk and {len(df_final)}
        remain. 'ADMET-Compliant (Oral)' = low hERG risk and good predicted absorption. 'Parenteral Candidate' = low hERG risk
        but poor predicted gut permeability. Both lists (before and after the hERG cut) are saved so the full picture can be
        plotted in 08.1.
    """)
    print("\n" + "=" * 105)
    print("ADMET PROFILING RESULTS SUMMARY")
    print("=" * 105)
    for _, r in df_final.iterrows():
        # Truncate and pad the name to precisely 28 characters to maintain alignment
        safe_name = str(r['Name'])[:28]
        print(f"{safe_name:<28} | hERG Prob: {r['hERG_Blocker_Prob']:.3f} | "
              f"Caco-2: {r['Caco2_Permeability']:>6.2f} | Profile: {r['ADMET_Profile']}")

    print(f"\n[EXPORT] Safe candidates ({len(df_final)}) saved to {OUTPUT_PATH}")

    # Calculate and format total execution time
    t_elapsed = time.time() - t0
    mins, secs = divmod(t_elapsed, 60)
    print(f"[TIME] Total execution: {int(mins)}m {int(secs)}s")
    print("[DONE] 08_admet_profiling.py complete.")


if __name__ == "__main__":
    main()