import concurrent.futures
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tdc.single_pred import ADME, Tox

# Link root directory to import paths_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import DOCKING_RESULTS_CSV, FDA_ADMET_CANDIDATES_CSV

INPUT_PATH = DOCKING_RESULTS_CSV
OUTPUT_PATH = FDA_ADMET_CANDIDATES_CSV
CORES = 48

# Thresholds
HERG_THRESH = 0.5         # Probability < 0.5 = Low cardiotoxicity risk
CACO2_THRESH = -5.15      # Permeability > -5.15 log(cm/s) = Moderate/High Oral Permeability


def lipinski_pass(smiles):
    """Evaluates Lipinski's Rule of Five."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    return (
        Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
        and Descriptors.MolLogP(mol) <= 5
    )


def evaluate_smiles(row):
    """Wrapper function to process each row in parallel."""
    smiles = row["SMILES"]
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
    print("=" * 90)
    print("3.5_ADMET.py — ADMET Profiling & Toxicity Risk Assessment")
    print("=" * 90)

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found at {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"[1] Total initial candidates from docking: {len(df)}")
    print(f"[1] Evaluating Lipinski rule of five on {CORES} cores...")

    # 1. Lipinski Filtering / Tagging
    valid_smiles = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=CORES) as executor:
        results = executor.map(evaluate_smiles, [row for _, row in df.iterrows()])
        for res in results:
            if res is not None:
                valid_smiles.add(res)

    df_filtered = df[df["SMILES"].isin(valid_smiles)].copy()
    print(f"[1] Candidates passing Lipinski: {len(df_filtered)}")

    if df_filtered.empty:
        print("[ERROR] No candidates passed Lipinski filter.")
        return

    smiles_candidates = df_filtered["SMILES"].tolist()
    X_cand = smiles_to_fps(smiles_candidates)

    # 2. hERG Training and Prediction (Cardiotoxicity Classifier)
    print("\n[2] Training hERG Cardiotoxicity Classifier (TDC Dataset)...")
    herg_dataset = Tox(name="hERG").get_data()
    X_train_herg = smiles_to_fps(herg_dataset["Drug"].tolist())
    y_train_herg = herg_dataset["Y"].values

    rf_herg = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=CORES)
    rf_herg.fit(X_train_herg, y_train_herg)
    df_filtered["hERG_Blocker_Prob"] = rf_herg.predict_proba(X_cand)[:, 1]

    # 3. Caco-2 Training and Prediction (Intestinal Permeability Regressor)
    print("\n[3] Training Caco-2 Permeability Regressor (TDC Dataset)...")
    caco_dataset = ADME(name="Caco2_Wang").get_data()
    X_train_caco = smiles_to_fps(caco_dataset["Drug"].tolist())
    y_train_caco = caco_dataset["Y"].values

    rf_caco = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=CORES)
    rf_caco.fit(X_train_caco, y_train_caco)
    df_filtered["Caco2_Permeability"] = rf_caco.predict(X_cand)

    # 4. ADMET Annotation & Non-Destructive Filtering
    # - hERG Blocker Prob < 0.5 (Safety filter: excludes cardiotoxic risks)
    # - Caco-2 Permeability used for Route Categorization (Oral vs Parenteral)

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

    # Filter out cardiotoxic compounds, but keep parenteral candidates (like Pralatrexate)
    df_final = df_filtered[df_filtered["hERG_Pass"]].copy()

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 90)
    print("ADMET PROFILING RESULTS SUMMARY")
    print("=" * 90)
    for _, r in df_final.iterrows():
        print(f"{str(r['Name']):<28} | hERG Prob: {r['hERG_Blocker_Prob']:.3f} | "
              f"Caco-2: {r['Caco2_Permeability']:.2f} | Profile: {r['ADMET_Profile']}")

    print(f"\n[EXPORT] Processed dataset ({len(df_final)} compounds) saved to {OUTPUT_PATH}")
    print("[DONE] 3.5_ADMET.py complete.")


if __name__ == "__main__":
    main()
