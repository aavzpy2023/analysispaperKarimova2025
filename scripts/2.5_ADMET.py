import concurrent.futures
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from tdc.single_pred import ADME, Tox

# Link root directory to import paths_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import FDA_ADMET_CANDIDATES_CSV, FDA_RAW_CANDIDATES_CSV

INPUT_PATH = FDA_RAW_CANDIDATES_CSV
OUTPUT_PATH = FDA_ADMET_CANDIDATES_CSV
CORES = 48


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
    """Generates Morgan fingerprints as Numpy arrays."""
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(n_bits))
    return np.array(fps)


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Total initial candidates: {len(df)}")
    print(f"Starting Lipinski filtering on {CORES} cores...")

    # 1. Lipinski Filtering
    valid_smiles = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=CORES) as executor:
        results = executor.map(evaluate_smiles, [row for _, row in df.iterrows()])
        for res in results:
            if res is not None:
                valid_smiles.add(res)

    df_filtered = df[df["SMILES"].isin(valid_smiles)].copy()
    print(f"Candidates post-Lipinski: {len(df_filtered)}")

    if df_filtered.empty:
        print("No candidates passed the Lipinski filter.")
        return

    smiles_candidates = df_filtered["SMILES"].tolist()
    X_cand = smiles_to_fps(smiles_candidates)

    # 2. hERG Training and Prediction (Classification)
    print("Downloading hERG TDC dataset and training classifier...")
    herg_dataset = Tox(name="hERG").get_data()
    X_train_herg = smiles_to_fps(herg_dataset["Drug"].tolist())
    y_train_herg = herg_dataset["Y"].values

    rf_herg = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=CORES)
    rf_herg.fit(X_train_herg, y_train_herg)
    # predict_proba[:, 1] returns the probability of being a hERG blocker (class 1)
    df_filtered["hERG_Blocker_Prob"] = rf_herg.predict_proba(X_cand)[:, 1]

    # 3. Caco-2 Training and Prediction (Regression)
    print("Downloading Caco2_Wang TDC dataset and training regressor...")
    caco_dataset = ADME(name="Caco2_Wang").get_data()
    X_train_caco = smiles_to_fps(caco_dataset["Drug"].tolist())
    y_train_caco = caco_dataset["Y"].values

    rf_caco = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=CORES)
    rf_caco.fit(X_train_caco, y_train_caco)
    df_filtered["Caco2_Permeability"] = rf_caco.predict(X_cand)

    # 4. Final Filter and Export
    # Caco2 > -5.15 log(cm/s) (moderate/high permeability)
    # hERG prob < 0.5 (low probability of cardiac toxicity)
    df_final = df_filtered[
        (df_filtered["Caco2_Permeability"] > -5.15)
        & (df_filtered["hERG_Blocker_Prob"] < 0.5)
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Final candidates saved to {OUTPUT_PATH}: {len(df_final)}")


if __name__ == "__main__":
    main()
