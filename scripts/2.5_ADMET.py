import concurrent.futures
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from tdc.oracles import Oracle
from tdc.single_pred import ADME

# Vincular el directorio raíz para importar paths_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import FDA_ADMET_CANDIDATES_CSV, FDA_RAW_CANDIDATES_CSV

# Configuración de rutas centralizadas desde paths_config
INPUT_PATH = FDA_RAW_CANDIDATES_CSV
OUTPUT_PATH = FDA_ADMET_CANDIDATES_CSV
CORES = 48


def lipinski_pass(smiles):
  """Evalúa la regla de los 5 de Lipinski."""
  mol = Chem.MolFromSmiles(smiles)
  if not mol:
    return False
  return (
      Descriptors.NumHDonors(mol) <= 5
      and Descriptors.NumHAcceptors(mol) <= 10
      and Descriptors.MolLogP(mol) <= 5
  )


def evaluate_smiles(row):
  """Función envoltura para procesar cada fila en paralelo."""
  smiles = row["SMILES"]
  return smiles if lipinski_pass(smiles) else None


def smiles_to_fps(smiles_list, radius=2, n_bits=2048):
  """Genera huellas moleculares Morgan en formato Array de Numpy."""
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
    print(f"Error: Archivo de entrada no encontrado en {INPUT_PATH}")
    return

  df = pd.read_csv(INPUT_PATH)
  print(f"Total de candidatos iniciales: {len(df)}")
  print(f"Iniciando filtrado Lipinski en {CORES} núcleos...")

  # 1. Filtrado de Lipinski en paralelo
  valid_smiles = set()
  with concurrent.futures.ProcessPoolExecutor(max_workers=CORES) as executor:
    results = executor.map(evaluate_smiles, [row for _, row in df.iterrows()])
    for res in results:
      if res is not None:
        valid_smiles.add(res)

  df_filtered = df[df["SMILES"].isin(valid_smiles)].copy()
  print(f"Candidatos post-Lipinski: {len(df_filtered)}")

  if df_filtered.empty:
    print("Ningún candidato superó el filtro de Lipinski.")
    return

  smiles_candidates = df_filtered["SMILES"].tolist()

  # 2. Predicción de toxicidad hERG con PyTDC
  print("Ejecutando oráculo de toxicidad hERG de TDC...")
  herg_oracle = Oracle(name="hERG")
  df_filtered["hERG_Blocker_Prob"] = [
      herg_oracle(s) for s in smiles_candidates
  ]

  # 3. Entrenamiento al vuelo y predicción de Caco-2
  print(
      "Descargando dataset Caco2_Wang de TDC y entrenando modelo surrogado"
      " (Morgan FP + RF)..."
  )
  caco_dataset = ADME(name="Caco2_Wang").get_data()
  caco_smiles = caco_dataset["Drug"].tolist()
  caco_labels = caco_dataset["Y"].values

  X_train_caco = smiles_to_fps(caco_smiles)
  rf_caco = RandomForestRegressor(
      n_estimators=100, random_state=42, n_jobs=CORES
  )
  rf_caco.fit(X_train_caco, caco_labels)

  X_cand = smiles_to_fps(smiles_candidates)
  df_filtered["Caco2_Permeability"] = rf_caco.predict(X_cand)

  # 4. Filtro final de farmacocinética / toxicidad y exportación
  df_final = df_filtered[
      (df_filtered["Caco2_Permeability"] > -5.15)
      & (df_filtered["hERG_Blocker_Prob"] < 0.5)
  ]

  os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
  df_final.to_csv(OUTPUT_PATH, index=False)
  print(f"✅ Candidatos finales guardados en {OUTPUT_PATH}: {len(df_final)}")


if __name__ == "__main__":
  main()
