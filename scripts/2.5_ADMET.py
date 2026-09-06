import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc.single_pred import ADME, Tox
import concurrent.futures

INPUT_PATH = 'results/FDA_Candidates_For_Docking.csv'
OUTPUT_PATH = 'results/ADMET_CANDIDATES_For_Docking.csv'
CORES = 48 # Puedes usar os.cpu_count() para detección automática

def lipinski_pass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    return (Descriptors.NumHDonors(mol) <= 5 and
            Descriptors.NumHAcceptors(mol) <= 10 and
            Descriptors.MolLogP(mol) <= 5)

def evaluate_smiles(row):
    """Función envoltura para procesar cada fila en paralelo."""
    smiles = row['SMILES']
    return row['SMILES'] if lipinski_pass(smiles) else None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Archivo no encontrado en {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Total de candidatos iniciales: {len(df)}")
    print(f"Iniciando filtrado RDKit en {CORES} núcleos...")

    # 1. Procesamiento en paralelo con ProcessPoolExecutor
    valid_smiles = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=CORES) as executor:
        results = executor.map(evaluate_smiles, [row for _, row in df.iterrows()])
        for res in results:
            if res is not None:
                valid_smiles.add(res)

    df_filtered = df[df['SMILES'].isin(valid_smiles)].copy()
    print(f"Candidatos post-Lipinski: {len(df_filtered)}")

    if df_filtered.empty:
        print("Ningún candidato superó el filtro de Lipinski.")
        return

    # 2. Predicciones ADMET (TDC - Procesamiento vectorizado/batch interno)
    print("Ejecutando oráculos de TDC en memoria...")
    caco2_model = ADME(name='Caco2_Wang')
    herg_model = Tox(name='hERG')

    smiles_list = df_filtered['SMILES'].tolist()
    df_filtered['Caco2_Permeability'] = caco2_model.predict(smiles_list)
    df_filtered['hERG_Blocker_Prob'] = herg_model.predict(smiles_list)

    # 3. Filtro final de toxicidad y exportación
    df_final = df_filtered[(df_filtered['Caco2_Permeability'] > -5.15) &
                           (df_filtered['hERG_Blocker_Prob'] < 0.5)]

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Candidatos finales guardados en {OUTPUT_PATH}: {len(df_final)}")

if __name__ == "__main__":
    main()
