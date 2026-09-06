import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc.single_pred import ADME, Tox

# 1. Definición de rutas basadas en el árbol del proyecto
INPUT_PATH = 'results/FDA_Candidates_For_Docking.csv'
OUTPUT_PATH = 'results/ADMET_CANDIDATES_For_Docking.csv'

def lipinski_pass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    return (Descriptors.NumHDonors(mol) <= 5 and
            Descriptors.NumHAcceptors(mol) <= 10 and
            Descriptors.MolLogP(mol) <= 5)

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: No se encontró el archivo {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Total de candidatos iniciales: {len(df)}")

    # 2. Filtro de Lipinski (RDKit)
    df['Lipinski_Pass'] = df['SMILES'].apply(lipinski_pass)
    df_filtered = df[df['Lipinski_Pass']].copy()
    print(f"Candidatos post-Lipinski: {len(df_filtered)}")

    if df_filtered.empty:
        print("Ningún candidato superó el filtro de Lipinski.")
        return

    # 3. Predicciones ADMET (TDC)
    print("Ejecutando oráculos de TDC (puede tomar unos minutos)...")
    caco2_model = ADME(name='Caco2_Wang')
    herg_model = Tox(name='hERG')

    smiles_list = df_filtered['SMILES'].tolist()
    df_filtered['Caco2_Permeability'] = caco2_model.predict(smiles_list)
    df_filtered['hERG_Blocker_Prob'] = herg_model.predict(smiles_list)

    # 4. Filtro final de toxicidad y absorción
    df_final = df_filtered[(df_filtered['Caco2_Permeability'] > -5.15) &
                           (df_filtered['hERG_Blocker_Prob'] < 0.5)]

    # 5. Exportar para 3DOCKING.py
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Candidatos finales guardados en {OUTPUT_PATH}: {len(df_final)}")

if __name__ == "__main__":
    main()
