import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko import PDBQTMolecule
from vina import Vina
import os
import warnings

# --- PROTOCOLO DE SILENCIO ---
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN DE FÍSICA (TUS COORDENADAS EXACTAS) ---
RECEPTOR_FILE = "receptor.pdbqt"
INPUT_CSV = "FDA_Candidates_For_Docking.csv" # El archivo generado por el modelo de consenso

# Coordenadas detectadas por prep_receptor.py
CENTER_X = 17.394
CENTER_Y = 68.757
CENTER_Z = -68.051
BOX_SIZE = 20.0    # Angstroms (Caja estándar generosa)
EXHAUSTIVENESS = 8 # Precisión de búsqueda (8 es estándar, 32 es muy lento)

def prepare_ligand_pdbqt(smiles, name):
    """Convierte SMILES -> Mol 3D -> PDBQT String"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        # 1. Añadir Hidrógenos (Crucial para física)
        mol = Chem.AddHs(mol)
        
        # 2. Generar Coordenadas 3D (Embedding)
        # Usamos randomSeed para reproducibilidad
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            # Si falla ETKDG, probar embedding aleatorio básico
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            
        # 3. Optimización de Energía (MMFF94)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass # Si falla el campo de fuerza, usamos la geometría cruda
        
        # 4. Conversión a PDBQT (Meeko)
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        return preparator.write_pdbqt_string()
    except Exception as e:
        # print(f"Error preparando {name}: {e}")
        return None

def run_docking_pipeline():
    print("="*80)
    print(f"INICIANDO VALIDACIÓN HÍBRIDA (ML + Vina)")
    print(f"Centro: [{CENTER_X}, {CENTER_Y}, {CENTER_Z}]")
    print("="*80)

    # 1. Cargar Candidatos
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] No se encuentra {INPUT_CSV}. Corre primero el script de predicción.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # ESTRATEGIA DE SELECCIÓN:
    # Tomamos el Top 15 absoluto + Los Fármacos de Referencia del Paper
    top_candidates = df.head(15).copy()
    
    targets = ['Pyrimethamine', 'Trimethoprim', 'Bisacodyl', 'Etodolac', 'Triamterene', 'Methotrexate']
    for t in targets:
        row = df[df['Name'].str.contains(t, case=False, na=False)]
        if not row.empty:
            top_candidates = pd.concat([top_candidates, row])
            
    # Eliminar duplicados si algún target ya estaba en el top
    top_candidates = top_candidates.drop_duplicates(subset=['CID'])
    
    print(f"[INFO] Se realizarán simulaciones para {len(top_candidates)} moléculas clave.")

    # 2. Configurar Motor Vina
    v = Vina(sf_name='vina')
    v.set_receptor(RECEPTOR_FILE)
    v.compute_vina_maps(center=[CENTER_X, CENTER_Y, CENTER_Z], box_size=[BOX_SIZE, BOX_SIZE, BOX_SIZE])

    results = []

    # 3. Bucle de Simulación
    print("\nPROCESANDO DOCKING...")
    print(f"{'Fármaco':<25} | {'Estado'}")
    print("-" * 40)

    for idx, row in top_candidates.iterrows():
        name = str(row['Name'])[:25]
        smiles = row['SMILES']
        ml_score = row['CONSENSUS_MEAN'] # Predicción de tu IA
        
        pdbqt_ligand = prepare_ligand_pdbqt(smiles, name)
        
        if pdbqt_ligand:
            try:
                v.set_ligand_from_string(pdbqt_ligand)
                v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=1)
                energy = v.energies(n_poses=1)[0][0] # Energía libre de unión (kcal/mol)
                
                print(f"{name:<25} | OK ({energy:.2f} kcal/mol)")
                
                results.append({
                    'Name': row['Name'],
                    'ML_pIC50': ml_score,
                    'Docking_Score': energy
                })
            except Exception as e:
                print(f"{name:<25} | Fallo Vina")
        else:
            print(f"{name:<25} | Fallo 3D")

    # 4. Resultados Finales
    final_df = pd.DataFrame(results).sort_values(by='Docking_Score') # Más negativo es mejor (más fuerte)
    
    print("\n" + "="*90)
    print("RESULTADOS FINALES DE VALIDACIÓN CRUZADA (IA vs FÍSICA)")
    print("="*90)
    print(f"{'Fármaco':<30} | {'IA (pIC50)':<10} | {'Física (kcal/mol)':<18} | {'Correlación'}")
    print("-" * 90)
    
    for _, r in final_df.iterrows():
        # Interpretación rápida
        dock = r['Docking_Score']
        ml = r['ML_pIC50']
        
        # Lógica de validación
        # Si ML es alto (>6.5) y Docking es fuerte (<-8.0) -> Validado
        # Si ML es bajo (<5.5) y Docking es débil (>-7.0) -> Validado (Correctamente inactivo)
        # Si discrepan -> Alerta
        
        if ml > 6.5 and dock < -8.0: status = "VALIDADO (+)"
        elif ml < 6.0 and dock > -7.5: status = "VALIDADO (-)"
        elif ml > 6.5 and dock > -7.0: status = "FALSO POSITIVO IA"
        elif ml < 6.0 and dock < -8.5: status = "FALSO NEGATIVO IA"
        else: status = "Neutro"
        
        name_clean = str(r['Name'])[:30]
        print(f"{name_clean:<30} | {ml:<10.4f} | {dock:<18.2f} | {status}")
        
    final_df.to_csv("Final_Validation_Hybrid.csv", index=False)
    print(f"\n[INFO] Resultados guardados en Final_Validation_Hybrid.csv")

if __name__ == "__main__":
    if not os.path.exists(RECEPTOR_FILE):
        print(f"[ERROR] No se encuentra {RECEPTOR_FILE}. Ejecuta primero prep_receptor.py")
    else:
        run_docking_pipeline()