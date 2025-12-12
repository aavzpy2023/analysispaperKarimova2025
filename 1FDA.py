import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, IsolationForest
from sklearn.linear_model import RidgeCV
import warnings
import time

# --- PROTOCOLO DE SILENCIO ---
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
TRAIN_FILE = "./V2-df_ic50_chmbl_CID_myFill.csv"
FDA_FILE = "PubChem_FDA-approved_NoInorganics.csv"
ALLOWED_ATOMS = set([1, 6, 7, 8, 9, 15, 16, 17, 35, 53]) 
MAX_MW = 1000.0
ISO_CONTAMINATION = 0.05

def get_fp(mol):
    if mol is None: return np.zeros((2048,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def check_atom_consistency(mol):
    if mol is None: return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ATOMS:
            return False
    return True

def run_comparison():
    print("="*70)
    print("COMPARATIVA FINAL: SINGLE RF vs. STACKING (RF+ET)")
    print("="*70)

    # 1. CARGA DE ENTRENAMIENTO
    print("\n[1] Entrenando Modelos Maestros...")
    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    train_mols = [Chem.MolFromSmiles(s) for s in df_train['Smiles']]
    X_train = np.array([get_fp(m) for m in train_mols])
    y_train = df_train['pIC50 Value'].values

    # --- MODELO A: SINGLE RF ---
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    
    # --- MODELO B: STACKING (RF + ET) ---
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ]
    model_stack = StackingRegressor(
        estimators=estimators, 
        final_estimator=RidgeCV(), 
        cv=5, n_jobs=-1, passthrough=False
    )
    model_stack.fit(X_train, y_train)
    
    # --- ISOLATION FOREST ---
    iso_forest = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    
    print("    -> Modelos entrenados correctamente.")

    # 2. PROCESAMIENTO FDA
    print("\n[2] Procesando FDA (Filtros Químicos)...")
    df_fda = pd.read_csv(FDA_FILE)
    df_fda.columns = df_fda.columns.str.strip() # Limpiar nombres columnas
    
    candidates = []
    
    for _, row in df_fda.iterrows():
        smi = row.get('isosmiles')
        if pd.isna(smi): smi = row.get('canonicalsmiles')
        if pd.isna(smi): continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        # Filtros Rápidos
        mw = row.get('mw')
        if pd.isna(mw): mw = Descriptors.MolWt(mol)
        else: mw = float(mw)
        
        if mw > MAX_MW: continue
        if not check_atom_consistency(mol): continue

        candidates.append({
            'CID': row.get('cid'),
            'Name': row.get('cmpdname'),
            'FP': get_fp(mol)
        })

    # 3. FILTRO OUTLIERS
    print(f"    -> Pre-candidatos: {len(candidates)}")
    X_fda = np.array([c['FP'] for c in candidates])
    iso_preds = iso_forest.predict(X_fda)
    
    final_list = [candidates[i] for i in range(len(candidates)) if iso_preds[i] == 1]
    print(f"    -> Finales (Inliers): {len(final_list)}")

    # 4. PREDICCIONES COMPARATIVAS
    print("\n[3] Generando Predicciones Cruzadas...")
    X_final = np.array([c['FP'] for c in final_list])
    
    preds_rf = model_rf.predict(X_final)
    preds_stack = model_stack.predict(X_final)
    
    results = []
    for i, item in enumerate(final_list):
        results.append({
            'CID': item['CID'],
            'Name': item['Name'],
            'RF_Pred': preds_rf[i],
            'Stack_Pred': preds_stack[i],
            'Diff': abs(preds_rf[i] - preds_stack[i])
        })
        
    df_res = pd.DataFrame(results).sort_values(by='Stack_Pred', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 CANDIDATOS - SEGÚN STACKING (RF+ET)")
    print("="*80)
    print(df_res[['Name', 'RF_Pred', 'Stack_Pred']].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("COMPARATIVA DE FÁRMACOS CLAVE")
    print("="*80)
    targets = ['Pyrimethamine', 'Trimethoprim', 'Bisacodyl', 'Etodolac', 'Triamterene', 'Methotrexate']
    
    for t in targets:
        match = df_res[df_res['Name'].str.contains(t, case=False, na=False)]
        if not match.empty:
            row = match.iloc[0]
            print(f"{t:<15} | RF: {row['RF_Pred']:.4f} | Stack: {row['Stack_Pred']:.4f} | Diff: {row['Diff']:.4f}")
        else:
            print(f"{t:<15} | No encontrado (Filtrado)")

if __name__ == "__main__":
    run_comparison()