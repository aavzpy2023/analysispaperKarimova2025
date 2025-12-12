import pandas as pd
import numpy as np
import warnings
import time
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors

# --- MODELOS ---
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
import lightgbm as lgb

# =========================================================
# PROTOCOLO DE SILENCIO
# =========================================================
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
        if atom.GetAtomicNum() not in ALLOWED_ATOMS: return False
    return True

# --- DEFINICIÓN DE LOS 10 MEJORES (TOP TIER) ---
def get_top_10_models():
    # Instancias base
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1)
    svm = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    pls = PLSRegression(n_components=10)
    
    def make_stack(est):
        return StackingRegressor(estimators=est, final_estimator=RidgeCV(), cv=5, n_jobs=-1, passthrough=False)

    models = [
        ("M01_ET+LGBM",       make_stack([('et', et), ('lgbm', lgbm)])),
        ("M02_ET+LGBM+PLS",   make_stack([('et', et), ('lgbm', lgbm), ('pls', pls)])),
        ("M03_RF+LGBM",       make_stack([('rf', rf), ('lgbm', lgbm)])),
        ("M04_RF+ET+LGBM",    make_stack([('rf', rf), ('et', et), ('lgbm', lgbm)])),
        ("M05_LGBM_Solo",     lgbm),
        ("M06_RF+LGBM+PLS",   make_stack([('rf', rf), ('lgbm', lgbm), ('pls', pls)])),
        ("M07_LGBM+PLS",      make_stack([('lgbm', lgbm), ('pls', pls)])),
        ("M08_RF+LGBM+SVM",   make_stack([('rf', rf), ('lgbm', lgbm), ('svm', svm)])),
        ("M09_ET+LGBM+SVM",   make_stack([('et', et), ('lgbm', lgbm), ('svm', svm)])),
        ("M10_RF+ET",         make_stack([('rf', rf), ('et', et)])) 
    ]
    return models

def run_unified_ranking():
    print("="*100)
    print("PROTOCOLO FINAL: RANKING UNIFICADO (Top Hits + Referencias)")
    print("="*100)

    # 1. ENTRENAMIENTO
    print("\n[1] Entrenando Tribunal de Modelos...")
    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    train_mols = [Chem.MolFromSmiles(s) for s in df_train['Smiles']]
    X_train = np.array([get_fp(m) for m in train_mols])
    y_train = df_train['pIC50 Value'].values

    trained_models = []
    model_definitions = get_top_10_models()
    model_col_names = [m[0] for m in model_definitions]
    
    for name, model in model_definitions:
        model.fit(X_train, y_train)
        trained_models.append((name, model))
        
    iso_forest = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    print("    -> Modelos listos.")

    # 2. PROCESAMIENTO FDA
    print("\n[2] Procesando FDA...")
    try:
        df_fda = pd.read_csv(FDA_FILE)
        df_fda.columns = df_fda.columns.str.strip()
    except:
        return
    
    candidates = []
    for _, row in df_fda.iterrows():
        smi = row.get('isosmiles')
        if pd.isna(smi): smi = row.get('canonicalsmiles')
        if pd.isna(smi): continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        mw = row.get('mw')
        if pd.isna(mw) or mw == '': mw = Descriptors.MolWt(mol)
        else: mw = float(mw)
        
        if mw > MAX_MW: continue
        if not check_atom_consistency(mol): continue

        candidates.append({'CID': row.get('cid'), 'Name': row.get('cmpdname'), 'FP': get_fp(mol), 'SMILES': smi})

    X_fda = np.array([c['FP'] for c in candidates])
    iso_preds = iso_forest.predict(X_fda)
    final_candidates = [candidates[i] for i in range(len(candidates)) if iso_preds[i] == 1]
    X_final = np.array([c['FP'] for c in final_candidates])
    
    print(f"    -> Candidatos Finales Válidos: {len(final_candidates)}")

    # 3. PREDICCIONES
    print("\n[3] Generando Votos...")
    model_predictions = {}
    for name, model in trained_models:
        model_predictions[name] = model.predict(X_final)
    
    detailed_results = []
    for i in range(len(final_candidates)):
        item = final_candidates[i]
        row = {'CID': item['CID'], 'Name': item['Name']}
        votes = []
        for name in model_col_names:
            pred = model_predictions[name][i]
            row[name] = pred
            votes.append(pred)
        
        row['CONSENSUS_MEAN'] = np.mean(votes)
        row['UNCERTAINTY_STD'] = np.std(votes)
        detailed_results.append(row)
        
    df_res = pd.DataFrame(detailed_results).sort_values(by='CONSENSUS_MEAN', ascending=False)
    
    # 4. CONSTRUCCIÓN DEL TABLERO UNIFICADO
    print("\n" + "="*160)
    print("RANKING ESTRATÉGICO UNIFICADO (Top 10 Nuevos + Referencias Paper)")
    print("="*160)
    
    # A. Obtener el Top 10 Absoluto
    top_10 = df_res.head(10).copy()
    top_10['Type'] = '[TOP]'
    
    # B. Obtener las Referencias del Paper
    targets = ['Trimetrexate', 'Methotrexate', 'Pyrimethamine', 'Trimethoprim', 'Bisacodyl', 'Etodolac', 'Triamterene']
    # Crear regex para buscar cualquiera de los targets
    pattern = '|'.join(targets)
    refs = df_res[df_res['Name'].str.contains(pattern, case=False, na=False)].copy()
    refs['Type'] = '[REF]'
    
    # C. Unir y Ordenar
    combined = pd.concat([top_10, refs]).drop_duplicates(subset=['CID'])
    combined = combined.sort_values(by='CONSENSUS_MEAN', ascending=False)
    
    # D. Imprimir
    header = f"{'Type':<6} | {'Drug Name':<25} | {'Mean':<6} | {'Std':<5}"
    for i in range(1, 11):
        header += f" | {f'M{i:02d}':<5}"
    print(header)
    print("-" * 160)
    
    for _, r in combined.iterrows():
        name = (r['Name'][:23] + '..') if len(str(r['Name'])) > 23 else r['Name']
        line = f"{r['Type']:<6} | {name:<25} | {r['CONSENSUS_MEAN']:.4f} | {r['UNCERTAINTY_STD']:.3f}"
        for m in model_col_names:
            line += f" | {r[m]:.3f}"
        print(line)

    print("\n[LEYENDA]")
    print("M01: ET+LGBM (El Ganador) | M05: LGBM Solo | M10: RF+ET (Clásico)")
    print("[TOP]: Nuevo descubrimiento | [REF]: Fármaco mencionado en el paper")

if __name__ == "__main__":
    run_unified_ranking()