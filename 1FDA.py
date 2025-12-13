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

# --- VALORES REALES (LITERATURA) PARA CÁLCULO DE ERROR ---
KNOWN_VALUES = {
    'Pyrimethamine': 6.56,
    'Trimethoprim': 5.57
}

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

def run_master_audit():
    print("="*100)
    print("AUDITORÍA MAESTRA: EFICIENCIA + EXACTITUD + PREDICCIÓN")
    print("="*100)

    # 1. ENTRENAMIENTO (CON METRICA DE TIEMPO)
    print("\n[1] Entrenando Tribunal de Modelos...")
    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=['Smiles', 'pIC50 Value'])
    train_mols = [Chem.MolFromSmiles(s) for s in df_train['Smiles']]
    X_train = np.array([get_fp(m) for m in train_mols])
    y_train = df_train['pIC50 Value'].values

    trained_models = []
    efficiency_stats = [] 
    model_definitions = get_top_10_models()
    model_col_names = [m[0] for m in model_definitions]
    
    for name, model in model_definitions:
        start_t = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_t
        print(f"    -> {name:<20} entrenado ({train_time:.2f}s)")
        
        trained_models.append((name, model))
        efficiency_stats.append({'Model': name, 'Train Time (s)': train_time})
        
    iso_forest = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)

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
    
    print(f"    -> Candidatos Finales: {len(final_candidates)}")

    # 3. PREDICCIÓN (CON METRICA DE TIEMPO)
    print("\n[3] Generando Votos e Inferencia...")
    model_predictions = {}
    
    for i, (name, model) in enumerate(trained_models):
        start_t = time.time()
        preds = model.predict(X_final)
        pred_time = time.time() - start_t
        
        model_predictions[name] = preds
        efficiency_stats[i]['Pred Time (s)'] = pred_time
        efficiency_stats[i]['Total Time (s)'] = efficiency_stats[i]['Train Time (s)'] + pred_time

    # 4. REPORTE 1: EFICIENCIA
    df_eff = pd.DataFrame(efficiency_stats).sort_values(by='Total Time (s)')
    print("\n" + "="*80)
    print("REPORTE 1: AUDITORÍA DE EFICIENCIA COMPUTACIONAL")
    print("="*80)
    print(df_eff.to_string(index=False))
    
    detailed_results = []
    for i in range(len(final_candidates)):
        item = final_candidates[i]
        row = {'CID': item['CID'], 'Name': item['Name'], 'SMILES': item['SMILES']}
        votes = []
        for name, _ in trained_models:
            votes.append(model_predictions[name][i])
        
        row['CONSENSUS_MEAN'] = np.mean(votes)
        row['UNCERTAINTY_STD'] = np.std(votes)
        
        # Guardar predicciones individuales para la tabla ancha
        for name in model_col_names:
             row[name] = model_predictions[name][i]
             
        detailed_results.append(row)
        
    df_res = pd.DataFrame(detailed_results).sort_values(by='CONSENSUS_MEAN', ascending=False)

    # 5. REPORTE 2: EXACTITUD (ERROR)
    print("\n" + "="*80)
    print("REPORTE 2: VALIDACIÓN DE EXACTITUD (ERROR EN CONTROLES)")
    print("="*80)
    print(f"{'Fármaco':<15} | {'Real':<6} | {'Predicho':<8} | {'Error Abs':<10} | {'Estado'}")
    print("-" * 80)
    
    for drug_name, real_val in KNOWN_VALUES.items():
        match = df_res[df_res['Name'].str.contains(drug_name, case=False, na=False)]
        if not match.empty:
            pred_val = match.iloc[0]['CONSENSUS_MEAN']
            error = abs(real_val - pred_val)
            status = "EXCELENTE" if error < 0.5 else "ACEPTABLE" if error < 1.0 else "ALERTA"
            print(f"{drug_name:<15} | {real_val:<6.2f} | {pred_val:<8.4f} | {error:<10.4f} | {status}")
        else:
            print(f"{drug_name:<15} | No encontrado (Filtrado)")

    # 6. REPORTE 3: RANKING UNIFICADO + EXPORTACIÓN
    print("\n" + "="*100)
    print("REPORTE 3: RANKING ESTRATÉGICO UNIFICADO")
    print("="*100)
    
    top_10 = df_res.head(10).copy()
    top_10['Type'] = '[TOP]'
    
    targets = ['Trimetrexate', 'Methotrexate', 'Pyrimethamine', 'Trimethoprim', 'Bisacodyl', 'Etodolac', 'Triamterene']
    pattern = '|'.join(targets)
    refs = df_res[df_res['Name'].str.contains(pattern, case=False, na=False)].copy()
    refs['Type'] = '[REF]'
    
    # Combinar Top 10 con Referencias
    combined = pd.concat([top_10, refs]).drop_duplicates(subset=['CID']).sort_values(by='CONSENSUS_MEAN', ascending=False)
    
    # --- EXPORTACIÓN PARA DOCKING ---
    docking_filename = "FDA_Candidates_For_Docking.csv"
    combined.to_csv(docking_filename, index=False)
    print(f"[EXPORTACIÓN] Archivo listo para 2DOCKING.py: {docking_filename}")
    
    # --- IMPRESIÓN DE TABLA ---
    header = f"{'Type':<6} | {'Drug Name':<25} | {'Mean':<6} | {'Std':<5}"
    for i in range(1, 11): header += f" | {f'M{i:02d}':<5}"
    print(header)
    print("-" * 130)
    
    for _, r in combined.iterrows():
        name_short = (r['Name'][:23] + '..') if len(str(r['Name'])) > 23 else r['Name']
        line = f"{r['Type']:<6} | {name_short:<25} | {r['CONSENSUS_MEAN']:.4f} | {r['UNCERTAINTY_STD']:.3f}"
        
        for m_name in model_col_names:
            val = r[m_name] # Ahora están en el DF
            line += f" | {val:.3f}"
        print(line)

if __name__ == "__main__":
    run_master_audit()
