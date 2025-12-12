import pandas as pd
import numpy as np
import itertools
import time
import warnings
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# --- MODELOS ---
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --- SILENCIO ---
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

def get_fp(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None: return np.zeros((2048,))
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros((2048,))

def run_exhaustive_search_metrics(filepath):
    print(f"[INFO] Cargando datos: {filepath}")
    df = pd.read_csv(filepath)
    df_clean = df.dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    
    print("[INFO] Generando Fingerprints...")
    X = np.array([get_fp(s) for s in df_clean['Smiles']])
    y = df_clean['pIC50 Value'].values
    
    # SPLIT 15%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # --- LOS 7 MAGNÍFICOS ---
    base_models = {
        'RF':   RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'ET':   ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'LGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1),
        'SVM':  SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
        'MLP':  MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
        'PLS':  PLSRegression(n_components=10),
        'kNN':  KNeighborsRegressor(n_neighbors=5, metric='cosine', n_jobs=-1)
    }
    
    results = []
    
    print("="*80)
    print(f"INICIANDO BARRIDO TOTAL CON MÉTRICAS (Train={len(y_train)}, Test={len(y_test)})")
    print("="*80)

    # 1. INDIVIDUALES
    print(">>> FASE 1: Individuales (7 Modelos)")
    for name, model in base_models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if y_pred.ndim > 1: y_pred = y_pred.flatten()
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - start
        
        results.append({
            'Tipo': '1-Single', 
            'Modelo': name, 
            'R2': r2, 
            'MAE': mae, 
            'Time': elapsed
        })
        print(f"   {name:<5} : R2={r2:.4f} | MAE={mae:.4f} | {elapsed:.2f}s")

    # 2. DUOS
    print("\n>>> FASE 2: Duos (21 Combinaciones)")
    model_items = list(base_models.items())
    
    for combo in itertools.combinations(model_items, 2):
        names = [c[0] for c in combo]
        combo_id = "+".join(names)
        
        start = time.time()
        reg = StackingRegressor(estimators=list(combo), final_estimator=RidgeCV(), cv=5, n_jobs=-1)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - start
        
        results.append({
            'Tipo': '2-Duo', 
            'Modelo': combo_id, 
            'R2': r2, 
            'MAE': mae, 
            'Time': elapsed
        })
        
        if r2 > 0.825: print(f"   {combo_id:<15} : R2={r2:.4f} (Top Duo)")

    # 3. TRIOS
    print("\n>>> FASE 3: Trios (35 Combinaciones)")
    for combo in itertools.combinations(model_items, 3):
        names = [c[0] for c in combo]
        combo_id = "+".join(names)
        
        start = time.time()
        reg = StackingRegressor(estimators=list(combo), final_estimator=RidgeCV(), cv=5, n_jobs=-1)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - start
        
        results.append({
            'Tipo': '3-Trio', 
            'Modelo': combo_id, 
            'R2': r2, 
            'MAE': mae, 
            'Time': elapsed
        })
        
        if r2 > 0.83: print(f"   {combo_id:<20} : R2={r2:.4f} (¡RÉCORD!)")

    # --- REPORTE FINAL ---
    df_res = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    
    print("\n" + "="*90)
    print("TOP 30 MEJORES ARQUITECTURAS (ORDENADO POR R2)")
    print("="*90)
    # Formato limpio para copiar
    print(df_res[['Tipo', 'Modelo', 'R2', 'MAE', 'Time']].head(30).to_string(index=False))
    print("-" * 90)
    
    best = df_res.iloc[0]
    print(f"\n[GANADOR] {best['Modelo']} | R2: {best['R2']:.4f} | MAE: {best['MAE']:.4f} | Tiempo: {best['Time']:.2f}s")

if __name__ == "__main__":
    run_exhaustive_search_metrics("./V2-df_ic50_chmbl_CID_myFill.csv")