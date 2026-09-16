import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Suppress scikit-learn feature name mismatch UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import centralized configurations
try:
    from paths_config import *
except ImportError:
    print("[FATAL] Could not import paths_config_6.py. Check your PYTHONPATH.")
    sys.exit(1)

# Import the exact same models used in 0STACK_6.py
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb
from xgboost import XGBRegressor

from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

# =========================================================
# EXPERIMENT CONFIGURATION
# =========================================================
N_PERMUTATIONS = 100
CV_FOLDS = 5
N_JOBS = 46 # Matches the workstation profile from 0STACK
RANDOM_STATE = 42

# =========================================================
# 1. DYNAMIC DISCOVERY OF THE BEST MODEL
# =========================================================
def get_winning_architecture():
    """Analyzes the Nested CV results to find the winning representation and architecture."""
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"[FATAL] {CHECKPOINT_FILE} does not exist. You must run 0STACK_6.py first.")
        sys.exit(1)

    df = pd.read_csv(CHECKPOINT_FILE)

    # Find the representation (Mode) with the highest average outer R2
    mode_perf = df.groupby('Mode')['R2_outer'].mean().sort_values(ascending=False)
    best_mode = mode_perf.index[0]

    # Find the most stable architecture (most frequently selected) for that Mode
    df_best_mode = df[df['Mode'] == best_mode]
    best_architecture = df_best_mode['Selected_Model'].value_counts().index[0]

    return best_mode, best_architecture

# =========================================================
# 2. ARCHITECTURE RECONSTRUCTION (WITHOUT .JOBLIB)
# =========================================================
def build_base_models(n_estimators=200):
    """Must be identical to the one in 0STACK_6.py"""
    return {
        'RF':   RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1),
        'ET':   ExtraTreesRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1),
        'LGBM': lgb.LGBMRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, verbosity=-1, n_jobs=1),
        'XGB':  XGBRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
        'SVM':  SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
        'kNN':  KNeighborsRegressor(n_neighbors=5, metric='cosine', n_jobs=1),
    }

def build_dynamic_model(architecture_name):
    """Rebuilds the stacking or single model dynamically based on its string name."""
    base_models = build_base_models()
    keys = architecture_name.split('+')

    if len(keys) == 1:
        return clone(base_models[keys[0]])
    else:
        return StackingRegressor(
            estimators=[(k, clone(base_models[k])) for k in keys],
            final_estimator=RidgeCV(),
            cv=5, n_jobs=1
        )

# Morgan feature extraction (Required if the winner uses Morgan fingerprints)
def get_morgan_fp(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None: return np.zeros((2048,), dtype=np.int8)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprintAsNumPy(m).astype(np.int8)
    except Exception:
        return np.zeros((2048,), dtype=np.int8)

# =========================================================
# 3. Y-RANDOMIZATION CORE
# =========================================================
def permutation_task(model, X, y_true, cv, seed):
    """Permutes the target variable and evaluates the model."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_true)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y_shuffled, cv=kf, scoring='r2', n_jobs=1)
    return np.mean(scores)

def main():
    print("\n" + "=" * 80)
    print("STARTING ROBUSTNESS TEST: Y-RANDOMIZATION")
    print("=" * 80)

    best_mode, best_architecture = get_winning_architecture()
    print(f"[AUDIT] Best representation detected: {best_mode}")
    print(f"[AUDIT] Most stable architecture detected: {best_architecture}")

    # 1. Load Data
    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    df = df.dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    y_true = df['pIC50 Value'].values

    # 2. Generate Feature Matrix X based on the winning mode
    print(f"[INFO] Calculating feature matrix for mode '{best_mode}'...")
    if 'morgan' in best_mode.lower():
        X_raw = np.array([get_morgan_fp(s) for s in df['Smiles']])
    else:
        print("[WARNING] Current Y-Randomization is optimized for Morgan-like representations. Adapt X calculation if necessary.")
        X_raw = np.array([get_morgan_fp(s) for s in df['Smiles']]) # Fallback

    # Convert X_raw array into DataFrame with column names to avoid LGBM feature name warning
    X = pd.DataFrame(X_raw, columns=[f"fp_{i}" for i in range(X_raw.shape[1])])

    model = build_dynamic_model(best_architecture)

    # 3. Compute True R2
    print("[INFO] Evaluating unpermuted R2 (True Score)...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    true_r2 = np.mean(cross_val_score(model, X, y_true, cv=kf, scoring='r2', n_jobs=N_JOBS))
    print(f"[RESULT] True R²: {true_r2:.4f}")

    # 4. Y-Randomization Loop
    print(f"[INFO] Executing {N_PERMUTATIONS} permutations distributed across {N_JOBS} threads...")
    start_time = time.time()
    random_scores = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(permutation_task)(clone(model), X, y_true, CV_FOLDS, seed=(RANDOM_STATE + i))
        for i in range(N_PERMUTATIONS)
    )

    # 5. Overfitting Diagnostics
    mean_random = np.mean(random_scores)
    std_random = np.std(random_scores)

    print("\n" + "-" * 40)
    print("Y-RANDOMIZATION INTEGRITY REPORT")
    print("-" * 40)
    print(f"Original R²:          {true_r2:.4f}")
    print(f"Y-Random R² (Mean):   {mean_random:.4f}")
    print(f"Y-Random R² (Std):    {std_random:.4f}")

    if true_r2 > (mean_random + 2.33 * std_random): # ~99% confidence interval
        print("\n[ACADEMIC VERDICT] ✅ SUCCESSFUL. The model is not memorizing noise.")
    else:
        print("\n[ACADEMIC VERDICT] ❌ ROBUSTNESS FAILURE. High probability of chance correlation.")

    print(f"Total computation time: {(time.time() - start_time):.1f} seconds")

if __name__ == "__main__":
    main()
