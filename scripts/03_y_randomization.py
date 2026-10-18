import os
import sys
import time
import warnings
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Suppress scikit-learn feature name mismatch UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Link root directory to import paths_config and logger_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import *
from logger_utils import setup_logger, explain

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

logger = logging.getLogger(__name__)


# =========================================================
# 1. DYNAMIC DISCOVERY OF THE BEST MODEL
# =========================================================
# Architecture of the deployed screening model (STACK_MEMBERS in step 04), evaluated on the Morgan fingerprints it uses.
DEPLOYED_MODE = 'morgan'
DEPLOYED_ARCHITECTURE = 'RF+XGB+SVM'


# =========================================================
# 2. ARCHITECTURE RECONSTRUCTION
# =========================================================
def build_base_models(n_estimators=200):
    """Rebuilds the dictionary of base regressors matching the primary pipeline."""
    return {
        'RF':   RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1),
        'ET':   ExtraTreesRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1),
        'LGBM': lgb.LGBMRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, verbosity=-1, n_jobs=1),
        'XGB':  XGBRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
        'SVM':  SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
        'kNN':  KNeighborsRegressor(n_neighbors=5, metric='cosine', n_jobs=1),
    }


def build_dynamic_model(architecture_name):
    """Rebuilds the stacking ensemble or single model dynamically based on its string identifier."""
    base_models = build_base_models()
    keys = architecture_name.split('+')

    if len(keys) == 1:
        return clone(base_models[keys[0]])
    else:
        return StackingRegressor(
            estimators=[(k, clone(base_models[k])) for k in keys],
            final_estimator=RidgeCV(),
            cv=CV_FOLDS,
            n_jobs=1
        )


def get_morgan_fp(smiles):
    """Calculates Morgan fingerprints (ECFP4, 2048 bits) for SMILES strings."""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return np.zeros((2048,), dtype=np.int8)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprintAsNumPy(m).astype(np.int8)
    except Exception:
        return np.zeros((2048,), dtype=np.int8)


# =========================================================
# 3. Y-RANDOMIZATION CORE
# =========================================================
def permutation_task(model, X, y_true, cv, seed):
    """Permutes the target variable and evaluates cross-validated R2 score."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_true)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y_shuffled, cv=kf, scoring='r2', n_jobs=1)
    return float(np.mean(scores))


def main():
    # Initialize unified logger
    log_file_path = os.path.join(LOGS_DIR, "03_y_randomization.log")
    setup_logger(log_file_path)

    logger.info("=" * 80)
    logger.info("03_y_randomization.py — Model Robustness Test (Y-Randomization)")
    logger.info("=" * 80)
    explain("What this step does", """
        Sanity check that the model really learned chemistry and did not just memorise noise or benefit from a data leak.

        Procedure (Y-randomization): the measured potencies (pIC50) are randomly shuffled between molecules, which destroys any
        true structure-activity relationship, and the same model is trained and cross-validated again. This is repeated many
        times. A healthy model scores well on the real data and about R2 = 0 (or negative) on the shuffled data.

        R2 is the fraction of the variation in potency that the model explains: 1.0 is perfect, 0 is no better than always
        predicting the average potency, negative is worse than that.
    """)

    best_mode, best_architecture = DEPLOYED_MODE, DEPLOYED_ARCHITECTURE
    logger.info(f"Deployed model -> representation: '{best_mode}', architecture: '{best_architecture}'")

    # 1. Load Data
    logger.info(f"Loading training data from: {TRAIN_FILE}")
    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    df = df.dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    y_true = df['pIC50 Value'].values
    logger.info(f"Successfully loaded {len(df)} valid SMILES and target values.")

    # 2. Feature Matrix Generation
    logger.info(f"Generating feature matrix X for representation mode '{best_mode}'...")
    X_raw = np.array([get_morgan_fp(s) for s in df['Smiles']])

    X = pd.DataFrame(X_raw, columns=[f"fp_{i}" for i in range(X_raw.shape[1])])
    logger.info(f"Feature matrix dimensions: {X.shape}")

    model = build_dynamic_model(best_architecture)

    # 3. Compute Baseline (True) R2 Score
    logger.info(f"Evaluating baseline unpermuted R2 score using {CV_FOLDS}-fold CV...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    true_r2 = float(np.mean(cross_val_score(model, X, y_true, cv=kf, scoring='r2', n_jobs=N_JOBS)))
    logger.info(f"Baseline Unpermuted R2: {true_r2:.4f}")

    # 4. Y-Randomization Permutation Loop
    logger.info(f"Starting {N_PERMUTATIONS} target permutations using {N_JOBS} worker threads...")
    start_time = time.time()
    random_scores = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(permutation_task)(clone(model), X, y_true, CV_FOLDS, seed=(RANDOM_STATE + i))
        for i in range(N_PERMUTATIONS)
    )

    elapsed_time = time.time() - start_time
    df_random = pd.DataFrame({'Permuted_R2': random_scores})
    csv_path = os.path.join(RESULTS_DIR, "03_y_randomization_scores.csv")
    df_random.to_csv(csv_path, index=False)
    logger.info(f"Y-randomization distribution saved to {csv_path}")

    mean_random = float(np.mean(random_scores))
    std_random = float(np.std(random_scores))
    z_score = (true_r2 - mean_random) / std_random if std_random > 0 else 0.0

    explain("How to read the report below", f"""
        'Baseline R2' is the real model ({true_r2:.2f}). 'Random R2' is what the same model achieves when potencies are shuffled
        ({mean_random:.2f} on average over {N_PERMUTATIONS} shuffles). The Z-score counts how many standard deviations the real
        result sits above the shuffled results; the pass criterion used here is real R2 > mean + 2.33 SD (~99% confidence).

        If the verdict is PASSED, the predictive signal is not a chance correlation. Note that this test uses Morgan
        fingerprints and the stacking architecture of the deployed screening model (RF+XGB+SVM, as in step 04).
    """)
    logger.info("-" * 80)
    logger.info("Y-RANDOMIZATION INTEGRITY REPORT")
    logger.info("-" * 80)
    logger.info(f"  Baseline R2 (True Target) : {true_r2:.4f}")
    logger.info(f"  Random R2 (Mean)          : {mean_random:.4f}")
    logger.info(f"  Random R2 (Std)           : {std_random:.4f}")
    logger.info(f"  Z-Score                   : {z_score:.2f}")

    # 99% confidence interval threshold (~2.33 std deviations)
    if true_r2 > (mean_random + 2.33 * std_random):
        logger.info("ACADEMIC VERDICT: PASSED. Model performance on true target is significantly superior to random chance (no memory leakage/overfitting).")
    else:
        logger.warning("ACADEMIC VERDICT: FAILED. High risk of chance correlation or data leakage.")

    # 5. Export Variables to LaTeX
    latex_file = os.path.join(LATEX_DIR, "yrandom_variables.tex")
    logger.info(f"Exporting Y-Randomization LaTeX variables to: {latex_file}")
    try:
        os.makedirs(os.path.dirname(latex_file), exist_ok=True)
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("% =====================================================\n")
            f.write("% Auto-generated by 03_y_randomization.py\n")
            f.write("% =====================================================\n\n")
            f.write(f"\\newcommand{{\\YRandomTrueRTwo}}{{{true_r2:.4f}}}\n")
            f.write(f"\\newcommand{{\\YRandomMeanRTwo}}{{{mean_random:.4f}}}\n")
            f.write(f"\\newcommand{{\\YRandomStdRTwo}}{{{std_random:.4f}}}\n")
            f.write(f"\\newcommand{{\\YRandomZScore}}{{{z_score:.1f}}}\n")
            f.write(f"\\newcommand{{\\YRandomArchitecture}}{{{best_architecture}}}\n")
            f.write(f"\\newcommand{{\\YRandomNPermutations}}{{{N_PERMUTATIONS}}}\n")
            f.write(f"\\newcommand{{\\YRandomCVFolds}}{{{CV_FOLDS}}}\n")
        logger.info("LaTeX variable export completed successfully.")
    except Exception as e:
        logger.error(f"Failed to export LaTeX variables: {e}")

    logger.info(f"Total execution time: {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()
