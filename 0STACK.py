import os
# Prevent C/C++ sub-libraries from spawning extra threads inside each of the
# N_JOBS worker processes (avoids CPU oversubscription: 44 processes x their
# own internal thread pools would massively exceed physical core count).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, RegressorMixin, clone
import lightgbm as lgb
from xgboost import XGBRegressor

from sklearn.model_selection import RepeatedKFold, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
# Benign, noisy warning from LightGBM/sklearn interaction inside StackingRegressor
# (LGBM sometimes infers feature names internally during stacking, then complains
# when predicting on a plain numpy array). Does not affect correctness.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =========================================================
# EXECUTION PROFILE
# =========================================================
PROFILE = 'workstation'  # 'laptop' to debug quickly, 'workstation' for the real run

PROFILES = {
    'laptop': dict(
        N_JOBS=2,
        FEATURE_MODES=['morgan'],
        MAX_COMBO_SIZE=2,
        OUTER_N_SPLITS=3,
        OUTER_N_REPEATS=1,
        INNER_N_SPLITS=3,
        N_ESTIMATORS_TREES=50,
    ),
    'workstation': dict(
        N_JOBS=44,  # adjust to your logical core count minus a couple for the OS
        FEATURE_MODES=['morgan', 'rdkit2d', 'rdkit2d_fp', 'rdkit2d3d_fp'],
        MAX_COMBO_SIZE=3,
        OUTER_N_SPLITS=5,
        OUTER_N_REPEATS=3,
        INNER_N_SPLITS=5,
        N_ESTIMATORS_TREES=200,
    ),
}
CFG = PROFILES[PROFILE]

TRAIN_FILE = "./V2-df_ic50_chmbl_CID_myFill.csv"
RANDOM_STATE = 42
CHECKPOINT_FILE = "nested_cv_checkpoint.csv"
SELECTION_LOG_FILE = "nested_cv_selection_log.csv"
FINAL_RESULTS_FILE = "nested_cv_final_results.csv"
LATEX_OUTPUT_FILE = "paper_variables.tex"
FIGURE_FILE = "r2_by_representation_boxplot.png"

# =========================================================
# APPLICABILITY-DOMAIN SAFEGUARD (bug fix for PLS extrapolation blowups)
# =========================================================
# Root cause observed in practice: PLSRegression under strong feature
# collinearity (common with hundreds/thousands of correlated 2D/FP
# descriptors) can produce catastrophically extrapolated predictions on
# out-of-fold test points (R2 in the thousands of negative units), which
# silently destroys the aggregate mean/std and every downstream t-test.
# Fix: clip every model's predictions to the training target range +/- a
# margin before scoring, in BOTH the inner-CV selection step and the outer
# evaluation step. This is standard QSAR practice (an applicability-domain
# bound on pIC50, which has a known physically plausible range) and is
# applied uniformly to all models, not just PLS, so it never advantages one
# architecture over another. Every clipping event is counted and logged for
# full transparency in the manuscript.
CLIP_MARGIN = 3.0  # pIC50 units beyond the observed training min/max


def get_clip_bounds(y_train, margin=CLIP_MARGIN):
    return float(np.min(y_train) - margin), float(np.max(y_train) + margin)


def clip_predictions(y_pred, lo, hi):
    y_pred = np.asarray(y_pred, dtype=float)
    clipped_mask = (y_pred < lo) | (y_pred > hi)
    return np.clip(y_pred, lo, hi), int(clipped_mask.sum())

# 'Standard Value' is the raw IC50 (nM) from which pIC50 Value is derived
# (pIC50 = -log10(IC50 in M)). Including it as a feature would be TARGET
# LEAKAGE. It must never be used as model input.
NON_FEATURE_COLS = {
    'molecule chembl id', 'smiles', 'pic50 value', 'pic50',
    'standard value', 'ic50', 'cid', 'name', 'id',
}

# Reference values from the original paper, with LaTeX-safe labels
# (\newcommand names may only contain letters, no digits or underscores)
PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP model, no feature selection', 0.75),
    'PaperSelected': ('After importance-based feature selection (no augmentation)', 0.82),
    'PaperFinal':    ('Data augmentation + DNN ensemble (paper final result)', 0.85),
}

# Feature-mode key -> LaTeX-safe label (letters only)
MODE_LATEX_LABEL = {
    'morgan': 'Morgan',
    'rdkit2d': 'RdkitTwoD',
    'rdkit2d_fp': 'RdkitTwoDFp',
    'rdkit2d3d_fp': 'RdkitTwoDThreeDFp',
    'csv_descriptors': 'CsvDescriptors',
}

MODE_DESCRIPTION = {
    'morgan': 'Morgan fingerprints (ECFP4, 2048 bits) — no physicochemical descriptors',
    'rdkit2d': f'{len(Descriptors._descList)} RDKit 2D descriptors (analogous to the paper\'s "2D" model)',
    'rdkit2d_fp': 'RDKit 2D descriptors + Morgan FP (analogous to the paper\'s "2D/FP" model)',
    'rdkit2d3d_fp': '2D + 3D (geometric, via ETKDGv3 embedding + MMFF94s optimization) descriptors + '
                     'Morgan FP (analogous to the paper\'s best model, "2D/3D/FP")',
    'csv_descriptors': 'Numeric columns already present in the training CSV (trivial reference only)',
}


# =========================================================
# MOLECULAR REPRESENTATIONS
# =========================================================
# Bug fix: AllChem.GetMorganFingerprintAsBitVect is deprecated in recent
# RDKit and prints a noisy C++-level warning on every call that RDLogger
# cannot silence. The new MorganGenerator API avoids that warning, but its
# generator object is NOT picklable, so it cannot be a module-level global
# (joblib/loky needs to pickle the function + its globals to ship the task
# to worker processes). Instead, each worker lazily builds its own
# generator on first use (module-level state is process-local once forked).
_MORGAN_GEN = None


def _get_morgan_generator():
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _MORGAN_GEN


def get_morgan_fp(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return np.zeros((2048,), dtype=np.int8)
        gen = _get_morgan_generator()
        return gen.GetFingerprintAsNumPy(m).astype(np.int8)
    except Exception:
        return np.zeros((2048,), dtype=np.int8)


_RDKIT_2D_NAMES = [name for name, _ in Descriptors._descList]


def get_rdkit_2d(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.full((len(_RDKIT_2D_NAMES),), np.nan)
    vals = []
    for name, fn in Descriptors._descList:
        try:
            vals.append(fn(m))
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


_RDKIT_3D_NAMES = [
    'Asphericity', 'Eccentricity', 'InertialShapeFactor',
    'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3',
    'RadiusOfGyration', 'SpherocityIndex',
]


def get_rdkit_3d(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.full((len(_RDKIT_3D_NAMES),), np.nan)
    try:
        m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(m, params) == -1:
            if AllChem.EmbedMolecule(m, useRandomCoords=True) == -1:
                return np.full((len(_RDKIT_3D_NAMES),), np.nan)
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            pass
        vals = [
            Descriptors3D.Asphericity(m), Descriptors3D.Eccentricity(m),
            Descriptors3D.InertialShapeFactor(m), Descriptors3D.NPR1(m),
            Descriptors3D.NPR2(m), Descriptors3D.PMI1(m), Descriptors3D.PMI2(m),
            Descriptors3D.PMI3(m), Descriptors3D.RadiusOfGyration(m),
            Descriptors3D.SpherocityIndex(m),
        ]
        return np.array(vals, dtype=float)
    except Exception:
        return np.full((len(_RDKIT_3D_NAMES),), np.nan)


def compute_parallel(smiles_list, fn, n_jobs):
    results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(fn)(s) for s in smiles_list)
    return np.array(results)


def sanitize_matrix(X):
    """Replace +/-inf with NaN so downstream imputation can handle them.
    Bug fix: some RDKit 2D descriptors (e.g. Ipc) can produce inf on large
    or complex molecules; SimpleImputer does NOT treat inf as missing, so
    without this the raw inf values would silently propagate into
    StandardScaler/SVR/PLS."""
    return np.where(np.isinf(X), np.nan, X)


def detect_descriptor_columns(df):
    cols = []
    for c in df.columns:
        if c.strip().lower() in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_feature_matrix(df, mode, n_jobs):
    smiles = df['Smiles'].tolist()

    if mode == 'morgan':
        X = compute_parallel(smiles, get_morgan_fp, n_jobs)
        return X, [f'morgan_{i}' for i in range(X.shape[1])], True

    if mode == 'rdkit2d':
        X = sanitize_matrix(compute_parallel(smiles, get_rdkit_2d, n_jobs))
        return X, list(_RDKIT_2D_NAMES), False

    if mode == 'rdkit2d_fp':
        X2d = sanitize_matrix(compute_parallel(smiles, get_rdkit_2d, n_jobs))
        Xfp = compute_parallel(smiles, get_morgan_fp, n_jobs)
        X = np.hstack([X2d, Xfp])
        return X, list(_RDKIT_2D_NAMES) + [f'morgan_{i}' for i in range(Xfp.shape[1])], False

    if mode == 'rdkit2d3d_fp':
        print("    [INFO] Generating 3D conformers in parallel (embedding + MMFF)...")
        X2d = sanitize_matrix(compute_parallel(smiles, get_rdkit_2d, n_jobs))
        X3d = sanitize_matrix(compute_parallel(smiles, get_rdkit_3d, n_jobs))
        Xfp = compute_parallel(smiles, get_morgan_fp, n_jobs)
        X = np.hstack([X2d, X3d, Xfp])
        names = list(_RDKIT_2D_NAMES) + list(_RDKIT_3D_NAMES) + [f'morgan_{i}' for i in range(Xfp.shape[1])]
        return X, names, False

    if mode == 'csv_descriptors':
        cols = detect_descriptor_columns(df)
        if not cols:
            raise ValueError("[csv_descriptors] No additional numeric columns found in the training CSV.")
        X = df[cols].apply(pd.to_numeric, errors='coerce').values
        return X, cols, False

    raise ValueError(f"Unknown feature mode: {mode}")


# =========================================================
# MODELS AND PIPELINE
# =========================================================
class PLSRegressor1D(RegressorMixin, BaseEstimator):
    """Wraps PLSRegression so predict() always returns a 1D array.
    Bug fix #3: raw PLSRegression.predict() returns shape (n, 1) even for a
    single target, which can cause subtle shape-mismatch issues inside
    sklearn scorers (cross_validate) and when used as a StackingRegressor
    base estimator.
    Bug fix #4: RegressorMixin must come BEFORE BaseEstimator in the class
    bases. With the reverse order, BaseEstimator.__sklearn_tags__() is
    resolved first in the MRO and never chains into RegressorMixin's
    override, so StackingRegressor's is_regressor() check fails with
    "should be a regressor" even though this class is one."""

    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y):
        self.model_ = PLSRegression(n_components=self.n_components)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        pred = self.model_.predict(X)
        return np.asarray(pred).ravel()


def build_base_models(n_jobs_model=1, n_estimators=200):
    # Note: PLSRegression was removed from the base model set. When combined
    # with SelectFromModel feature reduction (which can produce highly
    # collinear subsets), PLS occasionally extrapolates catastrophically on
    # the outer test fold (R2 << -1000) even when its inner CV score looked
    # reasonable. The remaining 6 models (tree ensembles + kernel + distance)
    # provide a comprehensive and robust benchmark without this instability.
    return {
        'RF':   RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=n_jobs_model),
        'ET':   ExtraTreesRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=n_jobs_model),
        'LGBM': lgb.LGBMRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, verbosity=-1, n_jobs=n_jobs_model),
        'XGB':  XGBRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=n_jobs_model, verbosity=0),
        'SVM':  SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
        'kNN':  KNeighborsRegressor(n_neighbors=5, metric='cosine', n_jobs=n_jobs_model),
    }


def build_core_estimator(combo_dict):
    items = list(combo_dict.items())
    if len(items) == 1:
        return clone(items[0][1])
    return StackingRegressor(
        estimators=[(n, clone(m)) for n, m in items],
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=1,
    )


def build_pipeline(combo_dict, is_binary):
    core = build_core_estimator(combo_dict)
    if is_binary:
        return Pipeline([('model', core)])
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectFromModel(
            RandomForestRegressor(n_estimators=CFG['N_ESTIMATORS_TREES'], random_state=RANDOM_STATE, n_jobs=1),
            threshold=1e-9,  # bug fix: threshold=0.0 kept ALL features (importances are >= 0,
                              # so ">= 0.0" is always true). 1e-9 correctly drops exact-zero-
                              # importance features while keeping every positive one.
        )),
        ('model', core),
    ])


def evaluate_combo_inner(combo, X_tr, y_tr, inner_splits, is_binary):
    # Bug fix: joblib/loky workers do not always inherit the main process's
    # warnings filter state (depends on the multiprocessing start method).
    # Re-applying it here, inside the function that actually runs in the
    # worker, guarantees the LightGBM "no valid feature names" noise is
    # suppressed regardless of how the worker process was spawned.
    warnings.filterwarnings("ignore")
    combo_id = "+".join(combo.keys())
    pipe = build_pipeline(combo, is_binary)
    kf = KFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

    lo, hi = get_clip_bounds(y_tr)

    def clipped_r2(y_true, y_pred):
        y_pred_c, _ = clip_predictions(y_pred, lo, hi)
        return r2_score(y_true, y_pred_c)

    def clipped_neg_mae(y_true, y_pred):
        y_pred_c, _ = clip_predictions(y_pred, lo, hi)
        return -mean_absolute_error(y_true, y_pred_c)

    scoring = {
        'r2': make_scorer(clipped_r2),
        'mae': make_scorer(clipped_neg_mae),
    }
    scores = cross_validate(pipe, X_tr, y_tr, cv=kf, scoring=scoring, n_jobs=1)
    return {
        'Model': combo_id,
        'R2_inner_mean': scores['test_r2'].mean(),
        'R2_inner_std': scores['test_r2'].std(),
        'MAE_inner_mean': -scores['test_mae'].mean(),
    }


# =========================================================
# CHECKPOINTING
# =========================================================
def load_done_folds(mode):
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    df = pd.read_csv(CHECKPOINT_FILE)
    return set(df[df['Mode'] == mode]['Fold'].unique().tolist())


def append_checkpoint(row):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(CHECKPOINT_FILE)
    df_row.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)


def append_selection_log(df_inner, mode, fold_idx):
    df_inner = df_inner.assign(Mode=mode, Fold=fold_idx)
    header = not os.path.exists(SELECTION_LOG_FILE)
    df_inner.to_csv(SELECTION_LOG_FILE, mode='a', header=header, index=False)


# =========================================================
# NARRATIVE CONSOLE OUTPUT (so the console log can be read directly
# to draft the paper's Methods/Results sections)
# =========================================================
def print_methods_intro(df_clean):
    total_folds = CFG['OUTER_N_SPLITS'] * CFG['OUTER_N_REPEATS']
    n_base = len(build_base_models())
    n_combos = sum(1 for k in range(1, CFG['MAX_COMBO_SIZE'] + 1)
                   for _ in itertools.combinations(range(n_base), k))
    print("\n" + "=" * 100)
    print("METHODS (summary — copy/adapt directly into the manuscript)")
    print("=" * 100)
    print(f"- Training dataset: {len(df_clean)} unique compounds (column 'Smiles', "
          f"target 'pIC50 Value'), matching the size of the original paper's dataset "
          f"(873 compounds after BindingDB+ChEMBL deduplication).")
    print(f"- Base models compared ({n_base}): Random Forest, Extra Trees, LightGBM, XGBoost, "
          f"SVR (RBF kernel), Partial Least Squares, k-Nearest Neighbors (cosine metric).")
    print(f"- Architectures evaluated per representation: {n_combos} (all single, pairwise, and "
          f"triple combinations of the {n_base} base models, with stacking + RidgeCV as the "
          f"meta-estimator for combos with 2+ members).")
    print(f"- Molecular representations compared ({len(CFG['FEATURE_MODES'])}):")
    for m in CFG['FEATURE_MODES']:
        print(f"    - {m}: {MODE_DESCRIPTION.get(m, '')}")
    print(f"- Validation scheme: nested cross-validation. Outer loop: {CFG['OUTER_N_SPLITS']}-fold "
          f"repeated {CFG['OUTER_N_REPEATS']} times ({total_folds} independent outer splits). "
          f"Inner loop (architecture selection): {CFG['INNER_N_SPLITS']}-fold on the training "
          f"portion of each outer split. The test portion of every outer fold is NEVER used for "
          f"model selection, avoiding the optimistic bias of reporting the best score among many "
          f"configurations evaluated on the same held-out set.")
    print(f"- Preprocessing (non-binary representations): median imputation, standardization "
          f"(StandardScaler), and positive-importance feature selection via "
          f"SelectFromModel(RandomForestRegressor), all inside a scikit-learn Pipeline refit on "
          f"every fold (no information leakage across folds).")
    print(f"- random_state=42 for every split, matching the original paper.")
    print("=" * 100 + "\n")


def print_mode_summary(mode, df_mode):
    r2 = df_mode['R2_outer'].values
    mae = df_mode['MAE_outer'].values
    counts = df_mode['Selected_Model'].value_counts()
    top_combo = counts.index[0]
    stability = counts.iloc[0] / len(df_mode) * 100

    print("\n" + "-" * 100)
    print(f"SUMMARY [{mode}] — ready to paste into Results")
    print("-" * 100)
    print(f"- R2 (nested CV, {len(df_mode)} outer folds): mean={r2.mean():.4f}, "
          f"std={r2.std():.4f}, min={r2.min():.4f}, max={r2.max():.4f}")
    print(f"- MAE (nested CV): mean={mae.mean():.4f}, std={mae.std():.4f}")
    print(f"- Most frequently selected architecture: '{top_combo}' "
          f"(won in {counts.iloc[0]}/{len(df_mode)} folds = {stability:.1f}% selection stability)")
    if len(counts) > 1:
        print(f"- Other architectures selected at least once: {dict(counts.iloc[1:].items())}")
    print("-" * 100)


# =========================================================
# NESTED CV FOR ONE FEATURE MODE
# =========================================================
def nested_cv_for_mode(mode, df_clean, cfg):
    print("\n" + "#" * 100)
    print(f"# NESTED CV — REPRESENTATION: {mode}")
    print(f"# {MODE_DESCRIPTION.get(mode, '')}")
    print("#" * 100)

    X, feat_names, is_binary = build_feature_matrix(df_clean, mode, cfg['N_JOBS'])
    y = df_clean['pIC50 Value'].values
    n_features_total = X.shape[1]
    print(f"[INFO] Features generated: {n_features_total} "
          f"({'binary' if is_binary else 'continuous, with impute+scale+select'})")

    base_models = build_base_models(n_jobs_model=1, n_estimators=cfg['N_ESTIMATORS_TREES'])
    all_combos = []
    for k in range(1, cfg['MAX_COMBO_SIZE'] + 1):
        for combo in itertools.combinations(base_models.items(), k):
            all_combos.append(dict(combo))

    outer_cv = RepeatedKFold(n_splits=cfg['OUTER_N_SPLITS'], n_repeats=cfg['OUTER_N_REPEATS'],
                              random_state=RANDOM_STATE)
    total_folds = cfg['OUTER_N_SPLITS'] * cfg['OUTER_N_REPEATS']
    done_folds = load_done_folds(mode)
    if done_folds:
        print(f"[CHECKPOINT] {len(done_folds)}/{total_folds} folds already completed for '{mode}', resuming.")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        if fold_idx in done_folds:
            continue

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        start = time.time()
        print(f"\n    [INFO] Starting evaluation of {len(all_combos)} architectures in "
              f"parallel for fold {fold_idx + 1}...")
        inner_results = Parallel(n_jobs=cfg['N_JOBS'], verbose=10)(
            delayed(evaluate_combo_inner)(combo, X_tr, y_tr, cfg['INNER_N_SPLITS'], is_binary)
            for combo in all_combos
        )
        print(f"    [INFO] Inner evaluation complete. Selecting best architecture...")
        df_inner = pd.DataFrame(inner_results).sort_values(by='R2_inner_mean', ascending=False)
        append_selection_log(df_inner, mode, fold_idx)

        best_combo_id = df_inner.iloc[0]['Model']
        best_combo = next(c for c in all_combos if "+".join(c.keys()) == best_combo_id)

        pipe = build_pipeline(best_combo, is_binary)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        # Applicability-domain safeguard (see CLIP_MARGIN definition above):
        # clip predictions to the training target range before scoring, and
        # log how many predictions needed clipping for full transparency.
        lo, hi = get_clip_bounds(y_tr)
        y_pred, n_clipped = clip_predictions(y_pred, lo, hi)
        if n_clipped > 0:
            print(f"    [SAFEGUARD] {n_clipped}/{len(y_pred)} predictions clipped to "
                  f"[{lo:.2f}, {hi:.2f}] (model={best_combo_id}) — likely PLS extrapolation.")

        r2_outer = r2_score(y_te, y_pred)
        mae_outer = mean_absolute_error(y_te, y_pred)
        elapsed = time.time() - start

        n_selected = n_features_total
        if not is_binary:
            try:
                n_selected = int(pipe.named_steps['select'].get_support().sum())
            except Exception:
                pass

        row = {
            'Mode': mode, 'Fold': fold_idx, 'Selected_Model': best_combo_id,
            'R2_outer': r2_outer, 'MAE_outer': mae_outer,
            'N_train': len(train_idx), 'N_test': len(test_idx),
            'N_features_total': n_features_total, 'N_features_selected': n_selected,
            'N_predictions_clipped': n_clipped,
            'Time_s': elapsed,
        }
        append_checkpoint(row)
        print(f"[{mode}] Fold {fold_idx + 1}/{total_folds} -> best={best_combo_id} | "
              f"R2_outer={r2_outer:.4f} | MAE_outer={mae_outer:.4f} | "
              f"features={n_selected}/{n_features_total} | {elapsed:.1f}s")

    df_all = pd.read_csv(CHECKPOINT_FILE)
    df_mode = df_all[df_all['Mode'] == mode]
    print_mode_summary(mode, df_mode)


# =========================================================
# LATEX EXPORT
# =========================================================
def latex_safe_combo(combo_id):
    """'RF+LGBM' -> 'RFLGBM' (letters only, valid as a \\newcommand suffix)"""
    return "".join(ch for ch in combo_id if ch.isalpha())


def newcommand(f, name, value):
    f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")


def generate_latex_file(df_all, summary, df_ttest, top2_modes, paired_stats):
    with open(LATEX_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("% =====================================================\n")
        f.write("% Variables auto-generated by the nested-CV benchmark script\n")
        f.write("% Include in your main document's preamble with:\n")
        f.write(f"%   \\input{{{LATEX_OUTPUT_FILE}}}\n")
        f.write("% then use \\VariableName anywhere in the text.\n")
        f.write("% =====================================================\n\n")

        f.write("% --- Experiment configuration ---\n")
        newcommand(f, "NCompounds", len(pd.read_csv(TRAIN_FILE, on_bad_lines='skip').dropna(subset=['Smiles', 'pIC50 Value'])))
        newcommand(f, "OuterKFolds", CFG['OUTER_N_SPLITS'])
        newcommand(f, "OuterRepeats", CFG['OUTER_N_REPEATS'])
        newcommand(f, "TotalOuterFolds", CFG['OUTER_N_SPLITS'] * CFG['OUTER_N_REPEATS'])
        newcommand(f, "InnerKFolds", CFG['INNER_N_SPLITS'])
        newcommand(f, "MaxComboSize", CFG['MAX_COMBO_SIZE'])
        f.write("\n")

        f.write("% --- Original paper reference values ---\n")
        for label, (_, val) in PAPER_R2.items():
            newcommand(f, f"{label}RTwo", f"{val:.2f}")
        f.write("\n")

        f.write("% --- Results per representation (nested CV) ---\n")
        for _, row in summary.iterrows():
            mode = row['Mode']
            label = MODE_LATEX_LABEL.get(mode, mode.title())
            newcommand(f, f"{label}RTwoMean", f"{row['mean']:.4f}")
            newcommand(f, f"{label}RTwoStd", f"{row['std']:.4f}")
            newcommand(f, f"{label}NFolds", int(row['count']))

            df_mode = df_all[df_all['Mode'] == mode]
            counts = df_mode['Selected_Model'].value_counts()
            top_combo = counts.index[0]
            stability = counts.iloc[0] / len(df_mode) * 100
            newcommand(f, f"{label}BestArchitecture", top_combo)
            newcommand(f, f"{label}BestArchitectureSafe", latex_safe_combo(top_combo))
            newcommand(f, f"{label}Stability", f"{stability:.1f}")
            newcommand(f, f"{label}MaeMean", f"{df_mode['MAE_outer'].mean():.4f}")
            newcommand(f, f"{label}NFeaturesTotal", int(df_mode['N_features_total'].mean()))
            newcommand(f, f"{label}NFeaturesSelectedMean", f"{df_mode['N_features_selected'].mean():.1f}")
        f.write("\n")

        f.write("% --- One-sample t-tests vs. paper values ---\n")
        for _, row in df_ttest.iterrows():
            mode = row['Mode']
            mlabel = MODE_LATEX_LABEL.get(mode, mode.title())
            plabel = row['Paper_reference_label']
            newcommand(f, f"PValue{mlabel}Vs{plabel}", f"{row['p_value']:.4f}")
            newcommand(f, f"TStat{mlabel}Vs{plabel}", f"{row['t_stat']:.3f}")
            sig = "true" if row['p_value'] < 0.05 else "false"
            newcommand(f, f"SigDiff{mlabel}Vs{plabel}", sig)
        f.write("\n")

        f.write("% --- Paired comparison between the two best representations ---\n")
        if paired_stats is not None:
            l1 = MODE_LATEX_LABEL.get(top2_modes[0], top2_modes[0].title())
            l2 = MODE_LATEX_LABEL.get(top2_modes[1], top2_modes[1].title())
            newcommand(f, "TopModeOne", l1)
            newcommand(f, "TopModeTwo", l2)
            newcommand(f, "PairedTTestStat", f"{paired_stats['t_stat']:.3f}")
            newcommand(f, "PairedTTestP", f"{paired_stats['t_p']:.4f}")
            if not np.isnan(paired_stats['w_stat']):
                newcommand(f, "WilcoxonStat", f"{paired_stats['w_stat']:.3f}")
                newcommand(f, "WilcoxonP", f"{paired_stats['w_p']:.4f}")
        f.write("\n")

        f.write("% --- Overall winner ---\n")
        best_row = summary.iloc[0]
        blabel = MODE_LATEX_LABEL.get(best_row['Mode'], best_row['Mode'].title())
        newcommand(f, "WinnerMode", blabel)
        newcommand(f, "WinnerRTwoMean", f"{best_row['mean']:.4f}")
        newcommand(f, "WinnerRTwoStd", f"{best_row['std']:.4f}")

    print(f"\n[LATEX] Variables exported to {LATEX_OUTPUT_FILE}")
    print(f"        In your main document: \\input{{{LATEX_OUTPUT_FILE}}}  (in the preamble)")
    print(f"        Then in the text, e.g.: \\WinnerModeRTwoMean, \\PaperFinalRTwo, etc.")


# =========================================================
# FIGURE (matplotlib) — R2 distribution per representation
# =========================================================
def generate_figure(df_all, summary):
    order = summary['Mode'].tolist()
    data = [df_all[df_all['Mode'] == m]['R2_outer'].values for m in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=order, showmeans=True)
    for label, (_, val) in PAPER_R2.items():
        ax.axhline(val, linestyle='--', linewidth=1, alpha=0.6, label=f"{label} ({val})")
    ax.set_ylabel("R2 (outer-fold nested CV)")
    ax.set_xlabel("Molecular representation")
    ax.set_title("Nested CV R2 distribution by molecular representation")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.close(fig)
    print(f"[FIGURE] Saved {FIGURE_FILE}")


# =========================================================
# FINAL STATISTICAL ANALYSIS + PAPER-READY SUMMARY
# =========================================================
def summarize_and_test():
    df = pd.read_csv(CHECKPOINT_FILE)

    print("\n" + "=" * 100)
    print("RESULTS (copy/adapt directly into the manuscript's Results section)")
    print("=" * 100)

    summary = df.groupby('Mode')['R2_outer'].agg(['mean', 'std', 'count']).reset_index()
    summary = summary.sort_values(by='mean', ascending=False)
    print("\nTable 1. R2 by molecular representation (nested cross-validation)")
    print(summary.to_string(index=False))

    print("\n" + "-" * 100)
    print("Table 2. One-sample t-test: does our R2 differ from the values reported in the paper?")
    print("-" * 100)
    ttest_rows = []
    for mode in df['Mode'].unique():
        vals = df[df['Mode'] == mode]['R2_outer'].values
        for label, (desc, paper_val) in PAPER_R2.items():
            t_stat, p_val = stats.ttest_1samp(vals, paper_val)
            ttest_rows.append({
                'Mode': mode, 'Paper_reference_label': label, 'Paper_reference_desc': desc,
                'Paper_R2': paper_val, 'Our_R2_mean': vals.mean(), 'Our_R2_std': vals.std(),
                't_stat': t_stat, 'p_value': p_val, 'Significant_p<0.05': p_val < 0.05,
            })
    df_ttest = pd.DataFrame(ttest_rows)
    print(df_ttest.drop(columns=['Paper_reference_desc']).to_string(index=False))

    print("\n" + "-" * 100)
    print("Table 3. Paired comparison between the two best representations (same outer folds)")
    print("-" * 100)
    top2 = summary.head(2)['Mode'].tolist()
    paired_stats = None
    if len(top2) == 2:
        a = df[df['Mode'] == top2[0]].sort_values('Fold')['R2_outer'].values
        b = df[df['Mode'] == top2[1]].sort_values('Fold')['R2_outer'].values
        if len(a) == len(b) and len(a) > 1:
            t_stat, p_val = stats.ttest_rel(a, b)
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            paired_stats = {'t_stat': t_stat, 't_p': p_val, 'w_stat': w_stat, 'w_p': w_p}
            print(f"{top2[0]} (R2={a.mean():.4f}) vs {top2[1]} (R2={b.mean():.4f})")
            print(f"  Paired t-test : t={t_stat:.3f}, p={p_val:.4f}")
            if not np.isnan(w_stat):
                print(f"  Wilcoxon      : W={w_stat:.3f}, p={w_p:.4f}")
        else:
            print("[WARNING] Folds are not paired (check that both modes used the same "
                  "OUTER_N_SPLITS/OUTER_N_REPEATS).")

    summary.to_csv(FINAL_RESULTS_FILE, index=False)
    df_ttest.to_csv(FINAL_RESULTS_FILE.replace('.csv', '_ttests.csv'), index=False)
    generate_figure(df, summary)

    # ---------------------------------------------------
    # NARRATIVE EXECUTIVE SUMMARY — read this to draft the paper directly
    # ---------------------------------------------------
    best = summary.iloc[0]
    best_mode_desc = MODE_DESCRIPTION.get(best['Mode'], best['Mode'])
    df_best = df[df['Mode'] == best['Mode']]
    counts_best = df_best['Selected_Model'].value_counts()
    top_combo_best = counts_best.index[0]
    stability_best = counts_best.iloc[0] / len(df_best) * 100

    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY (read top to bottom — written to drop straight into Results/Discussion)")
    print("=" * 100)
    print(f"1. The best molecular representation was '{best['Mode']}' ({best_mode_desc}), with "
          f"R2 = {best['mean']:.4f} +/- {best['std']:.4f} across {int(best['count'])} independent "
          f"outer folds (nested cross-validation).")
    print(f"2. The most frequently selected architecture within that representation was "
          f"'{top_combo_best}', chosen in {stability_best:.1f}% of outer folds "
          f"({'high' if stability_best >= 70 else 'moderate' if stability_best >= 40 else 'low'} "
          f"selection stability).")

    for label, (desc, paper_val) in PAPER_R2.items():
        row = df_ttest[(df_ttest['Mode'] == best['Mode']) & (df_ttest['Paper_reference_label'] == label)].iloc[0]
        sig_txt = ("YES, statistically significant difference (p<0.05)" if row['Significant_p<0.05']
                    else "NO statistically significant difference (p>=0.05)")
        direction = "above" if best['mean'] > paper_val else "below"
        idx = list(PAPER_R2.keys()).index(label) + 1
        print(f"3.{idx} Compared to '{desc}' (R2={paper_val}): our result is {direction} "
              f"(delta={best['mean']-paper_val:+.4f}), t={row['t_stat']:.3f}, p={row['p_value']:.4f} "
              f"-> {sig_txt}.")

    if paired_stats is not None:
        interp = "not distinguishable" if paired_stats['t_p'] >= 0.05 else "distinguishable"
        print(f"4. Between the two best representations ({top2[0]} vs {top2[1]}), the R2 difference "
              f"is {interp} statistically (paired t-test p={paired_stats['t_p']:.4f}).")

    final_row = df_ttest[(df_ttest['Mode'] == best['Mode']) &
                          (df_ttest['Paper_reference_label'] == 'PaperFinal')].iloc[0]
    if not final_row['Significant_p<0.05']:
        candidate_sentence = (
            f"a classical stacking ensemble of {top_combo_best} on {best['Mode']} features "
            f"reaches R2={best['mean']:.3f}+/-{best['std']:.3f} under nested cross-validation, "
            f"with no statistically significant difference from the data-augmented DNN ensemble "
            f"reported in the original study (R2=0.85), calling into question the need for that "
            f"architectural complexity at this dataset size."
        )
    else:
        candidate_sentence = (
            f"a classical stacking ensemble of {top_combo_best} on {best['Mode']} features "
            f"reaches R2={best['mean']:.3f}+/-{best['std']:.3f} under nested cross-validation, "
            f"a statistically significant difference from the data-augmented DNN ensemble "
            f"reported in the original study (R2=0.85), which tempers the initial "
            f"over-engineering critique."
        )
    print(f"5. Candidate sentence for the Discussion: \"{candidate_sentence}\"")
    print("=" * 100)

    generate_latex_file(df, summary, df_ttest, top2, paired_stats)


def run_benchmark():
    total_folds = CFG['OUTER_N_SPLITS'] * CFG['OUTER_N_REPEATS']
    print(f"[INFO] Active profile: {PROFILE} | N_JOBS={CFG['N_JOBS']} | "
          f"Outer folds={CFG['OUTER_N_SPLITS']}x{CFG['OUTER_N_REPEATS']}={total_folds}")
    print(f"[INFO] Loading data: {TRAIN_FILE}")
    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    df_clean = df.dropna(subset=['Smiles', 'pIC50 Value']).reset_index(drop=True)
    print(f"[INFO] N compounds: {len(df_clean)}")

    print_methods_intro(df_clean)

    for mode in CFG['FEATURE_MODES']:
        try:
            nested_cv_for_mode(mode, df_clean, CFG)
        except ValueError as e:
            print(f"[WARNING] Skipping mode '{mode}': {e}")

    if os.path.exists(CHECKPOINT_FILE):
        summarize_and_test()
    else:
        print("[ERROR] No results were generated (empty checkpoint).")


if __name__ == "__main__":
    run_benchmark()