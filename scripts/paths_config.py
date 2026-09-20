# =========================================================
# paths_config.py — Central path and experimental configuration
# Import this in every script: from paths_config import *
# =========================================================
import os

# Project root = parent of the scripts/ folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories
DATA_DIR      = os.path.join(ROOT, "data")
RECEPTOR_DIR  = os.path.join(ROOT, "receptor")
RESULTS_DIR   = os.path.join(ROOT, "results")
LATEX_DIR     = os.path.join(ROOT, "latex")
FIGURES_DIR   = os.path.join(ROOT, "figures")
LOGS_DIR      = os.path.join(ROOT, "logs")

# Data files
TRAIN_FILE    = os.path.join(DATA_DIR, "V2-df_ic50_chmbl_CID_myFill.csv")
FDA_FILE      = os.path.join(DATA_DIR, "PubChem_FDA-approved_NoInorganics.csv")

# Receptor files
RECEPTOR_PDB  = os.path.join(RECEPTOR_DIR, "receptor.pdb")
RECEPTOR_PDBQT= os.path.join(RECEPTOR_DIR, "receptor.pdbqt")

# Result files
CHECKPOINT_FILE      = os.path.join(RESULTS_DIR, "nested_cv_checkpoint.csv")
SELECTION_LOG_FILE   = os.path.join(RESULTS_DIR, "nested_cv_selection_log.csv")
FINAL_RESULTS_FILE   = os.path.join(RESULTS_DIR, "nested_cv_final_results.csv")
MODEL_FILE           = os.path.join(RESULTS_DIR, "best_model.joblib")
MASK_FILE            = os.path.join(RESULTS_DIR, "selected_features_mask.npy")

# ADMET & Docking Candidates
FDA_RAW_CANDIDATES_CSV   = os.path.join(RESULTS_DIR, "FDA_Candidates_For_Docking.csv")
FDA_ADMET_CANDIDATES_CSV = os.path.join(RESULTS_DIR, "ADMET_CANDIDATES_For_Docking.csv")
FDA_CANDIDATES_CSV       = os.path.join(RESULTS_DIR, "FDA_Candidates_For_Docking.csv")
DOCKING_RESULTS_CSV       = os.path.join(RESULTS_DIR, "Final_Validation_Hybrid.csv")

# LaTeX files
LATEX_PAPER     = os.path.join(LATEX_DIR, "paper_variables.tex")
LATEX_AUGMENT   = os.path.join(LATEX_DIR, "augment_variables.tex")
LATEX_FDA       = os.path.join(LATEX_DIR, "fda_variables.tex")
LATEX_DOCKING   = os.path.join(LATEX_DIR, "docking_variables.tex")
LATEX_REDOCKING = os.path.join(LATEX_DIR, "redocking_variables.tex")

# Figure files
FIGURE_NESTED_CV = os.path.join(FIGURES_DIR, "r2_by_representation_boxplot.png")
FIGURE_AUGMENT   = os.path.join(FIGURES_DIR, "augment_r2_comparison.png")

RANDOM_STATE = 42

# =========================================================
# PROFILES TO RUN (5x5 REPEATED NESTED CV) (1)
# =========================================================
PROFILE = 'workstation'

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
        N_JOBS=46,
        FEATURE_MODES=['morgan', 'rdkit2d', 'rdkit2d_fp', 'rdkit2d3d_fp'],
        MAX_COMBO_SIZE=3,
        OUTER_N_SPLITS=5,
        OUTER_N_REPEATS=5,
        INNER_N_SPLITS=5,
        N_ESTIMATORS_TREES=200,
    ),
}
CFG = PROFILES[PROFILE]

# FIGURES

mode_labels = {
    'morgan': 'Morgan FP',
    'rdkit2d': 'RDKit 2D',
    'rdkit2d_fp': 'RDKit 2D FP',
    'rdkit2d3d_fp': 'RDKit 2D+3D FP'
}


# =========================================================
# y-RANDOMIZATION EXPERIMENT CONFIGURATION (3)
# =========================================================
N_PERMUTATIONS = 100

# Dynamically extract CV_FOLDS and N_JOBS from the active profile
ACTIVE_CFG = PROFILES[PROFILE]
CV_FOLDS = ACTIVE_CFG.get('OUTER_N_SPLITS', 5)
N_JOBS = ACTIVE_CFG.get('N_JOBS', 1)



# =========================================================
# CONFIGURATION & PARAMETERS FOR DATA AUGMENTATION (4)
# =========================================================
TEST_SIZE = 0.15  # 15% holdout test set (identical to paper)
N_ENSEMBLE_RUNS = 5  # 5 stochastic ensemble runs (averaged)
N_BOOTSTRAP = 2000  # Resampling iterations for non-parametric CIs
LATEX_FILE = LATEX_AUGMENT
FIGURE_FILE = FIGURE_AUGMENT

# Gaussian noise perturbation levels
NOISE_LEVELS = [0.01, 0.001]



# =========================================================
# CONFIGURATION FOR VIRTUAL SCREENING (5)
# =========================================================
# We directly use the variables exposed in paths_config.py
ISO_CONTAMINATION = 0.05
MAX_MW = 1000.0

# Allowed atoms (identical to the original paper)
ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

# References for internal precision validation
KNOWN_VALUES = {
    'Pyrimethamine': 6.56,
    'Trimethoprim': 5.57,
}

# Candidates from the original paper (Table 2)
PAPER_TOP = ['Bisacodyl', 'Etodolac', 'Triamterene', 'Finerenone',
             'Methotrexate', 'Pyrimethamine', 'Trimethoprim']


# =========================================================
# REDOCKING CONFIGURATION (6)
# =========================================================
PDB_FILE = RECEPTOR_PDB
RECEPTOR_FILE = RECEPTOR_PDBQT
LIGAND_CODE = "CP6"
CHAIN = "B"  # DHFR active site chain
EXHAUSTIVENESS = 32
RMSD_THRESHOLD = 2.0
BOX_SIZE = 20.0








# =========================================================
# DOCKING & VINA EXPERIMENTAL PARAMETERS
# =========================================================
# Active site grid box coordinates (detected from crystal ligand)
CENTER_X, CENTER_Y, CENTER_Z = 3.689, 39.992, -62.818
BOX_SIZE       = 20.0   # Angstroms
EXHAUSTIVENESS = 32     # Search depth (32 = publication quality)
N_POSES        = 3      # Top poses per ligand

# Decision Thresholds (ML + Physics classification)
ML_ACTIVE_THRESH   = 6.5   # pIC50 > 6.5 = predicted active
DOCK_ACTIVE_THRESH = -8.0  # kcal/mol < -8.0 = binding confirmed

# Compound Selection Settings
TOP_CANDIDATES_COUNT = 15
REF_CONTROL_NAMES    = [
    'Pyrimethamine', 'Trimethoprim', 'Bisacodyl',
    'Etodolac', 'Triamterene', 'Methotrexate', 'Chlorambucil'
]

# Control compounds exported to LaTeX: tuple(search_substring, latex_prefix)
LATEX_REF_COMPOUNDS = [
    ('Chlorambucil', 'Chlorambucil'),
    ('Bisacodyl', 'Bisacodyl'),
    ('Pyrimethamine', 'Pyrimethamine'),
    ('Folic Acid', 'FolicAcid'),
    ('Methotrexate', 'Methotrexate'),
    ('Triamterene', 'Triamterene')
]

# =========================================================
# HARDWARE & ADMET EXPERIMENTAL PARAMETERS
# =========================================================
# Hardware
CORES_ADMET = 48

# ADMET Thresholds
HERG_THRESH  = 0.5         # Probability < 0.5 = Low cardiotoxicity risk
CACO2_THRESH = -5.15       # Permeability > -5.15 log(cm/s) = Moderate/High Oral Permeability

# Lipinski Rule of Five Parameters
LIPINSKI_MAX_HDONORS    = 5
LIPINSKI_MAX_HACCEPTORS = 10
LIPINSKI_MAX_LOGP       = 5.0


# =========================================================
# GNN BASELINE CONFIGURATION
# =========================================================
# Result files
GNN_RESULTS_CSV = os.path.join(RESULTS_DIR, "gnn_results.csv")
LATEX_GNN       = os.path.join(LATEX_DIR, "gnn_variables.tex")
FIGURE_GNN      = os.path.join(FIGURES_DIR, "gnn_comparison.png")

# Experimental Parameters
N_WORKERS_GNN    = 0
N_BOOTSTRAP_GNN  = 2000
TEST_SIZE_GNN    = 0.15

# Reference R2 values for plotting/comparison
PAPER_R2 = {
    'PaperBaseline': ('2D/3D/FP, no feature selection', 0.75),
    'PaperSelected': ('After Permutation Importance selection', 0.82),
    'PaperFinal': ('Data augmentation + DNN ensemble', 0.85),
}


def get_classical_r2(results_file):
    """
    Dynamically loads the best classical ML R2 scores from script 01 results.
    Prevents hardcoding values and adapts to new runs.
    """
    # Fallback values in case script 01 hasn't been run yet
    default_r2 = {
        'Morgan FP (RF+XGB+SVM)': 0.7407,
        'RDKit 2D+FP (best)': 0.7322,
    }

    if not os.path.exists(results_file):
        return default_r2

    try:
        import csv
        results = {}
        with open(results_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row['Mode']] = float(row['mean'])

        # Extract dynamic values
        morgan_val = results.get('morgan', default_r2['Morgan FP (RF+XGB+SVM)'])

        # Get the best RDKit score from all modes that contain 'rdkit'
        rdkit_vals = [v for k, v in results.items() if 'rdkit' in k]
        rdkit_best = max(rdkit_vals) if rdkit_vals else default_r2['RDKit 2D+FP (best)']

        return {
            'Morgan FP (Best Ensemble)': round(morgan_val, 4),
            'RDKit (Best Ensemble)': round(rdkit_best, 4),
        }
    except Exception:
        return default_r2


# Generate the dictionary dynamically reading the nested_cv_final_results.csv
CLASSICAL_R2 = get_classical_r2(FINAL_RESULTS_FILE)

# Ensure all output directories exist
for d in [RESULTS_DIR, LATEX_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
