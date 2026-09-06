# =========================================================
# paths_config.py — Central path configuration
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
FDA_CANDIDATES_CSV       = FDA_ADMET_CANDIDATES_CSV
DOCKING_RESULTS_CSV  = os.path.join(RESULTS_DIR, "Final_Validation_Hybrid.csv")

# LaTeX files
LATEX_PAPER     = os.path.join(LATEX_DIR, "paper_variables.tex")
LATEX_AUGMENT   = os.path.join(LATEX_DIR, "augment_variables.tex")
LATEX_FDA       = os.path.join(LATEX_DIR, "fda_variables.tex")
LATEX_DOCKING   = os.path.join(LATEX_DIR, "docking_variables.tex")
LATEX_REDOCKING = os.path.join(LATEX_DIR, "redocking_variables.tex")

# Figure files
FIGURE_NESTED_CV = os.path.join(FIGURES_DIR, "r2_by_representation_boxplot.png")
FIGURE_AUGMENT   = os.path.join(FIGURES_DIR, "augment_r2_comparison.png")

# Ensure all output directories exist
for d in [RESULTS_DIR, LATEX_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
