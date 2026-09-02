#!/usr/bin/env python3
"""
update_paths.py — Run this once after reorganize.sh to patch all scripts.
Place this in the scripts/ folder and run: python update_paths.py
"""
import os
import re

# Mapping: old hardcoded path strings -> new config variable
REPLACEMENTS = [
    # TRAIN_FILE
    (r'"./V2-df_ic50_chmbl_CID_myFill\.csv"', 'TRAIN_FILE'),
    (r"'./V2-df_ic50_chmbl_CID_myFill\.csv'", 'TRAIN_FILE'),
    # FDA_FILE
    (r'"PubChem_FDA-approved_NoInorganics\.csv"', 'FDA_FILE'),
    (r"'PubChem_FDA-approved_NoInorganics\.csv'", 'FDA_FILE'),
    # RECEPTOR
    (r'"receptor\.pdbqt"', 'RECEPTOR_PDBQT'),
    (r"'receptor\.pdbqt'", 'RECEPTOR_PDBQT'),
    (r'"receptor\.pdb"', 'RECEPTOR_PDB'),
    (r"'receptor\.pdb'", 'RECEPTOR_PDB'),
    # RESULTS
    (r'"nested_cv_checkpoint\.csv"', 'CHECKPOINT_FILE'),
    (r"'nested_cv_checkpoint\.csv'", 'CHECKPOINT_FILE'),
    (r'"nested_cv_selection_log\.csv"', 'SELECTION_LOG_FILE'),
    (r"'nested_cv_selection_log\.csv'", 'SELECTION_LOG_FILE'),
    (r'"nested_cv_final_results\.csv"', 'FINAL_RESULTS_FILE'),
    (r"'nested_cv_final_results\.csv'", 'FINAL_RESULTS_FILE'),
    (r'"best_model\.joblib"', 'MODEL_FILE'),
    (r"'best_model\.joblib'", 'MODEL_FILE'),
    (r'"selected_features_mask\.npy"', 'MASK_FILE'),
    (r"'selected_features_mask\.npy'", 'MASK_FILE'),
    (r'"FDA_Candidates_For_Docking\.csv"', 'FDA_CANDIDATES_CSV'),
    (r"'FDA_Candidates_For_Docking\.csv'", 'FDA_CANDIDATES_CSV'),
    (r'"Final_Validation_Hybrid\.csv"', 'DOCKING_RESULTS_CSV'),
    (r"'Final_Validation_Hybrid\.csv'", 'DOCKING_RESULTS_CSV'),
    # LATEX
    (r'"paper_variables\.tex"', 'LATEX_PAPER'),
    (r"'paper_variables\.tex'", 'LATEX_PAPER'),
    (r'"augment_variables\.tex"', 'LATEX_AUGMENT'),
    (r"'augment_variables\.tex'", 'LATEX_AUGMENT'),
    (r'"fda_variables\.tex"', 'LATEX_FDA'),
    (r"'fda_variables\.tex'", 'LATEX_FDA'),
    (r'"docking_variables\.tex"', 'LATEX_DOCKING'),
    (r"'docking_variables\.tex'", 'LATEX_DOCKING'),
    (r'"redocking_variables\.tex"', 'LATEX_REDOCKING'),
    (r"'redocking_variables\.tex'", 'LATEX_REDOCKING'),
    # FIGURES
    (r'"r2_by_representation_boxplot\.png"', 'FIGURE_NESTED_CV'),
    (r"'r2_by_representation_boxplot\.png'", 'FIGURE_NESTED_CV'),
    (r'"augment_r2_comparison\.png"', 'FIGURE_AUGMENT'),
    (r"'augment_r2_comparison\.png'", 'FIGURE_AUGMENT'),
]

IMPORT_LINE = "from paths_config import *\n"

SCRIPTS = [
    "0STACK.py",
    "1AUGMENT.py",
    "2FDA.py",
    "3DOCKING.py",
    "4REDOCKING_VALIDATION.py",
    "prep_receptor.py",
]

def patch_script(filename):
    if not os.path.exists(filename):
        print(f"  [SKIP] {filename} not found")
        return

    with open(filename, 'r') as f:
        content = f.read()

    # Add import if not already present
    if IMPORT_LINE.strip() not in content:
        # Insert after the last os.environ line or after imports block
        insert_after = "import warnings\n"
        if insert_after in content:
            content = content.replace(
                insert_after,
                insert_after + IMPORT_LINE,
                1
            )
        else:
            content = IMPORT_LINE + content

    # Apply path replacements
    n_replacements = 0
    for pattern, replacement in REPLACEMENTS:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            n_replacements += 1
            content = new_content

    # Remove old hardcoded TRAIN_FILE = ... lines since now from paths_config
    old_vars = [
        r'TRAIN_FILE\s*=\s*["\'].*?["\']',
        r'FDA_FILE\s*=\s*["\'].*?["\']',
        r'RECEPTOR_FILE\s*=\s*["\'].*?["\']',
        r'CHECKPOINT_FILE\s*=\s*["\'].*?["\']',
        r'SELECTION_LOG_FILE\s*=\s*["\'].*?["\']',
        r'FINAL_RESULTS_FILE\s*=\s*["\'].*?["\']',
        r'MODEL_FILE\s*=\s*["\'].*?["\']',
        r'MASK_FILE\s*=\s*["\'].*?["\']',
        r'OUTPUT_CSV\s*=\s*["\']FDA_Candidates.*?["\']',
        r'OUTPUT_CSV\s*=\s*["\']Final_Validation.*?["\']',
        r'LATEX_OUTPUT_FILE\s*=\s*["\'].*?["\']',
        r'FIGURE_FILE\s*=\s*["\'].*?["\']',
    ]
    for pattern in old_vars:
        content = re.sub(pattern + r'\n', '', content)

    with open(filename, 'w') as f:
        f.write(content)

    print(f"  [OK] {filename} — {n_replacements} path(s) updated")

if __name__ == "__main__":
    print("=" * 60)
    print("update_paths.py — Patching scripts with paths_config")
    print("=" * 60)
    for script in SCRIPTS:
        patch_script(script)
    print("\n[DONE] All scripts updated.")
    print("Verify with: grep -n 'TRAIN_FILE\|FDA_FILE\|RECEPTOR' *.py")
