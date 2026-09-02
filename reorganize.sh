#!/bin/bash
# =========================================================
# reorganize.sh — Project folder restructuring script
# Run from the project root: bash reorganize.sh
# =========================================================

set -e  # stop on any error

echo "=================================================="
echo "Reorganizing project structure..."
echo "=================================================="

# Create directories
mkdir -p data receptor scripts results latex figures logs archive

# ── DATA ──────────────────────────────────────────────
mv -v V2-df_ic50_chmbl_CID_myFill.csv data/ 2>/dev/null || true
mv -v PubChem_FDA-approved_NoInorganics.csv data/ 2>/dev/null || true

# ── RECEPTOR ──────────────────────────────────────────
mv -v receptor.pdb receptor.pdbqt receptor/ 2>/dev/null || true
mv -v crystal_ligand.pdb crystal_ligand.pdbqt receptor/ 2>/dev/null || true

# ── SCRIPTS (active) ──────────────────────────────────
mv -v 0STACK.py scripts/ 2>/dev/null || true
mv -v 1AUGMENT.py scripts/ 2>/dev/null || true
mv -v 2FDA.py scripts/ 2>/dev/null || true
mv -v 3DOCKING.py scripts/ 2>/dev/null || true
mv -v 4REDOCKING_VALIDATION.py scripts/ 2>/dev/null || true
mv -v prep_receptor.py scripts/ 2>/dev/null || true

# ── SCRIPTS (old versions → archive) ──────────────────
mv -v 1FDA.py archive/ 2>/dev/null || true
mv -v 2DOCKING.py archive/ 2>/dev/null || true
mv -v 4W_REDOCKING_VALIDATION.py archive/ 2>/dev/null || true

# ── RESULTS ───────────────────────────────────────────
mv -v nested_cv_checkpoint.csv results/ 2>/dev/null || true
mv -v nested_cv_final_results.csv results/ 2>/dev/null || true
mv -v nested_cv_final_results_ttests.csv results/ 2>/dev/null || true
mv -v nested_cv_selection_log.csv results/ 2>/dev/null || true
mv -v FDA_Candidates_For_Docking.csv results/ 2>/dev/null || true
mv -v FDA_Detailed_Votes_Full.csv results/ 2>/dev/null || true
mv -v Final_Validation_Hybrid.csv results/ 2>/dev/null || true
mv -v best_model.joblib results/ 2>/dev/null || true
mv -v selected_features_mask.npy results/ 2>/dev/null || true

# ── LATEX ─────────────────────────────────────────────
mv -v paper_variables.tex latex/ 2>/dev/null || true
mv -v augment_variables.tex latex/ 2>/dev/null || true
mv -v docking_variables.tex latex/ 2>/dev/null || true
mv -v redocking_variables.tex latex/ 2>/dev/null || true
mv -v fda_variables.tex latex/ 2>/dev/null || true

# ── FIGURES ───────────────────────────────────────────
mv -v augment_r2_comparison.png figures/ 2>/dev/null || true
mv -v r2_by_representation_boxplot.png figures/ 2>/dev/null || true

# ── LOGS ──────────────────────────────────────────────
mv -v 1AUGMENT_output.txt logs/ 2>/dev/null || true
mv -v 1FDA_output.txt logs/ 2>/dev/null || true
mv -v 2FDA_output.txt logs/ 2>/dev/null || true
mv -v 3DOCKING_output.txt logs/ 2>/dev/null || true

echo ""
echo "=================================================="
echo "Done. New structure:"
echo "=================================================="
tree . --dirsfirst 2>/dev/null || find . -type f | sort
