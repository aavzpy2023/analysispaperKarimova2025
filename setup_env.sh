#!/usr/bin/env bash
# One-command environment setup (Linux / macOS). Run from the repository root:
#     bash setup_env.sh
# Windows: run the three commands below by hand in an Anaconda Prompt.
set -euo pipefail

ENV_NAME="mlvsgnn_repro"

conda env create -f conda_environment.yml -n "$ENV_NAME"

# PyTDC must be installed WITHOUT its dependencies: it declares rdkit-pypi (an old RDKit build)
# that would overwrite the RDKit version pinned in requirements.txt. The packages it really
# needs at import time (fuzzywuzzy, huggingface_hub, setuptools<81) are already in requirements.txt.
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --no-deps PyTDC==0.4.1

echo
echo "RDKit distributions installed (expected: a single 'rdkit' line):"
conda run --no-capture-output -n "$ENV_NAME" python -m pip list 2>/dev/null | grep -i rdkit || true

conda run --no-capture-output -n "$ENV_NAME" python run_pipeline.py --check
echo
echo "Done. Activate with:  conda activate $ENV_NAME   and run:  python run_pipeline.py"
