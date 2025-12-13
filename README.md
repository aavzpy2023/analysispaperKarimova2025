# 🧬 Toxoplasma gondii: QSAR & Molecular Docking Pipeline

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Area](https://img.shields.io/badge/Area-Computational_Drug_Discovery-purple?style=for-the-badge)

> **Algorithmic Audit & Optimization of Drug Discovery Frameworks**
>
> An advanced computational pipeline that integrates **Ensemble Machine Learning** (Stacking Regressors) with **Bio-physical Simulations** (AutoDock Vina) to identify potent inhibitors of the *TgDHFR* enzyme. This project challenges and improves upon existing Deep Learning baselines by applying the principle of parsimony and rigorous validation.

---

## 🛠️ Tech Stack & Requirements

This project relies on a precise scientific stack. Below are the core libraries and system tools required for reproduction.

### 🐍 Python Dependencies (Core)
| Library | Version | Purpose |
| :--- | :--- | :--- |
| ![RDKit](https://img.shields.io/badge/RDKit-2025.9.3-orange) | `2025.9.3` | Cheminformatics & Fingerprint Generation |
| ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white) | `1.8.0` | Classical ML & Stacking Architectures |
| ![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green) | `4.6.0` | High-Efficiency Gradient Boosting |
| **AutoDock Vina** | `1.2.5` | Molecular Docking Engine (Python bindings) |
| **Meeko** | `0.5.0` | PDBQT Ligand Preparation |
| **PubChemPy** | `1.0.5` | Chemical Data Retrieval |
| **Gemmi** | `Latest` | Macromolecular Structure Handling |

### 🐧 System Dependencies (Ubuntu/Linux)
Molecular docking preparation requires specific system-level tools to handle PDB conversions.

```bash
# 1. Update Repositories
sudo apt-get update

# 2. OpenBabel: Critical for converting PDB -> PDBQT (Receptor preparation)
sudo apt-get install openbabel

# 3. RDKit System Dependencies (rendering support)
sudo apt-get install libxrender1 libxext6
```

## 🚀 Pipeline Architecture

The workflow follows a strict sequential logic, moving from statistical auditing to physical validation.



```Mermaid

graph TD
    A[Raw Training Data] -->|0STACK.py| B(Algorithmic Audit)
    B --> C{Select Champion Model}
    C -->|Stacking: ET+LGBM| D[1FDA.py]
    E[FDA Approved Database] --> D
    D -->|Consensus Prediction| F[Top 15 Candidates]
    G[Receptor PDB] -->|prep_receptor.py| H[Receptor PDBQT]
    F --> I[2DOCKING.py]
    H --> I
    I --> J[Hybrid Validation Report]
```

## 💻 Usage Guide (Step-by-Step)

**0. Environment Setup**

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python requirements
pip install -r requirements.txt
```

**1. Model Auditing & Benchmarking**

Trains 10 different architectures (RF, SVM, LGBM, Stacking) to determine the optimal strategy via Grid Search.

```bash
python 0STACK.py
```

- Input: PubChem_FDA-approved_NoInorganics.csv
- Output: FDA_Candidates_For_Docking.csv (Filtered list of high-potential candidates).

**3. Receptor Preparation**

Cleans the raw protein structure and automatically calculates the active site center based on the original ligand position.

```bash
python prep_receptor.py
```

- Input: receptor.pdb (Target Structure, e.g., 6AOG).
-Output: receptor.pdbqt and center_x, center_y, center_z coordinates.

**4. Hybrid Validation (Molecular Docking)**

Performs physics-based simulations on the top candidates identified by the AI.

```bash
python 2DOCKING.py
```

- Input: FDA_Candidates_For_Docking.csv + receptor.pdbqt
- Output: Final_Validation_Hybrid_V2.csv (Comparison of AI predicted pIC50 vs. Physical Binding Energy).





## 👤 Author
Andrey Vinajera Zamora
