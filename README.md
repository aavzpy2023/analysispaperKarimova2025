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

