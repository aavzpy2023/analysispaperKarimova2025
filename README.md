# 🧬 Toxoplasma gondii: QSAR & Molecular Docking Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Area](https://img.shields.io/badge/Area-Computational_Drug_Discovery-purple?style=for-the-badge)

> **Algorithmic Audit & Optimization of Drug Discovery Frameworks**
>
> An advanced computational pipeline that integrates **Ensemble Machine Learning** (Stacking Regressors), **ADMET Profiling** (TDC), and **Bio-physical Simulations** (AutoDock Vina) to identify potent inhibitors of the *TgDHFR* enzyme. This project challenges and improves upon existing Deep Learning (GNN) baselines by applying the principle of parsimony, robust heuristic filtering, and rigorous validation.

---

## 🛠️ Tech Stack & Requirements

This project relies on a highly reproducible scientific stack managed via Conda to prevent C++/CUDA compilation conflicts.

### 🐍 Core Ecosystem
| Library | Version | Purpose |
| :--- | :--- | :--- |
| ![RDKit](https://img.shields.io/badge/RDKit-2025.9.3-orange) | `2025.9.3` | Cheminformatics & Heuristic Filtering |
| ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white) | `1.8.0` | Classical ML & Stacking Architectures |
| **PyTorch / DGL** | `>=2.0.0` | Graph Neural Network Baselines |
| **PyTDC** | `>=0.4.1` | ADMET Oracles (hERG, Caco-2) |
| **AutoDock Vina** | `1.2.5` | Molecular Docking Engine (Python bindings) |
| **Meeko / Gemmi** | `Latest` | PDBQT Ligand Preparation & Structure Handling |

### 🐧 System Dependencies (Ubuntu/Linux)
Molecular docking preparation requires specific system-level tools to handle PDB conversions.

```bash
# Update Repositories and install OpenBabel (Critical for PDB -> PDBQT)
sudo apt-get update
sudo apt-get install openbabel libxrender1 libxext6
