"""Prepares the rigid TgDHFR receptor (PDB 6AOG) for AutoDock Vina and reports the active-site centre.

Protein atoms only (waters, ions, cofactors and the co-crystallized ligand are dropped). Hydrogens are added at pH 7.4 with
OpenBabel before the PDBQT conversion: without them the backbone nitrogens are typed as H-bond acceptors (NA) instead of
donors (N + HD), which changes the Vina hydrogen-bond term. Requires the `obabel` binary; not part of run_pipeline.py.
"""
import os
import subprocess

import numpy as np

from paths_config import RECEPTOR_PDB, RECEPTOR_PDBQT, RECEPTOR_DIR

PH = 7.4


def prepare_receptor(pdb_file, ligand_code="CP6"):
    coords, protein_lines = [], []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("HETATM") and ligand_code in line:
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    pass
            if line.startswith(("ATOM", "TER")):
                protein_lines.append(line)

    if not coords:
        print(f"[ERROR] Ligand {ligand_code} not found in {pdb_file}.")
        return None
    center = np.mean(coords, axis=0)
    print(f"[+] Active-site centre from {ligand_code}: {center}")

    clean_pdb = os.path.join(RECEPTOR_DIR, "receptor_clean.pdb")
    protonated_pdb = os.path.join(RECEPTOR_DIR, "receptor_clean_H.pdb")
    with open(clean_pdb, "w") as f:
        f.writelines(protein_lines)

    subprocess.run(["obabel", "-ipdb", clean_pdb, "-p", str(PH), "-opdb", "-O", protonated_pdb], check=True)
    subprocess.run(["obabel", "-ipdb", protonated_pdb, "-opdbqt", "-xr", "-O", RECEPTOR_PDBQT], check=True)
    for tmp in (clean_pdb, protonated_pdb):
        os.remove(tmp)
    print(f"[+] Receptor written to {RECEPTOR_PDBQT}")
    return center


if __name__ == "__main__":
    center = prepare_receptor(RECEPTOR_PDB, ligand_code="CP6")
    if center is not None:
        print(f"center_x = {center[0]:.3f}\ncenter_y = {center[1]:.3f}\ncenter_z = {center[2]:.3f}")
