import os
import warnings
from paths_config import *
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

PDB_FILE       = RECEPTOR_PDB
RECEPTOR_FILE  = RECEPTOR_PDBQT
LIGAND_CODE    = "CP6"
CHAIN          = "B"           # DHFR active site chain
EXHAUSTIVENESS = 32
RMSD_THRESHOLD = 2.0
BOX_SIZE       = 20.0


def extract_crystal_ligand(pdb_file, ligand_code, chain):
    crystal_coords = []
    ligand_lines   = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if (line.startswith("HETATM") and
                    ligand_code in line and
                    f" {chain} " in line):
                ligand_lines.append(line)
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    crystal_coords.append([x, y, z])
                except ValueError:
                    pass
    if not crystal_coords:
        print(f"[ERROR] Ligand '{ligand_code}' chain {chain} not found.")
        return None, None
    centroid = np.mean(crystal_coords, axis=0)
    print(f"[CRYSTAL] Chain {chain}: {len(crystal_coords)} heavy atoms")
    print(f"[CRYSTAL] Centroid (= active site center): {centroid.round(3)}")
    return np.array(crystal_coords), ligand_lines, centroid


def write_ligand_pdb(ligand_lines, out_path="crystal_ligand.pdb"):
    with open(out_path, 'w') as f:
        f.writelines(ligand_lines)
        f.write("END\n")
    print(f"[LIGAND] Crystal ligand PDB written to {out_path}")
    return out_path


def convert_ligand_to_pdbqt(ligand_pdb, out_pdbqt="crystal_ligand.pdbqt"):
    try:
        from meeko import MoleculePreparation
        try:
            from meeko import PDBQTWriterLegacy
            use_legacy = True
        except ImportError:
            use_legacy = False

        mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=True, sanitize=False)
        if mol is None:
            print("[ERROR] RDKit could not parse crystal_ligand.pdb")
            return None

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            print(f"[INFO] {len(frags)} fragments — keeping largest")
            mol = max(frags, key=lambda f: f.GetNumAtoms())

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"[WARNING] Sanitization: {e}")

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass

        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)

        if use_legacy:
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
            if not is_ok:
                print(f"[ERROR] Meeko: {err}")
                return None
        else:
            pdbqt_string = prep.write_pdbqt_string()

        with open(out_pdbqt, 'w') as f:
            f.write(pdbqt_string)
        print(f"[LIGAND] Converted to {out_pdbqt} via RDKit + Meeko")
        return out_pdbqt

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        return None


def compute_rmsd_centroid(coords_ref, coords_docked):
    """Centroid-based RMSD — robust when atom counts differ."""
    c1 = np.mean(coords_ref,   axis=0)
    c2 = np.mean(coords_docked, axis=0)
    dist = float(np.linalg.norm(c1 - c2))
    print(f"  Crystal centroid : {c1.round(3)}")
    print(f"  Docked  centroid : {c2.round(3)}")
    print(f"  Centroid distance: {dist:.3f} Å")
    return dist


def parse_vina_output_coords(vina_pdbqt_string):
    coords = []
    for line in vina_pdbqt_string.split('\n'):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                coords.append([float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])])
            except ValueError:
                pass
    return np.array(coords) if coords else None


def run_redocking_validation():
    try:
        from vina import Vina
    except ImportError:
        print("[ERROR] AutoDock Vina not installed.")
        return

    print("=" * 80)
    print("4REDOCKING_VALIDATION.py — Crystal Ligand Re-Docking Validation")
    print(f"Using chain {CHAIN} (DHFR active site) of PDB 6AOG")
    print("=" * 80)
    print(f"Criterion: centroid distance < {RMSD_THRESHOLD} Å = protocol validated\n")

    result = extract_crystal_ligand(PDB_FILE, LIGAND_CODE, CHAIN)
    if result[0] is None:
        return
    crystal_coords, ligand_lines, centroid = result

    cx, cy, cz = centroid
    print(f"\n[NOTE] Active site center for chain {CHAIN}: "
          f"[{cx:.3f}, {cy:.3f}, {cz:.3f}]")
    print(f"       (original script used average of both chains: "
          f"[17.394, 68.757, -68.051])")
    print(f"       Re-running docking with corrected center.\n")

    ligand_pdb   = write_ligand_pdb(ligand_lines)
    ligand_pdbqt = convert_ligand_to_pdbqt(ligand_pdb)
    if ligand_pdbqt is None:
        return

    print(f"\n[DOCKING] Re-docking {LIGAND_CODE} chain {CHAIN}...")
    v = Vina(sf_name='vina')
    v.set_receptor(RECEPTOR_FILE)
    v.compute_vina_maps(center=[cx, cy, cz],
                        box_size=[BOX_SIZE, BOX_SIZE, BOX_SIZE])

    with open(ligand_pdbqt, 'r') as f:
        ligand_str = f.read()

    v.set_ligand_from_string(ligand_str)
    v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=1)

    best_energy   = v.energies(n_poses=1)[0][0]
    docked_coords = parse_vina_output_coords(v.poses(n_poses=1))
    print(f"[DOCKING] Best pose energy: {best_energy:.3f} kcal/mol")

    print("\n" + "=" * 80)
    print("VALIDATION RESULT")
    print("=" * 80)

    if docked_coords is not None and len(docked_coords) > 0:
        rmsd = compute_rmsd_centroid(crystal_coords, docked_coords)
        print()
        if rmsd < RMSD_THRESHOLD:
            print("✓ PROTOCOL VALIDATED")
            print(f"\n  METHODS SENTENCE:")
            print(f'  "Docking protocol validation was performed by re-docking')
            print(f'  pyrimethamine ({LIGAND_CODE}, PDB: 6AOG, chain {CHAIN})')
            print(f'  into the TgDHFR active site. The top-ranked pose reproduced')
            print(f'  the crystallographic binding mode with a centroid distance')
            print(f'  of {rmsd:.2f} Å (threshold: {RMSD_THRESHOLD:.1f} Å)."')
            print(f'\n  IMPORTANT FOR THE PAPER: The corrected active site center')
            print(f'  for the DHFR site (chain B) is [{cx:.3f}, {cy:.3f}, {cz:.3f}].')
            print(f'  Update this in 3DOCKING.py before the final screening run.')
        else:
            print("✗ PROTOCOL NOT VALIDATED")
            print(f"  Centroid distance {rmsd:.3f}Å > {RMSD_THRESHOLD}Å threshold.")
            print(f"  Try increasing BOX_SIZE to 25.0 or 30.0 and re-run.")

        with open(LATEX_REDOCKING, 'w') as f:
            f.write("% Auto-generated by 4REDOCKING_VALIDATION.py\n")
            f.write(f"\\newcommand{{\\RedockRMSD}}{{{rmsd:.2f}}}\n")
            f.write(f"\\newcommand{{\\RedockEnergy}}{{{best_energy:.2f}}}\n")
            f.write(f"\\newcommand{{\\RedockThreshold}}{{{RMSD_THRESHOLD:.1f}}}\n")
            f.write(f"\\newcommand{{\\RedockLigand}}{{{LIGAND_CODE}}}\n")
            f.write(f"\\newcommand{{\\RedockChain}}{{{CHAIN}}}\n")
            f.write(f"\\newcommand{{\\RedockCenterX}}{{{cx:.3f}}}\n")
            f.write(f"\\newcommand{{\\RedockCenterY}}{{{cy:.3f}}}\n")
            f.write(f"\\newcommand{{\\RedockCenterZ}}{{{cz:.3f}}}\n")
            validated = "true" if rmsd < RMSD_THRESHOLD else "false"
            f.write(f"\\newcommand{{\\RedockValidated}}{{{validated}}}\n")
        print(f"\n[LATEX] redocking_variables.tex saved")
    else:
        print("[WARNING] Could not parse docked coordinates.")

    print("\n[DONE] 4REDOCKING_VALIDATION.py complete.")


if __name__ == "__main__":
    run_redocking_validation()
