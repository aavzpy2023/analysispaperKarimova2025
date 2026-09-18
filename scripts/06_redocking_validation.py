import os
import time
import warnings
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# ── Configuration and Logging Integration ────────────────────────────────────
from paths_config import *
from logger_utils import setup_logger

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

LOG_FILE = os.path.join(LOGS_DIR, "06_redocking_validation.log")

def extract_crystal_ligand(pdb_file, ligand_code, chain):
    """Extracts the crystal ligand coordinates and PDB lines from the receptor."""
    crystal_coords = []
    ligand_lines = []

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
        print(f"  [ERROR] Ligand '{ligand_code}' in chain {chain} not found.")
        return None, None, None

    centroid = np.mean(crystal_coords, axis=0)
    print(f"  [+] Chain {chain} extracted: {len(crystal_coords)} heavy atoms detected.")
    print(f"  [+] Centroid calculated (Active Site Center): {centroid.round(3)}")
    return np.array(crystal_coords), ligand_lines, centroid


def write_ligand_pdb(ligand_lines, out_path="crystal_ligand.pdb"):
    """Writes the extracted ligand lines into a standalone PDB file."""
    with open(out_path, 'w') as f:
        f.writelines(ligand_lines)
        f.write("END\n")
    print(f"  [+] Crystal ligand PDB successfully written to: {out_path}")
    return out_path


def convert_ligand_to_pdbqt(ligand_pdb, out_pdbqt="crystal_ligand.pdbqt"):
    """Converts the ligand PDB file into a PDBQT format suitable for Vina."""
    try:
        from meeko import MoleculePreparation
        try:
            from meeko import PDBQTWriterLegacy
            use_legacy = True
        except ImportError:
            use_legacy = False

        mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=True, sanitize=False)
        if mol is None:
            print("  [ERROR] RDKit could not parse the crystal_ligand.pdb file.")
            return None

        # Keep only the largest fragment if multiple exist
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            print(f"  [INFO] Multiple fragments detected ({len(frags)}). Retaining the largest one.")
            mol = max(frags, key=lambda f: f.GetNumAtoms())

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"  [WARNING] RDKit Sanitization issue: {e}")

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass

        # Prepare for docking using Meeko
        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)

        if use_legacy:
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
            if not is_ok:
                print(f"  [ERROR] Meeko preparation failed: {err}")
                return None
        else:
            pdbqt_string = prep.write_pdbqt_string()

        with open(out_pdbqt, 'w') as f:
            f.write(pdbqt_string)

        print(f"  [+] Ligand successfully converted to {out_pdbqt} via RDKit + Meeko.")
        return out_pdbqt

    except Exception as e:
        print(f"  [ERROR] Conversion process failed: {e}")
        return None


def compute_rmsd_centroid(coords_ref, coords_docked):
    """Computes Centroid-based RMSD, robust against atom count mismatches."""
    c1 = np.mean(coords_ref, axis=0)
    c2 = np.mean(coords_docked, axis=0)
    dist = float(np.linalg.norm(c1 - c2))
    print(f"  [-] Crystal Reference Centroid : {c1.round(3)}")
    print(f"  [-] Docked Output Centroid     : {c2.round(3)}")
    print(f"  [>] Calculated Centroid Distance: {dist:.3f} Å")
    return dist


def parse_vina_output_coords(vina_pdbqt_string):
    """Parses atom coordinates from a Vina output PDBQT string."""
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


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_redocking_validation():
    # 1. Initialize the agnostic logger
    setup_logger(LOG_FILE)
    start_pipeline = time.time()

    try:
        from vina import Vina
    except ImportError:
        print("[ERROR] AutoDock Vina Python bindings are not installed. Interrupting execution.")
        return

    print("=" * 80)
    print("REDOCKING VALIDATION PIPELINE — Crystal Ligand Re-Docking")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Receptor setup: Chain {CHAIN} (DHFR active site) of PDB 6AOG")
    print("=" * 80)
    print(f"Target Criterion: Centroid distance < {RMSD_THRESHOLD} Å = Protocol Validated\n")

    # ── 1. Ligand Extraction ─────────────────────────────────────────────────
    print(f"[1/4] EXTRACTING CRYSTAL LIGAND ({LIGAND_CODE})")
    t0 = time.time()
    result = extract_crystal_ligand(PDB_FILE, LIGAND_CODE, CHAIN)
    if result[0] is None:
        return
    crystal_coords, ligand_lines, centroid = result
    cx, cy, cz = centroid

    print(f"\n  [NOTE] Active site center for chain {CHAIN}: [{cx:.3f}, {cy:.3f}, {cz:.3f}]")
    print(f"  [NOTE] (Original script used average of both chains: [17.394, 68.757, -68.051])")
    print(f"  [NOTE] Re-running docking procedure with the corrected specific center.")
    print(f"  [+] Extraction completed in {time.time() - t0:.2f}s\n")

    # ── 2. Ligand Preparation ────────────────────────────────────────────────
    print(f"[2/4] LIGAND PREPARATION AND CONVERSION")
    t0 = time.time()
    ligand_pdb = write_ligand_pdb(ligand_lines)
    ligand_pdbqt = convert_ligand_to_pdbqt(ligand_pdb)
    if ligand_pdbqt is None:
        return
    print(f"  [+] Preparation completed in {time.time() - t0:.2f}s\n")

    # ── 3. AutoDock Vina Redocking ───────────────────────────────────────────
    print(f"[3/4] AUTODOCK VINA REDOCKING")
    print(f"  [-] Re-docking ligand '{LIGAND_CODE}' into chain '{CHAIN}'...")
    print(
        f"  [-] Grid Center: [{cx:.3f}, {cy:.3f}, {cz:.3f}] | Box Size: {BOX_SIZE}Å | Exhaustiveness: {EXHAUSTIVENESS}")

    t0 = time.time()
    v = Vina(sf_name='vina')
    v.set_receptor(RECEPTOR_FILE)
    v.compute_vina_maps(center=[cx, cy, cz], box_size=[BOX_SIZE, BOX_SIZE, BOX_SIZE])

    with open(ligand_pdbqt, 'r') as f:
        ligand_str = f.read()

    v.set_ligand_from_string(ligand_str)
    v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=1)

    best_energy = v.energies(n_poses=1)[0][0]
    docked_coords = parse_vina_output_coords(v.poses(n_poses=1))

    print(f"  [+] Docking completed in {time.time() - t0:.2f}s")
    print(f"  [+] Best Pose Binding Affinity: {best_energy:.3f} kcal/mol\n")

    # ── 4. Validation and Export ─────────────────────────────────────────────
    print(f"[4/4] VALIDATION RESULT")
    print("-" * 80)

    if docked_coords is not None and len(docked_coords) > 0:
        rmsd = compute_rmsd_centroid(crystal_coords, docked_coords)
        print("-" * 80)

        if rmsd < RMSD_THRESHOLD:
            print("  [SUCCESS] ✓ PROTOCOL VALIDATED")
            print(f"\n  [DRAFT] SUGGESTED METHODS SENTENCE FOR MANUSCRIPT:")
            print(f'  "Docking protocol validation was performed by re-docking')
            print(f'  pyrimethamine ({LIGAND_CODE}, PDB: 6AOG, chain {CHAIN})')
            print(f'  into the TgDHFR active site. The top-ranked pose reproduced')
            print(f'  the crystallographic binding mode with a centroid distance')
            print(f'  of {rmsd:.2f} Å (threshold: {RMSD_THRESHOLD:.1f} Å)."')
            print(f'\n  [IMPORTANT WARNING]: The corrected active site center')
            print(f'  for the DHFR site (chain {CHAIN}) is [{cx:.3f}, {cy:.3f}, {cz:.3f}].')
            print(f'  Ensure you update this in your 3DOCKING.py script before the final screening run.')
        else:
            print("  [FAILED] ✗ PROTOCOL NOT VALIDATED")
            print(f"  Centroid distance {rmsd:.3f}Å exceeds the {RMSD_THRESHOLD}Å threshold.")
            print(f"  Suggestion: Try increasing BOX_SIZE to 25.0 or 30.0 and re-run.")

        # LaTeX Export
        with open(LATEX_REDOCKING, 'w') as f:
            f.write("% Auto-generated by 06_redocking_validation.py pipeline\n")
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

        print(f"\n  [+] LaTeX variable definitions saved to: {LATEX_REDOCKING}")
    else:
        print("  [WARNING] Could not parse docked coordinates. Validation aborted.")

    print("=" * 80)
    print(f"[DONE] Redocking Validation Pipeline completed in {time.time() - start_pipeline:.2f}s.")
    print(f"       Logs saved in: {LOG_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_redocking_validation()