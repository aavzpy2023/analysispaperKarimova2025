"""
run_pipeline.py — Runs the full computational pipeline (scripts 01 -> 09.1) and logs every step.

Usage (from any directory):
    python run_pipeline.py                    # run everything
    python run_pipeline.py --list             # show the steps
    python run_pipeline.py --check            # verify Python dependencies, run nothing
    python run_pipeline.py --from 05          # resume from step 05 onwards
    python run_pipeline.py --only 04 04.1     # run selected steps only
    python run_pipeline.py --strict           # error (instead of silent skip) if ChemProp cannot load
Environment variables (optional): MLGNN_PROFILE (laptop|workstation), MLGNN_N_JOBS.
"""
import argparse
import importlib.util
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

PIPELINE = [
    "scripts/01_nested_cv_stacking.py",
    "scripts/01.1_build_figures.py",
    "scripts/02_statistical_tests.py",
    "scripts/03_y_randomization.py",
    "scripts/04_augmentation_training.py",
    "scripts/04.1_build_figures.py",
    "scripts/05_virtual_screening.py",
    "scripts/05.1_build_figures.py",
    "scripts/06_redocking_validation.py",
    "scripts/07_molecular_docking.py",
    "scripts/07.1_build_figures.py",
    "scripts/08_admet_profiling.py",
    "scripts/08.1_build_figures.py",
    "scripts/09_gnn_baseline.py",
    "scripts/09.1_build_figures.py",
]

# Import names always checked by --check, in addition to EVERY third-party module that
# scripts/*.py actually import (found automatically with `ast`, nothing is executed).
# meeko + vina (Python bindings) are used by steps 06/07; torch/lightning/chemprop by step 09.
# No external binary is needed by the pipeline: `obabel` is only used by scripts/prep_receptor.py,
# which is NOT part of the pipeline (receptor/receptor.pdbqt is versioned in the repository).
REQUIRED_MODULES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib", "rdkit", "xgboost",
                    "lightgbm", "meeko", "vina", "torch", "lightning", "chemprop"]
# PyPI distribution names printed in the environment snapshot
SNAPSHOT_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "rdkit",
                     "meeko", "vina", "torch", "lightning", "chemprop", "PyTDC", "seaborn",
                     "adjustText", "matplotlib", "joblib", "setuptools"]


def step_id(script):
    """'scripts/04.1_build_figures.py' -> '04.1'"""
    return os.path.basename(script).split("_", 1)[0]


class MasterStreamWriter:
    """Writes in real-time to both console and log files (individual and master)."""
    def __init__(self, log_filepaths):
        self.terminal = sys.stdout
        self.files = [open(fp, "a", encoding="utf-8") for fp in log_filepaths]

    def write(self, message):
        self.terminal.write(message)
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        self.terminal.flush()
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            f.close()


def environment_snapshot():
    lines = [f"Python {platform.python_version()} | {platform.system()} {platform.release()} | "
             f"{os.cpu_count()} CPU threads | profile={os.environ.get('MLGNN_PROFILE', 'workstation')}"]
    for pkg in SNAPSHOT_PACKAGES:
        try:
            lines.append(f"  {pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"  {pkg}: not installed")
    return "\n".join(lines)


def try_import(module):
    """Really imports `module` in a subprocess. Returns None if OK, else the last error line.
    (find_spec is not enough: e.g. dgl is 'installed' but fails to import with an unsupported torch.)"""
    r = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    if r.returncode == 0:
        return None
    lines = (r.stderr or "").strip().splitlines()
    return lines[-1][:160] if lines else "import failed"


def scan_imports():
    """Third-party top-level modules imported anywhere in scripts/*.py -> {module: {script, ...}}.
    Parsed with `ast` (nothing is executed); standard library and project modules are excluded."""
    import ast
    scripts_dir = os.path.join(ROOT, "scripts")
    if not os.path.isdir(scripts_dir):
        return {}
    stdlib = set(sys.stdlib_module_names)
    local = {os.path.splitext(f)[0] for f in os.listdir(scripts_dir) if f.endswith(".py")}
    found = {}
    for fname in sorted(os.listdir(scripts_dir)):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(scripts_dir, fname), encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=fname)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            else:
                continue
            for m in mods:
                if m not in stdlib and m not in local:
                    found.setdefault(m, set()).add(fname)
    return found


def check_dependencies():
    print(environment_snapshot())
    used = scan_imports()
    to_check = sorted(set(REQUIRED_MODULES) | set(used))
    print(f"\nThird-party modules imported by scripts/*.py ({len(used)}): {', '.join(sorted(used))}")
    print(f"Checking that {len(to_check)} modules really import (this can take a minute)...")
    bad = {m: e for m in to_check if (e := try_import(m))}
    print("\nModules that FAIL to import:", "none" if not bad else "")
    for m, e in bad.items():
        users = ", ".join(sorted(used.get(m, []))) or "listed in REQUIRED_MODULES"
        print(f"  - {m}: {e}\n      used by: {users}")
    print("\nNote: cross-check the list above against requirements.txt (module names differ from PyPI names,"
          " e.g. sklearn -> scikit-learn, tdc -> PyTDC).")
    return not bad


def execute_and_log(script_path, master_log_path):
    script_filename = os.path.basename(script_path)
    step_log_path = os.path.join(LOGS_DIR, script_filename.replace(".py", ".log"))

    with open(step_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== START LOG: {script_filename} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n\n")

    writer = MasterStreamWriter([step_log_path, master_log_path])

    # MLGNN_LOG_MANAGED: tells logger_utils not to open the step log itself (this runner owns it)
    env = dict(os.environ, MLGNN_LOG_MANAGED="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

    process = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        cwd=ROOT, env=env,
    )
    try:
        for line in iter(process.stdout.readline, ''):
            writer.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, script_path)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    finally:
        writer.write(f"\n=== END LOG: {script_filename} [EXIT CODE: {process.returncode}] ===\n\n")
        writer.close()


def select_steps(args):
    steps = list(PIPELINE)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {step_id(s) for s in steps}
        if unknown:
            sys.exit(f"Unknown step(s): {sorted(unknown)}. Use --list.")
        return [s for s in steps if step_id(s) in wanted]
    if args.start:
        ids = [step_id(s) for s in steps]
        if args.start not in ids:
            sys.exit(f"Unknown step '{args.start}'. Use --list.")
        return steps[ids.index(args.start):]
    return steps


def run_pipeline(steps=None):
    steps = steps if steps is not None else list(PIPELINE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(LOGS_DIR, f"master_pipeline_{timestamp}.log")

    print("=" * 80)
    print("INITIATING Q1-STANDARDIZED COMPUTATIONAL PIPELINE")
    print(f"Master execution log: {master_log_path}")
    print("=" * 80)
    snapshot = environment_snapshot()
    print(snapshot)
    with open(master_log_path, "w", encoding="utf-8") as f:
        f.write(snapshot + "\n\n")

    timings = []
    t_start = time.time()
    for script in steps:
        print(f"\n[>>>] Executing {script}...")
        t0 = time.time()
        try:
            execute_and_log(script, master_log_path)
        except subprocess.CalledProcessError as e:
            print(f"\n[FATAL ERROR] {script} failed with exit code {e.returncode}.")
            print("Halting pipeline to prevent cascading data corruption.")
            print(f"Fix the problem and resume with:  python run_pipeline.py --from {step_id(script)}")
            sys.exit(1)
        timings.append((script, time.time() - t0))
        print(f"[OK] {script} completed successfully ({timings[-1][1]:.0f}s).")

    print("\nStep timings:")
    for script, secs in timings:
        print(f"  {os.path.basename(script):<40} {secs / 60:7.1f} min")
    print(f"  {'TOTAL':<40} {(time.time() - t_start) / 60:7.1f} min")
    print("\n[SUCCESS] Entire pipeline executed flawlessly.")


def main():
    ap = argparse.ArgumentParser(description="Run the ML vs GNN computational pipeline.")
    ap.add_argument("--list", action="store_true", help="list pipeline steps and exit")
    ap.add_argument("--check", action="store_true", help="check Python dependencies and exit")
    ap.add_argument("--from", dest="start", metavar="STEP", help="resume from this step (e.g. 05 or 04.1)")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps (e.g. 04 04.1)")
    ap.add_argument("--strict", action="store_true",
                    help="fail instead of silently skipping a model whose libraries cannot be imported (step 09)")
    args = ap.parse_args()

    if args.list:
        for s in PIPELINE:
            print(f"{step_id(s):>5}  {s}")
        return
    if args.check:
        sys.exit(0 if check_dependencies() else 1)
    if args.strict:
        os.environ["MLGNN_STRICT"] = "1"
    run_pipeline(select_steps(args))


if __name__ == "__main__":
    main()
