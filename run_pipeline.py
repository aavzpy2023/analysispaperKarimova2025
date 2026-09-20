import os
import sys
import subprocess
from datetime import datetime

# Centralized logs directory
LOGS_DIR = os.path.abspath("./logs")
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
    "scripts/08_admet_profiling.py",
    "scripts/09_gnn_baseline.py"
]

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

def execute_and_log(script_path, master_log_path):
    script_filename = os.path.basename(script_path)
    log_filename = script_filename.replace(".py", ".log")
    step_log_path = os.path.join(LOGS_DIR, log_filename)

    # Clear individual log from previous executions
    with open(step_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== START LOG: {script_filename} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n\n")

    writer = MasterStreamWriter([step_log_path, master_log_path])

    # Capture unified stdout and stderr at system process level
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        for line in iter(process.stdout.readline, ''):
            writer.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, script_path)
    finally:
        writer.write(f"\n=== END LOG: {script_filename} [EXIT CODE: {process.returncode}] ===\n\n")
        writer.close()

def run_pipeline():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(LOGS_DIR, f"master_pipeline_{timestamp}.log")

    print("=" * 80)
    print("INITIATING Q1-STANDARDIZED COMPUTATIONAL PIPELINE")
    print(f"Master execution log: {master_log_path}")
    print("=" * 80)

    for script in PIPELINE:
        print(f"\n[>>>] Executing {script}...")
        try:
            execute_and_log(script, master_log_path)
            print(f"[OK] {script} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n[FATAL ERROR] {script} failed with exit code {e.returncode}.")
            print("Halting pipeline to prevent cascading data corruption.")
            sys.exit(1)

    print("\n[SUCCESS] Entire pipeline executed flawlessly.")

if __name__ == "__main__":
    run_pipeline()
