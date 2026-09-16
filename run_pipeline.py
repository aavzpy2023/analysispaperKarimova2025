import subprocess
import sys

# Define the validated sequential pipeline
PIPELINE = [
    "scripts/01_nested_cv_stacking.py",
    "scripts/02_statistical_tests.py",
    "scripts/03_y_randomization.py",
    "scripts/04_augmentation_training.py",
    "scripts/05_virtual_screening.py",
    "scripts/06_redocking_validation.py",
    "scripts/07_molecular_docking.py",
    "scripts/08_admet_profiling.py",
    "scripts/09_gnn_baseline.py"
]

def run_pipeline():
    print("=" * 80)
    print("INITIATING Q1-STANDARDIZED COMPUTATIONAL PIPELINE")
    print("=" * 80)

    for script in PIPELINE:
        print(f"\n[>>>] Executing {script}...")
        try:
            # Execute and pipe output to stdout
            result = subprocess.run([sys.executable, script], check=True)
            print(f"[OK] {script} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[FATAL ERROR] {script} failed with exit code {e.returncode}.")
            print("Halting pipeline to prevent cascading data corruption.")
            sys.exit(1)

    print("\n[SUCCESS] Entire pipeline executed flawlessly.")

if __name__ == "__main__":
    run_pipeline()
