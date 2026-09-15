import subprocess
import sys
import time

# =========================================================
# PIPELINE EXECUTION ORDER (Q1 Standard Validated)
# =========================================================
# Note: Wilcoxon has been removed and replaced by Nadeau-Bengio.
# Y-Randomization is now correctly positioned AFTER baseline modeling.
PIPELINE_SCRIPTS = [
    "scripts/0STACK.py",               # 1. Nested CV and hyperparameter selection
    "scripts/0.5YRANDOM.py",           # 2. Negative Control: Y-Randomization
    "scripts/01NadeauBengio.py",       # 3. Corrected statistical test
    "scripts/1AUGMENT.py",             # 4. Data augmentation
    "scripts/5GNNBASELINE.py",         # 5. Graph Neural Networks (GNN) baseline
    "scripts/2FDA.py",                 # 6. FDA candidates screening
    "scripts/3.5_ADMET.py",            # 7. Pharmacokinetic filters (ADMET)
    "scripts/3DOCKING.py",             # 8. Molecular docking
    "scripts/4REDOCKING_VALIDATION.py" # 9. RMSD Validation (Redocking)
]


def main():
    print("=" * 70)
    print(" STARTING Q1 VALIDATION PIPELINE ")
    print("=" * 70)

    total_start = time.time()

    for script in PIPELINE_SCRIPTS:
        print(f"\n[INFO] Executing: {script}...")
        step_start = time.time()

        # Execute the script using the current Python interpreter
        result = subprocess.run([sys.executable, script])

        # Strict error control: if a script fails, abort everything.
        if result.returncode != 0:
            print(f"\n[FATAL ERROR] Failure in {script} (Code {result.returncode}).")
            print("Halting pipeline execution to prevent cascading errors.")
            sys.exit(1)

        step_time = time.time() - step_start
        print(f"[SUCCESS] {script} completed in {step_time:.1f} seconds.")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f" PIPELINE SUCCESSFULLY COMPLETED IN {total_time:.1f} SECONDS. ")
    print(" CSV results and LaTeX variables are ready for the manuscript. ")
    print("=" * 70)

if __name__ == "__main__":
    main()
