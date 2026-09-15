import subprocess
import sys
import time

# Sequential list of scripts to execute.
# Note: Using the updated GNN version we created.
PIPELINE_SCRIPTS = [
    "scripts/0STACK.py",
    "scripts/01Wilcoxon.py",
    "scripts/2FDA.py",
    "scripts/3DOCKING.py",
    "scripts/3.5_ADMET.py",
    "scripts/4REDOCKING_VALIDATION.py",
    "scripts/5GNNBASELINE.py"
]

def main():
    print("=" * 70)
    print("🚀 STARTING Q1 VALIDATION PIPELINE")
    print("=" * 70)

    total_start = time.time()

    for script in PIPELINE_SCRIPTS:
        print(f"\n▶ Executing: {script}...")
        step_start = time.time()

        # Execute the script using the current Python interpreter
        result = subprocess.run([sys.executable, script])

        # Strict error control: if a script fails, abort everything.
        if result.returncode != 0:
            print(f"\n❌ [FATAL ERROR] Failure in {script} (Code {result.returncode}).")
            print("Halting pipeline execution to prevent cascading errors.")
            sys.exit(1)

        step_time = time.time() - step_start
        print(f"✅ {script} completed in {step_time:.1f} seconds.")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"🎉 PIPELINE SUCCESSFULLY COMPLETED IN {total_time:.1f} SECONDS.")
    print("CSV results and LaTeX variables are ready for the manuscript.")
    print("=" * 70)

if __name__ == "__main__":
    main()
