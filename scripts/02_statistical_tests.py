import os
import sys
import itertools
import logging
import pandas as pd
import numpy as np
from scipy import stats

# Link root directory to import paths_config and logger_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import CHECKPOINT_FILE, LATEX_DIR, LOGS_DIR, RESULTS_DIR
from logger_utils import setup_logger, explain
from nb_stats import nb_paired, holm, EQUIVALENCE_MARGIN_R2, EQUIVALENCE_MARGIN_MAE

NADEAU_BENGIO_LATEX_FILE = os.path.join(LATEX_DIR, "nadeau_bengio_variables.tex")

MODE_TAG = {"morgan": "Morgan", "rdkit2d": "TwoD", "rdkit2d_fp": "TwoDFp", "rdkit2d3d_fp": "TwoDThreeDFp"}


def all_pairs_nadeau_bengio(df):
    """Nadeau-Bengio corrected paired t-test and equivalence test (TOST) between every pair of representations, for R2
    and MAE. Folds are paired through the shared outer partitions; Holm adjustment is applied within each metric."""
    rows = []
    modes = [m for m in MODE_TAG if m in set(df["Mode"])]
    for metric, col, margin in (("RTwo", "R2_outer", EQUIVALENCE_MARGIN_R2), ("MAE", "MAE_outer", EQUIVALENCE_MARGIN_MAE)):
        block = []
        for a, b in itertools.combinations(modes, 2):
            da, db = df[df["Mode"] == a].sort_values("Fold"), df[df["Mode"] == b].sort_values("Fold")
            res = nb_paired(da[col].values - db[col].values, da["N_train"].mean(), da["N_test"].mean(), margin)
            block.append({"Metric": metric, "A": a, "B": b, "mean_diff_A_minus_B": res["mean_diff"], "t": res["t"],
                          "p": res["p"], "ci90_low": res["ci90_low"], "ci90_high": res["ci90_high"],
                          "equiv_margin": margin, "p_equiv": res["p_equiv"]})
        for row, p_adj in zip(block, holm([r["p"] for r in block])):
            row["p_holm"] = p_adj
        rows += block
    return pd.DataFrame(rows)


def main():
    # Initialize detailed logger
    log_file_path = os.path.join(LOGS_DIR, "02_statistical_tests.log")
    setup_logger(log_file_path)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("02_statistical_tests.py — Nadeau-Bengio Variance Correction Test")
    logger.info("=" * 80)
    explain("What this step does", """
        Step 01 produced one error value (MAE, in pIC50 units) for every test fold, for each way of describing the molecules
        (Morgan fingerprints vs. RDKit 2D + 3D descriptors + fingerprints). This step asks a single question: is the difference
        between those two descriptions real, or could it be just luck of how the compounds were split into folds?

        The test used is a paired t-test with the Nadeau-Bengio correction. A plain t-test would be over-confident here because
        the folds share most of their training compounds (they are not independent); the correction widens the uncertainty to
        compensate.

        How to read the result: p < 0.05 means the two descriptions truly differ in accuracy; p >= 0.05 means we cannot claim
        one is better than the other, and the simpler/cheaper one (Morgan) is then the reasonable choice.
    """)

    # Validate Checkpoint Data
    if not os.path.exists(CHECKPOINT_FILE):
        logger.error(f"Checkpoint file not found at: {CHECKPOINT_FILE}. Ensure 01_nested_cv_stacking.py completed.")
        return

    logger.info(f"Loading nested CV results from: {CHECKPOINT_FILE}")
    df = pd.read_csv(CHECKPOINT_FILE)
    logger.info(f"Total records loaded: {len(df)}. Available modes: {df['Mode'].unique().tolist()}")

    # Filter and sort by fold to guarantee exact pairing
    logger.info("Extracting performance metrics for 'morgan' and 'rdkit2d3d_fp' modes...")
    morgan = df[df['Mode'] == 'morgan'].sort_values('Fold')
    d3d = df[df['Mode'] == 'rdkit2d3d_fp'].sort_values('Fold')

    if len(morgan) == 0 or len(d3d) == 0:
        logger.error("Missing data for one or both required modes ('morgan', 'rdkit2d3d_fp'). Aborting test.")
        return

    # Structural Validation: Strict pairing by Fold ID
    if not np.array_equal(morgan['Fold'].values, d3d['Fold'].values):
        logger.error(
            "Fold IDs do not match exactly between 'morgan' and 'rdkit2d3d_fp'. Paired t-test requires strictly aligned distributions.")
        return

    logger.debug("Strict fold alignment validated successfully.")

    mae_morgan = morgan['MAE_outer'].values
    mae_d3d = d3d['MAE_outer'].values

    logger.info(f"Data points extracted: {len(mae_morgan)} outer folds per mode.")
    logger.info(
        f"Morgan MAE Distribution     -> Mean: {mae_morgan.mean():.4f} | Std: {mae_morgan.std():.4f} | Min: {mae_morgan.min():.4f} | Max: {mae_morgan.max():.4f}")
    logger.info(
        f"2D/3D/FP MAE Distribution -> Mean: {mae_d3d.mean():.4f} | Std: {mae_d3d.std():.4f} | Min: {mae_d3d.min():.4f} | Max: {mae_d3d.max():.4f}")

    # =========================================================
    # Nadeau-Bengio Corrected Paired t-test
    # =========================================================
    logger.info("Initializing Nadeau-Bengio variance correction calculations...")
    n_train_actual = morgan['N_train'].iloc[0]
    n_test_actual = morgan['N_test'].iloc[0]
    n_obs = len(mae_morgan)

    logger.debug(f"Test Parameters -> N_train: {n_train_actual}, N_test: {n_test_actual}, N_folds_total (obs): {n_obs}")

    diff = mae_morgan - mae_d3d
    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)

    logger.debug(f"Difference Array Statistics -> Mean diff: {mean_diff:.6f}, Variance of diff: {var_diff:.6f}")

    # Correction factor: 1/K*R + n_test/n_train
    variance_correction_term = (1.0 / n_obs) + (n_test_actual / n_train_actual)
    se_corrected = np.sqrt(var_diff * variance_correction_term)

    logger.debug(f"Correction Factor Term: {variance_correction_term:.6f}")
    logger.debug(f"Corrected Standard Error: {se_corrected:.6f}")

    if se_corrected > 0:
        t_corrected = mean_diff / se_corrected
        df_ttest = n_obs - 1
        p_corrected = 2 * stats.t.sf(np.abs(t_corrected), df=df_ttest)
    else:
        logger.warning(
            "Corrected standard error is exactly zero. This typically implies identical prediction outputs across all folds.")
        t_corrected = 0.0
        p_corrected = 1.0

    explain("How to read the numbers below", f"""
        Mean difference in MAE (Morgan minus 2D/3D/FP) = {mean_diff:+.4f} pIC50 units. A negative value means Morgan makes smaller
        errors on average. For scale: 0.3 pIC50 units is roughly a 2-fold change in potency (IC50).

        The t-statistic is the mean difference divided by its corrected uncertainty; |t| around 2 or more usually gives p < 0.05.
        Only {n_obs} folds are compared, so the test has limited power to detect small differences.
    """)
    logger.info("-" * 80)
    logger.info("Nadeau-Bengio Corrected Paired t-test Results:")
    logger.info(f"  t-statistic = {t_corrected:.3f}")
    logger.info(f"  p-value     = {p_corrected:.4f}")
    logger.info("-" * 80)

    if p_corrected >= 0.05:
        logger.info(
            "TEST CONCLUSION: No statistically significant difference between Morgan and 2D/3D/FP representations (p >= 0.05).")
    else:
        logger.info(
            "TEST CONCLUSION: Statistically significant difference observed between representations (p < 0.05).")

    pairwise = all_pairs_nadeau_bengio(df)
    pairwise_path = os.path.join(RESULTS_DIR, "02_pairwise_nb_tests.csv")
    pairwise.to_csv(pairwise_path, index=False)
    logger.info(f"All-pairs Nadeau-Bengio tests (Holm-adjusted within each metric) saved to: {pairwise_path}")
    logger.info("\n" + pairwise.round(4).to_string(index=False))

    # =========================================================
    # Export to LaTeX
    # =========================================================
    logger.info(f"Writing statistical variables to LaTeX file: {NADEAU_BENGIO_LATEX_FILE}")
    try:
        os.makedirs(os.path.dirname(NADEAU_BENGIO_LATEX_FILE), exist_ok=True)
        with open(NADEAU_BENGIO_LATEX_FILE, 'w', encoding='utf-8') as f:
            f.write("% =====================================================\n")
            f.write("% Auto-generated by 02_statistical_tests.py (NB Correction Only)\n")
            f.write("% Include in preamble with: \\input{latex/nadeau_bengio_variables.tex}\n")
            f.write("% =====================================================\n\n")
            f.write(f"\\newcommand{{\\CorrectedTStat}}{{{t_corrected:.3f}}}\n")
            f.write(f"\\newcommand{{\\CorrectedTP}}{{{p_corrected:.4f}}}\n")
            for _, r in pairwise.iterrows():
                tag = f"{r['Metric']}{MODE_TAG[r['A']]}Vs{MODE_TAG[r['B']]}"
                f.write(f"\\newcommand{{\\NBp{tag}}}{{{r['p']:.4f}}}\n")
                f.write(f"\\newcommand{{\\NBpHolm{tag}}}{{{r['p_holm']:.4f}}}\n")
                f.write(f"\\newcommand{{\\NBdiff{tag}}}{{{abs(r['mean_diff_A_minus_B']):.3f}}}\n")
                f.write(f"\\newcommand{{\\NBequivP{tag}}}{{{r['p_equiv']:.4f}}}\n")
                f.write(f"\\newcommand{{\\NBciLow{tag}}}{{{r['ci90_low']:+.3f}}}\n")
                f.write(f"\\newcommand{{\\NBciHigh{tag}}}{{{r['ci90_high']:+.3f}}}\n")
            f.write(f"\\newcommand{{\\EquivMarginRTwo}}{{{EQUIVALENCE_MARGIN_R2}}}\n")
        logger.info("LaTeX variable export completed successfully.")
    except Exception as e:
        logger.error(f"Failed to write LaTeX variables: {e}")


if __name__ == "__main__":
    main()