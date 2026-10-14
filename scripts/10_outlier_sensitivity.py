"""
10_outlier_sensitivity.py — Do the two extreme pIC50 values change the conclusions?

Compares the nested-CV results of the main set (results/) with the run that kept the extreme values (sensitivity/results/),
and checks how well a model trained without them predicts them.
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import (CFG, DATA_DIR, LATEX_DIR, LOGS_DIR, MAIN_FILE, RANDOM_STATE, RESULTS_DIR, SENS_DIR,
                          SENS_FILE, mode_labels)
from logger_utils import setup_logger, explain

RDLogger.DisableLog("rdApp.*")
TARGET = "pIC50 Value"
SENS_CHECKPOINT = os.path.join(SENS_DIR, "results", "nested_cv_checkpoint.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "10_outlier_sensitivity.csv")
OUT_DIAG_CSV = os.path.join(RESULTS_DIR, "10_outlier_diagnostic.csv")
OUT_TEX = os.path.join(LATEX_DIR, "outlier_variables.tex")
OUT_TABLE_TEX = os.path.join(LATEX_DIR, "outlier_table.tex")


def morgan_matrix(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.uint8)


def compare_runs(main, sens):
    rows = []
    for mode in main["Mode"].unique():
        a, b = main[main.Mode == mode], sens[sens.Mode == mode]
        if b.empty:
            continue
        # Splits differ (different n), so the folds are not paired: Welch test with the Nadeau-Bengio variance inflation
        n_tr, n_te = a["N_train"].mean(), a["N_test"].mean()
        rows.append({"Mode": mode, "n_folds_main": len(a), "n_folds_sens": len(b)})
        for metric, col in (("R2", "R2_outer"), ("MAE", "MAE_outer")):
            nb = 1 / len(a) + n_te / n_tr
            se = np.sqrt(nb * a[col].var(ddof=1) + nb * b[col].var(ddof=1))
            t = (b[col].mean() - a[col].mean()) / se
            df = len(a) + len(b) - 2
            rows[-1].update({f"{metric}_main": a[col].mean(), f"{metric}_main_sd": a[col].std(),
                             f"{metric}_sens": b[col].mean(), f"{metric}_sens_sd": b[col].std(),
                             f"{metric}_delta": b[col].mean() - a[col].mean(),
                             f"{metric}_p_NB": 2 * stats.t.sf(abs(t), df)})
    return pd.DataFrame(rows)


def diagnose_extremes():
    """Out-of-fold error of a Morgan/RandomForest model on the main set vs. its error on the removed extremes."""
    main = pd.read_csv(MAIN_FILE)
    ext = pd.read_csv(os.path.join(DATA_DIR, "V2-combined_bd_chmbl_unique_outliers.csv"))
    X, y = morgan_matrix(main["Smiles"]), main[TARGET].values
    rf = RandomForestRegressor(n_estimators=CFG["N_ESTIMATORS_TREES"], random_state=RANDOM_STATE, n_jobs=CFG["N_JOBS"])
    oof = cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
    rmse, abs_err = np.sqrt(np.mean((oof - y) ** 2)), np.abs(oof - y)
    pred = rf.fit(X, y).predict(morgan_matrix(ext["Smiles"]))
    out = ext[["cid", TARGET, "fence_violated"]].copy()
    out["predicted"] = pred
    out["abs_error"] = np.abs(pred - ext[TARGET].values)
    out["error_over_oof_rmse"] = out["abs_error"] / rmse
    out["error_percentile_in_main"] = [(abs_err < e).mean() * 100 for e in out["abs_error"]]
    return out, rmse


def main():
    setup_logger(os.path.join(LOGS_DIR, "10_outlier_sensitivity.log"))
    explain("What this step does", """
        Two extreme potencies were removed from the main dataset (outside the Tukey fences of the pIC50 distribution). Here the
        nested-CV benchmark run WITH them (sensitivity/) is compared with the main run, representation by representation, and a
        Morgan/Random Forest model trained without them is asked to predict them. If the scores barely move and the model
        predicts the extremes badly, they are unrepresentative of the series (outliers) but harmless to the conclusions; if the
        scores change materially, the exclusion is a modelling choice that must be reported.
    """)
    if not os.path.exists(SENS_CHECKPOINT):
        sys.exit(f"Sensitivity checkpoint not found: {SENS_CHECKPOINT}. Run steps S01/S02 first.")

    cmp_df = compare_runs(pd.read_csv(os.path.join(RESULTS_DIR, "nested_cv_checkpoint.csv")), pd.read_csv(SENS_CHECKPOINT))
    cmp_df.to_csv(OUT_CSV, index=False)
    print(cmp_df.round(4).to_string(index=False), flush=True)
    rank_main = cmp_df.sort_values("R2_main", ascending=False)["Mode"].tolist()
    rank_sens = cmp_df.sort_values("R2_sens", ascending=False)["Mode"].tolist()
    print(f"[RANK] by mean R2 without extremes: {rank_main}\n[RANK] by mean R2 with extremes:    {rank_sens}", flush=True)

    diag, rmse = diagnose_extremes()
    diag.to_csv(OUT_DIAG_CSV, index=False)
    print(f"[DIAG] Morgan+RF 5-fold OOF RMSE on the main set: {rmse:.3f} pIC50", flush=True)
    print(diag.round(3).to_string(index=False), flush=True)

    m = cmp_df.set_index("Mode").loc["morgan"]
    raw_y = pd.read_csv(os.path.join(DATA_DIR, "V2-combined_bd_chmbl_unique_full.csv"))[TARGET]
    q1, q3 = raw_y.quantile([0.25, 0.75])
    macros = {
        "OutlierNMain": len(pd.read_csv(MAIN_FILE)), "OutlierNSens": len(pd.read_csv(SENS_FILE)),
        "OutlierNRemoved": len(diag),
        "OutlierFenceLow": f"{q1 - 1.5 * (q3 - q1):.2f}", "OutlierFenceHigh": f"{q3 + 1.5 * (q3 - q1):.2f}",
        "OutlierMorganRTwoMain": f"{m.R2_main:.3f}", "OutlierMorganRTwoSens": f"{m.R2_sens:.3f}",
        "OutlierMorganDeltaRTwo": f"{m.R2_delta:+.3f}", "OutlierMorganRTwoP": f"{m.R2_p_NB:.2f}",
        "OutlierMinDeltaRTwo": f"{cmp_df.R2_delta.min():+.3f}", "OutlierMaxDeltaRTwo": f"{cmp_df.R2_delta.max():+.3f}",
        "OutlierMinDeltaMAE": f"{cmp_df.MAE_delta.min():+.3f}", "OutlierMaxDeltaMAE": f"{cmp_df.MAE_delta.max():+.3f}",
        "OutlierMinP": f"{min(cmp_df.R2_p_NB.min(), cmp_df.MAE_p_NB.min()):.2f}",
        "OutlierMaxP": f"{max(cmp_df.R2_p_NB.max(), cmp_df.MAE_p_NB.max()):.2f}",
        "OutlierSameRanking": "yes" if rank_main == rank_sens else "no",
        "OutlierOofRMSE": f"{rmse:.2f}",
    }
    for tag, side in (("High", "high"), ("Low", "low")):
        r = diag[diag.fence_violated == side].iloc[0]
        macros.update({f"Outlier{tag}PIC": f"{r[TARGET]:.2f}", f"Outlier{tag}Pred": f"{r.predicted:.2f}",
                       f"Outlier{tag}Err": f"{r.abs_error:.2f}", f"Outlier{tag}Ratio": f"{r.error_over_oof_rmse:.1f}"})
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by 10_outlier_sensitivity.py\n")
        for name, value in macros.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    header = (r"\resizebox{\textwidth}{!}{%" "\n" r"\begin{tabular}{lcccccccc}" "\n" r"\toprule" "\n"
              r"& \multicolumn{4}{c}{\textbf{$R^2$}} & \multicolumn{4}{c}{\textbf{MAE}} \\" "\n"
              r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}" "\n"
              r"\textbf{Representation} & Without & With & $\Delta$ & $p$ & Without & With & $\Delta$ & $p$ \\" "\n"
              r"\midrule" "\n")
    with open(OUT_TABLE_TEX, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by 10_outlier_sensitivity.py\n" + header)
        for _, r in cmp_df.sort_values("R2_main", ascending=False).iterrows():
            f.write(f"{mode_labels.get(r.Mode, r.Mode)} & {r.R2_main:.3f} & {r.R2_sens:.3f} & {r.R2_delta:+.3f} & "
                    f"{r.R2_p_NB:.2f} & {r.MAE_main:.3f} & {r.MAE_sens:.3f} & {r.MAE_delta:+.3f} & {r.MAE_p_NB:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n")
    print(f"[LATEX] {OUT_TEX}", flush=True)


if __name__ == "__main__":
    main()
