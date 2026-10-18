"""Nadeau-Bengio corrected paired comparisons for repeated cross-validation, shared by steps 02, 09, 12 and 13."""
import numpy as np
from scipy import stats

EQUIVALENCE_MARGIN_R2 = 0.05
EQUIVALENCE_MARGIN_MAE = 0.05


def nb_paired(diff, n_train, n_test, margin=None):
    """Corrected paired t-test on the per-fold differences `diff` (A minus B).

    The variance is inflated by (1/n + n_test/n_train) because folds of a repeated cross-validation share training data.
    With `margin`, also returns the two one-sided tests (TOST) for |mean difference| < margin: `p_equiv` is the larger of
    the two one-sided p-values, and [ci90_low, ci90_high] is the 90% interval, which lies inside (-margin, margin)
    exactly when equivalence is declared at the 5% level."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    mean = float(diff.mean())
    se = float(np.sqrt(diff.var(ddof=1) * (1.0 / n + n_test / n_train)))
    df = n - 1
    t = mean / se if se > 0 else 0.0
    out = {"mean_diff": mean, "se": se, "t": t, "p": 2 * stats.t.sf(abs(t), df=df) if se > 0 else 1.0}
    if margin is not None:
        t_crit = stats.t.ppf(0.95, df=df)
        out["ci90_low"], out["ci90_high"] = mean - t_crit * se, mean + t_crit * se
        if se > 0:
            p_lower = stats.t.sf((mean + margin) / se, df=df)
            p_upper = stats.t.cdf((mean - margin) / se, df=df)
            out["p_equiv"] = float(max(p_lower, p_upper))
        else:
            out["p_equiv"] = 0.0 if abs(mean) < margin else 1.0
    return out


def holm(p_values):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


CLASSICAL_MODE_LABEL = {'morgan': 'Morgan', 'rdkit2d': 'RdkitTwoD', 'rdkit2d_fp': 'RdkitTwoDFp',
                        'rdkit2d3d_fp': 'RdkitTwoDThreeDFp'}


def compare_to_classical(scores, checkpoint, prefix):
    """Compare a model evaluated fold by fold (`scores`: Fold, R2, MAE, N_train, N_test) against every classical
    representation of the step 01 `checkpoint` on the same folds: corrected t-test, Holm adjustment within each metric,
    and equivalence test. Folds are paired through their order and checked through the test-fold sizes.
    Returns (DataFrame of comparisons, dict of LaTeX macros named `{prefix}{Representation}{Quantity}{Metric}`)."""
    import pandas as pd
    rows = []
    for metric, s_col, c_col, margin in (("RTwo", "R2", "R2_outer", EQUIVALENCE_MARGIN_R2),
                                         ("Mae", "MAE", "MAE_outer", EQUIVALENCE_MARGIN_MAE)):
        block = []
        for mode, label in CLASSICAL_MODE_LABEL.items():
            c = checkpoint[checkpoint['Mode'] == mode].sort_values('Fold').reset_index(drop=True)
            if len(c) != len(scores) or not (c['N_test'].values == scores['N_test'].values).all():
                print(f"  [WARNING] Folds of '{mode}' do not match the evaluated model's folds; comparison skipped.", flush=True)
                continue
            res = nb_paired(scores[s_col].values - c[c_col].values, scores['N_train'].mean(), scores['N_test'].mean(), margin)
            block.append({"Metric": metric, "Classical": mode, "Label": label, "mean_diff_model_minus_classical": res["mean_diff"],
                          "t": res["t"], "p_raw": res["p"], "ci90_low": res["ci90_low"], "ci90_high": res["ci90_high"],
                          "equiv_margin": margin, "p_equiv": res["p_equiv"]})
        for row, p_adj in zip(block, holm([r["p_raw"] for r in block])):
            row["p_holm"] = p_adj
        rows += block
    macros = {}
    for r in rows:
        key = f"{prefix}{r['Label']}"
        macros[f"{key}Diff{r['Metric']}"] = f"{r['mean_diff_model_minus_classical']:+.4f}"
        macros[f"{key}P{r['Metric']}"] = f"{r['p_holm']:.3f}"
        macros[f"{key}EquivP{r['Metric']}"] = f"{r['p_equiv']:.3f}"
        macros[f"{key}CILow{r['Metric']}"] = f"{r['ci90_low']:+.3f}"
        macros[f"{key}CIHigh{r['Metric']}"] = f"{r['ci90_high']:+.3f}"
    for metric in ("RTwo", "Mae"):
        ps = [r['p_holm'] for r in rows if r['Metric'] == metric]
        if ps:
            macros[f"{prefix}MinP{metric}"] = f"{min(ps):.3f}"
    return pd.DataFrame(rows), macros
