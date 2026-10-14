"""
00_prepare_dataset.py — Builds the training sets from the raw curated table.

Input : data/V2-combined_bd_chmbl_unique_full.csv (every curated measurement, including extreme pIC50 values)
Output: *_dedup.csv               main analysis set (pIC50 outside the Tukey fences removed, duplicates collapsed)
        *_dedup_with_outliers.csv sensitivity set (same, but extreme pIC50 values kept; used by steps S01/S02/10)
        *_outliers.csv            the removed extreme values and the fence they violated
        *_duplicates.csv          every duplicated structure, with the row that was kept
Duplicates are rows with the same RDKit canonical (isomeric) SMILES; the first occurrence is kept.
"""
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import DATA_DIR, LOGS_DIR, RAW_FILE, MAIN_FILE, SENS_FILE
from logger_utils import setup_logger, explain

RDLogger.DisableLog("rdApp.*")

IQR_FACTOR = 1.5
TARGET = "pIC50 Value"
STEM = os.path.join(DATA_DIR, "V2-combined_bd_chmbl_unique")
OUTLIERS_CSV = STEM + "_outliers.csv"
DUPLICATES_CSV = STEM + "_duplicates.csv"


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def collapse_duplicates(df):
    """Returns (unique rows, duplicate report). The report holds every member of a duplicated group."""
    df = df.copy()
    df["canonical_smiles"] = df["Smiles"].map(canonical)
    if df["canonical_smiles"].isna().any():
        raise ValueError(f"{int(df['canonical_smiles'].isna().sum())} SMILES could not be parsed by RDKit")
    dup_mask = df.duplicated("canonical_smiles", keep=False)
    report = df[dup_mask].copy()
    report["group_id"] = report.groupby("canonical_smiles", sort=False).ngroup() + 1
    report["kept"] = ~report.duplicated("canonical_smiles", keep="first")
    report["kept"] = report["kept"].map({True: "KEPT", False: "REMOVED"})
    report = report[["group_id"] + [c for c in report.columns if c != "group_id"]]
    return df.drop_duplicates("canonical_smiles", keep="first").drop(columns="canonical_smiles"), report


def main():
    setup_logger(os.path.join(LOGS_DIR, "00_prepare_dataset.log"))
    explain("What this step does", """
        Builds the two training sets from the raw table. Extreme potencies are those outside Q1 - 1.5*IQR and Q3 + 1.5*IQR
        (Tukey fences) of the pIC50 distribution. They are removed for the main analysis; a second set keeps them so the
        pipeline can later measure how much they change the conclusions (sensitivity analysis, steps S01/S02/10).
        Then structures that appear more than once (same canonical SMILES, different identifiers) are collapsed to one row,
        because a duplicate in both the training and the test part of a fold would leak the answer.
    """)

    raw = pd.read_csv(RAW_FILE).dropna(subset=["Smiles", TARGET]).reset_index(drop=True)
    q1, q3 = raw[TARGET].quantile([0.25, 0.75])
    low, high = q1 - IQR_FACTOR * (q3 - q1), q3 + IQR_FACTOR * (q3 - q1)
    is_out = (raw[TARGET] < low) | (raw[TARGET] > high)
    print(f"[DATA] {len(raw)} raw rows | pIC50 Tukey fences [{low:.3f}, {high:.3f}] | {int(is_out.sum())} outside", flush=True)

    outliers = raw[is_out].assign(fence_violated=lambda d: d[TARGET].map(lambda v: "low" if v < low else "high"))
    outliers.to_csv(OUTLIERS_CSV, index=False)
    for _, r in outliers.iterrows():
        print(f"[OUTLIER] cid={r['cid']:.0f} pIC50={r[TARGET]:.3f} ({r['fence_violated']} fence)", flush=True)

    main_set, dup_report = collapse_duplicates(raw[~is_out])
    sens_set, dup_report_all = collapse_duplicates(raw)
    main_set.to_csv(MAIN_FILE, index=False)
    sens_set.to_csv(SENS_FILE, index=False)
    dup_report_all.to_csv(DUPLICATES_CSV, index=False)
    print(f"[DATA] main set: {len(main_set)} rows -> {os.path.basename(MAIN_FILE)}", flush=True)
    print(f"[DATA] sensitivity set (extremes kept): {len(sens_set)} rows -> {os.path.basename(SENS_FILE)}", flush=True)
    print(f"[DATA] {dup_report_all['group_id'].nunique()} duplicated structures "
          f"({int((dup_report_all['kept'] == 'REMOVED').sum())} rows removed) -> {os.path.basename(DUPLICATES_CSV)}", flush=True)
    if len(sens_set) - len(main_set) != len(outliers):
        print("[WARN] an extreme value is also a duplicate of another row; the two sets differ by a different count "
              "than the number of outliers", flush=True)


if __name__ == "__main__":
    main()
