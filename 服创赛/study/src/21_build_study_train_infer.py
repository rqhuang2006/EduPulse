from __future__ import annotations

import numpy as np
import pandas as pd

from study_common import DM_DIR, DWD_DIR, collect_feature_columns, ensure_dirs, latest_rows, safe_merge, write_csv


TERM_FEATURE_FILES = [
    "study_l1_grade_term.parquet",
    "study_l2_course_load_term.parquet",
    "study_l3_attendance_term.parquet",
    "study_l4_class_task_term.parquet",
    "study_l5_assignment_term.parquet",
    "study_l6_exam_quiz_term.parquet",
    "study_l7_online_activity_term.parquet",
]


def add_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    feature_cols = collect_feature_columns(result)
    if feature_cols:
        result["FEATURE_MISSING_RATE"] = result[feature_cols].isna().mean(axis=1)
        groups = sorted({col.split("_")[1] for col in feature_cols if "_" in col})
        covered = []
        for _, row in result[feature_cols].iterrows():
            hit = 0
            for group in groups:
                cols = [c for c in feature_cols if c.startswith(f"FEATURE_{group}_")]
                if cols and row[cols].notna().any():
                    hit += 1
            covered.append(hit / len(groups) if groups else np.nan)
        result["SOURCE_COVERAGE"] = covered
    else:
        result["FEATURE_MISSING_RATE"] = np.nan
        result["SOURCE_COVERAGE"] = np.nan
    result["DATA_QUALITY_FLAG"] = np.where(
        (result["FEATURE_MISSING_RATE"].fillna(1) > 0.7) | (result["SOURCE_COVERAGE"].fillna(0) < 0.3),
        "LOW_COVERAGE",
        "OK",
    )
    result["DOMAIN"] = "study"
    return result


def main() -> None:
    ensure_dirs()
    base_path = DWD_DIR / "study_l0_student_term_base.parquet"
    if base_path.exists():
        table = pd.read_parquet(base_path)
    else:
        table = pd.DataFrame(columns=["XH", "TERM_ID"])

    for filename in TERM_FEATURE_FILES:
        path = DWD_DIR / filename
        if path.exists():
            table = safe_merge(table, pd.read_parquet(path))

    label_path = DWD_DIR / "study_label_table.parquet"
    if label_path.exists():
        label = pd.read_parquet(label_path)
        table = table.merge(label, on=["XH", "TERM_ID"], how="left")
    else:
        table["LABEL"] = pd.NA

    table = add_quality_columns(table).drop_duplicates(["XH", "TERM_ID"])
    train = table[table["LABEL"].notna()].copy()
    infer = table[table["LABEL"].isna()].copy()
    if infer.empty and not table.empty:
        infer = latest_rows(table).copy()
    train["LABEL"] = pd.to_numeric(train["LABEL"], errors="coerce").astype("Int64")

    label_audit_cols = [c for c in ["XH", "TERM_ID", "NEXT_TERM_ID", "LABEL", "LABEL_SUBTYPE", "LABEL_REASON"] if c in train.columns]
    if label_audit_cols:
        write_csv(train[label_audit_cols], DM_DIR / "study_label_audit.csv")

    write_csv(train.drop(columns=["NEXT_TERM_ID", "LABEL_REASON"], errors="ignore"), DM_DIR / "study_train_table.csv")
    write_csv(infer.drop(columns=["LABEL", "LABEL_SUBTYPE", "NEXT_TERM_ID", "LABEL_REASON"], errors="ignore"), DM_DIR / "study_infer_table.csv")
    print(f"train/infer written: train={len(train)} infer={len(infer)}")


if __name__ == "__main__":
    main()
