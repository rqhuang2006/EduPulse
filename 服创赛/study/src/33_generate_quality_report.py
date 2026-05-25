from __future__ import annotations

import json

import numpy as np
import pandas as pd

from study_common import DM_DIR, ensure_dirs


def write_workbook_safely(sheets: dict[str, pd.DataFrame], output) -> None:
    temp = output.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(temp) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    try:
        temp.replace(output)
    except PermissionError:
        try:
            has_output = output.exists()
        except PermissionError:
            has_output = True
        if has_output:
            try:
                temp.unlink(missing_ok=True)
            except PermissionError:
                pass
            print(f"warning: {output} is locked; keeping existing file")
        else:
            fallback = output.with_suffix(".generated.xlsx")
            try:
                temp.replace(fallback)
            except PermissionError:
                pass
            print(f"warning: cannot write {output}; kept fallback workbook at {fallback}")


def read_csv_or_empty(name: str) -> pd.DataFrame:
    path = DM_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    ensure_dirs()
    train = read_csv_or_empty("study_train_table.csv")
    infer = read_csv_or_empty("study_infer_table.csv")
    prediction = read_csv_or_empty("study_prediction_output.csv")
    metrics_path = DM_DIR / "study_model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    feature_cols = sorted([c for c in pd.concat([train, infer], ignore_index=True, sort=False).columns if c.startswith("FEATURE_")])
    all_rows = pd.concat([train.assign(TABLE="train"), infer.assign(TABLE="infer")], ignore_index=True, sort=False)
    valid_train = train[train.get("LABEL", pd.Series(dtype=float)).notna()] if "LABEL" in train.columns else train
    key_success_rate = 1 - float(all_rows[["XH", "TERM_ID"]].isna().any(axis=1).mean()) if {"XH", "TERM_ID"}.issubset(all_rows.columns) and len(all_rows) else np.nan
    text_values = all_rows.astype("string") if len(all_rows) else pd.DataFrame()
    unknown_rate = float((text_values.eq("unknown") | text_values.eq("UNKNOWN") | text_values.eq("未分类")).any(axis=1).mean()) if len(all_rows) else np.nan
    model_feature_count = np.nan
    config_path = DM_DIR / "study_model_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_feature_count = len(config.get("feature_columns", []))

    overview = pd.DataFrame(
        [
            {"METRIC": "train_rows", "VALUE": len(train)},
            {"METRIC": "valid_train_rows", "VALUE": len(valid_train)},
            {"METRIC": "infer_rows", "VALUE": len(infer)},
            {"METRIC": "prediction_rows", "VALUE": len(prediction)},
            {"METRIC": "feature_count", "VALUE": len(feature_cols)},
            {"METRIC": "model_feature_count", "VALUE": model_feature_count},
            {"METRIC": "key_mapping_success_rate", "VALUE": key_success_rate},
            {"METRIC": "unknown_or_unclassified_row_rate", "VALUE": unknown_rate},
        ]
    )
    sample_stats = all_rows.groupby("TABLE", as_index=False).agg(ROWS=("XH", "count"), STUDENTS=("XH", "nunique"), TERMS=("TERM_ID", "nunique")) if not all_rows.empty else pd.DataFrame()
    source_coverage = all_rows.groupby("TABLE", as_index=False).agg(
        SOURCE_COVERAGE_MEAN=("SOURCE_COVERAGE", "mean"),
        FEATURE_MISSING_RATE_MEAN=("FEATURE_MISSING_RATE", "mean"),
    ) if {"SOURCE_COVERAGE", "FEATURE_MISSING_RATE", "TABLE"}.issubset(all_rows.columns) else pd.DataFrame()
    feature_missing = pd.DataFrame(
        [{"FEATURE_NAME": col, "MISSING_RATE": float(all_rows[col].isna().mean())} for col in feature_cols]
    )
    label_distribution = train["LABEL"].value_counts(dropna=False).rename_axis("LABEL").reset_index(name="COUNT") if "LABEL" in train.columns else pd.DataFrame()
    model_metrics = pd.json_normalize(metrics, sep=".") if metrics else pd.DataFrame()
    known_issues = pd.DataFrame(
        [
            {"ISSUE": "LOW_COVERAGE rows", "COUNT": int((all_rows.get("DATA_QUALITY_FLAG", pd.Series(dtype=str)) == "LOW_COVERAGE").sum())},
            {"ISSUE": "Rows without prediction", "COUNT": max(len(all_rows) - len(prediction), 0)},
        ]
    )
    raw_source_coverage = pd.DataFrame(
        [
            {"ITEM": "source_coverage_definition", "VALUE": "SOURCE_COVERAGE is row-level share of populated feature subject groups, not raw-file ingestion success."},
            {"ITEM": "key_mapping_success_rate", "VALUE": key_success_rate},
            {"ITEM": "feature_subject_group_count", "VALUE": len(sorted({c.split('_')[1] for c in feature_cols if '_' in c}))},
        ]
    )
    recommendations = pd.DataFrame(
        [
            {"PRIORITY": 1, "RECOMMENDATION": "Keep NEXT_TERM_ID and LABEL_REASON in label audit only, not in train table."},
            {"PRIORITY": 2, "RECOMMENDATION": "Do not use FEATURE_MISSING_RATE, SOURCE_COVERAGE, or DATA_QUALITY_FLAG as model features."},
            {"PRIORITY": 3, "RECOMMENDATION": "Improve L4/L5/L6 source coverage before business rollout."},
            {"PRIORITY": 4, "RECOMMENDATION": "Review degraded predictions separately because they indicate low source coverage or fallback scoring."},
        ]
    )

    output = DM_DIR / "study_quality_report.xlsx"
    write_workbook_safely(
        {
            "overview": overview,
            "sample_stats": sample_stats,
            "source_coverage": source_coverage,
            "feature_missing": feature_missing,
            "label_distribution": label_distribution,
            "model_metrics": model_metrics,
            "known_issues": known_issues,
            "raw_source_coverage": raw_source_coverage,
            "recommendations": recommendations,
        },
        output,
    )
    print(f"quality report written: {output}")


if __name__ == "__main__":
    main()
