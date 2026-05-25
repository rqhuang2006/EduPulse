from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from study_common import DELIVERABLE_DIR, DM_DIR, ensure_dirs, write_json


REQUIRED_FILES = {
    "data": [
        "study_train_table.csv",
        "study_infer_table.csv",
        "study_prediction_output.csv",
        "study_explanation_output.csv",
    ],
    "docs": [
        "study_feature_dictionary.xlsx",
        "study_quality_report.xlsx",
    ],
    "model": [
        "study_model.pkl",
        "study_model_config.json",
        "study_model_metrics.json",
    ],
    ".": ["README.md"],
}


def bundle_dir() -> Path:
    packaged = DELIVERABLE_DIR / "study"
    return packaged if packaged.exists() else DM_DIR


def check_unique(df: pd.DataFrame, name: str) -> list[dict]:
    issues = []
    if {"XH", "TERM_ID"}.issubset(df.columns):
        dup = int(df.duplicated(["XH", "TERM_ID"]).sum())
        if dup:
            issues.append({"check": f"{name}_unique_xh_term", "status": "FAIL", "detail": f"duplicates={dup}"})
        else:
            issues.append({"check": f"{name}_unique_xh_term", "status": "PASS", "detail": ""})
    return issues


def main() -> None:
    ensure_dirs()
    root = bundle_dir()
    issues: list[dict] = []
    for folder, files in REQUIRED_FILES.items():
        for filename in files:
            path = root / filename if folder == "." else root / folder / filename
            issues.append(
                {
                    "check": f"required_file_{folder}/{filename}",
                    "status": "PASS" if path.exists() else "FAIL",
                    "detail": str(path),
                }
            )

    data_root = root / "data" if (root / "data").exists() else root
    docs_root = root / "docs" if (root / "docs").exists() else root
    train = pd.read_csv(data_root / "study_train_table.csv") if (data_root / "study_train_table.csv").exists() else pd.DataFrame()
    infer = pd.read_csv(data_root / "study_infer_table.csv") if (data_root / "study_infer_table.csv").exists() else pd.DataFrame()
    pred = pd.read_csv(data_root / "study_prediction_output.csv") if (data_root / "study_prediction_output.csv").exists() else pd.DataFrame()
    exp = pd.read_csv(data_root / "study_explanation_output.csv") if (data_root / "study_explanation_output.csv").exists() else pd.DataFrame()

    issues.extend(check_unique(train, "train"))
    issues.extend(check_unique(infer, "infer"))
    train_features = sorted([c for c in train.columns if c.startswith("FEATURE_")])
    infer_features = sorted([c for c in infer.columns if c.startswith("FEATURE_")])
    issues.append(
        {
            "check": "train_infer_feature_columns_match",
            "status": "PASS" if train_features == infer_features else "FAIL",
            "detail": f"train_only={sorted(set(train_features)-set(infer_features))}; infer_only={sorted(set(infer_features)-set(train_features))}",
        }
    )

    if not pred.empty and not exp.empty:
        pred_key = pred[["XH", "TERM_ID", "SOURCE_TABLE"]].astype(str).drop_duplicates()
        exp_key = exp[["XH", "TERM_ID", "SOURCE_TABLE"]].astype(str).drop_duplicates()
        aligned = len(pred_key.merge(exp_key, on=["XH", "TERM_ID", "SOURCE_TABLE"], how="inner")) == len(pred_key) == len(exp_key)
        issues.append({"check": "prediction_explanation_alignment", "status": "PASS" if aligned else "FAIL", "detail": ""})
        explanation_required = {"TOP_FEATURE_1", "TOP_FEATURE_1_VALUE", "TOP_FEATURE_2", "TOP_FEATURE_2_VALUE", "TOP_FEATURE_3", "TOP_FEATURE_3_VALUE", "EXPLANATION_TEXT"}
        missing_exp = sorted(explanation_required - set(exp.columns))
        issues.append({"check": "explanation_required_columns", "status": "PASS" if not missing_exp else "FAIL", "detail": ",".join(missing_exp)})

    dictionary_path = docs_root / "study_feature_dictionary.xlsx"
    if dictionary_path.exists():
        dictionary = pd.read_excel(dictionary_path, sheet_name="features")
        known = set(dictionary["FEATURE_NAME"])
        all_features = set(train_features) | set(infer_features)
        missing = sorted(all_features - known)
        issues.append({"check": "feature_dictionary_complete", "status": "PASS" if not missing else "FAIL", "detail": ",".join(missing)})

    report = {"bundle_dir": str(root), "status": "PASS" if all(i["status"] == "PASS" for i in issues) else "FAIL", "checks": issues}
    write_json(report, DM_DIR / "study_validation_report.json")
    if root != DM_DIR:
        write_json(report, root / "docs" / "study_validation_report.json")
    pd.DataFrame(issues).to_excel(DM_DIR / "study_validation_report.xlsx", index=False)
    if root != DM_DIR:
        pd.DataFrame(issues).to_excel(root / "docs" / "study_validation_report.xlsx", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
