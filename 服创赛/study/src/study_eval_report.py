from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from study_common import write_json


def safe_auc(y_true: pd.Series, score: pd.Series) -> float | None:
    if len(y_true) == 0 or y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, score))


def classification_summary(y_true: pd.Series, score: pd.Series, threshold: float = 0.5) -> dict[str, Any]:
    pred = (score >= threshold).astype(int)
    return {
        "rows": int(len(y_true)),
        "auc": safe_auc(y_true, score),
        "f1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else None,
        "recall": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else None,
        "precision": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else None,
    }


def build_eval_reports(frame: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(frame["LABEL"], errors="coerce").fillna(0).astype(int)
    final_score = pd.to_numeric(frame["FINAL_SCORE"], errors="coerce").fillna(0.5)
    overall = classification_summary(y, final_score)
    overall["coverage"] = float(final_score.notna().mean())
    overall["degraded_ratio"] = float((frame.get("STUDY_DATA_MODE", pd.Series("core_only", index=frame.index)).astype(str) == "degraded_sparse").mean())

    mode_rows: list[dict[str, Any]] = []
    for mode in ["core_only", "core_plus_behavior", "degraded_sparse"]:
        mask = frame.get("STUDY_DATA_MODE", pd.Series("", index=frame.index)).astype(str) == mode
        if mask.any():
            row = {"study_data_mode": mode, **classification_summary(y.loc[mask], final_score.loc[mask])}
            mode_rows.append(row)

    subtype_rows: list[dict[str, Any]] = []
    subtype_series = frame.get("LABEL_SUBTYPE", pd.Series("unknown", index=frame.index)).astype(str)
    normal_mask = subtype_series == "normal"
    for subtype in ["single_fail", "overall_low"]:
        subtype_mask = subtype_series == subtype
        compare_mask = normal_mask | subtype_mask
        if compare_mask.any():
            subtype_target = subtype_mask.loc[compare_mask].astype(int)
            subtype_score = final_score.loc[compare_mask]
            row = {
                "label_subtype": subtype,
                "rows": int(compare_mask.sum()),
                "auc": safe_auc(subtype_target, subtype_score),
                "positive_rows": int(subtype_mask.sum()),
            }
            subtype_rows.append(row)

    zone_rows: list[dict[str, Any]] = []
    zone_series = frame.get("CONFIDENCE_ZONE", pd.Series("unknown", index=frame.index)).astype(str)
    for zone in ["high_conf_positive", "middle_correction", "high_conf_negative"]:
        mask = zone_series == zone
        if mask.any():
            row = {"confidence_zone": zone, **classification_summary(y.loc[mask], final_score.loc[mask])}
            zone_rows.append(row)

    return {
        "overall_metrics": overall,
        "mode_metrics": mode_rows,
        "subtype_metrics": subtype_rows,
        "confidence_zone_metrics": zone_rows,
    }


def write_eval_outputs(report: dict[str, Any], eval_report_path: Path, subgroup_metrics_path: Path, confidence_zone_path: Path) -> None:
    write_json(report, eval_report_path)
    pd.DataFrame(report.get("subtype_metrics", [])).to_csv(subgroup_metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(report.get("confidence_zone_metrics", [])).to_csv(confidence_zone_path, index=False, encoding="utf-8-sig")
