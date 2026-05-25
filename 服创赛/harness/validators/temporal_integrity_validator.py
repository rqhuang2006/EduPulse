from __future__ import annotations

from typing import Any


def validate_temporal_integrity(
    feature_specs: list[dict[str, Any]],
    *,
    label_window_start: str,
    label_window_end: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in feature_specs:
        feature_window_end = spec.get("feature_window_end")
        uses_post_label_data = bool(feature_window_end and feature_window_end > label_window_start)
        leakage_risk_level = "high" if uses_post_label_data else spec.get("leakage_risk_level", "low")
        row = {
            "feature_name": spec.get("feature_name", ""),
            "source_dataset": spec.get("source_dataset", ""),
            "feature_window_start": spec.get("feature_window_start", ""),
            "feature_window_end": feature_window_end or "",
            "label_window_start": label_window_start,
            "label_window_end": label_window_end,
            "uses_post_label_data": uses_post_label_data,
            "leakage_risk_level": leakage_risk_level,
            "note": spec.get("note", ""),
        }
        if uses_post_label_data and "repaired" not in row["note"]:
            row["note"] = (row["note"] + "; " if row["note"] else "") + "uses post-label data and must be fixed"
        rows.append(row)

    violating = [row for row in rows if row["uses_post_label_data"]]
    return {
        "ok": not violating,
        "rows": rows,
        "violation_count": len(violating),
        "conclusion": "no explicit post-label feature leakage detected" if not violating else "post-label leakage detected",
    }
