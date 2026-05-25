from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def _row_fingerprint(frame: pd.DataFrame, columns: list[str]) -> set[str]:
    if frame.empty or not columns:
        return set()
    safe = frame[columns].fillna("<NA>").astype(str)
    return {
        hashlib.sha1("|".join(row).encode("utf-8")).hexdigest()
        for row in safe.itertuples(index=False, name=None)
    }


def validate_split_integrity(
    split_frames: dict[str, pd.DataFrame],
    *,
    student_col: str = "student_id",
    row_key_columns: list[str] | None = None,
) -> dict[str, Any]:
    row_key_columns = list(row_key_columns or [student_col])
    student_sets: dict[str, set[str]] = {}
    row_sets: dict[str, set[str]] = {}
    unique_counts: dict[str, int] = {}

    for split_name, frame in split_frames.items():
        if frame.empty or student_col not in frame.columns:
            student_sets[split_name] = set()
            row_sets[split_name] = set()
            unique_counts[split_name] = 0
            continue
        normalized = frame.copy()
        normalized[student_col] = normalized[student_col].astype(str)
        student_sets[split_name] = set(normalized[student_col].dropna().tolist())
        unique_counts[split_name] = len(student_sets[split_name])
        available_row_columns = [column for column in row_key_columns if column in normalized.columns]
        row_sets[split_name] = _row_fingerprint(normalized, available_row_columns)

    split_names = list(split_frames)
    cross_student_overlap = 0
    row_overlap = 0
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            cross_student_overlap += len(student_sets[left].intersection(student_sets[right]))
            row_overlap += len(row_sets[left].intersection(row_sets[right]))

    return {
        "split_strategy": "student_id_isolated",
        "unique_students": unique_counts,
        "cross_split_student_overlap_count": cross_student_overlap,
        "row_overlap_count": row_overlap,
        "window_neighbor_leakage_detected": cross_student_overlap > 0 or row_overlap > 0,
        "conclusion": (
            "student-isolated split passed"
            if cross_student_overlap == 0 and row_overlap == 0
            else "split leakage detected"
        ),
    }
