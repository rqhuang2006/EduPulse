"""
Study Feature Engineering - Shared Module
==========================================
UNIFIED feature engineering functions used by BOTH training and inference.

This module eliminates the train/infer dual-path problem by providing
a single source of truth for:
- Temporal feature generation
- Interaction feature generation
- Targeted feature generation
- Feature layer inference

Usage:
    from study_feature_engine import (
        add_temporal_features,
        add_interaction_features,
        build_targeted_features,
        infer_feature_layer,
    )
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# Feature Layer Classification (SHARED)
# ============================================================================
def infer_feature_layer(col_name: str) -> str:
    """Classify a feature column into its layer: core/behavior/temporal/interaction."""
    c = str(col_name).lower()

    # Interaction markers (check first - highest priority)
    interaction_markers = ["__x__", "_x_", "cross_", "inter_", "discordance_", "workload_stress"]
    if any(marker in c for marker in interaction_markers):
        return "interaction"

    # Temporal markers
    temporal_markers = [
        "recent_", "prev_", "hist_", "delta_", "ratio_", "trend_",
        "slope_", "volatility_", "stability_", "rolling_", "window_",
        "chg_", "change_", "consecutive_decline_", "dist_from_worst_",
        "recovery_", "consecutive_",
    ]
    if any(marker in c for marker in temporal_markers):
        return "temporal"

    # Course risk markers (temporal-like since they involve critical thresholds)
    if c.startswith("course_risk_"):
        return "temporal"

    # Behavior markers
    behavior_markers = [
        "attendance", "library", "online", "assignment", "exam",
        "class_task", "video", "activity", "borrow", "night", "sleep",
        "consume", "consumption", "dorm", "internet", "schedule",
        "regularity",
    ]
    if any(marker in c for marker in behavior_markers):
        return "behavior"

    return "core"


def summarize_feature_layers(feature_cols: list[str]) -> dict[str, Any]:
    """Summarize feature counts and columns by layer."""
    summary = {"core": [], "behavior": [], "temporal": [], "interaction": []}
    for col in feature_cols:
        summary[infer_feature_layer(col)].append(str(col))

    counts = {key: len(value) for key, value in summary.items()}
    if counts["behavior"] > 0 or counts["temporal"] > 0 or counts["interaction"] > 0:
        if counts["behavior"] > 0 and (counts["temporal"] > 0 or counts["interaction"] > 0):
            mode = "core_plus_behavior_enhanced"
        elif counts["behavior"] > 0:
            mode = "core_plus_behavior"
        else:
            mode = "core_enhanced"
    else:
        mode = "core_only"

    return {"counts": counts, "columns": summary, "study_data_mode": mode}


# ============================================================================
# Temporal Features (SHARED)
# ============================================================================
def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for grade/course/behavior metrics.

    Generates per-student temporal features:
    - prev_*: previous term value
    - hist_*: historical mean (excluding current)
    - delta_*: change from previous
    - ratio_*: ratio to previous
    - delta_hist_*, ratio_hist_*: change/ratio to historical mean
    - consecutive_decline_*: count of consecutive declining terms
    - dist_from_worst_*: normalized position relative to student's range
    - recovery_*: flag for bounce-back after decline
    """
    frame = data.copy()
    if "XH" not in frame.columns or "TERM_ID" not in frame.columns:
        return frame

    candidate_features = [
        "FEATURE_GRADE_AVG_SCORE",
        "FEATURE_GRADE_FAIL_COUNT",
        "FEATURE_ATTENDANCE_ABNORMAL_RATE",
        "FEATURE_ASSIGNMENT_SUBMIT_RATE",
        "FEATURE_EXAM_SCORE_AVG",
        "FEATURE_LIBRARY_VISIT_COUNT",
    ]
    candidate_features = [column for column in candidate_features if column in frame.columns]
    if not candidate_features:
        return frame

    ordered = frame.copy()
    ordered["_TERM_SORT_KEY"] = _term_sort_key(ordered["TERM_ID"])
    ordered["_ROW_ID"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["XH", "_TERM_SORT_KEY", "TERM_ID", "_ROW_ID"]).reset_index(drop=True)

    for column in candidate_features:
        numeric = pd.to_numeric(ordered[column], errors="coerce")
        base = column.lower().replace("feature_", "")

        # Previous term value
        prev = numeric.groupby(ordered["XH"]).shift(1)
        ordered[f"prev_{base}"] = prev

        # Historical mean (excluding current and previous)
        hist = numeric.groupby(ordered["XH"]).transform(lambda s: s.shift(1).expanding().mean())
        ordered[f"hist_{base}"] = hist

        # Delta and ratio features
        ordered[f"delta_{base}"] = numeric - prev
        ordered[f"ratio_{base}"] = numeric / (prev.abs() + 1e-6)
        ordered[f"delta_hist_{base}"] = numeric - hist
        ordered[f"ratio_hist_{base}"] = numeric / (hist.abs() + 1e-6)

        # Consecutive decline count
        declined = (numeric < prev).astype(float)
        consecutive = declined.groupby(ordered["XH"]).cumsum()
        prev_consecutive = consecutive.groupby(ordered["XH"]).shift(1).fillna(0)
        ordered[f"consecutive_decline_{base}"] = prev_consecutive.astype(int)

        # Distance from worst (normalized position in student's range)
        cummin_shifted = numeric.groupby(ordered["XH"]).cummin().groupby(ordered["XH"]).shift(1)
        cummax_shifted = numeric.groupby(ordered["XH"]).cummax().groupby(ordered["XH"]).shift(1)
        range_val = cummax_shifted - cummin_shifted
        ordered[f"dist_from_worst_{base}"] = ((numeric - cummin_shifted) / (range_val + 1e-6)).fillna(0.5)

        # Recovery flag (bounced back after decline)
        ordered[f"recovery_{base}"] = ((numeric > prev) & (prev < hist)).fillna(False).astype(int)

    ordered = ordered.sort_values("_ROW_ID").drop(columns=["_TERM_SORT_KEY", "_ROW_ID"])
    return ordered


def _term_sort_key(term_id: pd.Series) -> pd.Series:
    """Create a sortable key for term IDs (e.g., '2023-1' -> 20231, '2023-2' -> 20232)."""
    return term_id.astype(str).str.replace("-", "").astype(int)


# ============================================================================
# Interaction Features (SHARED)
# ============================================================================
def add_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction/cross features between pairs of metrics.

    Generates:
    - cross__{col_a}__x__{col_b}: product of two features
    - discordance_attendance_grade: high attendance but low grades
    - discordance_effort_result: high assignment submission but low exam scores
    - workload_stress: many courses with low scores
    """
    frame = data.copy()

    # Standard cross-product pairs
    candidate_pairs = [
        ("FEATURE_GRADE_AVG_SCORE", "FEATURE_COURSE_SELECTED_COUNT"),
        ("FEATURE_GRADE_FAIL_COUNT", "FEATURE_COURSE_RETAKE_COUNT"),
        ("FEATURE_ATTENDANCE_ABNORMAL_RATE", "FEATURE_ASSIGNMENT_SUBMIT_RATE"),
        ("FEATURE_ASSIGNMENT_SCORE_AVG", "FEATURE_EXAM_SCORE_AVG"),
        ("FEATURE_LIBRARY_VISIT_COUNT", "FEATURE_GRADE_AVG_SCORE"),
        ("delta_grade_avg_score", "FEATURE_COURSE_SELECTED_COUNT"),
        ("delta_exam_score_avg", "FEATURE_ASSIGNMENT_SUBMIT_RATE"),
        ("ratio_assignment_submit_rate", "FEATURE_ATTENDANCE_ABNORMAL_RATE"),
    ]

    for col_a, col_b in candidate_pairs:
        if col_a in frame.columns and col_b in frame.columns:
            new_col = f"cross__{col_a.lower()}__x__{col_b.lower()}"
            frame[new_col] = (
                pd.to_numeric(frame[col_a], errors="coerce").fillna(0.0)
                * pd.to_numeric(frame[col_b], errors="coerce").fillna(0.0)
            )

    # Discordance features (semantically meaningful)
    def num_safe(name: str) -> pd.Series | None:
        return pd.to_numeric(frame[name], errors="coerce") if name in frame.columns else None

    # Discordance: high attendance but low grades
    a_rate = num_safe("FEATURE_ATTENDANCE_ABNORMAL_RATE")
    g_avg = num_safe("FEATURE_GRADE_AVG_SCORE")
    if a_rate is not None and g_avg is not None:
        attendance_good = 1.0 - a_rate.fillna(0.5)
        grade_bad = 1.0 - g_avg.fillna(70) / 100.0
        frame["discordance_attendance_grade"] = attendance_good * grade_bad

    # Discordance: high assignment submission but low exam scores
    a_submit = num_safe("FEATURE_ASSIGNMENT_SUBMIT_RATE")
    e_avg = num_safe("FEATURE_EXAM_SCORE_AVG")
    if a_submit is not None and e_avg is not None:
        effort = a_submit.fillna(0.5)
        exam_bad = 1.0 - e_avg.fillna(70) / 100.0
        frame["discordance_effort_result"] = effort * exam_bad

    # Workload stress: many courses with low scores
    c_count = num_safe("FEATURE_COURSE_SELECTED_COUNT")
    if c_count is not None and g_avg is not None:
        grade_pressure = 1.0 - g_avg.fillna(70) / 100.0
        frame["workload_stress"] = c_count.fillna(0) * grade_pressure

    return frame


# ============================================================================
# Targeted Features (SHARED)
# ============================================================================
def build_targeted_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build targeted features for subgroup analysis.

    Generates:
    - trend_decline_count_*: count of declining trends
    - trend_below_hist_*: flag for below historical mean
    - trend_drawdown_*: drawdown from peak
    - personal_gap_*: gap from personal historical mean
    - personal_ratio_*: ratio to personal historical mean
    - personal_z_*: z-score relative to personal history
    - feature_behavior_coverage_count: number of behavior families with data
    - feature_temporal_coverage_count: number of temporal features present
    - feature_behavior_missing_ratio: ratio of missing behavior features
    - imbalance_*: discordance between different feature types
    """
    frame = data.copy()

    def num(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce")

    def z_by_student(series: pd.Series) -> pd.Series:
        """
        Compute z-score using ONLY historical/past data (no leakage).
        
        FIXED: Uses shift(1) to ensure statistics are computed from past terms only.
        This prevents valid/test data from leaking into training statistics.
        """
        if "XH" not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        
        # Sort by student and term to ensure temporal order
        if "TERM_ID" in frame.columns:
            from study_feature_engine import _term_sort_key
            ordered = frame.copy()
            ordered["_TERM_SORT_KEY"] = _term_sort_key(ordered["TERM_ID"])
            ordered = ordered.sort_values(["XH", "_TERM_SORT_KEY"])
            
            # Compute expanding mean/std from PAST terms only
            numeric = pd.to_numeric(series, errors="coerce")
            grouped_mean = numeric.groupby(ordered["XH"]).transform(lambda s: s.shift(1).expanding().mean())
            grouped_std = numeric.groupby(ordered["XH"]).transform(lambda s: s.shift(1).expanding().std())
            
            result = (numeric - grouped_mean) / (grouped_std + 1e-6)
            return result.reindex(frame.index)
        else:
            # Fallback: if no TERM_ID, return NaN to avoid leakage
            return pd.Series(np.nan, index=frame.index, dtype="float64")

    # Base triplets for trend analysis
    base_triplets = [
        ("FEATURE_GRADE_AVG_SCORE", "prev_grade_avg_score", "hist_grade_avg_score", "grade_avg"),
        ("FEATURE_EXAM_SCORE_AVG", "prev_exam_score_avg", "hist_exam_score_avg", "exam_score"),
        ("FEATURE_LIBRARY_VISIT_COUNT", "prev_library_visit_count", "hist_library_visit_count", "library_visit"),
    ]

    for current_col, prev_col, hist_col, alias in base_triplets:
        if current_col in frame.columns:
            current = num(current_col)
            prev = num(prev_col)
            hist = num(hist_col)
            peak = current.groupby(frame["XH"]).cummax().shift(1)

            frame[f"trend_decline_count_{alias}"] = (
                current.lt(prev).fillna(False).astype(int) + prev.lt(hist).fillna(False).astype(int)
            )
            frame[f"trend_below_hist_{alias}"] = current.lt(hist).fillna(False).astype(int)
            frame[f"trend_drawdown_{alias}"] = (peak - current) / (peak.abs() + 1e-6)
            frame[f"personal_gap_{alias}"] = current - hist
            frame[f"personal_ratio_{alias}"] = current / (hist.abs() + 1e-6)
            frame[f"personal_z_{alias}"] = z_by_student(current)

    # Coverage features
    behavior_cols = [c for c in frame.columns if c.startswith("FEATURE_") and any(
        k in c for k in ["ATTENDANCE", "CLASS_", "ASSIGNMENT", "EXAM", "LIBRARY"]
    )]
    temporal_cols = [c for c in frame.columns if c.startswith(("prev_", "hist_", "delta_", "ratio_"))]

    frame["feature_behavior_coverage_count"] = frame[behavior_cols].notna().sum(axis=1) if behavior_cols else 0
    frame["feature_temporal_coverage_count"] = frame[temporal_cols].notna().sum(axis=1) if temporal_cols else 0
    frame["feature_behavior_missing_ratio"] = 1.0 - frame[behavior_cols].notna().mean(axis=1) if behavior_cols else 1.0
    frame["feature_has_complete_recent_windows"] = (
        frame[[c for c in ["prev_exam_score_avg", "hist_exam_score_avg", "prev_library_visit_count", "hist_library_visit_count"] if c in frame.columns]]
        .notna()
        .all(axis=1)
        .astype(int)
        if any(c in frame.columns for c in ["prev_exam_score_avg", "hist_exam_score_avg", "prev_library_visit_count", "hist_library_visit_count"])
        else 0
    )

    # Imbalance features
    if "FEATURE_GRADE_AVG_SCORE" in frame.columns and "FEATURE_LIBRARY_VISIT_COUNT" in frame.columns:
        frame["imbalance_grade_vs_library_z"] = z_by_student(num("FEATURE_GRADE_AVG_SCORE")) - z_by_student(num("FEATURE_LIBRARY_VISIT_COUNT"))

    if "FEATURE_ASSIGNMENT_SCORE_AVG" in frame.columns and "FEATURE_EXAM_SCORE_AVG" in frame.columns:
        frame["imbalance_assignment_vs_exam_level"] = num("FEATURE_ASSIGNMENT_SCORE_AVG") - num("FEATURE_EXAM_SCORE_AVG")

    if "delta_grade_avg_score" in frame.columns and "delta_library_visit_count" in frame.columns:
        frame["imbalance_grade_vs_library_delta"] = num("delta_grade_avg_score") - num("delta_library_visit_count")

    if "FEATURE_ATTENDANCE_EVENT_COUNT" in frame.columns and "delta_grade_avg_score" in frame.columns:
        frame["imbalance_attendance_stable_study_drop"] = (
            (num("FEATURE_ATTENDANCE_EVENT_COUNT").fillna(0) > 0).astype(int)
            * (num("delta_grade_avg_score").fillna(0) < 0).astype(int)
        )

    return frame


# ============================================================================
# Course-Level Risk Features (ENHANCED - for single_fail improvement)
# ============================================================================
def add_course_risk_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add course-level critical risk features targeting single_fail detection.

    These features focus on:
    - Distance to fail threshold (60)
    - Course score dispersion and instability
    - Near-fail indicators and marginal pass
    - Worst course performance and gaps
    - Behavior-result discordance
    - Continuous decline patterns
    """
    frame = data.copy()

    def num(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce")

    # ------------------------------------------------------------------------
    # Group 1: Basic risk indicators
    # ------------------------------------------------------------------------
    # Min score (worst course)
    if "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        frame["course_risk_min_score"] = num("FEATURE_GRADE_MIN_SCORE")

    # Gap between average and min (dispersion proxy)
    if "FEATURE_GRADE_AVG_SCORE" in frame.columns and "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        avg = num("FEATURE_GRADE_AVG_SCORE")
        min_s = num("FEATURE_GRADE_MIN_SCORE")
        frame["course_risk_avg_min_gap"] = avg - min_s

    # Fail count
    if "FEATURE_GRADE_FAIL_COUNT" in frame.columns:
        frame["course_risk_fail_count"] = num("FEATURE_GRADE_FAIL_COUNT")

    # ------------------------------------------------------------------------
    # Group 2: Critical threshold distance features
    # ------------------------------------------------------------------------
    # Distance to 60 (critical threshold) - average
    if "FEATURE_GRADE_AVG_SCORE" in frame.columns:
        frame["course_risk_distance_to_60"] = num("FEATURE_GRADE_AVG_SCORE") - 60

    # Distance to 60 - worst course
    if "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        frame["course_risk_min_distance_to_60"] = num("FEATURE_GRADE_MIN_SCORE") - 60

    # ------------------------------------------------------------------------
    # Group 3: Near-fail and marginal pass indicators
    # ------------------------------------------------------------------------
    # Near-fail indicator (60-65 range on worst course)
    if "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        min_s = num("FEATURE_GRADE_MIN_SCORE")
        frame["course_risk_near_fail"] = ((min_s >= 60) & (min_s <= 65)).astype(int)

    # Marginal pass count (courses between 60-70) - use fail_count as proxy
    if "FEATURE_GRADE_FAIL_COUNT" in frame.columns and "FEATURE_COURSE_SELECTED_COUNT" in frame.columns:
        fail_count = num("FEATURE_GRADE_FAIL_COUNT")
        course_count = num("FEATURE_COURSE_SELECTED_COUNT")
        # Estimate: if avg >= 60 but fail_count > 0, likely has marginal passes
        avg_score = num("FEATURE_GRADE_AVG_SCORE")
        frame["course_risk_marginal_pass_count"] = np.where(
            (avg_score >= 60) & (fail_count > 0),
            fail_count.clip(upper=course_count),
            0
        ).astype(float)

    # ------------------------------------------------------------------------
    # Group 4: Multi-course risk indicators
    # ------------------------------------------------------------------------
    # Multiple courses near danger line (use fail_count as proxy for multi-risk)
    if "FEATURE_GRADE_FAIL_COUNT" in frame.columns:
        fail_count = num("FEATURE_GRADE_FAIL_COUNT")
        frame["course_risk_multi_course_danger"] = (fail_count >= 2).astype(int)

    # ------------------------------------------------------------------------
    # Group 5: Score dispersion and instability
    # ------------------------------------------------------------------------
    # Score dispersion (std across score features)
    score_cols = [c for c in frame.columns if "SCORE" in c and c.startswith("FEATURE_")]
    if len(score_cols) >= 2:
        score_matrix = frame[score_cols].apply(pd.to_numeric, errors="coerce")
        frame["course_risk_dispersion"] = score_matrix.std(axis=1)
        frame["course_risk_dispersion_coef"] = score_matrix.std(axis=1) / (score_matrix.mean(axis=1) + 1e-6)

    # Score variance
    if len(score_cols) >= 2:
        frame["course_risk_score_variance"] = score_matrix.var(axis=1)

    # ------------------------------------------------------------------------
    # Group 6: Weakest course gap analysis
    # ------------------------------------------------------------------------
    # Gap from weakest course to 60
    if "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        min_score = num("FEATURE_GRADE_MIN_SCORE")
        frame["course_risk_weakest_gap_to_60"] = 60 - min_score
        frame["course_risk_weakest_gap_to_70"] = 70 - min_score

    # Weakest course is much worse than average (bottleneck indicator)
    if "FEATURE_GRADE_AVG_SCORE" in frame.columns and "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        avg = num("FEATURE_GRADE_AVG_SCORE")
        min_s = num("FEATURE_GRADE_MIN_SCORE")
        frame["course_risk_bottleneck_severity"] = avg - min_s
        frame["course_risk_bottleneck_ratio"] = (avg - min_s) / (avg + 1e-6)

    # ------------------------------------------------------------------------
    # Group 7: Course dispersion metrics
    # ------------------------------------------------------------------------
    # Course count dispersion proxy
    if "FEATURE_COURSE_SELECTED_COUNT" in frame.columns and "FEATURE_GRADE_FAIL_COUNT" in frame.columns:
        course_count = num("FEATURE_COURSE_SELECTED_COUNT")
        fail_count = num("FEATURE_GRADE_FAIL_COUNT")
        frame["course_risk_fail_rate"] = fail_count / (course_count + 1e-6)

    # ------------------------------------------------------------------------
    # Group 8: Behavior-result discordance features (IMPORTANT)
    # ------------------------------------------------------------------------
    # High attendance but low grades
    if "FEATURE_ATTENDANCE_ABNORMAL_RATE" in frame.columns and "FEATURE_GRADE_AVG_SCORE" in frame.columns:
        attendance_rate = 1.0 - num("FEATURE_ATTENDANCE_ABNORMAL_RATE").fillna(0.5)
        grade_normalized = num("FEATURE_GRADE_AVG_SCORE").fillna(70) / 100.0
        frame["discordance_attendance_grade"] = (attendance_rate - grade_normalized).clip(-1, 1)

    # High assignment submission but low exam scores
    if "FEATURE_ASSIGNMENT_SUBMIT_RATE" in frame.columns and "FEATURE_EXAM_SCORE_AVG" in frame.columns:
        submit_rate = num("FEATURE_ASSIGNMENT_SUBMIT_RATE").fillna(0.5)
        exam_normalized = num("FEATURE_EXAM_SCORE_AVG").fillna(70) / 100.0
        frame["discordance_effort_result"] = (submit_rate - exam_normalized).clip(-1, 1)

    # High course投入 but weakest course continues to worsen
    if "FEATURE_LIBRARY_VISIT_COUNT" in frame.columns and "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        library_count = num("FEATURE_LIBRARY_VISIT_COUNT")
        library_high = (library_count > library_count.median()).astype(int) if library_count.notna().any() else 0
        min_score_low = (num("FEATURE_GRADE_MIN_SCORE") < 65).astype(int)
        frame["discordance_library_grade"] = library_high * min_score_low

    # ------------------------------------------------------------------------
    # Group 9: Continuous decline indicators (using temporal features if available)
    # ------------------------------------------------------------------------
    # Continuous decline in weakest course (if prev_ features exist)
    if "prev_grade_min_score" in frame.columns and "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        prev_min = num("prev_grade_min_score")
        curr_min = num("FEATURE_GRADE_MIN_SCORE")
        frame["course_risk_consecutive_weakest_decline"] = (curr_min < prev_min).astype(int)

    # If we have multiple temporal windows, check for continuous decline
    if "prev_grade_avg_score" in frame.columns and "hist_grade_avg_score" in frame.columns:
        prev_avg = num("prev_grade_avg_score")
        hist_avg = num("hist_grade_avg_score")
        curr_avg = num("FEATURE_GRADE_AVG_SCORE")
        # Continuous decline: curr < prev < hist
        frame["course_risk_consecutive_decline"] = (
            (curr_avg < prev_avg) & (prev_avg < hist_avg)
        ).astype(int)

    return frame


# ============================================================================
# Full Feature Pipeline (SHARED - used by both train and infer)
# ============================================================================
def apply_feature_engineering(data: pd.DataFrame, include_course_risk: bool = True) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.

    This is the UNIFIED entry point for both training and inference.
    Ensures identical feature generation across both paths.

    Order:
    1. Temporal features (prev_*, hist_*, delta_*, etc.)
    2. Interaction features (cross_*, discordance_*, etc.)
    3. Targeted features (trend_*, personal_*, imbalance_*, etc.)
    4. Course risk features (course_risk_*)
    """
    frame = data.copy()
    frame = add_temporal_features(frame)
    frame = add_interaction_features(frame)
    frame = build_targeted_features(frame)
    if include_course_risk:
        frame = add_course_risk_features(frame)
    return frame
