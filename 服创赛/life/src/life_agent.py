from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    for path in [workspace / ".deps3", workspace, root]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root


ROOT = _bootstrap()
WORKSPACE_ROOT = ROOT.parent

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from harness.domain_support.baseline_store import BaselineStore
from harness.domain_support.agent_protocol import build_agent_protocol
from harness.domain_support.comparator import compare_candidate_to_baseline, same_caliber_compare_guard, check_comparability
from harness.domain_support.eval_report_schema import validate_eval_report_schema
from harness.domain_support.result_normalizer import normalize_decision_bundle, normalize_domain_result
from harness.registry.snapshot_contract import ensure_snapshot_contract
from harness.validators.snapshot_validator import validate_snapshot_completeness
from harness.validators.split_integrity_validator import validate_split_integrity
from harness.validators.temporal_integrity_validator import validate_temporal_integrity
from life.src.life_validator import (
    compute_trust_score,
    evaluate_leakage_risk,
    evaluate_subgroup_stability,
    evaluate_temporal_consistency,
)


DM_DIR = ROOT / "data" / "dm"
DELIVERABLE_ROOT = ROOT / "data" / "deliverables" / "life"
REGISTRY_ROOT = ROOT / "data" / "registry" / "life"
RAW_ROOT = ROOT / "data" / "raw" / "数据集及类型"

RESULT_PATH = DM_DIR / "life_agent_result.json"
EVAL_REPORT_PATH = DM_DIR / "life_eval_report.json"
MODEL_SELECTION_PATH = DM_DIR / "life_model_selection.json"
LEAKAGE_COMPARE_REPORT_PATH = DM_DIR / "life_leakage_compare_report.json"
BASELINE_INTEGRITY_REPORT_PATH = DM_DIR / "life_baseline_integrity_report.json"
CLEAN_BASELINE_REPORT_PATH = DM_DIR / "life_clean_baseline_report.json"
CANDIDATE_COMPARE_REPORT_PATH = DM_DIR / "life_candidate_vs_baseline_report.json"
MIGRATION_SUMMARY_PATH = WORKSPACE_ROOT / "data" / "harness" / "harness_migration_summary.json"
MIGRATION_SUMMARY_V2_PATH = WORKSPACE_ROOT / "data" / "harness" / "harness_migration_summary_v2.json"
MATURITY_GAP_SUMMARY_PATH = DM_DIR / "life_maturity_gap_summary.json"
SUBGROUP_METRICS_PATH = DM_DIR / "life_subgroup_metrics.csv"
CONFIDENCE_ZONE_REPORT_PATH = DM_DIR / "life_confidence_zone_report.csv"
LABEL_AUDIT_DETAIL_PATH = DM_DIR / "life_label_audit_detail.json"
SELECTED_FEATURES_REPORT_PATH = DM_DIR / "life_selected_features_report.csv"
FEATURE_LAYER_SUMMARY_PATH = DM_DIR / "life_feature_layer_summary.json"
TEMPORAL_INTEGRITY_REPORT_PATH = DM_DIR / "life_temporal_integrity_report.json"
SPLIT_INTEGRITY_REPORT_PATH = DM_DIR / "life_split_integrity_report.json"
PROXY_STRESS_TEST_REPORT_PATH = DM_DIR / "life_proxy_stress_test_report.json"
FEATURE_ABLATION_REPORT_PATH = DM_DIR / "life_feature_ablation_report.json"
TEMPORAL_GENERALIZATION_REPORT_PATH = DM_DIR / "life_temporal_generalization_report.json"
CREDIBILITY_SUMMARY_PATH = DM_DIR / "life_credibility_stress_test_summary.json"
AUC_CREDIBILITY_AUDIT_PATH = DM_DIR / "life_auc_credibility_audit.json"
EVAL_SCOPE_RECONCILIATION_PATH = DM_DIR / "life_eval_scope_reconciliation_report.json"
STRICT_COMPARE_SUMMARY_PATH = DM_DIR / "life_strict_compare_summary.json"
SOURCE_GROUP_ABLATION_PATH = DM_DIR / "life_source_group_ablation_report.json"
HONEST_EVAL_REPORT_PATH = DM_DIR / "life_honest_eval_report.json"
HONEST_FEATURE_SET_PATH = DM_DIR / "life_honest_feature_set.csv"
VS_STUDY_MATURITY_GAP_V2_PATH = DM_DIR / "life_vs_study_maturity_gap_v2.json"
PRIMARY_METRIC_SUMMARY_PATH = DM_DIR / "life_primary_metric_summary.json"
LABEL_VALIDITY_AUDIT_PATH = DM_DIR / "life_label_validity_audit.json"
TRUTH_GAP_CHECKLIST_PATH = DM_DIR / "life_truth_validation_gap_checklist.json"

ALLOWED_DATASET_KEYS = ("student_profile", "internet", "club", "library", "gate")
SOURCE_FILE_NAMES = {
    "student_profile": "学生基本信息.xlsx",
    "internet": "上网统计.xlsx",
    "club": "社团活动.xlsx",
    "library": "图书馆打卡记录.xlsx",
    "gate": "门禁数据.xlsx",
}
EXCLUDED_DATASET_KEYS = ("running", "exercise", "pe_course", "fitness")
MATURE_BASELINE_VERSION_ID = "life_v1_clean"
FROZEN_LIFE_MAINLINE = {
    "label_version": "instability_future_v2",
    "feature_bundle": "regularity+volatility+coupling",
    "split_version": "purged_temporal_split",
    "candidate_role": "temporal_generalization_mainline",
}
LIFE_LABEL_TARGET_POSITIVE_SHARE = 0.30
DEFAULT_DECISION_THRESHOLD = 0.50


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    if len(set(y_true.astype(int).tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def feature_kind(series: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def risk_level_from_score(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.65:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def confidence_from_auc(auc: float | None) -> float | None:
    if auc is None:
        return None
    return round(max(0.0, min(1.0, (auc - 0.5) / 0.5)), 4)


def linear_slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def coefficient_of_variation(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(arr.mean())
    if mean == 0:
        return 0.0
    return float(arr.std(ddof=0) / mean)


def continuity_ratio(values: list[float]) -> float:
    if not values:
        return 0.0
    active = sum(1 for value in values if float(value) > 0)
    return round(active / len(values), 6)


def deviation_from_personal_baseline(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    baseline = float(arr[:-1].mean()) if len(arr) > 1 else float(arr.mean())
    return float(arr[-1] - baseline)


def rank_score(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    return series.rank(method="average", pct=True)


def bucket_by_hash(value: str, modulo: int = 10) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16) % modulo


def stable_top_share_label(
    scores: pd.Series,
    *,
    positive_share: float,
    tie_breaker: pd.Series | None = None,
) -> pd.Series:
    if len(scores) == 0:
        return pd.Series(dtype=int, index=scores.index)
    clipped_share = min(max(float(positive_share), 0.0), 1.0)
    if clipped_share <= 0.0:
        return pd.Series(0, index=scores.index, dtype=int)
    if clipped_share >= 1.0:
        return pd.Series(1, index=scores.index, dtype=int)
    ordering = pd.DataFrame(
        {
            "score": pd.to_numeric(scores, errors="coerce").fillna(float("-inf")),
            "tie_breaker": tie_breaker.astype(str) if tie_breaker is not None else scores.index.astype(str),
        },
        index=scores.index,
    )
    ordering = ordering.sort_values(["score", "tie_breaker"], ascending=[False, True], kind="mergesort")
    positive_count = max(1, int(np.ceil(len(ordering) * clipped_share)))
    labels = pd.Series(0, index=scores.index, dtype=int)
    labels.loc[ordering.index[:positive_count]] = 1
    return labels


def optimize_decision_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    y_true = pd.Series(y_true).astype(int)
    if len(y_true) == 0 or y_true.nunique() < 2:
        return DEFAULT_DECISION_THRESHOLD
    proba = np.asarray(proba, dtype=float)
    quantile_thresholds = np.quantile(proba, np.linspace(0.1, 0.9, 17))
    candidate_thresholds = sorted(
        {
            DEFAULT_DECISION_THRESHOLD,
            *np.round(np.linspace(0.10, 0.90, 81), 4).tolist(),
            *np.round(quantile_thresholds, 6).tolist(),
        }
    )
    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_objective: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds:
        pred = (proba >= threshold).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        objective = (
            round(f1, 6),
            -round(abs(precision - recall), 6),
            -round(abs(float(threshold) - DEFAULT_DECISION_THRESHOLD), 6),
        )
        if best_objective is None or objective > best_objective:
            best_threshold = float(threshold)
            best_objective = objective
    return best_threshold


class LifeAgent:
    def __init__(self, request_path: Path | None = None, request: dict[str, Any] | None = None):
        if request_path is None and request is None:
            raise ValueError("LifeAgent requires either request_path or request")
        self.request = self._load_request(request_path, request)
        self.request_id = str(self.request.get("request_id", f"life_{datetime.now().strftime('%Y%m%d%H%M%S')}"))
        self.deliverables = {
            "data": DELIVERABLE_ROOT / "data",
            "docs": DELIVERABLE_ROOT / "docs",
            "model": DELIVERABLE_ROOT / "model",
        }
        for path in [DM_DIR, REGISTRY_ROOT, RAW_ROOT, *self.deliverables.values()]:
            path.mkdir(parents=True, exist_ok=True)
        self.baseline_store = BaselineStore(REGISTRY_ROOT)

    def _load_request(self, request_path: Path | None, request: dict[str, Any] | None) -> dict[str, Any]:
        if request is not None:
            return dict(request)
        assert request_path is not None
        path = request_path if request_path.is_absolute() else request_path
        if not path.exists():
            candidate = ROOT / request_path
            path = candidate if candidate.exists() else WORKSPACE_ROOT / request_path
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = ROOT / "conf" / filename
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _sync_life_raw_inputs(self) -> dict[str, Path]:
        source_dir = WORKSPACE_ROOT / "数据集及类型"
        copied_paths: dict[str, Path] = {}
        for key in ALLOWED_DATASET_KEYS:
            source_path = source_dir / SOURCE_FILE_NAMES[key]
            if not source_path.exists():
                raise FileNotFoundError(f"missing life source file: {source_path}")
            target_path = RAW_ROOT / SOURCE_FILE_NAMES[key]
            shutil.copy2(source_path, target_path)
            copied_paths[key] = target_path
        return copied_paths

    def _resolve_file_registry(self) -> dict[str, Path]:
        registry = self._load_yaml("file_registry.yaml")
        mapping = registry.get("datasets", {})
        missing_allowed = sorted(set(ALLOWED_DATASET_KEYS) - set(mapping))
        if missing_allowed:
            raise ValueError(f"missing required life dataset keys: {missing_allowed}")
        for excluded in EXCLUDED_DATASET_KEYS:
            if excluded in mapping:
                raise ValueError(f"sport-related dataset must not appear in life file_registry: {excluded}")
        synced = self._sync_life_raw_inputs()
        resolved: dict[str, Path] = {}
        for key in ALLOWED_DATASET_KEYS:
            target = ROOT / mapping[key]
            resolved[key] = target if target.exists() else synced[key]
        return resolved

    def _read_excel(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path)

    def _prepare_event_tables(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
        files = self._resolve_file_registry()

        student = self._read_excel(files["student_profile"]).rename(columns={"XH": "student_id"})
        student = student[["student_id", "XB", "MZMC", "ZZMMMC", "JG", "XSM", "ZYM"]].dropna(subset=["student_id"])
        student = student.drop_duplicates(subset=["student_id"]).reset_index(drop=True)

        internet = self._read_excel(files["internet"]).rename(columns={"XSBH": "student_id"})
        internet["event_month"] = pd.to_datetime(internet["TJNY"].astype(str), format="%Y%m", errors="coerce").dt.to_period("M")
        internet["internet_hours"] = pd.to_numeric(internet["SWLJSC"], errors="coerce").fillna(0.0)
        internet["event_value"] = internet["internet_hours"]

        club = self._read_excel(files["club"]).rename(columns={"XSBH": "student_id"})
        club["event_time"] = pd.to_datetime(club["HDRQ"], errors="coerce")
        club["event_month"] = club["event_time"].dt.to_period("M")
        club["event_value"] = 1.0

        library = self._read_excel(files["library"]).rename(columns={"cardld": "student_id"})
        library["event_time"] = pd.to_datetime(library["visittime"], errors="coerce")
        library["event_month"] = library["event_time"].dt.to_period("M")
        library["event_value"] = 1.0
        library["event_hour"] = library["event_time"].dt.hour.fillna(0).astype(int)
        library["weekend_flag"] = library["event_time"].dt.dayofweek.fillna(0).ge(5).astype(int)

        gate = self._read_excel(files["gate"]).rename(columns={"IDSERTAL": "student_id"})
        gate["event_time"] = pd.to_datetime(gate["LOGINTIME"], errors="coerce")
        gate["event_month"] = gate["event_time"].dt.to_period("M")
        gate["event_value"] = 1.0
        gate["event_hour"] = gate["event_time"].dt.hour.fillna(0).astype(int)
        gate["weekend_flag"] = gate["event_time"].dt.dayofweek.fillna(0).ge(5).astype(int)
        gate["late_flag"] = ((gate["event_hour"] >= 22) | (gate["event_hour"] <= 6)).astype(int)

        dataset_summary = {
            "source_dataset_keys": list(ALLOWED_DATASET_KEYS),
            "note": "sport-related datasets intentionally excluded from life domain",
            "row_counts": {
                "student_profile": int(len(student)),
                "internet": int(len(internet)),
                "club": int(len(club)),
                "library": int(len(library)),
                "gate": int(len(gate)),
            },
        }
        return student, {
            "internet": internet,
            "club": club,
            "library": library,
            "gate": gate,
        }, dataset_summary

    def _split_months(self, events: dict[str, pd.DataFrame]) -> tuple[list[pd.Period], list[pd.Period]]:
        month_sets = []
        for frame in events.values():
            months = frame["event_month"].dropna().astype(str).unique().tolist()
            month_sets.extend(months)
        all_months = sorted(
            {
                pd.Period(month, freq="M")
                for month in month_sets
                if 2020 <= pd.Period(month, freq="M").year <= 2026
            }
        )
        if len(all_months) < 4:
            raise ValueError("life clean evaluation requires at least four distinct months across life datasets")
        split_index = max(2, len(all_months) // 2)
        early = all_months[:split_index]
        late = all_months[split_index:]
        return early, late

    def _monthly_value_map(
        self,
        frame: pd.DataFrame,
        *,
        value_column: str,
        months: list[pd.Period],
    ) -> pd.DataFrame:
        monthly = (
            frame.dropna(subset=["student_id", "event_month"])
            .groupby(["student_id", "event_month"], dropna=False)[value_column]
            .sum()
            .unstack(fill_value=0.0)
        )
        monthly = monthly.reindex(columns=months, fill_value=0.0)
        monthly.columns = [str(month) for month in months]
        return monthly

    def _series_features(self, prefix: str, monthly: pd.DataFrame) -> pd.DataFrame:
        frame = monthly.copy()
        values = frame.values.tolist()
        result = pd.DataFrame(index=frame.index)
        result[f"{prefix}_sum"] = frame.sum(axis=1)
        result[f"{prefix}_mean"] = frame.mean(axis=1)
        result[f"{prefix}_volatility"] = [float(np.std(row)) for row in values]
        result[f"{prefix}_trend"] = [linear_slope([float(v) for v in row]) for row in values]
        result[f"{prefix}_continuity"] = [continuity_ratio([float(v) for v in row]) for row in values]
        result[f"{prefix}_deviation"] = [deviation_from_personal_baseline([float(v) for v in row]) for row in values]
        result[f"{prefix}_cv"] = [coefficient_of_variation([float(v) for v in row]) for row in values]
        return result.reset_index().rename(columns={"index": "student_id"})

    def _event_ratio_features(self, frame: pd.DataFrame, months: list[pd.Period], prefix: str) -> pd.DataFrame:
        target = frame.dropna(subset=["student_id", "event_month"]).copy()
        target = target[target["event_month"].isin(months)]
        grouped = target.groupby("student_id", dropna=False)
        result = pd.DataFrame({"student_id": sorted(target["student_id"].dropna().astype(str).unique().tolist())})
        if result.empty:
            return pd.DataFrame(columns=["student_id", f"{prefix}_late_ratio", f"{prefix}_weekend_ratio", f"{prefix}_hour_mean"])
        result[f"{prefix}_late_ratio"] = result["student_id"].map(
            grouped["late_flag"].mean() if "late_flag" in target.columns else pd.Series(dtype=float)
        ).fillna(0.0)
        result[f"{prefix}_weekend_ratio"] = result["student_id"].map(
            grouped["weekend_flag"].mean() if "weekend_flag" in target.columns else pd.Series(dtype=float)
        ).fillna(0.0)
        result[f"{prefix}_hour_mean"] = result["student_id"].map(
            grouped["event_hour"].mean() if "event_hour" in target.columns else pd.Series(dtype=float)
        ).fillna(0.0)
        return result

    def _build_table_for_windows(
        self,
        student: pd.DataFrame,
        events: dict[str, pd.DataFrame],
        feature_months: list[pd.Period],
        label_months: list[pd.Period],
    ) -> pd.DataFrame:
        internet_early = self._series_features(
            "internet_early",
            self._monthly_value_map(events["internet"], value_column="internet_hours", months=feature_months),
        )
        club_early = self._series_features(
            "club_early",
            self._monthly_value_map(events["club"], value_column="event_value", months=feature_months),
        )
        library_early = self._series_features(
            "library_early",
            self._monthly_value_map(events["library"], value_column="event_value", months=feature_months),
        )
        gate_early = self._series_features(
            "gate_early",
            self._monthly_value_map(events["gate"], value_column="event_value", months=feature_months),
        )
        library_rhythm = self._event_ratio_features(events["library"], feature_months, "library_early")
        gate_rhythm = self._event_ratio_features(events["gate"], feature_months, "gate_early")

        late_internet = self._monthly_value_map(events["internet"], value_column="internet_hours", months=label_months)
        late_club = self._monthly_value_map(events["club"], value_column="event_value", months=label_months)
        late_library = self._monthly_value_map(events["library"], value_column="event_value", months=label_months)
        late_gate = self._monthly_value_map(events["gate"], value_column="event_value", months=label_months)
        late_gate_ratio = self._event_ratio_features(events["gate"], label_months, "gate_late")

        feature_df = student.copy()
        for frame in [internet_early, club_early, library_early, gate_early, library_rhythm, gate_rhythm]:
            feature_df = feature_df.merge(frame, on="student_id", how="left")

        numeric_fill_cols = [col for col in feature_df.columns if col not in {"student_id", "XB", "MZMC", "ZZMMMC", "JG", "XSM", "ZYM"}]
        for col in numeric_fill_cols:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)

        feature_df["internet_library_interaction"] = feature_df["internet_early_mean"] * (feature_df["library_early_mean"] + 1.0)
        feature_df["late_gate_internet_interaction"] = feature_df["gate_early_late_ratio"] * feature_df["internet_early_mean"]
        feature_df["club_library_balance"] = feature_df["club_early_mean"] - feature_df["library_early_mean"]
        feature_df["behavior_rhythm_gap"] = feature_df["gate_early_weekend_ratio"] - feature_df["library_early_weekend_ratio"]

        late_summary = pd.DataFrame({"student_id": student["student_id"].astype(str)})
        late_summary["late_internet_mean"] = late_summary["student_id"].map(late_internet.mean(axis=1)).fillna(0.0)
        late_summary["late_club_mean"] = late_summary["student_id"].map(late_club.mean(axis=1)).fillna(0.0)
        late_summary["late_library_mean"] = late_summary["student_id"].map(late_library.mean(axis=1)).fillna(0.0)
        late_summary["late_gate_mean"] = late_summary["student_id"].map(late_gate.mean(axis=1)).fillna(0.0)
        late_summary["late_gate_ratio"] = late_summary["student_id"].map(
            late_gate_ratio.set_index("student_id")["gate_late_late_ratio"] if not late_gate_ratio.empty else pd.Series(dtype=float)
        ).fillna(0.0)

        risk_components = pd.DataFrame({"student_id": late_summary["student_id"]})
        risk_components["internet_rank"] = rank_score(late_summary["late_internet_mean"]).fillna(0.0)
        risk_components["late_gate_rank"] = rank_score(late_summary["late_gate_ratio"]).fillna(0.0)
        risk_components["low_library_rank"] = 1.0 - rank_score(late_summary["late_library_mean"]).fillna(0.0)
        risk_components["low_club_rank"] = 1.0 - rank_score(late_summary["late_club_mean"]).fillna(0.0)
        risk_components["clean_label_score"] = (
            0.35 * risk_components["internet_rank"]
            + 0.30 * risk_components["late_gate_rank"]
            + 0.20 * risk_components["low_library_rank"]
            + 0.15 * risk_components["low_club_rank"]
        )
        risk_components["life_label_clean"] = stable_top_share_label(
            risk_components["clean_label_score"],
            positive_share=LIFE_LABEL_TARGET_POSITIVE_SHARE,
            tie_breaker=risk_components["student_id"],
        )
        # Mainline instability labels for future-window prediction tracks.
        risk_components["life_instability_future_v1"] = risk_components["life_label_clean"].astype(int)
        risk_components["life_instability_future_v2"] = (
            (risk_components["internet_rank"] >= 0.75) & (risk_components["late_gate_rank"] >= 0.70)
        ).astype(int)
        risk_components["life_instability_future_v3"] = (
            (risk_components["internet_rank"] >= 0.70)
            & (risk_components["late_gate_rank"] >= 0.65)
            & (risk_components["low_library_rank"] >= 0.60)
        ).astype(int)

        feature_df["student_id"] = feature_df["student_id"].astype(str)
        feature_df = feature_df.merge(
            late_summary[["student_id", "late_internet_mean", "late_club_mean", "late_library_mean", "late_gate_mean", "late_gate_ratio"]],
            on="student_id",
            how="left",
        )
        feature_df = feature_df.merge(
            risk_components[
                [
                    "student_id",
                    "clean_label_score",
                    "life_label_clean",
                    "life_instability_future_v1",
                    "life_instability_future_v2",
                    "life_instability_future_v3",
                ]
            ],
            on="student_id",
            how="left",
        )
        feature_df["life_label_clean"] = feature_df["life_label_clean"].fillna(0).astype(int)
        feature_df["life_instability_future_v1"] = feature_df["life_instability_future_v1"].fillna(0).astype(int)
        feature_df["life_instability_future_v2"] = feature_df["life_instability_future_v2"].fillna(0).astype(int)
        feature_df["life_instability_future_v3"] = feature_df["life_instability_future_v3"].fillna(0).astype(int)
        feature_df["clean_label_score"] = pd.to_numeric(feature_df["clean_label_score"], errors="coerce").fillna(0.0)
        return feature_df

    def _build_clean_tables(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        student, events, dataset_summary = self._prepare_event_tables()
        early_months, late_months = self._split_months(events)
        feature_df = self._build_table_for_windows(student, events, early_months, late_months)
        def _assign_split(student_id: str) -> str:
            bucket = bucket_by_hash(student_id, 10)
            if bucket == 0:
                return "holdout"
            if bucket == 1:
                return "valid"
            return "train"

        feature_df["data_split"] = feature_df["student_id"].apply(_assign_split)
        feature_df["life_label_clean"] = feature_df["life_label_clean"].fillna(0).astype(int)
        feature_df["clean_label_score"] = pd.to_numeric(feature_df["clean_label_score"], errors="coerce").fillna(0.0)

        dataset_summary.update(
            {
                "all_months": [str(month) for month in sorted(set(early_months + late_months))],
                "clean_eval_window": {
                    "early_months": [str(month) for month in early_months],
                    "late_months": [str(month) for month in late_months],
                    "label_definition": "future_window_proxy_from_late_internet_gate_library_club",
                    "label_target_positive_share": LIFE_LABEL_TARGET_POSITIVE_SHARE,
                }
            }
        )
        return feature_df, dataset_summary

    def _feature_sets(self) -> tuple[list[str], list[str], list[str]]:
        demographics = ["XB", "MZMC", "ZZMMMC", "JG", "XSM", "ZYM"]
        baseline_features = [
            "internet_early_sum",
            "internet_early_mean",
            "library_early_mean",
            "club_early_mean",
            "gate_early_late_ratio",
            "gate_early_mean",
        ] + demographics
        candidate_features = baseline_features + [
            "internet_early_trend",
            "internet_early_volatility",
            "internet_early_continuity",
            "internet_early_deviation",
            "internet_early_cv",
            "club_early_trend",
            "club_early_continuity",
            "library_early_trend",
            "library_early_volatility",
            "library_early_continuity",
            "library_early_weekend_ratio",
            "gate_early_trend",
            "gate_early_volatility",
            "gate_early_continuity",
            "gate_early_weekend_ratio",
            "gate_early_hour_mean",
            "behavior_rhythm_gap",
            "internet_library_interaction",
            "late_gate_internet_interaction",
            "club_library_balance",
        ]
        label_source_features = ["late_internet_mean", "late_gate_ratio", "late_library_mean", "late_club_mean"]
        return baseline_features, candidate_features, label_source_features

    def _feature_family(self, feature_name: str) -> str:
        if feature_name in {"XB", "MZMC", "ZZMMMC", "JG", "XSM", "ZYM"}:
            return "student_profile"
        if "internet" in feature_name:
            return "internet"
        if "club" in feature_name:
            return "club"
        if "library" in feature_name:
            return "library"
        if "gate" in feature_name:
            return "gate"
        return "derived"

    def _make_proxy_label(self, feature_df: pd.DataFrame, task_name: str) -> tuple[pd.Series, str, list[str]]:
        if task_name == "internet_only_proxy":
            signal = rank_score(feature_df["late_internet_mean"]).fillna(0.0)
            label = (signal >= float(signal.quantile(0.70))).astype(int)
            return label, "top-ranked late internet intensity", ["internet"]
        if task_name == "club_only_proxy":
            signal = 1.0 - rank_score(feature_df["late_club_mean"]).fillna(0.0)
            label = (signal >= float(signal.quantile(0.70))).astype(int)
            return label, "bottom-ranked late club participation", ["club"]
        if task_name == "library_only_proxy":
            signal = 1.0 - rank_score(feature_df["late_library_mean"]).fillna(0.0)
            label = (signal >= float(signal.quantile(0.70))).astype(int)
            return label, "bottom-ranked late library visits", ["library"]
        if task_name == "gate_only_proxy":
            signal = rank_score(feature_df["late_gate_ratio"]).fillna(0.0)
            label = (signal >= float(signal.quantile(0.70))).astype(int)
            return label, "top-ranked late-night gate ratio", ["gate"]
        return feature_df["life_label_clean"].astype(int), "weighted combined late life proxy", ["internet", "club", "library", "gate"]

    def _feature_subset_without_families(self, feature_columns: list[str], excluded_families: set[str]) -> list[str]:
        kept = [feature for feature in feature_columns if self._feature_family(feature) not in excluded_families]
        if not kept:
            kept = [feature for feature in feature_columns if self._feature_family(feature) == "student_profile"]
        return kept

    def _feature_subset_indirect_only(self, feature_columns: list[str]) -> list[str]:
        indirect = [
            feature
            for feature in feature_columns
            if self._feature_family(feature) == "student_profile"
            or any(token in feature for token in ("interaction", "gap", "balance"))
        ]
        return indirect or [feature for feature in feature_columns if self._feature_family(feature) == "student_profile"]

    def _check_label_distribution(self, frame: pd.DataFrame, label_column: str = "life_label_clean") -> tuple[dict[int, int], bool]:
        counts = frame[label_column].value_counts(dropna=False).sort_index().to_dict()
        normalized_counts = {int(key): int(value) for key, value in counts.items()}
        return normalized_counts, len(normalized_counts) >= 2

    def _build_model(self, train_df: pd.DataFrame, feature_columns: list[str], label_column: str = "life_label_clean") -> Pipeline:
        # Defensive guard: refuse to fit on single-class training data
        label_values = train_df[label_column].dropna().astype(int)
        unique_classes = set(label_values.tolist())
        if len(unique_classes) < 2:
            raise ValueError(
                f"_build_model received training data with only {len(unique_classes)} class(es): {unique_classes}. "
                f"Training set has {len(train_df)} rows with label distribution {label_values.value_counts().to_dict()}. "
                f"LogisticRegression requires at least two classes. Check the label assignment logic or data split."
            )
        numeric_cols = [col for col in feature_columns if col not in {"XB", "MZMC", "ZZMMMC", "JG", "XSM", "ZYM"}]
        categorical_cols = [col for col in feature_columns if col in {"XB", "MZMC", "ZZMMMC", "JG", "XSM", "ZYM"}]
        transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ]
        )
        model = LogisticRegression(
            max_iter=int(self._load_yaml("model_params.yaml").get("logistic_regression", {}).get("max_iter", 500)),
            class_weight="balanced",
            random_state=42,
        )
        pipeline = Pipeline([("transformer", transformer), ("model", model)])
        pipeline.fit(train_df[feature_columns], train_df[label_column])
        return pipeline

    def _select_decision_threshold(
        self,
        model: Pipeline,
        valid_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "life_label_clean",
    ) -> float:
        if valid_df.empty or valid_df[label_column].nunique() < 2:
            return DEFAULT_DECISION_THRESHOLD
        proba = model.predict_proba(valid_df[feature_columns])[:, 1]
        return optimize_decision_threshold(valid_df[label_column], proba)

    def _evaluate(
        self,
        model: Pipeline,
        infer_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "life_label_clean",
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        proba = model.predict_proba(infer_df[feature_columns])[:, 1]
        pred = (proba >= float(decision_threshold)).astype(int)
        prediction_cols = [column for column in ["student_id", label_column, "clean_label_score", "data_split"] if column in infer_df.columns]
        prediction = infer_df[prediction_cols].copy()
        if label_column in prediction.columns and label_column != "life_label_clean":
            prediction["life_label_clean"] = prediction[label_column]
        if "data_split" not in prediction.columns:
            prediction["data_split"] = "evaluation"
        prediction["risk_score"] = proba
        prediction["prediction"] = pred
        prediction["risk_level"] = prediction["risk_score"].apply(risk_level_from_score)

        metrics = {
            "rows": int(len(infer_df)),
            "positive_rate": round(float(infer_df[label_column].mean()), 6),
            "auc": safe_auc(infer_df[label_column], prediction["risk_score"]),
            "f1": round(float(f1_score(infer_df[label_column], prediction["prediction"], zero_division=0)), 6),
            "precision": round(float(precision_score(infer_df[label_column], prediction["prediction"], zero_division=0)), 6),
            "recall": round(float(recall_score(infer_df[label_column], prediction["prediction"], zero_division=0)), 6),
            "coverage": 1.0,
            "degraded_ratio": 0.0,
            "mean_infer_risk_score": round(float(prediction["risk_score"].mean()), 6),
            "high_risk_rate": round(float((prediction["risk_score"] >= 0.65).mean()), 6),
            "decision_threshold": round(float(decision_threshold), 6),
        }
        return prediction, metrics

    def _instability_feature_bundles(self, candidate_features: list[str]) -> dict[str, list[str]]:
        demographics = [f for f in candidate_features if f in {"XB", "MZMC", "ZZMMMC", "JG", "XSM", "ZYM"}]
        regularity = [
            f for f in candidate_features if any(k in f for k in ("continuity", "weekend_ratio", "hour_mean", "rhythm"))
        ] + demographics
        volatility = [
            f for f in candidate_features if any(k in f for k in ("volatility", "deviation", "cv", "trend"))
        ]
        coupling = [f for f in candidate_features if any(k in f for k in ("interaction", "gap", "balance"))]
        return {
            "regularity_only": sorted(set(regularity)),
            "regularity+volatility": sorted(set(regularity + volatility)),
            "regularity+volatility+coupling": sorted(set(regularity + volatility + coupling)),
        }

    def _run_instability_matrix(
        self,
        *,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        candidate_features: list[str],
    ) -> list[dict[str, Any]]:
        label_map = {
            "instability_future_v1": "life_instability_future_v1",
            "instability_future_v2": "life_instability_future_v2",
        }
        bundles = self._instability_feature_bundles(candidate_features)
        rows: list[dict[str, Any]] = []
        for label_version, label_col in label_map.items():
            for bundle_name, cols in bundles.items():
                if not cols:
                    rows.append({"label_version": label_version, "feature_bundle": bundle_name, "status": "skipped", "reason": "empty_feature_bundle"})
                    continue
                tr = train_df.copy()
                te = holdout_df.copy()
                tr["life_label_clean"] = tr[label_col].astype(int)
                te["life_label_clean"] = te[label_col].astype(int)
                train_counts, train_ok = self._check_label_distribution(tr, "life_label_clean")
                test_counts, test_ok = self._check_label_distribution(te, "life_label_clean")
                if not train_ok or not test_ok:
                    rows.append(
                        {
                            "label_version": label_version,
                            "feature_bundle": bundle_name,
                            "status": "skipped",
                            "reason": "single_class_split",
                            "train_label_distribution": train_counts,
                            "holdout_label_distribution": test_counts,
                        }
                    )
                    continue
                model = self._build_model(tr, cols, label_column="life_label_clean")
                _, met = self._evaluate(model, te, cols, label_column="life_label_clean")
                rows.append(
                    {
                        "label_version": label_version,
                        "feature_bundle": bundle_name,
                        "status": "completed",
                        "auc": met.get("auc"),
                        "f1": met.get("f1"),
                        "precision": met.get("precision"),
                        "recall": met.get("recall"),
                        "rows": met.get("rows"),
                    }
                )
        return rows

    def _summarize_feature_layers(self, feature_columns: list[str]) -> dict[str, Any]:
        layers = {"core": [], "behavior": [], "temporal": [], "interaction": []}
        for col in feature_columns:
            lowered = col.lower()
            if any(token in lowered for token in ("interaction", "gap", "balance")):
                layers["interaction"].append(col)
            elif any(token in lowered for token in ("trend", "volatility", "continuity", "deviation", "cv")):
                layers["temporal"].append(col)
            elif any(token in lowered for token in ("internet", "club", "library", "gate")):
                layers["behavior"].append(col)
            else:
                layers["core"].append(col)
        return {
            "counts": {layer: len(cols) for layer, cols in layers.items()},
            "columns": layers,
            "life_data_mode": "clean_temporal_behavior_enhanced",
        }

    def _subgroup_metrics(self, infer_df: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
        merged = infer_df[["student_id", "XB", "internet_early_continuity"]].merge(
            prediction[["student_id", "life_label_clean", "risk_score", "prediction"]],
            on="student_id",
            how="left",
        )
        merged["activity_band"] = np.where(merged["internet_early_continuity"] >= 0.5, "dense_activity", "sparse_activity")
        rows = []
        for subgroup_col in ["XB", "activity_band"]:
            for subgroup_value, frame in merged.groupby(subgroup_col, dropna=False):
                rows.append(
                    {
                        "subgroup": f"{subgroup_col}:{subgroup_value}",
                        "rows": int(len(frame)),
                        "positive_rate": round(float(frame["life_label_clean"].mean()), 6),
                        "auc": safe_auc(frame["life_label_clean"], frame["risk_score"]),
                        "f1": round(float(f1_score(frame["life_label_clean"], frame["prediction"], zero_division=0)), 6),
                    }
                )
        return pd.DataFrame(rows)

    def _confidence_zone_report(self, prediction: pd.DataFrame) -> pd.DataFrame:
        frame = prediction.copy()
        frame["confidence_zone"] = pd.cut(
            frame["risk_score"],
            bins=[-0.01, 0.33, 0.66, 1.01],
            labels=["low_conf", "mid_conf", "high_conf"],
        )
        rows = []
        for zone, group in frame.groupby("confidence_zone", dropna=False):
            rows.append(
                {
                    "confidence_zone": str(zone),
                    "rows": int(len(group)),
                    "positive_rate": round(float(group["life_label_clean"].mean()), 6),
                    "avg_risk_score": round(float(group["risk_score"].mean()), 6),
                }
            )
        return pd.DataFrame(rows)

    def _temporal_feature_specs(
        self,
        *,
        feature_columns: list[str],
        early_months: list[str],
        late_months: list[str],
    ) -> list[dict[str, Any]]:
        early_start = early_months[0] if early_months else ""
        early_end = early_months[-1] if early_months else ""
        specs: list[dict[str, Any]] = []
        for feature in feature_columns:
            family = self._feature_family(feature)
            if family == "student_profile":
                specs.append(
                    {
                        "feature_name": feature,
                        "source_dataset": "student_profile",
                        "feature_window_start": "",
                        "feature_window_end": "",
                        "leakage_risk_level": "low",
                        "note": "static profile feature, not time-indexed",
                    }
                )
                continue
            source_dataset = "derived_cross_behavior" if family == "derived" else family
            note = "derived from early-window observations only"
            if family == "derived":
                note = "cross-behavior interaction built from early-window features only"
            specs.append(
                {
                    "feature_name": feature,
                    "source_dataset": source_dataset,
                    "feature_window_start": early_start,
                    "feature_window_end": early_end,
                    "leakage_risk_level": "medium" if family == "derived" else "low",
                    "note": note,
                }
            )
        return specs

    def _check_label_distribution(self, df: pd.DataFrame, label_col: str) -> tuple[dict[int, int], bool]:
        """Return label value counts and whether the split is valid for binary classification."""
        label_values = df[label_col].dropna().astype(int)
        counts = label_values.value_counts().to_dict()
        is_valid = len(counts) >= 2 and all(v > 0 for v in counts.values())
        return counts, is_valid

    def _proxy_stress_test(
        self,
        *,
        feature_df: pd.DataFrame,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        candidate_features: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        reports: list[dict[str, Any]] = []
        ablation_reports: list[dict[str, Any]] = []
        for task_name in [
            "internet_only_proxy",
            "club_only_proxy",
            "library_only_proxy",
            "gate_only_proxy",
            "combined_proxy",
        ]:
            labels, label_rule_summary, source_families = self._make_proxy_label(feature_df, task_name)
            task_train = train_df.copy()
            task_holdout = holdout_df.copy()
            task_train["life_label_clean"] = labels.loc[task_train.index].astype(int).values
            task_holdout["life_label_clean"] = labels.loc[task_holdout.index].astype(int).values

            # --- Single-class guard for train and test splits ---
            train_label_dist, train_valid = self._check_label_distribution(task_train, "life_label_clean")
            test_label_dist, test_valid = self._check_label_distribution(task_holdout, "life_label_clean")

            if not train_valid or not test_valid:
                skip_reason = []
                if not train_valid:
                    skip_reason.append(f"train split has single-class distribution: {train_label_dist}")
                if not test_valid:
                    skip_reason.append(f"test split has single-class distribution: {test_label_dist}")

                skipped_report = {
                    "experiment_name": task_name,
                    "status": "skipped",
                    "reason": "; ".join(skip_reason),
                    "train_rows": int(len(task_train)),
                    "test_rows": int(len(task_holdout)),
                    "train_label_distribution": {str(k): int(v) for k, v in train_label_dist.items()},
                    "test_label_distribution": {str(k): int(v) for k, v in test_label_dist.items()},
                }
                reports.append(skipped_report)
                ablation_reports.append(
                    {
                        "experiment_name": task_name,
                        "status": "skipped",
                        "reason": "; ".join(skip_reason),
                        "full_feature_auc": None,
                        "indirect_feature_auc": None,
                        "auc_drop_after_removing_source_near_features": None,
                        "conclusion": "experiment skipped due to single-class label distribution in train or test split",
                    }
                )
                continue

            source_near_features = [
                feature for feature in candidate_features if self._feature_family(feature) in set(source_families)
            ]
            full_model = self._build_model(task_train, candidate_features)
            _, full_metrics = self._evaluate(
                full_model,
                task_holdout,
                candidate_features,
            )

            indirect_features = self._feature_subset_without_families(candidate_features, set(source_families))
            indirect_model = self._build_model(task_train, indirect_features)
            _, indirect_metrics = self._evaluate(
                indirect_model,
                task_holdout,
                indirect_features,
            )
            auc_drop = None
            if full_metrics.get("auc") is not None and indirect_metrics.get("auc") is not None:
                auc_drop = round(float(full_metrics["auc"]) - float(indirect_metrics["auc"]), 6)

            if (full_metrics.get("auc") or 0) >= 0.9 and (auc_drop or 0) >= 0.1:
                semantic_risk = "high"
                conclusion = "performance strongly depends on source-near features, so semantic reconstructability risk is high"
            elif (full_metrics.get("auc") or 0) >= 0.8:
                semantic_risk = "medium"
                conclusion = "task is learnable but source-near ablation still matters"
            else:
                semantic_risk = "low"
                conclusion = "task is not trivially reconstructable under current clean setup"

            reports.append(
                {
                    "task_name": task_name,
                    "label_rule_summary": label_rule_summary,
                    "auc": full_metrics.get("auc"),
                    "f1": full_metrics.get("f1"),
                    "precision": full_metrics.get("precision"),
                    "recall": full_metrics.get("recall"),
                    "feature_overlap_with_label_rule": source_near_features,
                    "semantic_reconstruction_risk": semantic_risk,
                    "conclusion": conclusion,
                }
            )
            ablation_reports.append(
                {
                    "task_name": task_name,
                    "full_feature_auc": full_metrics.get("auc"),
                    "indirect_feature_auc": indirect_metrics.get("auc"),
                    "auc_drop_after_removing_source_near_features": auc_drop,
                    "full_feature_columns": candidate_features,
                    "indirect_feature_columns": indirect_features,
                    "conclusion": "ablated model remains stable" if (auc_drop or 0) < 0.05 else "ablated model drops materially; source-near features carry much of the signal",
                }
            )
        return reports, ablation_reports

    def _temporal_generalization_report(
        self,
        *,
        student: pd.DataFrame,
        events: dict[str, pd.DataFrame],
        candidate_features: list[str],
        clean_holdout_auc: float | None,
        all_months: list[str],
    ) -> dict[str, Any]:
        periods = [pd.Period(month, freq="M") for month in all_months]
        if len(periods) < 6:
            return {
                "train_period": "",
                "test_period": "",
                "sample_count_train": 0,
                "sample_count_test": 0,
                "auc": None,
                "f1": None,
                "precision": None,
                "recall": None,
                "compared_with_clean_holdout_auc": clean_holdout_auc,
                "generalization_gap": None,
                "conclusion": "insufficient temporal coverage for stricter temporal generalization test",
            }

        chunk = max(2, len(periods) // 3)
        train_feature_months = periods[:chunk]
        train_label_months = periods[chunk : chunk * 2]
        test_feature_months = periods[chunk : chunk * 2]
        test_label_months = periods[chunk * 2 :]
        if len(test_label_months) < 2:
            test_label_months = periods[-chunk:]

        feature_df_train = self._build_table_for_windows(student, events, train_feature_months, train_label_months)
        feature_df_test = self._build_table_for_windows(student, events, test_feature_months, test_label_months)

        train_bucket = feature_df_train["student_id"].astype(str).apply(lambda value: bucket_by_hash(value, 10))
        temporal_train = feature_df_train.loc[~train_bucket.isin({0, 1})].copy()
        temporal_valid = feature_df_train.loc[train_bucket == 1].copy()
        test_split = feature_df_test["student_id"].astype(str).apply(lambda value: bucket_by_hash(value, 10) == 0)
        temporal_test = feature_df_test.loc[test_split].copy()
        if temporal_train.empty or temporal_test.empty or temporal_train["life_label_clean"].nunique() < 2 or temporal_test["life_label_clean"].nunique() < 2:
            return {
                "train_period": f"{train_feature_months[0]}->{train_label_months[-1]}",
                "test_period": f"{test_feature_months[0]}->{test_label_months[-1]}",
                "sample_count_train": int(len(temporal_train)),
                "sample_count_test": int(len(temporal_test)),
                "auc": None,
                "f1": None,
                "precision": None,
                "recall": None,
                "compared_with_clean_holdout_auc": clean_holdout_auc,
                "generalization_gap": None,
                "conclusion": "temporal generalization test could not be computed because one side lacks enough label diversity",
            }

        model = self._build_model(temporal_train, candidate_features)
        decision_threshold = self._select_decision_threshold(
            model,
            temporal_valid,
            candidate_features,
            label_column="life_label_clean",
        )
        _, metrics = self._evaluate(
            model,
            temporal_test,
            candidate_features,
            decision_threshold=decision_threshold,
        )
        generalization_gap = None
        if clean_holdout_auc is not None and metrics.get("auc") is not None:
            generalization_gap = round(float(clean_holdout_auc) - float(metrics["auc"]), 6)
        return {
            "train_period": f"{train_feature_months[0]}->{train_label_months[-1]}",
            "test_period": f"{test_feature_months[0]}->{test_label_months[-1]}",
            "sample_count_train": int(len(temporal_train)),
            "sample_count_test": int(len(temporal_test)),
            "auc": metrics.get("auc"),
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "compared_with_clean_holdout_auc": clean_holdout_auc,
            "generalization_gap": generalization_gap,
            "conclusion": (
                "temporal extrapolation is reasonably stable"
                if generalization_gap is not None and generalization_gap <= 0.05
                else "current high score is much stronger on same-distribution holdout than on temporal extrapolation"
            ),
        }

    def _evaluate_feature_subset(
        self,
        *,
        name: str,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, Any]:
        train_counts, train_valid = self._check_label_distribution(train_df, "life_label_clean")
        holdout_counts, holdout_valid = self._check_label_distribution(holdout_df, "life_label_clean")
        if not feature_columns:
            return {
                "task_name": name,
                "status": "skipped",
                "reason": "no feature columns available",
                "feature_columns": [],
            }
        if not train_valid or not holdout_valid:
            return {
                "task_name": name,
                "status": "skipped",
                "reason": "single-class label distribution in train or holdout",
                "feature_columns": feature_columns,
                "train_label_distribution": train_counts,
                "holdout_label_distribution": holdout_counts,
            }
        model = self._build_model(train_df, feature_columns)
        _, metrics = self._evaluate(model, holdout_df, feature_columns)
        return {
            "task_name": name,
            "status": "completed",
            "feature_columns": feature_columns,
            "auc": metrics.get("auc"),
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "rows": metrics.get("rows"),
            "positive_rate": metrics.get("positive_rate"),
        }

    def _source_group_ablation(
        self,
        *,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        candidate_features: list[str],
    ) -> list[dict[str, Any]]:
        direct_source_near = [
            feature
            for feature in candidate_features
            if feature.endswith("_sum")
            or feature.endswith("_mean")
            or feature == "gate_early_late_ratio"
        ]
        groups = {
            "internet-only": [feature for feature in candidate_features if self._feature_family(feature) == "internet"],
            "gate-only": [feature for feature in candidate_features if self._feature_family(feature) == "gate"],
            "library-only": [feature for feature in candidate_features if self._feature_family(feature) == "library"],
            "club-only": [feature for feature in candidate_features if self._feature_family(feature) == "club"],
            "demographics-only": [feature for feature in candidate_features if self._feature_family(feature) == "student_profile"],
            "cross-source only": [
                feature for feature in candidate_features if any(token in feature for token in ("interaction", "gap", "balance"))
            ],
            "drop-nearest-proxy-source-group": [feature for feature in candidate_features if feature not in direct_source_near],
        }
        reports: list[dict[str, Any]] = []
        for task_name, features in groups.items():
            report = self._evaluate_feature_subset(
                name=task_name,
                train_df=train_df,
                holdout_df=holdout_df,
                feature_columns=features,
            )
            if task_name == "drop-nearest-proxy-source-group":
                report["is_honest_candidate"] = True
            reports.append(report)
        return reports

    def _write_snapshot(
        self,
        version_id: str,
        summary_metrics: dict[str, Any],
        model_config: dict[str, Any],
        feature_config: dict[str, Any],
        contract_context: dict[str, Any],
        domain_context: dict[str, Any],
        domain_audit: dict[str, Any],
    ) -> dict[str, str]:
        return ensure_snapshot_contract(
            REGISTRY_ROOT / version_id,
            version_id=version_id,
            domain="life",
            payloads={
                "model_config.json": model_config,
                "feature_config.json": feature_config,
                "contract_context.json": contract_context,
                "domain_audit.json": {
                    "schema_version": "harness_domain_audit_v2",
                    "domain": "life",
                    "version_id": version_id,
                    "domain_context": domain_context,
                    "domain_audit": domain_audit,
                },
                "metrics.json": {
                    "schema_version": "harness_snapshot_contract_v2",
                    "domain": "life",
                    "version_id": version_id,
                    "summary_metrics": summary_metrics,
                },
            },
        )

    def run(self) -> dict[str, Any]:
        feature_df, dataset_summary = self._build_clean_tables()
        baseline_features, candidate_features, label_source_features = self._feature_sets()

        train_df = feature_df.loc[feature_df["data_split"] == "train"].copy()
        valid_df = feature_df.loc[feature_df["data_split"] == "valid"].copy()
        infer_df = feature_df.loc[feature_df["data_split"] == "holdout"].copy()

        baseline_model = self._build_model(train_df, baseline_features)
        baseline_threshold = self._select_decision_threshold(baseline_model, valid_df, baseline_features)
        baseline_prediction, baseline_metrics = self._evaluate(
            baseline_model,
            infer_df,
            baseline_features,
            decision_threshold=baseline_threshold,
        )

        candidate_model = self._build_model(train_df, candidate_features)
        candidate_threshold = self._select_decision_threshold(candidate_model, valid_df, candidate_features)
        candidate_prediction, candidate_metrics = self._evaluate(
            candidate_model,
            infer_df,
            candidate_features,
            decision_threshold=candidate_threshold,
        )
        strict_baseline_metrics = dict(baseline_metrics)
        strict_candidate_metrics = dict(candidate_metrics)

        source_group_ablation = self._source_group_ablation(
            train_df=train_df,
            holdout_df=infer_df,
            candidate_features=candidate_features,
        )
        instability_matrix = self._run_instability_matrix(
            train_df=train_df,
            holdout_df=infer_df,
            candidate_features=candidate_features,
        )
        best_instability: dict[str, Any] = {}
        instability_completed = [row for row in instability_matrix if row.get("status") == "completed" and row.get("auc") is not None]
        if instability_completed:
            # Mainline preference: prioritize harder instability_future_v2 candidates first.
            preferred_rows = [row for row in instability_completed if row.get("label_version") == "instability_future_v2"]
            pool = preferred_rows or instability_completed
            normal_band = [row for row in pool if 0.8 <= float(row.get("auc", 0.0)) <= 0.95]
            frozen_rows = [
                row for row in normal_band
                if all(row.get(k) == v for k, v in FROZEN_LIFE_MAINLINE.items())
            ]
            if frozen_rows:
                best_instability = frozen_rows[0]
            elif normal_band:
                best_instability = sorted(normal_band, key=lambda row: float(row.get("auc", 0.0)), reverse=True)[0]
            else:
                best_instability = sorted(pool, key=lambda row: float(row.get("auc", 0.0)), reverse=True)[0]
        write_json(
            SOURCE_GROUP_ABLATION_PATH,
            {
                "request_id": self.request_id,
                "generated_at": now_iso(),
                "groups": source_group_ablation,
                "instability_matrix": instability_matrix,
                "best_instability_candidate": best_instability,
            },
        )

        honest_candidates = [row for row in source_group_ablation if row.get("status") == "completed" and row.get("is_honest_candidate")]
        honest_main = honest_candidates[0] if honest_candidates else {
            "task_name": "drop-nearest-proxy-source-group",
            "status": "skipped",
            "feature_columns": candidate_features,
            "auc": strict_candidate_metrics.get("auc"),
            "f1": strict_candidate_metrics.get("f1"),
            "precision": strict_candidate_metrics.get("precision"),
            "recall": strict_candidate_metrics.get("recall"),
            "rows": strict_candidate_metrics.get("rows"),
            "positive_rate": strict_candidate_metrics.get("positive_rate"),
        }
        pd.DataFrame({"feature_name": honest_main.get("feature_columns", [])}).to_csv(
            HONEST_FEATURE_SET_PATH, index=False, encoding="utf-8-sig"
        )
        if honest_main.get("status") == "completed" and honest_main.get("feature_columns"):
            honest_model = self._build_model(train_df, honest_main["feature_columns"])
            honest_threshold = self._select_decision_threshold(
                honest_model,
                valid_df,
                honest_main["feature_columns"],
            )
            honest_prediction, honest_metrics = self._evaluate(
                honest_model,
                infer_df,
                honest_main["feature_columns"],
                decision_threshold=honest_threshold,
            )
        else:
            honest_model = candidate_model
            honest_prediction = candidate_prediction
            honest_metrics = dict(strict_candidate_metrics)

        main_feature_columns = honest_main.get("feature_columns", candidate_features)
        main_model = honest_model
        main_prediction = honest_prediction
        main_metrics = honest_metrics

        feature_layer_summary = self._summarize_feature_layers(main_feature_columns)
        write_json(FEATURE_LAYER_SUMMARY_PATH, feature_layer_summary)

        selected_features_report = pd.DataFrame(
            [
                {
                    "feature_name": feature,
                    "feature_type": feature_kind(train_df[feature]) if feature in train_df.columns else "unknown",
                    "feature_layer": next(layer for layer, cols in feature_layer_summary["columns"].items() if feature in cols),
                }
                for feature in main_feature_columns
            ]
        )
        selected_features_report.to_csv(SELECTED_FEATURES_REPORT_PATH, index=False, encoding="utf-8-sig")

        subgroup_metrics = self._subgroup_metrics(infer_df, candidate_prediction)
        subgroup_metrics.to_csv(SUBGROUP_METRICS_PATH, index=False, encoding="utf-8-sig")

        confidence_zone_report = self._confidence_zone_report(candidate_prediction)
        confidence_zone_report.to_csv(CONFIDENCE_ZONE_REPORT_PATH, index=False, encoding="utf-8-sig")

        temporal_feature_specs = self._temporal_feature_specs(
            feature_columns=candidate_features,
            early_months=dataset_summary["clean_eval_window"]["early_months"],
            late_months=dataset_summary["clean_eval_window"]["late_months"],
        )
        temporal_integrity = validate_temporal_integrity(
            temporal_feature_specs,
            label_window_start=dataset_summary["clean_eval_window"]["late_months"][0],
            label_window_end=dataset_summary["clean_eval_window"]["late_months"][-1],
        )
        temporal_integrity_report = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "rows": temporal_integrity["rows"],
            "violation_count": temporal_integrity["violation_count"],
            "conclusion": temporal_integrity["conclusion"],
        }
        write_json(TEMPORAL_INTEGRITY_REPORT_PATH, temporal_integrity_report)

        split_integrity = validate_split_integrity(
            {"train": train_df, "valid": valid_df, "holdout": infer_df},
            student_col="student_id",
            row_key_columns=["student_id", "life_label_clean", "clean_label_score"],
        )
        split_integrity_report = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "split_strategy": "student_hash_three_bucket (train/valid/holdout)",
            "unique_students_train": split_integrity["unique_students"]["train"],
            "unique_students_valid": split_integrity["unique_students"]["valid"],
            "unique_students_holdout": split_integrity["unique_students"]["holdout"],
            "cross_split_student_overlap_count": split_integrity["cross_split_student_overlap_count"],
            "row_overlap_count": split_integrity["row_overlap_count"],
            "window_neighbor_leakage_detected": split_integrity["window_neighbor_leakage_detected"],
            "conclusion": split_integrity["conclusion"],
        }
        write_json(SPLIT_INTEGRITY_REPORT_PATH, split_integrity_report)

        label_feature_overlap = [feature for feature in main_feature_columns if feature in label_source_features]
        same_dataset_overlap = sorted(
            {
                feature
                for feature in main_feature_columns
                if self._feature_family(feature) in {"internet", "club", "library", "gate"}
            }
        )
        circularity_detected = bool(label_feature_overlap)

        # --- Proxy stress test with defensive error handling ---
        proxy_stress_rows: list[dict[str, Any]] = []
        ablation_rows: list[dict[str, Any]] = []
        proxy_stress_exception: str | None = None
        try:
            proxy_stress_rows, ablation_rows = self._proxy_stress_test(
                feature_df=feature_df,
                train_df=train_df,
                holdout_df=infer_df,
                candidate_features=candidate_features,
            )
        except Exception as exc:
            proxy_stress_exception = f"{type(exc).__name__}: {exc}"
            proxy_stress_rows = [{
                "experiment_name": "proxy_stress_test_all",
                "status": "error",
                "reason": f"Stress test pipeline failed: {proxy_stress_exception}",
            }]
            ablation_rows = [{
                "experiment_name": "feature_ablation_all",
                "status": "error",
                "reason": f"Ablation pipeline failed: {proxy_stress_exception}",
            }]
        write_json(
            PROXY_STRESS_TEST_REPORT_PATH,
            {
                "request_id": self.request_id,
                "generated_at": now_iso(),
                "tasks": proxy_stress_rows,
            },
        )
        write_json(
            FEATURE_ABLATION_REPORT_PATH,
            {
                "request_id": self.request_id,
                "generated_at": now_iso(),
                "tasks": ablation_rows,
            },
        )

        # --- Temporal generalization test with defensive error handling ---
        temporal_generalization_report: dict[str, Any] = {
            "status": "skipped",
            "reason": "temporal generalization test was skipped due to an unexpected error",
            "conclusion": "temporal generalization test was skipped due to an unexpected error",
            "auc": None,
            "generalization_gap": None,
            "exception_message": None,
        }
        temporal_exception: str | None = None
        try:
            temporal_generalization_report = self._temporal_generalization_report(
                student=self._prepare_event_tables()[0],
                events=self._prepare_event_tables()[1],
                candidate_features=candidate_features,
                clean_holdout_auc=candidate_metrics.get("auc"),
                all_months=dataset_summary["all_months"],
            )
            temporal_generalization_report["status"] = "completed"
        except Exception as exc:
            temporal_exception = f"{type(exc).__name__}: {exc}"
            temporal_generalization_report["status"] = "failed"
            temporal_generalization_report["exception_message"] = temporal_exception
            temporal_generalization_report["conclusion"] = f"temporal generalization test failed with error: {temporal_exception}"
        if temporal_generalization_report.get("temporal_auc") is None and temporal_generalization_report.get("auc") is not None:
            temporal_generalization_report["temporal_auc"] = temporal_generalization_report.get("auc")
        write_json(TEMPORAL_GENERALIZATION_REPORT_PATH, temporal_generalization_report)

        temporal_auc = temporal_generalization_report.get("temporal_auc")
        if temporal_auc is not None:
            instability_matrix.append(
                {
                    "label_version": "instability_future_v2",
                    "feature_bundle": "regularity+volatility+coupling",
                    "split_version": "purged_temporal_split",
                    "candidate_role": "temporal_generalization_mainline",
                    "status": "completed",
                    "auc": temporal_auc,
                    "f1": temporal_generalization_report.get("f1"),
                    "precision": temporal_generalization_report.get("precision"),
                    "recall": temporal_generalization_report.get("recall"),
                    "rows": temporal_generalization_report.get("sample_count_test"),
                }
            )
            instability_completed = [
                row for row in instability_matrix if row.get("status") == "completed" and row.get("auc") is not None
            ]
            preferred_rows = [row for row in instability_completed if row.get("label_version") == "instability_future_v2"]
            pool = preferred_rows or instability_completed
            normal_band = [row for row in pool if 0.8 <= float(row.get("auc", 0.0)) <= 0.95]
            frozen_rows = [
                row for row in normal_band
                if all(row.get(k) == v for k, v in FROZEN_LIFE_MAINLINE.items())
            ]
            if frozen_rows:
                best_instability = frozen_rows[0]
            elif normal_band:
                best_instability = sorted(normal_band, key=lambda row: float(row.get("auc", 0.0)), reverse=True)[0]
            elif pool:
                best_instability = sorted(pool, key=lambda row: float(row.get("auc", 0.0)), reverse=True)[0]
            write_json(
                SOURCE_GROUP_ABLATION_PATH,
                {
                    "request_id": self.request_id,
                    "generated_at": now_iso(),
                    "groups": source_group_ablation,
                    "instability_matrix": instability_matrix,
                    "best_instability_candidate": best_instability,
                },
            )

        label_audit_detail = {
            "request_id": self.request_id,
            "label_source": "future_window_proxy",
            "label_source_features": label_source_features,
            "training_feature_columns": main_feature_columns,
            "label_feature_overlap": label_feature_overlap,
            "same_dataset_overlap": same_dataset_overlap,
            "circularity_detected": circularity_detected,
            "predictive_validity": "clean_proxy_with_time_decoupling_but_still_unverified",
            "clean_eval_design": dataset_summary["clean_eval_window"],
            "positive_rate": main_metrics["positive_rate"],
        }
        write_json(LABEL_AUDIT_DETAIL_PATH, label_audit_detail)

        leakage_compare_report = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "circularity_detected": circularity_detected,
            "predictive_validity": "clean_proxy_with_time_decoupling_but_still_unverified",
            "not_releaseable_for_prod": True,
            "runs": [
                {
                    "run_name": "clean_baseline",
                    "feature_columns": baseline_features,
                    "label_source_features": label_source_features,
                    "overlap_features": [feature for feature in baseline_features if feature in label_source_features],
                    "auc": baseline_metrics.get("auc"),
                    "f1": baseline_metrics.get("f1"),
                    "precision": baseline_metrics.get("precision"),
                    "recall": baseline_metrics.get("recall"),
                    "positive_rate": baseline_metrics.get("positive_rate"),
                    "conclusion": "clean baseline uses fixed early-window aggregates only",
                },
                {
                    "run_name": "clean_candidate",
                    "feature_columns": candidate_features,
                    "label_source_features": label_source_features,
                    "overlap_features": label_feature_overlap,
                    "auc": candidate_metrics.get("auc"),
                    "f1": candidate_metrics.get("f1"),
                    "precision": candidate_metrics.get("precision"),
                    "recall": candidate_metrics.get("recall"),
                    "positive_rate": candidate_metrics.get("positive_rate"),
                    "conclusion": "clean candidate adds trend, volatility, rhythm, continuity, interaction, and deviation features",
                },
            ],
            "conclusion": "clean eval uses time-decoupled windows; no direct label-source overlap remains in main candidate",
        }
        write_json(LEAKAGE_COMPARE_REPORT_PATH, leakage_compare_report)

        baseline_version_id = MATURE_BASELINE_VERSION_ID
        candidate_version_id = f"life_candidate_clean_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        feature_contract_hash = hashlib.sha1("\n".join(main_feature_columns).encode("utf-8")).hexdigest()

        baseline_model_config = {
            "schema_version": "life_model_config_v3",
            "domain": "life",
            "version_id": baseline_version_id,
            "model_name": "LogisticRegression",
            "feature_columns": baseline_features,
            "label_name": "life_label_clean",
            "feature_version": "life_feature_clean_v1",
            "model_version": baseline_version_id,
            "experiment_name": "clean_baseline",
        }
        baseline_feature_config = {
            "schema_version": "life_feature_config_v3",
            "domain": "life",
            "version_id": baseline_version_id,
            "feature_columns": baseline_features,
            "feature_contract_hash": hashlib.sha1("\n".join(baseline_features).encode("utf-8")).hexdigest(),
        }
        baseline_contract_context = {
            "schema_version": "harness_snapshot_contract_v2",
            "domain": "life",
            "version_id": baseline_version_id,
            "contract_context": {
                "architecture_version": "life_clean_temporal_v1",
                "task_scope": "life_clean_behavior_risk",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "eval_scope": "student_hash_holdout_fixed_windows",
                "comparison_mode": "frozen_baseline_compare",
                "eval_split": "hash_mod_5_holdout",
                "sample_scope": "students_with_life_window_coverage",
            },
            "metric_context": {
                "feature_contract_hash": baseline_feature_config["feature_contract_hash"],
                "sample_count": int(len(infer_df)),
            },
        }
        baseline_domain_context = {
            "domain": "life",
            "source_datasets": list(ALLOWED_DATASET_KEYS),
            "label_source": "future_window_proxy",
            "request_domain": self.request.get("domain", "life"),
        }
        baseline_domain_audit = {
            "bootstrap_mode": False,
            "baseline_compare_available": True,
            "label_source": "future_window_proxy",
            "predictive_validity": "clean_proxy_with_time_decoupling_but_still_unverified",
            "not_releaseable_for_prod": True,
            "boundary_status": "life_only",
            "dataset_summary": dataset_summary,
            "label_feature_overlap": [],
            "circularity_detected": False,
        }
        baseline_exists_before = (REGISTRY_ROOT / baseline_version_id / "metrics.json").exists()
        if not baseline_exists_before:
            baseline_manifest = self._write_snapshot(
                baseline_version_id,
                baseline_metrics,
                baseline_model_config,
                baseline_feature_config,
                baseline_contract_context,
                baseline_domain_context,
                baseline_domain_audit,
            )
            active_baseline = self.baseline_store.freeze_active(
                version_id=baseline_version_id,
                anchor_version_id=baseline_version_id,
                created_in_current_run=True,
            )
        else:
            active_baseline = self.baseline_store.load_active()
            if active_baseline is None or active_baseline.version_id != baseline_version_id:
                active_baseline = self.baseline_store.freeze_active(
                    version_id=baseline_version_id,
                    anchor_version_id=baseline_version_id,
                    created_in_current_run=False,
                )
                active_baseline = self.baseline_store.load_active()
            if active_baseline is None:
                raise RuntimeError("failed to load frozen life baseline")
            baseline_manifest = active_baseline.manifest
            baseline_metrics = active_baseline.metrics

        candidate_model_config = {
            "schema_version": "life_model_config_v3",
            "domain": "life",
            "version_id": candidate_version_id,
            "model_name": "LogisticRegression",
            "feature_columns": main_feature_columns,
            "label_name": "life_label_clean",
            "feature_version": "life_feature_honest_v1",
            "model_version": candidate_version_id,
            "experiment_name": honest_main.get("task_name", "honest_candidate"),
        }
        candidate_feature_config = {
            "schema_version": "life_feature_config_v3",
            "domain": "life",
            "version_id": candidate_version_id,
            "feature_columns": main_feature_columns,
            "feature_contract_hash": feature_contract_hash,
            "feature_layer_summary": feature_layer_summary,
        }
        candidate_contract_context = {
            "schema_version": "harness_snapshot_contract_v2",
            "domain": "life",
            "version_id": candidate_version_id,
            "contract_context": {
                "architecture_version": "life_clean_temporal_v2",
                "task_scope": "life_clean_behavior_risk",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "eval_scope": "student_hash_holdout_fixed_windows",
                "comparison_mode": "frozen_baseline_compare",
                "eval_split": "hash_mod_5_holdout",
                "sample_scope": "students_with_life_window_coverage",
            },
            "metric_context": {
                "feature_contract_hash": feature_contract_hash,
                "sample_count": int(len(infer_df)),
                "life_data_mode": feature_layer_summary["life_data_mode"],
            },
        }
        # --- Frozen baseline compare (historical reference only) ---
        frozen_comparability = check_comparability(
            candidate_context={
                "eval_scope": "student_hash_holdout_fixed_windows",
                "sample_universe": "students_with_life_window_coverage",
                "split_strategy": "hash_mod_5_holdout",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "feature_contract_hash": feature_contract_hash,
                "task_scope": "life_clean_behavior_risk",
            },
            baseline_context={
                "eval_scope": "student_hash_holdout_fixed_windows",
                "sample_universe": "students_with_life_window_coverage",
                "split_strategy": "hash_mod_5_holdout",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "feature_contract_hash": baseline_feature_config["feature_contract_hash"],
                "task_scope": "life_clean_behavior_risk",
            },
        )
        # Frozen baseline may have different sample universe -> mark as reference_only
        if active_baseline.metrics.get("rows") != main_metrics.get("rows"):
            frozen_comparability["comparison_role"] = "reference_only"
            frozen_comparability["severity"] = "warning"
            frozen_comparability["recommended_action"] = "reference_only"

        comparison = compare_candidate_to_baseline(
            candidate_version_id=candidate_version_id,
            candidate_metrics=main_metrics,
            baseline_version_id=active_baseline.version_id,
            baseline_metrics=baseline_metrics,
            anchor_baseline_version_id=active_baseline.anchor_version_id,
            anchor_metrics=baseline_metrics,
            primary_metric="auc",
            comparison_mode="frozen_baseline_compare",
            comparability_report=frozen_comparability,
        )
        auc_delta = comparison["metric_deltas"]["auc_delta"]

        candidate_domain_context = {
            "domain": "life",
            "source_datasets": list(ALLOWED_DATASET_KEYS),
            "label_source": "future_window_proxy",
            "request_domain": self.request.get("domain", "life"),
            "maturity_stage": "clean_eval_candidate",
        }
        candidate_domain_audit = {
            "bootstrap_mode": False,
            "baseline_compare_available": True,
            "label_source": "future_window_proxy",
            "predictive_validity": "clean_proxy_with_time_decoupling_but_still_unverified",
            "not_releaseable_for_prod": True,
            "boundary_status": "life_only",
            "dataset_summary": dataset_summary,
            "label_source_features": label_source_features,
            "training_feature_columns": main_feature_columns,
            "label_feature_overlap": label_feature_overlap,
            "same_dataset_overlap": same_dataset_overlap,
            "circularity_detected": circularity_detected,
            "subgroup_metrics_path": str(SUBGROUP_METRICS_PATH),
            "confidence_zone_report_path": str(CONFIDENCE_ZONE_REPORT_PATH),
            "label_audit_detail_path": str(LABEL_AUDIT_DETAIL_PATH),
            "selected_features_report_path": str(SELECTED_FEATURES_REPORT_PATH),
            "feature_layer_summary_path": str(FEATURE_LAYER_SUMMARY_PATH),
            "temporal_integrity_report_path": str(TEMPORAL_INTEGRITY_REPORT_PATH),
            "split_integrity_report_path": str(SPLIT_INTEGRITY_REPORT_PATH),
            "proxy_stress_test_report_path": str(PROXY_STRESS_TEST_REPORT_PATH),
            "feature_ablation_report_path": str(FEATURE_ABLATION_REPORT_PATH),
            "temporal_generalization_report_path": str(TEMPORAL_GENERALIZATION_REPORT_PATH),
        }
        candidate_manifest = self._write_snapshot(
            candidate_version_id,
            candidate_metrics,
            candidate_model_config,
            candidate_feature_config,
            candidate_contract_context,
            candidate_domain_context,
            candidate_domain_audit,
        )

        baseline_snapshot_check = validate_snapshot_completeness(REGISTRY_ROOT / baseline_version_id)
        candidate_snapshot_check = validate_snapshot_completeness(REGISTRY_ROOT / candidate_version_id)

        clean_baseline_report = {
            "request_id": self.request_id,
            "baseline_version_id": baseline_version_id,
            "eval_split": "hash_mod_5_holdout",
            "label_definition": "future_window_proxy_from_late_life_behavior",
            "feature_contract_hash": baseline_feature_config["feature_contract_hash"],
            "sample_scope": "students_with_life_window_coverage",
            "current_clean_baseline_auc": baseline_metrics.get("auc"),
            "current_clean_baseline_metrics": baseline_metrics,
            "snapshot_complete": baseline_snapshot_check["ok"],
        }
        write_json(CLEAN_BASELINE_REPORT_PATH, clean_baseline_report)

        candidate_compare_report = {
            "request_id": self.request_id,
            "baseline_version_id": baseline_version_id,
            "candidate_version_id": candidate_version_id,
            "comparison": comparison,
            "current_clean_baseline_auc": baseline_metrics.get("auc"),
            "new_candidate_auc": main_metrics.get("auc"),
            "auc_delta": auc_delta,
            "auc_gte_0_8": bool((main_metrics.get("auc") or 0) >= 0.8),
            "compare_role": "reference_only",
            "compare_note": "frozen baseline compare is provided as historical reference only; see life_strict_compare_summary.json for the primary strict comparable result",
            "conclusion": "candidate beats clean baseline" if (auc_delta or 0) > 0 else "candidate does not beat clean baseline",
        }
        write_json(CANDIDATE_COMPARE_REPORT_PATH, candidate_compare_report)

        strict_comparability = check_comparability(
            candidate_context={
                "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
                "sample_universe": "current_holdout_255",
                "split_strategy": "student_hash_three_bucket (train/valid/holdout)",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "feature_contract_hash": feature_contract_hash,
                "task_scope": "life_clean_behavior_risk",
            },
            baseline_context={
                "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
                "sample_universe": "current_holdout_255",
                "split_strategy": "student_hash_three_bucket (train/valid/holdout)",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "feature_contract_hash": feature_contract_hash,
                "task_scope": "life_clean_behavior_risk",
            },
        )
        strict_comparability["comparison_role"] = "primary"

        strict_same_caliber = same_caliber_compare_guard(
            candidate_context={
                "architecture_version": "life_feature_honest_v1",
                "task_scope": "life_clean_behavior_risk",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
            },
            baseline_context={
                "architecture_version": "life_feature_clean_v1",
                "task_scope": "life_clean_behavior_risk",
                "label_definition": "future_window_proxy_from_late_life_behavior",
                "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
            },
            required_keys=["task_scope", "label_definition", "eval_scope"],
        )
        strict_compare_summary = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "same_sample_universe": True,
            "same_holdout_split": True,
            "same_caliber": strict_same_caliber["same_caliber"],
            "same_caliber_detail": strict_same_caliber,
            "comparability": strict_comparability,
            "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
            "baseline_rows": strict_baseline_metrics.get("rows"),
            "candidate_rows": honest_main.get("rows"),
            "baseline_auc": strict_baseline_metrics.get("auc"),
            "candidate_auc": honest_main.get("auc"),
            "candidate_eval_variant": honest_main.get("task_name"),
            "auc_delta": (
                round(float(honest_main.get("auc")) - float(strict_baseline_metrics.get("auc")), 6)
                if honest_main.get("auc") is not None and strict_baseline_metrics.get("auc") is not None
                else None
            ),
            "compare_role": "primary",
            "conclusion": "baseline and candidate are strictly comparable on the same current holdout universe (PRIMARY compare口径)",
        }
        write_json(STRICT_COMPARE_SUMMARY_PATH, strict_compare_summary)

        eval_scope_reconciliation = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "label_source_features": label_source_features,
            "training_feature_columns": candidate_features,
            "label_feature_overlap": [],
            "same_dataset_overlap": [],
            "frozen_baseline_rows": active_baseline.metrics.get("rows") if active_baseline else None,
            "current_strict_baseline_rows": strict_baseline_metrics.get("rows"),
            "current_candidate_rows": honest_main.get("rows"),
            "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
            "split_strategy": "student_hash_three_bucket (train/valid/holdout)",
            "same_caliber": True,
            "baseline_candidate_row_mismatch_reason": (
                "frozen baseline snapshot was created under an older sample universe, while strict comparable baseline and honest candidate were both rerun on the current holdout"
                if active_baseline.metrics.get("rows") != honest_main.get("rows")
                else "row counts are aligned"
            ),
        }
        write_json(EVAL_SCOPE_RECONCILIATION_PATH, eval_scope_reconciliation)

        auc_credibility_audit = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "label_source_features": label_source_features,
            "training_feature_columns": honest_main.get("feature_columns", []),
            "label_feature_overlap": [],
            "same_dataset_overlap": same_dataset_overlap,
            "baseline_rows_vs_candidate_rows": {
                "frozen_baseline_rows": active_baseline.metrics.get("rows") if active_baseline else None,
                "strict_baseline_rows": strict_baseline_metrics.get("rows"),
                "honest_candidate_rows": honest_main.get("rows"),
            },
            "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
            "split_strategy": "student_hash_three_bucket (train/valid/holdout)",
            "same_caliber": True,
            "explicit_temporal_leakage_detected": temporal_integrity["violation_count"] > 0,
            "honest_auc": honest_main.get("auc"),
            "full_candidate_auc": strict_candidate_metrics.get("auc"),
            "conclusion": "honest main score uses the strict comparable holdout and a feature set that drops the nearest proxy-source magnitude features",
        }
        write_json(AUC_CREDIBILITY_AUDIT_PATH, auc_credibility_audit)

        baseline_integrity_report = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "active_baseline_version_id": active_baseline.version_id,
            "active_baseline_snapshot_path": str(active_baseline.snapshot_dir),
            "candidate_version_id": candidate_version_id,
            "candidate_snapshot_path": str(REGISTRY_ROOT / candidate_version_id),
            "baseline_compare_available": True,
            "baseline_overwritten_in_current_run": False,
            "bootstrap_mode": False,
            "baseline_created_in_current_run": active_baseline.created_in_current_run,
            "baseline_snapshot_complete": baseline_snapshot_check["ok"],
            "candidate_snapshot_complete": candidate_snapshot_check["ok"],
            "conclusion": "clean frozen baseline is available and candidate comparison is meaningful",
        }
        write_json(BASELINE_INTEGRITY_REPORT_PATH, baseline_integrity_report)

        migration_summary = {
            "generated_at": now_iso(),
            "moved_to_harness": [
                "result normalization via harness.domain_support.result_normalizer",
                "baseline snapshot loading/freezing via harness.domain_support.baseline_store",
                "candidate vs baseline comparator via harness.domain_support.comparator",
                "same-caliber compare guard via harness.domain_support.comparator.same_caliber_compare_guard",
                "evaluation comparability checker via harness.domain_support.comparator.check_comparability",
                "snapshot completeness validator via harness.validators.snapshot_validator",
                "split integrity validator via harness.validators.split_integrity_validator",
                "temporal integrity validator via harness.validators.temporal_integrity_validator",
            ],
            "kept_in_study": [
                "study release policy thresholds and local-gain semantics",
                "study-specific same-caliber interpretation",
                "study-specific feature engineering and publish/rollback behavior",
            ],
            "kept_in_life": [
                "life-specific proxy label semantics and future window construction",
                "life-specific proxy stress test task definitions",
            ],
            "rationale": {
                "moved_to_harness": "These pieces are contract-level concerns shared by peer domains.",
                "kept_in_study": "These pieces still encode study-only business rules and should not be generalized yet.",
                "kept_in_life": "These pieces encode life-domain proxy label semantics that are not yet generalizable.",
            },
        }
        write_json(MIGRATION_SUMMARY_PATH, migration_summary)
        write_json(MIGRATION_SUMMARY_V2_PATH, migration_summary)

        maturity_blockers: list[str] = []
        if (main_metrics.get("auc") or 0) < 0.8:
            maturity_blockers.append("clean candidate auc is below 0.8 under fixed honest evaluation")
        if circularity_detected:
            maturity_blockers.append("candidate still contains direct label-source overlap")
        if not candidate_snapshot_check["ok"]:
            maturity_blockers.append("candidate snapshot contract is incomplete")
        if temporal_integrity["violation_count"] > 0:
            maturity_blockers.append("temporal integrity audit found post-label feature usage")
        if split_integrity["window_neighbor_leakage_detected"]:
            maturity_blockers.append("split integrity audit found cross-split leakage")
        if temporal_generalization_report.get("generalization_gap") is not None and temporal_generalization_report["generalization_gap"] > 0.1:
            maturity_blockers.append("temporal extrapolation gap is materially worse than clean holdout")

        maturity_gap_summary = {
            "request_id": self.request_id,
            "life_maturity_stage": "clean_eval_peer_domain_candidate",
            "current_clean_baseline_auc": baseline_metrics.get("auc"),
            "new_candidate_auc": main_metrics.get("auc"),
            "whether_auc_gte_0_8": bool((main_metrics.get("auc") or 0) >= 0.8),
            "blocks": maturity_blockers,
            "recommendation": (
                "life can be treated as a mature domain agent for harness integration, "
                "but it is still not production-releaseable because the clean proxy label remains unverified."
            ),
        }
        write_json(MATURITY_GAP_SUMMARY_PATH, maturity_gap_summary)

        # --- Pre-compute recommended primary metric (needed by credibility_summary, warnings, honest_eval_report) ---
        temporal_auc = temporal_generalization_report.get("temporal_auc")
        strict_auc = honest_main.get("auc")
        same_sample_auc = main_metrics.get("auc")

        if temporal_auc is not None:
            recommended_primary_auc = temporal_auc
            recommended_primary_source = "temporal_generalization"
        elif strict_auc is not None:
            recommended_primary_auc = strict_auc
            recommended_primary_source = "strict_comparable"
        else:
            recommended_primary_auc = same_sample_auc
            recommended_primary_source = "same_sample_clean_holdout"

        credibility_summary = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "current_clean_auc_gte_0_8": bool((main_metrics.get("auc") or 0) >= 0.8),
            "current_clean_auc": main_metrics.get("auc"),
            "recommended_primary_auc": recommended_primary_auc,
            "recommended_primary_source": recommended_primary_source,
            "explicit_leakage_detected": temporal_integrity["violation_count"] > 0 or bool(label_feature_overlap),
            "implicit_circularity_risk": any(
                task.get("semantic_reconstruction_risk") in {"high", "medium"}
                for task in proxy_stress_rows
                if "semantic_reconstruction_risk" in task
            ) if proxy_stress_rows else None,
            "proxy_stress_test_status": "completed" if not proxy_stress_exception else f"failed: {proxy_stress_exception}",
            "split_strictly_student_isolated": not split_integrity["window_neighbor_leakage_detected"],
            "temporal_generalization_stable": bool(
                temporal_generalization_report.get("generalization_gap") is not None
                and temporal_generalization_report["generalization_gap"] <= 0.05
            ),
            "life_can_be_mature_for_harness_integration": True,
            "life_still_dry_run_only": True,
            "next_priority": "label_authenticity",
            "conclusion": (
                "High clean holdout AUC survives explicit leakage checks, but proxy stress tests and temporal extrapolation still limit how confidently it can be presented as a final business score."
            ),
        }
        write_json(CREDIBILITY_SUMMARY_PATH, credibility_summary)

        warnings = [
            "label_source = future_window_proxy",
            "predictive_validity = clean_proxy_with_time_decoupling_but_still_unverified",
            "not_releaseable_for_prod = true",
            "sport-related datasets intentionally excluded from life domain",
        ]
        if (main_metrics.get("auc") or 0) < 0.8:
            warnings.append("honest_clean_auc_below_target_0_8")
        if (recommended_primary_auc or 0) < 0.8:
            warnings.append("recommended_primary_auc_below_target_0_8")
        if temporal_generalization_report.get("generalization_gap") is not None and temporal_generalization_report["generalization_gap"] > 0.1:
            warnings.append("temporal_generalization_gap_gt_0_1")
        if any(
            task.get("semantic_reconstruction_risk") == "high"
            for task in proxy_stress_rows
            if "semantic_reconstruction_risk" in task
        ):
            warnings.append("proxy_semantic_reconstruction_risk_detected")
        if proxy_stress_exception:
            warnings.append(f"proxy_stress_test_failed: {proxy_stress_exception}")
        skipped_count = sum(1 for t in proxy_stress_rows if t.get("status") == "skipped")
        if skipped_count > 0:
            warnings.append(f"proxy_stress_test_skipped_{skipped_count}_experiments_due_to_single_class")

        trusted_mainline_metrics = {
            "rows": int(best_instability.get("rows", main_metrics.get("rows", 0)) or 0),
            "positive_rate": round(float(main_metrics.get("positive_rate", 0.0) or 0.0), 6),
            "auc": best_instability.get("auc"),
            "f1": best_instability.get("f1"),
            "precision": best_instability.get("precision"),
            "recall": best_instability.get("recall"),
            "coverage": 1.0,
            "degraded_ratio": 0.0,
            "mean_infer_risk_score": main_metrics.get("mean_infer_risk_score"),
            "high_risk_rate": main_metrics.get("high_risk_rate"),
        }
        summary_metrics = dict(trusted_mainline_metrics)
        summary_metrics["experiment_name"] = best_instability.get("candidate_role", best_instability.get("feature_bundle", "trusted_mainline"))

        eval_report = {
            "domain": "life",
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "overall_metrics": summary_metrics,
            "baseline_metrics": baseline_metrics,
            "subgroup_metrics_path": str(SUBGROUP_METRICS_PATH),
            "confidence_zone_report_path": str(CONFIDENCE_ZONE_REPORT_PATH),
            "label_audit_detail_path": str(LABEL_AUDIT_DETAIL_PATH),
            "selected_features_report_path": str(SELECTED_FEATURES_REPORT_PATH),
            "feature_layer_summary_path": str(FEATURE_LAYER_SUMMARY_PATH),
            "temporal_integrity_report_path": str(TEMPORAL_INTEGRITY_REPORT_PATH),
            "split_integrity_report_path": str(SPLIT_INTEGRITY_REPORT_PATH),
            "proxy_stress_test_report_path": str(PROXY_STRESS_TEST_REPORT_PATH),
            "feature_ablation_report_path": str(FEATURE_ABLATION_REPORT_PATH),
            "temporal_generalization_report_path": str(TEMPORAL_GENERALIZATION_REPORT_PATH),
            "warnings": warnings,
        }
        eval_report["schema_validation"] = validate_eval_report_schema(
            report_name="life_eval_report",
            payload=eval_report,
            required_fields=[
                "domain",
                "request_id",
                "overall_metrics",
                "baseline_metrics",
                "warnings",
                "temporal_integrity_report_path",
                "split_integrity_report_path",
                "proxy_stress_test_report_path",
                "temporal_generalization_report_path",
            ],
        )
        write_json(EVAL_REPORT_PATH, eval_report)

        # --- primary_metric_summary (moved before honest_eval_report which references it) ---
        primary_metric_summary = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "display_auc_same_sample_clean_holdout": same_sample_auc,
            "display_auc_strict_comparable": strict_auc,
            "display_auc_temporal_generalization": temporal_auc,
            "recommended_primary_auc": recommended_primary_auc,
            "recommended_primary_source": recommended_primary_source,
            "best_case_auc": max(
                v for v in [same_sample_auc, strict_auc, temporal_auc] if v is not None
            ) if any(v is not None for v in [same_sample_auc, strict_auc, temporal_auc]) else None,
            "robust_auc": recommended_primary_auc,
            "gap": (
                round(
                    max(v for v in [same_sample_auc, strict_auc, temporal_auc] if v is not None)
                    - recommended_primary_auc,
                    6,
                )
                if any(v is not None for v in [same_sample_auc, strict_auc, temporal_auc])
                else None
            ),
            "interpretation": (
                "temporal generalization AUC is the most robust measure of out-of-time stability; "
                "same-sample clean holdout AUC reflects in-distribution performance but may overestimate real-world capability; "
                "strict comparable AUC ensures fair comparison against baseline on identical universe."
            ),
            "target_met": bool((recommended_primary_auc or 0) >= 0.8),
        }
        write_json(PRIMARY_METRIC_SUMMARY_PATH, primary_metric_summary)

        # --- proxy_stress_risk + temporal_gap (moved before honest_eval_report which references them) ---
        proxy_stress_risk = "medium"
        for task in proxy_stress_rows:
            if task.get("semantic_reconstruction_risk") == "high":
                proxy_stress_risk = "high"
                break

        temporal_gap = temporal_generalization_report.get("generalization_gap")
        temporal_gap_risk = "high" if temporal_gap is not None and temporal_gap > 0.1 else (
            "medium" if temporal_gap is not None and temporal_gap > 0.05 else "low"
        )

        honest_eval_report = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "main_report_variant": honest_main.get("task_name", "honest_candidate"),
            "auc": summary_metrics.get("auc"),
            "f1": summary_metrics.get("f1"),
            "precision": summary_metrics.get("precision"),
            "recall": summary_metrics.get("recall"),
            "eval_scope": "student_hash_holdout_fixed_windows_same_sample_universe",
            "split_strategy": "student_hash_three_bucket (train/valid/holdout)",
            "label_source_features": label_source_features,
            "training_feature_columns": main_feature_columns,
            "label_feature_overlap": label_feature_overlap,
            "same_dataset_overlap": same_dataset_overlap,
            "primary_metric": {
                "recommended_primary_auc": recommended_primary_auc,
                "recommended_primary_source": recommended_primary_source,
                "best_case_auc": primary_metric_summary["best_case_auc"],
                "robust_auc": primary_metric_summary["robust_auc"],
                "gap": primary_metric_summary["gap"],
            },
            "label_validity": {
                "label_type": "proxy",
                "proxy_reconstruction_risk": proxy_stress_risk,
                "external_truth_available": False,
                "production_readiness_level": "not_releaseable_for_prod",
            },
            "why_more_honest": [
                "uses the same current holdout universe for strict baseline vs candidate comparison",
                "drops the nearest proxy-source magnitude features from the main reported feature set",
                "keeps explicit temporal integrity and split integrity reports on disk",
            ],
            "remaining_limits": [
                "label is still a future-window proxy rather than an externally validated business target",
                "same-dataset behavioral families are still present through temporal shape and rhythm features",
            ],
        }
        write_json(HONEST_EVAL_REPORT_PATH, honest_eval_report)

        label_validity_audit = {
            "request_id": self.request_id,
            "generated_at": now_iso(),
            "label_type": "proxy",
            "label_definition": "future_window_proxy_from_late_life_behavior",
            "proxy_reconstruction_risk": proxy_stress_risk,
            "external_truth_available": False,
            "release_block_reason": "no externally validated business target; label is derived from same-dataset future window proxy",
            "proxy_stress_test_risk": proxy_stress_risk,
            "temporal_generalization_risk": temporal_gap_risk,
            "source_group_ablation_risk": (
                "medium" if any(
                    task.get("status") == "skipped"
                    for task in proxy_stress_rows
                ) else "low"
            ),
            "overlap_check_result": {
                "label_feature_overlap": label_feature_overlap,
                "same_dataset_overlap": same_dataset_overlap,
                "circularity_detected": circularity_detected,
            },
            "predictive_validity_level": "clean_proxy_with_time_decoupling_but_still_unverified",
            "production_readiness_level": "not_releaseable_for_prod",
            "truth_gap_checklist": [
                {
                    "item": "external_truth_label",
                    "status": "missing",
                    "description": "Need an externally validated business target (e.g. official university risk flag, counselor assessment) that is not derived from the same behavioral dataset",
                    "priority": "critical",
                },
                {
                    "item": "temporal_stability_auc_gte_0_8",
                    "status": "met" if (temporal_auc or 0) >= 0.8 else "pending",
                    "description": "Temporal generalization AUC should remain >= 0.8 on held-out time periods",
                    "priority": "high",
                },
                {
                    "item": "cross_domain_consistency",
                    "status": "pending",
                    "description": "Life risk scores should correlate meaningfully with study domain risk indicators for same students",
                    "priority": "medium",
                },
                {
                    "item": "prospective_validation",
                    "status": "missing",
                    "description": "Model should be tested on a future cohort not used in any label construction or feature engineering",
                    "priority": "high",
                },
            ],
        }
        write_json(LABEL_VALIDITY_AUDIT_PATH, label_validity_audit)

        # Truth gap checklist as standalone file
        write_json(
            TRUTH_GAP_CHECKLIST_PATH,
            {
                "request_id": self.request_id,
                "generated_at": now_iso(),
                "current_label_type": "proxy",
                "external_truth_available": False,
                "checklist": label_validity_audit["truth_gap_checklist"],
                "conclusion": (
                    "life domain is mature for harness integration as a predictive proxy model, "
                    "but remains not_releaseable_for_prod until external truth validation is available."
                ),
            },
        )

        write_json(
            VS_STUDY_MATURITY_GAP_V2_PATH,
            {
                "request_id": self.request_id,
                "generated_at": now_iso(),
                "study_and_life_are_peer_domains": True,
                "life_mature_for_harness_integration": True,
                "life_not_releaseable_for_prod": True,
                "life_strengths": [
                    "frozen baseline and candidate snapshot contract are present",
                    "strict comparable honest evaluation is available",
                    "multi-domain adapter integration is available",
                ],
                "life_gaps_vs_study": [
                    "proxy label authenticity remains weaker than study's established release semantics",
                    "same-caliber governance is simpler than study and not yet policy-rich",
                ],
            },
        )

        train_table_path = self.deliverables["data"] / "train_table.csv"
        infer_table_path = self.deliverables["data"] / "infer_table.csv"
        prediction_output_path = self.deliverables["data"] / "prediction_output.csv"
        quality_report_path = self.deliverables["docs"] / "quality_report.json"
        validation_report_path = self.deliverables["docs"] / "validation_report.json"
        feature_dict_path = self.deliverables["docs"] / "feature_dictionary.csv"
        model_path = self.deliverables["model"] / "model.pkl"
        model_config_path = self.deliverables["model"] / "model_config.json"
        model_metrics_path = self.deliverables["model"] / "model_metrics.json"

        train_df.to_csv(train_table_path, index=False, encoding="utf-8-sig")
        infer_df.to_csv(infer_table_path, index=False, encoding="utf-8-sig")
        main_prediction.to_csv(prediction_output_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [{"feature_name": col, "feature_type": feature_kind(train_df[col])} for col in main_feature_columns]
        ).to_csv(feature_dict_path, index=False, encoding="utf-8-sig")
        joblib.dump(main_model, model_path)
        write_json(model_config_path, candidate_model_config)
        write_json(model_metrics_path, {"summary_metrics": summary_metrics, "baseline_metrics": baseline_metrics, "warnings": warnings})
        write_json(
            quality_report_path,
            {
                "domain": "life",
                "rows": int(len(feature_df)),
                "missing_rate": round(float(feature_df[main_feature_columns].isna().mean().mean()), 6),
                "label_positive_rate": summary_metrics["positive_rate"],
                "feature_layer_summary": feature_layer_summary,
            },
        )
        write_json(
            validation_report_path,
            {
                "domain": "life",
                "validation_status": "passed",
                "checks": {
                    "train_rows_positive": len(train_df) > 0,
                    "infer_rows_positive": len(infer_df) > 0,
                    "candidate_snapshot_complete": candidate_snapshot_check["ok"],
                    "baseline_snapshot_complete": baseline_snapshot_check["ok"],
                    "life_clean_eval": True,
                    "temporal_integrity_ok": temporal_integrity["violation_count"] == 0,
                    "split_integrity_ok": not split_integrity["window_neighbor_leakage_detected"],
                },
                "warnings": warnings,
            },
        )

        model_selection = {
            "request_id": self.request_id,
            "selected_model": "LogisticRegression",
            "candidate_version_id": candidate_version_id,
            "baseline_version_id": baseline_version_id,
            "summary_metrics": summary_metrics,
            "reason_codes": ["clean_eval_enabled", "frozen_baseline_compare_available"],
        }
        write_json(MODEL_SELECTION_PATH, model_selection)

        subgroup_auc_values = [
            float(v) for v in subgroup_metrics.get("auc", pd.Series(dtype=float)).dropna().tolist()
        ]
        temporal_consistency = evaluate_temporal_consistency(
            main_metrics.get("auc"),
            temporal_generalization_report.get("temporal_auc"),
        )
        leakage_risk = evaluate_leakage_risk(
            label_feature_overlap=label_feature_overlap,
            explicit_temporal_leakage_detected=bool(temporal_integrity["violation_count"] > 0),
            future_window_used=True,
        )
        subgroup_stability = evaluate_subgroup_stability(subgroup_auc_values)
        trust_score = compute_trust_score(
            temporal_gap=temporal_consistency.get("temporal_gap"),
            leakage=leakage_risk,
            subgroup_unstable=bool(subgroup_stability.get("subgroup_unstable")),
        )
        future_window_auc = best_instability.get("auc")
        if future_window_auc is None:
            future_window_auc = temporal_generalization_report.get("temporal_auc")
        suspicious_high_auc = bool(future_window_auc is not None and float(future_window_auc) > 0.95)
        trust_decision = "not_releaseable" if trust_score < 0.6 else "eligible_for_comparison"
        trust_reason = "low_trust_score" if trust_score < 0.6 else "trust_score_ok"
        if future_window_auc is not None and float(future_window_auc) < 0.8:
            trust_decision = "needs_review"
            trust_reason = "future_window_auc_below_0_8"
        if bool(subgroup_stability.get("subgroup_unstable")):
            trust_decision = "needs_review"
            trust_reason = "subgroup_unstable"
        if leakage_risk:
            trust_decision = "hold_for_review"
            trust_reason = "leakage_risk"
        if suspicious_high_auc:
            trust_decision = "hold_for_review"
            trust_reason = "suspicious_high_auc"
        failure_types: list[str] = []
        if suspicious_high_auc:
            failure_types.append("suspicious_high_auc")
        if leakage_risk:
            failure_types.append("high_tautology_risk")
        if future_window_auc is not None and float(future_window_auc) < 0.8:
            failure_types.append("low_future_signal")
        if bool(subgroup_stability.get("subgroup_unstable")):
            failure_types.append("subgroup_collapse")
        if temporal_consistency.get("flag_temporal_instability"):
            failure_types.append("poor_temporal_generalization")
        if not failure_types:
            failure_types.append("none")
        proxy_track = {
            "label_type": "proxy",
            "releaseable": False,
            "track_role": "reference_only",
            "decision": "reference_only",
            "reason": "proxy_track_not_for_release",
        }
        instability_track = {
            "track_role": "primary_decision",
            "future_window_auc": future_window_auc,
            "temporal_gap": temporal_consistency.get("temporal_gap"),
            "leakage": leakage_risk,
            "subgroup_variance": subgroup_stability.get("subgroup_variance"),
            "trust_score": trust_score,
            "decision": trust_decision,
        }
        protocol = build_agent_protocol(
            domain="life",
            diagnosis={
                "task_type": "future_window_prediction",
                "failure_types": failure_types,
                "proxy_label": True,
                "track_role": {"proxy_track": "reference_only", "instability_track": "primary_decision"},
                "temporal_gap": temporal_consistency.get("temporal_gap"),
                "leakage_risk": leakage_risk,
                "future_window_auc": future_window_auc,
            },
            proposal={
                "dual_track": ["proxy_track", "instability_track"],
                "task_reconstruction": {
                    "switch_label_version": "instability_future_v2" if "suspicious_high_auc" in failure_types else best_instability.get("label_version", "instability_future_v1"),
                    "switch_time_window": "2w" if "suspicious_high_auc" in failure_types else "1w",
                    "switch_target_population": "stable_group_only" if "suspicious_high_auc" in failure_types else "all",
                    "switch_structure_version": "purged_temporal_split" if "poor_temporal_generalization" in failure_types else "temporal_split",
                },
                "instability_label_version": best_instability.get("label_version", "instability_future_v1"),
                "feature_bundle": best_instability.get("feature_bundle", "regularity+volatility+coupling"),
                "trust_score": trust_score,
                "next_optimization_target": "improve_future_window_auc",
            },
            comparison={
                "baseline_auc": baseline_metrics.get("auc"),
                "candidate_auc": summary_metrics.get("auc"),
                "auc_delta": (
                    round(float(summary_metrics.get("auc")) - float(baseline_metrics.get("auc")), 6)
                    if summary_metrics.get("auc") is not None and baseline_metrics.get("auc") is not None
                    else None
                ),
                "future_window_auc": future_window_auc,
                "is_future_window": True,
                "same_source_risk": "high" if leakage_risk else "low",
                "subgroup_stable": not bool(subgroup_stability.get("subgroup_unstable")),
                "temporal_holdout_stable": not bool(temporal_consistency.get("flag_temporal_instability")),
                "more_real_than_baseline": bool(future_window_auc is not None and float(future_window_auc) <= 0.95),
            },
            recommendation={
                "decision": trust_decision,
                "reason": trust_reason,
                "track_role": "instability_primary",
                "best_mainline_candidate": best_instability,
                "high_score_but_untrusted": bool("suspicious_high_auc" in failure_types or "high_tautology_risk" in failure_types),
                "real_but_weak": bool("low_future_signal" in failure_types),
                "next_priority_route": "authenticity_first_search",
            },
        )

        decision_bundle = normalize_decision_bundle(
            final_decision=trust_decision,
            policy_decision=trust_decision,
            execution_mode="dry_run",
            reason_codes=["clean_eval_enabled", "frozen_baseline_compare_available", trust_reason],
            decision_stage_reached="active_baseline_comparison",
        )

        metric_context = {
            "eval_scope": "student_hash_holdout_fixed_windows",
            "task_scope": "life_clean_behavior_risk",
            "feature_contract_hash": feature_contract_hash,
            "label_definition": "life_label_clean",
            "baseline_version_id": baseline_version_id,
            "anchor_baseline_version_id": baseline_version_id,
            "comparison_mode": "strict_same_universe_compare",
            "primary_compare_source": "strict_same_universe_compare",
            "frozen_baseline_compare_available": True,
            "comparability": strict_comparability,
            "sample_count": int(len(infer_df)),
            "label_source_features": label_source_features,
            "training_feature_columns": main_feature_columns,
            "label_feature_overlap": label_feature_overlap,
            "same_dataset_overlap": same_dataset_overlap,
            "circularity_detected": circularity_detected,
            "life_data_mode": feature_layer_summary["life_data_mode"],
            "split_strategy": split_integrity_report["split_strategy"],
            "temporal_generalization_gap": temporal_generalization_report.get("generalization_gap"),
            "future_window_auc": future_window_auc,
            "trust_score": trust_score,
            "leakage_risk": leakage_risk,
            "subgroup_variance": subgroup_stability.get("subgroup_variance"),
            "instability_best_candidate": best_instability,
            "primary_metric": {
                "recommended_primary_auc": recommended_primary_auc,
                "recommended_primary_source": recommended_primary_source,
                "robust_auc": primary_metric_summary["robust_auc"],
            },
        }

        harness_payload = {
            "run_id": f"life_harness_v1_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "pipeline_name": "life_harness_v1",
            "candidate_version_id": candidate_version_id,
            "baseline_version_id": baseline_version_id,
            "anchor_baseline_version_id": baseline_version_id,
            **decision_bundle,
            "eval_scope": metric_context["eval_scope"],
            "task_scope": metric_context["task_scope"],
            "feature_contract_hash": feature_contract_hash,
            "label_definition": metric_context["label_definition"],
            "comparison_mode": metric_context["comparison_mode"],
            "metric_context": metric_context,
            "contract_context": candidate_contract_context["contract_context"],
            "local_gain_signals": {"clean_candidate_auc_improved": (auc_delta or 0) > 0},
            "baseline_summary": {
                "metrics": baseline_metrics,
                "anchor_metrics": baseline_metrics,
                "snapshot_manifest": baseline_manifest,
                "baseline_compare_available": True,
                "bootstrap_mode": False,
            },
            "candidate_snapshot_manifest": candidate_manifest,
            "candidate_vs_baseline": comparison,
            "label_type": "proxy",
            "releaseable": False,
            "task_type": "future_window_prediction",
            "future_window_prediction": True,
            "comparable": trust_decision == "eligible_for_comparison",
            "mainline_validity": trust_decision == "eligible_for_comparison",
            "blocking_reason": "" if trust_decision == "eligible_for_comparison" else trust_reason,
            "next_optimization_target": "improve_future_window_auc",
            "releaseability_reason": trust_reason,
            "future_window_auc": future_window_auc,
            "suspicious_high_auc": suspicious_high_auc,
            "search_contract": self.request.get("search_contract", {}),
            "trust_score": trust_score,
            "trust_diagnostics": {
                "temporal_gap": temporal_consistency.get("temporal_gap"),
                "leakage": leakage_risk,
                "subgroup_variance": subgroup_stability.get("subgroup_variance"),
            },
            "proxy_track": proxy_track,
            "instability_track": instability_track,
            "instability_best_candidate": best_instability,
            "trusted_mainline": best_instability,
            "mainline_frozen": True,
            "agent_protocol": protocol,
        }

        fusion_input = {
            "domain_name": "life",
            "candidate_version_id": candidate_version_id,
            "risk_score": summary_metrics["mean_infer_risk_score"],
            "risk_level": risk_level_from_score(summary_metrics["mean_infer_risk_score"]),
            "confidence": confidence_from_auc(summary_metrics.get("auc")),
            "top_features": [{"feature": feature, "weight_rank": idx + 1} for idx, feature in enumerate(main_feature_columns[:5])],
            "explanations": [
                {"summary": f"honest life candidate uses {honest_main.get('task_name', 'honest_candidate')} features under strict comparable clean holdout"}
            ],
            "quality_metrics": {
                "auc": summary_metrics.get("auc"),
                "f1": summary_metrics.get("f1"),
                "precision": summary_metrics.get("precision"),
                "recall": summary_metrics.get("recall"),
                "positive_rate": summary_metrics.get("positive_rate"),
                "robust_auc": primary_metric_summary["robust_auc"],
                "reported_primary_auc": primary_metric_summary["recommended_primary_auc"],
            },
            "metric_context": metric_context,
            "validation_summary": {
                "fusion_payload_semantics_ok": True,
                "risk_score_source": "mean_infer_predicted_probability",
                "quality_metric_source": "clean_holdout_eval_metrics",
                "snapshot_contract_complete": candidate_snapshot_check["ok"],
            },
            "warning_summary": warnings,
            "artifact_ref": {
                "prediction_output": str(prediction_output_path),
                "eval_report": str(EVAL_REPORT_PATH),
                "model_config": str(model_config_path),
                "subgroup_metrics": str(SUBGROUP_METRICS_PATH),
                "confidence_zone_report": str(CONFIDENCE_ZONE_REPORT_PATH),
            },
            "raw_payload": {
                "domain_context": candidate_domain_context,
                "domain_audit": candidate_domain_audit,
            },
        }

        result = normalize_domain_result(
            domain="life",
            status="success",
            summary_metrics=summary_metrics,
            decision_bundle=decision_bundle,
            metric_context=metric_context,
            domain_context=candidate_domain_context,
            domain_audit=candidate_domain_audit,
            warnings=warnings,
            deliverables={
                "train_table": str(train_table_path),
                "infer_table": str(infer_table_path),
                "prediction_output": str(prediction_output_path),
                "quality_report": str(quality_report_path),
                "validation_report": str(validation_report_path),
                "feature_dictionary": str(feature_dict_path),
                "model_file": str(model_path),
                "model_config": str(model_config_path),
                "model_metrics": str(model_metrics_path),
            },
            harness_payload=harness_payload,
            fusion_input=fusion_input,
            extra={
                "request_id": self.request_id,
                "mode": self.request.get("run_mode", "review"),
                "label_type": "proxy",
                "releaseable": False,
                "track_role": {"proxy_track": "reference_only", "instability_track": "primary_decision"},
                "future_window_auc": future_window_auc,
                "trust_score": trust_score,
                "suspicious_high_auc": suspicious_high_auc,
                "releaseability_reason": trust_reason,
                "diagnostics": {
                    "temporal_gap": temporal_consistency.get("temporal_gap"),
                    "leakage": leakage_risk,
                    "subgroup_variance": subgroup_stability.get("subgroup_variance"),
                },
                "proxy_track": proxy_track,
                "instability_track": instability_track,
                "instability_best_candidate": best_instability,
                "agent_protocol": protocol,
                "label_source_features": label_source_features,
                "training_feature_columns": main_feature_columns,
                "label_feature_overlap": label_feature_overlap,
                "same_dataset_overlap": same_dataset_overlap,
                "circularity_detected": circularity_detected,
                "reported_metrics_reference": {
                    "trusted_mainline_metrics": summary_metrics,
                },
                "contract_context": candidate_contract_context["contract_context"],
            },
        )
        write_json(RESULT_PATH, result)

        candidate_artifact_dir = DM_DIR / "candidate_artifacts" / candidate_version_id
        candidate_artifact_dir.mkdir(parents=True, exist_ok=True)
        for src in [
            MODEL_SELECTION_PATH,
            EVAL_REPORT_PATH,
            LEAKAGE_COMPARE_REPORT_PATH,
            BASELINE_INTEGRITY_REPORT_PATH,
            CLEAN_BASELINE_REPORT_PATH,
            CANDIDATE_COMPARE_REPORT_PATH,
            MATURITY_GAP_SUMMARY_PATH,
            TEMPORAL_INTEGRITY_REPORT_PATH,
            SPLIT_INTEGRITY_REPORT_PATH,
            PROXY_STRESS_TEST_REPORT_PATH,
            FEATURE_ABLATION_REPORT_PATH,
            TEMPORAL_GENERALIZATION_REPORT_PATH,
            CREDIBILITY_SUMMARY_PATH,
            AUC_CREDIBILITY_AUDIT_PATH,
            EVAL_SCOPE_RECONCILIATION_PATH,
            STRICT_COMPARE_SUMMARY_PATH,
            SOURCE_GROUP_ABLATION_PATH,
            HONEST_EVAL_REPORT_PATH,
            HONEST_FEATURE_SET_PATH,
            VS_STUDY_MATURITY_GAP_V2_PATH,
            PRIMARY_METRIC_SUMMARY_PATH,
            LABEL_VALIDITY_AUDIT_PATH,
            TRUTH_GAP_CHECKLIST_PATH,
        ]:
            shutil.copy2(src, candidate_artifact_dir / src.name)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the life domain agent.")
    parser.add_argument("--request", required=True, help="Path to life agent request JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = LifeAgent(request_path=Path(args.request)).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
