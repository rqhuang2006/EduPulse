from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any


def _bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parents[1]
    workspace_root = root.parent
    vendor_dir = workspace_root / ".deps3"
    for path in [vendor_dir, workspace_root, root]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root


ROOT = _bootstrap_paths()
WORKSPACE_ROOT = ROOT.parent

import joblib
import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from study_feature_engine import (
    apply_feature_engineering,
    infer_feature_layer,
    summarize_feature_layers,
)
from study_routing_policy import apply_serving_policy, resolve_policy

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency guard
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

LOG_DIR = ROOT / "logs"
DM_DIR = ROOT / "data" / "dm"
DELIVERABLE_DOCS_DIR = ROOT / "data" / "deliverables" / "study" / "docs"
TRACE_PATH = LOG_DIR / "study_run_trace.jsonl"
DECISION_PATH = LOG_DIR / "study_decision_log.json"
RESULT_PATH = DM_DIR / "study_agent_result.json"
DELIVERABLE_RESULT_PATH = DELIVERABLE_DOCS_DIR / "study_agent_result.json"
QUALITY_JSON_PATH = DM_DIR / "study_quality_report.json"
DELIVERABLE_QUALITY_JSON_PATH = DELIVERABLE_DOCS_DIR / "study_quality_report.json"
FEATURE_LAYER_SUMMARY_PATH = DM_DIR / "study_feature_layer_summary.json"
INFER_FEATURE_LAYER_SUMMARY_PATH = DM_DIR / "study_infer_feature_layer_summary.json"
SELECTED_FEATURES_PATH = DM_DIR / "study_selected_features.csv"
LABEL_AUDIT_DETAIL_PATH = DM_DIR / "study_label_audit_detail.json"
POSITIVE_LABEL_PROFILE_PATH = DM_DIR / "study_positive_label_profile.csv"
MODEL_COMPARISON_PATH = DM_DIR / "study_model_comparison.csv"
MODEL_SELECTION_PATH = DM_DIR / "study_model_selection.json"
DELIVERABLE_MODEL_SELECTION_PATH = DELIVERABLE_DOCS_DIR / "study_model_selection.json"
THRESHOLD_TUNING_PATH = DM_DIR / "study_threshold_tuning.csv"
THRESHOLD_SELECTION_PATH = DM_DIR / "study_threshold_selection.json"
BRANCH_SCORES_PATH = DM_DIR / "study_branch_scores.csv"
FUSION_CONFIG_PATH = DM_DIR / "study_fusion_config.json"
LLM_ADVICE_PATH = DM_DIR / "study_llm_model_advice.json"
DELIVERABLE_LLM_ADVICE_PATH = DELIVERABLE_DOCS_DIR / "study_llm_model_advice.json"

REQUIRED_REQUEST_FIELDS = {
    "request_id",
    "domain",
    "term_id",
    "run_mode",
    "input_paths",
    "feature_version",
    "model_version",
    "enable_fallback",
    "enable_explanation",
}

REQUIRED_INPUT_PATHS = {
    "train_table",
    "infer_table",
    "prediction_output",
    "explanation_output",
    "quality_report",
    "validation_report",
    "feature_dictionary",
    "model_file",
    "model_config",
}

L456_PREFIXES = ("FEATURE_CLASS_", "FEATURE_ASSIGNMENT_", "FEATURE_EXAM_")
QUALITY_FEATURES = {"FEATURE_MISSING_RATE", "SOURCE_COVERAGE", "DATA_QUALITY_FLAG"}
UNKNOWN_TOKENS = {"", "nan", "none", "null", "unknown", "unk", "未映射", "未知", "缺失", "missing", "<na>"}
CORE_FAMILY_PREFIXES = {
    "grade": ("FEATURE_GRADE_", "FEATURE_CET_"),
    "course": ("FEATURE_COURSE_",),
}
BEHAVIOR_FAMILY_PREFIXES = {
    "attendance": ("FEATURE_ATTENDANCE_",),
    "class_task": ("FEATURE_CLASS_",),
    "assignment": ("FEATURE_ASSIGNMENT_",),
    "exam": ("FEATURE_EXAM_",),
    "library": ("FEATURE_LIBRARY_",),
    "online": ("FEATURE_ONLINE_",),
}
FEATURE_FAMILY_PREFIXES = {
    "grade": ("FEATURE_GRADE_", "FEATURE_CET_"),
    "course": ("FEATURE_COURSE_",),
    "attendance": ("FEATURE_ATTENDANCE_",),
    "class_task": ("FEATURE_CLASS_",),
    "assignment": ("FEATURE_ASSIGNMENT_",),
    "exam": ("FEATURE_EXAM_",),
    "library": ("FEATURE_LIBRARY_",),
    "online": ("FEATURE_ONLINE_",),
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def _remap_legacy_path(path_str: str, marker: str, target_root: Path) -> str | None:
    lowered = path_str.lower()
    marker_lower = marker.lower()
    idx = lowered.find(marker_lower)
    if idx < 0:
        return None
    suffix = path_str[idx + len(marker):].lstrip("\\/")
    parts = [part for part in PureWindowsPath(suffix).parts if part not in {"\\", "/"}]
    return str(target_root.joinpath(*parts)) if parts else str(target_root)


def normalize_workspace_path(path_str: str) -> str:
    if not isinstance(path_str, str) or not path_str:
        return path_str
    if "\\" not in path_str and "/" not in path_str:
        return path_str

    markers = [
        ("\\study\\data\\deliverables\\study\\", ROOT / "data" / "deliverables" / "study"),
        ("\\study\\data\\registry\\study\\", ROOT / "data" / "registry" / "study"),
        ("\\study\\data\\harness\\", ROOT / "data" / "harness"),
        ("\\study\\data\\dm\\", ROOT / "data" / "dm"),
        ("\\study\\logs\\", ROOT / "logs"),
        ("\\study\\conf\\", ROOT / "conf"),
        ("E:\\AAA\\data\\deliverables\\study\\", ROOT / "data" / "deliverables" / "study"),
        ("E:\\AAA\\data\\dm\\", ROOT / "data" / "dm"),
    ]
    for marker, target_root in markers:
        remapped = _remap_legacy_path(path_str, marker, target_root)
        if remapped:
            return remapped
    return path_str


def normalize_workspace_paths(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: normalize_workspace_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [normalize_workspace_paths(value) for value in payload]
    if isinstance(payload, str):
        return normalize_workspace_path(payload)
    return payload


def infer_feature_layer(col_name: str) -> str:
    c = str(col_name).lower()
    temporal_markers = [
        "recent_",
        "prev_",
        "hist_",
        "delta_",
        "ratio_",
        "trend_",
        "slope_",
        "volatility_",
        "stability_",
        "rolling_",
        "window_",
        "chg_",
        "change_",
        "consecutive_decline_",
        "dist_from_worst_",
        "recovery_",
    ]
    interaction_markers = ["__x__", "_x_", "cross_", "inter_", "discordance_", "workload_stress"]
    behavior_markers = [
        "attendance",
        "borrow",
        "library",
        "night",
        "sleep",
        "consume",
        "consumption",
        "dorm",
        "internet",
        "online",
        "class_participation",
        "activity",
        "schedule",
        "regularity",
        "assignment",
        "exam",
        "class_task",
        "video",
    ]

    if any(marker in c for marker in interaction_markers):
        return "interaction"
    if any(marker in c for marker in temporal_markers):
        return "temporal"
    if any(marker in c for marker in behavior_markers):
        return "behavior"
    return "core"


def summarize_feature_layers(feature_cols: list[str]) -> dict[str, Any]:
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

    return {
        "counts": counts,
        "columns": summary,
        "study_data_mode": mode,
    }


class StudyAgent:
    """Harness-facing agent for study execution and controlled train-mode enhancement."""

    def __init__(self, request_path: Path | None = None, request: dict[str, Any] | None = None):
        if request_path is None and request is None:
            raise ValueError("StudyAgent requires either request_path or request.")
        self.request_path = self._resolve_path(request_path) if request_path else None
        self.request = self._read_json(self.request_path) if self.request_path else dict(request or {})
        self.request_id = str(self.request.get("request_id", "unknown_request"))
        self.policy: dict[str, Any] = {}
        self.search_space: dict[str, Any] = {}
        self.paths: dict[str, Path] = {}
        self.model_config: dict[str, Any] = {}
        self.validation_report: dict[str, Any] = {}
        self.train = pd.DataFrame()
        self.infer = pd.DataFrame()
        self.prediction = pd.DataFrame()
        self.explanation = pd.DataFrame()
        self.quality_sheets: dict[str, pd.DataFrame] = {}
        self.quality_report: dict[str, Any] = {}
        self.row_quality = pd.DataFrame()
        self.row_modes = pd.DataFrame()
        self.selected_rows = pd.DataFrame()
        self.selected_predictions = pd.DataFrame()
        self.selected_explanations = pd.DataFrame()
        self.summary_metrics: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.decisions: list[dict[str, Any]] = []
        self.state_history: list[dict[str, Any]] = []
        self.status = "success"
        self.fallback_used = False
        self.scoring_model_name = ""
        self.scoring_model_source = ""
        self.study_data_mode = "core_only"
        self.study_quality_flag = "core_stable"
        self.row_level_study_data_mode = "core_only"
        self.row_level_study_quality_flag = "core_stable"
        self.feature_layer_summary: dict[str, Any] = {}
        self.infer_feature_layer_summary: dict[str, Any] = {}
        self.model_comparison = pd.DataFrame()
        self.threshold_tuning = pd.DataFrame()
        self.model_selection: dict[str, Any] = {}
        self.threshold_selection: dict[str, Any] = {}
        self.branch_scores = pd.DataFrame()
        self.fusion_config: dict[str, Any] = {}
        self.llm_review: dict[str, Any] = {}
        self.extra_result: dict[str, Any] = {}

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        DM_DIR.mkdir(parents=True, exist_ok=True)
        DELIVERABLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self._state("received", "success", "StudyAgent request received.")
        self._trace(
            "task_received",
            "success",
            "StudyAgent request received.",
            {"request_path": str(self.request_path) if self.request_path else "inline_request"},
        )

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path:
        if path is None:
            raise ValueError("path cannot be None")
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate
        root_candidate = ROOT / candidate
        if root_candidate.exists():
            return root_candidate
        return WORKSPACE_ROOT / candidate

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _trace(self, stage: str, status: str, message: str, key_metrics: dict[str, Any] | None = None) -> None:
        record = {
            "timestamp": now_iso(),
            "request_id": self.request_id,
            "stage": stage,
            "status": status,
            "message": message,
            "key_metrics": key_metrics or {},
        }
        with TRACE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")

    def _decision(self, decision: str, status: str, reason: str, metrics: dict[str, Any] | None = None) -> None:
        self.decisions.append(
            {
                "timestamp": now_iso(),
                "request_id": self.request_id,
                "decision": decision,
                "status": status,
                "reason": reason,
                "metrics": metrics or {},
            }
        )

    def _state(self, state: str, status: str, detail: str) -> None:
        self.state_history.append(
            {
                "timestamp": now_iso(),
                "request_id": self.request_id,
                "state": state,
                "status": status,
                "detail": detail,
            }
        )

    def _set_status(self, candidate: str) -> None:
        order = {"success": 0, "degraded": 1, "completed_with_hold": 1, "failed": 2}
        current = self.status if self.status in order else "success"
        if order.get(candidate, 0) > order[current]:
            self.status = candidate

    def _allowed(self, key: str) -> list[str]:
        value = self.search_space.get(key, [])
        return list(value or [])

    @staticmethod
    def _latest_stage_result(stage_results: list[Any], action_name: str) -> Any | None:
        for result in reversed(stage_results):
            if getattr(result, "action_name", "") == action_name:
                return result
        return None

    @staticmethod
    def _map_harness_status(record: Any) -> str:
        if getattr(record, "status", "") == "failed":
            return "failed"

        decision = getattr(getattr(record, "final_decision", None), "decision", "")
        if decision == "incomparable_candidate":
            return "reviewed_not_promotable"
        if decision in {"keep_baseline", "dry_run_only", "reject", "rollback_recommended"}:
            return "completed_with_hold"
        if decision == "promotion_pending_approval":
            return "pending_approval"
        if decision in {"promotion_recommended", "published"}:
            return "success"
        return "success"

    def load_data(self) -> None:
        policy_path = ROOT / "conf" / "study_agent_policy.yaml"
        search_space_path = ROOT / "conf" / "study_model_search_space.yaml"
        with policy_path.open("r", encoding="utf-8") as handle:
            self.policy = yaml.safe_load(handle) or {}
        with search_space_path.open("r", encoding="utf-8") as handle:
            self.search_space = yaml.safe_load(handle) or {}

        input_paths = self.request.get("input_paths", {})
        self.paths = {name: self._resolve_path(path) for name, path in input_paths.items()}

        self.model_config = normalize_workspace_paths(self._read_json(self.paths["model_config"]))
        self.validation_report = self._read_json(self.paths["validation_report"])
        self.train = pd.read_csv(self.paths["train_table"])
        self.infer = pd.read_csv(self.paths["infer_table"])
        if self.paths["prediction_output"].exists():
            self.prediction = pd.read_csv(self.paths["prediction_output"])
        if self.request.get("enable_explanation", False) and self.paths["explanation_output"].exists():
            self.explanation = pd.read_csv(self.paths["explanation_output"])
        if self.paths["quality_report"].exists() and self.paths["quality_report"].suffix.lower() in {".xlsx", ".xls"}:
            try:
                quality_book = pd.ExcelFile(self.paths["quality_report"])
                self.quality_sheets = {
                    sheet: pd.read_excel(self.paths["quality_report"], sheet_name=sheet) for sheet in quality_book.sheet_names
                }
            except Exception as exc:
                self._trace("quality_preload_skipped", "warning", "Existing quality workbook preload skipped.", {"error": str(exc)})

        self._trace(
            "data_loaded",
            "success",
            "Study input tables and optional artifacts loaded.",
            {
                "train_rows": len(self.train),
                "infer_rows": len(self.infer),
                "prediction_rows": len(self.prediction),
                "explanation_rows": len(self.explanation),
            },
        )

    def validate_input(self) -> None:
        self._state("validating", self.status, "Checking request schema, paths, and version alignment.")
        missing_fields = sorted(REQUIRED_REQUEST_FIELDS - set(self.request))
        missing_path_keys = sorted(REQUIRED_INPUT_PATHS - set(self.request.get("input_paths", {})))
        output_paths = {"prediction_output", "explanation_output", "quality_report"}
        missing_files = []
        for name, path in self.paths.items():
            if path.exists():
                continue
            if self.request.get("run_mode") == "infer" and name in output_paths:
                continue
            missing_files.append(name)
        errors = []
        if missing_fields:
            errors.append(f"missing request fields: {missing_fields}")
        if missing_path_keys:
            errors.append(f"missing input_paths keys: {missing_path_keys}")
        if missing_files:
            errors.append(f"missing files: {missing_files}")
        if self.request.get("domain") != "study":
            errors.append("domain must be study")
        if self.request.get("run_mode") not in {"train", "infer", "review", "publish", "rollback"}:
            errors.append("run_mode must be train, infer, review, publish, or rollback")
        if self.request.get("model_version") != self.model_config.get("model_version"):
            errors.append("request model_version does not match model_config")
        if self.request.get("feature_version") != self.model_config.get("feature_version"):
            errors.append("request feature_version does not match model_config")

        self._validate_train_options(errors)
        self._validate_llm_options(errors)

        if errors:
            self._set_status("failed")
            message = "; ".join(errors)
            self._decision("input_validation", "failed", message)
            self._trace("input_validated", "failed", message, {"error_count": len(errors)})
            raise ValueError(message)

        self._decision("input_validation", "success", "Request schema, domain, mode, versions, and paths are valid.")
        self._trace("input_validated", "success", "Request validated.", {"run_mode": self.request["run_mode"]})

    def _validate_train_options(self, errors: list[str]) -> None:
        requested_groups = self.request.get("candidate_feature_groups", self._allowed("feature_groups"))
        requested_models = self.request.get("candidate_models", self._allowed("candidate_models"))
        requested_thresholds = self.request.get("threshold_strategy", self._allowed("threshold_strategies"))
        if isinstance(requested_thresholds, str):
            requested_thresholds = [requested_thresholds]

        invalid_groups = sorted(set(requested_groups) - set(self._allowed("feature_groups")))
        invalid_models = sorted(set(requested_models) - set(self._allowed("candidate_models")))
        invalid_thresholds = sorted(set(requested_thresholds) - set(self._allowed("threshold_strategies")))
        if invalid_groups:
            errors.append(f"candidate_feature_groups outside controlled search space: {invalid_groups}")
        if invalid_models:
            errors.append(f"candidate_models outside controlled search space: {invalid_models}")
        if invalid_thresholds:
            errors.append(f"threshold_strategy outside controlled search space: {invalid_thresholds}")
        if self.request.get("enable_branch_fusion", False) and "weighted_branch_fusion" not in self._allowed("fusion_strategies"):
            errors.append("weighted_branch_fusion is not allowed by search space")
        if self.request.get("publish_selected_model", False):
            self.warnings.append("publish_selected_model was requested but this agent build does not overwrite formal artifacts.")

    def _validate_llm_options(self, errors: list[str]) -> None:
        provider = self.request.get("llm_provider", "mock")
        task_type = self.request.get("llm_task_type", "model_review")
        if provider not in {"mock", "qwen"}:
            errors.append("llm_provider must be mock or qwen")
        if task_type not in {"model_review", "selection_reason", "explanation_enhancement", "evolution_review"}:
            errors.append("llm_task_type must be model_review, selection_reason, explanation_enhancement, or evolution_review")
        for field in ["llm_temperature", "llm_timeout_seconds", "llm_max_tokens"]:
            if field in self.request and self.request[field] is None:
                errors.append(f"{field} cannot be null")

    def _write_table(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def _write_quality_report(self) -> None:
        write_json(QUALITY_JSON_PATH, self.quality_report)
        shutil.copy2(QUALITY_JSON_PATH, DELIVERABLE_QUALITY_JSON_PATH)
        requested = self.paths.get("quality_report")
        if requested and requested.suffix.lower() == ".json":
            write_json(requested, self.quality_report)
        elif requested and requested.suffix.lower() in {".xlsx", ".xls"}:
            try:
                requested.parent.mkdir(parents=True, exist_ok=True)
                with pd.ExcelWriter(requested) as writer:
                    pd.DataFrame(self.quality_report.get("summary", [{}])).to_excel(writer, sheet_name="summary", index=False)
                    pd.DataFrame(self.quality_report.get("field_coverage", [])).to_excel(writer, sheet_name="field_coverage", index=False)
                    pd.DataFrame(self.quality_report.get("primary_key_association", [])).to_excel(writer, sheet_name="pk_association", index=False)
                    pd.DataFrame(self.quality_report.get("unknown_value_stats", [])).to_excel(writer, sheet_name="unknown_stats", index=False)
                    pd.DataFrame(self.quality_report.get("data_mode_stats", [])).to_excel(writer, sheet_name="data_mode_stats", index=False)
                    pd.DataFrame(self.quality_report.get("row_samples", [])).to_excel(writer, sheet_name="row_samples", index=False)
            except Exception as exc:
                self._trace("quality_xlsx_skipped", "warning", "Quality workbook write skipped; JSON report remains available.", {"error": str(exc)})

    def _quality_report_output_path(self) -> str:
        return str(QUALITY_JSON_PATH)

    def _load_model_bundle(self) -> dict[str, Any]:
        bundle = joblib.load(self.paths["model_file"])
        if not isinstance(bundle, dict):
            raise ValueError("model bundle must be a dict with primary_model/fallback_model/config")
        return bundle

    @staticmethod
    def _feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")

    def _emit_feature_layer_summary(self, feature_cols: list[str], output_path: Path, scope: str) -> dict[str, Any]:
        summary = summarize_feature_layers(feature_cols)
        summary.update(
            {
                "request_id": self.request_id,
                "scope": scope,
                "generated_at": now_iso(),
                "feature_count": len(feature_cols),
            }
        )
        write_json(output_path, summary)
        print(f"{scope} feature layer counts = {summary['counts']}")
        print(f"{scope} study_data_mode = {summary['study_data_mode']}")
        return summary

    @staticmethod
    def _write_selected_features(feature_cols: list[str]) -> None:
        pd.DataFrame(
            {
                "feature_name": feature_cols,
                "feature_layer": [infer_feature_layer(name) for name in feature_cols],
            }
        ).to_csv(SELECTED_FEATURES_PATH, index=False, encoding="utf-8-sig")

    @staticmethod
    def _available_feature_layers(layer_summary: dict[str, Any]) -> list[str]:
        counts = layer_summary.get("counts", {}) if isinstance(layer_summary, dict) else {}
        return [layer for layer in ["core", "behavior", "temporal", "interaction"] if int(counts.get(layer, 0)) > 0]

    @staticmethod
    def _audit_label_distribution(df: pd.DataFrame, label_col: str = "LABEL", compare_cols: list[str] | None = None) -> dict[str, Any]:
        if label_col not in df.columns:
            return {}

        result: dict[str, Any] = {
            "label_counts": df[label_col].value_counts(dropna=False).to_dict(),
        }
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [column for column in numeric_cols if column != label_col]
        pos_df = df[df[label_col] == 1]
        neg_df = df[df[label_col] == 0]
        sampled_cols = [column for column in (compare_cols or numeric_cols[:20]) if column in df.columns]

        result["positive_count"] = len(pos_df)
        result["negative_count"] = len(neg_df)
        result["positive_mean_sample"] = {
            column: float(pd.to_numeric(pos_df[column], errors="coerce").mean()) for column in sampled_cols
        }
        result["negative_mean_sample"] = {
            column: float(pd.to_numeric(neg_df[column], errors="coerce").mean()) for column in sampled_cols
        }
        return result

    def _write_label_audit_outputs(self, df: pd.DataFrame, label_col: str, feature_cols: list[str]) -> None:
        audit = self._audit_label_distribution(df, label_col=label_col, compare_cols=feature_cols[:20])
        if not audit:
            return

        # Add label subtype analysis if LABEL_SUBTYPE column exists
        if "LABEL_SUBTYPE" in df.columns:
            subtype_counts = df["LABEL_SUBTYPE"].value_counts(dropna=False).to_dict()
            audit["label_subtype_counts"] = {str(k): int(v) for k, v in subtype_counts.items()}

            # Compute positive rates by subtype
            pos_by_subtype = {}
            for subtype in df["LABEL_SUBTYPE"].dropna().unique():
                subtype_df = df[df["LABEL_SUBTYPE"] == subtype]
                pos_count = int((subtype_df[label_col] == 1).sum())
                total_count = len(subtype_df)
                pos_rate = pos_count / max(total_count, 1)
                pos_by_subtype[str(subtype)] = {
                    "positive_count": pos_count,
                    "total_count": total_count,
                    "positive_rate": pos_rate,
                }
            audit["label_subtype_positive_rates"] = pos_by_subtype

        audit.update(
            {
                "request_id": self.request_id,
                "generated_at": now_iso(),
                "label_col": label_col,
            }
        )
        write_json(LABEL_AUDIT_DETAIL_PATH, audit)

        positive_cols = [column for column in feature_cols if column in df.columns][:20]
        positive_df = df[df[label_col] == 1].copy()
        if positive_df.empty or not positive_cols:
            pd.DataFrame().to_csv(POSITIVE_LABEL_PROFILE_PATH, index=False, encoding="utf-8-sig")
            return

        profile = positive_df[positive_cols].apply(pd.to_numeric, errors="coerce").describe().transpose()
        profile.to_csv(POSITIVE_LABEL_PROFILE_PATH, encoding="utf-8-sig")

    @staticmethod
    def _score_estimator(model: Any, x: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(x))[:, 1]
        if hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(x), dtype=float)
            return 1.0 / (1.0 + np.exp(-decision))
        return np.asarray(model.predict(x), dtype=float)

    @staticmethod
    def _risk_level(score: float) -> str:
        if pd.isna(score):
            return "unknown"
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _model_importance(model: Any, features: list[str]) -> pd.Series:
        estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.ravel(np.asarray(estimator.coef_, dtype=float))
        else:
            values = np.ones(len(features), dtype=float)
        if len(values) != len(features):
            values = np.ones(len(features), dtype=float)
        return pd.Series(np.abs(values), index=features, dtype="float64")

    def _contribution_frame(self, model: Any, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        if data.empty or not features:
            return pd.DataFrame(index=data.index, columns=features)
        numeric = self._feature_matrix(data, features).fillna(0.0).abs()
        importance = self._model_importance(model, features)
        return numeric.mul(importance, axis=1)

    @staticmethod
    def _family_columns(table: pd.DataFrame, family_map: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
        return {family: [c for c in table.columns if any(c.startswith(prefix) for prefix in prefixes)] for family, prefixes in family_map.items()}

    @staticmethod
    def _family_presence(table: pd.DataFrame, family_columns: dict[str, list[str]]) -> dict[str, pd.Series]:
        return {
            family: (table[cols].notna().any(axis=1) if cols else pd.Series(False, index=table.index))
            for family, cols in family_columns.items()
        }

    @staticmethod
    def _safe_predict_with_fallback(primary_model: Any, fallback_model: Any, x: pd.DataFrame) -> tuple[np.ndarray, bool, str | None]:
        try:
            return StudyAgent._score_estimator(primary_model, x), False, None
        except Exception as exc:
            return StudyAgent._score_estimator(fallback_model, x), True, str(exc)

    def _resolve_study_modes(self, table: pd.DataFrame) -> pd.DataFrame:
        core_family_columns = self._family_columns(table, CORE_FAMILY_PREFIXES)
        behavior_family_columns = self._family_columns(table, BEHAVIOR_FAMILY_PREFIXES)
        core_presence = self._family_presence(table, core_family_columns)
        behavior_presence = self._family_presence(table, behavior_family_columns)

        core_available = core_presence.get("grade", pd.Series(False, index=table.index)) & core_presence.get("course", pd.Series(False, index=table.index))
        behavior_family_hits = sum(mask.astype(int) for mask in behavior_presence.values()) if behavior_presence else pd.Series(0, index=table.index)
        behavior_available = behavior_family_hits >= int(self.model_config.get("data_mode_rules", {}).get("behavior_available_min_family_hits", 1))

        mode = np.where(~core_available, "degraded_sparse", np.where(behavior_available, "core_plus_behavior", "core_only"))
        quality_flag = np.where(
            mode == "degraded_sparse",
            "degraded_sparse",
            np.where(mode == "core_plus_behavior", "behavior_ready", "core_stable"),
        )
        mode_frame = pd.DataFrame(
            {
                "XH": table["XH"].astype("string"),
                "TERM_ID": table["TERM_ID"].astype("string"),
                "CORE_AVAILABLE": core_available.astype(bool),
                "BEHAVIOR_FAMILY_HITS": behavior_family_hits.astype(int),
                "BEHAVIOR_AVAILABLE": behavior_available.astype(bool),
                "STUDY_DATA_MODE": mode.astype(str),
                "STUDY_QUALITY_FLAG": quality_flag.astype(str),
            }
        )
        self.row_modes = mode_frame
        if not mode_frame.empty:
            self.row_level_study_data_mode = str(mode_frame["STUDY_DATA_MODE"].mode(dropna=False).iloc[0])
            self.row_level_study_quality_flag = str(mode_frame["STUDY_QUALITY_FLAG"].mode(dropna=False).iloc[0])
            if not self.feature_layer_summary:
                self.study_data_mode = self.row_level_study_data_mode
            self.study_quality_flag = self.row_level_study_quality_flag
        return mode_frame

    def _feature_family_success(self, table: pd.DataFrame) -> tuple[dict[str, pd.Series], list[dict[str, Any]]]:
        family_masks: dict[str, pd.Series] = {}
        summary_rows: list[dict[str, Any]] = []
        for family, prefixes in FEATURE_FAMILY_PREFIXES.items():
            cols = [c for c in table.columns if any(c.startswith(prefix) for prefix in prefixes)]
            if cols:
                mask = table[cols].notna().any(axis=1)
            else:
                mask = pd.Series(False, index=table.index)
            family_masks[family] = mask
            layer_status = "unavailable_behavior"
            if family in CORE_FAMILY_PREFIXES:
                layer_status = "stable_core" if float(mask.mean()) >= 0.70 else "degraded_core"
            else:
                rate = float(mask.mean()) if len(mask) else 0.0
                if rate >= 0.50:
                    layer_status = "stable_behavior"
                elif rate >= 0.05:
                    layer_status = "recoverable_behavior"
            summary_rows.append(
                {
                    "family": family,
                    "feature_columns": cols,
                    "join_success_rate": float(mask.mean()) if len(mask) else 0.0,
                    "missing_rows": int((~mask).sum()) if len(mask) else 0,
                    "layer_status": layer_status,
                }
            )
        return family_masks, summary_rows

    def _categorical_profile(self, table: pd.DataFrame, train_table: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for column in table.columns:
            if column in {"XH", "TERM_ID", "DOMAIN"} or column.startswith("FEATURE_"):
                continue
            series = table[column]
            if is_numeric_dtype(series) and not is_string_dtype(series):
                continue
            infer_values = series.astype("string").str.strip()
            train_series = train_table[column].astype("string").str.strip() if column in train_table.columns else pd.Series(dtype="string")
            missing_mask = infer_values.isna() | infer_values.str.lower().isin({"", "<na>"})
            unknown_mask = infer_values.str.lower().isin(UNKNOWN_TOKENS)
            train_set = {str(v).strip().lower() for v in train_series.dropna().tolist() if str(v).strip()}
            drift_mask = infer_values.notna() & ~infer_values.str.lower().isin(train_set | UNKNOWN_TOKENS)
            rows.append(
                {
                    "field": column,
                    "raw_missing_count": int(missing_mask.sum()),
                    "unknown_or_unmapped_count": int(unknown_mask.sum()),
                    "category_drift_count": int(drift_mask.sum()),
                    "examples": infer_values[drift_mask].dropna().unique().tolist()[:5],
                }
            )
        return rows

    def _build_quality_report(self, table: pd.DataFrame) -> None:
        mode_frame = self._resolve_study_modes(table)
        key_df = table[["XH", "TERM_ID"]].copy()
        pk_valid = key_df["XH"].notna() & key_df["TERM_ID"].notna()
        duplicate_mask = key_df.astype("string").duplicated(keep=False) if not key_df.empty else pd.Series(dtype=bool)
        categorical_stats = self._categorical_profile(table, self.train)
        family_masks, family_summary = self._feature_family_success(table)

        coverage_rows = []
        selected_columns = ["XH", "TERM_ID", "SOURCE_COVERAGE", "DATA_QUALITY_FLAG"] + self.model_config.get("feature_columns", [])
        for column in dict.fromkeys([c for c in selected_columns if c in table.columns]):
            series = table[column]
            missing_rate = float(series.isna().mean()) if len(series) else 0.0
            coverage_rows.append(
                {
                    "field": column,
                    "non_null_rate": 1.0 - missing_rate,
                    "missing_rate": missing_rate,
                    "distinct_count": int(series.nunique(dropna=True)),
                }
            )

        unknown_by_field = pd.DataFrame(categorical_stats)
        unknown_rows = pd.Series(False, index=table.index)
        category_drift_rows = pd.Series(False, index=table.index)
        for column in [r["field"] for r in categorical_stats]:
            values = table[column].astype("string").str.strip()
            unknown_rows = unknown_rows | values.str.lower().isin(UNKNOWN_TOKENS)
            train_series = self.train[column].astype("string").str.strip() if column in self.train.columns else pd.Series(dtype="string")
            train_set = {str(v).strip().lower() for v in train_series.dropna().tolist() if str(v).strip()}
            category_drift_rows = category_drift_rows | (values.notna() & ~values.str.lower().isin(train_set | UNKNOWN_TOKENS))

        low_source_coverage = pd.to_numeric(table.get("SOURCE_COVERAGE", pd.Series(np.nan, index=table.index)), errors="coerce").fillna(0.0) < float(
            self.policy.get("thresholds", {}).get("min_source_coverage", 0.3)
        )
        high_missing = pd.to_numeric(table.get("FEATURE_MISSING_RATE", pd.Series(np.nan, index=table.index)), errors="coerce").fillna(1.0) >= 0.60
        core_missing_count = sum((~mask).astype(int) for mask in family_masks.values()) if family_masks else pd.Series(0, index=table.index)
        join_failure_rows = low_source_coverage & (core_missing_count >= 2)
        garbage_rows = high_missing & ~unknown_rows & ~category_drift_rows
        quality_degraded = (~pk_valid) | duplicate_mask | join_failure_rows | garbage_rows | unknown_rows | category_drift_rows

        root_cause = np.select(
            [
                (~pk_valid) | duplicate_mask,
                join_failure_rows,
                unknown_rows,
                category_drift_rows,
                garbage_rows,
            ],
            [
                "primary_key_issue",
                "join_failure_or_low_coverage",
                "field_unmapped_or_unknown",
                "category_drift",
                "garbage_or_sparse_data",
            ],
            default="ok",
        )
        warnings = np.select(
            [
                root_cause == "primary_key_issue",
                root_cause == "join_failure_or_low_coverage",
                root_cause == "field_unmapped_or_unknown",
                root_cause == "category_drift",
                root_cause == "garbage_or_sparse_data",
            ],
            [
                "missing or duplicated primary key",
                "feature family join likely failed or source coverage too low",
                "unknown or unmapped categorical value detected",
                "category not seen in training baseline",
                "feature coverage too sparse for reliable inference",
            ],
            default="quality checks passed",
        )
        self.row_quality = pd.DataFrame(
            {
                "XH": table["XH"].astype("string"),
                "TERM_ID": table["TERM_ID"].astype("string"),
                "PK_VALID": pk_valid.astype(bool),
                "PK_DUPLICATED": duplicate_mask.astype(bool),
                "LOW_SOURCE_COVERAGE": low_source_coverage.astype(bool),
                "HIGH_MISSING_RATE": high_missing.astype(bool),
                "UNKNOWN_VALUE_DETECTED": unknown_rows.astype(bool),
                "CATEGORY_DRIFT_DETECTED": category_drift_rows.astype(bool),
                "JOIN_FAILURE_DETECTED": join_failure_rows.astype(bool),
                "QUALITY_DEGRADED": quality_degraded.astype(bool),
                "ROOT_CAUSE": root_cause.astype(str),
                "QUALITY_WARNING": warnings.astype(str),
            }
        )
        association_rate = float((pk_valid & ~duplicate_mask).mean()) if len(table) else 0.0
        summary = {
            "request_id": self.request_id,
            "rows": int(len(table)),
            "field_coverage_warning_count": int(sum(row["missing_rate"] > 0.5 for row in coverage_rows)),
            "primary_key_association_success_rate": association_rate,
            "unknown_or_unmapped_rows": int(unknown_rows.sum()),
            "category_drift_rows": int(category_drift_rows.sum()),
            "join_failure_rows": int(join_failure_rows.sum()),
            "garbage_or_sparse_rows": int(garbage_rows.sum()),
            "quality_degraded_rows": int(quality_degraded.sum()),
            "study_data_mode": self.study_data_mode,
            "study_quality_flag": self.study_quality_flag,
        }
        self.quality_report = {
            "request_id": self.request_id,
            "domain": self.request.get("domain"),
            "generated_at": now_iso(),
            "summary": [summary],
            "field_coverage": coverage_rows,
            "primary_key_association": [
                {
                    "metric": "rows_with_valid_primary_key",
                    "value": int(pk_valid.sum()),
                },
                {
                    "metric": "rows_with_duplicate_primary_key",
                    "value": int(duplicate_mask.sum()),
                },
                {
                    "metric": "primary_key_association_success_rate",
                    "value": association_rate,
                },
                *family_summary,
            ],
            "unknown_value_stats": categorical_stats,
            "data_mode_stats": mode_frame["STUDY_DATA_MODE"].value_counts(dropna=False).rename_axis("study_data_mode").reset_index(name="rows").to_dict(orient="records") if not mode_frame.empty else [],
            "row_samples": self.row_quality[self.row_quality["QUALITY_DEGRADED"]].head(50).to_dict(orient="records"),
        }

    def build_features(self) -> None:
        full_table = self.train if self.request["run_mode"] == "train" else self.infer
        full_table = self._augment_feature_table(full_table)
        table = full_table
        if self.request["term_id"] != "all":
            table = full_table[full_table["TERM_ID"].astype(str) == str(self.request["term_id"])].copy()
        self.selected_rows = table
        feature_cols = [
            c
            for c in table.columns
            if c.startswith("FEATURE_")
            or c.startswith(("prev_", "hist_", "delta_", "ratio_", "trend_", "volatility_", "stability_", "cross__", "consecutive_decline_", "dist_from_worst_", "recovery_", "discordance_", "workload_stress"))
        ]
        model_features = self.model_config.get("feature_columns", [])
        missing_model_features = sorted(set(model_features) - set(self.selected_rows.columns))
        if missing_model_features:
            if self.request["run_mode"] == "infer":
                for column in missing_model_features:
                    self.selected_rows[column] = 0.0
            else:
                self._set_status("failed")
                self._decision("feature_build", "failed", "Required model features are missing.", {"missing_features": missing_model_features})
                self._trace("feature_built", "failed", "Feature table missing model features.", {"missing_features": missing_model_features})
                raise ValueError(f"missing model features: {missing_model_features}")

        serving_features = list(
            dict.fromkeys(
                (self.model_config.get("core_feature_columns") or [])
                + (self.model_config.get("behavior_feature_columns") or [])
            )
        ) or model_features
        serving_features = [column for column in serving_features if column in self.selected_rows.columns]
        summary_path = FEATURE_LAYER_SUMMARY_PATH if self.request["run_mode"] == "train" else INFER_FEATURE_LAYER_SUMMARY_PATH
        layer_summary = self._emit_feature_layer_summary(serving_features or feature_cols, summary_path, self.request["run_mode"])
        if self.request["run_mode"] == "train":
            self.feature_layer_summary = layer_summary
        else:
            self.infer_feature_layer_summary = layer_summary
            self.feature_layer_summary = layer_summary
        self.study_data_mode = str(layer_summary.get("study_data_mode", self.study_data_mode))
        self._write_selected_features(serving_features or feature_cols)
        if self.request["run_mode"] == "train" and self.model_config.get("label_name", "LABEL") in self.selected_rows.columns:
            self._write_label_audit_outputs(self.selected_rows, self.model_config.get("label_name", "LABEL"), serving_features or feature_cols)

        self.summary_metrics.update(
            {
                "request_rows": int(len(table)),
                "feature_columns": int(len(feature_cols)),
                "model_feature_columns": int(len(model_features)),
                "study_data_mode": self.study_data_mode,
            }
        )
        self._decision("feature_build", "success", "Reused existing train/infer feature table for request.")
        self._trace("feature_built", "success", "Features resolved from existing delivery table.", self.summary_metrics)

    def quality_checked(self) -> None:
        table = self.selected_rows
        self._state("validating", self.status, "Running independent quality gate before inference.")
        self._build_quality_report(table)
        self._write_quality_report()
        summary = (self.quality_report.get("summary") or [{}])[0]
        low_coverage_rate = float(summary.get("join_failure_rows", 0)) / max(int(summary.get("rows", 0)), 1)
        l456_features = [c for c in table.columns if c.startswith(L456_PREFIXES)]
        l456_nonnull_rate = float(table[l456_features].notna().any(axis=1).mean()) if l456_features and len(table) else 0.0
        degraded_ratio = float(summary.get("quality_degraded_rows", 0)) / max(int(summary.get("rows", 0)), 1)
        sparse_ratio = float((self.row_modes.get("STUDY_DATA_MODE", pd.Series(dtype=str)) == "degraded_sparse").mean()) if not self.row_modes.empty else 0.0
        behavior_ratio = float((self.row_modes.get("STUDY_DATA_MODE", pd.Series(dtype=str)) == "core_plus_behavior").mean()) if not self.row_modes.empty else 0.0
        source_coverage_mean = float(pd.to_numeric(table.get("SOURCE_COVERAGE", pd.Series(np.nan, index=table.index)), errors="coerce").mean()) if len(table) else None
        low_coverage_threshold = float(self.policy.get("thresholds", {}).get("low_coverage_warning", 0.5))
        l456_threshold = float(self.policy.get("thresholds", {}).get("l456_min_nonnull_rate", 0.2))

        if sparse_ratio > low_coverage_threshold:
            self._set_status("degraded")
            self.warnings.append(f"degraded_sparse ratio is high: {sparse_ratio:.2%}")
        if l456_nonnull_rate < l456_threshold:
            self.warnings.append(f"L4/L5/L6 behavior coverage is low: {l456_nonnull_rate:.2%}")

        metrics = {
            "low_coverage_rate": low_coverage_rate,
            "source_coverage_mean": source_coverage_mean,
            "l456_nonnull_rate": l456_nonnull_rate,
            "primary_key_association_success_rate": summary.get("primary_key_association_success_rate"),
            "unknown_or_unmapped_rows": summary.get("unknown_or_unmapped_rows"),
            "category_drift_rows": summary.get("category_drift_rows"),
            "join_failure_rows": summary.get("join_failure_rows"),
            "garbage_or_sparse_rows": summary.get("garbage_or_sparse_rows"),
            "quality_degraded_rows": summary.get("quality_degraded_rows"),
            "study_data_mode": self.study_data_mode,
            "row_level_study_data_mode": self.row_level_study_data_mode,
            "study_quality_flag": self.study_quality_flag,
            "core_plus_behavior_ratio": behavior_ratio,
            "degraded_sparse_ratio": sparse_ratio,
        }
        self.summary_metrics.update(metrics)
        self._decision("quality_check", self.status, "Independent quality gate completed with field coverage, primary-key association, and unknown-value root causes.", metrics)
        self._trace("quality_checked", self.status, "Quality checks completed.", metrics)

    def run_model(self) -> None:
        self._state("scoring", self.status, "Scoring current request rows with serving model.")
        table = self.selected_rows.copy()
        if table.empty:
            self._set_status("failed")
            raise ValueError("no rows selected for inference")

        bundle = self._load_model_bundle()
        mode_frame = self.row_modes.set_index(["XH", "TERM_ID"]) if not self.row_modes.empty else pd.DataFrame()
        core_features = self.model_config.get("core_feature_columns") or self.model_config.get("feature_columns", [])
        behavior_features = self.model_config.get("behavior_feature_columns", [])
        behavior_corrector_features = self.model_config.get("behavior_corrector_feature_columns", behavior_features)
        subgroup_features = self.model_config.get("subgroup_feature_columns", [])
        selected_feature_file = SELECTED_FEATURES_PATH
        if selected_feature_file.exists():
            selected_feature_df = pd.read_csv(selected_feature_file)
            selected_feature_names = selected_feature_df["feature_name"].astype(str).tolist()
            for column in selected_feature_names:
                if column not in table.columns:
                    table[column] = 0.0
        infer_feature_cols = list(dict.fromkeys(core_features + behavior_corrector_features + subgroup_features))
        self.infer_feature_layer_summary = self._emit_feature_layer_summary(infer_feature_cols, INFER_FEATURE_LAYER_SUMMARY_PATH, "infer")
        self.study_data_mode = str(self.infer_feature_layer_summary.get("study_data_mode", self.study_data_mode))
        self._write_selected_features(infer_feature_cols)
        core_x = self._feature_matrix(table, core_features)
        behavior_x = self._feature_matrix(table, behavior_corrector_features) if behavior_corrector_features else pd.DataFrame(index=table.index)
        subgroup_x = self._feature_matrix(table, subgroup_features) if subgroup_features else pd.DataFrame(index=table.index)
        print(f"infer feature count = {core_x.shape[1] + behavior_x.shape[1] + subgroup_x.shape[1]}")

        # Phase 1: Core model prediction (always runs)
        core_prob, core_fallback_used, core_error = self._safe_predict_with_fallback(
            bundle.get("core_model", bundle["primary_model"]),
            bundle.get("core_fallback_model", bundle["fallback_model"]),
            core_x,
        )
        core_scores = np.asarray(core_prob, dtype=float)

        policy = resolve_policy(self.model_config.get("routing_policy"))
        core_confidence = np.abs(core_scores - 0.5) * 2.0
        middle_zone_candidates = (core_scores > policy.low_conf_lower) & (core_scores < policy.low_conf_upper)
        behavior_prob = np.full(len(table), np.nan)
        behavior_error = None
        subgroup_prob = np.full(len(table), np.nan)
        subgroup_error = None

        data_mode_series = (
            self.row_modes["STUDY_DATA_MODE"].astype(str)
            if not self.row_modes.empty
            else pd.Series(["core_only"] * len(table), index=table.index)
        )
        behavior_ready = data_mode_series.eq("core_plus_behavior").to_numpy()
        corrector_candidates = middle_zone_candidates & behavior_ready

        if bundle.get("behavior_model") is not None and behavior_corrector_features:
            behavior_input = behavior_x.copy()
            if "BASE_SCORE_OOF" in behavior_input.columns:
                behavior_input["BASE_SCORE_OOF"] = core_scores
            if corrector_candidates.any():
                behavior_scores, behavior_fallback_used, behavior_error = self._safe_predict_with_fallback(
                    bundle["behavior_model"],
                    bundle.get("behavior_fallback_model") or bundle["behavior_model"],
                    behavior_input.loc[corrector_candidates],
                )
                behavior_prob[corrector_candidates] = behavior_scores
                self.fallback_used = bool(core_fallback_used or behavior_fallback_used)
            else:
                self.fallback_used = bool(core_fallback_used)
        else:
            self.fallback_used = bool(core_fallback_used)

        if bundle.get("subgroup_model") is not None and subgroup_features:
            subgroup_input = subgroup_x.copy()
            if "BASE_SCORE_OOF" in subgroup_input.columns:
                subgroup_input["BASE_SCORE_OOF"] = core_scores
            if corrector_candidates.any():
                subgroup_scores, subgroup_fallback_used, subgroup_error = self._safe_predict_with_fallback(
                    bundle["subgroup_model"],
                    bundle.get("subgroup_fallback_model") or bundle["subgroup_model"],
                    subgroup_input.loc[corrector_candidates],
                )
                subgroup_prob[corrector_candidates] = subgroup_scores
                self.fallback_used = bool(self.fallback_used or subgroup_fallback_used)

        subtype_signal = np.where(
            table.get("LABEL_SUBTYPE", pd.Series("", index=table.index)).astype(str).eq("single_fail"),
            1.0,
            np.nan,
        )
        routed_frame = apply_serving_policy(
            base_score=core_scores,
            behavior_signal=behavior_prob,
            subgroup_signal=subgroup_prob,
            data_mode=data_mode_series,
            subtype_signal=subtype_signal,
            policy=policy,
        )
        routed_prob = routed_frame["FINAL_SCORE"].to_numpy()
        behavior_delta = routed_frame["BEHAVIOR_DELTA"].to_numpy()
        subgroup_delta = routed_frame["SUBGROUP_DELTA"].to_numpy()
        routing_detail = routed_frame["ROUTING_REASON"].astype(str).to_numpy()
        confidence_zone = routed_frame["CONFIDENCE_ZONE"].astype(str).to_numpy()
        blend_prob = np.clip(core_scores + behavior_delta, 0.0, 1.0)

        self.scoring_model_source = "layered_core_behavior"
        self.scoring_model_name = self.model_config.get("primary_model", "study_core_model")
        row_quality = self.row_quality.set_index(["XH", "TERM_ID"]) if not self.row_quality.empty else pd.DataFrame()
        quality_status = []
        root_causes = []
        row_modes = []
        quality_flags = []
        for _, row in table.iterrows():
            key = (str(row["XH"]), str(row["TERM_ID"]))
            if not row_quality.empty and key in row_quality.index:
                quality_status.append(bool(row_quality.loc[key, "QUALITY_DEGRADED"]))
                root_causes.append(str(row_quality.loc[key, "ROOT_CAUSE"]))
            else:
                quality_status.append(False)
                root_causes.append("ok")
            if not mode_frame.empty and key in mode_frame.index:
                row_modes.append(str(mode_frame.loc[key, "STUDY_DATA_MODE"]))
                quality_flags.append(str(mode_frame.loc[key, "STUDY_QUALITY_FLAG"]))
            else:
                row_modes.append("core_only")
                quality_flags.append("core_stable")

        failed = pd.Series(routed_prob).isna()
        degraded_mask = pd.Series(row_modes, index=table.index).eq("degraded_sparse") | self.fallback_used
        status = np.where(failed, "failed", np.where(degraded_mask, "degraded", "success"))
        predictions = pd.DataFrame(
            {
                "REQUEST_ID": self.request_id,
                "XH": table["XH"].astype("string"),
                "TERM_ID": table["TERM_ID"].astype("string"),
                "DOMAIN": self.model_config.get("domain", "study"),
                "BASE_SCORE": np.asarray(core_prob, dtype=float),
                "BEHAVIOR_DELTA": behavior_delta,
                "SUBGROUP_DELTA": subgroup_delta,
                "FINAL_SCORE": np.asarray(routed_prob, dtype=float),
                "STUDY_CORE_SCORE": np.asarray(core_prob, dtype=float),
                "STUDY_BEHAVIOR_SCORE": behavior_prob,
                "STUDY_SUBGROUP_SCORE": subgroup_prob,
                "STUDY_BLEND_SCORE": np.asarray(blend_prob, dtype=float),
                "STUDY_ROUTED_SCORE": np.asarray(routed_prob, dtype=float),
                "STUDY_FINAL_SCORE": np.asarray(routed_prob, dtype=float),
                "DOMAIN_SCORE": np.asarray(routed_prob, dtype=float),
                "DOMAIN_CONFIDENCE": core_confidence,
                "CONFIDENCE_ZONE": confidence_zone,
                "ROUTING_REASON": routing_detail,
                "ROUTING_DETAIL": routing_detail,
                "RISK_LEVEL": [self._risk_level(float(score)) for score in routed_prob],
                "PREDICTED_LABEL": (np.asarray(routed_prob, dtype=float) >= 0.5).astype(int),
                "MODEL_VERSION": self.model_config.get("model_version", "study_v1"),
                "FEATURE_VERSION": self.model_config.get("feature_version", "study_feature_v1"),
                "MODEL_NAME": self.scoring_model_name,
                "MODEL_SOURCE": self.scoring_model_source,
                "ROUTING_STRATEGY": "base_main_ranking; middle_zone_only->behavior_residual/single_fail_expert; degraded_sparse->base_only",
                "STATUS": status,
                "FALLBACK_USED": bool(self.fallback_used),
                "STUDY_DATA_MODE": row_modes,
                "STUDY_QUALITY_FLAG": quality_flags,
                "QUALITY_DEGRADED": quality_status,
                "QUALITY_ROOT_CAUSE": root_causes,
                "SOURCE_TABLE": self.request["run_mode"],
                "LABEL_SUBTYPE": table.get("LABEL_SUBTYPE", pd.Series(pd.NA, index=table.index)).astype("string"),
            }
        )
        self.selected_predictions = predictions
        self.prediction = predictions.copy()

        # Write to request path (canonical output location)
        self._write_table(self.paths["prediction_output"], predictions)

        # Sync to DM_DIR (internal tracking)
        self._write_table(DM_DIR / "study_prediction_output.csv", predictions)

        # Sync to deliverable docs
        deliverable_prediction_path = ROOT / "data" / "deliverables" / "study" / "data" / "study_prediction_output.csv"
        deliverable_prediction_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_table(deliverable_prediction_path, predictions)

        numeric_score = pd.to_numeric(predictions["DOMAIN_SCORE"], errors="coerce")
        if numeric_score.isna().any():
            self._set_status("failed")
            raise ValueError("invalid domain scores generated during live inference")

        status_counts = predictions["STATUS"].value_counts(dropna=False).to_dict()
        degraded_count = int((predictions.get("STATUS", pd.Series(dtype=str)).astype(str) == "degraded").sum())
        failed_count = int((predictions.get("STATUS", pd.Series(dtype=str)).astype(str) == "failed").sum())
        success_count = int((predictions.get("STATUS", pd.Series(dtype=str)).astype(str) == "success").sum())
        degraded_ratio = degraded_count / max(len(predictions), 1)
        if failed_count:
            self._set_status("failed")
        elif self.fallback_used or degraded_ratio > float(self.policy.get("thresholds", {}).get("degraded_ratio_warning", 0.5)):
            self._set_status("degraded")

        metrics = {
            "prediction_rows": int(len(predictions)),
            "success_count": success_count,
            "degraded_count": degraded_count,
            "failed_count": failed_count,
            "degraded_ratio": degraded_ratio,
            "status_counts": status_counts,
            "model_name": self.scoring_model_name,
            "model_source": self.scoring_model_source,
            "routing_strategy": "base_main_ranking; middle_zone_only->behavior_residual/single_fail_expert; degraded_sparse->base_only",
            "study_data_mode": self.study_data_mode,
            "row_level_study_data_mode": self.row_level_study_data_mode,
            "study_quality_flag": self.study_quality_flag,
            "core_only_count": int((predictions["STUDY_DATA_MODE"] == "core_only").sum()),
            "core_plus_behavior_count": int((predictions["STUDY_DATA_MODE"] == "core_plus_behavior").sum()),
            "degraded_sparse_count": int((predictions["STUDY_DATA_MODE"] == "degraded_sparse").sum()),
            "high_conf_positive_count": int((predictions["CONFIDENCE_ZONE"] == "high_conf_positive").sum()),
            "middle_correction_count": int((predictions["CONFIDENCE_ZONE"] == "middle_correction").sum()),
            "high_conf_negative_count": int((predictions["CONFIDENCE_ZONE"] == "high_conf_negative").sum()),
            "infer_feature_count": int(core_x.shape[1] + behavior_x.shape[1] + subgroup_x.shape[1]),
        }
        if core_error:
            metrics["core_model_error"] = core_error
        if behavior_error:
            metrics["behavior_model_error"] = behavior_error
        if subgroup_error:
            metrics["subgroup_model_error"] = subgroup_error
        self.summary_metrics.update(metrics)
        self._decision("model_execution", self.status, "Layered study inference completed with core model plus optional behavior module.", metrics)
        self._trace("model_executed", self.status, "Layered model result generated from live scoring.", metrics)

    def fallback_checked(self) -> None:
        fallback_series = self.selected_predictions.get("FALLBACK_USED", pd.Series(False, index=self.selected_predictions.index))
        self.fallback_used = bool(fallback_series.astype(str).str.lower().isin({"true", "1", "yes"}).any())
        model_file_exists = self.paths["model_file"].exists()
        if not model_file_exists and self.request.get("enable_fallback", False):
            self._set_status("degraded")
            self.warnings.append("Primary model artifact is missing; fallback policy would be required.")
        if self.fallback_used:
            self._set_status("degraded")
            self.warnings.append("Fallback scoring was used in the layered study model.")

        sparse_exists = bool((self.selected_predictions.get("STUDY_DATA_MODE", pd.Series(dtype=str)).astype(str) == "degraded_sparse").any())
        publish_allowed = bool(not self.fallback_used and not sparse_exists and self.status == "success")
        metrics = {
            "fallback_used": self.fallback_used,
            "model_file_exists": model_file_exists,
            "publish_candidate_allowed": publish_allowed,
            "study_data_mode": self.study_data_mode,
        }
        self.summary_metrics["publish_candidate_allowed"] = publish_allowed
        self._decision("fallback_check", self.status, "Fallback decision evaluated after live scoring.", metrics)
        self._decision("publish_candidate_gate", "success" if publish_allowed else "degraded", "Publish-candidate entry is allowed only when no fallback is used and inference is not degraded.", metrics)
        self._trace("fallback_checked", self.status, "Fallback check completed.", metrics)

    def generate_explanation(self) -> None:
        if not self.request.get("enable_explanation", False):
            self._decision("explanation", self.status, "Explanation disabled by request.")
            self._trace("explanation_generated", self.status, "Explanation disabled by request.", {})
            return

        self._state("explaining", self.status, "Generating explanation bound to current prediction rows.")
        bundle = self._load_model_bundle()
        table = self.selected_rows.reset_index(drop=True).copy()
        prediction = self.selected_predictions.reset_index(drop=True).copy()
        core_model = bundle.get("core_fallback_model") if self.fallback_used else bundle.get("core_model", bundle["primary_model"])
        behavior_model = None
        if bundle.get("behavior_model") is not None:
            behavior_model = bundle.get("behavior_fallback_model") if self.fallback_used and bundle.get("behavior_fallback_model") is not None else bundle.get("behavior_model")
        core_contrib = self._contribution_frame(core_model, table, self.model_config.get("core_feature_columns") or self.model_config.get("feature_columns", []))
        behavior_contrib = self._contribution_frame(behavior_model, table, self.model_config.get("behavior_feature_columns", [])) if behavior_model is not None else pd.DataFrame(index=table.index)

        rows = []
        for idx, row in table.iterrows():
            mode = str(prediction.iloc[idx].get("STUDY_DATA_MODE", "core_only"))
            contribution = core_contrib.loc[idx] if idx in core_contrib.index else pd.Series(dtype=float)
            if mode == "core_plus_behavior" and idx in behavior_contrib.index:
                contribution = pd.concat([contribution, behavior_contrib.loc[idx]]).groupby(level=0).sum()
            ranked = contribution.fillna(0.0).sort_values(ascending=False)
            top = [name for name in ranked.head(3).index if ranked.get(name, 0) > 0]
            while len(top) < 3:
                top.append("")
            pred_row = prediction.iloc[idx]
            values = [row.get(name, "") if name else "" for name in top]
            quality_warning = ""
            if not self.row_quality.empty:
                matched = self.row_quality[(self.row_quality["XH"].astype(str) == str(row["XH"])) & (self.row_quality["TERM_ID"].astype(str) == str(row["TERM_ID"]))]
                if not matched.empty:
                    quality_warning = str(matched.iloc[0]["QUALITY_WARNING"])
            bits = [f"{name}={row.get(name)}" for name in top if name]
            text = (
                f"本次推理为 {mode} 模式，核心学习分 {pred_row['STUDY_CORE_SCORE']:.4f}"
                + (f"，行为增强分 {pred_row['STUDY_BEHAVIOR_SCORE']:.4f}" if pd.notna(pred_row.get("STUDY_BEHAVIOR_SCORE")) else "")
                + f"，最终分 {pred_row['STUDY_FINAL_SCORE']:.4f}，主要特征为 "
                + "；".join(bits)
            ) if bits else f"本次推理为 {mode} 模式，当前特征贡献不足以形成稳定解释。"
            if quality_warning and quality_warning != "quality checks passed":
                text = f"{text}；质量提醒：{quality_warning}"
            rows.append(
                {
                    "REQUEST_ID": self.request_id,
                    "XH": row["XH"],
                    "TERM_ID": row["TERM_ID"],
                    "DOMAIN": self.model_config.get("domain", "study"),
                    "BASE_SCORE": pred_row.get("BASE_SCORE"),
                    "BEHAVIOR_DELTA": pred_row.get("BEHAVIOR_DELTA"),
                    "SUBGROUP_DELTA": pred_row.get("SUBGROUP_DELTA"),
                    "FINAL_SCORE": pred_row.get("FINAL_SCORE", pred_row.get("STUDY_FINAL_SCORE")),
                    "ROUTING_REASON": pred_row.get("ROUTING_REASON"),
                    "CONFIDENCE_ZONE": pred_row.get("CONFIDENCE_ZONE"),
                    "ENTERED_BEHAVIOR_CORRECTOR": bool(abs(float(pred_row.get("BEHAVIOR_DELTA", 0) or 0)) > 0),
                    "ENTERED_SINGLE_FAIL_EXPERT": bool(abs(float(pred_row.get("SUBGROUP_DELTA", 0) or 0)) > 0),
                    "STUDY_CORE_SCORE": pred_row["STUDY_CORE_SCORE"],
                    "STUDY_BEHAVIOR_SCORE": pred_row["STUDY_BEHAVIOR_SCORE"],
                    "STUDY_FINAL_SCORE": pred_row["STUDY_FINAL_SCORE"],
                    "DOMAIN_SCORE": pred_row["DOMAIN_SCORE"],
                    "DOMAIN_CONFIDENCE": pred_row["DOMAIN_CONFIDENCE"],
                    "RISK_LEVEL": pred_row["RISK_LEVEL"],
                    "STUDY_DATA_MODE": pred_row.get("STUDY_DATA_MODE"),
                    "STUDY_QUALITY_FLAG": pred_row.get("STUDY_QUALITY_FLAG"),
                    "MODEL_VERSION": self.model_config.get("model_version", "study_v1"),
                    "FEATURE_VERSION": self.model_config.get("feature_version", "study_feature_v1"),
                    "MODEL_NAME": self.scoring_model_name,
                    "MODEL_SOURCE": self.scoring_model_source,
                    "TOP_FEATURE_1": top[0],
                    "TOP_FEATURE_1_VALUE": values[0],
                    "TOP_FEATURE_2": top[1],
                    "TOP_FEATURE_2_VALUE": values[1],
                    "TOP_FEATURE_3": top[2],
                    "TOP_FEATURE_3_VALUE": values[2],
                    "QUALITY_ROOT_CAUSE": pred_row.get("QUALITY_ROOT_CAUSE"),
                    "EXPLANATION_TEXT": text,
                    "SOURCE_TABLE": self.request["run_mode"],
                }
            )

        explanations = pd.DataFrame(rows)
        self.selected_explanations = explanations
        self.explanation = explanations.copy()
        self._write_table(self.paths["explanation_output"], explanations)
        self._write_table(DM_DIR / "study_explanation_output.csv", explanations)
        mode_keys = self.selected_predictions[["XH", "TERM_ID", "DOMAIN"]].astype(str).drop_duplicates()
        exp_keys = explanations[["XH", "TERM_ID", "DOMAIN"]].astype(str).drop_duplicates()
        aligned = len(mode_keys.merge(exp_keys, on=["XH", "TERM_ID", "DOMAIN"], how="inner"))
        if aligned != len(mode_keys):
            self._set_status("degraded")
            self.warnings.append("Explanation output does not fully align with current prediction rows.")

        metrics = {"explanation_rows": int(len(explanations)), "aligned_prediction_rows": int(aligned), "model_used_for_explanation": self.scoring_model_name}
        self.summary_metrics.update(metrics)
        self._decision("explanation", self.status, "Explanation regenerated from current inference outputs and current model context.", metrics)
        self._trace("explanation_generated", self.status, "Explanation generation completed.", metrics)

    def run_llm_assistant(self) -> None:
        if not self.request.get("llm_enable", False):
            return
        if self.request["run_mode"] == "train" and not self.request.get("llm_use_for_model_review", True):
            return
        if self.request["run_mode"] == "infer" and not (
            self.request.get("llm_use_for_explanation", False) or self.request.get("llm_use_for_model_review", False)
        ):
            return
        try:
            try:
                from study_llm_assistant import LLM_REVIEW_PATH, build_llm_advice
            except ModuleNotFoundError:  # pragma: no cover - package import path
                from .study_llm_assistant import LLM_REVIEW_PATH, build_llm_advice

            self.llm_review = build_llm_advice(self.request)
            status = self.llm_review.get("response_status", "success")
            if status not in {"success", "skipped"}:
                self.warnings.append(f"LLM review degraded: {self.llm_review.get('error_message', 'unknown error')}")
                if self.request.get("llm_required", False):
                    self._set_status("failed")
            self.summary_metrics["llm_review_status"] = status
            self.summary_metrics["llm_review_provider"] = self.llm_review.get("provider")
            self._decision(
                "llm_review",
                "failed" if self.request.get("llm_required", False) and status not in {"success", "skipped"} else self.status,
                "LLM review completed without changing control decisions.",
                {
                    "provider": self.llm_review.get("provider"),
                    "requested_provider": self.llm_review.get("requested_provider"),
                    "response_status": status,
                    "review_path": str(LLM_REVIEW_PATH),
                },
            )
            self._trace(
                "llm_reviewed",
                self.status,
                "LLM model review/explanation advice completed.",
                {"provider": self.llm_review.get("provider"), "response_status": status},
            )
        except Exception as exc:
            self.warnings.append(f"LLM assistant failed: {exc}")
            if self.request.get("llm_required", False):
                self._set_status("failed")
            self._decision("llm_review", self.status, "LLM assistant failed non-fatally.", {"error": str(exc)})
            self._trace("llm_reviewed", self.status, "LLM assistant failed non-fatally.", {"error": str(exc)})

    def run_evolution(self) -> None:
        try:
            try:
                from study_evolution import StudyEvolutionEngine
            except ModuleNotFoundError:  # pragma: no cover
                from .study_evolution import StudyEvolutionEngine

            engine = StudyEvolutionEngine(self.request)
            evolution_result = engine.run()
            self.extra_result["evolution"] = evolution_result
            self.summary_metrics["evolution_candidate_count"] = evolution_result.get("candidate_count")
            self.summary_metrics["publish_candidate_version_id"] = evolution_result.get("publish_candidate", {}).get("version_id")
            self.model_selection = {
                "selected_primary_model": evolution_result.get("publish_candidate", {}).get("model_name"),
                "selected_feature_group": evolution_result.get("publish_candidate", {}).get("feature_group"),
                "selected_threshold_strategy": evolution_result.get("publish_candidate", {}).get("threshold_strategy"),
            }
            self._decision("evolution", "success", "Evolution layer generated comparison, selection, and publish candidate.", self.summary_metrics)
            self._trace("evolution_completed", self.status, "Evolution layer completed.", self.summary_metrics)
        except Exception as exc:
            self._set_status("failed")
            self.warnings.append(f"Evolution failed: {exc}")
            self._decision("evolution", "failed", str(exc))
            self._trace("evolution_completed", "failed", "Evolution layer failed.", {"error": str(exc)})

    def run_publish(self) -> None:
        try:
            try:
                from study_release_manager import StudyReleaseManager
            except ModuleNotFoundError:  # pragma: no cover
                from .study_release_manager import StudyReleaseManager

            manager = StudyReleaseManager(self.request)
            result = manager.publish(
                candidate_version_id=self.request.get("candidate_version_id"),
                dry_run=bool(self.request.get("dry_run", True)),
                require_approval=bool(self.request.get("require_approval", True)),
            )
            self.extra_result["release_action"] = result
            self.summary_metrics["release_action_status"] = result.get("status")
            self._decision("publish", result.get("status", "unknown"), "Release manager evaluated publish request.", result)
            self._trace("publish_evaluated", self.status, "Publish request evaluated.", {"status": result.get("status")})
        except Exception as exc:
            self._set_status("failed")
            self.warnings.append(f"Publish failed: {exc}")
            self._decision("publish", "failed", str(exc))

    def run_rollback(self) -> None:
        try:
            try:
                from study_release_manager import StudyReleaseManager
            except ModuleNotFoundError:  # pragma: no cover
                from .study_release_manager import StudyReleaseManager

            manager = StudyReleaseManager(self.request)
            result = manager.rollback(
                target_version_id=self.request.get("target_version_id"),
                dry_run=bool(self.request.get("dry_run", True)),
            )
            self.extra_result["release_action"] = result
            self.summary_metrics["release_action_status"] = result.get("status")
            self._decision("rollback", result.get("status", "unknown"), "Release manager evaluated rollback request.", result)
            self._trace("rollback_evaluated", self.status, "Rollback request evaluated.", {"status": result.get("status")})
        except Exception as exc:
            self._set_status("failed")
            self.warnings.append(f"Rollback failed: {exc}")
            self._decision("rollback", "failed", str(exc))

    def train_layered_model(self) -> None:
        self._state("scoring", self.status, "Training layered study core model and optional behavior module.")
        runpy.run_path(str(ROOT / "src" / "30_train_study_model.py"), run_name="__main__")

        # Sync DM_DIR artifacts to deliverable paths
        deliverable_model_dir = ROOT / "data" / "deliverables" / "study" / "model"
        deliverable_model_dir.mkdir(parents=True, exist_ok=True)

        # Core model artifacts
        for name in ["study_model.pkl", "study_model_config.json", "study_model_metrics.json"]:
            src = DM_DIR / name
            if src.exists():
                shutil.copy2(src, deliverable_model_dir / name)

        # Feature artifacts
        for name in ["study_feature_layer_summary.json", "study_selected_features.csv",
                     "study_feature_screening_report.json", "study_subgroup_feature_screening_report.json"]:
            src = DM_DIR / name
            dst = ROOT / "data" / "deliverables" / "study" / "docs" / name
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # Reload model_config from deliverable (canonical source for inference)
        self.model_config = self._read_json(deliverable_model_dir / "study_model_config.json")
        metrics = self.model_config.get("metrics", {}).get("core_model", {}).get("valid", {})
        self.summary_metrics.update(
            {
                "trained_architecture_version": self.model_config.get("architecture_version"),
                "trained_core_feature_count": len(self.model_config.get("core_feature_columns", [])),
                "trained_behavior_feature_count": len(self.model_config.get("behavior_feature_columns", [])),
                "trained_core_valid_auc": metrics.get("auc"),
                "trained_core_valid_f1": metrics.get("f1"),
                "trained_core_valid_recall": metrics.get("recall"),
            }
        )
        self._decision("layered_model_training", "success", "study_core_model and optional behavior module were trained and published to deliverables.", self.summary_metrics)
        self._trace("layered_model_trained", self.status, "Layered study model training completed.", self.summary_metrics)

    def run_train_enhancement(self) -> None:
        if not self.request.get("enable_model_search", True):
            self._decision("model_search", "success", "Model search disabled by request.")
            self._trace("model_executed", self.status, "Train mode skipped candidate search by request.", {})
            return

        label_name = self.model_config.get("label_name", "LABEL")
        if label_name not in self.selected_rows.columns:
            self._set_status("failed")
            raise ValueError(f"{label_name} is required for train mode")

        base_features = [
            c
            for c in self.selected_rows.columns
            if c not in {"XH", "TERM_ID", label_name, "DOMAIN"}
            and c not in QUALITY_FEATURES
            and (
                c.startswith("FEATURE_")
                or c.startswith(("prev_", "hist_", "delta_", "ratio_", "trend_", "volatility_", "stability_", "cross__"))
            )
        ]
        train_df = self.selected_rows.dropna(subset=[label_name]).copy()
        y = pd.to_numeric(train_df[label_name], errors="coerce")
        valid_mask = y.notna()
        train_df = train_df.loc[valid_mask].copy()
        y = y.loc[valid_mask].astype(int)
        if y.nunique() < 2:
            self._set_status("failed")
            raise ValueError("train mode requires at least two label classes")

        feature_groups = self.request.get("candidate_feature_groups") or self._allowed("feature_groups")
        if self.request.get("enable_llm_feature_synthesis", False) or "llm_synthesized" in feature_groups:
            train_df = self._add_llm_synthesized_features(train_df)

        train_df = self._augment_feature_table(train_df)
        base_features = [feature for feature in base_features if feature in train_df.columns]

        split_kwargs = {"test_size": 0.2, "random_state": 42, "stratify": y}
        train_idx, valid_idx = train_test_split(train_df.index, **split_kwargs)
        y_train = y.loc[train_idx]
        y_valid = y.loc[valid_idx]

        rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        branch_frames: list[pd.DataFrame] = []
        failures: list[str] = []
        model_names = self.request.get("candidate_models") or self._allowed("candidate_models")
        threshold_strategies = self.request.get("threshold_strategy") or self._allowed("threshold_strategies")
        if isinstance(threshold_strategies, str):
            threshold_strategies = [threshold_strategies]

        for feature_group in feature_groups:
            features = self._resolve_feature_group(feature_group, base_features, train_df, y)
            if not features:
                failures.append(f"{feature_group}: no usable features")
                continue
            feature_layer_summary = summarize_feature_layers(features)
            behavior_cols = feature_layer_summary["columns"]["behavior"]
            temporal_cols = feature_layer_summary["columns"]["temporal"]
            interaction_cols = feature_layer_summary["columns"]["interaction"]
            print(f"training feature group {feature_group} layer counts = {feature_layer_summary['counts']}")
            if not behavior_cols:
                self.warnings.append(f"behavior layer not used in training group {feature_group}")
            coverage = float(train_df[features].notna().mean().mean())
            X_train = train_df.loc[train_idx, features]
            X_valid = train_df.loc[valid_idx, features]
            print(f"train feature count = {len(features)}")
            for model_name in model_names:
                model = self._make_candidate_model(model_name)
                if model is None:
                    failures.append(f"{model_name}: dependency unavailable")
                    continue
                try:
                    model.fit(X_train, y_train)
                    proba = self._predict_proba(model, X_valid)
                    auc = self._safe_auc(y_valid, proba)
                    for strategy in threshold_strategies:
                        threshold, tuning = self._select_threshold(y_valid, proba, strategy, feature_group, model_name)
                        metrics = self._classification_metrics(y_valid, proba, threshold)
                        robustness = self._robustness_score(auc, metrics["f1"], metrics["recall"], coverage)
                        rows.append(
                            {
                                "request_id": self.request_id,
                                "feature_group": feature_group,
                                "model_name": model_name,
                                "threshold_strategy": strategy,
                                "threshold": threshold,
                                "auc": auc,
                                "f1": metrics["f1"],
                                "recall": metrics["recall"],
                                "precision": metrics["precision"],
                                "coverage": coverage,
                                "robustness_score": robustness,
                                "feature_layer_counts": feature_layer_summary["counts"],
                                "study_data_mode": feature_layer_summary["study_data_mode"],
                                "behavior_feature_count": len(behavior_cols),
                                "temporal_feature_count": len(temporal_cols),
                                "interaction_feature_count": len(interaction_cols),
                                "selected_as_primary": False,
                                "selected_as_challenger": False,
                                "selected_as_fallback": False,
                            }
                        )
                        threshold_rows.extend(tuning)
                    branch_frames.append(
                        pd.DataFrame(
                            {
                                "request_id": self.request_id,
                                "XH": train_df.loc[valid_idx, "XH"].astype(str).to_numpy(),
                                "TERM_ID": train_df.loc[valid_idx, "TERM_ID"].astype(str).to_numpy(),
                                "LABEL": y_valid.to_numpy(),
                                "feature_group": feature_group,
                                "model_name": model_name,
                                "score": proba,
                            }
                        )
                    )
                except Exception as exc:  # candidate failures should not stop the whole search
                    failures.append(f"{feature_group}/{model_name}: {exc}")

        if not rows:
            self._set_status("failed")
            raise ValueError("no candidate model completed successfully")

        comparison = pd.DataFrame(rows)
        comparison = self._mark_model_selection(comparison)
        self.model_comparison = comparison
        self.threshold_tuning = pd.DataFrame(threshold_rows)
        self.branch_scores = pd.concat(branch_frames, ignore_index=True) if branch_frames else pd.DataFrame()
        self.model_selection = self._build_model_selection(comparison, failures)
        self.threshold_selection = self._build_threshold_selection(comparison)
        if self.request.get("enable_branch_fusion", False):
            self.fusion_config = self._build_fusion_config(comparison)

        selected_feature_group = self.model_selection.get("selected_feature_group")
        selected_features = self._resolve_feature_group(selected_feature_group, base_features, train_df, y) if selected_feature_group else []
        if selected_features:
            self.feature_layer_summary = self._emit_feature_layer_summary(selected_features, FEATURE_LAYER_SUMMARY_PATH, "train_selected")
            self.study_data_mode = str(self.feature_layer_summary.get("study_data_mode", self.study_data_mode))
            self._write_selected_features(selected_features)
            self._write_label_audit_outputs(train_df, label_name, selected_features)

        comparison.to_csv(MODEL_COMPARISON_PATH, index=False, encoding="utf-8-sig")
        self.threshold_tuning.to_csv(THRESHOLD_TUNING_PATH, index=False, encoding="utf-8-sig")
        write_json(MODEL_SELECTION_PATH, self.model_selection)
        shutil.copy2(MODEL_SELECTION_PATH, DELIVERABLE_MODEL_SELECTION_PATH)
        write_json(THRESHOLD_SELECTION_PATH, self.threshold_selection)
        if self.request.get("enable_branch_fusion", False):
            self.branch_scores.to_csv(BRANCH_SCORES_PATH, index=False, encoding="utf-8-sig")
            write_json(FUSION_CONFIG_PATH, self.fusion_config)

        metrics = {
            "candidate_rows": int(len(comparison)),
            "selected_primary_model": self.model_selection.get("selected_primary_model"),
            "selected_challenger_model": self.model_selection.get("selected_challenger_model"),
            "selected_fallback_model": self.model_selection.get("selected_fallback_model"),
            "selected_feature_group": self.model_selection.get("selected_feature_group"),
            "selected_threshold_strategy": self.model_selection.get("selected_threshold_strategy"),
            "candidate_failures": len(failures),
        }
        self.summary_metrics.update(metrics)
        if failures:
            self._set_status("degraded")
            self.warnings.append(f"{len(failures)} candidate model runs failed or were skipped; see selection output.")
        if self.request.get("publish_selected_model", False):
            self.warnings.append("Selected model was not published; formal study_model.pkl/config were preserved.")
        self._decision("model_search", self.status, "Controlled candidate training and selection completed.", metrics)
        self._trace("model_executed", self.status, "Train-mode candidate search completed.", metrics)

    def _resolve_feature_group(self, group: str, base_features: list[str], data: pd.DataFrame, y: pd.Series) -> list[str]:
        synth_features = [c for c in data.columns if c.startswith("FEATURE_SYNTH_")]
        llm_features = [c for c in data.columns if c.startswith("FEATURE_LLM_")]
        candidate_pool = list(dict.fromkeys(base_features + synth_features + llm_features))

        if group == "all_features":
            return candidate_pool

        if group == "grade_dominant":
            keys = ("GRADE", "CET", "COURSE", "FAIL", "RETAKE", "SCORE")
            return [c for c in candidate_pool if any(key in c for key in keys)]

        if group == "attendance_task_dominant":
            keys = ("ATTENDANCE", "CLASS", "ASSIGNMENT", "EXAM", "VIDEO", "TASK")
            return [c for c in candidate_pool if any(key in c for key in keys)]

        if group == "online_activity_dominant":
            keys = ("ONLINE", "ACTIVITY", "LIBRARY", "VIDEO")
            return [c for c in candidate_pool if any(key in c for key in keys)]

        if group == "interaction_risk":
            base_keys = ("GRADE", "COURSE", "ATTENDANCE", "CLASS", "ASSIGNMENT", "EXAM", "LIBRARY", "ONLINE")
            base_selected = [c for c in candidate_pool if any(key in c for key in base_keys)]
            base_selected = self._topk_features(base_selected, data, y, k=16)
            return list(dict.fromkeys(synth_features + llm_features + base_selected))

        if group == "low_missing_robust":
            usable = []
            for feature in candidate_pool:
                if feature not in data.columns:
                    continue
                missing_rate = float(pd.to_numeric(data[feature], errors="coerce").isna().mean())
                if missing_rate <= 0.20:
                    usable.append(feature)

            if not usable:
                usable = candidate_pool

            scored = []
            for feature in usable:
                series = pd.to_numeric(data[feature], errors="coerce")
                if series.notna().sum() < 30 or series.nunique(dropna=True) <= 1:
                    continue

                filled = series.fillna(series.median())
                try:
                    auc = float(roc_auc_score(y, filled))
                    auc = max(auc, 1.0 - auc)
                except Exception:
                    auc = 0.5

                coverage = float(series.notna().mean())
                score = 0.75 * auc + 0.25 * coverage
                scored.append((feature, score))

            scored.sort(key=lambda item: item[1], reverse=True)
            return [feature for feature, _ in scored[:24]]

        if group == "topk_selected":
            return self._topk_features(candidate_pool, data, y, k=18)

        if group == "llm_synthesized":
            return llm_features

        return []

    def _add_llm_synthesized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic mock-LLM domain features from existing study signals."""
        frame = data.copy()

        def num(name: str) -> pd.Series:
            if name not in frame.columns:
                return pd.Series(0.0, index=frame.index)
            return pd.to_numeric(frame[name], errors="coerce")

        def clipped(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
            return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower, upper)

        grade_count = num("FEATURE_GRADE_COURSE_COUNT").fillna(0.0)
        grade_avg = num("FEATURE_GRADE_AVG_SCORE")
        grade_min = num("FEATURE_GRADE_MIN_SCORE")
        fail_count = num("FEATURE_GRADE_FAIL_COUNT").fillna(0.0)
        credit_sum = num("FEATURE_GRADE_CREDIT_SUM").fillna(0.0)
        selected_count = num("FEATURE_COURSE_SELECTED_COUNT").fillna(0.0)
        course_credit = num("FEATURE_COURSE_CREDIT_SUM").fillna(0.0)
        retake_count = num("FEATURE_COURSE_RETAKE_COUNT").fillna(0.0)
        attendance_events = num("FEATURE_ATTENDANCE_EVENT_COUNT").fillna(0.0)
        attendance_abnormal = num("FEATURE_ATTENDANCE_ABNORMAL_COUNT").fillna(0.0)
        attendance_rate = num("FEATURE_ATTENDANCE_ABNORMAL_RATE")
        assignment_count = num("FEATURE_ASSIGNMENT_COUNT")
        assignment_missing = num("FEATURE_ASSIGNMENT_MISSING_COUNT").fillna(0.0)
        assignment_submit = num("FEATURE_ASSIGNMENT_SUBMIT_RATE")
        exam_count = num("FEATURE_EXAM_COUNT")
        exam_missing = num("FEATURE_EXAM_MISSING_COUNT").fillna(0.0)
        exam_score = num("FEATURE_EXAM_SCORE_AVG")
        class_task = num("FEATURE_CLASS_TASK_COUNT")
        class_rate = num("FEATURE_CLASS_TASK_RATE_AVG")
        video_rate = num("FEATURE_CLASS_VIDEO_RATE_AVG")
        library_visits = num("FEATURE_LIBRARY_VISIT_COUNT").fillna(0.0)

        frame["FEATURE_LLM_GRADE_RISK_INDEX"] = clipped(
            0.35 * ((100.0 - grade_avg) / 100.0)
            + 0.25 * ((60.0 - grade_min).clip(lower=0.0) / 60.0)
            + 0.25 * (fail_count / (grade_count + 1.0))
            + 0.15 * (retake_count / (selected_count + 1.0))
        )
        frame["FEATURE_LLM_COURSE_PRESSURE_INDEX"] = clipped(
            0.45 * (selected_count / 10.0)
            + 0.35 * (course_credit / 30.0)
            + 0.20 * (credit_sum / 25.0)
        )
        frame["FEATURE_LLM_ATTENDANCE_RISK_INDEX"] = clipped(
            0.55 * attendance_rate.fillna(0.0)
            + 0.45 * (attendance_abnormal / (attendance_events + 1.0))
        )
        frame["FEATURE_LLM_ASSIGNMENT_EXAM_GAP_INDEX"] = clipped(
            0.30 * (assignment_missing / (assignment_count.fillna(0.0) + 1.0))
            + 0.25 * (1.0 - assignment_submit.fillna(1.0))
            + 0.25 * (exam_missing / (exam_count.fillna(0.0) + 1.0))
            + 0.20 * ((60.0 - exam_score).clip(lower=0.0) / 60.0).fillna(0.0)
        )
        frame["FEATURE_LLM_CLASS_ENGAGEMENT_GAP_INDEX"] = clipped(
            0.45 * (1.0 - class_rate.fillna(1.0))
            + 0.35 * (1.0 - video_rate.fillna(1.0))
            + 0.20 * (class_task.isna().astype(float))
        )
        frame["FEATURE_LLM_OFFLINE_ACTIVITY_WEAKNESS_INDEX"] = clipped(1.0 - np.log1p(library_visits) / np.log(21.0))
        frame["FEATURE_LLM_SCORE_VOLATILITY_INDEX"] = clipped((grade_avg - grade_min).abs() / 50.0)

        l456_cols = [
            "FEATURE_CLASS_TASK_COUNT",
            "FEATURE_CLASS_TASK_RATE_AVG",
            "FEATURE_CLASS_VIDEO_RATE_AVG",
            "FEATURE_ASSIGNMENT_COUNT",
            "FEATURE_ASSIGNMENT_SCORE_AVG",
            "FEATURE_ASSIGNMENT_MISSING_COUNT",
            "FEATURE_ASSIGNMENT_SUBMIT_RATE",
            "FEATURE_EXAM_COUNT",
            "FEATURE_EXAM_SCORE_AVG",
            "FEATURE_EXAM_MISSING_COUNT",
        ]
        present_l456 = frame[[c for c in l456_cols if c in frame.columns]].notna().any(axis=1)
        frame["FEATURE_LLM_L456_SIGNAL_ABSENCE"] = (~present_l456).astype(float)
        frame["FEATURE_LLM_COMPOSITE_STUDY_RISK"] = clipped(
            0.35 * frame["FEATURE_LLM_GRADE_RISK_INDEX"]
            + 0.20 * frame["FEATURE_LLM_ASSIGNMENT_EXAM_GAP_INDEX"]
            + 0.15 * frame["FEATURE_LLM_ATTENDANCE_RISK_INDEX"]
            + 0.15 * frame["FEATURE_LLM_CLASS_ENGAGEMENT_GAP_INDEX"]
            + 0.10 * frame["FEATURE_LLM_COURSE_PRESSURE_INDEX"]
            + 0.05 * frame["FEATURE_LLM_SCORE_VOLATILITY_INDEX"]
        )
        self._decision(
            "llm_feature_synthesis",
            "success",
            "Mock LLM created deterministic domain features for candidate model creation.",
            {"llm_feature_count": int(len([c for c in frame.columns if c.startswith("FEATURE_LLM_")]))},
        )
        return frame

    def _add_study_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()

        def num(name: str) -> pd.Series:
            if name not in frame.columns:
                return pd.Series(np.nan, index=frame.index, dtype="float64")
            return pd.to_numeric(frame[name], errors="coerce")

        def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            return a / b.replace(0, np.nan)

        def clipped(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
            return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower, upper)

        grade_count = num("FEATURE_GRADE_COURSE_COUNT").fillna(0.0)
        grade_avg = num("FEATURE_GRADE_AVG_SCORE")
        grade_min = num("FEATURE_GRADE_MIN_SCORE")
        fail_count = num("FEATURE_GRADE_FAIL_COUNT").fillna(0.0)
        credit_sum = num("FEATURE_GRADE_CREDIT_SUM").fillna(0.0)

        selected_count = num("FEATURE_COURSE_SELECTED_COUNT").fillna(0.0)
        course_credit = num("FEATURE_COURSE_CREDIT_SUM").fillna(0.0)
        retake_count = num("FEATURE_COURSE_RETAKE_COUNT").fillna(0.0)

        attendance_events = num("FEATURE_ATTENDANCE_EVENT_COUNT").fillna(0.0)
        attendance_abnormal = num("FEATURE_ATTENDANCE_ABNORMAL_COUNT").fillna(0.0)
        attendance_rate = num("FEATURE_ATTENDANCE_ABNORMAL_RATE").fillna(0.0)

        assignment_count = num("FEATURE_ASSIGNMENT_COUNT").fillna(0.0)
        assignment_missing = num("FEATURE_ASSIGNMENT_MISSING_COUNT").fillna(0.0)
        assignment_submit = num("FEATURE_ASSIGNMENT_SUBMIT_RATE").fillna(1.0)
        assignment_score = num("FEATURE_ASSIGNMENT_SCORE_AVG")

        exam_count = num("FEATURE_EXAM_COUNT").fillna(0.0)
        exam_missing = num("FEATURE_EXAM_MISSING_COUNT").fillna(0.0)
        exam_score = num("FEATURE_EXAM_SCORE_AVG")

        class_task_count = num("FEATURE_CLASS_TASK_COUNT").fillna(0.0)
        class_task_rate = num("FEATURE_CLASS_TASK_RATE_AVG").fillna(1.0)
        class_video_rate = num("FEATURE_CLASS_VIDEO_RATE_AVG").fillna(1.0)

        library_visits = num("FEATURE_LIBRARY_VISIT_COUNT").fillna(0.0)
        online_activity = num("FEATURE_ONLINE_ACTIVE_DAYS").fillna(0.0)

        frame["FEATURE_SYNTH_FAIL_RATIO"] = clipped(safe_div(fail_count, grade_count + 1.0))
        frame["FEATURE_SYNTH_RETAKE_RATIO"] = clipped(safe_div(retake_count, selected_count + 1.0))
        frame["FEATURE_SYNTH_SCORE_SPREAD"] = clipped((grade_avg - grade_min).abs() / 50.0)
        frame["FEATURE_SYNTH_LOW_SCORE_PRESSURE"] = clipped((60.0 - grade_min).clip(lower=0.0) / 60.0)

        frame["FEATURE_SYNTH_COURSE_LOAD_PER_CREDIT"] = clipped(safe_div(selected_count, course_credit + 1.0), 0.0, 3.0)
        frame["FEATURE_SYNTH_CREDIT_PRESSURE"] = clipped(course_credit / 30.0)
        frame["FEATURE_SYNTH_ATTENDANCE_RISK"] = clipped(
            0.55 * attendance_rate + 0.45 * safe_div(attendance_abnormal, attendance_events + 1.0).fillna(0.0)
        )

        frame["FEATURE_SYNTH_ASSIGNMENT_RISK"] = clipped(
            0.60 * safe_div(assignment_missing, assignment_count + 1.0).fillna(0.0)
            + 0.40 * (1.0 - assignment_submit)
        )

        frame["FEATURE_SYNTH_EXAM_RISK"] = clipped(
            0.55 * safe_div(exam_missing, exam_count + 1.0).fillna(0.0)
            + 0.45 * ((60.0 - exam_score).clip(lower=0.0) / 60.0).fillna(0.0)
        )

        frame["FEATURE_SYNTH_ASSIGNMENT_EXAM_GAP"] = clipped(
            (assignment_score - exam_score).abs() / 40.0
        )

        frame["FEATURE_SYNTH_ATTENDANCE_ASSIGNMENT_GAP"] = clipped(
            (attendance_rate - (1.0 - assignment_submit)).abs()
        )

        frame["FEATURE_SYNTH_CLASS_VIDEO_GAP"] = clipped((class_task_rate - class_video_rate).abs())
        frame["FEATURE_SYNTH_LIBRARY_PER_COURSE"] = clipped(safe_div(np.log1p(library_visits), np.log1p(selected_count + 1.0)), 0.0, 2.0)
        frame["FEATURE_SYNTH_ONLINE_PER_COURSE"] = clipped(safe_div(online_activity, selected_count + 1.0), 0.0, 5.0)

        frame["FEATURE_SYNTH_OFFLINE_ENGAGEMENT_WEAKNESS"] = clipped(
            1.0 - np.log1p(library_visits) / np.log(21.0)
        )

        frame["FEATURE_SYNTH_CLASS_SIGNAL_MISSING"] = (
            class_task_count.isna().astype(float)
            if "FEATURE_CLASS_TASK_COUNT" in frame.columns
            else pd.Series(1.0, index=frame.index)
        )

        frame["FEATURE_SYNTH_COMPOSITE_RISK"] = clipped(
            0.22 * frame["FEATURE_SYNTH_FAIL_RATIO"]
            + 0.12 * frame["FEATURE_SYNTH_RETAKE_RATIO"]
            + 0.15 * frame["FEATURE_SYNTH_LOW_SCORE_PRESSURE"]
            + 0.13 * frame["FEATURE_SYNTH_ATTENDANCE_RISK"]
            + 0.14 * frame["FEATURE_SYNTH_ASSIGNMENT_RISK"]
            + 0.14 * frame["FEATURE_SYNTH_EXAM_RISK"]
            + 0.10 * frame["FEATURE_SYNTH_SCORE_SPREAD"]
        )

        self._decision(
            "study_advanced_feature_synthesis",
            "success",
            "Advanced interaction features were created for study-domain candidate search.",
            {"advanced_feature_count": int(len([c for c in frame.columns if c.startswith("FEATURE_SYNTH_")]))},
        )
        return frame

    @staticmethod
    def _term_sort_key(series: pd.Series) -> pd.Series:
        term = series.astype("string").fillna("")
        extracted = term.str.extract(r"(?P<start>\d{4})-(?P<end>\d{4})-(?P<part>\d+)")
        start = pd.to_numeric(extracted["start"], errors="coerce").fillna(-1).astype(int)
        end = pd.to_numeric(extracted["end"], errors="coerce").fillna(-1).astype(int)
        part = pd.to_numeric(extracted["part"], errors="coerce").fillna(-1).astype(int)
        return start * 10000 + end * 10 + part

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
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
        ordered["_TERM_SORT_KEY"] = self._term_sort_key(ordered["TERM_ID"])
        ordered["_ROW_ID"] = np.arange(len(ordered))
        ordered = ordered.sort_values(["XH", "_TERM_SORT_KEY", "TERM_ID", "_ROW_ID"]).reset_index(drop=True)

        for column in candidate_features:
            numeric = pd.to_numeric(ordered[column], errors="coerce")
            prev = numeric.groupby(ordered["XH"]).shift(1)
            hist = numeric.groupby(ordered["XH"]).transform(lambda s: s.shift(1).expanding().mean())
            base = column.lower().replace("feature_", "")
            ordered[f"prev_{base}"] = prev
            ordered[f"hist_{base}"] = hist
            ordered[f"delta_{base}"] = numeric - prev
            ordered[f"ratio_{base}"] = numeric / (prev.abs() + 1e-6)
            ordered[f"delta_hist_{base}"] = numeric - hist
            ordered[f"ratio_hist_{base}"] = numeric / (hist.abs() + 1e-6)

            # New high-coverage temporal features
            declined = (numeric < prev).astype(float)
            consecutive = declined.groupby(ordered["XH"]).cumsum()
            prev_consecutive = consecutive.groupby(ordered["XH"]).shift(1).fillna(0)
            ordered[f"consecutive_decline_{base}"] = prev_consecutive.astype(int)

            cummin_shifted = numeric.groupby(ordered["XH"]).cummin().groupby(ordered["XH"]).shift(1)
            cummax_shifted = numeric.groupby(ordered["XH"]).cummax().groupby(ordered["XH"]).shift(1)
            range_val = cummax_shifted - cummin_shifted
            ordered[f"dist_from_worst_{base}"] = (
                (numeric - cummin_shifted) / (range_val + 1e-6)
            ).fillna(0.5)

            ordered[f"recovery_{base}"] = (
                (numeric > prev) & (prev < hist)
            ).fillna(False).astype(int)

        ordered = ordered.sort_values("_ROW_ID").drop(columns=["_TERM_SORT_KEY", "_ROW_ID"])
        return ordered

    @staticmethod
    def _add_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
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
                frame[new_col] = pd.to_numeric(frame[col_a], errors="coerce").fillna(0.0) * pd.to_numeric(frame[col_b], errors="coerce").fillna(0.0)

        # New semantically meaningful discordance features
        def num_safe(name: str) -> pd.Series | None:
            return pd.to_numeric(frame[name], errors="coerce") if name in frame.columns else None

        # Discordance: high attendance but low grades (effort not translating to results)
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

    def _add_subgroup_targeted_features(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()

        def num(name: str) -> pd.Series:
            if name not in frame.columns:
                return pd.Series(np.nan, index=frame.index, dtype="float64")
            return pd.to_numeric(frame[name], errors="coerce")

        def z_by_student(series: pd.Series) -> pd.Series:
            grouped_mean = series.groupby(frame["XH"]).transform("mean")
            grouped_std = series.groupby(frame["XH"]).transform("std").replace(0, np.nan)
            return (series - grouped_mean) / (grouped_std + 1e-6)

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
                frame[f"trend_decline_count_{alias}"] = current.lt(prev).fillna(False).astype(int) + prev.lt(hist).fillna(False).astype(int)
                frame[f"trend_below_hist_{alias}"] = current.lt(hist).fillna(False).astype(int)
                frame[f"trend_drawdown_{alias}"] = (peak - current) / (peak.abs() + 1e-6)
                frame[f"personal_gap_{alias}"] = current - hist
                frame[f"personal_ratio_{alias}"] = current / (hist.abs() + 1e-6)
                frame[f"personal_z_{alias}"] = z_by_student(current)

        behavior_cols = [c for c in frame.columns if c.startswith("FEATURE_") and any(k in c for k in ["ATTENDANCE", "CLASS_", "ASSIGNMENT", "EXAM", "LIBRARY"])]
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

        if "FEATURE_GRADE_AVG_SCORE" in frame.columns and "FEATURE_LIBRARY_VISIT_COUNT" in frame.columns:
            frame["imbalance_grade_vs_library_z"] = z_by_student(num("FEATURE_GRADE_AVG_SCORE")) - z_by_student(num("FEATURE_LIBRARY_VISIT_COUNT"))
        if "FEATURE_ASSIGNMENT_SCORE_AVG" in frame.columns and "FEATURE_EXAM_SCORE_AVG" in frame.columns:
            frame["imbalance_assignment_vs_exam_level"] = num("FEATURE_ASSIGNMENT_SCORE_AVG") - num("FEATURE_EXAM_SCORE_AVG")
        if "delta_grade_avg_score" in frame.columns and "delta_library_visit_count" in frame.columns:
            frame["imbalance_grade_vs_library_delta"] = num("delta_grade_avg_score") - num("delta_library_visit_count")
        if "FEATURE_ATTENDANCE_EVENT_COUNT" in frame.columns and "delta_grade_avg_score" in frame.columns:
            frame["imbalance_attendance_stable_study_drop"] = (
                (num("FEATURE_ATTENDANCE_EVENT_COUNT").fillna(0) > 0).astype(int) * (num("delta_grade_avg_score").fillna(0) < 0).astype(int)
            )
        return frame

    def _augment_feature_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Apply unified feature engineering pipeline (same as training path)."""
        return apply_feature_engineering(table, include_course_risk=True)

    def _topk_features(self, features: list[str], data: pd.DataFrame, y: pd.Series, k: int = 12) -> list[str]:
        scores = []
        for feature in features:
            if feature not in data.columns:
                continue

            series = pd.to_numeric(data[feature], errors="coerce")
            if series.notna().sum() < 30 or series.nunique(dropna=True) <= 1:
                continue

            filled = series.fillna(series.median())

            try:
                auc = float(roc_auc_score(y, filled))
                auc = max(auc, 1.0 - auc)
            except Exception:
                auc = 0.5

            try:
                corr = abs(float(filled.corr(y)))
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0

            coverage = float(series.notna().mean())
            score = 0.60 * auc + 0.25 * corr + 0.15 * coverage
            scores.append((feature, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [feature for feature, _ in scores[: min(k, len(scores))]]

    def _make_candidate_model(self, model_name: str) -> Any:
        if model_name == "LogisticRegression":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=800, class_weight="balanced", C=0.7, random_state=42)),
                ]
            )

        if model_name == "RandomForest":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=10,
                            min_samples_split=10,
                            min_samples_leaf=4,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

        if model_name == "ExtraTrees":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=10,
                            min_samples_split=8,
                            min_samples_leaf=3,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

        if model_name == "LightGBM":
            if LGBMClassifier is None:
                return None
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=300,
                            learning_rate=0.03,
                            num_leaves=31,
                            min_child_samples=25,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=0.2,
                            reg_lambda=0.4,
                            class_weight="balanced",
                            random_state=42,
                            verbose=-1,
                        ),
                    ),
                ]
            )

        if model_name == "XGBoost":
            if XGBClassifier is None:
                return None
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.03,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            min_child_weight=3,
                            reg_alpha=0.2,
                            reg_lambda=1.0,
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=2,
                        ),
                    ),
                ]
            )

        return None

    @staticmethod
    def _predict_proba(model: Any, x_valid: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_valid)
            return np.asarray(proba)[:, 1]
        decision = model.decision_function(x_valid)
        return 1.0 / (1.0 + np.exp(-decision))

    @staticmethod
    def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y_true, proba))
        except ValueError:
            return float("nan")

    def _select_threshold(
        self,
        y_true: pd.Series,
        proba: np.ndarray,
        strategy: str,
        feature_group: str,
        model_name: str,
    ) -> tuple[float, list[dict[str, Any]]]:
        grid = [0.5] if strategy == "default_0_5" else [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
        rows = []
        for threshold in grid:
            metrics = self._classification_metrics(y_true, proba, threshold)
            rows.append(
                {
                    "request_id": self.request_id,
                    "feature_group": feature_group,
                    "model_name": model_name,
                    "threshold_strategy": strategy,
                    "threshold": threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "positive_rate": metrics["positive_rate"],
                    "selected_threshold": False,
                    "selection_reason": "",
                }
            )
        tuning = pd.DataFrame(rows)
        if strategy == "default_0_5":
            selected_idx = tuning.index[0]
            reason = "fixed threshold 0.5"
        elif strategy == "best_f1":
            selected_idx = tuning.sort_values(["f1", "recall", "precision"], ascending=False).index[0]
            reason = "maximized validation F1"
        else:
            eligible = tuning[tuning["recall"] >= 0.7]
            if eligible.empty:
                selected_idx = tuning.sort_values(["recall", "f1"], ascending=False).index[0]
                reason = "maximized recall because no threshold reached recall>=0.70"
            else:
                selected_idx = eligible.sort_values(["f1", "precision"], ascending=False).index[0]
                reason = "best F1 among thresholds with recall>=0.70"
        tuning.loc[selected_idx, "selected_threshold"] = True
        tuning.loc[selected_idx, "selection_reason"] = reason
        selected_threshold = float(tuning.loc[selected_idx, "threshold"])
        return selected_threshold, tuning.to_dict(orient="records")

    @staticmethod
    def _classification_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
        pred = (proba >= threshold).astype(int)
        return {
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "positive_rate": float(pred.mean()),
        }

    @staticmethod
    def _robustness_score(auc: float, f1: float, recall: float, coverage: float) -> float:
        auc_value = 0.0 if np.isnan(auc) else auc
        return float(0.58 * auc_value + 0.22 * f1 + 0.12 * recall + 0.08 * coverage)

    def _mark_model_selection(self, comparison: pd.DataFrame) -> pd.DataFrame:
        comparison = comparison.copy().reset_index(drop=True)

        best_auc = float(comparison["auc"].max())
        auc_floor = max(best_auc - 0.01, 0.80)

        eligible = comparison[comparison["auc"] >= auc_floor].copy()
        if eligible.empty:
            eligible = comparison.copy()

        eligible = eligible.sort_values(["robustness_score", "f1", "recall"], ascending=False)
        primary_idx = eligible.index[0]
        comparison.loc[:, "selected_as_primary"] = False
        comparison.loc[:, "selected_as_challenger"] = False
        comparison.loc[:, "selected_as_fallback"] = False
        comparison.loc[primary_idx, "selected_as_primary"] = True

        remaining = comparison.drop(index=primary_idx).sort_values(["robustness_score", "auc", "f1"], ascending=False)
        if not remaining.empty:
            comparison.loc[remaining.index[0], "selected_as_challenger"] = True

        fallback_candidates = comparison[comparison["model_name"] == "LogisticRegression"]
        fallback_idx = fallback_candidates.index[0] if not fallback_candidates.empty else comparison.sort_values(
            ["coverage", "auc"], ascending=False
        ).index[0]
        comparison.loc[fallback_idx, "selected_as_fallback"] = True

        comparison = comparison.sort_values(
            ["selected_as_primary", "selected_as_challenger", "robustness_score", "auc", "f1"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

        return comparison

    def _build_model_selection(self, comparison: pd.DataFrame, failures: list[str]) -> dict[str, Any]:
        primary = comparison[comparison["selected_as_primary"]].iloc[0].to_dict()
        challenger = comparison[comparison["selected_as_challenger"]].iloc[0].to_dict() if comparison["selected_as_challenger"].any() else {}
        fallback = comparison[comparison["selected_as_fallback"]].iloc[0].to_dict()
        return {
            "request_id": self.request_id,
            "selected_primary_model": primary.get("model_name"),
            "selected_challenger_model": challenger.get("model_name"),
            "selected_fallback_model": fallback.get("model_name"),
            "selected_feature_group": primary.get("feature_group"),
            "selected_threshold_strategy": primary.get("threshold_strategy"),
            "selected_threshold": primary.get("threshold"),
            "selection_reason": "Selected by highest robustness_score; fallback prefers LogisticRegression when available. Formal model artifacts are not overwritten.",
            "comparison_summary": {
                "candidate_rows": int(len(comparison)),
                "best_auc": float(comparison["auc"].max()),
                "best_f1": float(comparison["f1"].max()),
                "best_recall": float(comparison["recall"].max()),
                "candidate_failures": failures,
            },
        }

    def _build_threshold_selection(self, comparison: pd.DataFrame) -> dict[str, Any]:
        primary = comparison[comparison["selected_as_primary"]].iloc[0]
        return {
            "request_id": self.request_id,
            "feature_group": primary["feature_group"],
            "model_name": primary["model_name"],
            "threshold_strategy": primary["threshold_strategy"],
            "selected_threshold": float(primary["threshold"]),
            "selection_reason": "Threshold inherited from the selected primary candidate row.",
        }

    def _build_fusion_config(self, comparison: pd.DataFrame) -> dict[str, Any]:
        top = comparison.drop_duplicates(["feature_group", "model_name"]).head(3).copy()
        total = float(top["robustness_score"].sum()) or 1.0
        branches = []
        for _, row in top.iterrows():
            branches.append(
                {
                    "feature_group": row["feature_group"],
                    "model_name": row["model_name"],
                    "weight": float(row["robustness_score"] / total),
                }
            )
        fusion_metrics = self._evaluate_branch_fusion(branches)
        return {
            "request_id": self.request_id,
            "fusion_strategy": "weighted_branch_fusion",
            "status": "candidate_only",
            "branches": branches,
            "validation_metrics": fusion_metrics,
            "note": "First-version domain branch fusion config; formal study model artifacts are not overwritten.",
        }

    def _evaluate_branch_fusion(self, branches: list[dict[str, Any]]) -> dict[str, Any]:
        if self.branch_scores.empty or not branches:
            return {}
        frames = []
        for branch in branches:
            mask = (
                (self.branch_scores["feature_group"].astype(str) == str(branch["feature_group"]))
                & (self.branch_scores["model_name"].astype(str) == str(branch["model_name"]))
            )
            frame = self.branch_scores.loc[mask, ["XH", "TERM_ID", "LABEL", "score"]].copy()
            frame["branch_key"] = f"{branch['feature_group']}__{branch['model_name']}"
            frames.append(frame)
        if not frames:
            return {}
        long_scores = pd.concat(frames, ignore_index=True)
        pivot = long_scores.pivot_table(index=["XH", "TERM_ID", "LABEL"], columns="branch_key", values="score", aggfunc="mean")
        pivot = pivot.reset_index()
        weights = {
            f"{branch['feature_group']}__{branch['model_name']}": float(branch["weight"])
            for branch in branches
            if f"{branch['feature_group']}__{branch['model_name']}" in pivot.columns
        }
        if not weights:
            return {}
        total = sum(weights.values()) or 1.0
        score = np.zeros(len(pivot))
        for key, weight in weights.items():
            score += pd.to_numeric(pivot[key], errors="coerce").fillna(0).to_numpy() * (weight / total)
        y_true = pivot["LABEL"].astype(int)
        threshold, tuning = self._select_threshold(y_true, score, "best_f1", "weighted_branch_fusion", "fusion")
        metrics = self._classification_metrics(y_true, score, threshold)
        metrics.update(
            {
                "auc": self._safe_auc(y_true, score),
                "threshold": threshold,
                "rows": int(len(pivot)),
                "branch_count": len(weights),
            }
        )
        selected_rows = [row for row in tuning if row.get("selected_threshold")]
        if selected_rows:
            metrics["selection_reason"] = selected_rows[0].get("selection_reason")
        return metrics

    def export_result(self) -> dict[str, Any]:
        harness_result = self.extra_result.get("harness_v1", {})
        if harness_result:
            metrics = harness_result.get("candidate_metrics", {}) or {}
        else:
            # Fix #2: Use candidate metrics when available, not just baseline model_config
            metrics_block = self.model_config.get("metrics", {})
            metrics = metrics_block.get("core_model", {}).get("valid", {}) if isinstance(metrics_block, dict) else {}

            # Override with evolution candidate metrics if available
            candidate_metrics = self.extra_result.get("release_action", {}).get("baseline_comparison", {}).get("candidate_metrics", {})
            if candidate_metrics:
                metrics = candidate_metrics
            elif self.extra_result.get("evolution_metrics"):
                publish_candidate = self.extra_result.get("evolution_metrics", {}).get("publish_candidate", {})
                if publish_candidate.get("metrics"):
                    metrics = publish_candidate["metrics"]

        self.summary_metrics.update(
            {
                "auc": self.summary_metrics.get("auc", metrics.get("auc")),
                "f1": self.summary_metrics.get("f1", metrics.get("f1")),
                "recall": self.summary_metrics.get("recall", metrics.get("recall")),
            }
        )

        # Harness output should use harness audit context as the primary source of truth.
        if harness_result:
            metric_context = harness_result.get("metric_context", {})
            self.study_data_mode = harness_result.get("study_data_mode") or metric_context.get("study_data_mode") or self.study_data_mode
            self.row_level_study_data_mode = harness_result.get("row_level_study_data_mode") or metric_context.get("row_level_study_data_mode") or self.row_level_study_data_mode
            self.summary_metrics.update(
                {
                    "eval_scope": metric_context.get("eval_scope"),
                    "task_scope": metric_context.get("task_scope"),
                    "study_data_mode": metric_context.get("study_data_mode"),
                    "row_level_study_data_mode": metric_context.get("row_level_study_data_mode"),
                }
            )
        # Fix #1: study_data_mode should reflect row-level reality, not just feature layer summary
        # Priority order:
        # 1. row_modes (actual inference data) - highest priority
        # 2. candidate chain_validation.data_mode_validation (evolution evaluation)
        # 3. feature_layer_summary (fallback - least accurate)
        elif self.row_modes.empty:
            # Publish mode: row_modes is empty, use candidate's data_mode_validation if available
            candidate_validation = (
                self.extra_result
                .get("release_action", {})
                .get("candidate", {})
                .get("chain_validation", {})
                .get("data_mode_validation", {})
            )
            if not candidate_validation:
                # Try evolution_metrics path
                candidate_validation = (
                    self.extra_result
                    .get("evolution_metrics", {})
                    .get("publish_candidate", {})
                    .get("chain_validation", {})
                    .get("data_mode_validation", {})
                )
            
            if candidate_validation:
                # Use candidate's actual evaluation ratios
                core_plus_ratio = candidate_validation.get("core_plus_behavior_ratio", 0)
                if core_plus_ratio > 0.05:  # >5% threshold
                    self.study_data_mode = "core_plus_behavior_enhanced"
                    self.row_level_study_data_mode = "core_plus_behavior"
        else:
            row_mode_counts = self.row_modes["STUDY_DATA_MODE"].value_counts(dropna=False)
            if row_mode_counts.get("core_plus_behavior", 0) > 0:
                # Enhanced samples exist - upgrade study_data_mode
                if self.study_data_mode == "core_only":
                    self.study_data_mode = "core_plus_behavior"

        if self.status == "degraded" and not self.warnings:
            self.warnings.append("Request completed with degraded status.")
        family_rows = [row for row in self.quality_report.get("primary_key_association", []) if row.get("family")]

        # Fix #4: available_layers should include behavior if row_modes OR candidate validation show it exists
        active_summary = self.infer_feature_layer_summary or self.feature_layer_summary
        available_layers = self._available_feature_layers(active_summary)
        if harness_result:
            mode_hints = {
                harness_result.get("study_data_mode", ""),
                harness_result.get("row_level_study_data_mode", ""),
                harness_result.get("metric_context", {}).get("study_data_mode", ""),
                harness_result.get("metric_context", {}).get("row_level_study_data_mode", ""),
            }
            if any(
                ("behavior" in hint) or hint == "mixed_overall_with_subgroups"
                for hint in mode_hints
                if hint
            ):
                if "behavior" not in available_layers:
                    available_layers.append("behavior")

        # Check row_modes first (train/infer mode)
        has_behavior_from_row_modes = not self.row_modes.empty and (self.row_modes["STUDY_DATA_MODE"] == "core_plus_behavior").any()
        
        # Check candidate validation (publish mode)
        has_behavior_from_candidate = False
        if self.row_modes.empty:
            candidate_validation = (
                self.extra_result
                .get("release_action", {})
                .get("candidate", {})
                .get("chain_validation", {})
                .get("data_mode_validation", {})
            )
            if not candidate_validation:
                candidate_validation = (
                    self.extra_result
                    .get("evolution_metrics", {})
                    .get("publish_candidate", {})
                    .get("chain_validation", {})
                    .get("data_mode_validation", {})
                )
            if candidate_validation:
                core_plus_ratio = candidate_validation.get("core_plus_behavior_ratio", 0)
                has_behavior_from_candidate = core_plus_ratio > 0.05

        if "behavior" not in available_layers and (has_behavior_from_row_modes or has_behavior_from_candidate):
            available_layers.append("behavior")

        recoverable_layers = [row["family"] for row in family_rows if row.get("layer_status") == "recoverable_behavior"]
        unavailable_layers = [row["family"] for row in family_rows if row.get("layer_status") == "unavailable_behavior"]

        result = {
            "request_id": self.request_id,
            "domain": self.request.get("domain"),
            "mode": self.request.get("run_mode"),
            "status": self.status,
            "fallback_used": self.fallback_used,
            "model_version": self.model_config.get("model_version"),
            "feature_version": self.model_config.get("feature_version"),
            "study_data_mode": self.study_data_mode,
            "row_level_study_data_mode": self.row_level_study_data_mode,
            "study_quality_flag": self.study_quality_flag,
            "prediction_output_path": str(self.paths.get("prediction_output", "")),
            "explanation_output_path": str(self.paths.get("explanation_output", "")),
            "quality_report_path": self._quality_report_output_path(),
            "validation_report_path": str(self.paths.get("validation_report", "")),
            "eval_report_path": str(self.model_config.get("eval_report_path", "")),
            "subgroup_metrics_path": str(self.model_config.get("subgroup_metrics_path", "")),
            "confidence_zone_report_path": str(self.model_config.get("confidence_zone_report_path", "")),
            "feature_layer_summary_path": str(FEATURE_LAYER_SUMMARY_PATH if self.request.get("run_mode") == "train" else INFER_FEATURE_LAYER_SUMMARY_PATH),
            "selected_features_path": str(SELECTED_FEATURES_PATH),
            "label_audit_detail_path": str(LABEL_AUDIT_DETAIL_PATH) if LABEL_AUDIT_DETAIL_PATH.exists() else None,
            "positive_label_profile_path": str(POSITIVE_LABEL_PROFILE_PATH) if POSITIVE_LABEL_PROFILE_PATH.exists() else None,
            "available_layers": available_layers,
            "recoverable_behavior_layers": recoverable_layers,
            "unavailable_behavior_layers": unavailable_layers,
            "summary_metrics": self.summary_metrics,
            "warnings": self.warnings,
        }
        if self.request.get("run_mode") == "train":
            result.update(
                {
                    "selected_primary_model": self.model_selection.get("selected_primary_model"),
                    "selected_challenger_model": self.model_selection.get("selected_challenger_model"),
                    "selected_fallback_model": self.model_selection.get("selected_fallback_model"),
                    "model_comparison_path": str(MODEL_COMPARISON_PATH),
                    "threshold_tuning_path": str(THRESHOLD_TUNING_PATH),
                    "model_selection_path": str(MODEL_SELECTION_PATH),
                    "threshold_selection_path": str(THRESHOLD_SELECTION_PATH),
                    "branch_scores_path": str(BRANCH_SCORES_PATH) if self.request.get("enable_branch_fusion", False) else None,
                    "fusion_config_path": str(FUSION_CONFIG_PATH) if self.request.get("enable_branch_fusion", False) else None,
                }
            )
        if self.request.get("llm_enable", False):
            result.update(
                {
                    "llm_provider": self.request.get("llm_provider", "mock"),
                    "llm_model": self.request.get("llm_model", "qwen-plus"),
                    "llm_task_type": self.request.get("llm_task_type", "model_review"),
                    "llm_review_path": str(DM_DIR / "study_llm_review.json"),
                    "llm_review_status": self.llm_review.get("response_status"),
                }
            )
        if self.extra_result:
            result.update(self.extra_result)

        result = normalize_workspace_paths(result)
        self._decision("final_output", self.status, "Standard result object exported for harness consumption.", {"result_path": str(RESULT_PATH)})
        write_json(RESULT_PATH, result)
        shutil.copy2(RESULT_PATH, DELIVERABLE_RESULT_PATH)
        decision_log = {
            "request_id": self.request_id,
            "domain": self.request.get("domain"),
            "mode": self.request.get("run_mode"),
            "status": self.status,
            "fallback_used": self.fallback_used,
            "warnings": self.warnings,
            "state_history": self.state_history,
            "decisions": self.decisions,
            "policy_snapshot": self.policy if self.policy.get("decision_log", {}).get("include_policy_snapshot", True) else {},
            "search_space_snapshot": self.search_space,
            "result_path": str(RESULT_PATH),
            "deliverable_result_path": str(DELIVERABLE_RESULT_PATH),
        }
        write_json(DECISION_PATH, decision_log)
        self._trace("result_exported", self.status, "StudyAgent result exported.", {"result_path": str(RESULT_PATH)})
        return result

    def run_harness_pipeline(self) -> None:
        workspace_root = ROOT.parent
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        if str(ROOT) not in sys.path:
            sys.path.insert(1, str(ROOT))
        from study_domain.study_pipeline import build_study_pipeline

        self.model_config = normalize_workspace_paths(
            self._read_json(ROOT / "data" / "deliverables" / "study" / "model" / "study_model_config.json")
        )
        self.paths = {name: self._resolve_path(Path(value)) for name, value in self.request.get("input_paths", {}).items()}
        runner = build_study_pipeline(ROOT, self.request)
        record, record_path = runner.run(self.request, ROOT)
        self.status = self._map_harness_status(record)
        self.warnings.extend([warning for warning in record.collected_warnings if warning not in self.warnings])

        eval_result = self._latest_stage_result(record.stage_results, "eval")
        eval_metrics = getattr(eval_result, "metrics", {}) if eval_result else {}
        eval_diagnostics = getattr(eval_result, "diagnostics", {}) if eval_result else {}
        baseline_comparison = eval_diagnostics.get("baseline_comparison", {})
        metric_context = {
            "eval_scope": record.eval_scope,
            "task_scope": record.task_scope,
            "study_data_mode": record.metric_context.get("study_data_mode") or record.domain_context.get("study_data_mode", ""),
            "row_level_study_data_mode": record.metric_context.get("row_level_study_data_mode") or record.domain_context.get("row_level_study_data_mode", ""),
            "feature_contract_hash": record.feature_contract_hash,
            "label_definition": record.label_definition,
            "baseline_version_id": record.baseline_version_id,
            "anchor_baseline_version_id": record.anchor_baseline_version_id,
            "comparison_mode": record.comparison_mode,
            "decision_stage_reached": record.decision_stage_reached,
        }

        self.study_data_mode = metric_context["study_data_mode"] or self.study_data_mode
        self.row_level_study_data_mode = metric_context["row_level_study_data_mode"] or self.row_level_study_data_mode
        if record.final_decision:
            self.summary_metrics["harness_final_decision"] = record.final_decision.decision
            self.summary_metrics["harness_reason_codes"] = record.final_decision.reason_codes
            self.summary_metrics["harness_execution_mode"] = getattr(record.final_decision, "execution_mode", "") or record.execution_mode
        self.summary_metrics["harness_policy_decision"] = record.policy_decision or (record.final_decision.decision if record.final_decision else "")
        self.summary_metrics.update(
            {
                "harness_stage_count": len(record.stage_results),
                "auc": eval_metrics.get("auc"),
                "f1": eval_metrics.get("f1"),
                "recall": eval_metrics.get("recall"),
                "eval_scope": metric_context["eval_scope"],
                "task_scope": metric_context["task_scope"],
                "study_data_mode": metric_context["study_data_mode"],
                "row_level_study_data_mode": metric_context["row_level_study_data_mode"],
            }
        )
        self.extra_result["harness_v1"] = {
            "run_id": record.run_id,
            "pipeline_name": record.pipeline_name,
            "run_record_path": str(record_path),
            "final_decision": record.final_decision.decision if record.final_decision else None,
            "policy_decision": record.policy_decision or (record.final_decision.decision if record.final_decision else None),
            "decision_reason_codes": record.final_decision.reason_codes if record.final_decision else [],
            "decision_stage_reached": record.decision_stage_reached or (
                record.final_decision.decision_stage_reached if record.final_decision else ""
            ),
            "gate_name": record.final_decision.gate_name if record.final_decision else "",
            "execution_mode": record.execution_mode or (getattr(record.final_decision, "execution_mode", "") if record.final_decision else ""),
            "collected_warnings": record.collected_warnings,
            "eval_scope": record.eval_scope,
            "task_scope": record.task_scope,
            "study_data_mode": metric_context["study_data_mode"],
            "feature_contract_hash": record.feature_contract_hash,
            "label_definition": record.label_definition,
            "baseline_version_id": record.baseline_version_id,
            "anchor_baseline_version_id": record.anchor_baseline_version_id,
            "comparison_mode": record.comparison_mode,
            "candidate_metrics": {
                "auc": eval_metrics.get("auc"),
                "f1": eval_metrics.get("f1"),
                "recall": eval_metrics.get("recall"),
            },
            "baseline_metrics": eval_diagnostics.get("baseline_metrics", {}),
            "metric_context": metric_context,
            "execution_status": record.status,
        }
        self._decision(
            "harness_v1",
            self.status,
            "Harness v1 pipeline executed through StudyAgent shell.",
            {"run_record_path": str(record_path), "decision": self.extra_result["harness_v1"]["final_decision"]},
        )
        self._trace("harness_v1_completed", self.status, "Harness v1 execution completed.", self.summary_metrics)

    def run(self) -> dict[str, Any]:
        try:
            if self.request.get("execution_engine") == "harness_v1" and self.request.get("run_mode") in {"train", "review"}:
                self.run_harness_pipeline()
                self._state("completed", self.status, "StudyAgent run finished via Harness v1.")
                return self.export_result()
            self.load_data()
            self.validate_input()
            if self.request["run_mode"] == "train":
                self.build_features()
                self.quality_checked()
                self.train_layered_model()
                if self.request.get("enable_model_search", False):
                    self.run_train_enhancement()
                if self.request.get("evolution_enable", False):
                    self.run_evolution()
                self.run_llm_assistant()
            elif self.request["run_mode"] == "infer":
                self.build_features()
                self.quality_checked()
                self.run_model()
                self.fallback_checked()
                self.generate_explanation()
                self.run_llm_assistant()
                try:
                    from study_release_manager import StudyReleaseManager
                except ModuleNotFoundError:  # pragma: no cover
                    from .study_release_manager import StudyReleaseManager

                self.extra_result["serving_monitor"] = StudyReleaseManager(self.request).maybe_rollback(
                    {
                        "degraded_rate": self.summary_metrics.get("degraded_count", 0) / max(self.summary_metrics.get("prediction_rows", 1), 1),
                        "error_rate": self.summary_metrics.get("failed_count", 0) / max(self.summary_metrics.get("prediction_rows", 1), 1),
                        "explanation_failure_rate": 0.0
                        if self.summary_metrics.get("aligned_prediction_rows") == self.summary_metrics.get("prediction_rows")
                        else 1.0,
                    }
                )
            elif self.request["run_mode"] == "review":
                self.run_llm_assistant()
            elif self.request["run_mode"] == "publish":
                self.run_publish()
            elif self.request["run_mode"] == "rollback":
                self.run_rollback()
            self._state("completed", self.status, "StudyAgent run finished.")
        except Exception as exc:
            self._set_status("failed")
            self.warnings.append(str(exc))
            self._decision("agent_failure", "failed", str(exc))
            self._state("failed", "failed", str(exc))
        return self.export_result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the study domain agent control layer.")
    parser.add_argument("--request", required=True, help="Path to study agent request JSON.")
    parser.add_argument("--print-llm-config", action="store_true", help="Print LLM provider call parameters before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_llm_config:
        try:
            from study_llm_assistant import print_llm_config
        except ModuleNotFoundError:  # pragma: no cover - package import path
            from .study_llm_assistant import print_llm_config

        request_path = Path(args.request)
        request_path = request_path if request_path.is_absolute() else ROOT / request_path
        print_llm_config(StudyAgent._read_json(request_path))
    result = StudyAgent(request_path=Path(args.request)).run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
