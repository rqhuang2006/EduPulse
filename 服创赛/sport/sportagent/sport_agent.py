from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from harness.artifact_manager import export_artifact
from harness.domain_support.agent_protocol import build_agent_protocol
from sport.sportagent.sport_evaluator import evaluate_model


REPO_ROOT = Path(__file__).resolve().parents[2]
SPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = SPORT_ROOT / "conf" / "sport_agent_policy.yaml"
BASELINE_MODEL_PATH = "data/deliverables/sport/model/sport_regression_model.joblib"
BASELINE_VERSION = "sport_baseline_v1"
FROZEN_SPORT_MAINLINE = {
    "label_version": "future_v3",
    "feature_bundle": "baseline+deviation+trend",
    "structure_version": "two_stage",
    "population_version": "recoverable",
}
SPORT_BLOCKED_FEATURE_COLUMNS = {
    "sid",
    "term_id",
    "year_start",
    "semester",
    "term_order",
    "zf_score",
    "zf_grade",
    "zf_label_v2",
    "zf_label_v3",
    "sport_label_future_v1",
    "sport_label_future_v2",
    "sport_label_future_v3",
}


def _minmax_inverted(values: list[Optional[float]]) -> list[Optional[float]]:
    valid = [v for v in values if v is not None]
    if not valid:
        return [None for _ in values]
    low = min(valid)
    high = max(valid)
    if high == low:
        return [0.5 if v is not None else None for v in values]

    output: list[Optional[float]] = []
    for value in values:
        if value is None:
            output.append(None)
            continue
        norm = (value - low) / (high - low)
        output.append(max(0.0, min(1.0, 1 - norm)))
    return output


def _safe_float(value: object) -> Optional[float]:
    try:
        if value in [None, ""]:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("-"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = value
                current_section = None
            else:
                data[key] = {}
                current_section = key
        elif line.startswith("-") and current_section and isinstance(data.get(current_section), dict):
            data[current_section].setdefault("items", []).append(line[1:].strip())
    return data


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass
class SportAgentResult:
    status: str
    prediction_output: Path
    rows: int
    warnings: list[str]


class SportAgent:
    def __init__(self, request: dict[str, Any], policy_path: Path = DEFAULT_POLICY_PATH):
        print(f"[{time.time()}] sport_agent.SportAgent.__init__ BEFORE", flush=True)
        self.request = request
        self.policy = _read_policy(policy_path)
        self.input_paths = dict(request.get("input_paths", {}))
        self.warnings: list[str] = []
        self.last_training_summary: dict[str, Any] = {}
        print(f"[{time.time()}] sport_agent.SportAgent.__init__ AFTER", flush=True)

    @classmethod
    def from_request_file(cls, request_path: Path, policy_path: Path = DEFAULT_POLICY_PATH) -> "SportAgent":
        request = json.loads(request_path.read_text(encoding="utf-8"))
        return cls(request=request, policy_path=policy_path)

    def validate_input(self) -> None:
        if self.request.get("domain") != "sport":
            raise ValueError("SportAgent only accepts domain='sport'.")
        run_mode = self.request.get("run_mode")
        if run_mode not in {"train", "infer"}:
            raise ValueError("SportAgent run_mode must be train or infer.")
        required = [
            "feature_dataset",
            "prediction_output",
            "prediction_test_output",
            "quality_report",
            "validation_report",
            "model_regression",
            "model_classification",
            "model_config",
            "metrics",
        ]
        for key in required:
            if not self.input_paths.get(key):
                raise ValueError(f"SportAgent missing input_paths.{key}")

    @staticmethod
    def _load_train_domain_module() -> Any:
        module_path = Path(__file__).resolve().parent / "train_domain_models.py"
        spec = importlib.util.spec_from_file_location("train_domain_models_runtime", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_features(self) -> pd.DataFrame:
        module = self._load_train_domain_module()

        data = module.build_sport_dataset()
        if data.empty:
            raise RuntimeError("SportAgent could not build sport feature dataset.")

        feature_path = _resolve_path(REPO_ROOT, self.input_paths["feature_dataset"])
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(feature_path, index=False, encoding="utf-8-sig")
        return data

    @staticmethod
    def _feature_dataset_requires_refresh(data: pd.DataFrame) -> bool:
        required_cols = {
            "recovery_activity_rebound",
            "rhythm_activity_vs_prev",
            "recovery_return_from_zero",
            "rhythm_run_cv4",
            "stage1_state_score",
            "stage1_engagement_score",
        }
        return any(col not in data.columns for col in required_cols)

    def _refresh_trusted_mainline_metrics(self, data: pd.DataFrame) -> None:
        trusted_mainline = dict(self.last_training_summary.get("trusted_mainline") or {})
        if not trusted_mainline:
            return

        # Always recompute trusted mainline thresholded metrics during infer/review so
        # top-level harness output reflects the latest scoring policy instead of stale snapshots.
        needs_refresh = True

        if needs_refresh:
            try:
                module = self._load_train_domain_module()
                refreshed = module._run_future_experiment(
                    data,
                    FROZEN_SPORT_MAINLINE["label_version"],
                    FROZEN_SPORT_MAINLINE["feature_bundle"],
                    FROZEN_SPORT_MAINLINE["structure_version"],
                    FROZEN_SPORT_MAINLINE["population_version"],
                )
                if isinstance(refreshed, dict):
                    trusted_mainline.update({k: v for k, v in refreshed.items() if v is not None})
                ablation = {}
                for bundle_name in ["baseline_only", "baseline+deviation", "baseline+deviation+trend"]:
                    bundle_result = module._run_future_experiment(
                        data,
                        FROZEN_SPORT_MAINLINE["label_version"],
                        bundle_name,
                        FROZEN_SPORT_MAINLINE["structure_version"],
                        FROZEN_SPORT_MAINLINE["population_version"],
                    )
                    ablation[bundle_name] = bundle_result.get("auc")
                trend_auc = _safe_float(ablation.get("baseline+deviation+trend"))
                no_trend_auc = _safe_float(ablation.get("baseline+deviation"))
                if trend_auc is not None and no_trend_auc is not None:
                    trusted_mainline["ablation_summary"] = {
                        "trend_drop_auc": float(trend_auc - no_trend_auc),
                        "by_bundle": ablation,
                    }
                stage1_diagnostics = module.analyze_stage1_subgroups(
                    data,
                    FROZEN_SPORT_MAINLINE["label_version"],
                    FROZEN_SPORT_MAINLINE["feature_bundle"],
                    FROZEN_SPORT_MAINLINE["structure_version"],
                    FROZEN_SPORT_MAINLINE["population_version"],
                )
                if isinstance(stage1_diagnostics, dict):
                    self.last_training_summary["stage1_subgroup_diagnostics"] = stage1_diagnostics
                    hardest = stage1_diagnostics.get("hardest_subgroup")
                    if isinstance(hardest, dict) and hardest:
                        self.last_training_summary["hardest_subgroup"] = hardest
            except Exception as exc:
                self.warnings.append(f"trusted_mainline_metric_refresh_failed:{exc}")

        self.last_training_summary["trusted_mainline"] = trusted_mainline
        self.last_training_summary["metrics"] = {
            "auc": trusted_mainline.get("auc"),
            "f1": trusted_mainline.get("f1"),
            "precision": trusted_mainline.get("precision"),
            "recall": trusted_mainline.get("recall"),
            "rows": trusted_mainline.get("rows"),
            "eval_rows": trusted_mainline.get("eval_rows"),
            "positive_count": trusted_mainline.get("positive_count"),
            "future_window_auc": trusted_mainline.get("auc"),
        }
        protocol = self.last_training_summary.get("agent_protocol", {})
        if isinstance(protocol, dict):
            diagnosis = protocol.get("diagnosis", {})
            if isinstance(diagnosis, dict):
                diagnosis["label_version"] = trusted_mainline.get("label_version")
                diagnosis["feature_bundle"] = trusted_mainline.get("feature_bundle")
                diagnosis["structure_version"] = trusted_mainline.get("structure_version")
                diagnosis["population_version"] = trusted_mainline.get("population_version")
            comparison = protocol.get("comparison", {})
            if isinstance(comparison, dict):
                comparison["baseline_auc"] = None
                comparison["candidate_auc"] = trusted_mainline.get("auc")
                comparison["delta_auc"] = None
                comparison["future_window_auc"] = trusted_mainline.get("auc")
                if trusted_mainline.get("ablation_summary"):
                    comparison["ablation_summary"] = trusted_mainline.get("ablation_summary")
            recommendation = protocol.get("recommendation", {})
            if isinstance(recommendation, dict):
                recommendation["best_mainline_candidate"] = trusted_mainline
            proposal = protocol.get("proposal", {})
            if isinstance(proposal, dict):
                task_reconstruction = proposal.get("task_reconstruction", {})
                if isinstance(task_reconstruction, dict):
                    task_reconstruction["switch_label_version"] = trusted_mainline.get("label_version")
                    task_reconstruction["switch_target_population"] = trusted_mainline.get("population_version")
                    task_reconstruction["switch_structure_version"] = trusted_mainline.get("structure_version")
        snapshot_path = _resolve_path(REPO_ROOT, "data/deliverables/sport/docs/sport_snapshot.json")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(self.last_training_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        diagnostics_path = _resolve_path(REPO_ROOT, "data/deliverables/sport/docs/sport_stage1_subgroup_diagnostics.json")
        diagnostics = self.last_training_summary.get("stage1_subgroup_diagnostics", {})
        diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    def quality_checked(self, data: pd.DataFrame) -> dict[str, Any]:
        total_rows = int(len(data))
        missing_ratio = float(data.isna().sum().sum() / (data.shape[0] * max(data.shape[1], 1)))
        min_rows = 30
        try:
            threshold_cfg = self.policy.get("thresholds", {})
            if isinstance(threshold_cfg, dict):
                min_rows = int(threshold_cfg.get("min_rows", min_rows))
        except Exception:
            pass

        status = "success"
        if total_rows < min_rows:
            status = "degraded"
            self.warnings.append(f"row_count_below_threshold:{total_rows}<{min_rows}")
        if missing_ratio > 0.6:
            status = "degraded"
            self.warnings.append(f"high_missing_ratio:{missing_ratio:.4f}")

        report = {
            "request_id": self.request.get("request_id"),
            "domain": "sport",
            "status": status,
            "summary": {
                "rows": total_rows,
                "columns": int(data.shape[1]),
                "missing_ratio": round(missing_ratio, 6),
            },
            "warnings": self.warnings,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

        quality_path = _resolve_path(REPO_ROOT, self.input_paths["quality_report"])
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def train_layered_model(self, data: pd.DataFrame) -> dict[str, Any]:
        module = self._load_train_domain_module()

        out_dir = _resolve_path(REPO_ROOT, self.input_paths["prediction_output"]).parent
        model_regression_target = _resolve_path(REPO_ROOT, self.input_paths["model_regression"])
        model_classification_target = _resolve_path(REPO_ROOT, self.input_paths["model_classification"])
        baseline_model_path = _resolve_path(REPO_ROOT, BASELINE_MODEL_PATH)
        prior_baseline_metrics: dict[str, Any] | None = None
        if baseline_model_path.exists():
            try:
                baseline_model = joblib.load(baseline_model_path)
                baseline_pred = baseline_model.predict(data[[c for c in data.columns if c not in {"sid", "term_id", "year_start", "semester", "term_order", "zf_score", "zf_grade"}]])
                prior_baseline_metrics = evaluate_model(data["zf_score"], baseline_pred)
            except Exception as exc:
                self.warnings.append(f"baseline_evaluation_failed:{exc}")

        label_version = str(self.request.get("label_version", "future_v1"))
        feature_bundle = str(self.request.get("feature_bundle", "baseline+deviation+trend"))
        structure_version = str(self.request.get("structure_version", "single_stage"))
        population_version = str(self.request.get("population_version", "all"))
        metrics = module.train_sport_models(
            data,
            out_dir,
            label_version=label_version,
            feature_bundle=feature_bundle,
            structure_version=structure_version,
            population_version=population_version,
        )

        # Mirror trained artifacts to the infer contract paths.
        reg_target = model_regression_target
        cls_target = model_classification_target

        reg_source: Optional[Path] = None
        cls_source: Optional[Path] = None
        if isinstance(metrics, dict):
            reg_info = metrics.get("regression", {})
            cls_info = metrics.get("classification", {})
            if isinstance(reg_info, dict):
                reg_path_raw = reg_info.get("model_path")
                if reg_path_raw:
                    reg_source = _resolve_path(REPO_ROOT, str(reg_path_raw))
            if isinstance(cls_info, dict):
                cls_path_raw = cls_info.get("model_path")
                if cls_path_raw:
                    cls_source = _resolve_path(REPO_ROOT, str(cls_path_raw))

        if reg_source is None:
            reg_source = out_dir / "regression" / "best_sport_regression_model.joblib"
        if cls_source is None:
            cls_source = out_dir / "classification" / "best_sport_classification_model.joblib"

        print(
            f"[{time.time()}] sport_agent.artifact_export SOURCE regression={reg_source} classification={cls_source}",
            flush=True,
        )
        if not reg_source.exists():
            raise FileNotFoundError(f"Missing trained regression source artifact: {reg_source}")
        if not cls_source.exists():
            raise FileNotFoundError(f"Missing trained classification source artifact: {cls_source}")

        export_artifact("sport", "regression_model", reg_source, reg_target)
        export_artifact("sport", "classification_model", cls_source, cls_target)
        print(
            f"[{time.time()}] sport_agent.artifact_export TARGET regression={reg_target} classification={cls_target}",
            flush=True,
        )

        candidate_model = joblib.load(reg_target)
        feature_cols = [c for c in data.columns if c not in SPORT_BLOCKED_FEATURE_COLUMNS]
        candidate_pred = candidate_model.predict(data[feature_cols])
        candidate_eval = evaluate_model(data["zf_score"], candidate_pred)
        baseline_eval = prior_baseline_metrics or candidate_eval
        candidate_auc = candidate_eval.get("auc")
        baseline_auc = baseline_eval.get("auc")
        mainline = metrics.get("mainline_experiments", {}) if isinstance(metrics, dict) else {}
        selected_exp: dict[str, Any] = {}
        selected_source = "best_normal_candidate"
        requested_match: dict[str, Any] = {}
        selection_mode = str(self.request.get("mainline_selection_mode", "") or "").strip().lower()
        if isinstance(mainline, dict):
            for row in mainline.get("matrix", []) if isinstance(mainline.get("matrix"), list) else []:
                if not isinstance(row, dict):
                    continue
                if (
                    row.get("label_version") == label_version
                    and row.get("feature_bundle") == feature_bundle
                    and row.get("structure_version") == structure_version
                    and row.get("population_version", "all") == population_version
                ):
                    requested_match = row
                    break

            if selection_mode == "controlled_candidate":
                selected_exp = (
                    mainline.get("controlled_candidate")
                    or requested_match
                    or mainline.get("best_normal_candidate")
                    or mainline.get("best_candidate")
                    or mainline.get("selected")
                    or {}
                )
                selected_source = "controlled_candidate"
            elif selection_mode == "requested_experiment":
                selected_exp = (
                    requested_match
                    or mainline.get("best_normal_candidate")
                    or mainline.get("best_candidate")
                    or mainline.get("selected")
                    or {}
                )
                selected_source = "requested_experiment"
            elif structure_version == "two_stage_conservative":
                selected_exp = (
                    requested_match
                    or mainline.get("controlled_candidate")
                    or mainline.get("best_normal_candidate")
                    or mainline.get("best_candidate")
                    or mainline.get("selected")
                    or {}
                )
                selected_source = "two_stage_conservative_request"
            else:
                selected_exp = (
                    mainline.get("best_normal_candidate")
                    or mainline.get("best_candidate")
                    or mainline.get("selected")
                    or {}
                )
        future_window_auc = selected_exp.get("auc") if isinstance(selected_exp, dict) else None
        suspicious_high_auc = bool(mainline.get("suspicious_high_auc")) if isinstance(mainline, dict) else False
        tautology_risk = str(mainline.get("tautology_risk", "unknown")) if isinstance(mainline, dict) else "unknown"
        if candidate_auc is None or baseline_auc is None:
            decision = "keep_baseline"
        else:
            delta_auc = float(candidate_auc - baseline_auc)
            if delta_auc > 0.005:
                decision = "promote_candidate"
            else:
                decision = "keep_baseline"
        if suspicious_high_auc:
            decision = "hold_for_review"
        if tautology_risk == "high":
            decision = "hold_for_review"
        if isinstance(future_window_auc, float) and future_window_auc < 0.8:
            decision = "hold_for_review"
        delta_auc = None
        if candidate_auc is not None and baseline_auc is not None:
            delta_auc = float(candidate_auc - baseline_auc)

        # Hardest subgroup based on classification error rate against zf_label_v3.
        hardest_subgroup: dict[str, Any] = {}
        if "zf_label_v3" in data.columns and "zf_grade" in data.columns:
            eval_df = data[["zf_label_v3", "zf_grade", "sid"]].copy()
            eval_df["pred_fail"] = (pd.Series(candidate_pred).astype(float) < 60.0).astype(int)
            eval_df["true_fail"] = (pd.to_numeric(data["zf_score"], errors="coerce").fillna(0.0) < 60.0).astype(int)
            eval_df["err"] = (eval_df["pred_fail"] != eval_df["true_fail"]).astype(int)
            rows = []
            for col in ["zf_label_v3", "zf_grade"]:
                for k, g in eval_df.groupby(col, dropna=False):
                    rows.append({"group": f"{col}:{k}", "rows": int(len(g)), "error_rate": float(g["err"].mean())})
            if rows:
                hardest_subgroup = sorted(rows, key=lambda x: (-x["error_rate"], -x["rows"]))[0]
        failure_types: list[str] = []
        if suspicious_high_auc:
            failure_types.append("suspicious_high_auc")
        if tautology_risk == "high":
            failure_types.append("high_tautology_risk")
        if isinstance(future_window_auc, float) and future_window_auc < 0.75:
            failure_types.append("low_future_signal")
        if hardest_subgroup and float(hardest_subgroup.get("error_rate", 0.0)) > 0.2:
            failure_types.append("subgroup_collapse")
        if not failure_types:
            failure_types.append("none")

        protocol = build_agent_protocol(
            domain="sport",
            diagnosis={
                "task_type": "future_window_prediction",
                "failure_types": failure_types,
                "label_versions": ["zf_grade", "zf_label_v2", "zf_label_v3", "sport_label_future_v1", "sport_label_future_v2", "sport_label_future_v3"],
                "label_version": selected_exp.get("label_version", label_version),
                "feature_bundle": selected_exp.get("feature_bundle", feature_bundle),
                "structure_version": selected_exp.get("structure_version", structure_version),
                "population_version": selected_exp.get("population_version", population_version),
                "future_window": "1-2 terms",
                "trend_bundle_enabled": selected_exp.get("feature_bundle") == "baseline+deviation+trend",
                "tautology_risk": tautology_risk,
                "trust_flags": {"suspicious_high_auc": suspicious_high_auc},
            },
            proposal={
                "candidate_model_path": str(reg_target),
                "task_reconstruction": {
                    "switch_label_version": "future_v3" if "low_future_signal" in failure_types else selected_exp.get("label_version", label_version),
                    "switch_time_window": "short_window_1_2_terms",
                    "switch_target_population": "recoverable" if "low_future_signal" in failure_types else selected_exp.get("population_version", population_version),
                    "switch_structure_version": "single_stage" if "low_future_signal" in failure_types else selected_exp.get("structure_version", structure_version),
                },
                "next_optimization_target": "improve_future_window_auc",
            },
            comparison={
                "baseline_auc": None,
                "candidate_auc": selected_exp.get("auc"),
                "delta_auc": None,
                "future_window_auc": future_window_auc,
                "is_future_window": True,
                "same_source_risk": tautology_risk,
                "subgroup_stable": not ("subgroup_collapse" in failure_types),
                "temporal_holdout_stable": bool(isinstance(future_window_auc, float) and future_window_auc >= 0.8),
                "more_real_than_baseline": bool(isinstance(future_window_auc, float) and future_window_auc >= 0.8),
                "ablation_summary": mainline.get("ablation_summary", {}) if isinstance(mainline, dict) else {},
            },
            recommendation={
                "decision": decision,
                "comparable": True,
                "best_mainline_candidate": selected_exp,
                "selected_experiment_source": selected_source,
                "high_score_but_untrusted": bool("suspicious_high_auc" in failure_types or "high_tautology_risk" in failure_types),
                "real_but_weak": bool("low_future_signal" in failure_types),
                "next_priority_route": "target_population_and_future_transition_search",
                "blocking_reason": "risk_gate_triggered" if decision == "hold_for_review" else "",
            },
        )

        snapshot = {
            "domain": "sport",
            "status": "completed",
            "decision": decision,
            "metrics": {
                "auc": selected_exp.get("auc"),
                "f1": selected_exp.get("f1"),
                "precision": selected_exp.get("precision"),
                "recall": selected_exp.get("recall"),
                "rows": selected_exp.get("rows"),
                "eval_rows": selected_exp.get("eval_rows"),
                "positive_count": selected_exp.get("positive_count"),
                "future_window_auc": future_window_auc,
            },
            "artifacts": {
                "baseline_model": str(baseline_model_path),
                "candidate_model": str(reg_target),
                "candidate_classification_model": str(cls_target),
            },
            "candidate_model_path": str(reg_target),
            "metrics_candidate": candidate_eval,
            "comparable": True,
            "baseline_version": BASELINE_VERSION,
            "hardest_subgroup": hardest_subgroup,
            "task_type": "future_window_prediction",
            "mainline_validity": decision != "hold_for_review",
            "tautology_risk": tautology_risk,
            "future_window_auc": future_window_auc,
            "suspicious_high_auc": suspicious_high_auc,
            "selected_experiment_source": selected_source,
            "search_contract": self.request.get("search_contract", {}),
            "trusted_mainline": selected_exp,
            "controlled_candidate": mainline.get("controlled_candidate", {}) if isinstance(mainline, dict) else {},
            "requested_experiment": requested_match,
            "mainline_frozen": True,
            "comparison_reference": {
                "baseline_auc": baseline_auc,
                "candidate_auc": candidate_auc,
                "delta_auc": delta_auc,
                "baseline_f1": baseline_eval.get("f1"),
                "candidate_f1": candidate_eval.get("f1"),
                "baseline_precision": baseline_eval.get("precision"),
                "candidate_precision": candidate_eval.get("precision"),
                "baseline_recall": baseline_eval.get("recall"),
                "candidate_recall": candidate_eval.get("recall"),
            },
            "ablation_summary": mainline.get("ablation_summary", {}) if isinstance(mainline, dict) else {},
            "experiment_matrix": mainline.get("matrix", []) if isinstance(mainline, dict) else [],
            "agent_protocol": protocol,
        }
        snapshot_path = _resolve_path(REPO_ROOT, "data/deliverables/sport/docs/sport_snapshot.json")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_training_summary = snapshot

        metrics_path = _resolve_path(REPO_ROOT, self.input_paths["metrics"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        model_config = {
            "domain": "sport",
            "model_version": self.request.get("model_version", "sport_v1"),
            "feature_version": self.request.get("feature_version", "sport_feature_v1"),
            "request_id": self.request.get("request_id"),
            "run_mode": self.request.get("run_mode"),
            "regression_model": str(_resolve_path(REPO_ROOT, self.input_paths["model_regression"])),
            "classification_model": str(_resolve_path(REPO_ROOT, self.input_paths["model_classification"])),
            "metrics": metrics,
        }
        model_cfg_path = _resolve_path(REPO_ROOT, self.input_paths["model_config"])
        model_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        model_cfg_path.write_text(json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")

        validation_path = _resolve_path(REPO_ROOT, self.input_paths["validation_report"])
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "domain": "sport",
                    "request_id": self.request.get("request_id"),
                    "metrics_summary": metrics,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return metrics

    def run_model(self, data: pd.DataFrame) -> SportAgentResult:
        model_regression = _resolve_path(REPO_ROOT, self.input_paths["model_regression"])
        model_classification = _resolve_path(REPO_ROOT, self.input_paths["model_classification"])
        pred_path = _resolve_path(REPO_ROOT, self.input_paths["prediction_output"])
        pred_test_path = _resolve_path(REPO_ROOT, self.input_paths["prediction_test_output"])

        if not model_regression.exists():
            raise FileNotFoundError(f"Missing regression model: {model_regression}")

        feature_cols = [c for c in data.columns if c not in SPORT_BLOCKED_FEATURE_COLUMNS]

        reg_model = joblib.load(model_regression)
        out = data[["sid", "term_id", "zf_score"]].copy()
        out["pred_zf_score"] = reg_model.predict(data[feature_cols])

        if model_classification.exists():
            try:
                cls_model = joblib.load(model_classification)
                out["pred_zf_grade"] = cls_model.predict(data[feature_cols])
            except Exception as exc:
                self.warnings.append(f"classification_prediction_failed:{exc}")
                out["pred_zf_grade"] = ""
        else:
            out["pred_zf_grade"] = ""

        raw_scores = [_safe_float(v) for v in out["pred_zf_score"].tolist()]
        risks = _minmax_inverted(raw_scores)
        out["pred_fail_proba"] = [0.5 if v is None else float(v) for v in risks]
        out["best_cls_model"] = "SportAgent"

        pred_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(pred_path, index=False, encoding="utf-8-sig")
        out.to_csv(pred_test_path, index=False, encoding="utf-8-sig")

        status = "degraded" if self.warnings else "success"
        return SportAgentResult(status=status, prediction_output=pred_path, rows=int(len(out)), warnings=self.warnings)

    def run(self) -> dict[str, Any]:
        print(f"[{time.time()}] sport_agent.SportAgent.run ENTER", flush=True)
        print(f"[{time.time()}] sport_agent.validate_input BEFORE", flush=True)
        self.validate_input()
        print(f"[{time.time()}] sport_agent.validate_input AFTER", flush=True)

        feature_path = _resolve_path(REPO_ROOT, self.input_paths["feature_dataset"])
        run_mode = str(self.request.get("run_mode"))

        print(f"[{time.time()}] sport_agent.feature_data_check BEFORE", flush=True)
        data = pd.DataFrame()
        if feature_path.exists():
            try:
                data = pd.read_csv(feature_path)
            except Exception as exc:
                self.warnings.append(f"feature_dataset_read_failed:{exc}")
                data = pd.DataFrame()

        if data.empty or self._feature_dataset_requires_refresh(data):
            print(f"[{time.time()}] sport_agent.build_sport_dataset/features BEFORE", flush=True)
            data = self._build_features()
            print(f"[{time.time()}] sport_agent.build_sport_dataset/features AFTER", flush=True)
        print(f"[{time.time()}] sport_agent.feature_data_check AFTER", flush=True)

        quality = self.quality_checked(data)

        if run_mode == "train":
            self.train_layered_model(data)
        elif not self.last_training_summary:
            snapshot_path = _resolve_path(REPO_ROOT, "data/deliverables/sport/docs/sport_snapshot.json")
            if snapshot_path.exists():
                try:
                    self.last_training_summary = json.loads(snapshot_path.read_text(encoding="utf-8"))
                except Exception:
                    self.last_training_summary = {}

        if self.last_training_summary:
            self._refresh_trusted_mainline_metrics(data)

        print(f"[{time.time()}] sport_agent.model_load BEFORE", flush=True)
        prediction = self.run_model(data)
        print(f"[{time.time()}] sport_agent.model_load AFTER", flush=True)

        result = {
            "status": prediction.status,
            "domain": "sport",
            "run_mode": run_mode,
            "request_id": self.request.get("request_id"),
            "rows": prediction.rows,
            "prediction_output": str(prediction.prediction_output),
            "warnings": prediction.warnings,
            "quality": quality,
            "decision": self.last_training_summary.get("decision", "keep_baseline"),
            "final_decision": self.last_training_summary.get("decision", "keep_baseline"),
            "policy_decision": self.last_training_summary.get("decision", "keep_baseline"),
            "decision_stage_reached": "active_baseline_comparison",
            "summary_metrics": self.last_training_summary.get("metrics", {"rows": prediction.rows, "status": prediction.status}),
            "harness_v1": {
                "domain": "sport",
                "status": "completed",
                "decision": self.last_training_summary.get("decision", "keep_baseline"),
                "comparable": True,
                "metrics": self.last_training_summary.get("metrics", {}),
                "artifacts": self.last_training_summary.get("artifacts", {}),
                "task_type": self.last_training_summary.get("task_type", "future_window_prediction"),
                "mainline_validity": self.last_training_summary.get("mainline_validity", True),
                "blocking_reason": "risk_gate_triggered" if self.last_training_summary.get("decision") == "hold_for_review" else "",
                "next_optimization_target": "improve_future_window_auc",
                "tautology_risk": self.last_training_summary.get("tautology_risk", ""),
                "suspicious_high_auc": self.last_training_summary.get("suspicious_high_auc", False),
                "baseline_version_id": BASELINE_VERSION,
                "candidate_model_path": self.last_training_summary.get("candidate_model_path"),
                "hardest_subgroup": self.last_training_summary.get("hardest_subgroup", {}),
                "trusted_mainline": self.last_training_summary.get("trusted_mainline", FROZEN_SPORT_MAINLINE),
                "mainline_frozen": self.last_training_summary.get("mainline_frozen", True),
                "agent_protocol": self.last_training_summary.get("agent_protocol", {}),
            },
        }
        print(f"[{time.time()}] sport_agent.result_return BEFORE", flush=True)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SportAgent with a JSON request file.")
    parser.add_argument("--request", type=Path, required=True, help="Path to sport agent request JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = SportAgent.from_request_file(args.request)
    result = agent.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
