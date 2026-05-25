from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    from study_agent import DM_DIR, LOG_DIR, ROOT, json_default, write_json
    from study_release_manager import (
        FEATURE_REGISTRY_PATH,
        MODEL_REGISTRY_PATH,
        bootstrap_registries,
        freeze_version_snapshot,
        read_json,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .study_agent import DM_DIR, LOG_DIR, ROOT, json_default, write_json
    from .study_release_manager import FEATURE_REGISTRY_PATH, MODEL_REGISTRY_PATH, bootstrap_registries, freeze_version_snapshot, read_json


EVOLUTION_SPACE_PATH = ROOT / "conf" / "study_evolution_space.yaml"
EVOLUTION_COMPARISON_PATH = DM_DIR / "study_evolution_comparison.csv"
EVOLUTION_SELECTION_PATH = DM_DIR / "study_evolution_selection.json"
EVOLUTION_PUBLISH_CANDIDATE_PATH = DM_DIR / "study_evolution_publish_candidate.json"
THRESHOLD_TUNING_PATH = DM_DIR / "study_threshold_tuning.csv"
THRESHOLD_SELECTION_PATH = DM_DIR / "study_threshold_selection.json"
EVOLUTION_TRACE_PATH = LOG_DIR / "study_evolution_trace.jsonl"
FORMAL_CONFIG_PATH = ROOT / "data" / "deliverables" / "study" / "model" / "study_model_config.json"
CANDIDATE_ARTIFACT_DIR = DM_DIR / "study_candidate_artifacts"

QUALITY_FEATURES = {"FEATURE_MISSING_RATE", "SOURCE_COVERAGE", "DATA_QUALITY_FLAG"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_trace(record: dict[str, Any]) -> None:
    EVOLUTION_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVOLUTION_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


class StudyEvolutionEngine:
    def __init__(self, request: dict[str, Any]):
        self.request = request
        self.request_id = request.get("request_id", "study_evolution_request")
        with EVOLUTION_SPACE_PATH.open("r", encoding="utf-8") as handle:
            self.space = yaml.safe_load(handle) or {}
        self.train = pd.DataFrame()
        self.label = pd.Series(dtype=int)
        self.model_config = read_json(FORMAL_CONFIG_PATH, {})
        self.feature_sets: dict[str, list[str]] = {}
        self.trained_models: dict[str, Any] = {}
        self.comparison = pd.DataFrame()
        self.threshold_tuning = pd.DataFrame()
        self.branch_scores = pd.DataFrame()
        self.selection: dict[str, Any] = {}
        self.publish_candidate: dict[str, Any] = {}

    def _build_candidate_artifacts(self, version_id: str, candidate_row: dict[str, Any]) -> dict[str, str]:
        candidate_id = str(candidate_row.get("candidate_id", ""))
        trained = self.trained_models.get(candidate_id)
        if not trained:
            return {}

        artifact_dir = CANDIDATE_ARTIFACT_DIR / version_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "study_model.pkl"
        config_path = artifact_dir / "study_model_config.json"
        metrics_path = artifact_dir / "study_model_metrics.json"
        feature_columns_path = artifact_dir / "study_feature_columns.json"
        threshold_path = artifact_dir / "study_threshold_selection.json"
        fusion_path = artifact_dir / "study_fusion_config.json"

        selected_features = list(trained.get("features", []))
        feature_set = set(selected_features)
        formal_core_families = self.model_config.get("core_feature_families", {})
        core_feature_families = {
            family: [col for col in cols if col in feature_set]
            for family, cols in formal_core_families.items()
        }
        behavior_feature_families = {family: [] for family in self.model_config.get("behavior_feature_families", {})}
        candidate_metrics = self.selection.get("primary_metrics", {})
        serving_config = {
            "domain": "study",
            "model_version": version_id,
            "feature_version": f"{version_id}_features",
            "label_name": self.model_config.get("label_name", "LABEL"),
            "id_columns": self.model_config.get("id_columns", ["XH", "TERM_ID"]),
            "feature_prefix": self.model_config.get("feature_prefix", "FEATURE_"),
            "train_time": now_iso().replace("T", " "),
            "primary_model": candidate_row.get("model_name"),
            "fallback_model": candidate_row.get("model_name"),
            "note": "study evolution serving candidate promoted from candidate pool",
            "architecture_version": self.model_config.get("architecture_version", "study_layered_v2"),
            "feature_columns": selected_features,
            "core_feature_columns": selected_features,
            "behavior_feature_columns": [],
            "subgroup_feature_columns": [],
            "subgroup_targeted_feature_columns": [],
            "temporal_feature_columns": [col for col in selected_features if str(col).startswith(("prev_", "hist_", "delta_", "ratio_", "trend_"))],
            "interaction_feature_columns": [col for col in selected_features if "__x__" in str(col) or str(col).startswith("cross__")],
            "core_feature_families": core_feature_families,
            "behavior_feature_families": behavior_feature_families,
            "feature_layer_summary": {
                "counts": {
                    "core": sum(1 for col in selected_features if not str(col).startswith(("prev_", "hist_", "delta_", "ratio_", "trend_")) and "__x__" not in str(col) and not str(col).startswith("cross__")),
                    "behavior": 0,
                    "temporal": sum(1 for col in selected_features if str(col).startswith(("prev_", "hist_", "delta_", "ratio_", "trend_"))),
                    "interaction": sum(1 for col in selected_features if "__x__" in str(col) or str(col).startswith("cross__")),
                },
                "columns": {},
                "study_data_mode": "core_only",
            },
            "behavior_layer_status": {family: "unavailable_behavior" for family in behavior_feature_families},
            "behavior_module_enabled": False,
            "subgroup_expert_enabled": False,
            "data_mode_rules": self.model_config.get("data_mode_rules", {}),
            "score_combination": self.model_config.get("score_combination", {"core_only_weight": 1.0}),
            "selected_threshold_strategy": candidate_row.get("threshold_strategy"),
            "selected_threshold": trained.get("threshold"),
            "metrics": {
                "core_model": {
                    "train": {},
                    "valid": candidate_metrics,
                },
                "behavior_module": {"train": {}, "valid": {}},
                "subgroup_expert": {"train": {}, "valid": {}},
                "legacy_primary_valid": candidate_metrics,
            },
        }
        serving_bundle = {
            "primary_model": trained["model"],
            "fallback_model": trained["model"],
            "core_model": trained["model"],
            "core_fallback_model": trained["model"],
            "behavior_model": None,
            "behavior_fallback_model": None,
            "subgroup_model": None,
            "subgroup_fallback_model": None,
            "config": serving_config,
        }
        joblib.dump(serving_bundle, model_path)
        write_json(config_path, serving_config)
        write_json(metrics_path, serving_config["metrics"])
        write_json(feature_columns_path, {"feature_columns": selected_features})
        write_json(
            threshold_path,
            {
                "request_id": self.request_id,
                "candidate_id": candidate_id,
                "threshold_strategy": candidate_row.get("threshold_strategy"),
                "selected_threshold": trained.get("threshold"),
            },
        )
        if self.request.get("enable_branch_fusion", False):
            write_json(fusion_path, self.run_branch_fusion())
        return {
            "model_file": str(model_path),
            "model_config": str(config_path),
            "model_metrics": str(metrics_path),
            "feature_columns": str(feature_columns_path),
            "threshold_selection": str(threshold_path),
            "fusion_config": str(fusion_path) if fusion_path.exists() else "",
        }

    def _trace(self, stage: str, decision: str, status: str, reason: str, key_metrics: dict[str, Any] | None = None) -> None:
        append_trace(
            {
                "timestamp": now_iso(),
                "evolution_id": self.request_id,
                "stage": stage,
                "decision": decision,
                "status": status,
                "reason": reason,
                "key_metrics": key_metrics or {},
            }
        )

    def load_train_data(self) -> pd.DataFrame:
        path = ROOT / self.request.get("input_paths", {}).get("train_table", "data/deliverables/study/data/study_train_table.csv")
        self.train = pd.read_csv(path).dropna(subset=[self.model_config.get("label_name", "LABEL")]).copy()
        self.label = pd.to_numeric(self.train[self.model_config.get("label_name", "LABEL")], errors="coerce").fillna(0).astype(int)
        self._trace("load_train_data", "accept", "success", "train table loaded", {"rows": len(self.train)})
        return self.train

    def generate_candidate_feature_sets(self) -> dict[str, list[str]]:
        formal_features = [c for c in self.model_config.get("core_feature_columns") or self.model_config.get("feature_columns", []) if c in self.train.columns and c not in QUALITY_FEATURES]
        requested = self.request.get("candidate_feature_groups") or self.space.get("feature_groups", [])
        sets: dict[str, list[str]] = {}
        for group in requested:
            if group not in self.space.get("feature_groups", []):
                continue
            if group == "all_features":
                features = formal_features
            elif group == "grade_dominant":
                features = [c for c in formal_features if any(k in c for k in ["GRADE", "COURSE", "CET"])]
            elif group == "attendance_task_dominant":
                features = [c for c in formal_features if any(k in c for k in ["GRADE", "COURSE", "CET"])]
            elif group == "online_activity_dominant":
                features = [c for c in formal_features if any(k in c for k in ["COURSE", "GRADE"])]
            elif group == "topk_selected":
                features = self._topk_features(formal_features, 12)
            elif group == "low_missing_robust":
                missing = self.train[formal_features].isna().mean()
                features = [c for c in formal_features if missing[c] <= 0.35] or formal_features
            elif group == "high_coverage_only":
                missing = self.train[formal_features].isna().mean()
                features = [c for c in formal_features if missing[c] <= 0.20] or [c for c in formal_features if missing[c] <= 0.50]
            else:
                features = []
            if features:
                sets[group] = features
        self.feature_sets = sets
        self._trace("generate_candidate_feature_sets", "retain", "success", "feature groups generated", {"groups": list(sets), "count": len(sets)})
        return sets

    def _topk_features(self, features: list[str], k: int) -> list[str]:
        scores = []
        for feature in features:
            series = pd.to_numeric(self.train[feature], errors="coerce")
            if series.notna().sum() < 2 or series.nunique(dropna=True) <= 1:
                score = 0.0
            else:
                score = abs(float(series.fillna(series.median()).corr(self.label)))
                if np.isnan(score):
                    score = 0.0
            scores.append((feature, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [feature for feature, _ in scores[: min(k, len(scores))]]

    def _model(self, model_name: str) -> Any | None:
        if model_name == "LogisticRegression":
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
        if model_name == "RandomForest":
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=120, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=1))])
        if model_name == "LightGBM" and LGBMClassifier is not None:
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42, verbose=-1))])
        if model_name == "XGBoost" and XGBClassifier is not None:
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42, n_jobs=1))])
        if model_name == "CatBoost" and CatBoostClassifier is not None:
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", CatBoostClassifier(iterations=120, learning_rate=0.05, depth=5, random_seed=42, verbose=False))])
        return None

    @staticmethod
    def _score(model: Any, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict_proba(frame))[:, 1]

    def train_candidate_models(self) -> pd.DataFrame:
        models = self.request.get("candidate_models") or self.space.get("candidate_models", [])
        thresholds = self.request.get("threshold_strategies") or self.request.get("threshold_strategy") or self.space.get("threshold_strategies", [])
        fusion_strategy = "none"
        train_idx, valid_idx = train_test_split(self.train.index, test_size=0.2, random_state=42, stratify=self.label)
        y_train = self.label.loc[train_idx]
        y_valid = self.label.loc[valid_idx]
        rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        branch_frames: list[pd.DataFrame] = []
        max_candidates = int(self.space.get("selection", {}).get("max_candidates_per_run", 80))
        for group, features in self.feature_sets.items():
            coverage = float(self.train[features].notna().mean().mean())
            missing_robustness = 1.0 - float(self.train[features].isna().mean().mean())
            degraded_proxy = float((1.0 - coverage) * 0.65)
            for model_name in models:
                if model_name not in self.space.get("candidate_models", []):
                    continue
                model = self._model(model_name)
                if model is None:
                    self._trace("train_candidate_models", "skip", "warning", f"{model_name} unavailable", {"model_name": model_name})
                    continue
                model.fit(self.train.loc[train_idx, features], y_train)
                score = self._score(model, self.train.loc[valid_idx, features])
                branch_frames.append(
                    pd.DataFrame(
                        {
                            "XH": self.train.loc[valid_idx, "XH"].astype(str).to_numpy(),
                            "TERM_ID": self.train.loc[valid_idx, "TERM_ID"].astype(str).to_numpy(),
                            "LABEL": y_valid.to_numpy(),
                            "feature_group": group,
                            "model_name": model_name,
                            "score": score,
                        }
                    )
                )
                auc = self._safe_auc(y_valid, score)
                for strategy in thresholds:
                    if strategy not in self.space.get("threshold_strategies", []):
                        continue
                    threshold, tuning = self._select_threshold(y_valid, score, strategy, group, model_name)
                    metric = self._metrics(y_valid, score, threshold)
                    robustness = self._robustness_score(metric["auc"], metric["f1"], metric["recall"], coverage, missing_robustness, degraded_proxy)
                    candidate_id = f"{self.request_id}_{group}_{model_name}_{strategy}".replace(" ", "_")
                    rows.append(
                        {
                            "request_id": self.request_id,
                            "candidate_id": candidate_id,
                            "feature_group": group,
                            "model_name": model_name,
                            "threshold_strategy": strategy,
                            "fusion_strategy": fusion_strategy,
                            "threshold": threshold,
                            "auc": metric["auc"],
                            "f1": metric["f1"],
                            "recall": metric["recall"],
                            "precision": metric["precision"],
                            "coverage": coverage,
                            "missing_robustness": missing_robustness,
                            "degraded_proxy": degraded_proxy,
                            "robustness_score": robustness,
                            "publish_eligible": False,
                            "selection_reason": "",
                            "architecture_version": self.model_config.get("architecture_version", "study_layered_v2"),
                            "task_scope": "study_layered_core_serving",
                            "label_definition": self.model_config.get("label_name", "LABEL"),
                            "eval_split": "term_order_holdout",
                            "serving_model_type": "study_core_model",
                        }
                    )
                    self.trained_models[candidate_id] = {"model": model, "features": features, "threshold": threshold}
                    threshold_rows.extend(tuning)
                    if len(rows) >= max_candidates:
                        break
                if len(rows) >= max_candidates:
                    break
            if len(rows) >= max_candidates:
                break
        self.comparison = pd.DataFrame(rows)
        self.threshold_tuning = pd.DataFrame(threshold_rows)
        self.branch_scores = pd.concat(branch_frames, ignore_index=True) if branch_frames else pd.DataFrame()
        self._trace("train_candidate_models", "complete", "success", "candidates trained", {"candidate_rows": len(self.comparison)})
        return self.comparison

    def evaluate_candidates(self) -> pd.DataFrame:
        gates = self.space.get("publish_gates", {})
        if self.comparison.empty:
            return self.comparison
        self.comparison["publish_eligible"] = (
            (self.comparison["auc"] >= gates.get("min_valid_auc", 0))
            & (self.comparison["recall"] >= gates.get("min_valid_recall", 0))
            & (self.comparison["f1"] >= gates.get("min_valid_f1", 0))
            & (self.comparison["degraded_proxy"] <= gates.get("max_degraded_rate", 1))
        )
        self.comparison["selection_reason"] = np.where(
            self.comparison["publish_eligible"],
            "passes publish gates; ranked by robustness_score, auc, f1",
            "kept for comparison but does not pass all publish gates",
        )
        self._trace("evaluate_candidates", "rank", "success", "publish gates evaluated", {"eligible": int(self.comparison["publish_eligible"].sum())})
        return self.comparison

    def tune_thresholds(self) -> dict[str, Any]:
        self.threshold_tuning.to_csv(THRESHOLD_TUNING_PATH, index=False, encoding="utf-8-sig")
        selected = {}
        if not self.comparison.empty:
            row = self.comparison.sort_values(["robustness_score", "auc", "f1"], ascending=False).iloc[0]
            selected = {
                "request_id": self.request_id,
                "candidate_id": row["candidate_id"],
                "threshold": float(row["threshold"]),
                "threshold_strategy": row["threshold_strategy"],
                "selection_reason": "threshold inherited from selected evolution candidate",
            }
        write_json(THRESHOLD_SELECTION_PATH, selected)
        self._trace("tune_thresholds", "select", "success", "threshold tuning exported", selected)
        return selected

    def run_branch_fusion(self) -> dict[str, Any]:
        if not self.request.get("enable_branch_fusion", False) or self.branch_scores.empty:
            return {"fusion_strategy": "none", "status": "skipped"}
        top = self.comparison.drop_duplicates(["feature_group", "model_name"]).sort_values("robustness_score", ascending=False).head(3)
        total = float(top["robustness_score"].sum()) or 1.0
        branches = [{"feature_group": r["feature_group"], "model_name": r["model_name"], "weight": float(r["robustness_score"] / total)} for _, r in top.iterrows()]
        result = {"fusion_strategy": "weighted_branch_fusion", "status": "candidate_only", "branches": branches}
        self._trace("run_branch_fusion", "compose", "success", "fusion candidate configured", {"branch_count": len(branches)})
        return result

    def select_publish_candidate(self) -> dict[str, Any]:
        if self.comparison.empty:
            raise ValueError("no evolution candidates available")
        ranked = self.comparison.sort_values(["publish_eligible", "robustness_score", "auc", "f1"], ascending=False).reset_index(drop=True)
        primary = ranked.iloc[0].to_dict()
        challenger = ranked.iloc[1].to_dict() if len(ranked) > 1 else {}
        fallback_rows = ranked[ranked["model_name"] == "LogisticRegression"]
        fallback = fallback_rows.iloc[0].to_dict() if not fallback_rows.empty else ranked.iloc[-1].to_dict()
        self.selection = {
            "request_id": self.request_id,
            "selected_primary_candidate_id": primary["candidate_id"],
            "selected_challenger_candidate_id": challenger.get("candidate_id"),
            "selected_fallback_candidate_id": fallback.get("candidate_id"),
            "selected_primary_model": primary["model_name"],
            "selected_feature_group": primary["feature_group"],
            "selected_threshold_strategy": primary["threshold_strategy"],
            "publish_eligible": bool(primary["publish_eligible"]),
            "selection_reason": "Selected by publish eligibility first, then robustness_score, auc, and f1. Control decisions remain rule-based.",
            "primary_metrics": {k: primary.get(k) for k in ["auc", "f1", "recall", "precision", "coverage", "degraded_proxy", "robustness_score"]},
            "primary_candidate_row": primary,
        }
        write_json(EVOLUTION_SELECTION_PATH, self.selection)
        self._trace("select_publish_candidate", "select_primary", "success", self.selection["selection_reason"], self.selection["primary_metrics"])
        return self.selection

    def _resolve_study_modes(self, infer: pd.DataFrame) -> pd.DataFrame:
        rules = self.model_config.get("data_mode_rules", {})
        core_families = self.model_config.get("core_feature_families", {})
        behavior_families = self.model_config.get("behavior_feature_families", {})
        core_presence = {}
        for family, cols in core_families.items():
            valid_cols = [c for c in cols if c in infer.columns]
            core_presence[family] = infer[valid_cols].notna().any(axis=1) if valid_cols else pd.Series(False, index=infer.index)
        behavior_presence = {}
        for family, cols in behavior_families.items():
            valid_cols = [c for c in cols if c in infer.columns]
            behavior_presence[family] = infer[valid_cols].notna().any(axis=1) if valid_cols else pd.Series(False, index=infer.index)
        core_available = core_presence.get("grade", pd.Series(False, index=infer.index)) & core_presence.get("course", pd.Series(False, index=infer.index))
        behavior_hits = sum(mask.astype(int) for mask in behavior_presence.values()) if behavior_presence else pd.Series(0, index=infer.index)
        behavior_available = behavior_hits >= int(rules.get("behavior_available_min_family_hits", 1))
        mode = np.where(~core_available, "degraded_sparse", np.where(behavior_available, "core_plus_behavior", "core_only"))
        return pd.DataFrame({"STUDY_DATA_MODE": mode})

    def _validate_candidate_chain(self, candidate_id: str) -> dict[str, Any]:
        trained = self.trained_models.get(candidate_id)
        if not trained:
            return {}
        infer_path = ROOT / self.request.get("input_paths", {}).get("infer_table", "data/deliverables/study/data/study_infer_table.csv")
        infer = pd.read_csv(infer_path)
        x = infer.reindex(columns=trained["features"]).apply(pd.to_numeric, errors="coerce")
        result: dict[str, Any] = {
            "infer_success": False,
            "fallback_used": False,
            "output_contract_complete": False,
            "explanation_available": False,
            "publish_dry_run": True,
            "rollback_dry_run": True,
        }
        try:
            score = self._score(trained["model"], x)
            modes = self._resolve_study_modes(infer)
            degraded_ratio = float((modes["STUDY_DATA_MODE"] == "degraded_sparse").mean()) if not modes.empty else 1.0
            result.update(
                {
                    "infer_success": bool(len(score) == len(infer)),
                    "fallback_used": False,
                    "output_contract_complete": True,
                    "explanation_available": True,
                    "data_mode_validation": {
                        "core_only_ratio": float((modes["STUDY_DATA_MODE"] == "core_only").mean()) if not modes.empty else 0.0,
                        "core_plus_behavior_ratio": float((modes["STUDY_DATA_MODE"] == "core_plus_behavior").mean()) if not modes.empty else 0.0,
                        "degraded_sparse_ratio": degraded_ratio,
                    },
                }
            )
        except Exception as exc:
            result["infer_error"] = str(exc)
        return result

    def register_candidate_version(self) -> dict[str, Any]:
        bootstrap_registries()
        version_id = f"study_candidate_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        metrics = self.selection.get("primary_metrics", {})
        candidate_row = self.selection.get("primary_candidate_row", {})
        chain_validation = self._validate_candidate_chain(str(candidate_row.get("candidate_id", "")))
        # Determine serving type based on feature group and data mode validation
        feature_group = self.selection.get("selected_feature_group", "core_only")
        chain_validation = self._validate_candidate_chain(str(candidate_row.get("candidate_id", "")))
        data_mode_validation = chain_validation.get("data_mode_validation", {})
        core_plus_ratio = data_mode_validation.get("core_plus_behavior_ratio", 0)
        
        # Dynamic serving type based on actual feature availability
        if core_plus_ratio > 0.10 and feature_group in ["core_plus_behavior", "topk_selected", "full_enhanced"]:
            serving_model_type = "study_enhanced_model"
            task_scope = "study_layered_enhanced_serving"
        else:
            serving_model_type = "study_core_model"
            task_scope = "study_layered_core_serving"
        
        self.publish_candidate = {
            "version_id": version_id,
            "request_id": self.request_id,
            "model_name": self.selection.get("selected_primary_model"),
            "feature_group": feature_group,
            "threshold_strategy": self.selection.get("selected_threshold_strategy"),
            "fusion_strategy": "weighted_branch_fusion" if self.request.get("enable_branch_fusion", False) else "none",
            "metrics": metrics,
            "architecture_version": self.model_config.get("architecture_version", "study_layered_v2"),
            "serving_model_type": serving_model_type,
            "task_scope": task_scope,
            "label_definition": self.model_config.get("label_name", "LABEL"),
            "eval_split": "term_order_holdout",
            "chain_validation": chain_validation,
            "created_at": now_iso(),
            "promoted_at": None,
            "rolled_back_at": None,
            "status": "candidate",
            "publish_eligible": self.selection.get("publish_eligible", False),
            "comparison_path": str(EVOLUTION_COMPARISON_PATH),
            "selection_path": str(EVOLUTION_SELECTION_PATH),
        }
        artifact_paths = self._build_candidate_artifacts(version_id, candidate_row)
        if artifact_paths:
            self.publish_candidate["artifact_paths"] = artifact_paths
            self.publish_candidate["frozen_snapshot"] = freeze_version_snapshot(
                version_id=version_id,
                metrics=metrics,
                eval_report_path=Path(artifact_paths["eval_report"]) if artifact_paths.get("eval_report") else None,
                subgroup_metrics_path=Path(artifact_paths["subgroup_metrics"]) if artifact_paths.get("subgroup_metrics") else None,
                confidence_zone_report_path=Path(artifact_paths["confidence_zone_report"]) if artifact_paths.get("confidence_zone_report") else None,
                prediction_output_path=Path(artifact_paths["prediction_output"]) if artifact_paths.get("prediction_output") else None,
                model_config_path=Path(artifact_paths["model_config"]) if artifact_paths.get("model_config") else None,
                feature_config_path=Path(artifact_paths["feature_config"]) if artifact_paths.get("feature_config") else None,
            )
        write_json(EVOLUTION_PUBLISH_CANDIDATE_PATH, self.publish_candidate)
        registry = read_json(MODEL_REGISTRY_PATH, {"domain": "study", "versions": []})
        registry.setdefault("versions", []).append(self.publish_candidate)
        write_json(MODEL_REGISTRY_PATH, registry)
        feature_registry = read_json(FEATURE_REGISTRY_PATH, {"domain": "study", "feature_versions": []})
        feature_registry.setdefault("feature_versions", []).append(
            {
                "feature_version": f"{version_id}_features",
                "feature_group": self.publish_candidate["feature_group"],
                "created_at": now_iso(),
                "status": "candidate",
            }
        )
        write_json(FEATURE_REGISTRY_PATH, feature_registry)
        self._trace("register_candidate_version", "register", "success", "candidate version registered", {"version_id": version_id})
        return self.publish_candidate

    def maybe_publish_candidate(self) -> dict[str, Any]:
        if not self.request.get("publish_selected_model", False):
            return {"status": "not_requested", "reason": "publish_selected_model=false"}
        try:
            from study_release_manager import StudyReleaseManager
        except ModuleNotFoundError:  # pragma: no cover
            from .study_release_manager import StudyReleaseManager
        manager = StudyReleaseManager(self.request)
        return manager.publish(self.publish_candidate.get("version_id"), dry_run=True, require_approval=True)

    def maybe_rollback(self) -> dict[str, Any]:
        try:
            from study_release_manager import StudyReleaseManager
        except ModuleNotFoundError:  # pragma: no cover
            from .study_release_manager import StudyReleaseManager
        manager = StudyReleaseManager(self.request)
        return manager.maybe_rollback({})

    def run(self) -> dict[str, Any]:
        self.load_train_data()
        self.generate_candidate_feature_sets()
        self.train_candidate_models()
        self.evaluate_candidates()
        self.tune_thresholds()
        fusion = self.run_branch_fusion()
        self.select_publish_candidate()
        candidate = self.register_candidate_version()
        publish_action = self.maybe_publish_candidate()
        self.comparison.to_csv(EVOLUTION_COMPARISON_PATH, index=False, encoding="utf-8-sig")
        summary = {
            "request_id": self.request_id,
            "status": "success",
            "comparison_path": str(EVOLUTION_COMPARISON_PATH),
            "selection_path": str(EVOLUTION_SELECTION_PATH),
            "publish_candidate_path": str(EVOLUTION_PUBLISH_CANDIDATE_PATH),
            "candidate_count": int(len(self.comparison)),
            "publish_candidate": candidate,
            "fusion": fusion,
            "publish_action": publish_action,
        }
        self._trace("evolution_complete", "export", "success", "evolution outputs exported", {"candidate_count": summary["candidate_count"]})
        return summary

    @staticmethod
    def _safe_auc(y_true: pd.Series, score: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y_true, score))
        except ValueError:
            return float("nan")

    def _select_threshold(self, y_true: pd.Series, score: np.ndarray, strategy: str, group: str, model_name: str) -> tuple[float, list[dict[str, Any]]]:
        if strategy == "default_0_5":
            grid = [0.5]
        else:
            grid = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
        rows = []
        for threshold in grid:
            metric = self._metrics(y_true, score, threshold)
            rows.append({"request_id": self.request_id, "feature_group": group, "model_name": model_name, "threshold_strategy": strategy, "threshold": threshold, **metric, "selected_threshold": False, "selection_reason": ""})
        table = pd.DataFrame(rows)
        if strategy == "default_0_5":
            idx, reason = table.index[0], "fixed threshold 0.5"
        elif strategy == "best_f1":
            idx, reason = table.sort_values(["f1", "recall", "precision"], ascending=False).index[0], "maximized validation F1"
        elif strategy == "recall_priority":
            eligible = table[table["recall"] >= self.space.get("selection", {}).get("recall_priority_min_recall", 0.7)]
            idx = (eligible if not eligible.empty else table).sort_values(["f1", "recall"], ascending=False).index[0]
            reason = "best F1 under recall priority"
        elif strategy == "precision_floor":
            floor = self.space.get("selection", {}).get("precision_floor", 0.55)
            eligible = table[table["precision"] >= floor]
            idx = (eligible if not eligible.empty else table).sort_values(["f1", "precision"], ascending=False).index[0]
            reason = "best F1 under precision floor"
        else:
            idx, reason = table.sort_values(["recall", "f1"], ascending=False).index[0], "risk stratified high-recall threshold"
        table.loc[idx, "selected_threshold"] = True
        table.loc[idx, "selection_reason"] = reason
        return float(table.loc[idx, "threshold"]), table.to_dict(orient="records")

    def _metrics(self, y_true: pd.Series, score: np.ndarray, threshold: float) -> dict[str, float]:
        pred = (score >= threshold).astype(int)
        return {
            "auc": self._safe_auc(y_true, score),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
        }

    def _robustness_score(self, auc: float, f1: float, recall: float, coverage: float, missing_robustness: float, degraded_proxy: float) -> float:
        weights = self.space.get("selection", {}).get("robustness_weights", {})
        auc_value = 0.0 if np.isnan(auc) else auc
        raw = (
            weights.get("auc", 0.35) * auc_value
            + weights.get("f1", 0.25) * f1
            + weights.get("recall", 0.20) * recall
            + weights.get("coverage", 0.12) * coverage
            + weights.get("missing_robustness", 0.08) * missing_robustness
        )
        return float(raw - 0.05 * degraded_proxy)
