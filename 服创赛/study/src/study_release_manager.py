from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from study_agent import DM_DIR, ROOT, json_default, normalize_workspace_paths, write_json
except ModuleNotFoundError:  # pragma: no cover
    from .study_agent import DM_DIR, ROOT, json_default, normalize_workspace_paths, write_json


REGISTRY_DIR = ROOT / "data" / "registry"
MODEL_REGISTRY_PATH = REGISTRY_DIR / "study_model_registry.json"
FEATURE_REGISTRY_PATH = REGISTRY_DIR / "study_feature_registry.json"
RELEASE_HISTORY_PATH = REGISTRY_DIR / "study_release_history.json"
CURRENT_SERVING_PATH = DM_DIR / "study_current_serving_version.json"
RELEASE_ACTION_LOG_PATH = DM_DIR / "study_release_action_log.jsonl"
RELEASE_TRACE_PATH = ROOT / "logs" / "study_release_trace.jsonl"
PUBLISH_CANDIDATE_PATH = DM_DIR / "study_evolution_publish_candidate.json"
RELEASE_POLICY_PATH = ROOT / "conf" / "study_release_policy.yaml"
FORMAL_CONFIG_PATH = ROOT / "data" / "deliverables" / "study" / "model" / "study_model_config.json"
LATEST_RESULT_PATH = DM_DIR / "study_infer_result_record.json"
VERSION_SNAPSHOT_DIR = REGISTRY_DIR / "study"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return normalize_workspace_paths(json.loads(path.read_text(encoding="utf-8")))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_workspace_paths(payload)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_json_file(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return normalize_workspace_paths(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def normalize_json_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    normalized = normalize_workspace_paths(raw)
    if normalized != raw:
        write_json_file(path, normalized)


def _relative_to_root(path_str: str) -> str:
    if not isinstance(path_str, str) or not path_str:
        return ""
    try:
        path = Path(path_str)
        if path.is_absolute():
            return str(path.relative_to(ROOT))
    except Exception:
        return ""
    return path_str.replace("/", "\\")


def _relativeize_path_map(payload: dict[str, Any] | None) -> dict[str, str]:
    payload = payload or {}
    result: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            rel = _relative_to_root(value)
            if rel:
                result[key] = rel
    return result


def _extract_feature_columns(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        columns = payload.get("feature_columns")
        if isinstance(columns, list):
            return [str(col) for col in columns]
    if isinstance(payload, list):
        return [str(col) for col in payload]
    return []


def _feature_contract_hash(feature_columns: list[str]) -> str:
    return hashlib.sha1("\n".join(feature_columns).encode("utf-8")).hexdigest() if feature_columns else ""


def build_snapshot_contract_payload(
    *,
    version_id: str,
    version_record: dict[str, Any],
    formal_config: dict[str, Any] | None = None,
    model_config_payload: dict[str, Any] | None = None,
    eval_report_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    formal_config = formal_config or {}
    model_config_payload = model_config_payload or {}
    eval_report_payload = eval_report_payload or {}
    contract = derive_version_contract(version_record, formal_config)
    overall_metrics = eval_report_payload.get("overall_metrics", {})
    return {
        "schema_version": "harness_snapshot_contract_v1",
        "domain": version_record.get("domain", "study"),
        "version_id": version_id,
        "created_at": version_record.get("created_at") or formal_config.get("train_time"),
        "model_name": version_record.get("model_name") or model_config_payload.get("primary_model"),
        "feature_group": version_record.get("feature_group"),
        "selection_policy": version_record.get("selection_policy"),
        "default_publish_decision": version_record.get("default_publish_decision"),
        "contract_context": contract,
        "metric_context": {
            "eval_scope": "publish_gate_metric" if overall_metrics else "",
            "sample_count": overall_metrics.get("rows"),
            "threshold_strategy": version_record.get("threshold_strategy") or model_config_payload.get("selected_threshold_strategy"),
        },
    }


def build_snapshot_feature_payload(
    *,
    version_id: str,
    version_record: dict[str, Any],
    formal_config: dict[str, Any] | None = None,
    model_config_payload: dict[str, Any] | None = None,
    feature_config_payload: Any = None,
) -> dict[str, Any]:
    formal_config = formal_config or {}
    model_config_payload = model_config_payload or {}
    feature_columns = _extract_feature_columns(feature_config_payload) or _extract_feature_columns(model_config_payload)
    feature_layer_summary = model_config_payload.get("feature_layer_summary", {}) if isinstance(model_config_payload, dict) else {}
    return {
        "schema_version": "harness_feature_contract_v1",
        "domain": version_record.get("domain", "study"),
        "version_id": version_id,
        "feature_version": model_config_payload.get("feature_version") or formal_config.get("feature_version") or f"{version_id}_features",
        "feature_group": version_record.get("feature_group"),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_contract_hash": _feature_contract_hash(feature_columns),
        "feature_layer_summary": feature_layer_summary,
    }


def build_snapshot_domain_audit_payload(
    *,
    version_id: str,
    version_record: dict[str, Any],
    model_config_payload: dict[str, Any] | None = None,
    feature_payload: dict[str, Any] | None = None,
    artifact_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_config_payload = model_config_payload or {}
    feature_payload = feature_payload or {}
    artifact_paths = artifact_paths or {}
    return {
        "schema_version": "harness_domain_audit_v1",
        "domain": version_record.get("domain", "study"),
        "version_id": version_id,
        "chain_validation": version_record.get("chain_validation", {}),
        "feature_layer_summary": model_config_payload.get("feature_layer_summary", {}),
        "behavior_layer_status": model_config_payload.get("behavior_layer_status", {}),
        "feature_contract_hash": feature_payload.get("feature_contract_hash", ""),
        "artifact_paths": artifact_paths,
    }


def prediction_summary_from_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import pandas as pd

        frame = pd.read_csv(path)
        summary = {"rows": int(len(frame))}
        for col in ["BASE_SCORE", "FINAL_SCORE", "DOMAIN_SCORE"]:
            if col in frame.columns:
                series = pd.to_numeric(frame[col], errors="coerce")
                summary[col] = {
                    "mean": float(series.mean()) if len(series) else None,
                    "min": float(series.min()) if len(series) else None,
                    "max": float(series.max()) if len(series) else None,
                }
        for col in ["CONFIDENCE_ZONE", "ROUTING_REASON", "STUDY_DATA_MODE"]:
            if col in frame.columns:
                summary[f"{col.lower()}_counts"] = frame[col].astype(str).value_counts(dropna=False).to_dict()
        return summary
    except Exception:
        return {}


def freeze_version_snapshot(
    *,
    version_id: str,
    metrics: dict[str, Any] | None,
    eval_report_path: Path | None,
    subgroup_metrics_path: Path | None,
    confidence_zone_report_path: Path | None,
    prediction_output_path: Path | None,
    model_config_path: Path | None,
    feature_config_path: Path | None,
    version_record: dict[str, Any] | None = None,
    formal_config: dict[str, Any] | None = None,
    artifact_paths: dict[str, Any] | None = None,
) -> dict[str, str]:
    snapshot_dir = (VERSION_SNAPSHOT_DIR / version_id).resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    version_record = version_record or {}
    formal_config = formal_config or {}
    artifact_paths = artifact_paths or {}

    metrics_path = snapshot_dir / "metrics.json"
    eval_snapshot = snapshot_dir / "eval_report.json"
    subgroup_snapshot = snapshot_dir / "subgroup_metrics.csv"
    confidence_snapshot = snapshot_dir / "confidence_zone_report.csv"
    prediction_summary_path = snapshot_dir / "prediction_summary.json"
    model_config_snapshot = snapshot_dir / "model_config.json"
    feature_config_snapshot = snapshot_dir / "feature_config.json"
    contract_context_snapshot = snapshot_dir / "contract_context.json"
    domain_audit_snapshot = snapshot_dir / "domain_audit.json"

    write_json_file(metrics_path, metrics or {})

    if not copy_if_exists(eval_report_path, eval_snapshot):
        write_json_file(eval_snapshot, {})
    if subgroup_metrics_path is not None and subgroup_metrics_path.exists():
        copy_if_exists(subgroup_metrics_path, subgroup_snapshot)
    elif not subgroup_snapshot.exists():
        subgroup_snapshot.write_text("", encoding="utf-8")
    if confidence_zone_report_path is not None and confidence_zone_report_path.exists():
        copy_if_exists(confidence_zone_report_path, confidence_snapshot)
    elif not confidence_snapshot.exists():
        confidence_snapshot.write_text("", encoding="utf-8")
    write_json_file(prediction_summary_path, prediction_summary_from_csv(prediction_output_path) if prediction_output_path else {})
    model_config_payload = read_json_file(model_config_path)
    feature_config_payload = read_json_file(feature_config_path)
    eval_report_payload = read_json_file(eval_snapshot) or {}

    contract_payload = build_snapshot_contract_payload(
        version_id=version_id,
        version_record=version_record,
        formal_config=formal_config,
        model_config_payload=model_config_payload if isinstance(model_config_payload, dict) else {},
        eval_report_payload=eval_report_payload if isinstance(eval_report_payload, dict) else {},
    )
    feature_payload = build_snapshot_feature_payload(
        version_id=version_id,
        version_record=version_record,
        formal_config=formal_config,
        model_config_payload=model_config_payload if isinstance(model_config_payload, dict) else {},
        feature_config_payload=feature_config_payload,
    )
    domain_audit_payload = build_snapshot_domain_audit_payload(
        version_id=version_id,
        version_record=version_record,
        model_config_payload=model_config_payload if isinstance(model_config_payload, dict) else {},
        feature_payload=feature_payload,
        artifact_paths=artifact_paths,
    )

    write_json_file(model_config_snapshot, model_config_payload if isinstance(model_config_payload, dict) and model_config_payload else contract_payload)
    write_json_file(feature_config_snapshot, feature_config_payload if isinstance(feature_config_payload, dict) and feature_config_payload else feature_payload)
    write_json_file(contract_context_snapshot, contract_payload)
    write_json_file(domain_audit_snapshot, domain_audit_payload)

    return {
        "snapshot_dir": str(snapshot_dir),
        "metrics": str(metrics_path),
        "eval_report": str(eval_snapshot),
        "subgroup_metrics": str(subgroup_snapshot),
        "confidence_zone_report": str(confidence_snapshot),
        "prediction_summary": str(prediction_summary_path),
        "model_config": str(model_config_snapshot),
        "feature_config": str(feature_config_snapshot),
        "contract_context": str(contract_context_snapshot),
        "domain_audit": str(domain_audit_snapshot),
    }


def validate_frozen_metrics_consistency(version: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    frozen = version.get("frozen_snapshot", {})
    metrics_path = Path(frozen["metrics"]) if frozen.get("metrics") else None
    if metrics_path is None or not metrics_path.exists():
        return False, {"reason": "frozen metrics missing"}
    frozen_metrics = read_json(metrics_path, {})
    registry_metrics = version.get("metrics", {}) or {}
    consistent = frozen_metrics == registry_metrics
    return consistent, {
        "reason": "ok" if consistent else "frozen metrics mismatch with registry",
        "frozen_metrics": frozen_metrics,
        "registry_metrics": registry_metrics,
    }


def snapshot_is_complete(snapshot: dict[str, Any] | None) -> bool:
    required_keys = [
        "metrics",
        "eval_report",
        "subgroup_metrics",
        "confidence_zone_report",
        "prediction_summary",
        "model_config",
        "feature_config",
        "contract_context",
        "domain_audit",
    ]
    if not snapshot:
        return False
    for key in required_keys:
        path_str = snapshot.get(key)
        if not path_str or not Path(path_str).exists():
            return False
    return True


def derive_version_contract(version: dict[str, Any], formal_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derive generic contract fields from persisted version evidence.

    This keeps same-caliber checks driven by stored version metadata instead
    of hard-coded assumptions about the active lane.
    """
    formal_config = formal_config or {}
    chain_validation = version.get("chain_validation", {}) or {}
    data_mode_validation = chain_validation.get("data_mode_validation", {}) or {}
    feature_group = version.get("feature_group")
    core_plus_ratio = float(data_mode_validation.get("core_plus_behavior_ratio", 0) or 0)

    if core_plus_ratio > 0.10 and feature_group in {"core_plus_behavior", "topk_selected", "full_enhanced"}:
        serving_model_type = "study_enhanced_model"
        task_scope = "study_layered_enhanced_serving"
    else:
        serving_model_type = "study_core_model"
        task_scope = "study_layered_core_serving"

    return {
        "architecture_version": version.get("architecture_version") or formal_config.get("architecture_version", "study_layered_v2"),
        "serving_model_type": version.get("serving_model_type") or serving_model_type,
        "task_scope": task_scope,
        "label_definition": version.get("label_definition") or formal_config.get("label_name", "LABEL"),
        "eval_split": version.get("eval_split") or "term_order_holdout",
    }


def read_policy() -> dict[str, Any]:
    if not RELEASE_POLICY_PATH.exists():
        return {}
    with RELEASE_POLICY_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def bootstrap_registries() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    DM_DIR.mkdir(parents=True, exist_ok=True)
    normalize_json_file(FORMAL_CONFIG_PATH)
    normalize_json_file(ROOT / "data" / "dm" / "study_model_config.json")
    normalize_json_file(MODEL_REGISTRY_PATH)
    normalize_json_file(CURRENT_SERVING_PATH)
    normalize_json_file(PUBLISH_CANDIDATE_PATH)
    policy = read_policy()
    serving_policy = policy.get("serving", {})
    formal_config = read_json(FORMAL_CONFIG_PATH, {})
    selection_policy = "model_selection_v1"
    publish_default_decision = "keep_baseline_unless_global_auc_stable_and_local_gain_confirmed"
    if not MODEL_REGISTRY_PATH.exists():
        version_id = serving_policy.get("bootstrap_version_id", formal_config.get("model_version", "study_v1"))
        # Fix #3: Use train_time for both created_at and promoted_at to avoid timestamp confusion
        train_time = formal_config.get("train_time", now_iso())
        frozen_snapshot = freeze_version_snapshot(
            version_id=version_id,
            metrics=formal_config.get("metrics", {}).get("core_model", {}).get("valid", formal_config.get("metrics", {}).get("valid", {})),
            eval_report_path=Path(formal_config.get("eval_report_path")) if formal_config.get("eval_report_path") else None,
            subgroup_metrics_path=Path(formal_config.get("subgroup_metrics_path")) if formal_config.get("subgroup_metrics_path") else None,
            confidence_zone_report_path=Path(formal_config.get("confidence_zone_report_path")) if formal_config.get("confidence_zone_report_path") else None,
            prediction_output_path=ROOT / "data" / "deliverables" / "study" / "data" / "study_prediction_output.csv",
            model_config_path=FORMAL_CONFIG_PATH,
            feature_config_path=DM_DIR / "study_selected_features.csv",
            version_record={
                "domain": "study",
                "version_id": version_id,
                "feature_group": "formal_study_v1",
                "selection_policy": selection_policy,
                "default_publish_decision": publish_default_decision,
            },
            formal_config=formal_config,
            artifact_paths={
                "model_config": str(FORMAL_CONFIG_PATH),
                "feature_config": str(DM_DIR / "study_selected_features.csv"),
            },
        )
        bootstrap_contract = derive_version_contract(
            {
                "feature_group": "formal_study_v1",
                "chain_validation": {},
                "architecture_version": formal_config.get("architecture_version", "study_layered_v2"),
            },
            formal_config,
        )
        registry = {
            "domain": "study",
            "active_baseline_version_id": version_id,
            "anchor_baseline_version_id": version_id,
            "versions": [
                {
                    "version_id": version_id,
                    "model_name": formal_config.get("primary_model", serving_policy.get("bootstrap_model_name")),
                    "feature_group": "formal_study_v1",
                    "threshold_strategy": "formal_config",
                    "fusion_strategy": "none",
                    "metrics": formal_config.get("metrics", {}).get("valid", {}),
                    "architecture_version": bootstrap_contract["architecture_version"],
                    "serving_model_type": bootstrap_contract["serving_model_type"],
                    "task_scope": bootstrap_contract["task_scope"],
                    "label_definition": bootstrap_contract["label_definition"],
                    "eval_split": bootstrap_contract["eval_split"],
                    "selection_policy": selection_policy,
                    "default_publish_decision": publish_default_decision,
                    "created_at": train_time,
                    "promoted_at": train_time,  # Use train_time, not now_iso()
                    "rolled_back_at": None,
                    "status": "published",
                    "artifact_paths": {
                        "model_config": str(FORMAL_CONFIG_PATH),
                        "model_file": str(ROOT / "data" / "deliverables" / "study" / "model" / "study_model.pkl"),
                    },
                    "baseline_role": "anchor_active",
                    "frozen_snapshot": frozen_snapshot,
                }
            ],
        }
        write_json(MODEL_REGISTRY_PATH, registry)
    else:
        registry = read_json(MODEL_REGISTRY_PATH, {"domain": "study", "versions": []})
        changed = False
        if "active_baseline_version_id" not in registry:
            published = [v for v in registry.get("versions", []) if v.get("status") == "published"]
            if published:
                registry["active_baseline_version_id"] = published[-1].get("version_id")
                changed = True
        if "anchor_baseline_version_id" not in registry:
            published = [v for v in registry.get("versions", []) if v.get("status") in {"published", "archived"}]
            if published:
                registry["anchor_baseline_version_id"] = published[0].get("version_id")
                changed = True
        for version in registry.get("versions", []):
            derived_contract = derive_version_contract(version, formal_config)
            version_id = version.get("version_id")
            if version_id == registry.get("active_baseline_version_id") == registry.get("anchor_baseline_version_id"):
                expected_role = "anchor_active"
            elif version_id == registry.get("active_baseline_version_id"):
                expected_role = "active_baseline"
            elif version_id == registry.get("anchor_baseline_version_id"):
                expected_role = "anchor_baseline"
            else:
                expected_role = version.get("baseline_role")
            if version.get("baseline_role") != expected_role:
                version["baseline_role"] = expected_role
                changed = True
            for field in ["architecture_version", "serving_model_type", "task_scope", "label_definition", "eval_split"]:
                if version.get(field) != derived_contract[field]:
                    version[field] = derived_contract[field]
                    changed = True
            if version.get("status") == "published":
                if version.get("selection_policy") != selection_policy:
                    version["selection_policy"] = selection_policy
                    changed = True
                if version.get("default_publish_decision") != publish_default_decision:
                    version["default_publish_decision"] = publish_default_decision
                    changed = True
            artifact_paths_relative = _relativeize_path_map(version.get("artifact_paths", {}))
            if artifact_paths_relative and version.get("artifact_paths_relative") != artifact_paths_relative:
                version["artifact_paths_relative"] = artifact_paths_relative
                changed = True
            frozen_snapshot_relative = _relativeize_path_map(version.get("frozen_snapshot", {}))
            if frozen_snapshot_relative and version.get("frozen_snapshot_relative") != frozen_snapshot_relative:
                version["frozen_snapshot_relative"] = frozen_snapshot_relative
                changed = True
            comparison_path_relative = _relative_to_root(version.get("comparison_path", ""))
            if comparison_path_relative and version.get("comparison_path_relative") != comparison_path_relative:
                version["comparison_path_relative"] = comparison_path_relative
                changed = True
            selection_path_relative = _relative_to_root(version.get("selection_path", ""))
            if selection_path_relative and version.get("selection_path_relative") != selection_path_relative:
                version["selection_path_relative"] = selection_path_relative
                changed = True
            if not version.get("frozen_snapshot") or not snapshot_is_complete(version.get("frozen_snapshot")):
                version_id = version.get("version_id", "unknown_version")
                artifact_paths = version.get("artifact_paths", {})
                frozen_snapshot = freeze_version_snapshot(
                    version_id=version_id,
                    metrics=version.get("metrics", {}),
                    eval_report_path=Path(artifact_paths["eval_report"]) if artifact_paths.get("eval_report") else None,
                    subgroup_metrics_path=Path(artifact_paths["subgroup_metrics"]) if artifact_paths.get("subgroup_metrics") else None,
                    confidence_zone_report_path=Path(artifact_paths["confidence_zone_report"]) if artifact_paths.get("confidence_zone_report") else None,
                    prediction_output_path=Path(artifact_paths["prediction_output"]) if artifact_paths.get("prediction_output") else None,
                    model_config_path=Path(artifact_paths["model_config"]) if artifact_paths.get("model_config") else None,
                    feature_config_path=Path(artifact_paths["feature_config"]) if artifact_paths.get("feature_config") else None,
                    version_record=version,
                    formal_config=formal_config,
                    artifact_paths=artifact_paths,
                )
                version["frozen_snapshot"] = frozen_snapshot
                changed = True
        if changed:
            write_json(MODEL_REGISTRY_PATH, registry)
    if not FEATURE_REGISTRY_PATH.exists():
        write_json(
            FEATURE_REGISTRY_PATH,
            {
                "domain": "study",
                "feature_versions": [
                    {
                        "feature_version": formal_config.get("feature_version", serving_policy.get("bootstrap_feature_version")),
                        "source": "formal_model_config",
                        "created_at": formal_config.get("train_time"),
                        "status": "published",
                    }
                ],
            },
        )
    if not RELEASE_HISTORY_PATH.exists():
        write_json(RELEASE_HISTORY_PATH, {"domain": "study", "actions": []})
    if not CURRENT_SERVING_PATH.exists():
        published = [v for v in read_json(MODEL_REGISTRY_PATH, {"versions": []}).get("versions", []) if v.get("status") == "published"]
        current = published[-1] if published else {}
        write_json(
            CURRENT_SERVING_PATH,
            {
                "domain": "study",
                "current_version_id": current.get("version_id"),
                "status": "published",
                "updated_at": now_iso(),
                "selection_policy": selection_policy,
                "default_publish_decision": publish_default_decision,
                "serving_model_type": current.get("serving_model_type"),
                "architecture_version": current.get("architecture_version"),
                "task_scope": current.get("task_scope"),
                "active_baseline_version_id": read_json(MODEL_REGISTRY_PATH, {}).get("active_baseline_version_id"),
                "anchor_baseline_version_id": read_json(MODEL_REGISTRY_PATH, {}).get("anchor_baseline_version_id"),
                "serving_version": current,
            },
        )
    else:
        current = read_json(CURRENT_SERVING_PATH, {})
        registry = read_json(MODEL_REGISTRY_PATH, {})
        versions = {v.get("version_id"): v for v in registry.get("versions", [])}
        active_version = versions.get(registry.get("active_baseline_version_id"), {})
        current["selection_policy"] = selection_policy
        current["default_publish_decision"] = publish_default_decision
        current["serving_model_type"] = active_version.get("serving_model_type")
        current["architecture_version"] = active_version.get("architecture_version")
        current["task_scope"] = active_version.get("task_scope")
        current["active_baseline_version_id"] = registry.get("active_baseline_version_id")
        current["anchor_baseline_version_id"] = registry.get("anchor_baseline_version_id")
        current["serving_version"] = active_version
        current["serving_version_relative"] = {
            "artifact_paths_relative": _relativeize_path_map(active_version.get("artifact_paths", {})),
            "frozen_snapshot_relative": _relativeize_path_map(active_version.get("frozen_snapshot", {})),
            "comparison_path_relative": _relative_to_root(active_version.get("comparison_path", "")),
            "selection_path_relative": _relative_to_root(active_version.get("selection_path", "")),
        }
        current["updated_at"] = now_iso()
        write_json(CURRENT_SERVING_PATH, current)


class StudyReleaseManager:
    def __init__(self, request: dict[str, Any] | None = None):
        self.request = request or {}
        bootstrap_registries()
        self.policy = read_policy()

    def _log(self, stage: str, decision: str, status: str, reason: str, key_metrics: dict[str, Any] | None = None) -> None:
        record = {
            "timestamp": now_iso(),
            "request_id": self.request.get("request_id", "release_request"),
            "stage": stage,
            "decision": decision,
            "status": status,
            "reason": reason,
            "key_metrics": key_metrics or {},
        }
        append_jsonl(RELEASE_ACTION_LOG_PATH, record)
        append_jsonl(RELEASE_TRACE_PATH, record)
        history = read_json(RELEASE_HISTORY_PATH, {"domain": "study", "actions": []})
        history.setdefault("actions", []).append(record)
        write_json(RELEASE_HISTORY_PATH, history)

    def registry_summary(self) -> dict[str, Any]:
        bootstrap_registries()
        registry = read_json(MODEL_REGISTRY_PATH, {"versions": []})
        serving = read_json(CURRENT_SERVING_PATH, {})
        return {
            "model_registry_path": str(MODEL_REGISTRY_PATH),
            "feature_registry_path": str(FEATURE_REGISTRY_PATH),
            "release_history_path": str(RELEASE_HISTORY_PATH),
            "version_count": len(registry.get("versions", [])),
            "current_serving": serving,
            "active_baseline_version_id": registry.get("active_baseline_version_id"),
            "anchor_baseline_version_id": registry.get("anchor_baseline_version_id"),
            "versions": registry.get("versions", []),
        }

    def current_serving(self) -> dict[str, Any]:
        bootstrap_registries()
        return read_json(CURRENT_SERVING_PATH, {})

    def _candidate_by_id(self, version_id: str | None) -> dict[str, Any]:
        candidate = read_json(PUBLISH_CANDIDATE_PATH, {})
        if version_id in {None, "", "latest"}:
            return candidate
        if candidate.get("version_id") == version_id:
            return candidate
        registry = read_json(MODEL_REGISTRY_PATH, {"versions": []})
        for version in registry.get("versions", []):
            if version.get("version_id") == version_id:
                return version
        return {}

    def _passes_gates(self, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        gates = self.policy.get("promotion_gates", {})
        metrics = candidate.get("metrics", {})
        reasons = []
        checks = {
            "min_valid_auc": metrics.get("auc", 0) >= gates.get("min_valid_auc", 0),
            "min_valid_recall": metrics.get("recall", 0) >= gates.get("min_valid_recall", 0),
            "min_valid_f1": metrics.get("f1", 0) >= gates.get("min_valid_f1", 0),
            "max_degraded_rate": metrics.get("degraded_proxy", 0) <= gates.get("max_degraded_rate", 1),
            "max_low_coverage_rate": metrics.get("low_coverage_rate", 0) <= gates.get("max_low_coverage_rate", 1),
        }
        for name, ok in checks.items():
            if not ok:
                reasons.append(f"{name} not satisfied")
        return not reasons, reasons

    def _version_by_id(self, version_id: str | None) -> dict[str, Any]:
        registry = read_json(MODEL_REGISTRY_PATH, {"versions": []})
        for version in registry.get("versions", []):
            if version.get("version_id") == version_id:
                return version
        return {}

    def _baseline_context(self, baseline_role: str = "active") -> dict[str, Any]:
        serving = self.current_serving().get("serving_version", {})
        registry = read_json(MODEL_REGISTRY_PATH, {})
        baseline_version_id = registry.get("anchor_baseline_version_id") if baseline_role == "anchor" else registry.get("active_baseline_version_id")
        baseline_version = self._version_by_id(baseline_version_id) if baseline_version_id else serving
        frozen = baseline_version.get("frozen_snapshot", {})
        baseline_metrics = read_json(Path(frozen["metrics"]), {}) if frozen.get("metrics") else {}
        baseline_eval_report = read_json(Path(frozen["eval_report"]), {}) if frozen.get("eval_report") else {}
        consistency_ok, consistency_detail = validate_frozen_metrics_consistency(baseline_version) if baseline_version else (False, {"reason": "baseline missing"})
        return {
            "serving_version": baseline_version,
            "baseline_metrics": baseline_metrics,
            "baseline_eval_report": baseline_eval_report,
            "frozen_snapshot": frozen,
            "baseline_role": baseline_role,
            "consistency_ok": consistency_ok,
            "consistency_detail": consistency_detail,
        }

    def _candidate_context(self, candidate: dict[str, Any]) -> dict[str, Any]:
        metrics = candidate.get("metrics", {})
        artifact_paths = candidate.get("artifact_paths", {})
        eval_report = read_json(Path(artifact_paths["eval_report"]), {}) if artifact_paths.get("eval_report") else {}
        return {
            "candidate": candidate,
            "candidate_metrics": metrics,
            "architecture_version": candidate.get("architecture_version"),
            "task_scope": candidate.get("task_scope", "unknown"),
            "chain_validation": candidate.get("chain_validation", {}),
            "candidate_eval_report": eval_report,
        }

    def _same_caliber_check(self, baseline: dict[str, Any], candidate_ctx: dict[str, Any]) -> tuple[bool, list[str]]:
        cfg = self.policy.get("model_selection_v1", {})
        reasons: list[str] = []
        baseline_version = baseline.get("serving_version", {})
        candidate = candidate_ctx.get("candidate", {})

        if not baseline.get("consistency_ok", False):
            reasons.append("baseline frozen metrics are inconsistent with registry")

        if cfg.get("require_same_layered_mode", True):
            if baseline_version.get("architecture_version") != candidate_ctx.get("architecture_version"):
                reasons.append("architecture_version not comparable to current layered serving model")
        if cfg.get("require_same_task_scope", True):
            baseline_scope = baseline_version.get("task_scope", "unknown")
            candidate_scope = candidate_ctx.get("task_scope", "unknown")
            if candidate_scope != baseline_scope:
                reasons.append("candidate task scope is not the same as current serving scope")
        if cfg.get("require_same_label_definition", True):
            if candidate.get("label_definition") not in {None, baseline_version.get("label_definition"), "LABEL"}:
                reasons.append("label definition mismatch")
        if cfg.get("require_same_eval_split", True):
            if candidate.get("eval_split") not in {None, "term_order_holdout"}:
                reasons.append("evaluation split mismatch")
        return not reasons, reasons

    def _chain_gate_check(self, baseline: dict[str, Any], candidate_ctx: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
        cfg = self.policy.get("model_selection_v1", {})
        candidate = candidate_ctx.get("candidate", {})
        chain = candidate_ctx.get("chain_validation", {})
        # Old-style evolution candidates do not carry the required chain evidence.
        checks = {
            "train_success": bool(candidate.get("status") in {"candidate", "published", "success"}) if cfg.get("require_train_success", True) else True,
            "infer_success": bool(chain.get("infer_success", False)) if cfg.get("require_infer_success", True) else True,
            "non_fallback_primary": bool(chain.get("fallback_used") is False) if cfg.get("require_non_fallback_primary", True) else True,
            "output_contract_complete": bool(chain.get("output_contract_complete", False)) if cfg.get("require_output_contract_complete", True) else True,
            "explanation_available": bool(chain.get("explanation_available", False)) if cfg.get("require_explanation_available", True) else True,
            "publish_dry_run": bool(chain.get("publish_dry_run", False)) if cfg.get("require_publish_dry_run", True) else True,
            "rollback_dry_run": bool(chain.get("rollback_dry_run", False)) if cfg.get("require_rollback_dry_run", True) else True,
        }
        reasons = [f"{name} not satisfied" for name, ok in checks.items() if not ok]
        baseline_chain = {
            "infer_success": True,
            "fallback_used": False,
            "output_contract_complete": True,
            "explanation_available": True,
            "study_data_mode": "frozen_snapshot",
        }
        return not reasons, reasons, {"candidate_chain": checks, "baseline_chain": baseline_chain}

    def _composite_score(self, metrics: dict[str, Any], degraded_ratio: float, deployability: float) -> float:
        weights = self.policy.get("model_selection_v1", {}).get("scoring_weights", {})
        auc = float(metrics.get("auc", 0) or 0)
        recall = float(metrics.get("recall", 0) or 0)
        f1 = float(metrics.get("f1", 0) or 0)
        stability = max(0.0, 1.0 - degraded_ratio)
        return (
            float(weights.get("auc", 0.35)) * auc
            + float(weights.get("recall", 0.30)) * recall
            + float(weights.get("f1", 0.20)) * f1
            + float(weights.get("data_stability", 0.10)) * stability
            + float(weights.get("deployability", 0.05)) * deployability
        )

    def _model_selection_v1_decision(self, candidate: dict[str, Any], eligible: bool, gate_reasons: list[str]) -> tuple[str, list[str], dict[str, Any]]:
        baseline = self._baseline_context("active")
        anchor_baseline = self._baseline_context("anchor")
        candidate_ctx = self._candidate_context(candidate)
        same_caliber_ok, same_caliber_reasons = self._same_caliber_check(baseline, candidate_ctx)
        chain_ok, chain_reasons, chain_snapshot = self._chain_gate_check(baseline, candidate_ctx)
        comparison = self._compare_with_baseline(candidate)
        comparison["selection_rule"] = "study_best_model_v1"
        comparison["same_caliber"] = {"ok": same_caliber_ok, "reasons": same_caliber_reasons}
        comparison["chain_validation"] = chain_snapshot
        comparison["anchor_baseline_version_id"] = anchor_baseline.get("serving_version", {}).get("version_id")
        comparison["anchor_baseline_metrics"] = anchor_baseline.get("baseline_metrics", {})
        comparison["anchor_delta"] = {
            "auc_delta": float(candidate.get("metrics", {}).get("auc", 0) - anchor_baseline.get("baseline_metrics", {}).get("auc", 0)),
            "f1_delta": float(candidate.get("metrics", {}).get("f1", 0) - anchor_baseline.get("baseline_metrics", {}).get("f1", 0)),
            "recall_delta": float(candidate.get("metrics", {}).get("recall", 0) - anchor_baseline.get("baseline_metrics", {}).get("recall", 0)),
        }

        reasons = list(gate_reasons)
        if not same_caliber_ok:
            reasons.extend(same_caliber_reasons)
        if not chain_ok:
            reasons.extend(chain_reasons)
        if not eligible:
            reasons.append("did not pass baseline metric floor")

        cfg = self.policy.get("model_selection_v1", {})
        candidate_metrics = candidate_ctx.get("candidate_metrics", {})
        baseline_metrics = baseline.get("baseline_metrics", {})
        candidate_degraded = float(candidate_metrics.get("degraded_proxy", candidate_metrics.get("degraded_ratio", 1)) or 1)
        baseline_eval = baseline.get("baseline_eval_report", {})
        baseline_degraded = float(
            baseline_eval.get("overall_metrics", {}).get(
                "degraded_ratio",
                baseline_metrics.get("degraded_proxy", baseline_metrics.get("degraded_ratio", 1)),
            )
            or 1
        )
        epsilon_auc_regression = float(cfg.get("epsilon_auc_regression", 0.003))

        floor_ok = (
            float(candidate_metrics.get("auc", 0) or 0) >= float(cfg.get("min_auc", 0.8))
            and float(candidate_metrics.get("recall", 0) or 0) >= float(cfg.get("min_recall", 0.6))
            and float(candidate_metrics.get("f1", 0) or 0) >= float(cfg.get("min_f1", 0.5))
            and candidate_degraded <= float(cfg.get("max_degraded_sparse_ratio", 0.10))
        )
        comparison["floor_ok"] = floor_ok
        comparison["candidate_degraded_ratio"] = candidate_degraded
        comparison["baseline_degraded_ratio"] = baseline_degraded
        comparison["epsilon_auc_regression"] = epsilon_auc_regression

        # ---- HARD GATE: same_caliber blocks all downstream promotion reasoning ----
        if not same_caliber_ok:
            if not reasons:
                reasons.append("candidate did not satisfy study best model v1 gates")
            comparison["local_gain_checks"] = {}
            comparison["local_gain_missing_reasons"] = ["same_caliber_failed"]
            comparison["decision_stage_reached"] = "same_caliber_gate"
            return "incomparable_candidate", reasons, comparison

        # Chain gate
        if not chain_ok or not eligible or not floor_ok:
            if not reasons:
                reasons.append("candidate did not satisfy study best model v1 gates")
            comparison["local_gain_checks"] = {}
            comparison["decision_stage_reached"] = "floor_gate"
            return "keep_baseline", reasons, comparison

        candidate_score = self._composite_score(candidate_metrics, candidate_degraded, 1.0 if chain_ok else 0.0)
        baseline_score = self._composite_score(baseline_metrics, baseline_degraded, 1.0)
        comparison["composite_scores"] = {"candidate": candidate_score, "baseline": baseline_score}

        deltas = comparison.get("metric_deltas", {})
        recall_better = float(deltas.get("recall_delta", 0) or 0) > 0
        f1_acceptable = float(candidate_metrics.get("f1", 0) or 0) >= float(cfg.get("min_f1", 0.5))
        auc_competitive = float(deltas.get("auc_delta", 0) or 0) >= -epsilon_auc_regression
        more_stable = candidate_degraded < baseline_degraded
        candidate_eval = candidate_ctx.get("candidate_eval_report", {})

        def _mode_auc(report: dict[str, Any], mode_name: str) -> float | None:
            for row in report.get("mode_metrics", []):
                if row.get("study_data_mode") == mode_name:
                    return row.get("auc")
            return None

        def _subtype_auc(report: dict[str, Any], subtype_name: str) -> float | None:
            for row in report.get("subtype_metrics", []):
                if row.get("label_subtype") == subtype_name:
                    return row.get("auc")
            return None

        single_fail_gain = (
            (float(_subtype_auc(candidate_eval, "single_fail")) - float(_subtype_auc(baseline_eval, "single_fail")))
            if _subtype_auc(candidate_eval, "single_fail") is not None and _subtype_auc(baseline_eval, "single_fail") is not None
            else None
        )
        core_plus_behavior_gain = (
            (float(_mode_auc(candidate_eval, "core_plus_behavior")) - float(_mode_auc(baseline_eval, "core_plus_behavior")))
            if _mode_auc(candidate_eval, "core_plus_behavior") is not None and _mode_auc(baseline_eval, "core_plus_behavior") is not None
            else None
        )
        comparison["priority_checks"] = {
            "recall_better": recall_better,
            "f1_acceptable": f1_acceptable,
            "auc_not_below_baseline_minus_epsilon": auc_competitive,
            "degraded_ratio_improved": more_stable,
            "single_fail_auc_gain": single_fail_gain,
            "core_plus_behavior_auc_gain": core_plus_behavior_gain,
        }
        local_gain_checks = {
            "single_fail_auc_gain": single_fail_gain is not None and single_fail_gain > 0,
            "core_plus_behavior_auc_gain": core_plus_behavior_gain is not None and core_plus_behavior_gain > 0,
            "degraded_ratio_improved": more_stable,
            "f1_improved": float(deltas.get("f1_delta", 0) or 0) > 0,
            "recall_improved": recall_better,
        }
        comparison["local_gain_checks"] = local_gain_checks

        # ---- TIGHTENED local-gain gate: require at least one real (true, non-null) signal ----
        any_local_gain_true = any(v for v in local_gain_checks.values() if v is True)

        if candidate_score > baseline_score and f1_acceptable and auc_competitive and any_local_gain_true:
            reasons.append("candidate passed global stability and local-gain publish rule")
            comparison["decision_stage_reached"] = "local_gain_gate"
            return "promotion_recommended", reasons, comparison

        # Local gain not proven — record which signals are missing
        local_gain_missing_reasons = []
        if not local_gain_checks.get("single_fail_auc_gain"):
            local_gain_missing_reasons.append("SINGLE_FAIL_GAIN_MISSING")
        if not local_gain_checks.get("core_plus_behavior_auc_gain"):
            local_gain_missing_reasons.append("CORE_PLUS_BEHAVIOR_GAIN_MISSING")
        if not local_gain_checks.get("degraded_ratio_improved"):
            local_gain_missing_reasons.append("DEGRADED_RATIO_NOT_IMPROVED")
        comparison["local_gain_missing_reasons"] = local_gain_missing_reasons
        comparison["decision_stage_reached"] = "local_gain_gate"

        reasons.append("serving baseline remains because local gain or global auc stability was not sufficient")
        return "keep_baseline", reasons, comparison

    def _compare_with_baseline(self, candidate: dict[str, Any]) -> dict[str, Any]:
        baseline = self._baseline_context("active")
        serving = baseline.get("serving_version", {})
        baseline_metrics = baseline.get("baseline_metrics", {})
        candidate_metrics = candidate.get("metrics", {})
        return {
            "baseline_version_id": serving.get("version_id"),
            "baseline_model_name": serving.get("model_name"),
            "baseline_role": baseline.get("baseline_role"),
            "baseline_frozen_snapshot": baseline.get("frozen_snapshot", {}),
            "baseline_consistency_ok": baseline.get("consistency_ok", False),
            "baseline_consistency_detail": baseline.get("consistency_detail", {}),
            "candidate_version_id": candidate.get("version_id"),
            "candidate_model_name": candidate.get("model_name"),
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "metric_deltas": {
                "auc_delta": float(candidate_metrics.get("auc", 0) - baseline_metrics.get("auc", 0)),
                "f1_delta": float(candidate_metrics.get("f1", 0) - baseline_metrics.get("f1", 0)),
                "recall_delta": float(candidate_metrics.get("recall", 0) - baseline_metrics.get("recall", 0)),
                "coverage_delta": float(candidate_metrics.get("coverage", 0) - baseline_metrics.get("coverage", 0)),
            },
        }

    def _policy_decision(self, candidate: dict[str, Any], eligible: bool, gate_reasons: list[str]) -> tuple[str, list[str], dict[str, Any]]:
        return self._model_selection_v1_decision(candidate, eligible, gate_reasons)

    def compare_caliber(self, candidate_version_id: str | None = None) -> dict[str, Any]:
        candidate = self._candidate_by_id(candidate_version_id)
        if not candidate:
            return {"status": "failed", "reason": "candidate not found"}
        baseline = self._baseline_context("active")
        candidate_ctx = self._candidate_context(candidate)
        same_caliber_ok, same_caliber_reasons = self._same_caliber_check(baseline, candidate_ctx)
        chain_ok, chain_reasons, chain_snapshot = self._chain_gate_check(baseline, candidate_ctx)
        comparison = self._compare_with_baseline(candidate)
        comparison["same_caliber"] = {"ok": same_caliber_ok, "reasons": same_caliber_reasons}
        comparison["chain_validation"] = chain_snapshot
        return {
            "status": "success",
            "candidate_version_id": candidate.get("version_id"),
            "baseline_version_id": comparison.get("baseline_version_id"),
            "same_caliber_ok": same_caliber_ok,
            "same_caliber_reasons": same_caliber_reasons,
            "chain_ok": chain_ok,
            "chain_reasons": chain_reasons,
            "comparison": comparison,
        }

    def publish(self, candidate_version_id: str | None = None, dry_run: bool | None = None, require_approval: bool | None = None) -> dict[str, Any]:
        candidate = self._candidate_by_id(candidate_version_id)
        if not candidate:
            result = {"status": "failed", "action": "publish", "reason": "candidate not found"}
            self._log("publish", "reject", "failed", result["reason"])
            return result

        # ---- Override candidate metrics with fresh eval results if available ----
        # The publish adapter may inject fresh_eval_metrics from the current
        # eval report so that baseline_comparison uses the live publish_gate_metric
        # evaluation, not stale frozen snapshot metrics from the evolution file.
        fresh_metrics = self.request.get("fresh_eval_metrics")
        if fresh_metrics:
            candidate = dict(candidate)  # shallow copy to avoid mutating the file cache
            candidate["metrics"] = {**candidate.get("metrics", {}), **fresh_metrics}

        release_cfg = self.policy.get("release", {})
        if dry_run is None:
            dry_run = bool(release_cfg.get("dry_run_default", True))
        if require_approval is None:
            require_approval = bool(release_cfg.get("require_human_approval_default", True))
        eligible, reasons = self._passes_gates(candidate)
        policy_decision, policy_reasons, comparison = self._policy_decision(candidate, eligible, reasons)
        print("eligible =", eligible)
        print("policy_decision =", policy_decision)
        print("reasons =", policy_reasons)
        print("require_approval =", require_approval, "dry_run =", dry_run)
        print("candidate_version_id =", candidate.get("version_id"))
        print("baseline_comparison =", comparison)

        # ---- Deterministic status determination ----
        # The status must reflect the actual release state, not just the policy label.
        decision_stage = comparison.get("decision_stage_reached", "unknown")

        if policy_decision == "incomparable_candidate":
            # same_caliber hard gate: candidate cannot be compared
            status = "rejected"
        elif policy_decision in {"promotion_recommended", "promotion_pending_approval"}:
            # Policy says promote; actual promotion depends on dry_run/require_approval
            if dry_run:
                status = "dry_run"
            elif require_approval:
                status = "pending_approval"
            else:
                status = "promoted"
        elif policy_decision == "keep_baseline":
            status = "dry_run" if dry_run else "rejected"
        else:
            # Fallback for any other decision
            status = "dry_run" if dry_run else "rejected"

        result = {
            "request_id": self.request.get("request_id"),
            "action": "publish",
            "candidate_version_id": candidate.get("version_id"),
            "dry_run": dry_run,
            "require_approval": require_approval,
            "publish_eligible": eligible,
            "policy_decision": policy_decision,
            "status": status,
            "reasons": policy_reasons,
            "baseline_comparison": comparison,
            "candidate": candidate,
            "decision_stage_reached": decision_stage,
            "created_at": now_iso(),
        }
        # ---- Actual promotion only when policy says promote AND no dry_run AND no approval required ----
        if eligible and policy_decision in {"promotion_recommended", "promotion_pending_approval"} and not dry_run and not require_approval:
            self._promote_candidate(candidate)
        self._log("publish", policy_decision, "success" if eligible else "warning", "; ".join(policy_reasons) or "publish gates evaluated", comparison)
        return result

    def _promote_candidate(self, candidate: dict[str, Any]) -> None:
        registry = read_json(MODEL_REGISTRY_PATH, {"domain": "study", "versions": []})
        for version in registry.get("versions", []):
            if version.get("status") == "published":
                version["status"] = "archived"
            if version.get("version_id") == registry.get("anchor_baseline_version_id"):
                version["baseline_role"] = "anchor_baseline"
            elif version.get("baseline_role") == "active_baseline":
                version["baseline_role"] = None
        promoted = dict(candidate)
        promoted["status"] = "published"
        promoted["promoted_at"] = now_iso()
        promoted.setdefault("rolled_back_at", None)
        promoted["baseline_role"] = "active_baseline" if registry.get("anchor_baseline_version_id") != promoted.get("version_id") else "anchor_active"
        registry.setdefault("versions", []).append(promoted)
        registry["active_baseline_version_id"] = promoted.get("version_id")
        write_json(MODEL_REGISTRY_PATH, registry)
        write_json(
            CURRENT_SERVING_PATH,
            {
                "domain": "study",
                "current_version_id": promoted.get("version_id"),
                "status": "published",
                "updated_at": now_iso(),
                "selection_policy": promoted.get("selection_policy", "model_selection_v1"),
                "default_publish_decision": promoted.get("default_publish_decision", "keep_baseline_unless_global_auc_stable_and_local_gain_confirmed"),
                "serving_model_type": promoted.get("serving_model_type", "study_core_model"),
                "architecture_version": promoted.get("architecture_version"),
                "active_baseline_version_id": registry.get("active_baseline_version_id"),
                "anchor_baseline_version_id": registry.get("anchor_baseline_version_id"),
                "serving_version": promoted,
            },
        )

    def rollback(self, target_version_id: str | None = None, dry_run: bool = True) -> dict[str, Any]:
        registry = read_json(MODEL_REGISTRY_PATH, {"versions": []})
        versions = registry.get("versions", [])
        stable = [v for v in versions if v.get("status") in {"published", "archived"}]
        target = None
        if target_version_id and target_version_id != "latest_stable":
            target = next((v for v in versions if v.get("version_id") == target_version_id), None)
        elif len(stable) >= 2:
            target = stable[-2]
        elif stable:
            target = stable[-1]
        if not target:
            result = {"status": "failed", "action": "rollback", "reason": "no stable target found", "dry_run": dry_run}
            self._log("rollback", "reject", "failed", result["reason"])
            return result
        result = {
            "request_id": self.request.get("request_id"),
            "action": "rollback",
            "target_version_id": target.get("version_id"),
            "dry_run": dry_run,
            "status": "dry_run" if dry_run else "rolled_back",
            "target": target,
            "created_at": now_iso(),
        }
        if not dry_run:
            for version in versions:
                if version.get("status") == "published":
                    version["status"] = "rolled_back"
                    version["rolled_back_at"] = now_iso()
                    if version.get("version_id") == registry.get("anchor_baseline_version_id"):
                        version["baseline_role"] = "anchor_baseline"
                if version.get("version_id") == target.get("version_id"):
                    version["status"] = "published"
                    version["promoted_at"] = now_iso()
                    version["baseline_role"] = "active_baseline" if registry.get("anchor_baseline_version_id") != version.get("version_id") else "anchor_active"
            registry["active_baseline_version_id"] = target.get("version_id")
            write_json(MODEL_REGISTRY_PATH, registry)
            write_json(
                CURRENT_SERVING_PATH,
                {
                    "domain": "study",
                    "current_version_id": target.get("version_id"),
                    "status": "published",
                    "updated_at": now_iso(),
                    "active_baseline_version_id": registry.get("active_baseline_version_id"),
                    "anchor_baseline_version_id": registry.get("anchor_baseline_version_id"),
                    "serving_version": target,
                },
            )
        self._log("rollback", result["status"], "success", "rollback target evaluated", {"target_version_id": target.get("version_id")})
        return result

    def maybe_rollback(self, infer_metrics: dict[str, Any]) -> dict[str, Any]:
        gates = self.policy.get("rollback_gates", {})
        degraded_rate = infer_metrics.get("degraded_rate", 0)
        error_rate = infer_metrics.get("error_rate", 0)
        explanation_failure_rate = infer_metrics.get("explanation_failure_rate", 0)
        if (
            error_rate > gates.get("infer_error_rate_threshold", 1)
            or degraded_rate > gates.get("degraded_rate_spike_threshold", 1)
            or explanation_failure_rate > gates.get("explanation_failure_rate_threshold", 1)
        ):
            return self.rollback(target_version_id="latest_stable", dry_run=True)
        self._log("rollback_monitor", "continue_serving", "success", "rollback gates not triggered", infer_metrics)
        return {"status": "not_triggered", "action": "rollback_monitor", "metrics": infer_metrics}
