from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DM_DIR = ROOT / "data" / "dm"
REGISTRY_ROOT = ROOT / "data" / "registry" / "life"
REPORT_PATH = DM_DIR / "life_domain_boundary_report.json"

FORBIDDEN_DATASET_KEYS = ("running", "exercise", "pe_course", "fitness")
FORBIDDEN_FEATURES = (
    "running_count",
    "running_state_mean",
    "exercise_count",
    "exercise_weeks",
    "pe_course_count",
    "fitness_score",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def contains_forbidden(payload: Any, forbidden_terms: tuple[str, ...]) -> list[str]:
    text = json.dumps(payload, ensure_ascii=False)
    return [term for term in forbidden_terms if term in text]


def main() -> None:
    file_registry_path = ROOT / "conf" / "file_registry.yaml"
    result_path = DM_DIR / "life_agent_result.json"
    registry_index_path = REGISTRY_ROOT / "baseline_index.json"

    checks: dict[str, Any] = {
        "file_registry_boundary_ok": True,
        "result_boundary_ok": True,
        "feature_config_boundary_ok": True,
        "domain_audit_boundary_ok": True,
        "feature_columns_boundary_ok": True,
    }
    violations: dict[str, list[str]] = {}

    file_registry_text = file_registry_path.read_text(encoding="utf-8")
    registry_hits = [term for term in FORBIDDEN_DATASET_KEYS if term in file_registry_text]
    if registry_hits:
        checks["file_registry_boundary_ok"] = False
        violations["file_registry"] = registry_hits

    result_payload = read_json(result_path)
    result_hits = contains_forbidden(result_payload, FORBIDDEN_DATASET_KEYS + FORBIDDEN_FEATURES)
    if result_hits:
        checks["result_boundary_ok"] = False
        violations["life_agent_result"] = result_hits

    candidate_version_id = result_payload.get("harness_v1", {}).get("candidate_version_id")
    candidate_dir = REGISTRY_ROOT / str(candidate_version_id)
    feature_config = read_json(candidate_dir / "feature_config.json")
    feature_hits = contains_forbidden(feature_config, FORBIDDEN_DATASET_KEYS + FORBIDDEN_FEATURES)
    if feature_hits:
        checks["feature_config_boundary_ok"] = False
        violations["feature_config"] = feature_hits

    domain_audit = read_json(candidate_dir / "domain_audit.json")
    audit_hits = contains_forbidden(domain_audit, FORBIDDEN_DATASET_KEYS + FORBIDDEN_FEATURES)
    if audit_hits:
        checks["domain_audit_boundary_ok"] = False
        violations["domain_audit"] = audit_hits

    model_config = read_json(candidate_dir / "model_config.json")
    feature_columns = model_config.get("feature_columns", [])
    feature_column_hits = [name for name in FORBIDDEN_FEATURES if name in feature_columns]
    if feature_column_hits:
        checks["feature_columns_boundary_ok"] = False
        violations["feature_columns"] = feature_column_hits

    report = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "domain": "life",
        "candidate_version_id": candidate_version_id,
        "checks": checks,
        "violations": violations,
        "summary": {
            "boundary_ok": all(checks.values()),
            "statement": "sport-related datasets intentionally excluded from life domain",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
