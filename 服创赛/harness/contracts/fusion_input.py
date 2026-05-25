from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FusionInputContract:
    """Normalized fusion input that every domain agent must export.

    This contract is domain-agnostic. Each domain agent (study, life, sport)
    produces one of these, and the fusion layer combines them.
    """

    # Domain identification
    domain_name: str
    candidate_version_id: str | None = None

    # Risk assessment
    risk_score: float | None = None
    risk_level: str | None = None  # e.g., "low", "medium", "high", "critical"
    confidence: float | None = None

    # Explainability
    top_features: list[dict[str, Any]] = field(default_factory=list)
    explanations: list[dict[str, Any]] = field(default_factory=list)

    # Metrics and validation
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    metric_context: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    warning_summary: list[str] = field(default_factory=list)

    # Artifact references
    artifact_ref: dict[str, Any] = field(default_factory=dict)

    # Raw payload for domain-specific data not covered by generic fields
    raw_payload: dict[str, Any] = field(default_factory=dict)


def validate_fusion_input(
    payload: FusionInputContract | dict[str, Any],
    *,
    allow_stub: bool = False,
) -> dict[str, Any]:
    """Validate a normalized fusion payload and return an auditable summary."""

    candidate = asdict(payload) if isinstance(payload, FusionInputContract) else dict(payload)
    errors: list[str] = []
    warnings: list[str] = []

    domain_name = str(candidate.get("domain_name") or "").strip()
    if not domain_name:
        errors.append("domain_name_missing")

    risk_level = candidate.get("risk_level")
    allowed_risk_levels = {"low", "medium", "high", "critical", "stub", None, ""}
    if risk_level not in allowed_risk_levels:
        errors.append("risk_level_invalid")

    confidence = candidate.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        errors.append("confidence_not_numeric")

    risk_score = candidate.get("risk_score")
    if risk_score is None:
        if not allow_stub:
            warnings.append("risk_score_missing")
    elif not isinstance(risk_score, (int, float)):
        errors.append("risk_score_not_numeric")

    explanations = candidate.get("explanations", [])
    if not isinstance(explanations, list):
        errors.append("explanations_not_list")

    artifact_ref = candidate.get("artifact_ref", {})
    if not isinstance(artifact_ref, dict):
        errors.append("artifact_ref_not_object")

    metric_context = candidate.get("metric_context", {})
    if not isinstance(metric_context, dict):
        errors.append("metric_context_not_object")

    quality_metrics = candidate.get("quality_metrics", {})
    if not isinstance(quality_metrics, dict):
        errors.append("quality_metrics_not_object")

    validation_summary = candidate.get("validation_summary", {})
    if not isinstance(validation_summary, dict):
        errors.append("validation_summary_not_object")
    else:
        if "fusion_payload_semantics_ok" not in validation_summary:
            warnings.append("fusion_payload_semantics_flag_missing")
        if "risk_score_source" not in validation_summary:
            warnings.append("risk_score_source_missing")
        if "quality_metric_source" not in validation_summary:
            warnings.append("quality_metric_source_missing")

    warning_summary = candidate.get("warning_summary", [])
    if not isinstance(warning_summary, list):
        errors.append("warning_summary_not_list")

    auc = quality_metrics.get("auc") if isinstance(quality_metrics, dict) else None
    if isinstance(risk_score, (int, float)) and isinstance(auc, (int, float)) and float(risk_score) == float(auc):
        warnings.append("risk_score_matches_auc_check_semantics")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized_payload": candidate,
        "is_stub": allow_stub or risk_level == "stub",
    }
