from __future__ import annotations

from typing import Any


def normalize_decision_bundle(
    *,
    final_decision: str,
    policy_decision: str | None = None,
    execution_mode: str = "",
    reason_codes: list[str] | None = None,
    decision_stage_reached: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "final_decision": final_decision,
        "policy_decision": policy_decision or final_decision,
        "execution_mode": execution_mode,
        "reason_codes": list(reason_codes or []),
        "decision_stage_reached": decision_stage_reached,
        **(extra or {}),
    }


def normalize_domain_result(
    *,
    domain: str,
    status: str,
    summary_metrics: dict[str, Any],
    decision_bundle: dict[str, Any],
    metric_context: dict[str, Any],
    domain_context: dict[str, Any],
    domain_audit: dict[str, Any],
    warnings: list[str],
    deliverables: dict[str, Any],
    harness_payload: dict[str, Any],
    fusion_input: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "status": status,
        **decision_bundle,
        "summary_metrics": summary_metrics,
        "metric_context": metric_context,
        "domain_context": domain_context,
        "domain_audit": domain_audit,
        "warnings": warnings,
        "deliverables": deliverables,
        "harness_v1": harness_payload,
        "fusion_input": fusion_input,
        **(extra or {}),
    }
