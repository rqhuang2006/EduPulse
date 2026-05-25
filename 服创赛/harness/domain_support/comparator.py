from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Universal comparability checker (pre-compare gate)
# ---------------------------------------------------------------------------

COMPARABILITY_KEYS = [
    "eval_scope",
    "sample_universe",
    "split_strategy",
    "label_definition",
    "feature_contract_hash",
    "task_scope",
]


def check_comparability(
    *,
    candidate_context: dict[str, Any],
    baseline_context: dict[str, Any],
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Check whether candidate and baseline are strictly comparable.

    Returns a unified comparability report with:
      - comparable: true/false
      - severity: info / warning / blocking
      - mismatches: list of mismatch descriptions
      - recommended_action: strict_rebuild / reference_only / block_decision
      - comparison_role: primary / reference_only
    """
    required_keys = required_keys or COMPARABILITY_KEYS
    mismatches: list[str] = []

    for key in required_keys:
        candidate_value = candidate_context.get(key)
        baseline_value = baseline_context.get(key)
        if candidate_value != baseline_value:
            mismatches.append(
                f"{key}: candidate={candidate_value!r} baseline={baseline_value!r}"
            )

    comparable = len(mismatches) == 0

    # Determine severity and recommended action
    blocking_keys = {"label_definition", "sample_universe", "eval_scope"}
    warning_keys = {"split_strategy", "feature_contract_hash"}
    has_blocking = any(
        m.startswith(key + ":") for key in blocking_keys for m in mismatches
    )
    has_warning = any(
        m.startswith(key + ":") for key in warning_keys for m in mismatches
    )

    if has_blocking:
        severity = "blocking"
        recommended_action = "block_decision"
        comparison_role = "reference_only"
    elif has_warning:
        severity = "warning"
        recommended_action = "reference_only"
        comparison_role = "reference_only"
    else:
        severity = "info"
        recommended_action = "strict_rebuild"
        comparison_role = "primary"

    return {
        "comparable": comparable,
        "severity": severity,
        "mismatches": mismatches,
        "recommended_action": recommended_action,
        "comparison_role": comparison_role,
        "required_keys": required_keys,
        "conclusion": (
            "candidate and baseline are strictly comparable"
            if comparable
            else "candidate and baseline are NOT strictly comparable"
        ),
    }


def same_caliber_compare_guard(
    *,
    candidate_context: dict[str, Any],
    baseline_context: dict[str, Any],
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    required_keys = required_keys or ["architecture_version", "task_scope", "label_definition", "eval_scope"]
    mismatches: list[str] = []
    for key in required_keys:
        candidate_value = candidate_context.get(key)
        baseline_value = baseline_context.get(key)
        if candidate_value != baseline_value:
            mismatches.append(f"{key}: candidate={candidate_value!r} baseline={baseline_value!r}")
    return {
        "same_caliber": not mismatches,
        "required_keys": required_keys,
        "mismatches": mismatches,
        "conclusion": "candidate and baseline are comparable" if not mismatches else "candidate and baseline are not same-caliber",
    }


def compare_candidate_to_baseline(
    *,
    candidate_version_id: str,
    candidate_metrics: dict[str, Any],
    baseline_version_id: str | None,
    baseline_metrics: dict[str, Any],
    anchor_baseline_version_id: str | None = None,
    anchor_metrics: dict[str, Any] | None = None,
    primary_metric: str = "auc",
    comparison_mode: str = "frozen_baseline_compare",
    comparability_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_metrics = anchor_metrics or baseline_metrics

    def _delta(metric_name: str) -> float | None:
        candidate_value = candidate_metrics.get(metric_name)
        baseline_value = baseline_metrics.get(metric_name)
        if candidate_value is None or baseline_value is None:
            return None
        return round(float(candidate_value) - float(baseline_value), 6)

    return {
        "candidate_version_id": candidate_version_id,
        "baseline_version_id": baseline_version_id,
        "anchor_baseline_version_id": anchor_baseline_version_id or baseline_version_id,
        "comparison_mode": comparison_mode,
        "primary_metric": primary_metric,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "anchor_metrics": anchor_metrics,
        "metric_deltas": {
            "auc_delta": _delta("auc"),
            "f1_delta": _delta("f1"),
            "precision_delta": _delta("precision"),
            "recall_delta": _delta("recall"),
        },
        "comparability": comparability_report or {
            "same_sample_universe": True,
            "same_eval_scope": True,
            "same_label_definition": True,
            "strict_comparable": True,
            "comparison_role": "primary",
        },
    }
