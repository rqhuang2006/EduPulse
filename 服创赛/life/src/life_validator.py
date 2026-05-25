from __future__ import annotations

from typing import Any


def evaluate_temporal_consistency(auc_same_sample: float | None, auc_temporal: float | None) -> dict[str, Any]:
    if auc_same_sample is None or auc_temporal is None:
        return {"temporal_gap": None, "flag_temporal_instability": False}
    gap = abs(float(auc_same_sample) - float(auc_temporal))
    return {"temporal_gap": gap, "flag_temporal_instability": gap > 0.1}


def evaluate_leakage_risk(
    *,
    label_feature_overlap: list[str],
    explicit_temporal_leakage_detected: bool,
    future_window_used: bool,
) -> bool:
    # Future-window label usage itself is expected for mainline prediction and is not leakage.
    _ = future_window_used
    return bool(label_feature_overlap) or explicit_temporal_leakage_detected


def evaluate_subgroup_stability(subgroup_auc_values: list[float]) -> dict[str, Any]:
    if not subgroup_auc_values:
        return {"subgroup_variance": None, "subgroup_unstable": False}
    spread = max(subgroup_auc_values) - min(subgroup_auc_values)
    return {"subgroup_variance": float(spread), "subgroup_unstable": spread > 0.15}


def compute_trust_score(
    *,
    temporal_gap: float | None,
    leakage: bool,
    subgroup_unstable: bool,
) -> float:
    score = 1.0
    if temporal_gap is not None and temporal_gap > 0.1:
        score -= 0.4
    if leakage:
        score -= 0.4
    if subgroup_unstable:
        score -= 0.2
    return max(0.0, round(score, 4))
