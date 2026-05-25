from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoutingPolicy:
    low_conf_lower: float = 0.35
    low_conf_upper: float = 0.65
    behavior_alpha: float = 0.15
    subgroup_beta: float = 0.20


def resolve_policy(config: dict[str, Any] | None = None) -> RoutingPolicy:
    config = config or {}
    return RoutingPolicy(
        low_conf_lower=float(config.get("low_conf_lower", 0.35)),
        low_conf_upper=float(config.get("low_conf_upper", 0.65)),
        behavior_alpha=float(config.get("behavior_alpha", 0.15)),
        subgroup_beta=float(config.get("subgroup_beta", 0.20)),
    )


def confidence_zone(base_score: np.ndarray | pd.Series, policy: RoutingPolicy) -> np.ndarray:
    score = np.asarray(base_score, dtype=float)
    return np.where(
        score <= policy.low_conf_lower,
        "high_conf_negative",
        np.where(score >= policy.low_conf_upper, "high_conf_positive", "middle_correction"),
    )


def apply_serving_policy(
    base_score: np.ndarray | pd.Series,
    behavior_signal: np.ndarray | pd.Series | None,
    subgroup_signal: np.ndarray | pd.Series | None,
    data_mode: pd.Series | np.ndarray | None,
    policy: RoutingPolicy,
    subtype_signal: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    output_index = base_score.index if isinstance(base_score, pd.Series) else None
    base = np.asarray(base_score, dtype=float)
    behavior = np.asarray(behavior_signal, dtype=float) if behavior_signal is not None else np.full(len(base), np.nan)
    subgroup = np.asarray(subgroup_signal, dtype=float) if subgroup_signal is not None else np.full(len(base), np.nan)
    subtype = np.asarray(subtype_signal, dtype=float) if subtype_signal is not None else np.full(len(base), np.nan)
    mode = np.asarray(pd.Series(data_mode if data_mode is not None else ["core_only"] * len(base)).astype(str))

    zone = confidence_zone(base, policy)
    in_middle = zone == "middle_correction"
    degraded = mode == "degraded_sparse"

    behavior_delta = np.zeros(len(base), dtype=float)
    subgroup_delta = np.zeros(len(base), dtype=float)

    valid_behavior = in_middle & ~degraded & ~np.isnan(behavior)
    if valid_behavior.any():
        behavior_delta[valid_behavior] = policy.behavior_alpha * (behavior[valid_behavior] - base[valid_behavior])

    valid_subgroup = in_middle & ~degraded & ~np.isnan(subgroup)
    if valid_subgroup.any():
        subgroup_strength = np.clip((subgroup[valid_subgroup] - 0.5) * 2.0, 0.0, 1.0)
        subtype_gate = np.where(np.isnan(subtype[valid_subgroup]), 1.0, np.clip(subtype[valid_subgroup], 0.0, 1.0))
        subgroup_delta[valid_subgroup] = policy.subgroup_beta * subgroup_strength * subtype_gate

    final = np.clip(base + behavior_delta + subgroup_delta, 0.0, 1.0)
    routing_reason = np.full(len(base), "base_only_confident", dtype=object)
    routing_reason[in_middle] = "middle_zone_base_only"
    routing_reason[valid_behavior] = "middle_zone_behavior_residual"
    routing_reason[valid_subgroup] = "middle_zone_single_fail_expert"
    routing_reason[valid_behavior & valid_subgroup] = "middle_zone_behavior_plus_single_fail_expert"
    routing_reason[degraded] = "degraded_sparse_base_only"

    return pd.DataFrame(
        {
            "BASE_SCORE": base,
            "BEHAVIOR_DELTA": behavior_delta,
            "SUBGROUP_DELTA": subgroup_delta,
            "FINAL_SCORE": final,
            "ROUTING_REASON": routing_reason,
            "CONFIDENCE_ZONE": zone,
        },
        index=output_index,
    )
