from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Canonical decision labels for the harness decision layer.
# Each label has a single, unambiguous meaning:
#   reject                  – hard gate failure (chain, contract, dry-run failure)
#   keep_baseline           – candidate passed gates but did not beat baseline
#   incomparable_candidate  – same_caliber gate failed; candidate cannot be compared
#   dry_run_only            – candidate has publish evidence but error-level issues prevent accept
#   promotion_recommended   – policy passed; candidate ready but no publish action executed yet
#   promotion_pending_approval – policy passed; require_approval=true, awaiting human review
#   published               – candidate has actually been promoted to serving
#   rollback_recommended    – serving health gate triggered rollback suggestion
DECISION_LABELS = frozenset([
    "reject",
    "keep_baseline",
    "incomparable_candidate",
    "dry_run_only",
    "promotion_recommended",
    "promotion_pending_approval",
    "published",
    "rollback_recommended",
])

# Decision-stage names corresponding to the explicit precedence order.
DECISION_STAGES = [
    "contract_chain_gate",
    "same_caliber_gate",
    "floor_gate",
    "anchor_regression_gate",
    "active_baseline_comparison",
    "local_gain_gate",
    "approval_release_stage",
]


@dataclass
class PolicyDecision:
    decision: str
    reason_codes: list[str]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    # Execution-mode label kept separate from the strategy decision.
    # Examples: "dry_run", "pending_approval", "promoted".
    execution_mode: str = ""
    # Which decision stage determined the outcome (from DECISION_STAGES)
    decision_stage_reached: str = ""
    # Name of the specific gate that made the final call (empty if not gate-decided)
    gate_name: str = ""
    # Warnings collected during the run, persisted for auditability
    collected_warnings: list[str] = field(default_factory=list)
