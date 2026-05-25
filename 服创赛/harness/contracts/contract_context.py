from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContractContext:
    """Generic contract context supplied by domain agents.

    Contains the information needed for same-caliber checks,
    baseline comparisons, and local gain evaluation without
    embedding domain-specific field names.
    """

    # Baseline identification
    baseline_version_id: str | None = None
    anchor_baseline_version_id: str | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    anchor_baseline_metrics: dict[str, Any] = field(default_factory=dict)

    # Candidate identification
    candidate_version_id: str | None = None
    candidate_metrics: dict[str, Any] = field(default_factory=dict)

    # Scope descriptors (domain-specific meaning, generic names)
    architecture_version: str | None = None
    task_scope: str | None = None
    label_definition: str | None = None
    eval_split: str | None = None
    data_mode: str | None = None

    # Chain/contract validation results
    chain_validation: dict[str, Any] = field(default_factory=dict)

    # Local gain signals (domain-specific flags, generic container)
    local_gain_flags: dict[str, Any] = field(default_factory=dict)

    # Metric context for auditability
    metric_context: dict[str, Any] = field(default_factory=dict)

    # Frozen snapshot references
    baseline_frozen_snapshot: dict[str, Any] = field(default_factory=dict)
    candidate_frozen_snapshot: dict[str, Any] = field(default_factory=dict)
