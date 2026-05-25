from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_result import ActionResult
from .artifact import ArtifactRef
from .policy import PolicyDecision

@dataclass
class PipelineContext:
    run_id: str
    pipeline_name: str
    domain: str
    request: dict[str, Any]
    root_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    baseline_info: dict[str, Any] = field(default_factory=dict)
    candidate_info: dict[str, Any] = field(default_factory=dict)
    domain_context: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    stage_results: list[ActionResult] = field(default_factory=list)
    final_decision: PolicyDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ActionResult) -> None:
        self.stage_results.append(result)
        self.artifacts.extend(result.artifacts)

    def add_artifacts(self, artifacts: list[ArtifactRef]) -> None:
        self.artifacts.extend(artifacts)

    def latest_result(self, action_name: str) -> ActionResult | None:
        for result in reversed(self.stage_results):
            if result.action_name == action_name:
                return result
        return None


@dataclass
class RunRecord:
    run_id: str
    pipeline_name: str
    domain: str
    stage_results: list[ActionResult]
    final_decision: PolicyDecision | None
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)

    run_type: str = "single_domain"
    domain_name: str = ""

    eval_scope: str = ""
    task_scope: str = ""
    feature_contract_hash: str = ""
    label_definition: str = ""
    baseline_version_id: str = ""
    anchor_baseline_version_id: str = ""
    comparison_mode: str = ""
    policy_decision: str = ""
    execution_mode: str = ""
    decision_stage_reached: str = ""
    collected_warnings: list[str] = field(default_factory=list)

    domain_context: dict[str, Any] = field(default_factory=dict)
    metric_context: dict[str, Any] = field(default_factory=dict)
    domain_audit: dict[str, Any] = field(default_factory=dict)
    multi_domain_audit: dict[str, Any] = field(default_factory=dict)
