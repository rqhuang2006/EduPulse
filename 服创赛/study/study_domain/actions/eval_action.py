from __future__ import annotations

from pathlib import Path

from harness.actions.base import BaseAction
from harness.contracts import ActionResult, ValidationResult
from harness.validators.artifact_validator import ArtifactValidator
from study_domain.adapters.eval_adapter import StudyEvalAdapter
from study_domain.validators.metric_validator import MetricValidator
from study_domain.validators.subgroup_validator import SubgroupValidator


class EvalAction(BaseAction):
    name = "eval"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.adapter = StudyEvalAdapter(root_dir)
        self.metric_validator = MetricValidator()
        self.subgroup_validator = SubgroupValidator()
        self.artifact_validator = ArtifactValidator()

    def run(self, context) -> ActionResult:
        payload = self.adapter.execute(context.request)
        context.baseline_info = payload["diagnostics"].get("baseline_ids", {})
        context.candidate_info.update({"metrics": payload["metrics"]})
        metric_context = payload["diagnostics"].get("metric_context", {})
        context.metadata.update(metric_context)
        validations = []
        validations.extend(self.metric_validator.validate(payload["metrics"], payload["diagnostics"].get("baseline_metrics", {})))
        validations.extend(
            self.subgroup_validator.validate(
                payload["diagnostics"].get("subtype_metrics", {}) | payload["diagnostics"].get("mode_metrics", {}),
                {},
            )
        )
        validations.extend(self.artifact_validator.validate(payload["artifacts"]))
        validations.append(
            ValidationResult(
                validator_name=self.name,
                passed=bool(metric_context.get("feature_contract_hash")),
                severity="error" if not metric_context.get("feature_contract_hash") else "info",
                reason_code="feature_contract_hash_present" if metric_context.get("feature_contract_hash") else "feature_contract_hash_missing",
                message="Metric context exported for harness comparison.",
                details=metric_context,
            )
        )
        action_metrics = {**payload["metrics"], **metric_context}
        return ActionResult(
            action_name=self.name,
            status=payload["status"],
            metrics=action_metrics,
            diagnostics=payload["diagnostics"],
            artifacts=payload["artifacts"],
            validations=validations,
            message=payload["message"],
        )
