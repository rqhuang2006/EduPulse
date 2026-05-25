from __future__ import annotations

from pathlib import Path

from harness.actions.base import BaseAction
from harness.contracts import ActionResult, ValidationResult
from harness.validators.artifact_validator import ArtifactValidator
from study_domain.adapters.diagnose_adapter import StudyDiagnoseAdapter


class DiagnoseAction(BaseAction):
    name = "diagnose"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.adapter = StudyDiagnoseAdapter(root_dir)
        self.artifact_validator = ArtifactValidator()

    def run(self, context) -> ActionResult:
        payload = self.adapter.execute(context.request)
        validations = self.artifact_validator.validate(payload["artifacts"])
        screening = payload["diagnostics"].get("feature_screening", {})
        if screening.get("dropped_feature_count", 0):
            validations.append(
                ValidationResult(
                    validator_name="DiagnoseAction",
                    passed=True,
                    severity="info",
                    reason_code="feature_screening_present",
                    message="Feature screening report available for policy review.",
                    details=screening,
                )
            )
        return ActionResult(
            action_name=self.name,
            status=payload["status"],
            diagnostics=payload["diagnostics"],
            artifacts=payload["artifacts"],
            validations=validations,
            message=payload["message"],
        )
