from __future__ import annotations

from pathlib import Path

from harness.actions.base import BaseAction
from harness.contracts import ActionResult
from harness.validators.artifact_validator import ArtifactValidator
from study_domain.adapters.publish_adapter import StudyPublishAdapter


class RollbackAction(BaseAction):
    name = "rollback"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.adapter = StudyPublishAdapter(root_dir)
        self.artifact_validator = ArtifactValidator()

    def run(self, context) -> ActionResult:
        payload = self.adapter.execute_rollback(context.request)
        validations = self.artifact_validator.validate(payload["artifacts"])
        return ActionResult(
            action_name=self.name,
            status=payload["status"],
            metrics=payload["metrics"],
            diagnostics=payload["diagnostics"],
            artifacts=payload["artifacts"],
            validations=validations,
            message=payload["message"],
        )
