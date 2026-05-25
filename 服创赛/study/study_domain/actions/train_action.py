from __future__ import annotations

from pathlib import Path

from harness.actions.base import BaseAction
from harness.contracts import ActionResult
from harness.validators.artifact_validator import ArtifactValidator
from study_domain.adapters.train_adapter import StudyTrainAdapter
from study_domain.validators.data_validator import DataValidator


class TrainAction(BaseAction):
    name = "train"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.adapter = StudyTrainAdapter(root_dir)
        self.data_validator = DataValidator()
        self.artifact_validator = ArtifactValidator()

    def run(self, context) -> ActionResult:
        train_path = self.root_dir / "data" / "dm" / "study_train_table.csv"
        validations = self.data_validator.validate_train_source(train_path)
        if any(v.severity == "block" and not v.passed for v in validations):
            return ActionResult(action_name=self.name, status="failed", validations=validations, message="Training blocked by data validation.")

        payload = self.adapter.execute(context.request)
        context.candidate_info = {"artifacts": [artifact.uri for artifact in payload["artifacts"]]}
        validations.extend(self.artifact_validator.validate(payload["artifacts"]))
        return ActionResult(
            action_name=self.name,
            status=payload["status"],
            artifacts=payload["artifacts"],
            validations=validations,
            message=payload["message"],
        )
