from __future__ import annotations

import runpy
from pathlib import Path

from harness.contracts import ArtifactRef


class StudyTrainAdapter:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.dm_dir = root_dir / "data" / "dm"
        self.deliverable_dir = root_dir / "data" / "deliverables" / "study"

    def execute(self, request: dict) -> dict:
        runpy.run_path(str(self.root_dir / "src" / "30_train_study_model.py"), run_name="__main__")
        artifacts = [
            ArtifactRef("study_model_bundle", "model_bundle", str(self.dm_dir / "study_model.pkl")),
            ArtifactRef("study_model_config", "model_config", str(self.dm_dir / "study_model_config.json")),
            ArtifactRef("study_model_metrics", "metrics_report", str(self.dm_dir / "study_model_metrics.json")),
            ArtifactRef("study_eval_report", "metrics_report", str(self.dm_dir / "study_eval_report.json")),
            ArtifactRef("study_subgroup_metrics", "metrics_report", str(self.dm_dir / "study_subgroup_metrics.csv")),
            ArtifactRef("study_confidence_zone_report", "metrics_report", str(self.dm_dir / "study_confidence_zone_report.csv")),
        ]
        return {
            "status": "success",
            "artifacts": artifacts,
            "message": "Study training pipeline executed through existing domain trainer.",
        }
