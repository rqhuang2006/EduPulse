from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.contracts import ArtifactRef


class StudyDiagnoseAdapter:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.dm_dir = root_dir / "data" / "dm"

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def execute(self, request: dict) -> dict:
        feature_screening = self._read_json(self.dm_dir / "study_feature_screening_report.json", {})
        subgroup_screening = self._read_json(self.dm_dir / "study_subgroup_feature_screening_report.json", {})
        quality_report = self._read_json(self.dm_dir / "study_quality_report.json", {})
        layer_summary = self._read_json(self.dm_dir / "study_feature_layer_summary.json", {})
        diagnostic_summary = {
            "feature_screening": {
                "kept_feature_count": feature_screening.get("kept_feature_count"),
                "dropped_feature_count": feature_screening.get("dropped_feature_count"),
                "dropped_counts_by_reason": feature_screening.get("dropped_counts_by_reason", {}),
            },
            "subgroup_screening": {
                "kept_feature_count": subgroup_screening.get("kept_feature_count"),
                "kept_counts_by_layer": subgroup_screening.get("kept_counts_by_layer", {}),
            },
            "quality_report": {
                "degraded_row_count": quality_report.get("degraded_row_count"),
                "join_failure_rows": quality_report.get("join_failure_rows"),
                "source_coverage_mean": quality_report.get("source_coverage_mean"),
            },
            "feature_layers": layer_summary.get("counts", {}),
        }
        artifacts = [
            ArtifactRef("diagnostic_summary", "diagnostic_report", str(self.dm_dir / "study_feature_screening_report.json")),
            ArtifactRef("subgroup_screening", "diagnostic_report", str(self.dm_dir / "study_subgroup_feature_screening_report.json")),
            ArtifactRef("quality_report", "diagnostic_report", str(self.dm_dir / "study_quality_report.json")),
        ]
        return {
            "status": "success",
            "artifacts": artifacts,
            "diagnostics": diagnostic_summary,
            "message": "Structural diagnostics assembled from study domain artifacts.",
        }
