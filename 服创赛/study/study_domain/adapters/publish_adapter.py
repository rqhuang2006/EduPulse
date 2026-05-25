from __future__ import annotations

import json
from pathlib import Path

from src.study_release_manager import StudyReleaseManager

from harness.contracts import ArtifactRef


class StudyPublishAdapter:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.dm_dir = root_dir / "data" / "dm"

    @staticmethod
    def _read_json(path: Path, default: dict | None = None) -> dict:
        if default is None:
            default = {}
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def execute_publish(self, request: dict) -> dict:
        # Inject fresh eval metrics BEFORE creating the manager,
        # so StudyReleaseManager.__init__ captures the enriched request.
        eval_report = self._read_json(self.dm_dir / "study_eval_report.json", {})
        model_metrics = self._read_json(self.dm_dir / "study_model_metrics.json", {})
        if eval_report.get("overall_metrics"):
            fresh_metrics = {
                "auc": eval_report["overall_metrics"].get("auc"),
                "f1": eval_report["overall_metrics"].get("f1"),
                "recall": eval_report["overall_metrics"].get("recall"),
                "precision": eval_report["overall_metrics"].get("precision"),
                "coverage": eval_report["overall_metrics"].get("coverage"),
                "degraded_ratio": eval_report["overall_metrics"].get("degraded_ratio"),
            }
            request = {**request, "fresh_eval_metrics": fresh_metrics}
        elif model_metrics.get("core_model", {}).get("valid"):
            fresh_metrics = dict(model_metrics["core_model"]["valid"])
            request = {**request, "fresh_eval_metrics": fresh_metrics}

        manager = StudyReleaseManager(request)
        result = manager.publish(
            candidate_version_id=request.get("candidate_version_id"),
            dry_run=True,
            require_approval=False,
        )
        release_status = result.get("status")
        return {
            "status": "failed" if release_status == "failed" else "success",
            "metrics": {
                "policy_decision": result.get("policy_decision"),
                "release_status": release_status,
            },
            "diagnostics": result,
            "artifacts": [ArtifactRef("publish_dry_run", "publish_bundle", str(self.dm_dir / "study_publish_dry_run_record.json"))],
            "message": "Publish dry-run evaluated through release manager.",
        }

    def execute_rollback(self, request: dict) -> dict:
        manager = StudyReleaseManager(request)
        result = manager.rollback(
            target_version_id=request.get("target_version_id"),
            dry_run=True,
        )
        return {
            "status": "success" if result.get("status") in {"dry_run", "rolled_back"} else "failed",
            "metrics": {"rollback_status": result.get("status")},
            "diagnostics": result,
            "artifacts": [ArtifactRef("rollback_dry_run", "rollback_bundle", str(self.dm_dir / "study_rollback_dry_run_record.json"))],
            "message": "Rollback dry-run evaluated through release manager.",
        }
