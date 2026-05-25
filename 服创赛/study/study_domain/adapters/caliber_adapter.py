from __future__ import annotations

from pathlib import Path

from src.study_release_manager import StudyReleaseManager


class StudyCaliberAdapter:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def execute(self, request: dict) -> dict:
        manager = StudyReleaseManager(request)
        return manager.compare_caliber(candidate_version_id=request.get("candidate_version_id"))
