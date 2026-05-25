from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.registry.snapshot_contract import REQUIRED_SNAPSHOT_FILES


def validate_snapshot_completeness(snapshot_dir: Path) -> dict[str, Any]:
    missing = [filename for filename in REQUIRED_SNAPSHOT_FILES if not (snapshot_dir / filename).exists()]
    return {
        "ok": not missing,
        "snapshot_dir": str(snapshot_dir),
        "required_files": list(REQUIRED_SNAPSHOT_FILES),
        "missing_files": missing,
    }
