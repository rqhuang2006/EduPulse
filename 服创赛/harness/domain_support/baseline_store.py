from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BaselineSnapshot:
    version_id: str
    anchor_version_id: str
    snapshot_dir: Path
    metrics: dict[str, Any]
    anchor_metrics: dict[str, Any]
    manifest: dict[str, str]
    compare_available: bool
    bootstrap_mode: bool
    created_in_current_run: bool = False


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class BaselineStore:
    """Generic frozen baseline registry for domain adapters and agents."""

    def __init__(self, registry_root: Path):
        self.registry_root = registry_root
        self.index_path = registry_root / "baseline_index.json"

    def load_active(self) -> BaselineSnapshot | None:
        index_payload = read_json(self.index_path, {})
        if not index_payload:
            return None

        version_id = str(index_payload.get("active_baseline_version_id") or "").strip()
        if not version_id:
            return None
        anchor_version_id = str(index_payload.get("anchor_baseline_version_id") or version_id)
        is_frozen = bool(index_payload.get("is_frozen", False))

        snapshot_dir = self.registry_root / version_id
        metrics_payload = read_json(snapshot_dir / "metrics.json", {})
        anchor_payload = read_json(self.registry_root / anchor_version_id / "metrics.json", metrics_payload)

        manifest = {
            filename: str(snapshot_dir / filename)
            for filename in (
                "model_config.json",
                "feature_config.json",
                "contract_context.json",
                "domain_audit.json",
                "metrics.json",
            )
        }
        compare_available = is_frozen and snapshot_dir.exists() and (snapshot_dir / "metrics.json").exists()
        return BaselineSnapshot(
            version_id=version_id,
            anchor_version_id=anchor_version_id,
            snapshot_dir=snapshot_dir,
            metrics=metrics_payload.get("summary_metrics", metrics_payload),
            anchor_metrics=anchor_payload.get("summary_metrics", anchor_payload),
            manifest=manifest,
            compare_available=compare_available,
            bootstrap_mode=not compare_available,
        )

    def freeze_active(
        self,
        *,
        version_id: str,
        anchor_version_id: str | None = None,
        created_in_current_run: bool = False,
    ) -> BaselineSnapshot:
        anchor = anchor_version_id or version_id
        write_json(
            self.index_path,
            {
                "active_baseline_version_id": version_id,
                "anchor_baseline_version_id": anchor,
                "is_frozen": True,
                "created_in_current_run": created_in_current_run,
            },
        )
        snapshot = self.load_active()
        if snapshot is None:
            raise RuntimeError("failed to load frozen baseline after writing baseline index")
        snapshot.created_in_current_run = created_in_current_run
        return snapshot
