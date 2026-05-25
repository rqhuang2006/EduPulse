from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactManifest:
    """Generic artifact manifest for a harness run.

    Tracks all artifacts produced by domain agents during execution.
    """

    run_id: str
    domain: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
