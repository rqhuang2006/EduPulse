from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactRef:
    name: str
    kind: str
    uri: str
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
