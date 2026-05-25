from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .artifact import ArtifactRef
from .validation import ValidationResult


@dataclass
class ActionResult:
    action_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    validations: list[ValidationResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    message: str = ""
