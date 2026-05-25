from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    validator_name: str
    passed: bool
    severity: str
    reason_code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
