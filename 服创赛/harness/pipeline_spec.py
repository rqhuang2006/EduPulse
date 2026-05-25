from __future__ import annotations

from dataclasses import dataclass, field

from harness.actions.base import BaseAction


@dataclass
class PipelineSpec:
    name: str
    pre_policy_actions: list[BaseAction] = field(default_factory=list)
    post_policy_actions: list[BaseAction] = field(default_factory=list)
