from __future__ import annotations

from abc import ABC, abstractmethod

from harness.contracts import ActionResult, PipelineContext


class BaseAction(ABC):
    name: str

    @abstractmethod
    def run(self, context: PipelineContext) -> ActionResult:
        raise NotImplementedError
