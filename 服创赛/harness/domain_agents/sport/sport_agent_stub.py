from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from harness.contracts import ContractContext, FusionInputContract
from harness.domain_agents.base import BaseDomainAgent


class SportAgentStub(BaseDomainAgent):
    """Stub sport agent – returns ``not_implemented`` status.

    Kept for backward-compatibility and for any path that explicitly
    wants the stub behaviour.  The real adapter lives in
    ``sport_agent_adapter.py``.
    """

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir or Path(".")

    @property
    def domain_name(self) -> str:
        return "sport"

    def run_domain_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "status": "not_implemented",
            "final_decision": {"decision": "dry_run_only", "summary": "Sport domain not implemented (stub)"},
            "policy_decision": "dry_run_only",
            "execution_mode": "dry_run",
            "decision_stage_reached": "",
            "metrics": {},
            "metric_context": {},
            "domain_context": {},
            "domain_audit": {},
            "validation_summary": [],
            "warning_summary": ["sport agent is a stub"],
            "artifact_ref": None,
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": self.__class__.__name__,
                "execution_path": "orchestrator -> stub (no real agent)",
                "domain_pipeline_called": False,
                "request_domain": request.get("domain"),
            },
            "raw_result": {},
        }

    def train(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "tolerated", "message": "Sport domain training delegated to impl"}

    def eval(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "tolerated", "message": "Sport domain eval delegated to impl"}

    def predict(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "tolerated", "message": "Sport domain prediction delegated to impl"}

    def build_candidate(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "tolerated"}

    def load_baseline(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"status": "tolerated"}

    def get_contract_context(self, request: dict[str, Any], context: dict[str, Any]) -> ContractContext:
        return ContractContext()

    def get_metric_pack(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def get_local_gain_signals(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def export_fusion_payload(self, request: dict[str, Any], context: dict[str, Any]) -> FusionInputContract:
        return FusionInputContract(
            domain_name=self.domain_name,
            risk_level="stub",
            warning_summary=[f"{self.domain_name} agent is tolerated"],
        )

    def build_fusion_input(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "risk_score": None,
            "risk_level": run_result.get("status"),
            "confidence": None,
            "top_features": [],
            "explanations": [],
            "artifact_ref": None,
            "metric_context": run_result.get("metric_context", {}),
            "validation_summary": run_result.get("validation_summary", []),
            "warning_summary": run_result.get("warning_summary", []),
        }
