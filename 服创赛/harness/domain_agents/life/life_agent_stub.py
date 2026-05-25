from __future__ import annotations

from typing import Any, Dict

from harness.domain_agents.base.base_domain_agent import BaseDomainAgent


class LifeAgentStub(BaseDomainAgent):
    domain_name = "life"

    def train(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def eval(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def predict(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def build_candidate(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def load_baseline(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_contract_context(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_metric_pack(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_local_gain_signals(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def export_fusion_payload(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "risk_score": None,
            "risk_level": None,
            "confidence": None,
            "top_features": [],
            "explanations": [],
            "artifact_ref": None,
            "metric_context": {},
            "validation_summary": [],
            "warning_summary": [f"{self.domain_name} agent is currently a stub"],
        }

    def run_domain_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "status": "not_implemented",
            "final_decision": {
                "decision": "dry_run_only",
                "summary": "life agent is not implemented yet",
            },
            "metrics": {},
            "metric_context": {},
            "validation_summary": [],
            "warning_summary": ["life agent is currently a stub"],
            "artifact_ref": None,
            "raw_result": {},
        }

    def build_fusion_input(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "risk_score": None,
            "risk_level": None,
            "confidence": None,
            "top_features": [],
            "explanations": [],
            "artifact_ref": None,
            "metric_context": run_result.get("metric_context", {}),
            "validation_summary": run_result.get("validation_summary", []),
            "warning_summary": run_result.get("warning_summary", []),
        }