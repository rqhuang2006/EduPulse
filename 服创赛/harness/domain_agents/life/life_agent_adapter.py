from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.contracts import ContractContext, FusionInputContract
from harness.domain_agents.base import BaseDomainAgent


class LifeAgentAdapter(BaseDomainAgent):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.domain_root = root_dir / "life"

    @property
    def domain_name(self) -> str:
        return "life"

    def _run_agent(self, request: dict[str, Any]) -> dict[str, Any]:
        from life.src.life_agent import LifeAgent

        return LifeAgent(request=request).run()

    def _read_result(self) -> dict[str, Any]:
        result_path = self.domain_root / "data" / "dm" / "life_agent_result.json"
        if not result_path.exists():
            return {}
        return json.loads(result_path.read_text(encoding="utf-8"))

    def train(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(request)

    def eval(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(request)

    def predict(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(request)

    def build_candidate(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return self._run_agent(request)

    def load_baseline(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        result = self._read_result()
        harness_v1 = result.get("harness_v1", {})
        baseline_summary = harness_v1.get("baseline_summary", {})
        return {
            "baseline_version_id": harness_v1.get("baseline_version_id"),
            "anchor_baseline_version_id": harness_v1.get("anchor_baseline_version_id"),
            "baseline_metrics": baseline_summary.get("metrics", {}),
            "anchor_baseline_metrics": baseline_summary.get("anchor_metrics", {}),
            "frozen_snapshot": baseline_summary.get("snapshot_manifest", {}),
        }

    def get_contract_context(self, request: dict[str, Any], context: dict[str, Any]) -> ContractContext:
        result = self._read_result()
        harness_v1 = result.get("harness_v1", {})
        summary_metrics = result.get("summary_metrics", {})
        return ContractContext(
            baseline_version_id=harness_v1.get("baseline_version_id"),
            anchor_baseline_version_id=harness_v1.get("anchor_baseline_version_id"),
            candidate_version_id=harness_v1.get("candidate_version_id"),
            baseline_metrics=harness_v1.get("baseline_summary", {}).get("metrics", {}),
            anchor_baseline_metrics=harness_v1.get("baseline_summary", {}).get("anchor_metrics", {}),
            candidate_metrics=summary_metrics,
            architecture_version=harness_v1.get("contract_context", {}).get("architecture_version"),
            task_scope=harness_v1.get("task_scope"),
            label_definition=harness_v1.get("label_definition"),
            eval_split=harness_v1.get("contract_context", {}).get("eval_split"),
            local_gain_flags=harness_v1.get("local_gain_signals", {}),
            metric_context=harness_v1.get("metric_context", {}),
            baseline_frozen_snapshot=harness_v1.get("baseline_summary", {}).get("snapshot_manifest", {}),
            candidate_frozen_snapshot=harness_v1.get("candidate_snapshot_manifest", {}),
        )

    def get_metric_pack(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        contract = self.get_contract_context(request, context)
        return {
            "candidate_metrics": contract.candidate_metrics,
            "baseline_metrics": contract.baseline_metrics,
            "anchor_baseline_metrics": contract.anchor_baseline_metrics,
            "metric_context": contract.metric_context,
        }

    def get_local_gain_signals(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return self.get_contract_context(request, context).local_gain_flags

    def export_fusion_payload(self, request: dict[str, Any], context: dict[str, Any]) -> FusionInputContract:
        result = self._read_result()
        harness_v1 = result.get("harness_v1", {})
        fusion_input = result.get("fusion_input", {})
        return FusionInputContract(
            domain_name="life",
            candidate_version_id=harness_v1.get("candidate_version_id"),
            risk_score=fusion_input.get("risk_score"),
            risk_level=fusion_input.get("risk_level"),
            confidence=fusion_input.get("confidence"),
            top_features=fusion_input.get("top_features", []),
            explanations=fusion_input.get("explanations", []),
            quality_metrics=fusion_input.get("quality_metrics", {}),
            metric_context=harness_v1.get("metric_context", {}),
            validation_summary=fusion_input.get("validation_summary", {}),
            warning_summary=fusion_input.get("warning_summary", result.get("warnings", [])),
            artifact_ref=fusion_input.get("artifact_ref", {}),
            raw_payload=fusion_input.get("raw_payload", {}),
        )

    def run_domain_pipeline(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self._run_agent(request)
        harness_v1 = result.get("harness_v1", {})
        return {
            "domain_name": self.domain_name,
            "status": "success" if result.get("status") != "failed" else "failed",
            "system_status": "multi_domain_ready" if result.get("status") != "failed" else "completed_with_hold",
            "final_decision": result.get("final_decision"),
            "policy_decision": result.get("policy_decision"),
            "reason": ",".join(result.get("reason_codes", [])) if isinstance(result.get("reason_codes"), list) else result.get("reason_codes"),
            "execution_mode": result.get("execution_mode"),
            "decision_stage_reached": result.get("decision_stage_reached"),
            "metrics": result.get("summary_metrics", {}),
            "metric_context": harness_v1.get("metric_context", {}),
            "domain_context": result.get("domain_context", {}),
            "domain_audit": result.get("domain_audit", {}),
            "validation_summary": result.get("fusion_input", {}).get("validation_summary", {}),
            "warning_summary": result.get("warnings", []),
            "artifact_ref": result.get("fusion_input", {}).get("artifact_ref", {}),
            "harness_v1": harness_v1,
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": self.__class__.__name__,
                "execution_path": "orchestrator -> adapter.run_domain_pipeline -> life_agent",
                "domain_pipeline_called": True,
                "request_domain": request.get("domain"),
            },
            "raw_result": result,
        }
