from __future__ import annotations

import sys
from pathlib import Path
from typing import Any,Dict

from harness.contracts import (
    ContractContext,
    FusionInputContract,
    PipelineContext,
    RunRecord,
    normalize_decision_bundle,
    normalize_domain_result,
)
from harness.domain_agents.base import BaseDomainAgent


class StudyAgentAdapter(BaseDomainAgent):
    """Adapter that bridges the existing StudyAgent to the BaseDomainAgent interface.

    This is the first concrete domain implementation. It delegates to the
    existing study_domain pipeline while implementing the generic interface.
    """

    def run_domain_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        from study.src.study_agent import StudyAgent

        result = StudyAgent(request=request).run()
        decision_bundle = normalize_decision_bundle(
            final_decision=result.get("final_decision") or "",
            policy_decision=result.get("policy_decision") or result.get("final_decision") or "",
            execution_mode=result.get("execution_mode", ""),
            reason_codes=result.get("harness_v1", {}).get("decision_reason_codes", []),
            decision_stage_reached=result.get("harness_v1", {}).get("decision_stage_reached", ""),
        )
        normalized = normalize_domain_result(
            domain=self.domain_name,
            status=result.get("status", "unknown"),
            summary_metrics=result.get("summary_metrics", {}),
            decision_bundle=decision_bundle,
            metric_context=result.get("harness_v1", {}),
            domain_context=result.get("domain_context", {}),
            domain_audit=result.get("domain_audit", {}),
            warnings=result.get("warnings", []),
            deliverables=result.get("deliverables", {}),
            harness_payload=result.get("harness_v1", {}),
            fusion_input=result.get("fusion_input", {}),
        )

        return {
            "domain_name": self.domain_name,
            "status": normalized.get("status", "unknown"),
            "final_decision": normalized.get("final_decision"),
            "policy_decision": normalized.get("policy_decision"),
            "execution_mode": normalized.get("execution_mode"),
            "decision_stage_reached": normalized.get("decision_stage_reached"),
            "metrics": normalized.get("summary_metrics", {}),
            "metric_context": normalized.get("harness_v1", {}),
            "domain_context": normalized.get("domain_context", {}),
            "domain_audit": normalized.get("domain_audit", {}),
            "validation_summary": result.get("validation_summary", []),
            "warning_summary": normalized.get("warnings", []),
            "artifact_ref": normalized.get("fusion_input", {}).get("artifact_ref"),
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": self.__class__.__name__,
                "execution_path": "orchestrator -> adapter.run_domain_pipeline -> study_agent",
                "domain_pipeline_called": True,
                "request_domain": request.get("domain"),
            },
            "raw_result": normalized,
        }

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        for path in [root_dir, root_dir / "src"]:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    @property
    def domain_name(self) -> str:
        return "study"

    def train(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute study training via the existing pipeline."""
        from study_domain.actions.train_action import TrainAction

        action = TrainAction(self.root_dir)
        pipeline_context = context.get("pipeline_context")
        if pipeline_context:
            result = action.run(pipeline_context)
            return {
                "status": result.status,
                "metrics": result.metrics,
                "artifacts": [a.__dict__ if hasattr(a, "__dict__") else {} for a in result.artifacts],
                "diagnostics": result.diagnostics,
                "message": result.message,
            }
        return {"status": "failed", "message": "No pipeline context provided"}

    def eval(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute study evaluation via the existing pipeline."""
        from study_domain.actions.eval_action import EvalAction

        action = EvalAction(self.root_dir)
        pipeline_context = context.get("pipeline_context")
        if pipeline_context:
            result = action.run(pipeline_context)
            return {
                "status": result.status,
                "metrics": result.metrics,
                "artifacts": [a.__dict__ if hasattr(a, "__dict__") else {} for a in result.artifacts],
                "diagnostics": result.diagnostics,
                "message": result.message,
            }
        return {"status": "failed", "message": "No pipeline context provided"}

    def predict(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute study prediction — delegated to existing study agent inference path."""
        # Study prediction is handled by the legacy study agent path.
        # For harness v1, this is not called directly.
        return {
            "status": "success",
            "metrics": {},
            "message": "Prediction handled by legacy study agent inference path",
        }

    def build_candidate(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Build a study candidate from training outputs."""
        # The study pipeline builds candidates implicitly during train/eval.
        # This method returns the candidate metadata from the eval action.
        eval_result = self.eval(request, context)
        return {
            "status": eval_result.get("status"),
            "metrics": eval_result.get("metrics", {}),
            "diagnostics": eval_result.get("diagnostics", {}),
        }

    def load_baseline(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Load study baselines via the existing BaselineRegistry."""
        from study_domain.registry.baseline_registry import BaselineRegistry

        registry = BaselineRegistry(self.root_dir)
        baselines = registry.current_baselines()
        return {
            "baseline_version_id": baselines.get("active_baseline_id"),
            "anchor_baseline_version_id": baselines.get("anchor_baseline_id"),
            "baseline_metrics": baselines.get("active_baseline", {}).get("metrics", {}),
            "anchor_baseline_metrics": baselines.get("anchor_baseline", {}).get("metrics", {}),
            "frozen_snapshot": baselines.get("active_baseline", {}).get("frozen_snapshot", {}),
        }

    def get_contract_context(self, request: dict[str, Any], context: dict[str, Any]) -> ContractContext:
        """Build a ContractContext from study eval outputs."""
        from study_domain.registry.baseline_registry import BaselineRegistry

        registry = BaselineRegistry(self.root_dir)
        baselines = registry.current_baselines()
        active_baseline = baselines.get("active_baseline", {})
        anchor_baseline = baselines.get("anchor_baseline", {})

        # Read eval report if available
        eval_report_path = self.root_dir / "data" / "dm" / "study_eval_report.json"
        model_config_path = self.root_dir / "data" / "deliverables" / "study" / "model" / "study_model_config.json"

        candidate_metrics = {}
        metric_context = {}
        local_gain_flags = {}

        if eval_report_path.exists():
            import json
            eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
            overall = eval_report.get("overall_metrics", {})
            candidate_metrics = {
                "auc": overall.get("auc"),
                "f1": overall.get("f1"),
                "recall": overall.get("recall"),
                "degraded_ratio": overall.get("degraded_ratio"),
            }

            mode_metrics = {row.get("study_data_mode"): row for row in eval_report.get("mode_metrics", [])}
            subtype_metrics = {row.get("label_subtype"): row for row in eval_report.get("subtype_metrics", [])}

            frozen_snapshot = active_baseline.get("frozen_snapshot", {})
            frozen_eval_report = {}
            if isinstance(frozen_snapshot, dict):
                candidate_eval_ref = frozen_snapshot.get("eval_report", {})
                if isinstance(candidate_eval_ref, dict):
                    frozen_eval_report = candidate_eval_ref
                elif isinstance(candidate_eval_ref, str):
                    eval_ref_path = Path(candidate_eval_ref)
                    if eval_ref_path.exists():
                        frozen_eval_report = json.loads(eval_ref_path.read_text(encoding="utf-8"))

            active_mode = {row.get("study_data_mode"): row for row in frozen_eval_report.get("mode_metrics", [])}
            active_subtype = {row.get("label_subtype"): row for row in frozen_eval_report.get("subtype_metrics", [])}

            local_gain_flags = {
                "single_fail_auc_gain": subtype_metrics.get("single_fail", {}).get("auc", 0) > active_subtype.get("single_fail", {}).get("auc", 0),
                "core_plus_behavior_auc_gain": mode_metrics.get("core_plus_behavior", {}).get("auc", 0) > active_mode.get("core_plus_behavior", {}).get("auc", 0),
                "degraded_ratio_improved": candidate_metrics.get("degraded_ratio", 1) < active_baseline.get("metrics", {}).get("degraded_ratio", 1),
            }

        if model_config_path.exists():
            import json
            import hashlib
            model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
            feature_columns = model_config.get("feature_columns", [])
            feature_contract_hash = hashlib.sha1("\n".join(feature_columns).encode("utf-8")).hexdigest() if feature_columns else None
            metric_context = {
                "feature_contract_hash": feature_contract_hash,
                "label_version": model_config.get("label_name", "LABEL"),
            }

        return ContractContext(
            baseline_version_id=baselines.get("active_baseline_id"),
            anchor_baseline_version_id=baselines.get("anchor_baseline_id"),
            baseline_metrics=active_baseline.get("metrics", {}),
            anchor_baseline_metrics=anchor_baseline.get("metrics", {}),
            candidate_metrics=candidate_metrics,
            architecture_version="study_layered_v2",
            task_scope="study_layered_core_serving",
            label_definition="LABEL",
            eval_split="term_order_holdout",
            local_gain_flags=local_gain_flags,
            metric_context=metric_context,
            baseline_frozen_snapshot=active_baseline.get("frozen_snapshot", {}),
        )

    def get_metric_pack(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Return study metric pack for policy evaluation."""
        contract_ctx = self.get_contract_context(request, context)
        return {
            "candidate_metrics": contract_ctx.candidate_metrics,
            "baseline_metrics": contract_ctx.baseline_metrics,
            "anchor_baseline_metrics": contract_ctx.anchor_baseline_metrics,
            "metric_context": contract_ctx.metric_context,
            "local_gain_flags": contract_ctx.local_gain_flags,
        }

    def get_local_gain_signals(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Return study-specific local gain signals."""
        contract_ctx = self.get_contract_context(request, context)
        return contract_ctx.local_gain_flags

    def export_fusion_payload(self, request: dict[str, Any], context: dict[str, Any]) -> FusionInputContract:
        """Export a normalized FusionInputContract for study domain."""
        contract_ctx = self.get_contract_context(request, context)
        metric_pack = self.get_metric_pack(request, context)

        auc = contract_ctx.candidate_metrics.get("auc")
        risk_level = "low" if auc and auc >= 0.85 else "medium" if auc and auc >= 0.80 else "high"

        return FusionInputContract(
            domain_name="study",
            candidate_version_id=contract_ctx.candidate_version_id,
            risk_score=auc,
            risk_level=risk_level,
            confidence=None,
            top_features=[],
            explanations=[],
            metric_context=metric_pack,
            validation_summary={
                "local_gain_flags": contract_ctx.local_gain_flags,
                "metric_context": contract_ctx.metric_context,
            },
            warning_summary=[],
            artifact_ref={
                "eval_report": str(self.root_dir / "data" / "dm" / "study_eval_report.json"),
                "model_config": str(self.root_dir / "data" / "deliverables" / "study" / "model" / "study_model_config.json"),
            },
        )

    def run_harness_pipeline(self, request: dict[str, Any], context: PipelineContext) -> RunRecord:
        """Run the full study harness pipeline using the existing study_domain pipeline.

        This is the primary entry point that maintains backward compatibility
        with the existing study harness flow.
        """
        from study_domain.study_pipeline import build_study_pipeline

        runner = build_study_pipeline(self.root_dir, request)
        record, record_path = runner.run(request, self.root_dir)
        return record
