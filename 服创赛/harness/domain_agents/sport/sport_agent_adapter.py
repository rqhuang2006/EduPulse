from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from harness.contracts import ContractContext, FusionInputContract, normalize_decision_bundle, normalize_domain_result
from harness.domain_agents.base import BaseDomainAgent


class SportAgentAdapter(BaseDomainAgent):
    """Adapter that bridges the real SportAgent to the BaseDomainAgent interface.

    This is the *real* adapter – not a stub.  It calls
    ``sport.src.sport_agent.SportAgent`` and normalizes the raw result
    into the harness multi-domain contract.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.domain_root = root_dir / "sport"

    # ------------------------------------------------------------------
    # domain_name
    # ------------------------------------------------------------------
    @property
    def domain_name(self) -> str:
        return "sport"

    # ------------------------------------------------------------------
    # Core pipeline – the single entry-point the orchestrator calls
    # ------------------------------------------------------------------
    def run_domain_pipeline(self, request: dict[str, Any]) -> dict[str, Any]:
        import warnings as _w
        from sport.src.sport_agent import SportAgent

        print(f"[{time.time()}] sport_adapter.run_domain_pipeline ENTER", flush=True)
        # Build a request dict that the raw SportAgent expects.
        print(f"[{time.time()}] sport_adapter._build_sport_request BEFORE", flush=True)
        sport_request = self._build_sport_request(request)
        print(f"[{time.time()}] sport_adapter._build_sport_request AFTER", flush=True)

        try:
            print(f"[{time.time()}] sport_adapter.SportAgent INIT BEFORE", flush=True)
            agent = SportAgent(request=sport_request)
            print(f"[{time.time()}] sport_adapter.SportAgent INIT AFTER", flush=True)
            raw = agent.run()
        except Exception as exc:
            _w.warn(f"SportAgent execution failed ({type(exc).__name__}): {exc}")
            print(f"[{time.time()}] sport_adapter.run_domain_pipeline EXIT (FAILED)", flush=True)
            return {
                "domain_name": self.domain_name,
                "status": "failed",
                "final_decision": "not_available",
                "policy_decision": "not_available",
                "execution_mode": request.get("run_mode", "infer"),
                "decision_stage_reached": "agent_error",
                "metrics": {},
                "metric_context": {},
                "domain_context": {},
                "domain_audit": {},
                "validation_summary": [],
                "warning_summary": [f"SportAgent execution failed: {type(exc).__name__}: {exc}"],
                "artifact_ref": None,
                "agent_trace": {
                    "invoked_via_harness": True,
                    "adapter_name": self.__class__.__name__,
                    "execution_path": "orchestrator -> adapter.run_domain_pipeline -> sport_agent (FAILED)",
                    "domain_pipeline_called": True,
                    "request_domain": request.get("domain"),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
                "raw_result": {},
            }

        decision_bundle = normalize_decision_bundle(
            final_decision=raw.get("final_decision") or raw.get("status", "unknown"),
            policy_decision=raw.get("policy_decision") or raw.get("status", "unknown"),
            execution_mode=raw.get("execution_mode", request.get("run_mode", "infer")),
            reason_codes=raw.get("reason_codes", []),
            decision_stage_reached=raw.get("decision_stage_reached", ""),
        )
        normalized = normalize_domain_result(
            domain=self.domain_name,
            status=raw.get("status", "unknown"),
            summary_metrics=self._extract_summary_metrics(raw),
            decision_bundle=decision_bundle,
            metric_context=raw.get("metric_context", {}),
            domain_context=raw.get("domain_context", {}),
            domain_audit=raw.get("domain_audit", {}),
            warnings=raw.get("warnings", []),
            deliverables=raw.get("deliverables", {}),
            harness_payload=raw.get("harness_v1", {}),
            fusion_input=raw.get("fusion_input", {}),
        )

        output_status = normalized.get("status", "unknown")
        if raw.get("decision") and output_status == "success":
            output_status = "completed"

        output = {
            "domain_name": self.domain_name,
            "status": output_status,
            "final_decision": normalized.get("final_decision"),
            "policy_decision": normalized.get("policy_decision"),
            "decision": raw.get("decision", normalized.get("final_decision")),
            "comparable": bool((raw.get("harness_v1") or {}).get("comparable", True)),
            "execution_mode": normalized.get("execution_mode"),
            "decision_stage_reached": normalized.get("decision_stage_reached"),
            "metrics": normalized.get("summary_metrics", {}),
            "metric_context": normalized.get("harness_v1", {}),
            "domain_context": normalized.get("domain_context", {}),
            "domain_audit": normalized.get("domain_audit", {}),
            "validation_summary": raw.get("validation_summary", []),
            "warning_summary": normalized.get("warnings", []),
            "artifact_ref": normalized.get("fusion_input", {}).get("artifact_ref"),
            "harness_v1": normalized.get("harness_v1", {}),
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": self.__class__.__name__,
                "execution_path": "orchestrator -> adapter.run_domain_pipeline -> sport_agent",
                "domain_pipeline_called": True,
                "request_domain": request.get("domain"),
            },
            "raw_result": normalized,
        }
        print(f"[{time.time()}] sport_adapter.run_domain_pipeline EXIT", flush=True)
        return output

    # ------------------------------------------------------------------
    # ABC method implementations (minimal – delegated to SportAgent)
    # ------------------------------------------------------------------
    def train(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from sport.src.sport_agent import SportAgent

        sport_request = self._build_sport_request({**request, "run_mode": "train"})
        raw = SportAgent(request=sport_request).run()
        return {"status": raw.get("status", "unknown"), "metrics": self._extract_summary_metrics(raw)}

    def eval(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from sport.src.sport_agent import SportAgent

        sport_request = self._build_sport_request({**request, "run_mode": "infer"})
        raw = SportAgent(request=sport_request).run()
        return {"status": raw.get("status", "unknown"), "metrics": self._extract_summary_metrics(raw)}

    def predict(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from sport.src.sport_agent import SportAgent

        sport_request = self._build_sport_request({**request, "run_mode": "infer"})
        raw = SportAgent(request=sport_request).run()
        return {"status": raw.get("status", "unknown"), "metrics": self._extract_summary_metrics(raw)}

    def build_candidate(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raw = self._run_sport(request)
        return {"status": raw.get("status", "unknown"), "metrics": self._extract_summary_metrics(raw)}

    def load_baseline(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "baseline_version_id": None,
            "anchor_baseline_version_id": None,
            "baseline_metrics": {},
            "anchor_baseline_metrics": {},
            "frozen_snapshot": {},
        }

    def get_contract_context(self, request: dict[str, Any], context: dict[str, Any]) -> ContractContext:
        return ContractContext(
            architecture_version="sport_layered_v1",
            task_scope="sport_体测_体育课",
            label_definition="zf_score",
            eval_split="term_order_holdout",
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
        contract = self.get_contract_context(request, context)
        metric_pack = self.get_metric_pack(request, context)
        return FusionInputContract(
            domain_name="sport",
            candidate_version_id=contract.candidate_version_id,
            risk_score=None,
            risk_level="medium",
            confidence=None,
            top_features=[],
            explanations=[],
            quality_metrics={},
            metric_context=metric_pack,
            validation_summary={"fusion_payload_semantics_ok": True, "risk_score_source": "sport_adapter", "quality_metric_source": "sport_adapter"},
            warning_summary=[],
            artifact_ref={},
            raw_payload={},
        )

    def build_fusion_input(self, run_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "risk_score": run_result.get("risk_score"),
            "risk_level": run_result.get("risk_level", run_result.get("status")),
            "confidence": run_result.get("confidence"),
            "top_features": run_result.get("top_features", []),
            "explanations": run_result.get("explanations", []),
            "artifact_ref": run_result.get("artifact_ref"),
            "metric_context": run_result.get("metric_context", {}),
            "validation_summary": run_result.get("validation_summary", []),
            "warning_summary": run_result.get("warning_summary", []),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_sport(self, request: dict[str, Any]) -> dict[str, Any]:
        from sport.src.sport_agent import SportAgent

        sport_request = self._build_sport_request(request)
        return SportAgent(request=sport_request).run()

    def _build_sport_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Merge the incoming harness request with the sport request template.

        The SportAgent expects ``input_paths`` and ``run_mode`` at minimum.
        If the caller already supplied them, keep them.  Otherwise fall back
        to the infer template shipped with the sport package.
        """
        # Normalize run_mode: SportAgent only accepts "train" or "infer".
        # For harness "review" mode (and any other unknown value), default to "infer".
        raw_mode = request.get("run_mode", "infer")
        sport_run_mode = raw_mode if raw_mode in {"train", "infer"} else "infer"

        # If the request already carries sport-specific input_paths (i.e. contains
        # keys that only the sport template defines), it is usable directly.
        _SPORT_REQUIRED_PATH_KEYS = {"feature_dataset", "model_regression", "model_classification"}
        if (request.get("input_paths")
                and _SPORT_REQUIRED_PATH_KEYS.issubset(request["input_paths"].keys())):
            return {**request, "run_mode": sport_run_mode, "domain": "sport"}

        # Load the sport infer template so we get correct input_paths and domain.
        template_path = self.domain_root / "input" / "sport_agent_request.infer.json"
        if template_path.exists():
            template = json.loads(template_path.read_text(encoding="utf-8"))
            # Overlay only harness-level fields that are safe to pass through.
            # Do NOT overlay domain, input_paths, or run_mode from the harness
            # request – those belong to the originating domain (e.g. study).
            _PASSTHROUGH_KEYS = {
                "request_id", "term_id", "feature_version", "model_version",
                "enable_fallback", "enable_quality_check", "enable_explanation",
                "fallback_to_mock", "llm_enable", "llm_required",
                "serving_version", "search_contract",
            }
            for k, v in request.items():
                if k in _PASSTHROUGH_KEYS:
                    template[k] = v
            template["run_mode"] = sport_run_mode
            return template

        # Last resort – build a minimal sport request; SportAgent.validate_input
        # will raise with a clear error if required fields are still missing.
        return {**request, "run_mode": sport_run_mode, "domain": "sport"}

    @staticmethod
    def _extract_summary_metrics(raw: dict[str, Any]) -> dict[str, Any]:
        """Pull a small, normalized metric dict from the raw SportAgent result."""
        summary_metrics = raw.get("summary_metrics")
        if isinstance(summary_metrics, dict) and summary_metrics:
            return summary_metrics

        harness_metrics = (raw.get("harness_v1") or {}).get("metrics")
        if isinstance(harness_metrics, dict) and harness_metrics:
            return harness_metrics

        quality = raw.get("quality", {})
        if isinstance(quality, dict):
            return {
                "rows": quality.get("rows", raw.get("rows")),
                "status": quality.get("status", raw.get("status")),
            }
        return {"rows": raw.get("rows"), "status": raw.get("status")}
