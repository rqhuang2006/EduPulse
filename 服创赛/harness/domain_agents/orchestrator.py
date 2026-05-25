from __future__ import annotations

import time
from typing import Any, Dict, List

from harness.domain_agents.registry import DomainAgentRegistry


class HarnessOrchestrator:
    """Domain-agnostic orchestrator.

    Responsibilities:
    - run one domain through its own pipeline
    - run multiple domains and collect normalized outputs
    - return aggregated multi-domain results
    """

    def __init__(self, registry: DomainAgentRegistry) -> None:
        self.registry = registry

    def run_single_domain(self, domain_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.registry.get(domain_name)
        domain_request = {**request, "domain": domain_name, "search_contract": self._build_search_contract(domain_name, request)}
        run_result = agent.run_domain_pipeline(domain_request)
        fusion_input = agent.build_fusion_input(run_result)

        return {
            "run_type": "single_domain",
            "domain_name": domain_name,
            "domain_result": run_result,
            "fusion_input": fusion_input,
            "final_decision": run_result.get("final_decision", {}),
            "status": run_result.get("status", "unknown"),
            "warnings": run_result.get("warning_summary", []),
            "validations": run_result.get("validation_summary", []),
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": run_result.get("agent_trace", {}).get("adapter_name", agent.__class__.__name__),
                "execution_path": run_result.get("agent_trace", {}).get(
                    "execution_path",
                    "orchestrator -> adapter.run_domain_pipeline",
                ),
                "domain_pipeline_called": run_result.get("agent_trace", {}).get("domain_pipeline_called", True),
                "request_domain": domain_request.get("domain"),
            },
        }

    def run_multi_domain(self, domains: List[str], request: Dict[str, Any]) -> Dict[str, Any]:
        domain_results: Dict[str, Any] = {}
        fusion_inputs: List[Dict[str, Any]] = []
        warnings: List[str] = []
        validations: List[Dict[str, Any]] = []

        # Observability: track timing and order
        domain_timings: Dict[str, float] = {}
        domain_start_order: List[str] = []
        domain_end_order: List[str] = []
        adapter_names: Dict[str, str] = {}
        domain_statuses: Dict[str, str] = {}

        for domain_name in domains:
            agent = self.registry.get(domain_name)
            adapter_class = agent.__class__.__name__
            domain_request = {**request, "domain": domain_name, "search_contract": self._build_search_contract(domain_name, request)}

            # START log
            domain_start_order.append(domain_name)
            start_time = time.time()
            print(f"\n{'='*60}", flush=True)
            print(f"START domain={domain_name}", flush=True)
            print(f"adapter={adapter_class}", flush=True)
            print(f"{'='*60}", flush=True)

            try:
                run_result = agent.run_domain_pipeline(domain_request)
                elapsed = time.time() - start_time
                status = run_result.get("status", "unknown")

                # END log
                domain_end_order.append(domain_name)
                domain_timings[domain_name] = elapsed
                adapter_names[domain_name] = adapter_class
                domain_statuses[domain_name] = status

                print(f"\n{'='*60}", flush=True)
                print(f"END domain={domain_name}", flush=True)
                print(f"status={status}", flush=True)
                print(f"elapsed_seconds={elapsed:.2f}", flush=True)
                print(f"{'='*60}", flush=True)

                fusion_input = agent.build_fusion_input(run_result)

                domain_results[domain_name] = run_result
                fusion_inputs.append(fusion_input)
                warnings.extend(run_result.get("warning_summary", []))
                validations.extend(run_result.get("validation_summary", []))

            except Exception as e:
                elapsed = time.time() - start_time
                domain_timings[domain_name] = elapsed
                adapter_names[domain_name] = adapter_class
                domain_statuses[domain_name] = "failed"
                domain_end_order.append(domain_name)

                # FAIL log
                print(f"\n{'='*60}", flush=True)
                print(f"FAIL domain={domain_name}", flush=True)
                print(f"exception_type={type(e).__name__}", flush=True)
                print(f"exception_message={str(e)}", flush=True)
                print(f"elapsed_seconds={elapsed:.2f}", flush=True)
                print(f"{'='*60}", flush=True)

                # Preserve debug info - do not swallow exception completely
                run_result = {
                    "domain_name": domain_name,
                    "status": "failed",
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "agent_trace": {
                        "invoked_via_harness": True,
                        "adapter_name": adapter_class,
                        "execution_path": "orchestrator -> adapter.run_domain_pipeline",
                        "domain_pipeline_called": True,
                        "request_domain": domain_request.get("domain"),
                    },
                }
                domain_results[domain_name] = run_result

        final_decision = self._aggregate_multi_domain_decision(domain_results)

        # Determine slowest domain
        slowest_domain = max(domain_timings, key=domain_timings.get) if domain_timings else None

        return {
            "run_type": "multi_domain",
            "domain_results": domain_results,
            "fusion_inputs": fusion_inputs,
            "final_decision": final_decision,
            "warnings": warnings,
            "validations": validations,
            "status": "success",
            "system_status": final_decision.get("system_status", "unknown"),
            # Observability fields
            "domain_timings": domain_timings,
            "domain_start_order": domain_start_order,
            "domain_end_order": domain_end_order,
            "adapter_names": adapter_names,
            "domain_statuses": domain_statuses,
            "slowest_domain": slowest_domain,
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": "HarnessOrchestrator",
                "execution_path": "orchestrator -> adapter.run_domain_pipeline",
                "domain_pipeline_called": True,
                "request_domain": request.get("domain"),
            },
        }

    def _aggregate_multi_domain_decision(self, domain_results: Dict[str, Any]) -> Dict[str, Any]:
        statuses = {name: result.get("status") for name, result in domain_results.items()}
        stub_domains = [name for name, status in statuses.items() if status == "not_implemented"]
        ready_domains = [name for name, status in statuses.items() if status not in {"failed", "not_implemented"}]

        if ready_domains and stub_domains:
            system_status = "partial_domain_ready"
            decision = "dry_run_only"
            summary = "Partial multi-domain output is available; stub domains were tolerated."
        elif ready_domains and len(ready_domains) == len(domain_results):
            system_status = "multi_domain_ready"
            decision = "multi_domain_completed"
            summary = "All requested domains completed."
        elif stub_domains and len(stub_domains) == len(domain_results):
            system_status = "dry_run_only"
            decision = "dry_run_only"
            summary = "All requested domains are stubbed."
        else:
            system_status = "completed_with_hold"
            decision = "dry_run_only"
            summary = "At least one domain failed or was held."

        domain_summaries: Dict[str, Any] = {}
        for domain_name, result in domain_results.items():
            hv1 = result.get("harness_v1", {}) if isinstance(result.get("harness_v1"), dict) else {}
            if not hv1 and isinstance(result.get("raw_result"), dict):
                raw_hv1 = result["raw_result"].get("harness_v1")
                if isinstance(raw_hv1, dict):
                    hv1 = raw_hv1
            metrics = hv1.get("metrics", {}) if isinstance(hv1.get("metrics"), dict) else {}
            resolved_decision = (
                hv1.get("decision")
                or hv1.get("final_decision")
                or hv1.get("policy_decision")
                or result.get("final_decision")
                or result.get("policy_decision")
            )
            future_auc = metrics.get("future_window_auc")
            candidate_auc = metrics.get("candidate_auc")
            tautology_risk = result.get("tautology_risk") or hv1.get("tautology_risk") or ""
            suspicious_high_auc = bool(result.get("suspicious_high_auc") or hv1.get("suspicious_high_auc"))
            blocking_reason = str(hv1.get("blocking_reason", ""))
            mainline_validity = bool(hv1.get("mainline_validity", True))
            # Domain-specific selection policy profiles.
            if domain_name == "life":
                if suspicious_high_auc:
                    blocking_reason = blocking_reason or "suspicious_high_auc"
                    mainline_validity = False
                if str(tautology_risk).lower() == "high":
                    blocking_reason = blocking_reason or "high_tautology_risk"
                    mainline_validity = False
            elif domain_name == "sport":
                if str(hv1.get("task_type", "")) != "future_window_prediction":
                    blocking_reason = blocking_reason or "non_future_window_candidate"
                    mainline_validity = False
                if str(tautology_risk).lower() == "high":
                    blocking_reason = blocking_reason or "high_tautology_risk"
                    mainline_validity = False
                if isinstance(future_auc, (float, int)) and float(future_auc) < 0.8:
                    blocking_reason = blocking_reason or "future_window_auc_below_threshold"
                    mainline_validity = False
            domain_summaries[domain_name] = {
                "status": result.get("status"),
                "decision": resolved_decision,
                "decision_semantics": self._decision_semantics(
                    resolved_decision,
                    mainline_validity,
                    blocking_reason,
                ),
                "comparable": bool(hv1.get("comparable", False)),
                "mainline_task_type": hv1.get("task_type", "same_window_classification"),
                "mainline_validity": mainline_validity,
                "blocking_reason": blocking_reason,
                "trusted_mainline": hv1.get("trusted_mainline", {}),
                "mainline_frozen": bool(hv1.get("mainline_frozen", False)),
                "next_optimization_target": hv1.get("next_optimization_target", ""),
            }

        return {
            "decision": decision,
            "summary": summary,
            "system_status": system_status,
            "decision_vocabulary": self._decision_vocabulary(),
            "stub_domains": stub_domains,
            "ready_domains": ready_domains,
            "domains": domain_summaries,
            "domain_decisions": {
                domain_name: (
                    (
                        (result.get("harness_v1", {}) if isinstance(result.get("harness_v1"), dict) else {})
                        or (
                            result.get("raw_result", {}).get("harness_v1", {})
                            if isinstance(result.get("raw_result"), dict) and isinstance(result.get("raw_result", {}).get("harness_v1"), dict)
                            else {}
                        )
                    ).get("decision")
                    or (
                        (result.get("harness_v1", {}) if isinstance(result.get("harness_v1"), dict) else {})
                        or (
                            result.get("raw_result", {}).get("harness_v1", {})
                            if isinstance(result.get("raw_result"), dict) and isinstance(result.get("raw_result", {}).get("harness_v1"), dict)
                            else {}
                        )
                    ).get("final_decision")
                    or (
                        (result.get("harness_v1", {}) if isinstance(result.get("harness_v1"), dict) else {})
                        or (
                            result.get("raw_result", {}).get("harness_v1", {})
                            if isinstance(result.get("raw_result"), dict) and isinstance(result.get("raw_result", {}).get("harness_v1"), dict)
                            else {}
                        )
                    ).get("policy_decision")
                    or result.get("final_decision", {})
                ) for domain_name, result in domain_results.items()
            },
        }

    @staticmethod
    def _decision_vocabulary() -> Dict[str, str]:
        return {
            "promote_candidate": "trusted mainline beats baseline enough to replace the current baseline",
            "keep_baseline": "trusted mainline is valid and comparable, but the current serving choice remains baseline",
            "eligible_for_comparison": "trusted mainline passes authenticity gates and can enter cross-domain comparison",
            "hold_for_review": "candidate exists but authenticity or validity gates failed and review is required",
        }

    @staticmethod
    def _decision_semantics(decision: Any, mainline_validity: bool, blocking_reason: str) -> str:
        decision_text = str(decision or "")
        if decision_text == "promote_candidate":
            return "trusted_and_promoted"
        if decision_text == "keep_baseline":
            return "trusted_but_baseline_kept"
        if decision_text == "eligible_for_comparison":
            return "trusted_and_ready_for_comparison"
        if decision_text == "hold_for_review":
            return "blocked_pending_review"
        if mainline_validity and not blocking_reason:
            return "trusted_but_baseline_kept"
        return "needs_review_or_followup"

    def _build_search_contract(self, domain_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        shared = {
            "allowed_actions": ["label_version", "time_window", "target_population", "structure_version"],
            "forbidden_actions": ["change_output_contract", "change_study_domain_logic"],
        }
        if domain_name == "life":
            return {
                **shared,
                "domain_policy_profile": "life_authenticity_first",
                "optimize_priority": "authenticity",
                "round_goal": "de_fake_then_raise",
                "selection_policy": "filter_suspicious_then_select_best",
            }
        if domain_name == "sport":
            return {
                **shared,
                "domain_policy_profile": "sport_predictability_first",
                "optimize_priority": "predictability",
                "round_goal": "raise_true_future_signal",
                "selection_policy": "filter_non_future_or_tautology_then_select_best",
            }
        return {
            **shared,
            "domain_policy_profile": "default",
            "optimize_priority": "balanced",
            "round_goal": request.get("round_goal", "normal_optimization"),
            "selection_policy": "default",
        }
