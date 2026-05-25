from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap(root: Path) -> None:
    for path in [root / ".deps3", root]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the harness multi-domain orchestrator.")
    parser.add_argument("--request", required=True, help="Path to a study/life-style request JSON.")
    parser.add_argument("--domains", nargs="*", default=["study", "life", "sport"], help="Domains to execute in order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    _bootstrap(root)

    from harness.domain_agents.life import LifeAgentAdapter
    from harness.domain_agents.orchestrator import HarnessOrchestrator
    from harness.domain_agents.registry import DomainAgentRegistry
    from harness.domain_agents.sport import SportAgentAdapter
    from harness.domain_agents.study import StudyAgentAdapter

    request_path = Path(args.request)
    if not request_path.is_absolute():
        request_path = root / request_path
    request = json.loads(request_path.read_text(encoding="utf-8"))

    registry = DomainAgentRegistry()
    registry.register(StudyAgentAdapter(root / "study"))
    registry.register(LifeAgentAdapter(root))
    registry.register(SportAgentAdapter(root))
    orchestrator = HarnessOrchestrator(registry)

    if len(args.domains) == 1:
        domain_name = args.domains[0]
        orchestrator_result = orchestrator.run_single_domain(domain_name, request)
    else:
        orchestrator_result = orchestrator.run_multi_domain(args.domains, request)

    output_dir = root / "data" / "harness" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_dir = root / "data" / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    run_name = "single_domain" if len(args.domains) == 1 else "multi_domain"
    output_path = output_dir / f"{run_name}_{'_'.join(args.domains)}.json"
    output_path.write_text(json.dumps(orchestrator_result, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "run_record_path": str(output_path),
        "run_type": orchestrator_result.get("run_type"),
        "system_status": orchestrator_result.get("system_status"),
        "final_decision": orchestrator_result.get("final_decision"),
        "domain_results": orchestrator_result.get("domain_results") or {
            orchestrator_result.get("domain_name"): orchestrator_result.get("domain_result")
        },
        "agent_trace": orchestrator_result.get("agent_trace", {}),
    }
    if len(args.domains) > 1:
        (harness_dir / "multi_domain_smoke_result.json").write_text(
            json.dumps(orchestrator_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        domain_results = orchestrator_result.get("domain_results") or {}

        # Determine adapter active status
        study_adapter_active = bool(
            domain_results.get("study", {}).get("agent_trace", {}).get("adapter_name") == "StudyAgentAdapter"
        )
        life_adapter_active = bool(
            domain_results.get("life", {}).get("agent_trace", {}).get("adapter_name") == "LifeAgentAdapter"
        )
        sport_adapter_active = bool(
            domain_results.get("sport", {}).get("agent_trace", {}).get("adapter_name") == "SportAgentAdapter"
        )

        debug_summary = {
            "run_record_path": str(output_path),
            "requested_domains": args.domains,
            "system_status": orchestrator_result.get("system_status"),
            "final_decision": orchestrator_result.get("final_decision"),
            "domain_statuses": {
                name: {
                    "status": result.get("status"),
                    "final_decision": result.get("final_decision"),
                    "adapter_name": result.get("agent_trace", {}).get("adapter_name"),
                }
                for name, result in domain_results.items()
            },
            # New observability fields
            "domain_timings": orchestrator_result.get("domain_timings", {}),
            "domain_start_order": orchestrator_result.get("domain_start_order", []),
            "domain_end_order": orchestrator_result.get("domain_end_order", []),
            "adapter_names": orchestrator_result.get("adapter_names", {}),
            "slowest_domain": orchestrator_result.get("slowest_domain"),
            "sport_adapter_active": sport_adapter_active,
            "study_adapter_active": study_adapter_active,
            "life_adapter_active": life_adapter_active,
            "sport_stub_tolerated": bool(
                domain_results.get("sport", {}).get("status") == "not_implemented"
                and orchestrator_result.get("system_status") in {"partial_domain_ready", "completed_with_hold", "dry_run_only"}
            ),
        }
        (harness_dir / "multi_domain_debug_summary.json").write_text(
            json.dumps(debug_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
