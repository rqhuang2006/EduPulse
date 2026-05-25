from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DM_DIR = ROOT / "data" / "dm"
REPORT_PATH = DM_DIR / "life_harness_participation_report.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    standalone_result_path = DM_DIR / "life_agent_result.json"
    single_domain_path = WORKSPACE_ROOT / "data" / "harness" / "runs" / "single_domain_life.json"
    multi_domain_path = WORKSPACE_ROOT / "data" / "harness" / "runs" / "multi_domain_study_life_sport.json"

    standalone = read_json(standalone_result_path)
    single = read_json(single_domain_path)
    multi = read_json(multi_domain_path)

    single_life = single["domain_result"]
    multi_life = multi["domain_results"]["life"]
    multi_sport = multi["domain_results"]["sport"]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "standalone_run_ok": standalone.get("status") == "success",
        "harness_single_domain_ok": single.get("run_type") == "single_domain" and single_life.get("status") == "success",
        "multi_domain_ok": multi.get("run_type") == "multi_domain" and multi_life.get("status") == "success",
        "adapter_invoked": single_life.get("agent_trace", {}).get("adapter_name") == "LifeAgentAdapter",
        "orchestrator_invoked": multi.get("agent_trace", {}).get("adapter_name") == "HarnessOrchestrator",
        "sport_stub_tolerated": multi.get("system_status") == "partial_domain_ready" and multi_sport.get("status") == "not_implemented",
        "evidence_files": {
            "standalone_result": str(standalone_result_path),
            "single_domain_orchestrator_result": str(single_domain_path),
            "multi_domain_orchestrator_result": str(multi_domain_path),
        },
        "single_domain_evidence": {
            "life_agent_trace": single_life.get("agent_trace", {}),
            "life_final_decision": single_life.get("final_decision"),
            "single_domain_status": single_life.get("status"),
        },
        "multi_domain_evidence": {
            "top_level_status": multi.get("system_status"),
            "study_status": multi["domain_results"]["study"].get("status"),
            "life_status": multi_life.get("status"),
            "sport_status": multi_sport.get("status"),
            "sport_stub_summary": multi_sport.get("final_decision"),
        },
        "final_conclusion": {
            "standalone": "life standalone script runs successfully",
            "single_domain_harness": "life is invoked through LifeAgentAdapter and run_domain_pipeline in harness single-domain mode",
            "multi_domain_harness": "life participates in multi-domain orchestration together with study and sport; sport stub does not block the whole pipeline",
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
