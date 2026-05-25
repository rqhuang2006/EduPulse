"""Run harness tests and append results to the life eval report."""
import sys, os, traceback, json
from pathlib import Path

sys.path.insert(0, r"c:\Users\39527\Documents\Playground\服创赛")
os.chdir(r"c:\Users\39527\Documents\Playground\服创赛")

results = {"test_run": True, "tests": []}

def record(name, passed, detail=""):
    results["tests"].append({"name": name, "passed": passed, "detail": detail})

# Test 1: sport stub
try:
    from harness.domain_agents.sport import SportAgentStub
    sport = SportAgentStub()
    result = sport.run_domain_pipeline({})
    assert result["status"] == "stub"
    assert result["system_status"] == "stub"
    assert result["domain_name"] == "sport"
    record("sport_stub", True)
except Exception as e:
    record("sport_stub", False, str(e))

# Test 2: fusion valid
try:
    from harness.contracts.fusion_input import validate_fusion_input, FusionInputContract
    valid = validate_fusion_input(FusionInputContract(domain_name="life", risk_score=0.78, risk_level="medium"))
    assert valid["ok"] == True
    record("fusion_valid", True)
except Exception as e:
    record("fusion_valid", False, str(e))

# Test 3: fusion invalid
try:
    invalid = validate_fusion_input({"domain_name": "", "risk_level": "weird"})
    assert invalid["ok"] == False
    assert "domain_name_missing" in invalid["errors"]
    record("fusion_invalid", True)
except Exception as e:
    record("fusion_invalid", False, str(e))

# Test 4: snapshot contract
try:
    import tempfile
    from harness.registry.snapshot_contract import ensure_snapshot_contract, REQUIRED_SNAPSHOT_FILES
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = ensure_snapshot_contract(Path(tmpdir) / "t1", version_id="t1", domain="test")
        assert len(manifest) == len(REQUIRED_SNAPSHOT_FILES)
        for fname, fpath in manifest.items():
            content = Path(fpath).read_text(encoding="utf-8").strip()
            assert content != "{}"
    record("snapshot_contract", True)
except Exception as e:
    record("snapshot_contract", False, str(e))

# Test 5: orchestrator partial
try:
    from harness.domain_agents.base import BaseDomainAgent
    from harness.domain_agents.orchestrator import HarnessOrchestrator
    from harness.contracts import ContractContext
    from harness.recorder.run_recorder import RunRecorder
    from harness.registry.artifact_registry import ArtifactRegistry

    class DReady(BaseDomainAgent):
        @property
        def domain_name(self): return "study"
        def train(self,r,c): return {}
        def eval(self,r,c): return {}
        def predict(self,r,c): return {}
        def build_candidate(self,r,c): return {}
        def load_baseline(self,r,c): return {}
        def get_contract_context(self,r,c): return ContractContext()
        def get_metric_pack(self,r,c): return {}
        def get_local_gain_signals(self,r,c): return {}
        def export_fusion_payload(self,r,c):
            return FusionInputContract(domain_name="study", risk_score=0.8, risk_level="low")
        def run_domain_pipeline(self,r):
            return {"domain_name":"study","status":"success","system_status":"multi_domain_ready",
                    "final_decision":"promotion_recommended","policy_decision":"promotion_recommended",
                    "execution_mode":"dry_run","decision_stage_reached":"approval_release_stage",
                    "metric_context":{},"domain_context":{},"domain_audit":{},"warning_summary":[]}

    class DStub(DReady):
        @property
        def domain_name(self): return "sport"
        def export_fusion_payload(self,r,c):
            return FusionInputContract(domain_name="sport", risk_level="stub", warning_summary=["stub"])
        def run_domain_pipeline(self,r):
            raise NotImplementedError

    root = Path(".") / ".tmp" / "unittest_q4"
    root.mkdir(parents=True, exist_ok=True)
    orch = HarnessOrchestrator(agents=[DReady(), DStub()], recorder=RunRecorder(root),
                               artifact_registry=ArtifactRegistry(root), root_dir=root)
    rec, _, _ = orch.run_multi_domain({"run_id":"q4"}, ["study","sport"])
    assert rec.system_status == "partial_domain_ready"
    assert rec.final_decision.decision == "dry_run_only"
    record("orch_partial", True)
except Exception as e:
    record("orch_partial", False, str(e))

# Test 6: orchestrator all ready
try:
    class DReady2(DReady):
        @property
        def domain_name(self): return "life"
        def export_fusion_payload(self,r,c):
            return FusionInputContract(domain_name="life", risk_score=0.8, risk_level="low")
        def run_domain_pipeline(self,r):
            return {"domain_name":"life","status":"success","system_status":"multi_domain_ready",
                    "final_decision":"promotion_recommended","policy_decision":"promotion_recommended",
                    "execution_mode":"dry_run","decision_stage_reached":"approval_release_stage",
                    "metric_context":{},"domain_context":{},"domain_audit":{},"warning_summary":[]}

    root2 = Path(".") / ".tmp" / "unittest_q4b"
    root2.mkdir(parents=True, exist_ok=True)
    orch2 = HarnessOrchestrator(agents=[DReady(), DReady2()], recorder=RunRecorder(root2),
                                artifact_registry=ArtifactRegistry(root2), root_dir=root2)
    rec2, _, _ = orch2.run_multi_domain({"run_id":"q4b"}, ["study","life"])
    assert rec2.system_status == "multi_domain_ready"
    assert rec2.final_decision.decision == "promotion_recommended"
    record("orch_all_ready", True)
except Exception as e:
    record("orch_all_ready", False, str(e))

# Test 7: orchestrator fusion invalid
try:
    class DBadFusion(DReady):
        @property
        def domain_name(self): return "bad"
        def export_fusion_payload(self,r,c):
            return FusionInputContract(domain_name="", risk_level="unknown")
        def run_domain_pipeline(self,r):
            return {"domain_name":"bad","status":"success","system_status":"multi_domain_ready",
                    "final_decision":"promotion_recommended","policy_decision":"promotion_recommended",
                    "execution_mode":"dry_run","decision_stage_reached":"approval_release_stage",
                    "metric_context":{},"domain_context":{},"domain_audit":{},"warning_summary":[]}

    root3 = Path(".") / ".tmp" / "unittest_q4c"
    root3.mkdir(parents=True, exist_ok=True)
    orch3 = HarnessOrchestrator(agents=[DReady(), DBadFusion()], recorder=RunRecorder(root3),
                                artifact_registry=ArtifactRegistry(root3), root_dir=root3)
    rec3, _, _ = orch3.run_multi_domain({"run_id":"q4c"}, ["study","bad"])
    assert rec3.system_status == "completed_with_hold"
    assert "fusion_payload_invalid" in rec3.final_decision.reason_codes
    record("orch_fusion_invalid", True)
except Exception as e:
    record("orch_fusion_invalid", False, str(e))

# Summary
passed = sum(1 for t in results["tests"] if t["passed"])
failed = sum(1 for t in results["tests"] if not t["passed"])
results["passed"] = passed
results["failed"] = failed
results["total"] = passed + failed

# Write to eval report path (known to be writable)
out_path = Path(r"c:\Users\39527\Documents\Playground\服创赛\life\data\dm\life_eval_report.json")
existing = {}
if out_path.exists():
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except:
        pass
existing["harness_test_results"] = results
out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
