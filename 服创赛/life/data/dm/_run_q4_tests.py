import sys, os, traceback
sys.path.insert(0, r"c:\Users\39527\Documents\Playground\服创赛")
os.chdir(r"c:\Users\39527\Documents\Playground\服创赛")

result_path = r"c:\Users\39527\Documents\Playground\服创赛\life\data\dm\_test_results.txt"
lines = []

try:
    lines.append(f"Python: {sys.version}")
    lines.append(f"Exe: {sys.executable}")
    lines.append(f"CWD: {os.getcwd()}")
except Exception as e:
    lines.append(f"Env info failed: {e}")

# Sport stub test
try:
    from harness.domain_agents.sport import SportAgentStub
    sport = SportAgentStub()
    result = sport.run_domain_pipeline({})
    assert result["status"] == "stub"
    assert result["system_status"] == "stub"
    assert result["domain_name"] == "sport"
    lines.append("TEST sport_stub: PASS")
except Exception as e:
    lines.append(f"TEST sport_stub: FAIL - {traceback.format_exc()}")

# Fusion valid
try:
    from harness.contracts.fusion_input import validate_fusion_input
    from harness.contracts import FusionInputContract
    valid = validate_fusion_input(FusionInputContract(domain_name="life", risk_score=0.78, risk_level="medium"))
    assert valid["ok"] == True
    lines.append("TEST fusion_valid: PASS")
except Exception as e:
    lines.append(f"TEST fusion_valid: FAIL - {traceback.format_exc()}")

# Fusion invalid
try:
    invalid = validate_fusion_input({"domain_name": "", "risk_level": "weird"})
    assert invalid["ok"] == False
    assert "domain_name_missing" in invalid["errors"]
    lines.append("TEST fusion_invalid: PASS")
except Exception as e:
    lines.append(f"TEST fusion_invalid: FAIL - {traceback.format_exc()}")

# Snapshot contract
try:
    import tempfile
    from pathlib import Path
    from harness.registry.snapshot_contract import ensure_snapshot_contract, REQUIRED_SNAPSHOT_FILES
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = ensure_snapshot_contract(Path(tmpdir) / "t1", version_id="t1", domain="test")
        assert len(manifest) == len(REQUIRED_SNAPSHOT_FILES)
        for fname, fpath in manifest.items():
            content = Path(fpath).read_text(encoding="utf-8").strip()
            assert content != "{}"
    lines.append("TEST snapshot_contract: PASS")
except Exception as e:
    lines.append(f"TEST snapshot_contract: FAIL - {traceback.format_exc()}")

# Orchestrator: partial
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

    class DStub(BaseDomainAgent):
        @property
        def domain_name(self): return "sport"
        def train(self,r,c): return {}
        def eval(self,r,c): return {}
        def predict(self,r,c): return {}
        def build_candidate(self,r,c): return {}
        def load_baseline(self,r,c): return {}
        def get_contract_context(self,r,c): return ContractContext()
        def get_metric_pack(self,r,c): return {}
        def get_local_gain_signals(self,r,c): return {}
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
    lines.append("TEST orch_partial: PASS")
except Exception as e:
    lines.append(f"TEST orch_partial: FAIL - {traceback.format_exc()}")

# Orchestrator: all ready
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
    lines.append("TEST orch_all_ready: PASS")
except Exception as e:
    lines.append(f"TEST orch_all_ready: FAIL - {traceback.format_exc()}")

# Orchestrator: fusion invalid
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
    lines.append("TEST orch_fusion_invalid: PASS")
except Exception as e:
    lines.append(f"TEST orch_fusion_invalid: FAIL - {traceback.format_exc()}")

# Summary
p = sum(1 for l in lines if l.startswith("TEST") and "PASS" in l)
f = sum(1 for l in lines if l.startswith("TEST") and "FAIL" in l)
lines.append(f"\nSUMMARY: {p} passed, {f} failed, {p+f} total")

with open(result_path, "w", encoding="utf-8") as out:
    out.write("\n".join(lines))
