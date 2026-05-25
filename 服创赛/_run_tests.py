import unittest
import sys
import os
import traceback as tb_module

os.chdir(r"c:\Users\39527\Documents\Playground\服创赛")
sys.path.insert(0, ".")

results_lines = []

try:
    from harness.contracts import ContractContext, FusionInputContract
    from harness.contracts.fusion_input import validate_fusion_input
    from harness.domain_agents.base import BaseDomainAgent
    from harness.domain_agents.orchestrator import HarnessOrchestrator
    from harness.domain_agents.sport import SportAgentStub
    from harness.recorder.run_recorder import RunRecorder
    from harness.registry.artifact_registry import ArtifactRegistry
    from harness.registry.snapshot_contract import ensure_snapshot_contract, REQUIRED_SNAPSHOT_FILES
    results_lines.append("IMPORT: OK - all modules imported")
except Exception as e:
    results_lines.append(f"IMPORT: FAIL - {e}")
    with open("_test_result.txt", "w") as f:
        f.write("\n".join(results_lines))
    sys.exit(1)

# Test 1: Sport stub returns structured result
try:
    sport = SportAgentStub()
    result = sport.run_domain_pipeline({})
    assert result["status"] == "stub"
    assert result["system_status"] == "stub"
    assert result["domain_name"] == "sport"
    results_lines.append("TEST sport_stub: PASS")
except Exception as e:
    results_lines.append(f"TEST sport_stub: FAIL - {e}")

# Test 2: Fusion validator - valid
try:
    valid = validate_fusion_input(FusionInputContract(domain_name="life", risk_score=0.78, risk_level="medium"))
    assert valid["ok"] == True
    results_lines.append("TEST fusion_valid: PASS")
except Exception as e:
    results_lines.append(f"TEST fusion_valid: FAIL - {e}")

# Test 3: Fusion validator - invalid domain_name
try:
    invalid = validate_fusion_input({"domain_name": "", "risk_level": "weird"})
    assert invalid["ok"] == False
    assert "domain_name_missing" in invalid["errors"]
    results_lines.append("TEST fusion_invalid_domain: PASS")
except Exception as e:
    results_lines.append(f"TEST fusion_invalid_domain: FAIL - {e}")

# Test 4: Fusion validator - invalid risk_score
try:
    invalid = validate_fusion_input({"domain_name": "x", "risk_score": "bad", "risk_level": "low"})
    assert invalid["ok"] == False
    assert "risk_score_not_numeric" in invalid["errors"]
    results_lines.append("TEST fusion_invalid_risk_score: PASS")
except Exception as e:
    results_lines.append(f"TEST fusion_invalid_risk_score: FAIL - {e}")

# Test 5: Snapshot contract
try:
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        manifest = ensure_snapshot_contract(Path(tmpdir) / "test_v1", version_id="test_v1", domain="test")
        assert len(manifest) == len(REQUIRED_SNAPSHOT_FILES)
        for fname, fpath in manifest.items():
            content = Path(fpath).read_text(encoding="utf-8").strip()
            assert content != "{}", f"{fname} is empty"
    results_lines.append("TEST snapshot_contract: PASS")
except Exception as e:
    results_lines.append(f"TEST snapshot_contract: FAIL - {e}")

# Test 6: Multi-domain orchestrator with dummy agents
try:
    from pathlib import Path

    class DummyReady(BaseDomainAgent):
        @property
        def domain_name(self): return "study"
        def train(self, r, c): return {}
        def eval(self, r, c): return {}
        def predict(self, r, c): return {}
        def build_candidate(self, r, c): return {}
        def load_baseline(self, r, c): return {}
        def get_contract_context(self, r, c): return ContractContext()
        def get_metric_pack(self, r, c): return {}
        def get_local_gain_signals(self, r, c): return {}
        def export_fusion_payload(self, r, c):
            return FusionInputContract(domain_name="study", risk_score=0.8, risk_level="low")
        def run_domain_pipeline(self, r):
            return {
                "domain_name": "study", "status": "success",
                "system_status": "multi_domain_ready",
                "final_decision": "promotion_recommended",
                "policy_decision": "promotion_recommended",
                "execution_mode": "dry_run",
                "decision_stage_reached": "approval_release_stage",
                "metric_context": {}, "domain_context": {},
                "domain_audit": {}, "warning_summary": [],
            }

    class DummyStub(BaseDomainAgent):
        @property
        def domain_name(self): return "sport"
        def train(self, r, c): return {}
        def eval(self, r, c): return {}
        def predict(self, r, c): return {}
        def build_candidate(self, r, c): return {}
        def load_baseline(self, r, c): return {}
        def get_contract_context(self, r, c): return ContractContext()
        def get_metric_pack(self, r, c): return {}
        def get_local_gain_signals(self, r, c): return {}
        def export_fusion_payload(self, r, c):
            return FusionInputContract(domain_name="sport", risk_level="stub", warning_summary=["stub"])
        def run_domain_pipeline(self, r):
            raise NotImplementedError

    root = Path(".") / ".tmp" / "unittest_verify"
    root.mkdir(parents=True, exist_ok=True)
    orch = HarnessOrchestrator(
        agents=[DummyReady(), DummyStub()],
        recorder=RunRecorder(root),
        artifact_registry=ArtifactRegistry(root),
        root_dir=root,
    )
    record, _, _ = orch.run_multi_domain({"run_id": "verify"}, ["study", "sport"])
    assert record.system_status == "partial_domain_ready"
    assert record.final_decision.decision == "dry_run_only"
    results_lines.append("TEST orchestrator_partial: PASS")
except Exception as e:
    results_lines.append(f"TEST orchestrator_partial: FAIL - {e}")

# Test 7: All domains ready
try:
    class DummyReady2(BaseDomainAgent):
        @property
        def domain_name(self): return "life"
        def train(self, r, c): return {}
        def eval(self, r, c): return {}
        def predict(self, r, c): return {}
        def build_candidate(self, r, c): return {}
        def load_baseline(self, r, c): return {}
        def get_contract_context(self, r, c): return ContractContext()
        def get_metric_pack(self, r, c): return {}
        def get_local_gain_signals(self, r, c): return {}
        def export_fusion_payload(self, r, c):
            return FusionInputContract(domain_name="life", risk_score=0.8, risk_level="low")
        def run_domain_pipeline(self, r):
            return {
                "domain_name": "life", "status": "success",
                "system_status": "multi_domain_ready",
                "final_decision": "promotion_recommended",
                "policy_decision": "promotion_recommended",
                "execution_mode": "dry_run",
                "decision_stage_reached": "approval_release_stage",
                "metric_context": {}, "domain_context": {},
                "domain_audit": {}, "warning_summary": [],
            }

    root2 = Path(".") / ".tmp" / "unittest_verify2"
    root2.mkdir(parents=True, exist_ok=True)
    orch2 = HarnessOrchestrator(
        agents=[DummyReady(), DummyReady2()],
        recorder=RunRecorder(root2),
        artifact_registry=ArtifactRegistry(root2),
        root_dir=root2,
    )
    record2, _, _ = orch2.run_multi_domain({"run_id": "verify2"}, ["study", "life"])
    assert record2.system_status == "multi_domain_ready"
    assert record2.final_decision.decision == "promotion_recommended"
    results_lines.append("TEST orchestrator_all_ready: PASS")
except Exception as e:
    results_lines.append(f"TEST orchestrator_all_ready: FAIL - {e}")

# Test 8: Invalid fusion
try:
    class DummyBadFusion(BaseDomainAgent):
        @property
        def domain_name(self): return "bad"
        def train(self, r, c): return {}
        def eval(self, r, c): return {}
        def predict(self, r, c): return {}
        def build_candidate(self, r, c): return {}
        def load_baseline(self, r, c): return {}
        def get_contract_context(self, r, c): return ContractContext()
        def get_metric_pack(self, r, c): return {}
        def get_local_gain_signals(self, r, c): return {}
        def export_fusion_payload(self, r, c):
            return FusionInputContract(domain_name="", risk_level="unknown")
        def run_domain_pipeline(self, r):
            return {
                "domain_name": "bad", "status": "success",
                "system_status": "multi_domain_ready",
                "final_decision": "promotion_recommended",
                "policy_decision": "promotion_recommended",
                "execution_mode": "dry_run",
                "decision_stage_reached": "approval_release_stage",
                "metric_context": {}, "domain_context": {},
                "domain_audit": {}, "warning_summary": [],
            }

    root3 = Path(".") / ".tmp" / "unittest_verify3"
    root3.mkdir(parents=True, exist_ok=True)
    orch3 = HarnessOrchestrator(
        agents=[DummyReady(), DummyBadFusion()],
        recorder=RunRecorder(root3),
        artifact_registry=ArtifactRegistry(root3),
        root_dir=root3,
    )
    record3, _, _ = orch3.run_multi_domain({"run_id": "verify3"}, ["study", "bad"])
    assert record3.system_status == "completed_with_hold"
    assert "fusion_payload_invalid" in record3.final_decision.reason_codes
    results_lines.append("TEST orchestrator_fusion_invalid: PASS")
except Exception as e:
    results_lines.append(f"TEST orchestrator_fusion_invalid: FAIL - {e}")

# Summary
passed = sum(1 for l in results_lines if "PASS" in l)
failed = sum(1 for l in results_lines if "FAIL" in l)
results_lines.append(f"\nSUMMARY: {passed} passed, {failed} failed, {passed+failed} total")

with open("_test_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))
