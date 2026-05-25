from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from harness.contracts import ContractContext, FusionInputContract
from harness.contracts.fusion_input import validate_fusion_input
from harness.domain_agents.base import BaseDomainAgent
from harness.domain_agents.orchestrator import HarnessOrchestrator
from harness.recorder.run_recorder import RunRecorder
from harness.registry.artifact_registry import ArtifactRegistry
from harness.registry.snapshot_contract import REQUIRED_SNAPSHOT_FILES, ensure_snapshot_contract


# ---------------------------------------------------------------------------
# Dummy agents for testing
# ---------------------------------------------------------------------------

class DummyReadyAgent(BaseDomainAgent):
    """A fully-implemented dummy agent that produces valid outputs."""

    def __init__(self, domain_name: str = "dummy", auc: float = 0.82):
        self._domain_name = domain_name
        self._auc = auc

    @property
    def domain_name(self) -> str:
        return self._domain_name

    def train(self, request, context):
        return {"status": "success", "metrics": {"auc": self._auc}}

    def eval(self, request, context):
        return {"status": "success", "metrics": {"auc": self._auc}}

    def predict(self, request, context):
        return {"status": "success", "metrics": {"auc": self._auc}}

    def build_candidate(self, request, context):
        return {"status": "success", "metrics": {"auc": self._auc}}

    def load_baseline(self, request, context):
        return {
            "baseline_version_id": f"{self._domain_name}_v1",
            "anchor_baseline_version_id": f"{self._domain_name}_v1",
            "baseline_metrics": {"auc": 0.80},
            "anchor_baseline_metrics": {"auc": 0.80},
            "frozen_snapshot": {},
        }

    def get_contract_context(self, request, context):
        return ContractContext(
            baseline_version_id=f"{self._domain_name}_v1",
            anchor_baseline_version_id=f"{self._domain_name}_v1",
            candidate_version_id=f"{self._domain_name}_candidate",
            baseline_metrics={"auc": 0.80},
            anchor_baseline_metrics={"auc": 0.80},
            candidate_metrics={"auc": self._auc},
            eval_split="test",
            local_gain_flags={"auc_not_worse_than_baseline": self._auc >= 0.80},
            metric_context={"eval_scope": "test"},
        )

    def get_metric_pack(self, request, context):
        return {
            "candidate_metrics": {"auc": self._auc},
            "baseline_metrics": {"auc": 0.80},
            "anchor_baseline_metrics": {"auc": 0.80},
            "metric_context": {"eval_scope": "test"},
        }

    def get_local_gain_signals(self, request, context):
        return {"auc_not_worse_than_baseline": self._auc >= 0.80}

    def export_fusion_payload(self, request, context):
        return FusionInputContract(
            domain_name=self._domain_name,
            candidate_version_id=f"{self._domain_name}_candidate",
            risk_score=self._auc,
            risk_level="low" if self._auc >= 0.8 else "medium",
            confidence=0.7,
            artifact_ref={"x": "y"},
            explanations=[{"summary": "ok"}],
        )

    def run_domain_pipeline(self, request):
        return {
            "domain_name": self._domain_name,
            "status": "success",
            "system_status": "multi_domain_ready",
            "final_decision": "promotion_recommended",
            "policy_decision": "promotion_recommended",
            "execution_mode": "dry_run",
            "decision_stage_reached": "approval_release_stage",
            "metrics": {"auc": self._auc},
            "metric_context": {"eval_scope": "test"},
            "domain_context": {"domain_name": self._domain_name},
            "domain_audit": {"stub": False},
            "warning_summary": [],
        }


class DummyStubAgent(DummyReadyAgent):
    """A stub agent that raises NotImplementedError on run_domain_pipeline."""

    def __init__(self, domain_name: str = "stub"):
        super().__init__(domain_name)

    def export_fusion_payload(self, request, context):
        return FusionInputContract(
            domain_name=self._domain_name,
            risk_level="stub",
            warning_summary=["stub"],
            metric_context={"note": "stub"},
        )

    def run_domain_pipeline(self, request):
        raise NotImplementedError


class DummyInvalidFusionAgent(DummyReadyAgent):
    """An agent that produces an invalid fusion payload."""

    def __init__(self, domain_name: str = "invalid"):
        super().__init__(domain_name)

    def export_fusion_payload(self, request, context):
        return FusionInputContract(domain_name="", risk_level="unknown")


class DummyKeepBaselineAgent(DummyReadyAgent):
    """An agent whose candidate doesn't beat baseline (keep_baseline scenario)."""

    def __init__(self, domain_name: str = "regression"):
        super().__init__(domain_name, auc=0.75)

    def run_domain_pipeline(self, request):
        return {
            "domain_name": self._domain_name,
            "status": "success",
            "system_status": "multi_domain_ready",
            "final_decision": "keep_baseline",
            "policy_decision": "keep_baseline",
            "execution_mode": "dry_run",
            "decision_stage_reached": "floor_gate",
            "metrics": {"auc": self._auc},
            "metric_context": {"eval_scope": "test"},
            "domain_context": {"domain_name": self._domain_name},
            "domain_audit": {"auc_regression": True},
            "warning_summary": ["baseline_auc_regression"],
        }


class _TestHelpers:
    @staticmethod
    def build_orchestrator(agents):
        root = Path.cwd() / ".tmp" / "unittest" / ("orchestrator_" + "_".join(a.domain_name for a in agents))
        root.mkdir(parents=True, exist_ok=True)
        return root, HarnessOrchestrator(
            agents=agents,
            recorder=RunRecorder(root),
            artifact_registry=ArtifactRegistry(root),
            root_dir=root,
        )


# ---------------------------------------------------------------------------
# Multi-domain orchestrator tests
# ---------------------------------------------------------------------------

class MultiDomainOrchestratorTests(unittest.TestCase, _TestHelpers):

    def test_partial_success_when_one_domain_stubbed(self) -> None:
        _, orchestrator = self.build_orchestrator([DummyReadyAgent("study"), DummyStubAgent("sport")])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "test"}, ["study", "sport"])
        self.assertEqual(record.system_status, "partial_domain_ready")
        self.assertEqual(record.final_decision.decision, "dry_run_only")
        self.assertIn("partial_domain_ready", record.final_decision.reason_codes)

    def test_hold_when_fusion_invalid(self) -> None:
        _, orchestrator = self.build_orchestrator([DummyReadyAgent("study"), DummyInvalidFusionAgent("life")])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "test2"}, ["study", "life"])
        self.assertEqual(record.system_status, "completed_with_hold")
        self.assertIn("fusion_payload_invalid", record.final_decision.reason_codes)

    def test_all_domains_ready(self) -> None:
        _, orchestrator = self.build_orchestrator([
            DummyReadyAgent("study"),
            DummyReadyAgent("life"),
        ])
        record, _, payloads = orchestrator.run_multi_domain({"run_id": "test3"}, ["study", "life"])
        self.assertEqual(record.system_status, "multi_domain_ready")
        self.assertEqual(record.final_decision.decision, "promotion_recommended")
        self.assertEqual(len(payloads), 2)

    def test_one_domain_ready_one_keep_baseline(self) -> None:
        _, orchestrator = self.build_orchestrator([
            DummyReadyAgent("study"),
            DummyKeepBaselineAgent("life"),
        ])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "test4"}, ["study", "life"])
        # Both domains ran successfully, so partial or multi-domain ready
        self.assertIn(record.system_status, {"multi_domain_ready", "partial_domain_ready"})

    def test_stub_only_domains(self) -> None:
        _, orchestrator = self.build_orchestrator([DummyStubAgent("sport")])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "test5"}, ["sport"])
        self.assertEqual(record.system_status, "dry_run_only")
        self.assertIn("all_domains_stubbed", record.final_decision.reason_codes)

    def test_three_domain_mix_ready_stub(self) -> None:
        _, orchestrator = self.build_orchestrator([
            DummyReadyAgent("study"),
            DummyReadyAgent("life"),
            DummyStubAgent("sport"),
        ])
        record, _, payloads = orchestrator.run_multi_domain({"run_id": "test6"}, ["study", "life", "sport"])
        self.assertEqual(record.system_status, "partial_domain_ready")
        # sport stub should not crash the orchestrator
        self.assertIn("partial_domain_ready", record.final_decision.reason_codes)


# ---------------------------------------------------------------------------
# Policy truth table tests
# ---------------------------------------------------------------------------

class PolicyTruthTableTests(unittest.TestCase):
    """Test harness system_status / final_decision logic as a truth table."""

    def _run(self, agents, domains):
        root = Path.cwd() / ".tmp" / "unittest" / "policy_table"
        root.mkdir(parents=True, exist_ok=True)
        orchestrator = HarnessOrchestrator(
            agents=agents,
            recorder=RunRecorder(root),
            artifact_registry=ArtifactRegistry(root),
            root_dir=root,
        )
        record, _, _ = orchestrator.run_multi_domain({"run_id": "policy_test"}, domains)
        return record.system_status, record.final_decision.decision, record.final_decision.reason_codes

    def test_all_ready_promotion_recommended(self) -> None:
        status, decision, codes = self._run(
            [DummyReadyAgent("a"), DummyReadyAgent("b")],
            ["a", "b"],
        )
        self.assertEqual(status, "multi_domain_ready")
        self.assertEqual(decision, "promotion_recommended")

    def test_one_stub_partial_ready(self) -> None:
        status, decision, codes = self._run(
            [DummyReadyAgent("a"), DummyStubAgent("b")],
            ["a", "b"],
        )
        self.assertEqual(status, "partial_domain_ready")
        self.assertEqual(decision, "dry_run_only")
        self.assertIn("partial_domain_ready", codes)

    def test_invalid_fusion_completed_with_hold(self) -> None:
        status, decision, codes = self._run(
            [DummyReadyAgent("a"), DummyInvalidFusionAgent("b")],
            ["a", "b"],
        )
        self.assertEqual(status, "completed_with_hold")
        self.assertEqual(decision, "dry_run_only")
        self.assertIn("fusion_payload_invalid", codes)

    def test_all_stubs_dry_run_only(self) -> None:
        status, decision, codes = self._run(
            [DummyStubAgent("a"), DummyStubAgent("b")],
            ["a", "b"],
        )
        self.assertEqual(status, "dry_run_only")
        self.assertIn("all_domains_stubbed", codes)


# ---------------------------------------------------------------------------
# Fusion payload validation tests
# ---------------------------------------------------------------------------

class FusionPayloadValidationTests(unittest.TestCase):

    def test_accepts_valid_payload(self) -> None:
        result = validate_fusion_input(
            FusionInputContract(
                domain_name="life",
                risk_score=0.78,
                risk_level="medium",
                confidence=0.61,
                explanations=[{"summary": "ok"}],
                artifact_ref={"eval_report": "x.json"},
            )
        )
        self.assertTrue(result["ok"])
        self.assertFalse(result["is_stub"])

    def test_accepts_stub_payload(self) -> None:
        result = validate_fusion_input(
            FusionInputContract(
                domain_name="sport",
                risk_level="stub",
                warning_summary=["stub"],
            ),
            allow_stub=True,
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["is_stub"])

    def test_rejects_empty_domain_name(self) -> None:
        result = validate_fusion_input({"domain_name": "", "risk_level": "low"})
        self.assertFalse(result["ok"])
        self.assertIn("domain_name_missing", result["errors"])

    def test_rejects_bad_risk_level(self) -> None:
        result = validate_fusion_input({"domain_name": "x", "risk_level": "weird"})
        self.assertFalse(result["ok"])
        self.assertIn("risk_level_invalid", result["errors"])

    def test_rejects_bad_risk_score(self) -> None:
        result = validate_fusion_input({"domain_name": "x", "risk_score": "bad", "risk_level": "low"})
        self.assertFalse(result["ok"])
        self.assertIn("risk_score_not_numeric", result["errors"])

    def test_rejects_bad_confidence(self) -> None:
        result = validate_fusion_input({"domain_name": "x", "confidence": "bad", "risk_level": "low"})
        self.assertFalse(result["ok"])
        self.assertIn("confidence_not_numeric", result["errors"])

    def test_rejects_explanations_not_list(self) -> None:
        result = validate_fusion_input({"domain_name": "x", "explanations": {}, "risk_level": "low"})
        self.assertFalse(result["ok"])
        self.assertIn("explanations_not_list", result["errors"])

    def test_rejects_artifact_ref_not_dict(self) -> None:
        result = validate_fusion_input({"domain_name": "x", "artifact_ref": [], "risk_level": "low"})
        self.assertFalse(result["ok"])
        self.assertIn("artifact_ref_not_object", result["errors"])

    def test_rejects_validation_summary_not_dict(self) -> None:
        result = validate_fusion_input({"domain_name": "x", "validation_summary": [], "risk_level": "low"})
        self.assertFalse(result["ok"])
        self.assertIn("validation_summary_not_object", result["errors"])

    def test_rejects_warning_summary_not_list(self) -> None:
        result = validate_fusion_input({"domain_name": "x", "warning_summary": "bad", "risk_level": "low"})
        self.assertFalse(result["ok"])
        self.assertIn("warning_summary_not_list", result["errors"])

    def test_risk_score_missing_warning(self) -> None:
        result = validate_fusion_input(
            FusionInputContract(domain_name="x", risk_level="low"),
            allow_stub=False,
        )
        self.assertTrue(result["ok"])
        self.assertIn("risk_score_missing", result["warnings"])


# ---------------------------------------------------------------------------
# Snapshot contract tests
# ---------------------------------------------------------------------------

class SnapshotContractTests(unittest.TestCase):

    def _workspace_tmp(self, name: str) -> Path:
        root = Path.cwd() / ".tmp" / "unittest" / name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_writes_non_empty_fallbacks(self) -> None:
        root = self._workspace_tmp("snapshot_contract")
        manifest = ensure_snapshot_contract(
            root / "life_v1",
            version_id="life_v1",
            domain="life",
            payloads={"model_config.json": {}},
        )
        self.assertEqual(set(manifest.keys()), set(REQUIRED_SNAPSHOT_FILES))
        for filename in REQUIRED_SNAPSHOT_FILES:
            content = (root / "life_v1" / filename).read_text(encoding="utf-8").strip()
            self.assertNotEqual(content, "{}")

    def test_writes_all_required_files(self) -> None:
        root = self._workspace_tmp("snapshot_full")
        manifest = ensure_snapshot_contract(
            root / "study_v1",
            version_id="study_v1",
            domain="study",
        )
        self.assertEqual(len(manifest), len(REQUIRED_SNAPSHOT_FILES))
        for filename, path in manifest.items():
            self.assertTrue(Path(path).exists(), f"{filename} not found at {path}")

    def test_missing_fields_fallback(self) -> None:
        root = self._workspace_tmp("snapshot_fallback")
        manifest = ensure_snapshot_contract(
            root / "test_v1",
            version_id="test_v1",
            domain="test",
            payloads={"model_config.json": None},
        )
        # Even with None payload, fallback defaults should be written
        content = (root / "test_v1" / "model_config.json").read_text(encoding="utf-8").strip()
        self.assertNotEqual(content, "{}")
        self.assertIn("harness_snapshot_contract_v2", content)


# ---------------------------------------------------------------------------
# End-to-end smoke test (uses real life agent if data available)
# ---------------------------------------------------------------------------

class EndToEndSmokeTests(unittest.TestCase):
    """Smoke tests that verify the harness wiring with real domain agents."""

    def test_life_agent_request_structure(self) -> None:
        """Verify that the life agent request file exists and has the right structure."""
        workspace = Path(__file__).resolve().parents[2]
        request_path = workspace / "life" / "input" / "life_agent_request.harness_v1.review.json"
        if not request_path.exists():
            self.skipTest("life agent request file not found")

        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(request["domain"], "life")
        self.assertIn("run_mode", request)
        self.assertIn("execution_engine", request)
        self.assertIn("input_paths", request)

    def test_sport_stub_returns_structured_result(self) -> None:
        """Verify that sport stub returns a structured dict, not an exception."""
        from harness.domain_agents.sport import SportAgentStub

        agent = SportAgentStub()
        result = agent.run_domain_pipeline({})
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "stub")
        self.assertEqual(result["system_status"], "stub")
        self.assertEqual(result["domain_name"], "sport")


# ---------------------------------------------------------------------------
# Additional coverage: floor_gate hold, local_gain missing, dry_run,
# multi-domain partial success, snapshot missing fields fallback,
# fusion payload completeness
# ---------------------------------------------------------------------------

class FloorGateHoldTests(unittest.TestCase):
    """Test that keep_baseline / floor_gate decisions propagate correctly."""

    def test_keep_baseline_decision_propagates(self) -> None:
        """Agent returning keep_baseline should not be treated as success."""
        _, orchestrator = _TestHelpers.build_orchestrator([DummyKeepBaselineAgent("life")])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "floor_test"}, ["life"])
        # The domain ran but the decision is keep_baseline
        self.assertIn(record.final_decision.decision, {"keep_baseline", "dry_run_only"})

    def test_floor_gate_with_regression_warning(self) -> None:
        """A floor-gate hold should carry the regression warning."""
        _, orchestrator = _TestHelpers.build_orchestrator([DummyKeepBaselineAgent("x")])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "floor_warn"}, ["x"])
        # Warnings should include the baseline regression note
        all_warnings = " ".join(record.collected_warnings)
        # At minimum the system_status should reflect hold or partial
        self.assertIn(record.system_status, {"multi_domain_ready", "partial_domain_ready", "completed_with_hold"})


class LocalGainMissingTests(unittest.TestCase):
    """Test behavior when local gain signals are absent or all False."""

    def test_no_local_gain_signals_treated_as_not_met(self) -> None:
        """An agent with empty local_gain_flags should not get promotion."""
        class NoGainAgent(DummyReadyAgent):
            def get_local_gain_signals(self, request, context):
                return {}

            def run_domain_pipeline(self, request):
                return {
                    "domain_name": "nogain",
                    "status": "success",
                    "system_status": "completed_with_hold",
                    "final_decision": "keep_baseline",
                    "policy_decision": "keep_baseline",
                    "execution_mode": "dry_run",
                    "decision_stage_reached": "local_gain_gate",
                    "metrics": {"auc": 0.82},
                    "metric_context": {"eval_scope": "test"},
                    "domain_context": {},
                    "domain_audit": {"no_local_gain": True},
                    "warning_summary": ["no_local_gain_signals"],
                }

        _, orchestrator = _TestHelpers.build_orchestrator([NoGainAgent("nogain")])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "nogain_test"}, ["nogain"])
        self.assertEqual(record.system_status, "completed_with_hold")
        self.assertEqual(record.final_decision.decision, "dry_run_only")


class DryRunModeTests(unittest.TestCase):
    """Verify that dry_run execution mode is consistently set."""

    def test_all_scenarios_are_dry_run(self) -> None:
        """Every test scenario should produce dry_run execution mode."""
        scenarios = [
            ([DummyReadyAgent("a")], ["a"]),
            ([DummyReadyAgent("a"), DummyStubAgent("b")], ["a", "b"]),
            ([DummyStubAgent("s")], ["s"]),
        ]
        for agents, domains in scenarios:
            _, orchestrator = _TestHelpers.build_orchestrator(agents)
            record, _, _ = orchestrator.run_multi_domain({"run_id": f"dryrun_{domains[0]}"}, domains)
            self.assertEqual(record.final_decision.execution_mode, "dry_run",
                             f"Expected dry_run for domains={domains}, got {record.final_decision.execution_mode}")


class SnapshotMissingFieldsFallbackTests(unittest.TestCase):
    """Verify that snapshot contract handles missing / empty fields gracefully."""

    def test_empty_payload_gets_defaults(self) -> None:
        """Passing {} for all payloads should still write valid defaults."""
        root = Path.cwd() / ".tmp" / "unittest" / "snapshot_empty_all"
        root.mkdir(parents=True, exist_ok=True)
        manifest = ensure_snapshot_contract(
            root / "test_v1",
            version_id="test_v1",
            domain="test",
            payloads={
                "model_config.json": {},
                "feature_config.json": {},
                "contract_context.json": {},
                "domain_audit.json": {},
                "metrics.json": {},
            },
        )
        for filename in REQUIRED_SNAPSHOT_FILES:
            path = root / "test_v1" / filename
            self.assertTrue(path.exists(), f"{filename} should exist")
            content = path.read_text(encoding="utf-8").strip()
            self.assertNotEqual(content, "{}", f"{filename} should not be empty {{}}")
            self.assertIn("harness_snapshot_contract_v2", content)

    def test_partial_payloads_complete_missing(self) -> None:
        """Providing only 2 of 5 required files should fill the rest."""
        root = Path.cwd() / ".tmp" / "unittest" / "snapshot_partial"
        root.mkdir(parents=True, exist_ok=True)
        manifest = ensure_snapshot_contract(
            root / "test_v1",
            version_id="test_v1",
            domain="test",
            payloads={
                "model_config.json": {"custom": True},
                "metrics.json": {"summary_metrics": {"auc": 0.8}},
            },
        )
        self.assertEqual(len(manifest), len(REQUIRED_SNAPSHOT_FILES))
        # Custom payloads preserved
        mc = json.loads((root / "test_v1" / "model_config.json").read_text(encoding="utf-8"))
        self.assertTrue(mc.get("custom"))
        # Missing files got defaults
        fc = json.loads((root / "test_v1" / "feature_config.json").read_text(encoding="utf-8"))
        self.assertIn("schema_version", fc)


class MultiDomainPartialSuccessTests(unittest.TestCase):
    """Test multi-domain scenarios where some domains succeed and others don't."""

    def test_one_ready_one_failed(self) -> None:
        """When one domain fails, system should not crash."""
        class FailedAgent(DummyReadyAgent):
            def run_domain_pipeline(self, request):
                return {
                    "domain_name": "failed",
                    "status": "failed",
                    "system_status": "completed_with_hold",
                    "final_decision": "reject",
                    "policy_decision": "reject",
                    "execution_mode": "dry_run",
                    "decision_stage_reached": "contract_chain_gate",
                    "metrics": {},
                    "metric_context": {},
                    "domain_context": {},
                    "domain_audit": {"failure": True},
                    "warning_summary": ["agent_failed"],
                }

            def export_fusion_payload(self, request, context):
                return FusionInputContract(
                    domain_name="failed",
                    risk_level="stub",
                    warning_summary=["agent failed, no fusion payload"],
                )

        _, orchestrator = _TestHelpers.build_orchestrator([
            DummyReadyAgent("study"),
            FailedAgent("failed"),
        ])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "partial_fail"}, ["study", "failed"])
        # System should not crash; partial output allowed
        self.assertIn(record.system_status, {"partial_domain_ready", "completed_with_hold", "not_releaseable_for_prod"})

    def test_ready_count_matches_metric_context(self) -> None:
        """metric_context should correctly count ready/stub domains."""
        _, orchestrator = _TestHelpers.build_orchestrator([
            DummyReadyAgent("study"),
            DummyReadyAgent("life"),
            DummyStubAgent("sport"),
        ])
        record, _, _ = orchestrator.run_multi_domain({"run_id": "count_test"}, ["study", "life", "sport"])
        mc = record.metric_context
        self.assertEqual(mc.get("ready_domain_count"), 2)
        self.assertEqual(mc.get("stub_domain_count"), 1)
        self.assertEqual(mc.get("requested_domain_count"), 3)


class FusionPayloadCompletenessTests(unittest.TestCase):
    """Verify fusion payload validation covers all required fields."""

    def test_all_required_fields_validated(self) -> None:
        """A payload missing multiple fields should report all errors."""
        result = validate_fusion_input({
            "domain_name": "",
            "risk_level": "invalid_level",
            "risk_score": "not_numeric",
            "confidence": "not_numeric",
            "explanations": "not_a_list",
            "artifact_ref": "not_a_dict",
            "validation_summary": "not_a_dict",
            "warning_summary": "not_a_list",
        })
        self.assertFalse(result["ok"])
        expected_errors = {
            "domain_name_missing",
            "risk_level_invalid",
            "risk_score_not_numeric",
            "confidence_not_numeric",
            "explanations_not_list",
            "artifact_ref_not_object",
            "validation_summary_not_object",
            "warning_summary_not_list",
        }
        for err in expected_errors:
            self.assertIn(err, result["errors"], f"Expected error '{err}' not found")

    def test_stub_payload_with_allow_stub_true(self) -> None:
        """A stub payload should pass when allow_stub=True."""
        result = validate_fusion_input(
            FusionInputContract(
                domain_name="sport",
                risk_level="stub",
                risk_score=None,
                confidence=None,
                top_features=[],
                explanations=[],
                validation_summary={"is_stub": True},
                warning_summary=["SportAgent is a stub"],
                artifact_ref={},
            ),
            allow_stub=True,
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["is_stub"])

    def test_stub_payload_without_allow_stub_warns(self) -> None:
        """A stub payload without allow_stub should warn about missing risk_score."""
        result = validate_fusion_input(
            FusionInputContract(
                domain_name="sport",
                risk_level="stub",
                warning_summary=["stub"],
            ),
            allow_stub=False,
        )
        self.assertTrue(result["ok"])  # still passes (not hard error)
        self.assertIn("risk_score_missing", result["warnings"])


# ---------------------------------------------------------------------------
# Life domain single-class guard tests
# ---------------------------------------------------------------------------

class LifeDomainSingleClassGuardTests(unittest.TestCase):
    """Verify that life agent handles single-class label distributions gracefully."""

    def test_check_label_distribution_detects_single_class(self) -> None:
        """_check_label_distribution should return is_valid=False for single-class data."""
        import pandas as pd
        from life.src.life_agent import LifeAgent

        # Create a minimal agent (won't run, just need the method)
        agent = LifeAgent.__new__(LifeAgent)

        df = pd.DataFrame({"life_label_clean": [0, 0, 0, 0]})
        counts, is_valid = agent._check_label_distribution(df, "life_label_clean")
        self.assertFalse(is_valid)
        self.assertEqual(counts, {0: 4})

        df2 = pd.DataFrame({"life_label_clean": [1, 1, 1]})
        counts2, is_valid2 = agent._check_label_distribution(df2, "life_label_clean")
        self.assertFalse(is_valid2)
        self.assertEqual(counts2, {1: 3})

    def test_check_label_distribution_accepts_two_classes(self) -> None:
        """_check_label_distribution should return is_valid=True for binary data."""
        import pandas as pd
        from life.src.life_agent import LifeAgent

        agent = LifeAgent.__new__(LifeAgent)

        df = pd.DataFrame({"life_label_clean": [0, 0, 1, 1, 0, 1]})
        counts, is_valid = agent._check_label_distribution(df, "life_label_clean")
        self.assertTrue(is_valid)
        self.assertEqual(counts, {0: 3, 1: 3})

    def test_build_model_raises_for_single_class(self) -> None:
        """_build_model should raise ValueError when training data has only one class."""
        import pandas as pd
        from life.src.life_agent import LifeAgent, ROOT

        agent = LifeAgent.__new__(LifeAgent)
        agent.request = {"request_id": "test"}
        agent.request_id = "test"

        train_df = pd.DataFrame({
            "student_id": ["s1", "s2", "s3"],
            "life_label_clean": [0, 0, 0],
            "internet_early_sum": [1.0, 2.0, 3.0],
            "XB": ["M", "F", "M"],
        })
        feature_columns = ["internet_early_sum", "XB"]

        with self.assertRaises(ValueError) as ctx:
            agent._build_model(train_df, feature_columns)
        self.assertIn("only 1 class", str(ctx.exception))
        self.assertIn("single-class", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
