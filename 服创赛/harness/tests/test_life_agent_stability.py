from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from life.src.life_agent import LifeAgent


class LifeAgentStabilityTests(unittest.TestCase):
    def _agent(self) -> LifeAgent:
        agent = LifeAgent.__new__(LifeAgent)
        agent.request = {"request_id": "test"}
        agent.request_id = "test"
        return agent

    def test_check_label_distribution_single_class(self) -> None:
        agent = self._agent()
        counts, valid = agent._check_label_distribution(pd.DataFrame({"life_label_clean": [0, 0, 0]}), "life_label_clean")
        self.assertEqual(counts, {0: 3})
        self.assertFalse(valid)

    def test_build_model_raises_on_single_class(self) -> None:
        agent = self._agent()
        frame = pd.DataFrame(
            {
                "life_label_clean": [1, 1, 1],
                "internet_early_sum": [0.1, 0.2, 0.3],
                "XB": ["M", "F", "M"],
            }
        )
        with self.assertRaises(ValueError):
            agent._build_model(frame, ["internet_early_sum", "XB"])

    def test_proxy_stress_test_skips_single_class(self) -> None:
        agent = self._agent()
        feature_df = pd.DataFrame(
            {
                "student_id": ["s1", "s2", "s3"],
                "life_label_clean": [0, 0, 0],
                "late_internet_mean": [1.0, 1.0, 1.0],
                "late_gate_ratio": [0.0, 0.0, 0.0],
                "late_library_mean": [1.0, 1.0, 1.0],
                "late_club_mean": [1.0, 1.0, 1.0],
                "internet_early_sum": [0.1, 0.2, 0.3],
                "internet_early_mean": [0.1, 0.2, 0.3],
                "library_early_mean": [0.0, 0.0, 0.0],
                "club_early_mean": [0.0, 0.0, 0.0],
                "gate_early_late_ratio": [0.0, 0.0, 0.0],
                "gate_early_mean": [0.0, 0.0, 0.0],
                "XB": ["M", "F", "M"],
                "MZMC": ["a", "a", "a"],
                "ZZMMMC": ["a", "a", "a"],
                "JG": ["a", "a", "a"],
                "XSM": ["a", "a", "a"],
                "ZYM": ["a", "a", "a"],
            }
        )
        candidate_features = [
            "internet_early_sum",
            "internet_early_mean",
            "library_early_mean",
            "club_early_mean",
            "gate_early_late_ratio",
            "gate_early_mean",
            "XB",
            "MZMC",
            "ZZMMMC",
            "JG",
            "XSM",
            "ZYM",
        ]
        reports, _ = agent._proxy_stress_test(
            feature_df=feature_df,
            train_df=feature_df.iloc[:2].copy(),
            holdout_df=feature_df.iloc[2:].copy(),
            candidate_features=candidate_features,
        )
        self.assertTrue(reports)
        self.assertEqual(reports[0]["status"], "skipped")

    def test_run_survives_diagnostic_failure(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        request_path = workspace / "life" / "input" / "life_agent_request.harness_v1.review.json"
        if not request_path.exists():
            self.skipTest("life request file not found")
        agent = LifeAgent(request_path=request_path)
        with mock.patch.object(LifeAgent, "_proxy_stress_test", side_effect=RuntimeError("boom")):
            result = agent.run()
        self.assertIsInstance(result, dict)
        self.assertIn("summary_metrics", result)
        self.assertIn("warnings", result)


if __name__ == "__main__":
    unittest.main()
