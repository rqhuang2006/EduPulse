from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.contracts.fusion_input import FusionInputContract, validate_fusion_input
from harness.registry.snapshot_contract import REQUIRED_SNAPSHOT_FILES, ensure_snapshot_contract


class FusionAndSnapshotTests(unittest.TestCase):
    def _workspace_tmp(self, name: str) -> Path:
        root = Path.cwd() / ".tmp" / "unittest" / name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_fusion_validator_accepts_valid_payload(self) -> None:
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

    def test_fusion_validator_rejects_invalid_payload(self) -> None:
        result = validate_fusion_input(
            {
                "domain_name": "",
                "risk_score": "bad",
                "risk_level": "weird",
                "confidence": "bad",
                "explanations": {},
                "artifact_ref": [],
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("domain_name_missing", result["errors"])

    def test_snapshot_contract_writes_non_empty_fallbacks(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
