from __future__ import annotations

from pathlib import Path

from harness.actions.base import BaseAction
from harness.contracts import ActionResult, ValidationResult
from study_domain.adapters.caliber_adapter import StudyCaliberAdapter


class CaliberAction(BaseAction):
    name = "same_caliber"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.adapter = StudyCaliberAdapter(root_dir)

    def run(self, context) -> ActionResult:
        payload = self.adapter.execute(context.request)
        diagnostics = payload.get("comparison", {})
        same_caliber_ok = bool(payload.get("same_caliber_ok", False))
        reason_codes = payload.get("same_caliber_reasons", []) + payload.get("chain_reasons", [])
        validations = [
            ValidationResult(
                validator_name=self.name,
                passed=same_caliber_ok,
                severity="block" if not same_caliber_ok else "info",
                reason_code="same_caliber_ok" if same_caliber_ok else "same_caliber_failed",
                message="Candidate comparability checked before selection.",
                details={
                    "candidate_version_id": payload.get("candidate_version_id"),
                    "baseline_version_id": payload.get("baseline_version_id"),
                    "reason_codes": reason_codes,
                },
            )
        ]
        context.metadata["same_caliber"] = {
            "ok": same_caliber_ok,
            "reason_codes": reason_codes,
        }
        return ActionResult(
            action_name=self.name,
            status=payload.get("status", "success"),
            diagnostics=diagnostics,
            validations=validations,
            message="Same-caliber gate evaluated before selection policy.",
        )
