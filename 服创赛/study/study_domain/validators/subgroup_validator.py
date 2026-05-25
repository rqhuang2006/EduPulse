from __future__ import annotations

from harness.contracts import ValidationResult


class SubgroupValidator:
    name = "SubgroupValidator"

    def validate(self, subgroup_metrics: dict[str, dict], baseline_subgroup_metrics: dict[str, dict] | None = None) -> list[ValidationResult]:
        baseline_subgroup_metrics = baseline_subgroup_metrics or {}
        results: list[ValidationResult] = []
        for subgroup in ["single_fail", "overall_low", "core_plus_behavior", "core_only"]:
            current = subgroup_metrics.get(subgroup, {})
            baseline = baseline_subgroup_metrics.get(subgroup, {})
            if not current:
                continue
            current_auc = float(current.get("auc", 0) or 0)
            baseline_auc = float(baseline.get("auc", 0) or 0)
            delta = current_auc - baseline_auc if baseline else None
            passed = delta is None or delta >= -0.01
            results.append(
                ValidationResult(
                    validator_name=self.name,
                    passed=passed,
                    severity="error" if not passed else "info",
                    reason_code=f"{subgroup}_stable" if passed else f"{subgroup}_degraded",
                    message=f"{subgroup} AUC={current_auc:.4f}" + (f", delta={delta:.4f}" if delta is not None else ""),
                    details={"subgroup": subgroup, "auc": current_auc, "baseline_auc": baseline_auc, "auc_delta": delta},
                )
            )
        return results
