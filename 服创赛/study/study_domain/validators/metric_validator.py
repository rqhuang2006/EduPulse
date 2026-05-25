from __future__ import annotations

from harness.contracts import ValidationResult


class MetricValidator:
    name = "MetricValidator"

    def validate(self, metrics: dict, baseline_metrics: dict | None = None) -> list[ValidationResult]:
        baseline_metrics = baseline_metrics or {}
        auc = float(metrics.get("auc", 0) or 0)
        recall = float(metrics.get("recall", 0) or 0)
        f1 = float(metrics.get("f1", 0) or 0)
        baseline_auc = float(baseline_metrics.get("auc", 0) or 0)
        results = [
            ValidationResult(
                validator_name=self.name,
                passed=auc >= 0.80,
                severity="block" if auc < 0.80 else "info",
                reason_code="auc_floor_ok" if auc >= 0.80 else "auc_below_floor",
                message=f"Candidate AUC={auc:.4f}",
                details={"auc": auc},
            ),
            ValidationResult(
                validator_name=self.name,
                passed=recall >= 0.60,
                severity="warning" if recall < 0.60 else "info",
                reason_code="recall_floor_ok" if recall >= 0.60 else "recall_below_floor",
                message=f"Candidate Recall={recall:.4f}",
                details={"recall": recall},
            ),
            ValidationResult(
                validator_name=self.name,
                passed=f1 >= 0.50,
                severity="warning" if f1 < 0.50 else "info",
                reason_code="f1_floor_ok" if f1 >= 0.50 else "f1_below_floor",
                message=f"Candidate F1={f1:.4f}",
                details={"f1": f1},
            ),
        ]
        if baseline_auc:
            delta = auc - baseline_auc
            results.append(
                ValidationResult(
                    validator_name=self.name,
                    passed=delta >= -0.003,
                    severity="error" if delta < -0.003 else "info",
                    reason_code="baseline_auc_delta_ok" if delta >= -0.003 else "baseline_auc_regression",
                    message=f"AUC delta vs baseline={delta:.4f}",
                    details={"candidate_auc": auc, "baseline_auc": baseline_auc, "auc_delta": delta},
                )
            )
        return results
