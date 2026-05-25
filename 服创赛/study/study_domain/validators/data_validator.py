from __future__ import annotations

from pathlib import Path

import pandas as pd

from harness.contracts import ValidationResult


class DataValidator:
    name = "DataValidator"

    def validate_train_source(self, train_path: Path, label_col: str = "LABEL") -> list[ValidationResult]:
        if not train_path.exists():
            return [
                ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    severity="block",
                    reason_code="train_table_missing",
                    message=f"Training table not found: {train_path}",
                )
            ]

        frame = pd.read_csv(train_path, nrows=200)
        results: list[ValidationResult] = []
        results.append(
            ValidationResult(
                validator_name=self.name,
                passed=label_col in frame.columns,
                severity="block" if label_col not in frame.columns else "info",
                reason_code="label_column_present" if label_col in frame.columns else "label_column_missing",
                message="Label column validated." if label_col in frame.columns else f"{label_col} missing from training data.",
                details={"columns": frame.columns.tolist()},
            )
        )
        if label_col in frame.columns:
            label_series = pd.to_numeric(frame[label_col], errors="coerce")
            label_ok = label_series.nunique(dropna=True) >= 2
            results.append(
                ValidationResult(
                    validator_name=self.name,
                    passed=label_ok,
                    severity="block" if not label_ok else "info",
                    reason_code="label_cardinality_ok" if label_ok else "label_cardinality_insufficient",
                    message="Label cardinality validated." if label_ok else "Training label has insufficient class diversity.",
                    details={"sample_rows_checked": int(len(frame))},
                )
            )
        return results
