from __future__ import annotations

from pathlib import Path

from harness.contracts import ArtifactRef, ValidationResult


class ArtifactValidator:
    name = "ArtifactValidator"

    def validate(self, artifacts: list[ArtifactRef]) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for artifact in artifacts:
            exists = Path(artifact.uri).exists()
            results.append(
                ValidationResult(
                    validator_name=self.name,
                    passed=exists,
                    severity="block" if not exists else "info",
                    reason_code="artifact_exists" if exists else "artifact_missing",
                    message=f"{artifact.name} -> {artifact.uri}",
                    details={"kind": artifact.kind, "uri": artifact.uri},
                )
            )
        return results
