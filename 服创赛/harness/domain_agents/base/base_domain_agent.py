from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from harness.contracts.contract_context import ContractContext
from harness.contracts.fusion_input import FusionInputContract


class BaseDomainAgent(ABC):
    """Abstract interface all domain agents (study, life, sport) must implement.

    Each domain agent is responsible for:
    - Domain-specific feature construction
    - Domain-specific train / eval / predict
    - Domain-specific explainability
    - Domain-specific baseline loading
    - Domain-specific local-gain signals

    The harness core interacts with domain agents ONLY through this interface.
    """

    domain_name: str

    @abstractmethod
    def train(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def eval(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_candidate(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_baseline(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_contract_context(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_metric_pack(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_local_gain_signals(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def export_fusion_payload(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_domain_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Single entrypoint for orchestrator."""
        raise NotImplementedError

    def build_fusion_input(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "risk_score": run_result.get("risk_score"),
            "risk_level": run_result.get("risk_level"),
            "confidence": run_result.get("confidence"),
            "top_features": run_result.get("top_features", []),
            "explanations": run_result.get("explanations", []),
            "artifact_ref": run_result.get("artifact_ref"),
            "metric_context": run_result.get("metric_context", {}),
            "validation_summary": run_result.get("validation_summary", []),
            "warning_summary": run_result.get("warning_summary", []),
        }

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain identifier (e.g., 'study', 'life', 'sport')."""

    # ---- Core pipeline methods ----

    @abstractmethod
    def train(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute domain-specific training.

        Returns:
            dict with keys: status, metrics, artifacts, diagnostics, message
        """

    @abstractmethod
    def eval(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute domain-specific evaluation.

        Returns:
            dict with keys: status, metrics, artifacts, diagnostics, message
        """

    @abstractmethod
    def predict(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute domain-specific prediction/inference.

        Returns:
            dict with keys: status, metrics, artifacts, diagnostics, message
        """

    # ---- Baseline and candidate management ----

    @abstractmethod
    def build_candidate(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Build a candidate artifact for evaluation.

        Returns:
            dict with candidate metadata, metrics, and artifact references
        """

    @abstractmethod
    def load_baseline(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Load the active and anchor baselines for this domain.

        Returns:
            dict with baseline_version_id, anchor_baseline_version_id,
            baseline_metrics, anchor_baseline_metrics, frozen_snapshot
        """

    # ---- Contract and metric context ----

    @abstractmethod
    def get_contract_context(self, request: dict[str, Any], context: dict[str, Any]) -> ContractContext:
        """Build a generic ContractContext for same-caliber and comparison logic.

        The harness core consumes this context — it must NOT contain
        domain-specific field names in the top-level structure.
        """

    @abstractmethod
    def get_metric_pack(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Return the domain's metric pack for policy evaluation.

        Returns:
            dict with candidate_metrics, baseline_metrics, metric_context
        """

    @abstractmethod
    def get_local_gain_signals(self, request: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Return domain-specific local gain signals.

        Returns:
            dict of {signal_name: bool} indicating whether each gain
            criterion is met. The harness policy evaluates these generically.
        """

    # ---- Fusion ----

    @abstractmethod
    def export_fusion_payload(self, request: dict[str, Any], context: dict[str, Any]) -> FusionInputContract:
        """Export a normalized FusionInputContract for future fusion.

        Every domain agent produces one of these. The fusion layer
        combines them without needing to know domain-specific internals.
        """
