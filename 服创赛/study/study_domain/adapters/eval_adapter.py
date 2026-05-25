from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from harness.contracts import ArtifactRef
from study_domain.registry.baseline_registry import BaselineRegistry


class StudyEvalAdapter:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.dm_dir = root_dir / "data" / "dm"
        self.baseline_registry = BaselineRegistry(root_dir)

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def execute(self, request: dict) -> dict:
        eval_report = self._read_json(self.dm_dir / "study_eval_report.json", {})
        model_metrics = self._read_json(self.dm_dir / "study_model_metrics.json", {})
        model_config = self._read_json(self.root_dir / "data" / "deliverables" / "study" / "model" / "study_model_config.json", {})
        baselines = self.baseline_registry.current_baselines()
        active_baseline = baselines.get("active_baseline", {})
        anchor_baseline = baselines.get("anchor_baseline", {})

        overall = eval_report.get("overall_metrics", {})
        candidate_metrics = {
            "auc": overall.get("auc", model_metrics.get("core_model", {}).get("valid", {}).get("auc")),
            "f1": overall.get("f1", model_metrics.get("core_model", {}).get("valid", {}).get("f1")),
            "recall": overall.get("recall", model_metrics.get("core_model", {}).get("valid", {}).get("recall")),
            "precision": overall.get("precision"),
            "coverage": overall.get("coverage"),
            "degraded_ratio": overall.get("degraded_ratio"),
        }

        mode_metrics = {row.get("study_data_mode"): row for row in eval_report.get("mode_metrics", [])}
        subtype_metrics = {row.get("label_subtype"): row for row in eval_report.get("subtype_metrics", [])}
        confidence_metrics = {row.get("confidence_zone"): row for row in eval_report.get("confidence_zone_metrics", [])}

        active_metrics = active_baseline.get("metrics", {})
        active_eval = self._read_json(Path(active_baseline.get("frozen_snapshot", {}).get("eval_report", "")), {})
        active_mode_metrics = {row.get("study_data_mode"): row for row in active_eval.get("mode_metrics", [])}
        active_subtype_metrics = {row.get("label_subtype"): row for row in active_eval.get("subtype_metrics", [])}

        local_gain_flags = {
            "single_fail_auc_gain": subtype_metrics.get("single_fail", {}).get("auc", 0) > active_subtype_metrics.get("single_fail", {}).get("auc", 0),
            "core_plus_behavior_auc_gain": mode_metrics.get("core_plus_behavior", {}).get("auc", 0) > active_mode_metrics.get("core_plus_behavior", {}).get("auc", 0),
            "f1_improved": float(candidate_metrics.get("f1", 0) or 0) > float(active_metrics.get("f1", 0) or 0),
            "recall_improved": float(candidate_metrics.get("recall", 0) or 0) > float(active_metrics.get("recall", 0) or 0),
            "degraded_ratio_improved": float(candidate_metrics.get("degraded_ratio", 1) or 1) < float(active_metrics.get("degraded_ratio", 1) or 1),
        }

        feature_columns = model_config.get("feature_columns", []) or []
        feature_contract_hash = hashlib.sha1("\n".join(feature_columns).encode("utf-8")).hexdigest() if feature_columns else None
        metric_context = {
            "eval_scope": "publish_gate_metric",
            "data_mode": "mixed_overall_with_subgroups",
            "feature_contract_hash": feature_contract_hash,
            "label_version": model_config.get("label_name", "LABEL"),
            "threshold_regime": "threshold_not_applied_auc_primary",
            "sample_count": overall.get("rows"),
        }

        artifacts = [
            ArtifactRef("eval_report", "metrics_report", str(self.dm_dir / "study_eval_report.json")),
            ArtifactRef("subgroup_metrics", "metrics_report", str(self.dm_dir / "study_subgroup_metrics.csv")),
            ArtifactRef("confidence_zone_report", "metrics_report", str(self.dm_dir / "study_confidence_zone_report.csv")),
        ]
        return {
            "status": "success",
            "metrics": candidate_metrics,
            "artifacts": artifacts,
            "diagnostics": {
                "metric_context": metric_context,
                "baseline_metrics": active_metrics,
                "anchor_baseline_metrics": anchor_baseline.get("metrics", {}),
                "mode_metrics": mode_metrics,
                "subtype_metrics": subtype_metrics,
                "confidence_metrics": confidence_metrics,
                "local_gain_flags": local_gain_flags,
                "baseline_ids": {
                    "active": baselines.get("active_baseline_id"),
                    "anchor": baselines.get("anchor_baseline_id"),
                },
            },
            "message": "Candidate evaluation and baseline comparison completed.",
        }
