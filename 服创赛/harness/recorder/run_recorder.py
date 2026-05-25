from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from harness.contracts import PipelineContext, PolicyDecision, RunRecord


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _collect_warnings(context: PipelineContext) -> list[str]:
    """Collect warning messages from stage validations, action messages, and policy output."""
    warnings: list[str] = []

    for result in context.stage_results:
        for v in result.validations:
            if (not v.passed) and v.severity == "warning":
                msg = f"[{result.action_name}] {v.validator_name}: {v.reason_code} - {v.message}"
                if msg not in warnings:
                    warnings.append(msg)

        if result.message and "warn" in result.message.lower():
            msg = f"[{result.action_name}] {result.message}"
            if msg not in warnings:
                warnings.append(msg)

    if context.final_decision and getattr(context.final_decision, "collected_warnings", None):
        for w in context.final_decision.collected_warnings:
            if w not in warnings:
                warnings.append(w)

    return warnings


def _extract_metric_context(context: PipelineContext) -> dict[str, str]:
    """Extract generic audit fields from eval/publish diagnostics.

    This helper is generic at harness level:
    - keeps only common comparable fields at top-level RunRecord
    - leaves domain-specific details to domain_audit
    """
    eval_result = context.latest_result("eval")
    publish_result = context.latest_result("publish")

    mc = eval_result.diagnostics.get("metric_context", {}) if eval_result else {}
    publish_diag = publish_result.diagnostics if publish_result else {}
    comparison = publish_diag.get("baseline_comparison", {})
    candidate_diag = publish_diag.get("candidate", {})
    same_caliber = comparison.get("same_caliber", {})

    task_scope = ""
    if candidate_diag.get("task_scope"):
        task_scope = str(candidate_diag.get("task_scope"))
    elif mc.get("task_scope"):
        task_scope = str(mc.get("task_scope"))
    elif same_caliber.get("candidate_task_scope"):
        task_scope = str(same_caliber.get("candidate_task_scope"))

    return {
        "eval_scope": str(mc.get("eval_scope", "")),
        "task_scope": task_scope,
        "feature_contract_hash": str(mc.get("feature_contract_hash", "")),
        "label_definition": str(mc.get("label_definition", mc.get("label_version", ""))),
        "baseline_version_id": str(comparison.get("baseline_version_id", "")),
        "anchor_baseline_version_id": str(comparison.get("anchor_baseline_version_id", "")),
        "comparison_mode": str(comparison.get("comparison_mode", comparison.get("selection_rule", ""))),
        "decision_stage_reached": str(comparison.get("decision_stage_reached", "")),
    }


def _extract_domain_context(context: PipelineContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract domain-facing context without teaching harness domain semantics.

    The recorder only forwards already-exported domain diagnostics so adapters and
    agents can consume them later. It does not invent domain logic.
    """
    eval_result = context.latest_result("eval")
    if not eval_result:
        return {}, {}

    diagnostics = getattr(eval_result, "diagnostics", {}) or {}
    metric_context = diagnostics.get("metric_context", {}) or {}
    eval_metrics = getattr(eval_result, "metrics", {}) or {}

    data_mode = (
        metric_context.get("study_data_mode")
        or metric_context.get("row_level_study_data_mode")
        or metric_context.get("data_mode")
        or eval_metrics.get("study_data_mode")
        or eval_metrics.get("row_level_study_data_mode")
        or eval_metrics.get("data_mode")
        or ""
    )
    if not data_mode:
        return {}, {}

    forwarded_metric_context = {
        "study_data_mode": str(data_mode),
        "row_level_study_data_mode": str(
            metric_context.get("row_level_study_data_mode") or data_mode
        ),
    }
    forwarded_domain_context = dict(forwarded_metric_context)
    return forwarded_metric_context, forwarded_domain_context


def _extract_domain_audit(context: PipelineContext) -> dict[str, Any]:
    """Return domain-specific audit payload.

    Harness does not interpret domain internals.
    It only stores them under domain_audit for later inspection.
    """
    if not context.domain_context:
        return {}

    if context.domain in context.domain_context and isinstance(context.domain_context[context.domain], dict):
        return context.domain_context[context.domain]

    return context.domain_context


def _extract_multi_domain_audit(context: PipelineContext) -> dict[str, Any]:
    """Return multi-domain audit payload from context metadata."""
    return {
        "domain_results": context.metadata.get("domain_results", {}),
        "domains_executed": context.metadata.get("domains_executed", []),
        "fusion_inputs": context.metadata.get("fusion_inputs", []),
    }


class RunRecorder:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.run_dir = root_dir / "data" / "harness" / "runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def finalize(self, context: PipelineContext, decision: PolicyDecision | None) -> tuple[RunRecord, Path]:
        status = "failed" if any(result.status == "failed" for result in context.stage_results) else "success"
        collected_warnings = _collect_warnings(context)
        audit_fields = _extract_metric_context(context)
        forwarded_metric_context, forwarded_domain_context = _extract_domain_context(context)
        domain_audit = _extract_domain_audit(context)

        decision_stage = ""
        if decision and getattr(decision, "decision_stage_reached", ""):
            decision_stage = decision.decision_stage_reached
        elif audit_fields.get("decision_stage_reached"):
            decision_stage = audit_fields["decision_stage_reached"]

        record = RunRecord(
            run_id=context.run_id,
            pipeline_name=context.pipeline_name,
            domain=context.domain,
            stage_results=context.stage_results,
            final_decision=decision,
            started_at=context.metadata.get("started_at", now_iso()),
            finished_at=now_iso(),
            status=status,
            metadata={
                **context.metadata,
                "baseline_info": context.baseline_info,
                "candidate_info": context.candidate_info,
            },
            run_type="single_domain",
            domain_name=context.domain,
            collected_warnings=collected_warnings,
            eval_scope=audit_fields.get("eval_scope", ""),
            task_scope=audit_fields.get("task_scope", ""),
            feature_contract_hash=audit_fields.get("feature_contract_hash", ""),
            label_definition=audit_fields.get("label_definition", ""),
            baseline_version_id=audit_fields.get("baseline_version_id", ""),
            anchor_baseline_version_id=audit_fields.get("anchor_baseline_version_id", ""),
            comparison_mode=audit_fields.get("comparison_mode", ""),
            policy_decision=decision.decision if decision else "",
            execution_mode=getattr(decision, "execution_mode", "") if decision else "",
            decision_stage_reached=decision_stage,
            domain_context=forwarded_domain_context,
            metric_context=forwarded_metric_context,
            domain_audit=domain_audit,
            multi_domain_audit={},
        )

        path = self.run_dir / f"{context.run_id}.json"
        path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        return record, path

    def finalize_multi_domain(
        self,
        context: PipelineContext,
        decision: PolicyDecision | None,
        collected_warnings: list[str] | None = None,
    ) -> tuple[RunRecord, Path]:
        status = "failed" if any(result.status == "failed" for result in context.stage_results) else "success"
        warnings = collected_warnings or _collect_warnings(context)
        multi_domain_audit = _extract_multi_domain_audit(context)

        record = RunRecord(
            run_id=context.run_id,
            pipeline_name=context.pipeline_name,
            domain=context.domain,
            stage_results=context.stage_results,
            final_decision=decision,
            started_at=context.metadata.get("started_at", now_iso()),
            finished_at=now_iso(),
            status=status,
            metadata={
                **context.metadata,
                "multi_domain": True,
            },
            run_type="multi_domain",
            domain_name="",
            collected_warnings=warnings,
            eval_scope="",
            task_scope="",
            feature_contract_hash="",
            label_definition="",
            baseline_version_id="",
            anchor_baseline_version_id="",
            comparison_mode="",
            decision_stage_reached="",
            domain_audit={},
            multi_domain_audit=multi_domain_audit,
        )

        path = self.run_dir / f"{context.run_id}.json"
        path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        return record, path
