from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from study_agent import (
        DELIVERABLE_DOCS_DIR,
        DM_DIR,
        FUSION_CONFIG_PATH,
        MODEL_COMPARISON_PATH,
        MODEL_SELECTION_PATH,
        ROOT,
        THRESHOLD_SELECTION_PATH,
        THRESHOLD_TUNING_PATH,
        json_default,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .study_agent import (
        DELIVERABLE_DOCS_DIR,
        DM_DIR,
        FUSION_CONFIG_PATH,
        MODEL_COMPARISON_PATH,
        MODEL_SELECTION_PATH,
        ROOT,
        THRESHOLD_SELECTION_PATH,
        THRESHOLD_TUNING_PATH,
        json_default,
        write_json,
    )


FORMAL_CONFIG_PATH = ROOT / "data" / "deliverables" / "study" / "model" / "study_model_config.json"
LLM_REVIEW_PATH = DM_DIR / "study_llm_review.json"
EVOLUTION_LLM_REVIEW_PATH = DM_DIR / "study_evolution_llm_review.json"
DELIVERABLE_LLM_REVIEW_PATH = DELIVERABLE_DOCS_DIR / "study_llm_review.json"
LLM_EXPLANATION_PREVIEW_PATH = DM_DIR / "study_llm_explanation_preview.json"
LLM_TRACE_PATH = ROOT / "logs" / "study_llm_trace.jsonl"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_request(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    request_path = Path(path)
    if not request_path.is_absolute():
        request_path = ROOT / request_path
    return load_json(request_path)


def llm_config_from_request(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "llm_enable": bool(request.get("llm_enable", False)),
        "llm_provider": request.get("llm_provider", "mock"),
        "llm_model": request.get("llm_model") or DEFAULT_QWEN_MODEL,
        "llm_task_type": request.get("llm_task_type", "model_review"),
        "llm_temperature": float(request.get("llm_temperature", 0.2)),
        "llm_max_tokens": int(request.get("llm_max_tokens", 1200)),
        "llm_timeout_seconds": int(request.get("llm_timeout_seconds", 60)),
        "llm_use_for_explanation": bool(request.get("llm_use_for_explanation", False)),
        "llm_use_for_model_review": bool(request.get("llm_use_for_model_review", True)),
        "fallback_to_mock": bool(request.get("fallback_to_mock", True)),
        "llm_required": bool(request.get("llm_required", False)),
    }


def print_llm_config(request: dict[str, Any]) -> None:
    config = llm_config_from_request(request)
    printable = {
        "llm_provider": config["llm_provider"],
        "llm_model": config["llm_model"],
        "llm_task_type": config["llm_task_type"],
        "llm_temperature": config["llm_temperature"],
        "llm_max_tokens": config["llm_max_tokens"],
        "llm_timeout_seconds": config["llm_timeout_seconds"],
        "DASHSCOPE_BASE_URL": os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
        "DASHSCOPE_API_KEY_detected": bool(os.environ.get("DASHSCOPE_API_KEY")),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=json_default))


def append_llm_trace(record: dict[str, Any]) -> None:
    LLM_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LLM_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def weighted_score(row: pd.Series, mode: str) -> float:
    if mode == "auc_priority":
        weights = {"auc": 0.45, "f1": 0.2, "recall": 0.15, "precision": 0.1, "coverage": 0.1}
    elif mode == "business_risk":
        weights = {"auc": 0.3, "f1": 0.25, "recall": 0.25, "precision": 0.1, "coverage": 0.1}
    else:
        weights = {"auc": 0.35, "f1": 0.25, "recall": 0.2, "precision": 0.1, "coverage": 0.1}
    return float(sum(float(row.get(metric, 0) or 0) * weight for metric, weight in weights.items()))


def candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "feature_group",
        "model_name",
        "threshold_strategy",
        "threshold",
        "auc",
        "f1",
        "recall",
        "precision",
        "coverage",
        "robustness_score",
        "auc_priority_score",
        "business_risk_score",
    ]
    return {key: row.get(key) for key in keys if key in row}


def collect_model_context() -> dict[str, Any]:
    comparison = pd.read_csv(MODEL_COMPARISON_PATH) if MODEL_COMPARISON_PATH.exists() else pd.DataFrame()
    formal_config = load_json(FORMAL_CONFIG_PATH)
    model_selection = load_json(MODEL_SELECTION_PATH)
    threshold_selection = load_json(THRESHOLD_SELECTION_PATH)
    fusion_config = load_json(FUSION_CONFIG_PATH)
    evolution_comparison_path = DM_DIR / "study_evolution_comparison.csv"
    evolution_selection = load_json(DM_DIR / "study_evolution_selection.json")
    publish_candidate = load_json(DM_DIR / "study_evolution_publish_candidate.json")
    evolution_top = []
    if evolution_comparison_path.exists():
        evolution_comparison = pd.read_csv(evolution_comparison_path)
        if not evolution_comparison.empty and "robustness_score" in evolution_comparison.columns:
            evolution_top = evolution_comparison.sort_values("robustness_score", ascending=False).head(5).to_dict(orient="records")

    if not comparison.empty:
        comparison["auc_priority_score"] = comparison.apply(lambda row: weighted_score(row, "auc_priority"), axis=1)
        comparison["business_risk_score"] = comparison.apply(lambda row: weighted_score(row, "business_risk"), axis=1)
        comparison["balanced_score"] = comparison.apply(lambda row: weighted_score(row, "balanced"), axis=1)
        top_auc = comparison.sort_values("auc", ascending=False).head(5).to_dict(orient="records")
        top_robust = comparison.sort_values("robustness_score", ascending=False).head(5).to_dict(orient="records")
        top_business = comparison.sort_values("business_risk_score", ascending=False).head(5).to_dict(orient="records")
    else:
        top_auc, top_robust, top_business = [], [], []

    return {
        "formal_model_config": formal_config,
        "model_selection": model_selection,
        "threshold_selection": threshold_selection,
        "fusion_config": fusion_config,
        "top_candidates_by_auc": [candidate_summary(row) for row in top_auc],
        "top_candidates_by_robustness": [candidate_summary(row) for row in top_robust],
        "top_candidates_by_business_risk_score": [candidate_summary(row) for row in top_business],
        "evolution_selection": evolution_selection,
        "evolution_publish_candidate": publish_candidate,
        "top_evolution_candidates": evolution_top,
    }


def build_prompt(context: dict[str, Any], task_type: str) -> str:
    compact_context = json.dumps(context, ensure_ascii=False, indent=2, default=json_default)
    if len(compact_context) > 9000:
        compact_context = compact_context[:9000] + "\n...<truncated>"
    return f"""你是学习域 StudyAgent 的模型评测助手。你只负责读取已有模型比较/选型/阈值调优结果并给出建议，不允许决定 degraded/fallback/failed，也不允许声称已经训练或发布模型。

任务类型: {task_type}

请输出中文 Markdown，包含:
1. 当前主模型/候选模型是否合理
2. challenger 是否值得替换正式模型
3. AUC、F1、Recall、Precision 的优缺点
4. degraded 风险和数据覆盖风险说明
5. 下一步优化建议
6. 如果看到 AUC 0.95 的历史结果，应如何审计其可信度

输入上下文:
{compact_context}
"""


def parsed_summary_from_text(text: str, provider: str, context: dict[str, Any]) -> dict[str, Any]:
    formal_valid = context.get("formal_model_config", {}).get("metrics", {}).get("valid", {})
    selection = context.get("model_selection", {})
    top_auc = context.get("top_candidates_by_auc", [{}])[0] if context.get("top_candidates_by_auc") else {}
    return {
        "provider": provider,
        "formal_valid_auc": formal_valid.get("auc"),
        "formal_valid_f1": formal_valid.get("f1"),
        "formal_valid_recall": formal_valid.get("recall"),
        "selected_primary_model": selection.get("selected_primary_model"),
        "selected_feature_group": selection.get("selected_feature_group"),
        "best_candidate_auc": top_auc.get("auc"),
        "best_candidate_model": top_auc.get("model_name"),
        "contains_response": bool(text.strip()),
    }


def mock_response(context: dict[str, Any], task_type: str) -> str:
    formal_valid = context.get("formal_model_config", {}).get("metrics", {}).get("valid", {})
    selection = context.get("model_selection", {})
    top_auc = context.get("top_candidates_by_auc", [{}])[0] if context.get("top_candidates_by_auc") else {}
    top_business = context.get("top_candidates_by_business_risk_score", [{}])[0] if context.get("top_candidates_by_business_risk_score") else {}
    return (
        "## Mock LLM Model Review\n\n"
        f"- 正式模型 valid AUC={formal_valid.get('auc')}, F1={formal_valid.get('f1')}, Recall={formal_valid.get('recall')}。\n"
        f"- 当前选型 primary={selection.get('selected_primary_model')}，feature_group={selection.get('selected_feature_group')}。\n"
        f"- 候选最高 AUC 模型={top_auc.get('model_name')} / {top_auc.get('feature_group')}，AUC={top_auc.get('auc')}。\n"
        f"- 业务风险分更优候选={top_business.get('model_name')} / {top_business.get('feature_group')}，Recall={top_business.get('recall')}。\n"
        "- 建议：若 harness 以 AUC 为主，暂不替换正式主模型；若业务以召回为主，可发布 challenger 做并行观察。\n"
        "- AUC 0.95 需要重点审计标签泄露、时间穿越、同学期成绩直接决定标签、以及随机切分泄露。\n"
        f"- 任务类型: {task_type}。\n"
    )


def call_qwen(prompt: str, config: dict[str, Any]) -> str:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set; cannot call qwen provider.")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL)
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is not installed; cannot call DashScope compatible API.") from exc

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=config["llm_timeout_seconds"])
    response = client.chat.completions.create(
        model=config["llm_model"],
        messages=[
            {"role": "system", "content": "你是严谨的教育数据建模评测助手，只基于给定结果作答。"},
            {"role": "user", "content": prompt},
        ],
        temperature=config["llm_temperature"],
        max_tokens=config["llm_max_tokens"],
    )
    return response.choices[0].message.content or ""


def build_llm_advice(request: dict[str, Any] | None = None, provider: str | None = None) -> dict[str, Any]:
    request = dict(request or {})
    config = llm_config_from_request(request)
    if provider:
        config["llm_provider"] = provider
    request_id = str(request.get("request_id", "study_llm_standalone_request"))
    task_type = config["llm_task_type"]
    source_paths = {
        "model_comparison": str(MODEL_COMPARISON_PATH),
        "model_selection": str(MODEL_SELECTION_PATH),
        "threshold_tuning": str(THRESHOLD_TUNING_PATH),
        "threshold_selection": str(THRESHOLD_SELECTION_PATH),
        "formal_model_config": str(FORMAL_CONFIG_PATH),
        "fusion_config": str(FUSION_CONFIG_PATH),
    }
    context = collect_model_context()
    prompt = build_prompt(context, task_type)
    prompt_preview = prompt[:1200]
    provider_name = config["llm_provider"]
    start = time.perf_counter()
    response_text = ""
    response_status = "success"
    error_message = ""
    used_provider = provider_name

    try:
        if not config["llm_enable"]:
            response_status = "skipped"
            response_text = "LLM disabled by request."
        elif provider_name == "mock":
            response_text = mock_response(context, task_type)
        elif provider_name == "qwen":
            response_text = call_qwen(prompt, config)
        else:
            raise ValueError(f"Unsupported llm_provider: {provider_name}")
    except Exception as exc:
        response_status = "degraded"
        error_message = str(exc)
        if config["fallback_to_mock"]:
            used_provider = "mock"
            response_text = mock_response(context, task_type)
            response_text = f"[Qwen call failed, fallback to mock]\n原因: {error_message}\n\n{response_text}"
        else:
            response_text = ""

    latency_ms = int((time.perf_counter() - start) * 1000)
    review = {
        "request_id": request_id,
        "provider": used_provider,
        "requested_provider": provider_name,
        "model": config["llm_model"],
        "task_type": task_type,
        "prompt_preview": prompt_preview,
        "response_text": response_text,
        "parsed_summary": parsed_summary_from_text(response_text, used_provider, context),
        "source_paths": source_paths,
        "created_at": now_iso(),
        "response_status": response_status,
        "error_message": error_message,
        "latency_ms": latency_ms,
    }
    write_json(LLM_REVIEW_PATH, review)
    if task_type == "evolution_review":
        write_json(EVOLUTION_LLM_REVIEW_PATH, review)
    DELIVERABLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LLM_REVIEW_PATH, DELIVERABLE_LLM_REVIEW_PATH)
    if task_type == "explanation_enhancement" or config["llm_use_for_explanation"]:
        write_json(
            LLM_EXPLANATION_PREVIEW_PATH,
            {
                "request_id": request_id,
                "provider": used_provider,
                "model": config["llm_model"],
                "response_text": response_text,
                "created_at": review["created_at"],
            },
        )

    append_llm_trace(
        {
            "timestamp": now_iso(),
            "request_id": request_id,
            "provider": provider_name,
            "effective_provider": used_provider,
            "model": config["llm_model"],
            "task_type": task_type,
            "source_files": list(source_paths.values()),
            "prompt_length": len(prompt),
            "response_status": response_status,
            "latency_ms": latency_ms,
            "error_message": error_message,
        }
    )
    return review


def run_llm_assistant(request_path: str | Path | None = None, provider: str | None = None, print_config_only: bool = False) -> dict[str, Any]:
    request = read_request(request_path)
    if print_config_only:
        print_llm_config(request)
        return {}
    return build_llm_advice(request=request, provider=provider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run study LLM assistant over model comparison outputs.")
    parser.add_argument("--request", help="Path to StudyAgent request JSON.", default=None)
    parser.add_argument("--provider", choices=["mock", "qwen"], help="Override llm_provider from request.", default=None)
    parser.add_argument("--print-config", action="store_true", help="Print LLM call parameters without calling provider.")
    args = parser.parse_args()
    result = run_llm_assistant(args.request, provider=args.provider, print_config_only=args.print_config)
    if result:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
