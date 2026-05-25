from __future__ import annotations

import csv
import json
import math
import os
import site
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
site.addsitedir(site.getusersitepackages())
site.addsitedir(str(ROOT / "streamlit_runtime"))

import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None


A14_DIR = ROOT / "outputs" / "a14"
OUT_DIR = ROOT / "outputs"
OUT_NEXT_STUDY_DIR = ROOT / "outputs_next" / "study"
ASSET_DIR = ROOT / "assets"
MODELING_ROOT = ROOT / "fuchuangsai2" / "output"
MODELING_REPORT_DIR = MODELING_ROOT / "reports"
MAP_FUSION_PATH = ROOT / "fuchuang_shapmapl" / "fuchuang_final" / "output" / "map" / "individual" / "overall_map_fusion.csv"
STUDY_SHAP_PATH = ROOT / "fuchuang_shapmapl" / "fuchuang_final" / "output" / "shap" / "individual" / "shap_individual_study_risk_cls_real_feature.csv"
LIFE_SHAP_PATH = ROOT / "fuchuang_shapmapl" / "fuchuang_final" / "output" / "shap" / "individual" / "shap_life_individual_simple_1775823941.csv"
SPORT_SHAP_PATH = ROOT / "fuchuang_shapmapl" / "fuchuang_final" / "output" / "shap" / "individual" / "shap_individual_sport_risk.csv"

DEFAULT_HARNESS_BASE_URL = os.getenv("HARNESS_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_HARNESS_TIMEOUT_SEC = float(os.getenv("HARNESS_TIMEOUT_SEC", "8"))
DEFAULT_HARNESS_RUN_TIMEOUT_SEC = float(os.getenv("HARNESS_RUN_TIMEOUT_SEC", "360"))
DEFAULT_HARNESS_DOMAINS = ["study", "life", "sport"]
BACKEND_ROOT = Path(os.getenv("HARNESS_BACKEND_ROOT", str(ROOT.parent / "服创赛")))
BACKEND_HARNESS_RUNS_DIR = Path(os.getenv("HARNESS_RUNS_DIR", str(BACKEND_ROOT / "data" / "harness" / "runs")))
BACKEND_FRONTEND_BUNDLE_PATH = Path(
    os.getenv(
        "FRONTEND_BUNDLE_PATH",
        str(BACKEND_ROOT / "data" / "harness" / "frontend_bundle" / "latest_frontend_bundle.json"),
    )
)
DEFAULT_HARNESS_REQUEST = {
    "request_id": "frontend_run_001",
    "domain": "study",
    "run_mode": "review",
    "execution_engine": "harness_v1",
    "input_paths": {},
}

DECISION_LABELS_ZH = {
    "promote_candidate": "提升候选",
    "keep_baseline": "保持基线",
    "eligible_for_comparison": "可进入比较",
    "hold_for_review": "暂缓待复核",
    "running": "运行中",
    "failed": "失败",
}

DIMENSION_LABELS = {
    "study": "学习",
    "life": "生活",
    "sport": "运动",
    "学习": "学习",
    "生活": "生活",
    "运动": "运动",
    "unknown": "未知",
    "": "未知",
}

SPORT_DOMINANT_MIN_RISK = 0.40
STUDY_PATTERN_MIN_RISK = 0.35

MAP_LABELS = {
    "M": "动机",
    "A": "能力",
    "P": "提示",
    "动机": "动机",
    "能力": "能力",
    "提示": "提示",
    "环境": "提示",
    "": "未知",
}

STATUS_LABELS = {
    "running": "运行中",
    "pending": "等待中",
    "queued": "排队中",
    "ready": "已就绪",
    "multi_domain_ready": "多域结果已就绪",
    "completed": "已完成",
    "complete": "已完成",
    "done": "已完成",
    "success": "成功",
    "succeeded": "成功",
    "failed": "失败",
    "error": "异常",
    "unknown": "未知",
    "": "未知",
}

CONTRACT_METRIC_ORDER = ["auc", "f1", "precision", "recall", "rows", "eval_rows", "positive_count"]

PREDICTOR_SCHEMAS = {
    "study": [
        {"key": "avg_score", "label": "平均成绩", "type": "float", "default": 80.0},
        {"key": "avg_gpa", "label": "平均绩点", "type": "float", "default": 3.0},
        {"key": "course_count", "label": "课程数", "type": "int", "default": 6},
        {"key": "fail_count", "label": "不及格门数", "type": "int", "default": 0},
        {"key": "select_course_count", "label": "选课次数", "type": "float", "default": 6.0},
        {"key": "attend_record_count", "label": "考勤记录数", "type": "float", "default": 20.0},
        {"key": "absent_count", "label": "缺勤次数", "type": "float", "default": 0.0},
        {"key": "absent_rate", "label": "缺勤率", "type": "float", "default": 0.0},
        {"key": "sign_count", "label": "签到次数", "type": "float", "default": 10.0},
        {"key": "hw_submit_count", "label": "作业提交数", "type": "float", "default": 6.0},
        {"key": "exam_submit_count", "label": "考试提交数", "type": "float", "default": 2.0},
        {"key": "online_bfb", "label": "线上学习活跃度", "type": "float", "default": 85.0},
        {"key": "task_job_rate", "label": "任务完成率", "type": "float", "default": 0.75},
        {"key": "task_test_avg", "label": "测试均分", "type": "float", "default": 8.0},
        {"key": "task_work_avg", "label": "作业均分", "type": "float", "default": 8.0},
        {"key": "task_exam_avg", "label": "考试均分", "type": "float", "default": 8.0},
        {"key": "task_reply_num", "label": "讨论回复数", "type": "float", "default": 3.0},
    ],
    "life": [
        {"key": "club_event_count", "label": "活动参与次数", "type": "int", "default": 6},
        {"key": "lib_visit_count", "label": "图书馆到访次数", "type": "float", "default": 4.0},
        {"key": "lib_unique_gate", "label": "图书馆出入入口数", "type": "float", "default": 1.0},
        {"key": "door_event_count", "label": "门禁事件数", "type": "float", "default": 20.0},
        {"key": "door_unique_ctrl", "label": "门禁点位数", "type": "float", "default": 2.0},
    ],
    "sport": [
        {"key": "bmi_count", "label": "体测记录数", "type": "int", "default": 1},
        {"key": "fhl_mean", "label": "肺活量均值", "type": "float", "default": 2600.0},
        {"key": "ws_mean", "label": "50米成绩均值", "type": "float", "default": 8.8},
        {"key": "ldty_mean", "label": "立定跳远均值", "type": "float", "default": 180.0},
        {"key": "zwtqq_mean", "label": "坐位体前屈均值", "type": "float", "default": 12.0},
        {"key": "bb_mean", "label": "仰卧起坐/引体向上均值", "type": "float", "default": 20.0},
        {"key": "ywqz_mean", "label": "1000/800米成绩均值", "type": "float", "default": 4.0},
        {"key": "pe_course_count", "label": "体育课程数", "type": "float", "default": 1.0},
        {"key": "daily_daka_sum", "label": "日常打卡总次数", "type": "float", "default": 80.0},
        {"key": "daily_week_count", "label": "打卡周数", "type": "float", "default": 10.0},
        {"key": "run_punch_count", "label": "跑步打卡次数", "type": "float", "default": 50.0},
        {"key": "run_state_mean", "label": "跑步状态均值", "type": "float", "default": 4.0},
    ],
}

FIELD_ALIASES = {
    "student_id": ["student_id", "sid", "学号", "学生id", "学生编号"],
    "term_id": ["term_id", "学期", "term"],
}
for domain_fields in PREDICTOR_SCHEMAS.values():
    for field in domain_fields:
        FIELD_ALIASES[field["key"]] = [field["key"], field["label"]]


COLUMN_LABELS = {
    "student_id": "学生编号",
    "sid": "学生编号",
    "term_id": "学期编号",
    "risk_level": "风险等级",
    "total_risk": "综合风险",
    "dominant_dimension": "主导维度",
    "main_dimension": "主导维度",
    "dominant_MAP": "主导机制",
    "pattern_label": "行为模式",
    "pattern_reason": "模式说明",
    "priority": "优先级",
    "domain": "领域",
    "model": "模型",
    "status": "状态",
    "system_status": "系统状态",
    "decision": "决策",
    "final_decision": "系统决策",
    "source": "数据来源",
    "_source": "数据来源",
    "generated_at": "生成时间",
    "updated_at": "更新时间",
    "built_from_run_id": "运行编号",
    "file": "文件",
    "type": "类型",
    "size_kb": "大小(KB)",
    "student_count": "学生人数",
    "sample_count": "样本数",
    "avg_total_risk": "平均综合风险",
    "auc": "AUC",
    "f1": "F1",
    "precision": "精确率",
    "recall": "召回率",
    "rows": "样本数",
    "eval_rows": "评估样本数",
    "positive_count": "正样本数",
    "study_risk": "学习风险",
    "life_risk": "生活风险",
    "sport_risk": "运动风险",
}
for domain_fields in PREDICTOR_SCHEMAS.values():
    for field in domain_fields:
        COLUMN_LABELS[field["key"]] = field["label"]


def repair_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if "茂" in value or any(token in value for token in ["鐎", "閻", "閸", "鐠", "缂", "闂", "璇", "娅", "鍨", "楂", "涓", "浣"]):
        for encoding in ("gbk", "gb18030"):
            try:
                return value.encode(encoding).decode("utf-8")
            except Exception:
                continue
    return value


def repair_object(value: Any) -> Any:
    if isinstance(value, dict):
        return {repair_object(k): repair_object(v) for k, v in value.items()}
    if isinstance(value, list):
        return [repair_object(item) for item in value]
    return repair_text(value)


def safe_float(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        result = float(value)
        return None if math.isnan(result) else round(result, digits)
    except Exception:
        return None


def fmt_num(value: Any, digits: int = 3) -> str:
    number = safe_float(value, digits)
    return "-" if number is None else f"{number:.{digits}f}"


def fmt_pct(value: Any) -> str:
    number = safe_float(value)
    return "-" if number is None else f"{number * 100:.1f}%"


def file_stamp(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


@st.cache_data
def load_json(path: Path, _stamp: int | None = None) -> Any:
    if not path.exists():
        return {}
    return repair_object(json.loads(path.read_text(encoding="utf-8-sig")))


@st.cache_data
def load_csv(path: Path, _stamp: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return repair_dataframe(pd.read_csv(path))


def repair_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    fixed = df.copy()
    for col in fixed.columns:
        if fixed[col].dtype == object:
            fixed[col] = fixed[col].map(repair_text)
    return fixed


def _pick_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _clamp_zero_one(value: Any) -> float | None:
    number = safe_float(value, 6)
    if number is None:
        return None
    return max(0.0, min(1.0, float(number)))


def _blank_like_mask(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return series.isna() | lowered.isin(["", "nan", "none", "null", "-"])


def _humanize_model_feature_name(feature_name: Any) -> str:
    feature_map = {
        "FEATURE_COURSE_CREDIT_SUM": "课程负担相关信号",
        "FEATURE_GRADE_AVG_SCORE": "平均成绩波动信号",
        "FEATURE_GRADE_CREDIT_SUM": "学业完成度相关信号",
        "FEATURE_GRADE_MIN_SCORE": "课程最低成绩波动信号",
        "FEATURE_TASK_JOB_RATE": "任务完成率信号",
        "FEATURE_ONLINE_BFB": "线上学习活跃信号",
        "FEATURE_ATTEND_RECORD_COUNT": "出勤参与信号",
        "club_active_flag": "社团活动参与情况",
        "club_event_count": "校园活动参与频率",
        "lib_visit_count": "图书馆到访频率",
        "lib_unique_gate": "图书馆出入规律",
        "door_event_count": "门禁出入活跃度",
        "door_unique_ctrl": "日常活动范围变化",
        "zf_score": "综合体测成绩",
        "pred_zf_score": "体测表现趋势",
        "pred_fail_proba": "体测风险概率",
        "daily_daka_sum": "日常打卡持续性",
        "daily_week_count": "周度运动持续性",
        "run_punch_count": "跑步打卡完成度",
        "run_state_mean": "跑步状态稳定性",
        "pe_course_count": "体育课程参与情况",
        "pe_unique_course": "体育课程覆盖情况",
        "ywqz_mean": "耐力表现信号",
        "ws_mean": "速度表现信号",
        "ldty_mean": "爆发力表现信号",
        "zwtqq_mean": "柔韧性表现信号",
        "bb_mean": "力量表现信号",
        "fhl_mean": "肺活量表现信号",
        "bmi_count": "体测记录完整度",
        "year_start": "年级阶段变化",
        "semester": "学期阶段变化",
        "term_order": "学期顺序变化",
    }
    raw = str(feature_name or "").strip()
    if not raw:
        return ""
    if raw in feature_map:
        return feature_map[raw]
    if raw.startswith("FEATURE_"):
        return "关键行为信号"
    return raw


def _load_domain_prediction_fallback(domain: str) -> pd.DataFrame:
    path = _pick_existing_path(
        OUT_DIR / domain / "predictions_full.csv",
        OUT_DIR / domain / "predictions_test.csv",
    )
    if path is None:
        return pd.DataFrame()
    df = load_csv(path, file_stamp(path))
    if df.empty:
        return pd.DataFrame()
    sid_col = "sid" if "sid" in df.columns else ("student_id" if "student_id" in df.columns else "")
    value_col = "pred_prob" if domain == "life" else "pred_fail_proba"
    if not sid_col or value_col not in df.columns:
        return pd.DataFrame()
    working = df[[sid_col, value_col]].copy()
    working[sid_col] = working[sid_col].astype(str)
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[value_col])
    if working.empty:
        return pd.DataFrame()
    result = working.groupby(sid_col, as_index=False)[value_col].mean()
    result[value_col] = result[value_col].map(_clamp_zero_one)
    return result.rename(columns={sid_col: "student_id", value_col: f"{domain}_risk_fallback"})


def _load_domain_shap_fallback(domain: str) -> pd.DataFrame:
    path = {
        "study": STUDY_SHAP_PATH,
        "life": LIFE_SHAP_PATH,
        "sport": SPORT_SHAP_PATH,
    }.get(domain)
    if path is None or not path.exists():
        return pd.DataFrame()
    df = load_csv(path, file_stamp(path))
    if df.empty or "sid" not in df.columns or "feature" not in df.columns:
        return pd.DataFrame()
    working = df[["sid", "feature", "shap_value"]].copy()
    working["sid"] = working["sid"].astype(str)
    working["feature"] = working["feature"].map(_humanize_model_feature_name)
    working["shap_value"] = pd.to_numeric(working["shap_value"], errors="coerce").abs()
    working = working.dropna(subset=["shap_value"])
    working = working[working["feature"].astype(str).str.strip() != ""]
    if working.empty:
        return pd.DataFrame()

    rows: list[dict[str, str]] = []
    for student_id, group in working.groupby("sid"):
        top_features = (
            group.sort_values("shap_value", ascending=False)["feature"]
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        payload: dict[str, str] = {"student_id": student_id}
        for index in range(3):
            payload[f"{domain}_shap_top{index + 1}"] = top_features[index] if index < len(top_features) else ""
        rows.append(payload)
    return pd.DataFrame(rows)


def _complete_missing_domain_risks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    risk_cols = ["study_risk", "life_risk", "sport_risk"]
    for col in risk_cols + ["total_risk"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    for domain in ["life", "sport"]:
        fallback_df = _load_domain_prediction_fallback(domain)
        target_col = f"{domain}_risk"
        if fallback_df.empty or target_col not in working.columns:
            continue
        working = working.merge(fallback_df, on="student_id", how="left")
        fallback_col = f"{domain}_risk_fallback"
        if fallback_col in working.columns:
            working[target_col] = working[target_col].fillna(working[fallback_col])
            working = working.drop(columns=[fallback_col])

    def _fill_row(row: pd.Series) -> pd.Series:
        values = {col: pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0] for col in risk_cols}
        total = pd.to_numeric(pd.Series([row.get("total_risk")]), errors="coerce").iloc[0]
        if pd.notna(total):
            known_sum = sum(float(value) for value in values.values() if pd.notna(value))
            missing_cols = [col for col, value in values.items() if pd.isna(value)]
            if len(missing_cols) == 1:
                values[missing_cols[0]] = _clamp_zero_one(total * 3 - known_sum)
            elif len(missing_cols) == 2:
                shared_value = _clamp_zero_one((total * 3 - known_sum) / 2)
                for col in missing_cols:
                    values[col] = shared_value
        for col, value in values.items():
            row[col] = value
        return row

    working = working.apply(_fill_row, axis=1)
    medians = {
        col: float(working[col].dropna().median()) if col in working.columns and not working[col].dropna().empty else 0.5
        for col in risk_cols
    }
    for col in risk_cols:
        if col in working.columns:
            working[col] = working[col].fillna(medians[col]).map(_clamp_zero_one)
    return working


def _fill_missing_shap_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "student_id" not in df.columns:
        return df
    working = df.copy()
    default_shap = {
        "study": ["课程成绩波动信号", "任务完成情况", "课堂与线上参与节律"],
        "life": ["作息与校园活动节律", "图书馆到访规律", "门禁与日常出入节律"],
        "sport": ["日常打卡持续性", "体育课程参与情况", "体测与身体状态表现"],
    }
    for domain in ["study", "life", "sport"]:
        fallback_df = _load_domain_shap_fallback(domain)
        if not fallback_df.empty:
            working = working.merge(fallback_df, on="student_id", how="left", suffixes=("", "_fallback"))
        for idx in range(1, 4):
            base_col = f"{domain}_shap_top{idx}"
            fallback_col = f"{base_col}_fallback"
            if base_col not in working.columns:
                working[base_col] = ""
            if str(working[base_col].dtype) != "object":
                working[base_col] = working[base_col].astype(object)
            mask = _blank_like_mask(working[base_col])
            if fallback_col in working.columns:
                working.loc[mask, base_col] = working.loc[mask, fallback_col]
                working = working.drop(columns=[fallback_col])
                mask = _blank_like_mask(working[base_col])
            working.loc[mask, base_col] = default_shap[domain][idx - 1]
    return working


def data_source_to_cn(value: Any) -> str:
    key = str(value or "")
    return {
        "contract_live": "后端 HTTP 实时",
        "backend_file": "后端文件桥接",
        "backend_http_bundle": "后端 HTTP Bundle",
        "backend_file_bundle": "后端 Bundle 文件",
        "local_fallback": "前端本地回退",
    }.get(key, key or "未知")


def decision_to_cn(value: Any) -> str:
    key = str(value or "").strip()
    return DECISION_LABELS_ZH.get(key, repair_text(key) if key else "未知")


def status_to_cn(value: Any) -> str:
    key = repair_text(str(value or "").strip())
    return STATUS_LABELS.get(str(key).lower(), key or "未知")


def dimension_to_cn(value: Any) -> str:
    return DIMENSION_LABELS.get(str(value or ""), repair_text(str(value or "")) or "未知")


def map_to_cn(value: Any) -> str:
    return MAP_LABELS.get(str(value or ""), repair_text(str(value or "")) or "未知")


def column_to_cn(value: Any) -> str:
    key = str(value or "").strip()
    return COLUMN_LABELS.get(key, repair_text(key) if key else "")


def display_text_to_cn(value: Any, column: Any = "") -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"

    text = repair_text(str(value)).strip()
    if not text:
        return "-"

    column_key = str(column or "").strip()
    lowered = text.lower()
    if lowered in {"none", "nan", "null"}:
        return "-"
    if lowered == "full":
        return "完整"

    if column_key in {"domain", "领域", "dominant_dimension", "主导维度", "main_dimension", "维度"}:
        return dimension_to_cn(text)
    if column_key in {"decision", "业务决策", "系统决策", "final_decision"}:
        return decision_to_cn(text)
    if column_key in {"status", "执行状态", "系统状态", "system_status"}:
        return status_to_cn(text)
    if column_key in {"source", "数据来源", "_source"}:
        return data_source_to_cn(text)
    if column_key in {"dominant_MAP", "主导机制", "MAP机制"}:
        return map_to_cn(text)

    if lowered in DIMENSION_LABELS:
        return dimension_to_cn(lowered)
    if lowered in STATUS_LABELS:
        return status_to_cn(lowered)
    if lowered in DECISION_LABELS_ZH:
        return decision_to_cn(lowered)
    if text in MAP_LABELS or lowered in MAP_LABELS:
        return map_to_cn(text)
    return text


def prettify_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    display_df = df.copy()
    original_columns = list(display_df.columns)
    for col in original_columns:
        if display_df[col].dtype == object or str(display_df[col].dtype) == "bool":
            display_df[col] = display_df[col].map(lambda value, current_col=col: display_text_to_cn(value, current_col))
    return display_df.rename(columns={col: column_to_cn(col) for col in original_columns})


def sanitize_contract_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for key in CONTRACT_METRIC_ORDER:
        value = payload.get(key)
        number = safe_float(value, 6)
        if number is not None:
            result[key] = float(number)
    return result


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout_sec: float = DEFAULT_HARNESS_TIMEOUT_SEC) -> dict[str, Any] | None:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            content = response.read().decode("utf-8")
            parsed = json.loads(content) if content else {}
            return parsed if isinstance(parsed, dict) else {}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None


def _harness_result_has_domains(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        return False
    final_decision = result.get("final_decision", {})
    if not isinstance(final_decision, dict):
        return False
    domains = final_decision.get("domains", {})
    return isinstance(domains, dict) and bool(domains)


def _harness_result_is_terminal(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        return False
    final_decision = result.get("final_decision", {})
    status = (
        result.get("system_status")
        or result.get("status")
        or (final_decision.get("system_status") if isinstance(final_decision, dict) else "")
        or (final_decision.get("decision") if isinstance(final_decision, dict) else "")
    )
    return str(status or "").strip().lower() not in {"", "running"}


def _read_backend_latest_harness_file(runs_dir: Path = BACKEND_HARNESS_RUNS_DIR) -> dict[str, Any] | None:
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    preferred_terminal: list[tuple[Path, dict[str, Any]]] = []
    preferred_running: list[tuple[Path, dict[str, Any]]] = []
    fallback: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if path.name.startswith("multi_domain"):
            fallback.append((path, payload))
        if _harness_result_has_domains(payload):
            if _harness_result_is_terminal(payload):
                preferred_terminal.append((path, payload))
            else:
                preferred_running.append((path, payload))
    selected = (
        preferred_terminal[0]
        if preferred_terminal
        else (preferred_running[0] if preferred_running else (fallback[0] if fallback else None))
    )
    if selected is None:
        return None
    path, payload = selected
    return {"run_record_path": str(path), "result": payload, "_source": "file"}


def get_frontend_bundle(base_url: str = DEFAULT_HARNESS_BASE_URL) -> dict[str, Any] | None:
    payload = _http_json("GET", f"{base_url}/harness/frontend_bundle/latest")
    if isinstance(payload, dict) and payload.get("schema_version") == "frontend_bundle_v1":
        payload["_source"] = "http_bundle"
        return repair_object(payload)
    if BACKEND_FRONTEND_BUNDLE_PATH.exists():
        try:
            file_payload = json.loads(BACKEND_FRONTEND_BUNDLE_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
        if isinstance(file_payload, dict) and file_payload.get("schema_version") == "frontend_bundle_v1":
            file_payload["_source"] = "file_bundle"
            return repair_object(file_payload)
    return None


def get_harness_latest(base_url: str = DEFAULT_HARNESS_BASE_URL) -> dict[str, Any] | None:
    payload = _http_json("GET", f"{base_url}/harness/result/latest")
    if isinstance(payload, dict):
        payload["_source"] = "http"
        return repair_object(payload)
    return _read_backend_latest_harness_file()


def build_harness_request_payload() -> dict[str, Any]:
    payload = dict(DEFAULT_HARNESS_REQUEST)
    payload["request_id"] = f"frontend_run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return payload


def trigger_harness_run(
    base_url: str = DEFAULT_HARNESS_BASE_URL,
    domains: list[str] | None = None,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload = {
        "domains": domains or DEFAULT_HARNESS_DOMAINS,
        "request": request_payload or DEFAULT_HARNESS_REQUEST,
    }
    return _http_json("POST", f"{base_url}/harness/run", payload=payload, timeout_sec=DEFAULT_HARNESS_RUN_TIMEOUT_SEC)


def normalize_harness_result(raw_payload: Any) -> dict[str, Any]:
    raw = repair_object(raw_payload) if isinstance(raw_payload, dict) else {}
    result = raw.get("result", {}) if isinstance(raw.get("result", {}), dict) else {}
    final_decision = result.get("final_decision", {}) if isinstance(result.get("final_decision", {}), dict) else {}
    domains = final_decision.get("domains", {}) if isinstance(final_decision.get("domains", {}), dict) else {}
    domain_results = result.get("domain_results", {}) if isinstance(result.get("domain_results", {}), dict) else {}

    def text_or_empty(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"none", "null"} else repair_text(text)

    normalized_domains: dict[str, dict[str, Any]] = {}
    normalized_domain_results: dict[str, dict[str, Any]] = {}
    for domain_name in DEFAULT_HARNESS_DOMAINS:
        domain_data = domains.get(domain_name, {}) if isinstance(domains.get(domain_name, {}), dict) else {}
        domain_result_data = domain_results.get(domain_name, {}) if isinstance(domain_results.get(domain_name, {}), dict) else {}
        normalized_domains[domain_name] = {
            "status": text_or_empty(domain_data.get("status", "")),
            "decision": text_or_empty(domain_data.get("decision", "")),
            "trusted_mainline": sanitize_contract_metrics(domain_data.get("trusted_mainline", {})),
            "mainline_task_type": text_or_empty(domain_data.get("mainline_task_type", "")),
            "mainline_validity": bool(domain_data.get("mainline_validity", False)),
            "mainline_frozen": bool(domain_data.get("mainline_frozen", False)),
            "next_optimization_target": text_or_empty(domain_data.get("next_optimization_target", "")),
        }
        normalized_domain_results[domain_name] = {
            "status": text_or_empty(domain_result_data.get("status", "")),
            "metrics": sanitize_contract_metrics(domain_result_data.get("metrics", {})),
            "metric_context": domain_result_data.get("metric_context", {}) if isinstance(domain_result_data.get("metric_context", {}), dict) else {},
            "raw_result": domain_result_data.get("raw_result", {}) if isinstance(domain_result_data.get("raw_result", {}), dict) else {},
        }
    return {
        "available": bool(normalized_domains),
        "source": text_or_empty(raw.get("_source", "")),
        "run_record_path": text_or_empty(raw.get("run_record_path", "")),
        "system_status": text_or_empty(result.get("system_status", "")),
        "final_decision": text_or_empty(final_decision.get("decision", "")),
        "domains": normalized_domains,
        "domain_results": normalized_domain_results,
    }


def get_contract_domain_metrics(harness: dict[str, Any], domain_name: str) -> dict[str, float]:
    domain_results = harness.get("domain_results", {}) if isinstance(harness.get("domain_results", {}), dict) else {}
    domain_info = domain_results.get(domain_name, {}) if isinstance(domain_results.get(domain_name, {}), dict) else {}
    metrics = domain_info.get("metrics", {}) if isinstance(domain_info.get("metrics", {}), dict) else {}
    if metrics:
        return metrics
    final_domains = harness.get("domains", {}) if isinstance(harness.get("domains", {}), dict) else {}
    fallback_info = final_domains.get(domain_name, {}) if isinstance(final_domains.get(domain_name, {}), dict) else {}
    return fallback_info.get("trusted_mainline", {}) if isinstance(fallback_info.get("trusted_mainline", {}), dict) else {}


@st.cache_data
def build_eval_df_from_model_reports(_stamp: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    legacy_reports = [
        ("学习", MODELING_REPORT_DIR / "metrics_summary.json"),
        ("生活", MODELING_REPORT_DIR / "life_metrics_summary.json"),
        ("运动", MODELING_REPORT_DIR / "sports_metrics_summary.json"),
    ]
    for domain_label, path in legacy_reports:
        report = load_json(path, file_stamp(path))
        if not isinstance(report, dict):
            continue
        models = report.get("models", {}) or {}
        for model_name, metrics in models.items():
            metrics = metrics or {}
            rows.append(
                {
                    "domain": domain_label,
                    "task": "分类",
                    "model": repair_text(str(model_name)),
                    "best_val_threshold": metrics.get("threshold"),
                    "val_f1": metrics.get("f1"),
                    "val_auc": metrics.get("cv_auc_mean"),
                    "test_auc": metrics.get("roc_auc"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "samples": metrics.get("samples"),
                }
            )
    if rows:
        return pd.DataFrame(rows)

    def append_current_row(domain_label: str, task_label: str, metrics: dict[str, Any], dataset: dict[str, Any] | None = None) -> None:
        if not isinstance(metrics, dict) or not metrics:
            return
        sample_count = metrics.get("samples")
        if sample_count is None and isinstance(dataset, dict):
            sample_count = dataset.get("test_samples")
        if sample_count is None and isinstance(dataset, dict):
            sample_count = dataset.get("total_samples")

        val_f1 = metrics.get("best_val_f1")
        if val_f1 is None:
            val_f1 = metrics.get("val_f1")
        if val_f1 is None:
            val_f1 = metrics.get("f1")
        if val_f1 is None:
            val_f1 = metrics.get("macro_f1")

        val_auc = metrics.get("best_val_auc")
        if val_auc is None:
            val_auc = metrics.get("val_auc")
        if val_auc is None:
            val_auc = metrics.get("auc")

        test_auc = metrics.get("test_auc")
        if test_auc is None:
            test_auc = metrics.get("roc_auc")
        if test_auc is None:
            test_auc = metrics.get("auc")

        rows.append(
            {
                "domain": domain_label,
                "task": task_label,
                "model": repair_text(str(metrics.get("best_model", "-"))),
                "best_val_threshold": metrics.get("best_threshold"),
                "val_f1": val_f1,
                "val_auc": val_auc,
                "test_auc": test_auc,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "samples": sample_count,
            }
        )

    study_report = load_json(OUT_DIR / "metrics.json", file_stamp(OUT_DIR / "metrics.json"))
    if isinstance(study_report, dict):
        append_current_row("学习", "分类", study_report.get("classification", {}) or {}, study_report.get("dataset", {}) or {})
        append_current_row("学习", "回归", study_report.get("regression", {}) or {}, study_report.get("dataset", {}) or {})

    life_report = load_json(OUT_DIR / "life" / "metrics.json", file_stamp(OUT_DIR / "life" / "metrics.json"))
    if isinstance(life_report, dict):
        life_payload = life_report.get("life", life_report)
        append_current_row("生活", "分类", life_payload if isinstance(life_payload, dict) else {}, {})

    sport_report = load_json(OUT_DIR / "sport" / "metrics.json", file_stamp(OUT_DIR / "sport" / "metrics.json"))
    if isinstance(sport_report, dict):
        append_current_row("运动", "分类", sport_report.get("classification", {}) or {}, {})
        append_current_row("运动", "回归", sport_report.get("regression", {}) or {}, {})

    return pd.DataFrame(rows)


@st.cache_data
def build_eval_df_from_current_outputs(_stamp: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_eval_row(
        *,
        domain_code: str,
        domain_label: str,
        task_type: str,
        task_name: str,
        target_name: str,
        metrics: dict[str, Any],
        dataset: dict[str, Any] | None = None,
        sample_scope: str = "评估样本",
    ) -> None:
        if not isinstance(metrics, dict) or not metrics:
            return

        sample_count = metrics.get("samples")
        if sample_count is None and isinstance(dataset, dict):
            sample_count = dataset.get("test_samples")
        if sample_count is None and isinstance(dataset, dict):
            sample_count = dataset.get("total_samples")

        val_f1 = metrics.get("best_val_f1")
        if val_f1 is None:
            val_f1 = metrics.get("val_f1")
        if val_f1 is None:
            val_f1 = metrics.get("f1")
        if val_f1 is None:
            val_f1 = metrics.get("macro_f1")

        val_auc = metrics.get("best_val_auc")
        if val_auc is None:
            val_auc = metrics.get("val_auc")
        if val_auc is None:
            val_auc = metrics.get("auc")

        rows.append(
            {
                "domain_code": domain_code,
                "domain": domain_label,
                "task_type": task_type,
                "task_name": task_name,
                "target_name": target_name,
                "model": repair_text(str(metrics.get("best_model", "-"))),
                "best_val_threshold": metrics.get("best_threshold"),
                "val_f1": val_f1,
                "val_auc": val_auc,
                "r2": metrics.get("best_val_r2", metrics.get("r2")),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "samples": sample_count,
                "sample_scope": sample_scope,
            }
        )

    study_report = load_json(OUT_DIR / "metrics.json", file_stamp(OUT_DIR / "metrics.json"))
    if isinstance(study_report, dict):
        study_dataset = study_report.get("dataset", {}) or {}
        append_eval_row(
            domain_code="study",
            domain_label="学习",
            task_type="classification",
            task_name="学业预警分类",
            target_name="fail_flag",
            metrics=study_report.get("classification", {}) or {},
            dataset=study_dataset,
            sample_scope="测试集样本",
        )
        append_eval_row(
            domain_code="study",
            domain_label="学习",
            task_type="regression",
            task_name="学业成绩回归",
            target_name="avg_score",
            metrics=study_report.get("regression", {}) or {},
            dataset=study_dataset,
            sample_scope="测试集样本",
        )

    life_report = load_json(OUT_DIR / "life" / "metrics.json", file_stamp(OUT_DIR / "life" / "metrics.json"))
    if isinstance(life_report, dict):
        life_payload = life_report.get("life", life_report)
        append_eval_row(
            domain_code="life",
            domain_label="生活",
            task_type="classification",
            task_name="生活活跃分类",
            target_name="club_active_flag",
            metrics=life_payload if isinstance(life_payload, dict) else {},
            dataset={},
            sample_scope="评估样本",
        )

    sport_report = load_json(OUT_DIR / "sport" / "metrics.json", file_stamp(OUT_DIR / "sport" / "metrics.json"))
    if isinstance(sport_report, dict):
        append_eval_row(
            domain_code="sport",
            domain_label="运动",
            task_type="classification",
            task_name="体测等级分类",
            target_name="zf_grade",
            metrics=sport_report.get("classification", {}) or {},
            dataset={},
            sample_scope="测试集样本",
        )
        append_eval_row(
            domain_code="sport",
            domain_label="运动",
            task_type="regression",
            task_name="体测总分回归",
            target_name="zf_score",
            metrics=sport_report.get("regression", {}) or {},
            dataset={},
            sample_scope="测试集样本",
        )

    return pd.DataFrame(rows)


def _artifact_path(bundle: dict[str, Any], key: str, fallback: Path) -> Path:
    artifact_paths = bundle.get("artifact_paths", {}) if isinstance(bundle.get("artifact_paths", {}), dict) else {}
    candidate = Path(str(artifact_paths.get(key, "")))
    return candidate if candidate.exists() else fallback


def choose_dominant_dimension(study_risk: Any, life_risk: Any, sport_risk: Any) -> str:
    scores = {
        "study": pd.to_numeric(pd.Series([study_risk]), errors="coerce").iloc[0],
        "life": pd.to_numeric(pd.Series([life_risk]), errors="coerce").iloc[0],
        "sport": pd.to_numeric(pd.Series([sport_risk]), errors="coerce").iloc[0],
    }
    ranked = [(label, float(score)) for label, score in scores.items() if pd.notna(score)]
    if not ranked:
        return "unknown"
    ranked.sort(key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    if top_label == "sport" and top_score < SPORT_DOMINANT_MIN_RISK and len(ranked) > 1:
        return ranked[1][0]
    return top_label


def remap_pattern_label(pattern_label: Any, dominant_dimension: Any) -> str:
    label = repair_text(pattern_label)
    dominant = str(dominant_dimension or "").strip().lower()
    if label == "运动薄弱型":
        if dominant == "study":
            return "学习主导风险型"
        if dominant == "life":
            return "生活失衡型"
        return "运动薄弱型"
    return label


def remap_pattern_label_by_risk(pattern_label: Any, dominant_dimension: Any, study_risk: Any, sport_risk: Any) -> str:
    label = repair_text(pattern_label)
    dominant = str(dominant_dimension or "").strip().lower()
    study_value = pd.to_numeric(pd.Series([study_risk]), errors="coerce").iloc[0]
    sport_value = pd.to_numeric(pd.Series([sport_risk]), errors="coerce").iloc[0]
    if label == repair_text("运动薄弱型"):
        if pd.notna(study_value) and float(study_value) >= STUDY_PATTERN_MIN_RISK:
            return repair_text("学习主导风险型")
        if dominant == "life":
            return repair_text("生活失衡型")
        return repair_text("运动薄弱型")
    return label


def remap_pattern_label_by_thresholds(
    pattern_label: Any,
    dominant_dimension: Any,
    study_risk: Any,
    sport_risk: Any,
) -> str:
    label = repair_text(pattern_label)
    dominant = str(dominant_dimension or "").strip().lower()
    study_value = pd.to_numeric(pd.Series([study_risk]), errors="coerce").iloc[0]
    sport_value = pd.to_numeric(pd.Series([sport_risk]), errors="coerce").iloc[0]

    if label == repair_text("运动薄弱型"):
        if pd.notna(sport_value) and float(sport_value) >= SPORT_DOMINANT_MIN_RISK:
            return repair_text("运动薄弱型")
        if pd.notna(study_value) and float(study_value) >= STUDY_PATTERN_MIN_RISK:
            return repair_text("学习主导风险型")
        if dominant == "life":
            return repair_text("生活失衡型")
        return repair_text("能力不足型")

    if label == repair_text("学习主导风险型"):
        if pd.notna(study_value) and float(study_value) >= STUDY_PATTERN_MIN_RISK:
            return repair_text("学习主导风险型")
        return repair_text("能力不足型")

    return label


def _prepare_master_df(master_df: pd.DataFrame) -> pd.DataFrame:
    df = repair_dataframe(master_df)
    if df.empty:
        return df
    numeric_cols = ["life_risk", "study_risk", "sport_risk", "total_risk", "M_score", "A_score", "P_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "student_id" in df.columns:
        df["student_id"] = df["student_id"].astype(str)
    df = _complete_missing_domain_risks(df)
    df = _fill_missing_shap_columns(df)

    map_fusion_df = _load_map_fusion_scores()
    if not map_fusion_df.empty:
        df = df.drop(columns=[col for col in ["M_score", "A_score", "P_score", "dominant_MAP"] if col in df.columns])
        df = df.merge(map_fusion_df, on="student_id", how="left")

    if {"study_risk", "life_risk", "sport_risk"}.issubset(df.columns):
        df["dominant_dimension"] = df.apply(
            lambda row: choose_dominant_dimension(
                row.get("study_risk"),
                row.get("life_risk"),
                row.get("sport_risk"),
            ),
            axis=1,
        )
    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].fillna("未知").astype(str).map(repair_text)
    if "dominant_dimension" in df.columns:
        df["dominant_dimension"] = df["dominant_dimension"].fillna("unknown").astype(str).map(repair_text)
    if "dominant_MAP" in df.columns:
        df["dominant_MAP"] = df["dominant_MAP"].fillna("").astype(str).map(repair_text)
    text_cols = ["pattern_label", "pattern_reason", "profile_text", "intervention_type", "intervention_text", "priority"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).map(repair_text)
    if "pattern_label" in df.columns and "dominant_dimension" in df.columns:
        df["pattern_label"] = df.apply(
            lambda row: remap_pattern_label_by_thresholds(
                row.get("pattern_label"),
                row.get("dominant_dimension"),
                row.get("study_risk"),
                row.get("sport_risk"),
            ),
            axis=1,
        )
    return df


def _build_outputs_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(A14_DIR.glob("*")):
        rows.append(
            {
                "file": path.name,
                "type": "目录" if path.is_dir() else path.suffix.lstrip("."),
                "size_kb": round(path.stat().st_size / 1024, 1) if path.is_file() else None,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return rows


def _load_map_fusion_scores() -> pd.DataFrame:
    if not MAP_FUSION_PATH.exists():
        return pd.DataFrame()
    try:
        map_df = pd.read_csv(MAP_FUSION_PATH, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if map_df.empty or "student_id" not in map_df.columns:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["student_id"] = map_df["student_id"].astype(str)
    result["M_score"] = pd.to_numeric(map_df.get("整体M"), errors="coerce")
    result["A_score"] = pd.to_numeric(map_df.get("整体A"), errors="coerce")
    result["P_score"] = pd.to_numeric(map_df.get("整体P"), errors="coerce")

    dominant_raw = map_df.get("整体主导类型", pd.Series(dtype=str)).fillna("").astype(str)
    result["dominant_MAP"] = dominant_raw.map(
        lambda value: "M" if value.startswith("M主导") else ("A" if value.startswith("A主导") else ("P" if value.startswith("P主导") else ""))
    )
    return result


@st.cache_data
def prepare_data(_cache_version: str = "pattern_remap_v6") -> dict[str, Any]:
    frontend_bundle = get_frontend_bundle()
    bundle_source = str(frontend_bundle.get("_source", "")) if isinstance(frontend_bundle, dict) else ""
    bundle_harness_raw = frontend_bundle.get("harness") if isinstance(frontend_bundle, dict) and isinstance(frontend_bundle.get("harness"), dict) else None
    if isinstance(bundle_harness_raw, dict):
        bundle_harness_raw = dict(bundle_harness_raw)
        bundle_harness_raw["_source"] = "http" if bundle_source == "http_bundle" else "file"
    latest_harness_raw = get_harness_latest()

    bundle_run_path = str(frontend_bundle.get("run_record_path", "")) if isinstance(frontend_bundle, dict) else ""
    latest_run_path = str(latest_harness_raw.get("run_record_path", "")) if isinstance(latest_harness_raw, dict) else ""
    latest_result = latest_harness_raw.get("result", {}) if isinstance(latest_harness_raw, dict) and isinstance(latest_harness_raw.get("result", {}), dict) else {}
    latest_system_status = str(latest_result.get("system_status", ""))

    prefer_latest_harness = bool(
        isinstance(latest_harness_raw, dict)
        and (
            not isinstance(bundle_harness_raw, dict)
            or (latest_run_path and latest_run_path != bundle_run_path)
            or latest_system_status == "running"
        )
    )
    harness_latest_raw = latest_harness_raw if prefer_latest_harness else bundle_harness_raw
    harness_data = normalize_harness_result(harness_latest_raw)
    harness_ready = bool(harness_data.get("system_status") and harness_data.get("domains"))

    if not prefer_latest_harness and bundle_source == "http_bundle":
        data_source = "backend_http_bundle"
    elif not prefer_latest_harness and bundle_source == "file_bundle":
        data_source = "backend_file_bundle"
    elif harness_ready and harness_data.get("source") == "http":
        data_source = "contract_live"
    elif harness_ready and harness_data.get("source") == "file":
        data_source = "backend_file"
    else:
        data_source = "local_fallback"

    bundle_meta = frontend_bundle if isinstance(frontend_bundle, dict) else {}
    master_path = _artifact_path(bundle_meta, "fusion_student_master_table", A14_DIR / "fusion_student_master_table.csv")
    pattern_path = _artifact_path(bundle_meta, "pattern_summary", A14_DIR / "pattern_summary.csv")
    intervention_path = _artifact_path(bundle_meta, "student_intervention", A14_DIR / "student_intervention.csv")
    report_path = _artifact_path(bundle_meta, "student_full_report_multi_agent", A14_DIR / "student_full_report_multi_agent.json")
    group_profile_path = _artifact_path(bundle_meta, "group_profile", A14_DIR / "group_profile.json")
    demo_case_path = _artifact_path(bundle_meta, "demo_case_student", A14_DIR / "demo_case_student.json")

    master_df = _prepare_master_df(load_csv(master_path, file_stamp(master_path)))
    pattern_df = load_csv(pattern_path, file_stamp(pattern_path))
    intervention_df = load_csv(intervention_path, file_stamp(intervention_path))
    report_records = load_json(report_path, file_stamp(report_path))
    report_records = report_records if isinstance(report_records, list) else []
    report_records = repair_object(report_records)
    report_map = {str(item.get("student_id", "")): item for item in report_records if isinstance(item, dict)}

    if not master_df.empty and report_map:
        if "profile_text" not in master_df.columns:
            master_df["profile_text"] = ""
        if "intervention_text" not in master_df.columns:
            master_df["intervention_text"] = ""
        for idx, student_id in master_df["student_id"].astype(str).items():
            report = report_map.get(student_id, {})
            if report:
                if not str(master_df.at[idx, "profile_text"]).strip():
                    master_df.at[idx, "profile_text"] = repair_text(report.get("report_agent") or report.get("summary") or "")
                if not str(master_df.at[idx, "intervention_text"]).strip():
                    master_df.at[idx, "intervention_text"] = repair_text(report.get("intervention_text") or "")

    group_profile = load_json(group_profile_path, file_stamp(group_profile_path))
    demo_case = load_json(demo_case_path, file_stamp(demo_case_path))
    outputs_catalog = _build_outputs_catalog()
    eval_df = build_eval_df_from_current_outputs(
        max(
            file_stamp(MODELING_REPORT_DIR / "metrics_summary.json"),
            file_stamp(MODELING_REPORT_DIR / "life_metrics_summary.json"),
            file_stamp(MODELING_REPORT_DIR / "sports_metrics_summary.json"),
            file_stamp(OUT_DIR / "metrics.json"),
            file_stamp(OUT_DIR / "life" / "metrics.json"),
            file_stamp(OUT_DIR / "sport" / "metrics.json"),
        )
    )

    study_feature_df = load_csv(OUT_DIR / "feature_dataset.csv", file_stamp(OUT_DIR / "feature_dataset.csv"))
    life_feature_df = load_csv(OUT_DIR / "life" / "feature_dataset.csv", file_stamp(OUT_DIR / "life" / "feature_dataset.csv"))
    sport_feature_df = load_csv(OUT_DIR / "sport" / "feature_dataset.csv", file_stamp(OUT_DIR / "sport" / "feature_dataset.csv"))
    sport_pred_path = OUT_DIR / "sport" / "predictions_full.csv"
    if not sport_pred_path.exists():
        sport_pred_path = OUT_DIR / "sport" / "predictions_test.csv"
    sport_pred_df = load_csv(sport_pred_path, file_stamp(sport_pred_path))

    return {
        "frontend_bundle": bundle_meta,
        "data_source": data_source,
        "harness": harness_data,
        "harness_raw": harness_latest_raw if isinstance(harness_latest_raw, dict) else {},
        "master_df": master_df,
        "pattern_df": pattern_df,
        "intervention_df": intervention_df,
        "report_records": report_records,
        "report_map": report_map,
        "group_profile": group_profile if isinstance(group_profile, dict) else {},
        "demo_case": demo_case if isinstance(demo_case, dict) else {},
        "outputs_catalog": outputs_catalog,
        "eval_df": eval_df,
        "study_feature_df": study_feature_df,
        "life_feature_df": life_feature_df,
        "sport_feature_df": sport_feature_df,
        "sport_pred_df": sport_pred_df,
    }


def detect_csv_columns(columns: list[str]) -> dict[str, str]:
    normalized = {str(col).strip().lower(): str(col) for col in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            hit = normalized.get(str(alias).strip().lower())
            if hit:
                mapping[canonical] = hit
                break
    return mapping


def load_uploaded_table(uploaded_file: Any) -> pd.DataFrame:
    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return repair_dataframe(pd.read_excel(uploaded_file))
    return repair_dataframe(pd.read_csv(uploaded_file))


def build_predictor_defaults(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    frame_map = {
        "study": data["study_feature_df"],
        "life": data["life_feature_df"],
        "sport": data["sport_feature_df"],
    }
    for domain, schema in PREDICTOR_SCHEMAS.items():
        defaults: dict[str, Any] = {}
        df = frame_map[domain]
        for field in schema:
            key = field["key"]
            fallback = field["default"]
            if not df.empty and key in df.columns:
                series = pd.to_numeric(df[key], errors="coerce")
                value = series.median()
                defaults[key] = fallback if pd.isna(value) else float(value)
            else:
                defaults[key] = fallback
        result[domain] = defaults
    return result


def standardize_record(record: dict[str, Any], data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    defaults = build_predictor_defaults(data)
    standardized: dict[str, dict[str, Any]] = {}
    for domain, schema in PREDICTOR_SCHEMAS.items():
        domain_record = defaults[domain].copy()
        for field in schema:
            if field["key"] in record and record[field["key"]] not in ["", None]:
                domain_record[field["key"]] = record[field["key"]]
        standardized[domain] = domain_record
    standardized["meta"] = {
        "student_id": str(record.get("student_id", "new_student")),
        "term_id": str(record.get("term_id", "2025-2026-T1")),
    }
    return standardized


@st.cache_resource
def load_prediction_runtime() -> dict[str, Any] | None:
    if joblib is None:
        return None
    try:
        study_models_dir = OUT_DIR / "models"
        study_backend = "legacy"
        study_meta = load_json(study_models_dir / "model_metadata.json", file_stamp(study_models_dir / "model_metadata.json"))
        study_reg = joblib.load(study_models_dir / "best_regression_model.joblib")
        study_cls = joblib.load(study_models_dir / "best_classification_model.joblib")
        study_bundle = None
        study_feature_columns: list[str] = []

        aaa_models_dir = OUT_NEXT_STUDY_DIR / "models"
        if (aaa_models_dir / "study_model.pkl").exists():
            study_backend = "aaa"
            study_bundle = joblib.load(aaa_models_dir / "study_model.pkl")
            study_meta = load_json(aaa_models_dir / "model_metadata.json", file_stamp(aaa_models_dir / "model_metadata.json")) or study_meta
            study_reg = None
            study_cls = None
            study_feature_columns = list((study_bundle.get("config", {}) or {}).get("feature_columns", []))

        return {
            "study_backend": study_backend,
            "study_reg": study_reg,
            "study_cls": study_cls,
            "study_bundle": study_bundle,
            "study_feature_columns": study_feature_columns,
            "study_meta": study_meta,
            "life_model": joblib.load(OUT_DIR / "life" / "models" / "best_life_model.joblib"),
            "life_metrics": load_json(OUT_DIR / "life" / "metrics.json", file_stamp(OUT_DIR / "life" / "metrics.json")),
            "sport_model": joblib.load(OUT_DIR / "sport" / "regression" / "best_sport_regression_model.joblib"),
            "sport_metrics": load_json(OUT_DIR / "sport" / "metrics.json", file_stamp(OUT_DIR / "sport" / "metrics.json")),
        }
    except Exception:
        return None


def build_feature_row(
    df: pd.DataFrame,
    standardized: dict[str, Any],
    id_cols: list[str],
    label_cols: list[str],
    student_id: str,
    term_id: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    base: dict[str, Any] = {}
    numeric_cols = set(df.select_dtypes(include="number").columns)
    for col in df.columns:
        if col in id_cols:
            continue
        if col in label_cols:
            value = pd.to_numeric(df[col], errors="coerce").median() if col in numeric_cols else (df[col].mode().iloc[0] if not df[col].mode().empty else 0)
            base[col] = 0 if pd.isna(value) else value
        elif col in standardized:
            base[col] = standardized[col]
        elif col in numeric_cols:
            value = pd.to_numeric(df[col], errors="coerce").median()
            base[col] = 0 if pd.isna(value) else float(value)
        else:
            mode = df[col].mode()
            base[col] = "" if mode.empty else mode.iloc[0]
    base["sid"] = student_id
    base["term_id"] = term_id
    return pd.DataFrame([base], columns=df.columns)


def score_new_student(standardized: dict[str, dict[str, Any]], data: dict[str, Any]) -> dict[str, Any] | None:
    runtime = load_prediction_runtime()
    if runtime is None:
        return None
    student_id = standardized["meta"]["student_id"]
    term_id = standardized["meta"]["term_id"]

    study_df = build_feature_row(
        data["study_feature_df"],
        standardized["study"],
        ["sid", "term_id"],
        ["avg_score", "avg_gpa", "fail_count", "fail_flag"],
        student_id,
        term_id,
    )
    life_df = build_feature_row(
        data["life_feature_df"],
        standardized["life"],
        ["sid", "term_id", "year_start", "semester", "term_order"],
        ["club_active_flag", "club_event_count"],
        student_id,
        term_id,
    )
    sport_df = build_feature_row(
        data["sport_feature_df"],
        standardized["sport"],
        ["sid", "term_id", "year_start", "semester", "term_order"],
        ["zf_score", "zf_grade"],
        student_id,
        term_id,
    )

    study_features = [c for c in study_df.columns if c not in ["sid", "term_id", "avg_score", "avg_gpa", "fail_count", "fail_flag"]]
    life_features = [c for c in life_df.columns if c not in ["sid", "term_id", "year_start", "semester", "term_order", "club_active_flag", "club_event_count"]]
    sport_features = [c for c in sport_df.columns if c not in ["sid", "term_id", "year_start", "semester", "term_order", "zf_score", "zf_grade"]]

    if runtime.get("study_backend") == "aaa" and runtime.get("study_bundle") is not None:
        bundle = runtime["study_bundle"]
        feature_columns = list(runtime.get("study_feature_columns", []))
        if feature_columns:
            x = pd.DataFrame([standardized["study"]]).reindex(columns=feature_columns)
            x = x.apply(pd.to_numeric, errors="coerce")
            model = bundle.get("primary_model")
            if hasattr(model, "predict_proba"):
                study_prob = float(model.predict_proba(x)[:, 1][0])
            else:
                study_prob = float(model.predict(x)[0])
        else:
            study_prob = 0.5
        study_score = None
        study_threshold = float(runtime["study_meta"].get("classification", {}).get("best_threshold", 0.5))
    else:
        study_prob = float(runtime["study_cls"].predict_proba(study_df[study_features])[:, 1][0])
        study_score = float(runtime["study_reg"].predict(study_df[study_features])[0])
        study_threshold = float(runtime["study_meta"].get("classification", {}).get("best_threshold", 0.5))

    life_prob = float(runtime["life_model"].predict_proba(life_df[life_features])[:, 1][0])
    life_threshold = float(runtime["life_metrics"].get("best_threshold", 0.5))
    sport_pred = float(runtime["sport_model"].predict(sport_df[sport_features])[0])

    sport_reference = pd.to_numeric(data["sport_pred_df"].get("pred_zf_score"), errors="coerce") if not data["sport_pred_df"].empty else pd.Series(dtype=float)
    ref_min = sport_reference.min() if not sport_reference.empty else sport_pred
    ref_max = sport_reference.max() if not sport_reference.empty else sport_pred
    if pd.isna(ref_min) or pd.isna(ref_max) or ref_max == ref_min:
        sport_risk = 0.5
    else:
        sport_risk = 1 - ((sport_pred - ref_min) / (ref_max - ref_min))
    sport_risk = max(0.0, min(1.0, float(sport_risk)))

    total_risk = float((study_prob + life_prob + sport_risk) / 3)
    master_df = data["master_df"]
    valid_total = master_df["total_risk"].dropna()
    high_cut = valid_total.quantile(0.8) if not valid_total.empty else 0.66
    low_cut = valid_total.quantile(0.2) if not valid_total.empty else 0.33
    if total_risk >= high_cut:
        risk_level = "高风险"
    elif total_risk <= low_cut:
        risk_level = "低风险"
    else:
        risk_level = "中风险"

    dimension_scores = {"学习": study_prob, "生活": life_prob, "运动": sport_risk}
    dominant_dimension = choose_dominant_dimension(study_prob, life_prob, sport_risk)
    return {
        "student_id": student_id,
        "term_id": term_id,
        "study_risk": study_prob,
        "life_risk": life_prob,
        "sport_risk": sport_risk,
        "pred_avg_score": study_score,
        "study_threshold": study_threshold,
        "life_threshold": life_threshold,
        "sport_pred_score": sport_pred,
        "total_risk": total_risk,
        "risk_level": risk_level,
        "dominant_dimension": dominant_dimension,
    }
