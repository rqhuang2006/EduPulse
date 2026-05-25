from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st


TEACHER_PAGES = [
    "总览工作台",
    "成果总览",
    "指标验证台",
    "群体画像",
    "学生档案",
    "预警干预",
    "新增预测",
    "AI 助手",
    "链路追踪",
]

STUDENT_PAGES = [
    "我的主页",
    "状态自评",
    "AI 树洞",
]


def ensure_state(master_df: pd.DataFrame) -> None:
    student_ids = master_df.get("student_id", pd.Series(dtype=str)).astype(str).tolist() if not master_df.empty else []
    defaults: dict[str, Any] = {
        "view_role": "教师/管理端",
        "current_page": TEACHER_PAGES[0],
        "pending_page_nav": None,
        "pending_selected_student": None,
        "selected_student": student_ids[0] if student_ids else "",
        "assistant_history": [],
        "assistant_pending_prompt": "",
        "self_assessment_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if student_ids and st.session_state.selected_student not in student_ids:
        st.session_state.selected_student = student_ids[0]
    if st.session_state.current_page not in TEACHER_PAGES + STUDENT_PAGES:
        st.session_state.current_page = TEACHER_PAGES[0]


def get_pages(view_role: str) -> list[str]:
    return TEACHER_PAGES if view_role == "教师/管理端" else STUDENT_PAGES


def apply_filters(
    master_df: pd.DataFrame,
    risk_levels: list[str] | None = None,
    dimensions: list[str] | None = None,
    patterns: list[str] | None = None,
    min_total_risk: float = 0.0,
) -> pd.DataFrame:
    if master_df.empty:
        return master_df.copy()
    filtered = master_df.copy()
    if risk_levels:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    if dimensions:
        filtered = filtered[filtered["dominant_dimension"].isin(dimensions)]
    if patterns:
        filtered = filtered[filtered["pattern_label"].isin(patterns)]
    filtered = filtered[filtered["total_risk"].fillna(0) >= min_total_risk]
    return filtered.reset_index(drop=True)


def get_student_row(df: pd.DataFrame, student_id: str) -> dict[str, Any]:
    if df.empty or not student_id or "student_id" not in df.columns:
        return {}
    hit = df[df["student_id"].astype(str) == str(student_id)]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def parse_student_id_from_text(prompt: str, student_ids: list[str]) -> str | None:
    if not prompt:
        return None
    id_set = set(student_ids)
    for token in re.findall(r"[A-Za-z0-9_]{6,}", prompt):
        if token in id_set:
            return token
    return None
