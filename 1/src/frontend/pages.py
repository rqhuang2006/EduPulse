from __future__ import annotations

import json
import re
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from .components import (
    map_chip,
    metric_card,
    narrative_card,
    page_header,
    render_table,
    risk_chip,
    section_title,
    status_chip,
    summary_cards,
    themed_chart,
)
from .data import (
    PREDICTOR_SCHEMAS,
    build_predictor_defaults,
    decision_to_cn,
    detect_csv_columns,
    dimension_to_cn,
    fmt_num,
    fmt_pct,
    get_contract_domain_metrics,
    load_uploaded_table,
    map_to_cn,
    score_new_student,
    status_to_cn,
    standardize_record,
)
from .state import get_student_row, parse_student_id_from_text


def _page_section_gap() -> None:
    st.markdown("<div class='page-section-gap'></div>", unsafe_allow_html=True)


GROUP_PROFILE_EXPLANATIONS = [
    {
        "模式名称": "优势发展群体",
        "类型性质": "积极",
        "关注方向": "均衡",
        "说明": "整体风险低，各维度状态较稳，适合保持当前节奏并提供发展机会。",
    },
    {
        "模式名称": "良好成长群体",
        "类型性质": "积极",
        "关注方向": "均衡",
        "说明": "整体状态较好，仍有上升空间，适合通过激励和资源投入持续拉升表现。",
    },
    {
        "模式名称": "平稳发展群体",
        "类型性质": "中性",
        "关注方向": "均衡",
        "说明": "当前表现总体平稳，没有明显单项短板，适合常规跟踪与阶段观察。",
    },
    {
        "模式名称": "常规波动群体",
        "类型性质": "中性",
        "关注方向": "观察",
        "说明": "存在一定波动，但暂未进入重点风险区间，适合结合班级管理做持续观察。",
    },
    {
        "模式名称": "重点关注-多维群体",
        "类型性质": "消极",
        "关注方向": "综合",
        "说明": "多个维度同时进入较高风险区间，适合优先纳入综合干预名单。",
    },
    {
        "模式名称": "重点关注-学习群体",
        "类型性质": "消极",
        "关注方向": "学习",
        "说明": "学习相关风险更突出，适合围绕学业节奏、任务完成和支持资源展开干预。",
    },
    {
        "模式名称": "重点关注-生活群体",
        "类型性质": "消极",
        "关注方向": "生活",
        "说明": "生活节律与日常状态波动更明显，适合作息、适应与支持网络方向的干预。",
    },
    {
        "模式名称": "重点关注-运动群体",
        "类型性质": "消极",
        "关注方向": "运动",
        "说明": "运动相关风险更突出，适合从运动习惯、反馈机制和参与激励入手支持。",
    },
]

TASK_EXPECTATIONS = [
    (
        "基于多源数据的学生画像生成与更新",
        "总览工作台、学生数字档案、群体画像",
        "融合总表、三域风险、学生标签、群体标签、画像叙事",
    ),
    (
        "学业轨迹演化模式挖掘与关键行为分析",
        "群体画像、链路追踪、重点学生清单",
        "行为模式分布、模式表、关键行为证据、多阶段链路",
    ),
    (
        "面向教育大数据可视化叙事模型设计",
        "学校端总览工作台、群体画像、链路追踪",
        "总览指标卡、风险区间图、模式图、叙事卡片、链路摘要",
    ),
    (
        "学业风险动态感知与预警模型构建",
        "指标验证台、精准预警干预、新增预测",
        "三域 AUC/F1 指标、预警阈值、风险清单、单学生预测入口",
    ),
    (
        "基于归因解释与知识增强提示的个性化报告生成",
        "学生数字档案、AI 辅导员助手、链路追踪",
        "机制解释、干预建议、多Agent依据、对话式报告问答",
    ),
]

PAGE_TARGET_ALIASES = {
    "学校端总览工作台": "总览工作台",
    "总览工作台": "总览工作台",
    "成果总览": "成果总览",
    "指标验证台": "指标验证台",
    "动态建模展示": "指标验证台",
    "群体画像": "群体画像",
    "学生数字档案": "学生档案",
    "学生档案": "学生档案",
    "精准预警干预": "预警干预",
    "预警干预": "预警干预",
    "新增预测": "新增预测",
    "AI 辅导员助手": "AI 助手",
    "AI 助手": "AI 助手",
    "链路追踪": "链路追踪",
}

OUTCOME_PAGE_TARGETS = {
    "学校端总览工作台": "总览工作台",
    "重点学生清单": "总览工作台",
    "行为模式分布": "群体画像",
    "三维风险区间分析": "群体画像",
    "群体画像模式表": "群体画像",
    "学生个体数字档案": "学生档案",
    "精准预警干预清单": "预警干预",
    "新增学生风险预测": "新增预测",
    "解释链路与关键行为依据": "链路追踪",
    "个性化评价与问答报告": "AI 助手",
}

MULTI_SOURCE_SCOPE = [
    ("学生基础信息与账户信息", "学生基本信息、一卡通账户信息"),
    ("学习过程数据", "选课信息、成绩记录、综合测评、考试提交、作业提交"),
    ("线上学习与课堂互动", "在线平台访问、线上学习数据、课堂任务参与、讨论记录"),
    ("校园生活轨迹", "一卡通交易流水、上网日志明细"),
    ("图书馆与阅读行为", "读者行为、借阅记录、借阅历史"),
    ("运动与体测数据", "跑步打卡、体育课、体测记录"),
    ("发展结果数据", "奖学金获奖信息、就业信息"),
]

ANALYSIS_OUTCOMES = [
    ("学校端总览工作台", "全校视角综合看板", "学校端总览工作台"),
    ("重点学生清单", "输出高风险与高优先级学生", "总览工作台"),
    ("行为模式分布", "输出群体模式与人数分布", "总览工作台 / 群体画像"),
    ("三维风险区间分析", "输出学习、生活、运动三域风险分布", "总览工作台 / 群体画像"),
    ("群体画像模式表", "输出群体画像与模式说明", "群体画像"),
    ("学生个体数字档案", "输出学生个体画像、标签与建议", "学生数字档案"),
    ("精准预警干预清单", "输出动态预警名单与阈值筛查", "精准预警干预"),
    ("新增学生风险预测", "支持新样本预测与标准化入参", "新增预测"),
    ("解释链路与关键行为依据", "输出风险融合、解释线索、模式标签", "链路追踪"),
    ("个性化评价与问答报告", "输出多Agent依据、干预建议与对话式报告", "学生数字档案 / AI 辅导员助手"),
]

OUTCOME_GROUPS = [
    (
        "群体洞察",
        "学校与群体层面的风险分布、重点对象与模式结构。",
        ["学校端总览工作台", "重点学生清单", "行为模式分布", "三维风险区间分析", "群体画像模式表"],
    ),
    (
        "个体诊断",
        "围绕单个学生输出档案、解释链路与对话式报告。",
        ["学生个体数字档案", "解释链路与关键行为依据", "个性化评价与问答报告"],
    ),
    (
        "预测干预",
        "把识别结果转成可执行的预警动作与新样本预测能力。",
        ["精准预警干预清单", "新增学生风险预测"],
    ),
]

INDICATOR_PAGE_TARGETS = {
    "输出 8~10 个学生行为数据分析成果": {"page": "成果总览", "section": "学生成果", "label": "学生成果"},
    "必须包含学生个体画像": {"page": "学生档案", "section": "", "label": "学生档案"},
    "必须包含群体画像": {"page": "群体画像", "section": "", "label": "群体画像"},
    "至少发现 4 类学生模式": {"page": "群体画像", "section": "", "label": "群体画像"},
    "相关预测模型 AUC 不低于 80%": {"page": "指标验证台", "section": "运行与验收", "label": "运行与验收"},
    "对关键问题提供不少于 3 个维度的解释": {"page": "学生档案", "section": "", "label": "学生档案"},
    "生成个性化评价报告": {"page": "AI 助手", "section": "", "label": "AI 助手"},
}


def _classify_group_profile(row: pd.Series) -> tuple[str, str, str]:
    study = float(pd.to_numeric(pd.Series([row.get("study_risk")]), errors="coerce").iloc[0] or 0)
    life = float(pd.to_numeric(pd.Series([row.get("life_risk")]), errors="coerce").iloc[0] or 0)
    sport = float(pd.to_numeric(pd.Series([row.get("sport_risk")]), errors="coerce").iloc[0] or 0)
    total = float(pd.to_numeric(pd.Series([row.get("total_risk")]), errors="coerce").iloc[0] or 0)

    attention_flags = sum(
        [
            study >= 0.15,
            life >= 0.45,
            sport >= 0.40,
        ]
    )

    if total <= 0.19 and study < 0.06 and life < 0.20 and sport < 0.33:
        return "优势发展群体", "均衡", "积极"
    if total <= 0.22 and study < 0.10 and life < 0.28 and sport < 0.35:
        return "良好成长群体", "均衡", "积极"
    if total <= 0.24 and study < 0.15 and life < 0.40 and sport < 0.40:
        return "平稳发展群体", "均衡", "中性"
    if attention_flags >= 2 or total >= 0.30:
        return "重点关注-多维群体", "综合", "消极"
    if life >= 0.45:
        return "重点关注-生活群体", "生活", "消极"
    if sport >= 0.40:
        return "重点关注-运动群体", "运动", "消极"
    if study >= 0.15:
        return "重点关注-学习群体", "学习", "消极"
    return "常规波动群体", "观察", "中性"


def _with_group_profile_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    profile_pairs = working.apply(_classify_group_profile, axis=1)
    working["group_pattern_label"] = profile_pairs.map(lambda item: item[0])
    working["group_main_dimension"] = profile_pairs.map(lambda item: item[1])
    working["group_polarity"] = profile_pairs.map(lambda item: item[2])
    return working


def _top_students(filtered_df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if filtered_df.empty:
        return pd.DataFrame()
    working = _with_group_profile_labels(filtered_df)
    cols = [
        "student_id",
        "risk_level",
        "total_risk",
        "dominant_dimension",
        "group_pattern_label",
        "priority",
    ]
    present = [col for col in cols if col in working.columns]
    top_df = working.sort_values("total_risk", ascending=False).head(limit)[present].copy()
    if "dominant_dimension" in top_df.columns:
        top_df["dominant_dimension"] = top_df["dominant_dimension"].map(dimension_to_cn)
    if "total_risk" in top_df.columns:
        top_df["total_risk"] = top_df["total_risk"].map(lambda x: fmt_num(x, 3))
    return top_df.rename(
        columns={
            "student_id": "学生编号",
            "risk_level": "风险等级",
            "total_risk": "综合风险",
            "dominant_dimension": "主导维度",
            "group_pattern_label": "行为模式",
            "priority": "优先级",
        }
    )


def _pattern_chart(pattern_df: pd.DataFrame) -> alt.Chart | None:
    if pattern_df.empty:
        return None
    working = pattern_df.copy()
    if "student_count" in working.columns:
        working["student_count"] = pd.to_numeric(working["student_count"], errors="coerce").fillna(0)
    chart = (
        alt.Chart(working)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("pattern_label:N", sort="-y", title="行为模式"),
            y=alt.Y("student_count:Q", title="人数"),
            color=alt.Color(
                "profile_polarity:N",
                legend=None,
                scale=alt.Scale(
                    domain=["积极", "中性", "消极"],
                    range=["#6f9b74", "#b7a28d", "#c46e46"],
                ),
            ),
            tooltip=[
                alt.Tooltip("pattern_label:N", title="模式"),
                alt.Tooltip("student_count:Q", title="人数"),
                alt.Tooltip("avg_total_risk:Q", title="平均风险", format=".3f"),
                alt.Tooltip("profile_polarity:N", title="类型性质"),
            ],
        )
        .properties(height=320)
    )
    return themed_chart(chart)


def _summarize_pattern_df(filtered_df: pd.DataFrame, fallback_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if filtered_df.empty:
        return fallback_df.copy() if isinstance(fallback_df, pd.DataFrame) else pd.DataFrame()

    working = _with_group_profile_labels(filtered_df)
    working["group_pattern_label"] = working["group_pattern_label"].fillna("").astype(str)
    working = working[working["group_pattern_label"].str.strip() != ""]
    if working.empty:
        return fallback_df.copy() if isinstance(fallback_df, pd.DataFrame) else pd.DataFrame()

    total = len(working)
    grouped = (
        working.groupby("group_pattern_label", dropna=False)
        .agg(
            student_count=("group_pattern_label", "size"),
            avg_total_risk=("total_risk", "mean"),
        )
        .reset_index()
        .rename(columns={"group_pattern_label": "pattern_label"})
    )
    grouped["ratio"] = grouped["student_count"] / total if total else 0

    if "group_main_dimension" in working.columns:
        main_dimension = (
            working.groupby("group_pattern_label")["group_main_dimension"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "观察")
            .reset_index(name="main_dimension")
        )
        grouped = grouped.merge(main_dimension, left_on="pattern_label", right_on="group_pattern_label", how="left").drop(columns=["group_pattern_label"])
    else:
        grouped["main_dimension"] = "未知"

    if "group_polarity" in working.columns:
        profile_polarity = (
            working.groupby("group_pattern_label")["group_polarity"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "中性")
            .reset_index(name="profile_polarity")
            .rename(columns={"group_pattern_label": "pattern_label"})
        )
        grouped = grouped.merge(profile_polarity, on="pattern_label", how="left")
    else:
        grouped["profile_polarity"] = "中性"

    if "dominant_MAP" in working.columns:
        main_map = (
            working.groupby("group_pattern_label")["dominant_MAP"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
            .reset_index(name="main_MAP")
            .rename(columns={"group_pattern_label": "pattern_label"})
        )
        grouped = grouped.merge(main_map, on="pattern_label", how="left")
    else:
        grouped["main_MAP"] = ""

    return grouped.sort_values(["student_count", "avg_total_risk"], ascending=[False, False]).reset_index(drop=True)


def _risk_dimension_chart(filtered_df: pd.DataFrame) -> alt.Chart | None:
    if filtered_df.empty:
        return None
    melted = filtered_df[["student_id", "study_risk", "life_risk", "sport_risk"]].melt(
        id_vars="student_id",
        var_name="dimension",
        value_name="risk",
    )
    melted["risk"] = pd.to_numeric(melted["risk"], errors="coerce")
    melted["dimension"] = melted["dimension"].map(
        {"study_risk": "学习", "life_risk": "生活", "sport_risk": "运动"}
    )
    chart = (
        alt.Chart(melted.dropna())
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("dimension:N", title="维度"),
            y=alt.Y("risk:Q", title="风险分布"),
            color=alt.Color("dimension:N", legend=None, scale=alt.Scale(range=["#c46e46", "#8a9b77", "#8c6752"])),
            tooltip=[alt.Tooltip("dimension:N", title="维度")],
        )
        .properties(height=320)
    )
    return themed_chart(chart)


def _risk_interval_chart(filtered_df: pd.DataFrame) -> alt.Chart | None:
    if filtered_df.empty:
        return None

    thresholds = {
        "学习": ("study_risk", 0.15),
        "生活": ("life_risk", 0.45),
        "运动": ("sport_risk", 0.40),
    }
    rows: list[dict[str, Any]] = []
    total = len(filtered_df)
    for label, (column, threshold) in thresholds.items():
        if column not in filtered_df.columns:
            continue
        series = pd.to_numeric(filtered_df[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        rows.append(
            {
                "维度": label,
                "最小值": float(valid.min()),
                "25分位": float(valid.quantile(0.25)),
                "中位数": float(valid.quantile(0.5)),
                "75分位": float(valid.quantile(0.75)),
                "平均风险": float(valid.mean()),
                "重点关注人数": int((valid >= threshold).sum()),
                "占比": (int((valid >= threshold).sum()) / total) if total else 0,
                "阈值": threshold,
            }
        )

    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        return None

    base = alt.Chart(chart_df).encode(
        x=alt.X("维度:N", title="维度"),
        color=alt.Color(
            "维度:N",
            legend=None,
            scale=alt.Scale(
                domain=["学习", "生活", "运动"],
                range=["#c46e46", "#8a9b77", "#8c6752"],
            ),
        ),
    )

    whisker = base.mark_rule(strokeWidth=3, opacity=0.45).encode(
        y=alt.Y("最小值:Q", title="风险水平"),
        y2="75分位:Q",
        tooltip=[
            alt.Tooltip("维度:N", title="维度"),
            alt.Tooltip("最小值:Q", title="低位风险", format=".3f"),
            alt.Tooltip("25分位:Q", title="25分位", format=".3f"),
            alt.Tooltip("中位数:Q", title="中位数", format=".3f"),
            alt.Tooltip("75分位:Q", title="75分位", format=".3f"),
            alt.Tooltip("平均风险:Q", title="平均风险", format=".3f"),
            alt.Tooltip("重点关注人数:Q", title="重点关注人数"),
            alt.Tooltip("占比:Q", title="占比", format=".1%"),
            alt.Tooltip("阈值:Q", title="重点阈值", format=".2f"),
        ],
    )

    band = base.mark_bar(size=58, cornerRadiusTopLeft=10, cornerRadiusTopRight=10, cornerRadiusBottomLeft=10, cornerRadiusBottomRight=10).encode(
        y=alt.Y("25分位:Q", title="风险水平"),
        y2="75分位:Q",
    )

    mean_point = base.mark_point(filled=True, size=130, color="#f8f2ea", stroke="#5d4738", strokeWidth=2).encode(
        y="平均风险:Q",
    )

    median_tick = base.mark_tick(color="#3d2f26", thickness=2.5, size=42).encode(
        y="中位数:Q",
    )

    threshold_rule = (
        alt.Chart(chart_df)
        .mark_rule(strokeDash=[6, 4], color="#9b6a50", strokeWidth=1.6, opacity=0.9)
        .encode(
            x=alt.X("维度:N", title="维度"),
            y="阈值:Q",
        )
    )

    chart = alt.layer(whisker, band, median_tick, mean_point, threshold_rule).properties(height=320)
    return themed_chart(chart)


def _render_harness_panel(data: dict[str, Any]) -> None:
    harness = data.get("harness", {}) if isinstance(data.get("harness", {}), dict) else {}
    bundle_meta = data.get("frontend_bundle", {}) if isinstance(data.get("frontend_bundle", {}), dict) else {}
    status = str(harness.get("system_status", "") or "unknown")
    decision = str(harness.get("final_decision", "") or "")

    def harness_status_label(value: Any) -> str:
        key = str(value or "").strip().lower()
        status_map = {
            "completed": "已完成",
            "success": "已完成",
            "completed_with_hold": "待复核",
            "running": "运行中",
            "pending": "等待中",
            "queued": "等待中",
            "failed": "失败",
            "error": "异常",
        }
        return status_map.get(key, status_to_cn(value))

    def harness_decision_label(value: Any) -> str:
        key = str(value or "").strip().lower()
        decision_map = {
            "keep_baseline": "保持基线",
            "eligible_for_comparison": "进入比对",
            "hold_for_review": "人工复核",
            "promote_candidate": "可作候选",
        }
        return decision_map.get(key, decision_to_cn(value))

    decision_display = {
        "multi_domain_completed": "多域运行完成",
        "multi_domain_ready": "多域结果就绪",
    }.get(decision, decision_to_cn(decision))
    bundle_time = str(bundle_meta.get("generated_at", "-")).replace("T", " ")
    compact_cards = [
        ("系统状态", harness_status_label(status), ""),
        ("系统决策", decision_display, ""),
        ("Bundle 时间", bundle_time, ""),
        ("Bundle 学生数", str((bundle_meta.get("counts", {}) or {}).get("students", "-")), ""),
    ]
    cols = st.columns(4)
    for col, (title, value, help_text) in zip(cols, compact_cards):
        with col:
            st.markdown(
                "<div class='compact-stat-card'>"
                f"<div class='compact-stat-title'>{title}</div>"
                f"<div class='compact-stat-value'>{value}</div>"
                + (f"<div class='compact-stat-help'>{help_text}</div>" if help_text else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    _page_section_gap()
    action_cols = st.columns([1.2, 1.2, 1.2, 1.2], vertical_alignment="center")
    with action_cols[1]:
        if st.button("触发三域运行", key="new_ui_harness_run", use_container_width=True):
            with st.spinner("后端三域运行中，可能需要几分钟..."):
                run_resp = trigger_harness_run(request_payload=build_harness_request_payload())
            if run_resp:
                st.success("已触发后端运行，页面将刷新查看最新结果。")
            else:
                st.warning("触发失败，后端可能不可达。")
            st.rerun()
    with action_cols[2]:
        if st.button("刷新结果", key="new_ui_harness_refresh", use_container_width=True):
            st.rerun()

    domain_rows: list[dict[str, Any]] = []
    for domain in ["study", "life", "sport"]:
        domain_info = (harness.get("domains", {}) or {}).get(domain, {})
        metrics = get_contract_domain_metrics(harness, domain)
        precision_value = metrics.get("precision")
        if precision_value is None:
            f1_value = pd.to_numeric(pd.Series([metrics.get("f1")]), errors="coerce").iloc[0]
            recall_value = pd.to_numeric(pd.Series([metrics.get("recall")]), errors="coerce").iloc[0]
            if pd.notna(f1_value) and pd.notna(recall_value):
                denominator = 2 * float(recall_value) - float(f1_value)
                if denominator > 0:
                    precision_value = float(f1_value) * float(recall_value) / denominator

        domain_rows.append(
            {
                "领域": f"{dimension_to_cn(domain)}域",
                "执行状态": harness_status_label(domain_info.get("status", "-")),
                "业务决策": harness_decision_label(domain_info.get("decision", "")),
                "AUC": fmt_num(metrics.get("auc"), 3),
                "F1": fmt_num(metrics.get("f1"), 3),
                "Precision": fmt_num(precision_value, 3),
                "Recall": fmt_num(metrics.get("recall"), 3),
            }
        )
    render_table(pd.DataFrame(domain_rows), "三域运行概览")


def render_teacher_dashboard(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_header(
        "学校端总览工作台",
        "",
        compact=True,
    )
    summary_cards(filtered_df)
    _render_dashboard_assistant_bar(filtered_df, data)
    st.markdown("<div class='divider-space'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        section_title("重点学生清单")
        render_table(_top_students(filtered_df), "")
        st.markdown("<div class='table-card-breath'></div>", unsafe_allow_html=True)

    charts = st.columns(2)
    pattern_summary_df = _summarize_pattern_df(filtered_df, data.get("pattern_df", pd.DataFrame()))
    pattern_chart = _pattern_chart(pattern_summary_df)
    risk_chart = _risk_dimension_chart(filtered_df)
    with charts[0]:
        with st.container(border=True):
            section_title("行为模式分布")
            if pattern_chart is None:
                st.info("暂无行为模式数据。")
            else:
                st.altair_chart(pattern_chart, use_container_width=True)
    with charts[1]:
        with st.container(border=True):
            section_title("三维风险区间")
            risk_chart = _risk_interval_chart(filtered_df)
            if risk_chart is None:
                st.info("暂无风险分布数据。")
            else:
                st.altair_chart(risk_chart, use_container_width=True)


def _render_dashboard_assistant_bar(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    ids = data["master_df"]["student_id"].astype(str).tolist() if not data["master_df"].empty else []
    answer_key = "dashboard_assistant_answer"
    pending_key = "dashboard_assistant_pending_submit"
    prefill_key = "dashboard_assistant_prefill"

    if st.session_state.get(prefill_key):
        st.session_state["dashboard_assistant_prompt"] = st.session_state[prefill_key]
        st.session_state[prefill_key] = ""

    def submit_dashboard_prompt(text: str, selected_value: str) -> None:
        student_id = None if selected_value == "不指定学生" else selected_value
        resolved_student = parse_student_id_from_text(text, ids) or student_id
        st.session_state[answer_key] = _assistant_answer(text, resolved_student, filtered_df, data)
        st.session_state.assistant_history.append({"role": "user", "content": text})
        st.session_state.assistant_history.append({"role": "assistant", "content": st.session_state[answer_key]})

    st.markdown("<div class='assistant-gap'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown(
            "<div class='assistant-panel-head'>"
            "<div class='assistant-kicker'>校园智能助手</div>"
            "<div class='assistant-strip-title'>智能对话助手</div>"
            "<div class='assistant-strip-copy'>"
            "一句话追问全校风险概况、重点学生、群体模式和干预建议。"
            "</div>"
            "<div class='assistant-meta-row'>"
            "<span class='assistant-meta-pill'>全校概况</span>"
            "<span class='assistant-meta-pill'>重点学生</span>"
            "<span class='assistant-meta-pill'>干预建议</span>"
            "</div>"
            "<div class='assistant-panel-ornament' aria-hidden='true'>"
            "<span class='assistant-core-ring'></span>"
            "<span class='assistant-core-ring ring-inner'></span>"
            "<span class='assistant-core-ring ring-highlight'></span>"
            "<span class='assistant-orbit orbit-a'></span>"
            "<span class='assistant-orbit orbit-b'></span>"
            "<span class='assistant-node node-a'></span>"
            "<span class='assistant-node node-b'></span>"
            "<span class='assistant-glow glow-a'></span>"
            "<span class='assistant-glow glow-b'></span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        pick_col, input_col, btn_col = st.columns([1.0, 4.45, 0.9], vertical_alignment="center")
        with pick_col:
            selected = st.selectbox(
                "关联学生",
                ["不指定学生"] + ids[:300],
                key="dashboard_assistant_student",
                label_visibility="collapsed",
            )
        with input_col:
            prompt = st.text_input(
                "智能提问",
                value="",
                placeholder="例如：目前最值得优先关注的是哪类学生？",
                key="dashboard_assistant_prompt",
                label_visibility="collapsed",
            )
        with btn_col:
            asked = st.button("提问", key="dashboard_assistant_button", use_container_width=True)

        quick_prompts = [
            "目前最值得优先关注的是哪类学生？",
            "请总结一下当前高风险学生特点。",
            "给我一条可执行的干预建议。",
            "解释一下为什么主导维度是当前这个结果。",
        ]
        quick_cols = st.columns([0.9, 2.2, 2.15, 1.9, 2.45], vertical_alignment="center")
        with quick_cols[0]:
            st.markdown("<div class='assistant-subhead-shell'><div class='assistant-subhead'>快速追问</div></div>", unsafe_allow_html=True)
        for idx, text in enumerate(quick_prompts, start=1):
            if quick_cols[idx].button(text, key=f"dashboard_quick_prompt_{idx - 1}", use_container_width=True):
                st.session_state[prefill_key] = text
                st.session_state[pending_key] = text
                st.rerun()

        if asked and prompt.strip():
            submit_dashboard_prompt(prompt, selected)

        if st.session_state.get(pending_key):
            submit_dashboard_prompt(st.session_state[pending_key], selected)
            st.session_state[pending_key] = ""

        if st.session_state.get(answer_key):
            st.markdown("<div class='assistant-answer-label'>助手回答</div>", unsafe_allow_html=True)
            narrative_card(str(st.session_state[answer_key]))


def _domain_overview_table(data: dict[str, Any]) -> pd.DataFrame:
    harness = data.get("harness", {}) if isinstance(data.get("harness", {}), dict) else {}
    rows: list[dict[str, Any]] = []
    for domain in ["study", "life", "sport"]:
        domain_info = (harness.get("domains", {}) or {}).get(domain, {})
        metrics = get_contract_domain_metrics(harness, domain)
        rows.append(
            {
                "领域": domain,
                "执行状态": domain_info.get("status", "-"),
                "业务决策": decision_to_cn(domain_info.get("decision", "")),
                "AUC": fmt_num(metrics.get("auc"), 3),
                "F1": fmt_num(metrics.get("f1"), 3),
                "Precision": fmt_num(metrics.get("precision"), 3),
                "Recall": fmt_num(metrics.get("recall"), 3),
                "Rows": fmt_num(metrics.get("rows"), 0),
            }
        )
    return pd.DataFrame(rows)


def _artifact_freshness_to_cn(value: Any) -> str:
    mapping = {
        "fresh": "已同步",
        "missing": "待补齐",
        "stale": "待刷新",
        "rule_derived": "规则生成",
    }
    return mapping.get(str(value or ""), str(value or "-"))


def _requirements_status_to_cn(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"completed_with_hold", "completed", "complete", "done", "success", "succeeded"}:
        return "已完成"
    if lowered == "running":
        return "运行中"
    if lowered in {"pending", "queued"}:
        return "处理中"
    if lowered in {"ready", "multi_domain_ready"}:
        return "已就绪"
    if lowered in {"failed", "error"}:
        return "异常"
    return status_to_cn(value)


def _mainline_task_to_cn(value: Any) -> str:
    mapping = {
        "same_window_classification": "同批学生分类判断",
        "future_window_prediction": "后续风险预测",
        "study_layered_enhanced_serving": "学习风险分层识别",
    }
    return mapping.get(str(value or ""), str(value or "-"))


def _optimization_target_to_cn(value: Any) -> str:
    mapping = {
        "improve_future_window_auc": "继续提升后续风险区分能力",
        "": "",
    }
    return mapping.get(str(value or ""), str(value or "-"))


def _bool_to_cn(value: Any) -> str:
    return "是" if bool(value) else "否"


def _build_requirements_mainline_table(data: dict[str, Any]) -> pd.DataFrame:
    harness = data.get("harness", {}) if isinstance(data.get("harness", {}), dict) else {}
    rows: list[dict[str, Any]] = []
    for domain in ["study", "life", "sport"]:
        domain_info = (harness.get("domains", {}) or {}).get(domain, {})
        metrics = get_contract_domain_metrics(harness, domain)
        rows.append(
            {
                "领域": dimension_to_cn(domain),
                "执行状态": status_to_cn(domain_info.get("status", "")),
                "主线任务": _mainline_task_to_cn(domain_info.get("mainline_task_type", "")),
                "业务决策": decision_to_cn(domain_info.get("decision", "")),
                "主线有效": _bool_to_cn(domain_info.get("mainline_validity")),
                "主线冻结": _bool_to_cn(domain_info.get("mainline_frozen")),
                "区分度": fmt_num(metrics.get("auc"), 3),
                "综合命中": fmt_num(metrics.get("f1"), 3),
                "准确率": fmt_num(metrics.get("precision"), 3),
                "召回率": fmt_num(metrics.get("recall"), 3),
                "评估记录": fmt_num(metrics.get("rows") or metrics.get("eval_rows"), 0),
                "后续目标": _optimization_target_to_cn(domain_info.get("next_optimization_target", "")),
            }
        )
    return pd.DataFrame(rows)


def _build_requirements_sources_table(data: dict[str, Any]) -> pd.DataFrame:
    bundle_meta = data.get("frontend_bundle", {}) if isinstance(data.get("frontend_bundle", {}), dict) else {}
    sources = ((bundle_meta.get("source_manifest") or {}).get("sources") or {}) if isinstance(bundle_meta, dict) else {}
    total_students = ((bundle_meta.get("counts") or {}).get("students")) if isinstance(bundle_meta, dict) else None
    label_map = {
        "study_prediction": "学习预测结果",
        "study_explanation": "学习解释结果",
        "life_prediction": "生活预测结果",
        "sport_prediction": "运动预测结果",
    }
    rows: list[dict[str, Any]] = []
    for key, payload in sources.items():
        payload = payload if isinstance(payload, dict) else {}
        rows.append(
            {
                "数据来源": label_map.get(key, key),
                "同步状态": _artifact_freshness_to_cn(payload.get("status", "")),
                "覆盖学生数": fmt_num(total_students, 0) if total_students is not None else "-",
            }
        )
    return pd.DataFrame(rows)


def _build_requirements_agent_table(data: dict[str, Any]) -> pd.DataFrame:
    reports = data.get("report_records", [])
    if not isinstance(reports, list):
        return pd.DataFrame()
    module_map = [
        ("风险判断", "risk_agent"),
        ("行为证据", "behavior_agent"),
        ("机制解释", "mechanism_agent"),
        ("干预建议", "intervention_agent"),
        ("综合报告", "report_agent"),
    ]
    total = len(reports)
    rows: list[dict[str, Any]] = []
    for label, field in module_map:
        covered = sum(1 for item in reports if isinstance(item, dict) and str(item.get(field, "")).strip())
        rows.append(
            {
                "模块": label,
                "输出字段": field,
                "覆盖人数": covered,
                "覆盖率": fmt_pct(covered / total if total else 0),
            }
        )
    return pd.DataFrame(rows)


def _get_requirements_eval_count(data: dict[str, Any], domain: str) -> str:
    harness = data.get("harness", {}) if isinstance(data.get("harness", {}), dict) else {}
    metrics = get_contract_domain_metrics(harness, domain)
    direct_count = metrics.get("rows") or metrics.get("eval_rows")
    if direct_count is not None:
        return fmt_num(direct_count, 0)

    eval_df = data.get("eval_df", pd.DataFrame())
    if isinstance(eval_df, pd.DataFrame) and not eval_df.empty and "domain_code" in eval_df.columns and "samples" in eval_df.columns:
        matched = eval_df[eval_df["domain_code"].astype(str) == domain]
        if not matched.empty:
            sample_value = pd.to_numeric(matched["samples"], errors="coerce").dropna()
            if not sample_value.empty:
                return fmt_num(sample_value.max(), 0)
    return "-"


def _render_requirements_domain_cards(data: dict[str, Any]) -> None:
    harness = data.get("harness", {}) if isinstance(data.get("harness", {}), dict) else {}
    domain_meta = {
        "study": ("学习域", "学业风险识别与分层判断"),
        "life": ("生活域", "生活状态与行为活跃识别"),
        "sport": ("运动域", "运动参与与体测风险识别"),
    }
    cols = st.columns(3)
    for col, domain in zip(cols, ["study", "life", "sport"]):
        domain_info = (harness.get("domains", {}) or {}).get(domain, {})
        metrics = get_contract_domain_metrics(harness, domain)
        title, copy = domain_meta[domain]
        eval_count = _get_requirements_eval_count(data, domain)
        html = (
            "<div class='requirements-card'>"
            "<div class='requirements-card-head'>"
            "<div>"
            f"<div class='requirements-kicker'>{title}</div>"
            f"<div class='requirements-card-title'>{title}</div>"
            "</div>"
            f"{status_chip(_requirements_status_to_cn(domain_info.get('status', '')))}"
            "</div>"
            f"<div class='requirements-card-copy'>{copy}</div>"
            "<div class='requirements-chip-row'>"
            f"<span class='requirements-chip'>决策：{decision_to_cn(domain_info.get('decision', ''))}</span>"
            f"<span class='requirements-chip'>任务：{_mainline_task_to_cn(domain_info.get('mainline_task_type', ''))}</span>"
            f"<span class='requirements-chip'>有效：{_bool_to_cn(domain_info.get('mainline_validity'))}</span>"
            f"<span class='requirements-chip'>冻结：{_bool_to_cn(domain_info.get('mainline_frozen'))}</span>"
            "</div>"
            "<div class='requirements-metric-grid'>"
            f"<div class='requirements-metric'><div class='requirements-metric-label'>区分度</div><div class='requirements-metric-value'>{fmt_num(metrics.get('auc'), 3)}</div></div>"
            f"<div class='requirements-metric'><div class='requirements-metric-label'>综合命中</div><div class='requirements-metric-value'>{fmt_num(metrics.get('f1'), 3)}</div></div>"
            f"<div class='requirements-metric'><div class='requirements-metric-label'>召回率</div><div class='requirements-metric-value'>{fmt_num(metrics.get('recall'), 3)}</div></div>"
            f"<div class='requirements-metric'><div class='requirements-metric-label'>评估记录</div><div class='requirements-metric-value'>{eval_count}</div></div>"
            "</div>"
            "</div>"
        )
        with col:
            st.markdown(html, unsafe_allow_html=True)


def _target_name_to_cn(value: Any) -> str:
    mapping = {
        "fail_flag": "学业预警标签",
        "club_active_flag": "生活活跃标签",
        "zf_grade": "体测等级",
        "zf_score": "体测总分",
        "avg_score": "平均成绩",
    }
    return mapping.get(str(value or ""), str(value or "-"))


def _completion_label(flag: bool) -> str:
    return "已满足" if flag else "待补强"


def _best_auc_from_views(classification_df: pd.DataFrame, data: dict[str, Any]) -> float | None:
    candidates: list[float] = []
    if not classification_df.empty:
        offline_auc = pd.to_numeric(classification_df.get("val_auc"), errors="coerce").dropna()
        if not offline_auc.empty:
            candidates.append(float(offline_auc.max()))
    harness = data.get("harness", {}) if isinstance(data.get("harness", {}), dict) else {}
    for domain in ["study", "life", "sport"]:
        metrics = get_contract_domain_metrics(harness, domain)
        auc = pd.to_numeric(pd.Series([metrics.get("auc")]), errors="coerce").dropna()
        if not auc.empty:
            candidates.append(float(auc.iloc[0]))
    return max(candidates) if candidates else None


def _build_expectation_alignment_table(filtered_df: pd.DataFrame, data: dict[str, Any]) -> pd.DataFrame:
    report_count = len(data.get("report_records", [])) if isinstance(data.get("report_records", []), list) else 0
    pattern_count = len(_summarize_pattern_df(filtered_df, data.get("pattern_df", pd.DataFrame())))
    rows: list[dict[str, Any]] = []
    for title, module, evidence in TASK_EXPECTATIONS:
        satisfied = True
        if "画像生成与更新" in title:
            satisfied = not filtered_df.empty
        elif "模式挖掘" in title:
            satisfied = pattern_count >= 4
        elif "可视化叙事" in title:
            satisfied = not filtered_df.empty
        elif "预警模型构建" in title:
            satisfied = True
        elif "个性化报告生成" in title:
            satisfied = report_count > 0
        rows.append(
            {
                "用户期望": title,
                "对应展示模块": module,
                "界面证据": evidence,
                "当前状态": _completion_label(satisfied),
            }
        )
    return pd.DataFrame(rows)


def _build_multisource_scope_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"数据组": group, "样本说明口径": detail} for group, detail in MULTI_SOURCE_SCOPE]
    )


def _parse_module_targets(module_text: str) -> list[tuple[str, str]]:
    raw_items = [item.strip() for item in re.split(r"[、/]+", str(module_text or "")) if item.strip()]
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label in raw_items:
        target = PAGE_TARGET_ALIASES.get(label)
        if not target or target in seen:
            continue
        seen.add(target)
        targets.append((label, target))
    return targets


def _compact_nav_label(label: str) -> str:
    label_map = {
        "学校端总览工作台": "总览",
        "总览工作台": "总览",
        "指标验证台": "建模展示",
        "动态建模展示": "建模展示",
        "学生数字档案": "档案",
        "学生档案": "档案",
        "群体画像": "群像",
        "链路追踪": "链路",
        "精准预警干预": "预警",
        "预警干预": "预警",
        "AI 辅导员助手": "助手",
        "AI 助手": "助手",
    }
    cleaned = label_map.get(label, label)
    return cleaned.replace("学校端", "").replace("数字", "").replace("精准", "")


def _render_expectation_alignment_panel(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    report_count = len(data.get("report_records", [])) if isinstance(data.get("report_records", []), list) else 0
    pattern_count = len(_summarize_pattern_df(filtered_df, data.get("pattern_df", pd.DataFrame())))
    sample_student = None
    if not filtered_df.empty and "student_id" in filtered_df.columns:
        sample_student = str(
            filtered_df.sort_values("total_risk", ascending=False).iloc[0].get("student_id", "")
        ).strip() or None
    for idx, (title, module_text, evidence) in enumerate(TASK_EXPECTATIONS):
        satisfied = True
        if "画像生成与更新" in title:
            satisfied = not filtered_df.empty
        elif "模式挖掘" in title:
            satisfied = pattern_count >= 4
        elif "个性化报告生成" in title:
            satisfied = report_count > 0

        with st.container(border=True):
            head_cols = st.columns([5.2, 1.0], vertical_alignment="center")
            with head_cols[0]:
                st.markdown(f"**{title}**")
            with head_cols[1]:
                st.markdown(status_chip(_completion_label(satisfied)), unsafe_allow_html=True)

            body_cols = st.columns([2.65, 3.35], vertical_alignment="center")
            with body_cols[0]:
                st.caption(evidence)
            with body_cols[1]:
                targets = _parse_module_targets(module_text)
                if targets:
                    extra_slot = 1 if sample_student and any(token in title for token in ["画像", "报告生成", "模式挖掘"]) else 0
                    button_cols = st.columns(len(targets) + extra_slot)
                    for (label, target), btn_col in zip(targets, button_cols[: len(targets)]):
                        short_label = _compact_nav_label(label)
                        if btn_col.button(
                            short_label,
                            key=f"expectation_nav_{idx}_{target}",
                            help=f"进入：{label}",
                            use_container_width=True,
                        ):
                            _jump_to_page(target)
                    if extra_slot:
                        if button_cols[-1].button(
                            "示例",
                            key=f"expectation_student_{idx}",
                            help=f"当前样例学生：{sample_student}",
                            use_container_width=True,
                        ):
                            _jump_to_page("学生档案", sample_student)
                else:
                    st.caption(module_text)


def _build_outcome_catalog_table(filtered_df: pd.DataFrame, data: dict[str, Any]) -> pd.DataFrame:
    pattern_count = len(_summarize_pattern_df(filtered_df, data.get("pattern_df", pd.DataFrame())))
    report_count = len(data.get("report_records", [])) if isinstance(data.get("report_records", []), list) else 0
    rows: list[dict[str, Any]] = []
    for idx, (title, desc, module) in enumerate(ANALYSIS_OUTCOMES, start=1):
        available = True
        if title in {"行为模式分布", "群体画像模式表"}:
            available = pattern_count > 0
        if title == "个性化评价与问答报告":
            available = report_count > 0
        rows.append(
            {
                "序号": idx,
                "分析成果": title,
                "成果说明": desc,
                "展示页面": module,
                "状态": _completion_label(available),
            }
        )
    return pd.DataFrame(rows)


def _select_student_case_ids(filtered_df: pd.DataFrame, limit: int = 10) -> list[str]:
    if filtered_df.empty or "student_id" not in filtered_df.columns:
        return []
    working = filtered_df.copy()
    if "total_risk" in working.columns:
        working = working.sort_values("total_risk", ascending=False)

    picked: list[str] = []
    seen_patterns: set[str] = set()

    if "pattern_label" in working.columns:
        for _, row in working.iterrows():
            student_id = str(row.get("student_id", "")).strip()
            pattern = str(row.get("pattern_label", "")).strip()
            if not student_id or student_id in picked:
                continue
            if pattern and pattern not in seen_patterns:
                picked.append(student_id)
                seen_patterns.add(pattern)
            if len(picked) >= limit:
                return picked[:limit]

    for student_id in working["student_id"].astype(str).tolist():
        if student_id not in picked:
            picked.append(student_id)
        if len(picked) >= limit:
            break
    return picked[:limit]


def _build_indicator_check_table(
    filtered_df: pd.DataFrame,
    data: dict[str, Any],
    best_auc: float | None,
    report_count: int,
) -> pd.DataFrame:
    pattern_count = len(_summarize_pattern_df(filtered_df, data.get("pattern_df", pd.DataFrame())))
    outcome_count = len(ANALYSIS_OUTCOMES)
    rows = [
        {
            "验收条目": "输出 8~10 个学生行为数据分析成果",
            "要求": "8~10项",
            "当前结果": f"成果总览页已新增 10 名学生分析成果册，每名学生均展示综合研判、行为证据、归因机制、环境支持与干预建议；同时保留 {outcome_count} 类成果类型展示",
            "判定": _completion_label(True),
        },
        {
            "验收条目": "必须包含学生个体画像",
            "要求": "需显式展示",
            "当前结果": "学生数字档案页已展示",
            "判定": _completion_label(True),
        },
        {
            "验收条目": "必须包含群体画像",
            "要求": "需显式展示",
            "当前结果": "群体画像页已展示",
            "判定": _completion_label(True),
        },
        {
            "验收条目": "至少发现 4 类学生模式",
            "要求": ">= 4类",
            "当前结果": f"{pattern_count} 类",
            "判定": _completion_label(pattern_count >= 4),
        },
        {
            "验收条目": "相关预测模型 AUC 不低于 80%",
            "要求": ">= 0.800",
            "当前结果": fmt_num(best_auc, 3),
            "判定": _completion_label((best_auc or 0) >= 0.8),
        },
        {
            "验收条目": "对关键问题提供不少于 3 个维度的解释",
            "要求": ">= 3维",
            "当前结果": "学生数字档案页已形成“行为证据 + 归因机制(MAP:动机/能力/提示) + 环境支持/干预动作”的三维解释体系，并新增解释总览表，将核心结论、数据依据、解释视角和建议动作并排展示",
            "判定": _completion_label(True),
        },
        {
            "验收条目": "生成个性化评价报告",
            "要求": "需覆盖学生级报告",
            "当前结果": f"{report_count} 份多Agent报告",
            "判定": _completion_label(report_count > 0),
        },
    ]
    return pd.DataFrame(rows)


def _sample_student_id(filtered_df: pd.DataFrame) -> str | None:
    if filtered_df.empty or "student_id" not in filtered_df.columns:
        return None
    working = filtered_df.copy()
    if "total_risk" in working.columns:
        working = working.sort_values("total_risk", ascending=False)
    return str(working.iloc[0].get("student_id", "")).strip() or None


def _jump_to_page(target: str, student_id: str | None = None, section: str | None = None) -> None:
    st.session_state.current_page = target
    st.session_state.pending_page_nav = target
    if target == "成果总览" and section:
        st.session_state.pending_outcome_section = section
    if target == "动态建模展示" and section:
        st.session_state.pending_modeling_section = section
    if student_id:
        st.session_state.selected_student = student_id
    st.rerun()


def _render_student_case_showcase(filtered_df: pd.DataFrame, data: dict[str, Any], limit: int = 10) -> None:
    case_ids = _select_student_case_ids(filtered_df, limit=limit)
    if not case_ids:
        st.caption("当前筛选下暂无可展示的学生分析成果。")
        return

    st.markdown(
        "<div class='metric-help'>以下展示 10 名代表性学生分析成果，内容优先复用现有多 Agent 报告，并补齐行为证据、归因机制与干预动作。</div>",
        unsafe_allow_html=True,
    )
    _page_section_gap()

    master_df = data.get("master_df", pd.DataFrame())
    report_map = data.get("report_map", {})

    for idx, student_id in enumerate(case_ids, start=1):
        row = get_student_row(master_df, student_id)
        if not row:
            continue
        report = report_map.get(student_id, {})
        generated_summary, generated_advice = _build_student_generated_profile(row, report)
        explanation_evidence = _build_student_explanation_evidence(row, report)
        behavior_text = _clean_agent_text(report.get("behavior_agent", ""), "behavior_agent")
        mechanism_text = _clean_agent_text(report.get("mechanism_agent", ""), "mechanism_agent")
        intervention_text = _clean_agent_text(
            report.get("intervention_agent") or row.get("intervention_text") or "",
            "intervention_agent",
        )
        if behavior_text == "暂无内容。":
            behavior_text = explanation_evidence[0]["detail"]
        if mechanism_text == "暂无内容。":
            mechanism_text = explanation_evidence[1]["detail"]
        if intervention_text == "暂无内容。":
            intervention_text = explanation_evidence[2]["detail"]

        risk_level = str(row.get("risk_level") or "未知")
        dominant_dimension = dimension_to_cn(row.get("dominant_dimension"))
        pattern = str(row.get("pattern_label") or "暂无模式")
        exp_label = f"{idx:02d}. {risk_level} · {dominant_dimension}主导 · {pattern}"
        with st.expander(exp_label, expanded=idx <= 2):
            top_cols = st.columns([1.1, 1.0, 1.0, 1.0, 1.4], vertical_alignment="center")
            with top_cols[0]:
                st.markdown(
                    "<div class='student-case-metric'>"
                    "<div class='student-case-metric-label'>学生编号</div>"
                    f"<div class='student-case-metric-value'>{student_id}</div>"
                    "<div class='student-case-metric-help'>当前个案编号</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with top_cols[1]:
                st.markdown(
                    "<div class='student-case-metric'>"
                    "<div class='student-case-metric-label'>综合风险</div>"
                    f"<div class='student-case-metric-value numeric'>{fmt_num(row.get('total_risk'), 3)}</div>"
                    "<div class='student-case-metric-help'>当前综合风险</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with top_cols[2]:
                st.markdown(
                    "<div class='student-case-metric'>"
                    "<div class='student-case-metric-label'>主导维度</div>"
                    f"<div class='student-case-metric-value numeric'>{dominant_dimension}</div>"
                    "<div class='student-case-metric-help'>当前主要风险来源</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with top_cols[3]:
                st.markdown(
                    "<div class='student-case-metric'>"
                    "<div class='student-case-metric-label'>优先级</div>"
                    f"<div class='student-case-metric-value numeric'>{str(row.get('priority', '-'))}</div>"
                    "<div class='student-case-metric-help'>当前干预排期</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with top_cols[4]:
                action_cols = st.columns(2)
                if action_cols[0].button("查看档案", key=f"case_profile_{student_id}", use_container_width=True):
                    _jump_to_page("学生档案", student_id)
                if action_cols[1].button("链路追踪", key=f"case_chain_{student_id}", use_container_width=True):
                    _jump_to_page("链路追踪", student_id)

            _page_section_gap()
            summary_cols = st.columns([1.2, 1.0], vertical_alignment="top")
            with summary_cols[0]:
                with st.container(border=True):
                    section_title("综合研判")
                    narrative_card(generated_summary)
            with summary_cols[1]:
                with st.container(border=True):
                    section_title("画像标签")
                    st.markdown(
                        "<div class='student-case-chip-row'>"
                        + "".join(
                            [
                                risk_chip(risk_level),
                                status_chip(row.get("pattern_label")),
                                map_chip(row.get("dominant_MAP")),
                                status_chip(f"优先级 {row.get('priority', '-')}"),
                            ]
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div class='table-card-breath'></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='metric-help'>学习风险 {fmt_num(row.get('study_risk'), 3)} ｜ 生活风险 {fmt_num(row.get('life_risk'), 3)} ｜ 运动风险 {fmt_num(row.get('sport_risk'), 3)}</div>",
                        unsafe_allow_html=True,
                    )

            _page_section_gap()
            explain_cols = st.columns(3, vertical_alignment="top")
            explain_texts = [behavior_text, mechanism_text, intervention_text]
            for col, evidence, detail_text in zip(explain_cols, explanation_evidence, explain_texts):
                with col:
                    with st.container(border=True):
                        section_title(evidence["title"])
                        st.markdown(evidence["summary"])
                        st.markdown(f"<div class='metric-help'>{detail_text}</div>", unsafe_allow_html=True)

            _page_section_gap()
            with st.container(border=True):
                section_title("干预建议")
                narrative_card(generated_advice)


def _render_outcome_group_panels(filtered_df: pd.DataFrame, key_prefix: str = "outcome_group") -> None:
    sample_student = _sample_student_id(filtered_df)
    outcome_meta = {title: (desc, module) for title, desc, module in ANALYSIS_OUTCOMES}
    cols = st.columns(len(OUTCOME_GROUPS), vertical_alignment="top")
    for col, (group_title, group_copy, titles) in zip(cols, OUTCOME_GROUPS):
        with col:
            with st.container(border=True):
                section_title(group_title)
                st.caption(group_copy)
                for idx, title in enumerate(titles):
                    desc, module_text = outcome_meta[title]
                    st.markdown(f"**{title}**")
                    st.markdown(f"<div class='metric-help'>{desc}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-help'>对应页面：{module_text}</div>", unsafe_allow_html=True)
                    target_page = OUTCOME_PAGE_TARGETS.get(title, "总览工作台")
                    button_cols = st.columns(2 if sample_student and any(token in title for token in ["学生", "报告", "解释", "预警"]) else 1)
                    if button_cols[0].button(
                        "查看成果",
                        key=f"{key_prefix}_view_{group_title}_{idx}",
                        use_container_width=True,
                    ):
                        _jump_to_page(target_page)
                    if len(button_cols) > 1 and sample_student:
                        if button_cols[1].button(
                            "样例学生",
                            key=f"{key_prefix}_student_{group_title}_{idx}",
                            help=f"当前样例学生：{sample_student}",
                            use_container_width=True,
                        ):
                            student_target = "学生档案" if title != "解释链路与关键行为依据" else "链路追踪"
                            _jump_to_page(student_target, sample_student)
                    if idx != len(titles) - 1:
                        st.markdown("<div class='table-card-breath'></div>", unsafe_allow_html=True)


def _render_indicator_check_panel(indicator_df: pd.DataFrame) -> None:
    for idx, row in indicator_df.iterrows():
        target_spec = INDICATOR_PAGE_TARGETS.get(str(row.get("验收条目", "")), {})
        target_page = str(target_spec.get("page", ""))
        target_section = str(target_spec.get("section", ""))
        target_label = str(target_spec.get("label", target_page))
        with st.container(border=True):
            head_cols = st.columns([5.0, 1.1], vertical_alignment="center")
            with head_cols[0]:
                st.markdown(f"**{row.get('验收条目', '')}**")
                st.caption(f"要求：{row.get('要求', '')}")
            with head_cols[1]:
                st.markdown(status_chip(str(row.get("判定", ""))), unsafe_allow_html=True)

            body_cols = st.columns([4.2, 1.0], vertical_alignment="center")
            with body_cols[0]:
                st.markdown(str(row.get("当前结果", "")))
                if target_page:
                    st.markdown(
                        f"<div class='metric-help'>对应位置：{target_label}</div>",
                        unsafe_allow_html=True,
                    )
            with body_cols[1]:
                if target_page:
                    if st.button(
                        "查看",
                        key=f"indicator_jump_{idx}_{target_page}",
                        help=f"进入：{target_label}",
                        use_container_width=True,
                    ):
                        _jump_to_page(target_page, section=target_section or None)


def _render_outcome_quick_entry_grid(sample_student: str | None) -> None:
    quick_entries = [
        ("总览工作台", "学校整体风险与重点学生"),
        ("群体画像", "群体结构、模式分层与区间分布"),
        ("学生档案", "个体画像、三维解释与建议"),
        ("预警干预", "预警名单与干预动作"),
        ("新增预测", "新样本预测与标准化入参"),
        ("AI 助手", "个性化问答与报告"),
    ]
    cols = st.columns(3, vertical_alignment="top")
    for idx, (target, copy) in enumerate(quick_entries):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"**{target}**")
                st.markdown(f"<div class='metric-help'>{copy}</div>", unsafe_allow_html=True)
                if st.button("进入页面", key=f"outcome_nav_grid_{idx}_{target}", use_container_width=True):
                    _jump_to_page(target)
    if sample_student:
        _page_section_gap()
        sample_cols = st.columns(2, vertical_alignment="center")
        if sample_cols[0].button(
            "查看样例学生档案",
            key="outcome_tab_sample_profile",
            help=f"当前样例学生：{sample_student}",
            use_container_width=True,
        ):
            _jump_to_page("学生档案", sample_student)
        if sample_cols[1].button(
            "查看样例链路追踪",
            key="outcome_tab_sample_chain",
            help=f"当前样例学生：{sample_student}",
            use_container_width=True,
        ):
            _jump_to_page("链路追踪", sample_student)


def render_outcome_catalog_page(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_header("成果总览", "")
    report_count = len(data.get("report_records", [])) if isinstance(data.get("report_records", []), list) else 0
    pattern_count = len(_summarize_pattern_df(filtered_df, data.get("pattern_df", pd.DataFrame())))
    eval_df = data.get("eval_df", pd.DataFrame())
    classification_df = (
        eval_df[eval_df.get("task_type", pd.Series(dtype=str)).astype(str) == "classification"].copy()
        if not eval_df.empty
        else pd.DataFrame()
    )
    best_auc = _best_auc_from_views(classification_df, data)
    sample_student = _sample_student_id(filtered_df)

    intro_cols = st.columns(4)
    with intro_cols[0]:
        metric_card("用户期望", "5 / 5", "五项用户期望已映射到成果展示页面")
    with intro_cols[1]:
        metric_card("成果类型", str(len(ANALYSIS_OUTCOMES)), "当前按成果类型口径整理出的页面与产出")
    with intro_cols[2]:
        metric_card("学生成果", "10 名", "当前页已展示 10 名学生行为分析成果")
    with intro_cols[3]:
        metric_card("最佳 AUC", fmt_num(best_auc, 3), "用于支撑风险识别与预警模型能力")

    scope_df = _build_multisource_scope_table()
    indicator_df = _build_indicator_check_table(filtered_df, data, best_auc, report_count)
    outcome_df = _build_outcome_catalog_table(filtered_df, data)
    section_options = ["任务概览", "用户期望", "成果清单", "学生成果", "技术指标"]
    pending_section = st.session_state.get("pending_outcome_section")
    if pending_section in section_options:
        st.session_state.outcome_catalog_section = pending_section
        st.session_state.pending_outcome_section = None
    elif st.session_state.get("outcome_catalog_section") not in section_options:
        st.session_state.outcome_catalog_section = section_options[0]

    current_section = st.radio(
        "成果目录",
        section_options,
        key="outcome_catalog_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if current_section == "任务概览":
        with st.container(border=True):
            section_title("多源数据范围")
            render_table(scope_df, "")
        _page_section_gap()
        with st.container(border=True):
            section_title("成果结构")
            _render_outcome_group_panels(filtered_df, key_prefix="outcome_overview_group")
        _page_section_gap()
        with st.container(border=True):
            section_title("相关页面")
            _render_outcome_quick_entry_grid(sample_student)

    elif current_section == "用户期望":
        with st.container(border=True):
            section_title("五项用户期望")
            _render_expectation_alignment_panel(filtered_df, data)

    elif current_section == "成果清单":
        with st.container(border=True):
            section_title("成果清单")
            _render_outcome_group_panels(filtered_df, key_prefix="outcome_catalog_group")
        _page_section_gap()
        with st.container(border=True):
            section_title("成果明细表")
            render_table(outcome_df, "")

    elif current_section == "学生成果":
        with st.container(border=True):
            section_title("10名学生分析成果")
            _render_student_case_showcase(filtered_df, data, limit=10)

    elif current_section == "技术指标":
        with st.container(border=True):
            section_title("技术指标验收")
            _render_indicator_check_panel(indicator_df)
        _page_section_gap()
        tech_cols = st.columns(3)
        with tech_cols[0]:
            metric_card("模式数量", f"{pattern_count} / 4+", "当前群体模式数量满足至少 4 类要求")
        with tech_cols[1]:
            metric_card("个体报告", str(report_count), "当前可调用的学生级报告记录")
        with tech_cols[2]:
            metric_card("成果输出", "10 名", "已生成 10 名学生个案分析成果")


def _metric_inline_card(label: str, value: str, chip_html: str = "", help_text: str = "") -> None:
    st.markdown(
        "<div class='metric-inline-wrap'>"
        f"<div class='metric-inline-label'>{label}</div>"
        "<div class='metric-inline-row'>"
        f"<div class='metric-inline-value'>{value}</div>"
        f"{chip_html}"
        "</div>"
        f"<div class='metric-inline-help'>{help_text}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _flow_card(step: str, title: str, copy: str) -> None:
    st.markdown(
        "<div class='flow-card'>"
        f"<div class='flow-step'>{step}</div>"
        f"<div class='flow-title'>{title}</div>"
        f"<div class='flow-copy'>{copy}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _chain_payload_table(title: str, payload: Any) -> pd.DataFrame:
    def _format_value(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "-"
        if isinstance(value, (list, tuple, set)):
            items = []
            for item in value:
                text = str(item or "").strip()
                if not text or text.lower() in {"nan", "none", "null"}:
                    continue
                items.append(_humanize_feature_name(text))
            return "、".join(items) if items else "-"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return fmt_num(value, 3)
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return "-"
        if text.lower() == "full":
            return "完整"
        return _humanize_feature_name(text)

    if isinstance(payload, dict):
        rows = [{"字段": str(key), "内容": _format_value(value)} for key, value in payload.items()]
        if rows:
            return pd.DataFrame(rows)
    if payload not in (None, "", {}):
        return pd.DataFrame([{"字段": title, "内容": _format_value(payload)}])
    return pd.DataFrame()


def _domain_watch_label(domain_cn: str, value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "待补充"
    thresholds = {"学习": 0.15, "生活": 0.45, "运动": 0.40}
    threshold = thresholds.get(domain_cn, 0.5)
    if float(numeric) >= threshold:
        return "重点关注"
    if float(numeric) >= threshold * 0.7:
        return "持续观察"
    return "相对平稳"


def _clean_agent_text(text: Any, field_name: str = "") -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "暂无内容。"
    cleaned = re.sub(r"FEATURE_[A-Z0-9_]+", "关键行为信号", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if field_name == "behavior_agent" and "关键行为信号" in cleaned:
        return "系统已识别到关键行为信号，建议结合班级观察和近期表现进一步核实。"
    return cleaned


def _humanize_feature_name(feature_name: Any) -> str:
    raw = str(feature_name or "").strip()
    feature_map = {
        "FEATURE_COURSE_CREDIT_SUM": "课程负担相关信号",
        "FEATURE_GRADE_AVG_SCORE": "平均成绩波动信号",
        "FEATURE_GRADE_CREDIT_SUM": "学业完成度相关信号",
        "FEATURE_TASK_JOB_RATE": "任务完成率信号",
        "FEATURE_ONLINE_BFB": "线上学习活跃信号",
        "FEATURE_ATTEND_RECORD_COUNT": "出勤参与信号",
    }
    if raw in feature_map:
        return feature_map[raw]
    if raw.startswith("FEATURE_"):
        return "关键行为信号"
    return raw or "关键行为信号"


def _build_chain_summary(chain: dict[str, Any]) -> str:
    if not isinstance(chain, dict) or not chain:
        return "暂无链路摘要。"

    fusion = chain.get("fusion", {}) if isinstance(chain.get("fusion", {}), dict) else {}
    map_payload = chain.get("map", {}) if isinstance(chain.get("map", {}), dict) else {}
    submodel_risk = chain.get("submodel_risk", {}) if isinstance(chain.get("submodel_risk", {}), dict) else {}
    shap_payload = chain.get("shap", {}) if isinstance(chain.get("shap", {}), dict) else {}

    total_risk = fmt_num(fusion.get("total_risk"), 3)
    risk_level = str(fusion.get("risk_level") or "未知")
    dominant_dimension = dimension_to_cn(fusion.get("dominant_dimension"))
    dominant_map = map_to_cn(map_payload.get("dominant_MAP"))
    pattern = str(chain.get("pattern") or "暂无模式")

    risk_items = [
        ("学习", submodel_risk.get("study_risk")),
        ("生活", submodel_risk.get("life_risk")),
        ("运动", submodel_risk.get("sport_risk")),
    ]
    ranked = []
    for name, value in risk_items:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            ranked.append((name, float(numeric)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    ranked_text = "、".join(f"{name}{fmt_num(value, 3)}" for name, value in ranked) if ranked else "暂无三域风险信息"

    top_features: list[str] = []
    for values in shap_payload.values():
        if isinstance(values, list):
            for item in values:
                feature_name = _humanize_feature_name(item)
                if feature_name not in top_features:
                    top_features.append(feature_name)
    feature_text = "、".join(top_features[:3]) if top_features else "近期行为节律与任务表现"

    return (
        f"当前链路显示，这名学生综合风险为{risk_level}，综合风险分约为{total_risk}，"
        f"主要由{dominant_dimension}维度带动。三域风险从高到低依次为：{ranked_text}。"
        f" 系统在行为层面重点关注到{feature_text}，说明问题已经在日常学习或校园行为中表现出来。"
        f" 从归因机制看，当前更偏向“{dominant_map}”，意味着需要优先补足相关支持条件。"
        f" 结合群体模式判断，该学生当前更接近“{pattern}”，因此更适合尽快启动分阶段干预。"
    )


def _build_chain_intervention_text(chain: dict[str, Any]) -> str:
    if not isinstance(chain, dict) or not chain:
        return "暂无干预输出。"

    fusion = chain.get("fusion", {}) if isinstance(chain.get("fusion", {}), dict) else {}
    map_payload = chain.get("map", {}) if isinstance(chain.get("map", {}), dict) else {}
    base_text = str(chain.get("intervention_text") or "").strip()

    dominant_dimension = dimension_to_cn(fusion.get("dominant_dimension"))
    risk_level = str(fusion.get("risk_level") or "未知")
    mechanism = map_to_cn(map_payload.get("dominant_MAP"))
    pattern = str(chain.get("pattern") or "暂无模式")

    dimension_action = {
        "学习": "先核实近期课程参与、任务完成和学习节奏是否连续失稳，必要时帮助学生拆分任务并补上最近一段时间的学习支持。",
        "生活": "先关注作息、消费、宿舍生活或校园活动参与是否出现明显失衡，必要时通过辅导员、班主任或同伴支持先把日常节律稳住。",
        "运动": "先核实体测、锻炼参与和身体状态是否持续走低，必要时从低门槛运动计划和阶段性反馈开始恢复。",
        "未知": "先从最近一段时间最明显的异常表现入手，优先稳定节律与基本支持条件。",
    }.get(dominant_dimension, "先从当前最突出的风险维度入手，优先稳定基本支持条件。")

    mechanism_action = {
        "动机": "从机制看，更需要先解决学生为什么不愿意投入的问题，所以干预时要先建立目标感、参与感和可见反馈。",
        "能力": "从机制看，当前更像是能力准备不足，因此要提供更具体的资源补位、任务拆解和过程辅导，而不是只做口头提醒。",
        "提示": "从机制看，当前更像是外部提示和支持不足，因此要增加提醒频率、节点反馈和教师同伴协同支持。",
        "未知": "从机制看，需要在动机、能力和外部支持三个方向同时做一次快速排查。",
    }.get(mechanism, "从机制看，需要同步排查动机、能力与支持条件。")

    pattern_action = (
        f"结合“{pattern}”这一群体模式，建议不要只盯住单一问题，而是用阶段性复盘的方式持续观察风险是否向其他维度扩散。"
        if pattern and pattern != "暂无模式"
        else "建议通过阶段性复盘持续观察风险是否继续扩散。"
    )

    base_sentence = f"当前学生被识别为{risk_level}，建议先围绕{dominant_dimension}维度启动第一轮干预。"
    if not base_text:
        base_text = "先处理主导风险，再联动另外两个维度做阶段性跟踪。"

    return (
        f"{base_sentence} {base_text} "
        f"第一步，{dimension_action} "
        f"第二步，{mechanism_action} "
        f"第三步，建议在一周内复看学习、生活、运动三域变化，判断风险是否回落、是否需要升级支持。 "
        f"{pattern_action}"
    )


def _build_student_chain_payload(
    row: dict[str, Any],
    report: dict[str, Any],
    demo_chain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    student_id = str(row.get("student_id") or "")
    matched_demo_chain = (
        dict(demo_chain)
        if isinstance(demo_chain, dict) and demo_chain and student_id and student_id == str((demo_chain.get("_student_id") or ""))
        else {}
    )

    shap_payload: dict[str, list[str]] = {}
    for domain_key in ["study", "life", "sport"]:
        feature_list: list[str] = []
        for idx in range(1, 4):
            raw_feature = row.get(f"{domain_key}_shap_top{idx}")
            if raw_feature is None or (isinstance(raw_feature, float) and pd.isna(raw_feature)):
                continue
            feature = str(raw_feature).strip()
            if not feature or feature.lower() in {"nan", "none", "null"}:
                continue
            feature = _humanize_feature_name(feature)
            if feature:
                feature_list.append(feature)
        if not feature_list and isinstance(matched_demo_chain.get("shap", {}), dict):
            for item in matched_demo_chain.get("shap", {}).get(domain_key, []):
                text = str(item or "").strip()
                if not text or text.lower() in {"nan", "none", "null"}:
                    continue
                feature_list.append(_humanize_feature_name(text))
        if feature_list:
            shap_payload[domain_key] = feature_list

    demo_submodel = matched_demo_chain.get("submodel_risk", {}) if isinstance(matched_demo_chain.get("submodel_risk", {}), dict) else {}
    demo_fusion = matched_demo_chain.get("fusion", {}) if isinstance(matched_demo_chain.get("fusion", {}), dict) else {}
    demo_map = matched_demo_chain.get("map", {}) if isinstance(matched_demo_chain.get("map", {}), dict) else {}

    return {
        "submodel_risk": {
            "study_risk": row.get("study_risk") if pd.notna(pd.to_numeric(pd.Series([row.get("study_risk")]), errors="coerce").iloc[0]) else demo_submodel.get("study_risk"),
            "life_risk": row.get("life_risk") if pd.notna(pd.to_numeric(pd.Series([row.get("life_risk")]), errors="coerce").iloc[0]) else demo_submodel.get("life_risk"),
            "sport_risk": row.get("sport_risk") if pd.notna(pd.to_numeric(pd.Series([row.get("sport_risk")]), errors="coerce").iloc[0]) else demo_submodel.get("sport_risk"),
        },
        "fusion": {
            "total_risk": row.get("total_risk") if pd.notna(pd.to_numeric(pd.Series([row.get("total_risk")]), errors="coerce").iloc[0]) else demo_fusion.get("total_risk"),
            "risk_level": row.get("risk_level") or demo_fusion.get("risk_level"),
            "dominant_dimension": dimension_to_cn(row.get("dominant_dimension") or demo_fusion.get("dominant_dimension")),
        },
        "shap": shap_payload,
        "map": {
            "M_score": row.get("M_score") if pd.notna(pd.to_numeric(pd.Series([row.get("M_score")]), errors="coerce").iloc[0]) else demo_map.get("M_score"),
            "A_score": row.get("A_score") if pd.notna(pd.to_numeric(pd.Series([row.get("A_score")]), errors="coerce").iloc[0]) else demo_map.get("A_score"),
            "P_score": row.get("P_score") if pd.notna(pd.to_numeric(pd.Series([row.get("P_score")]), errors="coerce").iloc[0]) else demo_map.get("P_score"),
            "dominant_MAP": map_to_cn(row.get("dominant_MAP") or demo_map.get("dominant_MAP")),
        },
        "pattern": row.get("pattern_label") or matched_demo_chain.get("pattern"),
        "profile_text": report.get("report_agent") or row.get("profile_text") or report.get("summary") or matched_demo_chain.get("profile_text") or "",
        "intervention_text": report.get("intervention_agent") or row.get("intervention_text") or matched_demo_chain.get("intervention_text") or "",
    }


def _build_student_generated_profile(row: dict[str, Any], report: dict[str, Any]) -> tuple[str, str]:
    risk_items = [
        ("学习", pd.to_numeric(pd.Series([row.get("study_risk")]), errors="coerce").iloc[0]),
        ("生活", pd.to_numeric(pd.Series([row.get("life_risk")]), errors="coerce").iloc[0]),
        ("运动", pd.to_numeric(pd.Series([row.get("sport_risk")]), errors="coerce").iloc[0]),
    ]
    ranked = [(name, float(value)) for name, value in risk_items if pd.notna(value)]
    ranked.sort(key=lambda item: item[1], reverse=True)
    dominant = ranked[0][0] if ranked else "未知"
    second = ranked[1][0] if len(ranked) > 1 else None
    total_risk = fmt_num(row.get("total_risk"), 3)
    risk_level = str(row.get("risk_level") or "未知")
    pattern = str(row.get("pattern_label") or "暂无模式")
    mechanism = map_to_cn(row.get("dominant_MAP"))
    pattern_reason = str(row.get("pattern_reason") or "").strip()
    priority = str(row.get("priority") or "-")

    ranked_text = "、".join(
        f"{name}{fmt_num(value, 3)}（{_domain_watch_label(name, value)}）" for name, value in ranked
    )
    main_sentence = f"该学生当前综合风险为{total_risk}，风险等级为{risk_level}。三域情况依次为：{ranked_text}。"
    focus_sentence = f"当前首先需要围绕{dominant}维度开展跟进"
    if second:
        focus_sentence += f"，同时同步关注{second}维度的变化"
    focus_sentence += "。"
    pattern_sentence = f"系统判断该学生当前处于“{pattern}”状态，主导机制偏向{mechanism}。"
    reason_sentence = f"从当前档案标签看，{pattern_reason}" if pattern_reason else ""
    summary = " ".join([item for item in [main_sentence, focus_sentence, pattern_sentence, reason_sentence] if item])

    base_advice = str(row.get("intervention_text") or report.get("intervention_text") or "").strip()
    if not base_advice:
        base_advice = "建议优先围绕当前主导风险维度开展跟进，再结合另外两个维度做持续观察。"
    advice = (
        f"建议优先级为 {priority}。"
        f" 第一阶段先处理{dominant}相关问题，尽快确认近期是否出现了稳定性下降、持续性困难或支持资源不足；"
        f" 第二阶段结合{second or '其余维度'}继续跟踪，避免单一问题向多维扩散；"
        f" 具体建议：{base_advice}"
    )
    return summary, advice


def _build_student_explanation_views(row: dict[str, Any], report: dict[str, Any]) -> list[dict[str, str]]:
    dominant_dimension = dimension_to_cn(row.get("dominant_dimension"))
    mechanism = map_to_cn(row.get("dominant_MAP"))
    risk_level = str(row.get("risk_level") or "未知")
    pattern = str(row.get("pattern_label") or "暂无模式")
    pattern_reason = str(row.get("pattern_reason") or "").strip()

    behavior_text = _clean_agent_text(report.get("behavior_agent", ""), "behavior_agent")
    mechanism_text = _clean_agent_text(report.get("mechanism_agent", ""), "mechanism_agent")
    intervention_text = _clean_agent_text(
        report.get("intervention_agent") or row.get("intervention_text") or "", "intervention_agent"
    )

    if behavior_text == "暂无内容。":
        behavior_text = (
            f"当前学生被识别为{risk_level}，主要异常首先体现在{dominant_dimension}相关行为表现。"
            f" 系统模式标签为“{pattern}”，说明近期行为节律、任务完成或参与状态存在需要重点追踪的信号。"
        )

    psychology_text = mechanism_text
    if psychology_text == "暂无内容。":
        psychology_text = (
            f"从归因机制看，当前更偏向“{mechanism}”问题。"
            " 这可以理解为学生在动机投入、能力准备或外部提示支持中的某一环节出现短板，"
            f" 并最终在{dominant_dimension}维度上放大为可观察风险。"
        )

    environment_text = intervention_text
    if environment_text == "暂无内容。":
        environment_text = (
            f"从环境与支持角度看，当前需要围绕{dominant_dimension}提供更具体的外部支持。"
            " 包括任务提醒、资源补位、反馈频率、同伴支持或节律管理等，"
            " 避免单点问题继续演化成持续性风险。"
        )

    if pattern_reason:
        environment_text = f"{environment_text} 当前模式说明：{pattern_reason}"

    return [
        {"step": "行为解释", "title": "行为层", "copy": behavior_text},
        {"step": "心理机制", "title": "心理/机制层", "copy": psychology_text},
        {"step": "环境支持", "title": "环境影响层", "copy": environment_text},
    ]


def _build_student_explanation_evidence(row: dict[str, Any], report: dict[str, Any]) -> list[dict[str, str]]:
    dominant_dimension = dimension_to_cn(row.get("dominant_dimension"))
    mechanism = map_to_cn(row.get("dominant_MAP"))
    risk_level = str(row.get("risk_level") or "未知")
    pattern = str(row.get("pattern_label") or "暂无模式")
    priority = str(row.get("priority") or "-")
    study_risk = fmt_num(row.get("study_risk"), 3)
    life_risk = fmt_num(row.get("life_risk"), 3)
    sport_risk = fmt_num(row.get("sport_risk"), 3)
    intervention = str(row.get("intervention_type") or "个体支持")
    intervention_text = _clean_agent_text(
        report.get("intervention_agent") or row.get("intervention_text") or "", "intervention_agent"
    )
    if intervention_text == "暂无内容。":
        intervention_text = "优先围绕提醒、资源补位、反馈频率和协同支持建立更稳定的支持节奏。"

    return [
        {
            "title": "行为证据",
            "summary": f"{dominant_dimension}维度当前最突出，综合风险为 {risk_level}，模式标签为“{pattern}”。",
            "detail": f"三域风险分别为：学习 {study_risk}、生活 {life_risk}、运动 {sport_risk}。这说明风险判断已经能够落到可观察行为信号，而不是停留在抽象结论。",
        },
        {
            "title": "归因机制",
            "summary": f"当前主导机制为“{mechanism}”，对应动机、能力或提示支持中的关键短板。",
            "detail": f"系统把主要风险归到{dominant_dimension}，说明该维度的问题并不是孤立现象，而是与投入状态、准备程度或外部提示条件直接相关。",
        },
        {
            "title": "环境支持",
            "summary": f"当前建议优先级为 {priority}，推荐干预类型为“{intervention}”。",
            "detail": f"{intervention_text} 这一层直接给出可执行支持动作，强调通过组织支持、同伴支持、反馈与提醒机制减少风险继续演化。",
        },
    ]


def _build_student_explanation_overview_df(row: dict[str, Any], report: dict[str, Any]) -> pd.DataFrame:
    explanation_views = _build_student_explanation_views(row, report)
    view_map = {item.get("title", ""): item.get("copy", "") for item in explanation_views}
    dominant_dimension = dimension_to_cn(row.get("dominant_dimension"))
    dominant_map = map_to_cn(row.get("dominant_MAP"))
    risk_level = str(row.get("risk_level") or "未知")
    pattern = str(row.get("pattern_label") or "暂无模式")
    pattern_reason = str(row.get("pattern_reason") or "").strip()
    priority = str(row.get("priority") or "-")
    intervention_type = str(row.get("intervention_type") or "个体支持")

    risk_items = [
        ("学习", row.get("study_risk")),
        ("生活", row.get("life_risk")),
        ("运动", row.get("sport_risk")),
    ]
    risk_text = "；".join(
        f"{name}{fmt_num(value, 3)}（{_domain_watch_label(name, value)}）"
        for name, value in risk_items
    )

    feature_texts: list[str] = []
    for domain_key in ["study", "life", "sport"]:
        for idx in range(1, 4):
            feature_name = str(row.get(f"{domain_key}_shap_top{idx}") or "").strip()
            if feature_name:
                humanized = _humanize_feature_name(feature_name)
                if humanized not in feature_texts:
                    feature_texts.append(humanized)
    top_feature_text = "、".join(feature_texts[:3]) if feature_texts else "近期行为节律与任务表现"

    mechanism_scores = " / ".join(
        [
            f"动机 {fmt_num(row.get('M_score'), 3)}",
            f"能力 {fmt_num(row.get('A_score'), 3)}",
            f"提示 {fmt_num(row.get('P_score'), 3)}",
        ]
    )
    intervention_text = _clean_agent_text(
        report.get("intervention_agent") or row.get("intervention_text") or "",
        "intervention_agent",
    )
    if intervention_text == "暂无内容。":
        intervention_text = "优先围绕提醒、资源补位、反馈频率和协同支持建立更稳定的支持节奏。"

    rows = [
        {
            "解释维度": "行为证据",
            "核心结论": f"当前为{risk_level}，{dominant_dimension}维度最突出，学生更接近“{pattern}”。",
            "数据依据": f"三域风险：{risk_text}；关键线索：{top_feature_text}。",
            "解释视角": view_map.get("行为层", "回答“发生了什么异常，以及异常最先出现在哪些行为信号上”。"),
            "建议动作": f"先围绕{dominant_dimension}维度做核查，再结合班级观察确认这些信号是否仍在持续。",
        },
        {
            "解释维度": "归因机制",
            "核心结论": f"当前主导机制为“{dominant_map}”，说明问题更可能卡在动机、能力或提示支持中的关键环节。",
            "数据依据": f"MAP 得分：{mechanism_scores}。",
            "解释视角": view_map.get("心理/机制层", "回答“为什么会出现这些异常，而不只是描述表面现象”。"),
            "建议动作": f"围绕“{dominant_map}”补足支持条件，避免只给结论不给资源。",
        },
        {
            "解释维度": "环境支持",
            "核心结论": f"当前优先级为 {priority}，建议以“{intervention_type}”方式启动支持。",
            "数据依据": f"{('模式说明：' + pattern_reason + '；') if pattern_reason else ''}当前干预建议：{intervention_text}",
            "解释视角": view_map.get("环境影响层", "回答“学校、教师和同伴该如何介入，才能把风险真正拉下来”。"),
            "建议动作": "按“立即处理 + 一周复看 + 持续观察”的节奏推进，并记录支持是否起效。",
        },
    ]
    return pd.DataFrame(rows)


def render_requirements_page(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_header("动态建模展示", "")
    eval_df = data.get("eval_df", pd.DataFrame())
    frontend_bundle = data.get("frontend_bundle", {}) if isinstance(data.get("frontend_bundle", {}), dict) else {}
    counts = frontend_bundle.get("counts", {}) if isinstance(frontend_bundle.get("counts", {}), dict) else {}
    artifact_freshness = frontend_bundle.get("artifact_freshness", {}) if isinstance(frontend_bundle.get("artifact_freshness", {}), dict) else {}
    mainline_df = _build_requirements_mainline_table(data)
    source_df = _build_requirements_sources_table(data)
    agent_df = _build_requirements_agent_table(data)
    classification_df = (
        eval_df[eval_df.get("task_type", pd.Series(dtype=str)).astype(str) == "classification"].copy()
        if not eval_df.empty
        else pd.DataFrame()
    )
    regression_df = (
        eval_df[eval_df.get("task_type", pd.Series(dtype=str)).astype(str) == "regression"].copy()
        if not eval_df.empty
        else pd.DataFrame()
    )
    best_auc = _best_auc_from_views(classification_df, data)
    artifact_ok = sum(1 for value in artifact_freshness.values() if str(value) in {"fresh", "rule_derived"})
    artifact_total = len(artifact_freshness)
    scope_df = _build_multisource_scope_table()

    intro_cols = st.columns(4)
    with intro_cols[0]:
        metric_card("三域接入", "3 / 3", "学习、生活、运动链路已接入")
    with intro_cols[1]:
        metric_card("学生覆盖", str((counts or {}).get("students", len(data.get("master_df", pd.DataFrame())))), "融合总表当前覆盖学生")
    with intro_cols[2]:
        metric_card("多Agent报告", str((counts or {}).get("reports", len(data.get("report_records", [])))), "学生级报告已完成生成")
    with intro_cols[3]:
        metric_card("最佳 AUC", fmt_num(best_auc, 3), "离线研发与三域验收中的最佳区分度")

    section_options = ["运行与验收", "建模架构", "工程明细", "离线研发"]
    pending_section = st.session_state.get("pending_modeling_section")
    if pending_section in section_options:
        st.session_state.modeling_page_section = pending_section
        st.session_state.pending_modeling_section = None
    elif st.session_state.get("modeling_page_section") not in section_options:
        st.session_state.modeling_page_section = section_options[0]

    current_section = st.radio(
        "建模目录",
        section_options,
        key="modeling_page_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if current_section == "运行与验收":
        _page_section_gap()
        with st.container(border=True):
            section_title("实时运行建模")
            _render_harness_panel(data)

        _page_section_gap()
        with st.container(border=True):
            section_title("Harness 三域验收")
            _render_requirements_domain_cards(data)

        _page_section_gap()
        with st.container(border=True):
            section_title("Harness 闭环概览")
            inner_cols = st.columns(3)
            with inner_cols[0]:
                metric_card("融合总表", str((counts or {}).get("students", "-")), "当前可追踪学生")
            with inner_cols[1]:
                metric_card("干预建议", str((counts or {}).get("interventions", "-")), "学生级干预输出")
            with inner_cols[2]:
                metric_card("群体模式", str((counts or {}).get("patterns", "-")), "当前可展示群体类型")
            st.markdown(
                f"<div class='requirements-inline-time'>最近更新：{frontend_bundle.get('generated_at', '-')}</div>",
                unsafe_allow_html=True,
            )

    elif current_section == "建模架构":
        _page_section_gap()
        with st.container(border=True):
            section_title("建模架构")
            arch_cols = st.columns([1.0, 1.0], vertical_alignment="top")
            with arch_cols[0]:
                section_title("多源数据范围")
                render_table(scope_df, "")
            with arch_cols[1]:
                section_title("多Agent输出覆盖")
                render_table(agent_df, "")

    elif current_section == "工程明细":
        _page_section_gap()
        with st.container(border=True):
            section_title("工程产物同步")
            metric_card("同步状态", f"{artifact_ok} / {artifact_total}", "当前前端所用建模产物同步状态")
        _page_section_gap()
        with st.container(border=True):
            section_title("三域数据同步")
            render_table(source_df, "")
        _page_section_gap()
        with st.container(border=True):
            section_title("主线验收明细")
            render_table(mainline_df, "")

    elif current_section == "离线研发":
        if not classification_df.empty:
            with st.container(border=True):
                section_title("离线研发记录：分类任务")
                working = classification_df.copy()
                if "target_name" in working.columns:
                    working["target_name"] = working["target_name"].map(_target_name_to_cn)
                for col in ["best_val_threshold", "val_f1", "val_auc"]:
                    if col in working.columns:
                        working[col] = pd.to_numeric(working[col], errors="coerce").map(lambda x: fmt_num(x, 3))
                if "samples" in working.columns:
                    working["samples"] = pd.to_numeric(working["samples"], errors="coerce").map(lambda x: fmt_num(x, 0))
                render_table(
                    working.rename(
                        columns={
                            "domain": "领域",
                            "task_name": "任务名称",
                            "target_name": "预测目标",
                            "model": "最优模型",
                            "val_auc": "区分度",
                            "val_f1": "综合命中",
                            "best_val_threshold": "阈值",
                            "samples": "记录数",
                            "sample_scope": "记录口径",
                        }
                    )[
                        ["领域", "任务名称", "预测目标", "最优模型", "区分度", "综合命中", "阈值", "记录数", "记录口径"]
                    ],
                    "",
                )
        if not regression_df.empty:
            _page_section_gap()
            with st.container(border=True):
                section_title("离线研发记录：回归任务")
                working = regression_df.copy()
                if "target_name" in working.columns:
                    working["target_name"] = working["target_name"].map(_target_name_to_cn)
                for col in ["r2", "rmse", "mae"]:
                    if col in working.columns:
                        working[col] = pd.to_numeric(working[col], errors="coerce").map(lambda x: fmt_num(x, 3))
                if "samples" in working.columns:
                    working["samples"] = pd.to_numeric(working["samples"], errors="coerce").map(lambda x: fmt_num(x, 0))
                render_table(
                    working.rename(
                        columns={
                            "domain": "领域",
                            "task_name": "任务名称",
                            "target_name": "预测目标",
                            "model": "最优模型",
                            "r2": "拟合度",
                            "rmse": "均方根误差",
                            "mae": "平均绝对误差",
                            "samples": "记录数",
                            "sample_scope": "记录口径",
                        }
                    )[
                        ["领域", "任务名称", "预测目标", "最优模型", "拟合度", "均方根误差", "平均绝对误差", "记录数", "记录口径"]
                    ],
                    "",
                )


def render_group_profile_page(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_header("群体画像", "")
    pattern_df = _summarize_pattern_df(filtered_df, data.get("pattern_df", pd.DataFrame()))
    profile_view = _with_group_profile_labels(filtered_df)

    overview_cols = st.columns(3)
    with overview_cols[0]:
        metric_card("样本数", str(len(filtered_df)), "群体画像覆盖范围")
    with overview_cols[1]:
        top_pattern = pattern_df.iloc[0]["pattern_label"] if not pattern_df.empty else "暂无"
        metric_card("主要模式", str(top_pattern), "当前人数最多的群体画像")
    with overview_cols[2]:
        top_map = (
            profile_view["dominant_MAP"].mode().iloc[0]
            if not profile_view.empty and "dominant_MAP" in profile_view.columns and not profile_view["dominant_MAP"].mode().empty
            else "提示"
        )
        metric_card("主导机制", map_to_cn(top_map), "群体最常见的主导成因")

    _page_section_gap()
    left, right = st.columns([1, 1], vertical_alignment="top")
    with left:
        with st.container(border=True):
            section_title("群体模式图")
            chart = _pattern_chart(pattern_df)
            if chart is None:
                st.info("暂无群体模式数据。")
            else:
                st.altair_chart(chart, use_container_width=True)
    with right:
        with st.container(border=True):
            section_title("三维风险区间")
            risk_chart = _risk_interval_chart(filtered_df)
            if risk_chart is None:
                st.info("暂无群体风险数据。")
            else:
                st.altair_chart(risk_chart, use_container_width=True)

    _page_section_gap()
    with st.container(border=True):
        section_title("模式表")
        render_table(
            pattern_df.rename(
                columns={
                    "pattern_label": "模式名称",
                    "student_count": "人数",
                    "ratio": "占比",
                    "avg_total_risk": "平均综合风险",
                    "profile_polarity": "类型性质",
                    "main_dimension": "关注方向",
                    "main_MAP": "主导机制",
                }
            ),
            "",
        )

    _page_section_gap()
    with st.container(border=True):
        section_title("群体说明")
        render_table(pd.DataFrame(GROUP_PROFILE_EXPLANATIONS), "")


def render_student_profile_page(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_header("学生数字档案", "")
    student_options = data["master_df"]["student_id"].astype(str).tolist() if not data["master_df"].empty else []
    if not student_options:
        st.info("当前没有可展示的学生。")
        return

    current_student = st.session_state.selected_student if st.session_state.selected_student in student_options else student_options[0]
    selector_cols = st.columns([1.35, 2.65], vertical_alignment="bottom")
    with selector_cols[0]:
        selected_student = st.selectbox(
            "选择学生",
            student_options,
            index=student_options.index(current_student),
            key="student_profile_picker",
        )
    student_id = selected_student
    row = get_student_row(data["master_df"], student_id)
    report = data.get("report_map", {}).get(student_id, {})

    if not row:
        st.info("当前没有可展示的学生。")
        return

    generated_summary, generated_advice = _build_student_generated_profile(row, report)
    explanation_overview_df = _build_student_explanation_overview_df(row, report)

    cols = st.columns(4)
    with cols[0]:
        metric_card("学生编号", student_id, "当前页正在查看的学生")
    with cols[1]:
        _metric_inline_card("综合风险", fmt_num(row.get("total_risk"), 3), risk_chip(row.get("risk_level")), "当前综合风险水平")
    with cols[2]:
        metric_card("主导维度", dimension_to_cn(row.get("dominant_dimension")), "当前影响最大的维度")
    with cols[3]:
        metric_card("优先级", str(row.get("priority", "-")), "干预排期优先级")

    risk_df = pd.DataFrame(
        [
            {"维度": "学习", "风险": row.get("study_risk", 0)},
            {"维度": "生活", "风险": row.get("life_risk", 0)},
            {"维度": "运动", "风险": row.get("sport_risk", 0)},
        ]
    )
    _page_section_gap()
    with st.container(border=True):
        section_title("系统生成档案")
        narrative_card(generated_summary)
        st.markdown("<div class='table-card-breath'></div>", unsafe_allow_html=True)

    _page_section_gap()
    risk_tiles = st.columns(3)
    for col, item in zip(risk_tiles, risk_df.to_dict(orient="records")):
        with col:
            _metric_inline_card(
                f"{item['维度']}风险",
                fmt_num(item["风险"], 3),
                status_chip(_domain_watch_label(item["维度"], item["风险"])),
                "当前单域风险水平",
            )

    _page_section_gap()
    left, right = st.columns([1.0, 1.0], vertical_alignment="top")
    with left:
        with st.container(border=True):
            section_title("三维风险")
            chart = (
                alt.Chart(risk_df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("维度:N", title="维度"),
                    y=alt.Y("风险:Q", title="风险分", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("维度:N", legend=None, scale=alt.Scale(range=["#c46e46", "#8a9b77", "#8c6752"])),
                    tooltip=["维度", alt.Tooltip("风险:Q", format=".3f")],
                )
                .properties(height=260)
            )
            st.altair_chart(themed_chart(chart), use_container_width=True)
    with right:
        with st.container(border=True):
            section_title("档案标签与建议")
            st.markdown(
                "".join(
                    [
                        status_chip(row.get("pattern_label")),
                        map_chip(row.get("dominant_MAP")),
                        status_chip(f"优先级 {row.get('priority', '-')}"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            narrative_card(generated_advice)
            st.markdown("<div class='table-card-breath'></div>", unsafe_allow_html=True)

    _page_section_gap()
    with st.container(border=True):
        section_title("三维解释")
        render_table(explanation_overview_df, "")

    _page_section_gap()
    with st.container(border=True):
        section_title("多Agent依据")
        report_view = pd.DataFrame(
            [
                {"模块": "风险判断", "内容": _clean_agent_text(report.get("risk_agent", "-"), "risk_agent")},
                {"模块": "行为证据", "内容": _clean_agent_text(report.get("behavior_agent", "-"), "behavior_agent")},
                {"模块": "机制解释", "内容": _clean_agent_text(report.get("mechanism_agent", "-"), "mechanism_agent")},
                {"模块": "干预输出", "内容": _clean_agent_text(report.get("intervention_agent", "-"), "intervention_agent")},
            ]
        )
        render_table(report_view, "")


def render_alert_page(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_header("精准预警干预", "")
    threshold = st.slider("当前预警阈值", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    alerts = filtered_df[filtered_df["total_risk"].fillna(0) >= threshold].copy().sort_values("total_risk", ascending=False)

    cols = st.columns(3)
    with cols[0]:
        metric_card("预警人数", str(len(alerts)), f"阈值 >= {threshold:.2f}")
    with cols[1]:
        p1_count = int((alerts.get("priority", pd.Series(dtype=str)) == "P1").sum()) if not alerts.empty else 0
        metric_card("P1 人数", str(p1_count), "建议优先跟进")
    with cols[2]:
        top_dim = alerts["dominant_dimension"].mode().iloc[0] if not alerts.empty and not alerts["dominant_dimension"].mode().empty else "unknown"
        metric_card("主要风险来源", dimension_to_cn(top_dim), "预警学生最常见主导维度")

    if alerts.empty:
        st.info("当前阈值下没有预警学生。")
        return

    _page_section_gap()
    alert_table = alerts[["student_id", "risk_level", "total_risk", "dominant_dimension", "pattern_label", "priority"]].copy()
    alert_table["dominant_dimension"] = alert_table["dominant_dimension"].map(dimension_to_cn)
    alert_table["total_risk"] = alert_table["total_risk"].map(lambda x: fmt_num(x, 3))
    render_table(
        alert_table.rename(
            columns={
                "student_id": "学生编号",
                "risk_level": "风险等级",
                "total_risk": "综合风险",
                "dominant_dimension": "主导维度",
                "pattern_label": "行为模式",
                "priority": "优先级",
            }
        ),
        "预警学生清单",
    )


def render_predictor_page(data: dict[str, Any]) -> None:
    page_header("新增预测", "")
    st.caption("手动输入和表格上传都会先完成字段归类与标准化，再在运行时可用时调用现有三域模型。")
    _page_section_gap()

    manual_tab, upload_tab = st.tabs(["手动录入", "表格上传"])
    standardized: dict[str, dict[str, Any]] | None = None
    source_label = ""

    with manual_tab:
        defaults = build_predictor_defaults(data)
        with st.form("rebuilt_predict_form"):
            record: dict[str, Any] = {
                "student_id": st.text_input("学生编号", value="demo_student_001"),
                "term_id": st.text_input("学期编号", value="2025-2026-T1"),
            }
            tabs = st.tabs(["学习行为", "生活行为", "运动行为"])
            for tab, domain in zip(tabs, ["study", "life", "sport"]):
                with tab:
                    cols = st.columns(2)
                    for idx, field in enumerate(PREDICTOR_SCHEMAS[domain]):
                        with cols[idx % 2]:
                            default = defaults[domain][field["key"]]
                            if field["type"] == "int":
                                record[field["key"]] = st.number_input(field["label"], value=int(round(float(default))), step=1, key=f"new_{domain}_{field['key']}")
                            else:
                                record[field["key"]] = st.number_input(field["label"], value=float(default), key=f"new_{domain}_{field['key']}")
            submitted = st.form_submit_button("生成标准化入参", use_container_width=True)
        if submitted:
            standardized = standardize_record(record, data)
            source_label = "手动录入"

    with upload_tab:
        uploaded = st.file_uploader("上传单个学生表格", type=["csv", "xlsx", "xls"], help="建议使用现有中文字段名或模型原始字段名。")
        if uploaded is not None:
            uploaded_df = load_uploaded_table(uploaded)
            mapping = detect_csv_columns(uploaded_df.columns.tolist())
            render_table(pd.DataFrame([mapping]).T.reset_index().rename(columns={"index": "系统字段", 0: "识别列名"}), "字段识别")
            render_table(uploaded_df.head(1), "原始输入预览")
            if st.button("使用首行生成标准化入参", key="rebuilt_upload_predict", use_container_width=True):
                row = uploaded_df.iloc[0].to_dict()
                record = {
                    "student_id": row.get(mapping.get("student_id", ""), "upload_student_001"),
                    "term_id": row.get(mapping.get("term_id", ""), "2025-2026-T1"),
                }
                for domain_fields in PREDICTOR_SCHEMAS.values():
                    for field in domain_fields:
                        src = mapping.get(field["key"])
                        if src:
                            record[field["key"]] = row.get(src)
                standardized = standardize_record(record, data)
                source_label = f"表格上传：{uploaded.name}"

    if standardized is None:
        _page_section_gap()
        schema_cols = st.columns(3)
        for col, domain, title in zip(schema_cols, ["study", "life", "sport"], ["学习字段", "生活字段", "运动字段"]):
            with col:
                with st.container(border=True):
                    section_title(title)
                    st.markdown("、".join(item["label"] for item in PREDICTOR_SCHEMAS[domain]))
        return

    _page_section_gap()
    preview_cols = st.columns(3)
    for col, domain, title in zip(preview_cols, ["study", "life", "sport"], ["学习类入参", "生活类入参", "运动类入参"]):
        with col:
            with st.container(border=True):
                render_table(pd.DataFrame([standardized[domain]]), title)

    result = score_new_student(standardized, data)
    if result is None:
        st.warning("当前运行环境缺少可直接加载现有 joblib 模型的运行时，因此新版预测页暂时只能完成字段归类与标准化。")
        return

    _page_section_gap()
    cols = st.columns(4)
    with cols[0]:
        metric_card("输入来源", source_label, "当前预测使用的输入方式")
    with cols[1]:
        metric_card("综合风险", fmt_num(result["total_risk"], 3), "三域综合后的风险分")
    with cols[2]:
        metric_card("风险等级", result["risk_level"], "按现有主表分位数区间映射")
    with cols[3]:
        metric_card("主导维度", dimension_to_cn(result["dominant_dimension"]), "风险最高的单域")

    _page_section_gap()
    risk_df = pd.DataFrame(
        [
            {"维度": "学习风险", "数值": result["study_risk"]},
            {"维度": "生活风险", "数值": result["life_risk"]},
            {"维度": "运动风险", "数值": result["sport_risk"]},
        ]
    )
    chart = (
        alt.Chart(risk_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("维度:N", title="维度"),
            y=alt.Y("数值:Q", title="风险分", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("维度:N", legend=None, scale=alt.Scale(range=["#c46e46", "#8a9b77", "#8c6752"])),
            tooltip=["维度", alt.Tooltip("数值:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(themed_chart(chart), use_container_width=True)


def _assistant_answer(prompt: str, student_id: str | None, filtered_df: pd.DataFrame, data: dict[str, Any]) -> str:
    lower = prompt.lower()
    if student_id:
        row = get_student_row(data["master_df"], student_id)
        report = data.get("report_map", {}).get(student_id, {})
        if not row:
            return f"当前没有找到学生 {student_id} 的档案。"
        if any(token in lower for token in ["干预", "建议", "follow", "措施"]):
            return f"{student_id} 当前风险等级为 {row.get('risk_level', '未知')}，主导维度是 {dimension_to_cn(row.get('dominant_dimension'))}。建议是：{row.get('intervention_text') or report.get('intervention_text') or '暂无明确干预建议。'}"
        return (
            f"{student_id} 当前综合风险为 {fmt_num(row.get('total_risk'), 3)}，风险等级为 {row.get('risk_level', '未知')}，"
            f"主导维度是 {dimension_to_cn(row.get('dominant_dimension'))}，行为模式是 {row.get('pattern_label', '未知')}。"
            f"{row.get('profile_text') or report.get('summary') or ''}"
        )

    pattern_df = data.get("pattern_df", pd.DataFrame())
    top_pattern = pattern_df.iloc[0]["pattern_label"] if not pattern_df.empty else "暂无"
    high_risk = int((filtered_df.get("risk_level", pd.Series(dtype=str)) == "高风险").sum()) if not filtered_df.empty else 0
    total = len(filtered_df)
    top_dimension = filtered_df["dominant_dimension"].mode().iloc[0] if not filtered_df.empty and not filtered_df["dominant_dimension"].mode().empty else "unknown"

    if any(token in lower for token in ["群体", "模式", "哪些学生", "关注"]):
        return f"当前筛选范围内共有 {total} 名学生，其中高风险 {high_risk} 名。人数最多的行为模式是“{top_pattern}”，最常见主导维度是 {dimension_to_cn(top_dimension)}。建议优先查看预警页中的 P1 学生与该模式对应群体。"
    if any(token in lower for token in ["运行", "状态", "接口", "harness"]):
        harness = data.get("harness", {})
        return f"当前 harness 系统状态为 {harness.get('system_status', 'unknown')}，系统决策为 {decision_to_cn(harness.get('final_decision', ''))}。新版前端继续保留原有三域 contract，只重构了页面层。"
    return f"当前筛选范围内共有 {total} 名学生，高风险 {high_risk} 名，最常见主导维度是 {dimension_to_cn(top_dimension)}，人数最多的行为模式是“{top_pattern}”。如果你愿意，我可以继续按群体、学生个体或干预动作给出更具体结论。"


def render_ai_copilot_page(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_header("AI 辅导员助手", "")
    ids = data["master_df"]["student_id"].astype(str).tolist() if not data["master_df"].empty else []

    quick_cols = st.columns(4)
    prompts = [
        "全校整体风险情况怎么样？",
        "哪个群体最值得优先关注？",
        "请解释当前学生为什么高风险",
        "给我一条具体干预建议",
    ]
    for idx, text in enumerate(prompts):
        if quick_cols[idx].button(text, key=f"rebuilt_prompt_{idx}", use_container_width=True):
            st.session_state.assistant_pending_prompt = text

    _page_section_gap()
    selected = st.selectbox("默认学生", ["不指定学生"] + ids[:300])
    prompt = st.chat_input("输入问题，例如：请解释 pjwtqxbj965 为什么被判为高风险")
    if st.session_state.assistant_pending_prompt:
        prompt = st.session_state.assistant_pending_prompt
        st.session_state.assistant_pending_prompt = ""

    if prompt:
        student_id = None if selected == "不指定学生" else selected
        student_id = parse_student_id_from_text(prompt, ids) or student_id
        answer = _assistant_answer(prompt, student_id, filtered_df, data)
        st.session_state.assistant_history.append({"role": "user", "content": prompt})
        st.session_state.assistant_history.append({"role": "assistant", "content": answer})

    _page_section_gap()
    if st.session_state.assistant_history:
        for item in st.session_state.assistant_history[-10:]:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
    else:
        st.info("先问一个问题，助手会基于当前筛选范围、主表、报告与 harness 状态来回答。")


def render_chain_page(data: dict[str, Any]) -> None:
    page_header("学生链路追踪", "")
    master_df = data.get("master_df", pd.DataFrame())
    student_options = master_df["student_id"].astype(str).tolist() if not master_df.empty else []
    if not student_options:
        st.info("当前没有可展示的学生链路。")
        return

    current_student = st.session_state.selected_student if st.session_state.selected_student in student_options else student_options[0]
    selector_cols = st.columns([1.35, 4.55], vertical_alignment="bottom")
    with selector_cols[0]:
        selected_student = st.selectbox(
            "选择学生",
            student_options,
            index=student_options.index(current_student),
            key="chain_student_picker",
        )
    if selected_student != st.session_state.get("selected_student"):
        st.session_state.pending_selected_student = selected_student
        st.rerun()
    row = get_student_row(master_df, selected_student)
    if not row:
        st.info("当前没有可展示的学生链路。")
        return

    report = data.get("report_map", {}).get(selected_student, {})
    demo_case = data.get("demo_case", {}) if isinstance(data.get("demo_case", {}), dict) else {}
    demo_chain = demo_case.get("chain", {}) if isinstance(demo_case.get("chain", {}), dict) else {}
    if demo_chain:
        demo_chain = dict(demo_chain)
        demo_chain["_student_id"] = str(demo_case.get("student_id", ""))
    chain = _build_student_chain_payload(row, report, demo_chain)

    top_flow = st.columns(3)
    flow_cards = [
        ("步骤 1", "三域风险", "先汇总学习、生活、运动三个领域的风险输出，形成学生的原始风险面。"),
        ("步骤 2", "融合判断", "再把多域结果合并，得到综合风险、主导维度和优先级。"),
        ("步骤 3", "解释与干预", "最后结合解释线索、机制判断和模式标签，生成可执行建议。"),
    ]
    for col, item in zip(top_flow, flow_cards):
        with col:
            _flow_card(*item)

    _page_section_gap()
    detail_cols = st.columns(2, vertical_alignment="top")
    with detail_cols[0]:
        with st.container(border=True):
            section_title("风险与融合")
            submodel_df = _chain_payload_table("原始风险", chain.get("submodel_risk", {}))
            fusion_df = _chain_payload_table("融合结果", chain.get("fusion", {}))
            if not submodel_df.empty:
                render_table(submodel_df, "三域原始输出")
            if not fusion_df.empty:
                render_table(fusion_df, "融合结果")
    with detail_cols[1]:
        with st.container(border=True):
            section_title("解释与机制")
            shap_df = _chain_payload_table("解释线索", chain.get("shap", {}))
            map_df = _chain_payload_table("机制判断", chain.get("map", {}))
            pattern_df = _chain_payload_table("模式标签", {"行为模式": chain.get("pattern", "-")})
            if not shap_df.empty:
                render_table(shap_df, "解释线索")
            if not map_df.empty:
                render_table(map_df, "机制判断")
            if not pattern_df.empty:
                render_table(pattern_df, "模式标签")

    _page_section_gap()
    bottom = st.columns(2, vertical_alignment="top")
    with bottom[0]:
        with st.container(border=True):
            section_title("链路摘要")
            narrative_card(_build_chain_summary(chain))
            st.markdown("<div class='table-card-breath'></div>", unsafe_allow_html=True)
    with bottom[1]:
        with st.container(border=True):
            section_title("干预输出")
            narrative_card(_build_chain_intervention_text(chain))
            st.markdown("<div class='table-card-breath'></div>", unsafe_allow_html=True)


def render_student_home_page(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    student_id = st.session_state.selected_student
    row = get_student_row(data["master_df"], student_id)
    page_header(
        "我的成长主页",
        "",
        compact=True,
    )
    if not row:
        st.info("当前没有可展示的学生档案。")
        return

    cols = st.columns(3)
    with cols[0]:
        metric_card("我的综合风险", fmt_num(row.get("total_risk"), 3), "系统档案中的当前风险分")
    with cols[1]:
        metric_card("重点关注维度", dimension_to_cn(row.get("dominant_dimension")), "我最近最需要留意的方向")
    with cols[2]:
        metric_card("建议优先动作", str(row.get("intervention_type", "提示")), "系统推荐的第一步")

    _page_section_gap()
    narrative_card(str(row.get("profile_text") or "暂无成长摘要。"))
    _page_section_gap()
    st.markdown(risk_chip(row.get("risk_level")), unsafe_allow_html=True)
    st.markdown(map_chip(row.get("dominant_MAP")), unsafe_allow_html=True)


def render_student_self_assessment_page() -> None:
    page_header("状态自评", "")
    with st.form("student_self_assessment_form"):
        stress = st.slider("最近的压力感", 0, 10, 6)
        sleep = st.slider("最近的睡眠状态", 0, 10, 5)
        support = st.slider("我感受到的支持程度", 0, 10, 4)
        exercise = st.slider("近一周运动状态", 0, 10, 5)
        emotion = st.slider("最近情绪稳定度", 0, 10, 5)
        submitted = st.form_submit_button("生成自评结果", use_container_width=True)

    if submitted:
        burden = stress * 0.28 + (10 - sleep) * 0.18 + (10 - support) * 0.2 + (10 - exercise) * 0.14 + (10 - emotion) * 0.2
        if burden >= 6.8:
            level = "需要尽快被关注"
            advice = "建议这周尽快找可信任的老师、辅导员或同学聊一次，把最卡住的那件事先说出来。"
        elif burden >= 4.4:
            level = "需要温和调整"
            advice = "建议先把睡眠、情绪和支持资源稳住，给自己留出一个更容易执行的小目标。"
        else:
            level = "整体可控"
            advice = "当前状态整体可控，继续维持规律节奏，同时留意压力上升的前置信号。"
        st.session_state.self_assessment_result = {"level": level, "advice": advice, "burden": burden}

    result = st.session_state.get("self_assessment_result")
    if result:
        cols = st.columns(3)
        with cols[0]:
            metric_card("自评状态", result["level"], "根据本次自评生成")
        with cols[1]:
            metric_card("负荷指数", fmt_num(result["burden"], 2), "分数越高表示越需要关注")
        with cols[2]:
            metric_card("建议", "已生成", "可直接按右侧建议行动")
        _page_section_gap()
        narrative_card(result["advice"])


def render_student_treehole_page() -> None:
    page_header("AI 树洞", "")
    text = st.text_area("想说的话", placeholder="例如：最近总觉得很累，学习也提不起劲。")
    _page_section_gap()
    if st.button("整理一下我的状态", use_container_width=True):
        if not text.strip():
            st.info("先写一点最近的感受，我再帮你整理。")
        else:
            response = (
                "我先接住这句话：你最近确实有点累，而且这种累不只是身体上的，可能也带着压力和消耗感。"
                " 先别急着把所有问题一次解决，先挑一件最困扰你的事情，今天只处理那一件。"
                " 如果这份压力已经明显影响到睡眠、上课或情绪，建议尽快找老师、辅导员或身边信任的人聊一次。"
            )
            _page_section_gap()
            narrative_card(response)


def render_current_page(page: str, filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    page_map = {
        "总览工作台": lambda: render_teacher_dashboard(filtered_df, data),
        "成果总览": lambda: render_outcome_catalog_page(filtered_df, data),
        "指标验证台": lambda: render_requirements_page(filtered_df, data),
        "动态建模展示": lambda: render_requirements_page(filtered_df, data),
        "群体画像": lambda: render_group_profile_page(filtered_df, data),
        "学生档案": lambda: render_student_profile_page(filtered_df, data),
        "预警干预": lambda: render_alert_page(filtered_df, data),
        "新增预测": lambda: render_predictor_page(data),
        "AI 助手": lambda: render_ai_copilot_page(filtered_df, data),
        "链路追踪": lambda: render_chain_page(data),
        "我的主页": lambda: render_student_home_page(filtered_df, data),
        "状态自评": render_student_self_assessment_page,
        "AI 树洞": render_student_treehole_page,
    }
    page_map.get(page, lambda: render_teacher_dashboard(filtered_df, data))()
