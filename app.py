from __future__ import annotations

import json
import math
import re
import site
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
site.addsitedir(site.getusersitepackages())
site.addsitedir(str(ROOT / "streamlit_runtime"))

import altair as alt
import pandas as pd
import streamlit as st


A14_DIR = ROOT / "outputs" / "a14"
DIMENSION_LABELS = {"life": "生活", "study": "学习", "sport": "运动", "unknown": "未知", "": "未知"}
MAP_LABELS = {"M": "动机", "A": "能力", "P": "提示", "": "未知"}
RISK_COLORS = {"高风险": "#7aa2f7", "中风险": "#4fd1c5", "低风险": "#9ae6b4", "未知": "#718096"}


def repair_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if "ï" in value or any(token in value for token in ["瀛", "鐢", "鍔", "璇", "缁", "闄", "诲", "櫓"]):
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


@st.cache_data
def load_json(path: Path) -> Any:
    if not path.exists():
        return {} if path.suffix == ".json" else []
    return repair_object(json.loads(path.read_text(encoding="utf-8")))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [repair_text(col) for col in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(repair_text)
    return df


@st.cache_data
def prepare_data() -> dict[str, Any]:
    group_profile = load_json(A14_DIR / "group_profile.json")
    demo_case = load_json(A14_DIR / "demo_case_student.json")
    master_df = load_csv(A14_DIR / "fusion_student_master_table.csv")
    pattern_df = load_csv(A14_DIR / "pattern_summary.csv")
    intervention_df = load_csv(A14_DIR / "student_intervention.csv")
    reports = load_json(A14_DIR / "student_full_report_multi_agent.json")

    numeric_cols = ["life_risk", "study_risk", "sport_risk", "total_risk", "M_score", "A_score", "P_score"]
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    if "student_id" in master_df.columns:
        master_df["student_id"] = master_df["student_id"].astype(str)
    master_df["dominant_dimension_display"] = master_df.get("dominant_dimension", pd.Series(["unknown"] * len(master_df))).map(
        lambda x: DIMENSION_LABELS.get(str(x), str(x))
    )
    master_df["dominant_map_display"] = master_df.get("dominant_MAP", pd.Series([""] * len(master_df))).map(
        lambda x: MAP_LABELS.get(str(x), str(x))
    )
    if "profile_text" not in master_df.columns:
        master_df["profile_text"] = ""
    if "pattern_label" not in master_df.columns:
        master_df["pattern_label"] = "未知"
    if "risk_level" not in master_df.columns:
        master_df["risk_level"] = "未知"
    master_df = master_df.sort_values(["total_risk", "student_id"], ascending=[False, True]).reset_index(drop=True)

    for col in ["student_count", "ratio", "avg_total_risk"]:
        if col in pattern_df.columns:
            pattern_df[col] = pd.to_numeric(pattern_df[col], errors="coerce")

    report_map = {}
    if isinstance(reports, list):
        for item in reports:
            student_id = str(item.get("student_id", ""))
            if student_id:
                report_map[student_id] = repair_object(item)

    intervention_map = {}
    if not intervention_df.empty and "student_id" in intervention_df.columns:
        intervention_df["student_id"] = intervention_df["student_id"].astype(str)
        for _, row in intervention_df.iterrows():
            intervention_map[str(row["student_id"])] = row.to_dict()

    return {
        "group_profile": repair_object(group_profile),
        "demo_case": repair_object(demo_case),
        "master_df": master_df,
        "pattern_df": pattern_df,
        "intervention_df": intervention_df,
        "report_map": report_map,
        "intervention_map": intervention_map,
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #07121e;
            --bg-1: #0c1827;
            --bg-2: #112033;
            --panel: rgba(17, 29, 45, 0.82);
            --panel-strong: rgba(15, 24, 38, 0.92);
            --line: rgba(183, 205, 229, 0.10);
            --line-strong: rgba(183, 205, 229, 0.16);
            --text-0: #f1f5f9;
            --text-1: #d7e3f4;
            --text-2: #a8bad3;
            --accent: #00d4aa;
            --accent-soft: #5eead4;
        }
        .stApp {background:radial-gradient(circle at 85% 8%, rgba(0,212,170,.10), transparent 18%),radial-gradient(circle at 10% 0%, rgba(94,234,212,.10), transparent 24%),linear-gradient(180deg,#08111d 0%,#0d1726 46%,#09111b 100%);color:#eef4ff;}
        html, body, [class*="css"]  {
            font-family: "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif;
            letter-spacing: .01em;
        }
        header[data-testid="stHeader"] {background:transparent;height:0;}
        [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu {display:none !important;}
        .block-container {padding-top:1.35rem;padding-bottom:2.4rem;max-width:1240px;}
        section[data-testid="stSidebar"] {background:radial-gradient(circle at top, rgba(0,212,170,.10), transparent 18%),linear-gradient(180deg,#07111d 0%,#0d1829 52%,#0a1322 100%);border-right:1px solid rgba(255,255,255,.06);min-width:300px !important;max-width:300px !important;}
        section[data-testid="stSidebar"] * {color:#d9e8ff !important;}
        section[data-testid="stSidebar"] .block-container {padding-top:1.35rem;padding-left:1.05rem;padding-right:1.05rem;}
        .hero-card,.metric-card,.panel-card,.quick-card{box-shadow:0 16px 36px rgba(0,0,0,.16);backdrop-filter:blur(10px);}
        .hero-card{position:relative;overflow:hidden;background:linear-gradient(180deg, rgba(18,28,43,.92), rgba(12,20,32,.96));border:1px solid rgba(0,212,170,.12);border-radius:24px;padding:1.45rem 1.45rem 1.25rem 1.45rem;margin-bottom:1.1rem;}
        .hero-card::after{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:1px;background:linear-gradient(90deg, rgba(0,212,170,.0), rgba(0,212,170,.45), rgba(0,212,170,.0));}
        .main-title{font-size:2.05rem;font-weight:700;color:var(--text-0);letter-spacing:-.015em;margin-bottom:.35rem;line-height:1.08;text-align:center;}
        .sub-title{color:var(--text-1);margin:0 auto 1rem auto;font-size:.98rem;max-width:760px;line-height:1.8;text-align:center;}
        .metric-card{background:linear-gradient(135deg, #131f31 0%, #16213e 100%);border:1px solid rgba(0,212,170,.14);border-radius:16px;padding:1rem 1.05rem .95rem 1.05rem;min-height:124px;}
        .metric-label{font-size:.79rem;color:var(--text-2);letter-spacing:.05em;text-transform:uppercase;}
        .metric-value{font-size:2rem;font-weight:700;color:#00d4aa;margin-top:.6rem;line-height:1;}
        .metric-help{color:#9bb0c9;font-size:.78rem;margin-top:.45rem;line-height:1.55;}
        .panel-card{background:linear-gradient(180deg, rgba(16,26,40,.90), rgba(12,19,30,.96));border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:1.05rem 1.1rem;height:100%;}
        .section-title{color:var(--text-0);font-size:1.02rem;font-weight:650;margin:.15rem 0 .85rem 0;letter-spacing:.01em;}
        .badge{display:inline-block;padding:.38rem .74rem;border-radius:999px;margin:.14rem .28rem .14rem 0;background:rgba(0,212,170,.08);border:1px solid rgba(0,212,170,.14);color:#ecf8ff;font-size:.8rem;}
        .quick-card{background:linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:.92rem;margin-bottom:.72rem;}
        .narrative-card{background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.06);border-left:3px solid rgba(0,212,170,.48);border-radius:16px;padding:1rem 1rem 1rem 1.05rem;line-height:1.8;color:#eef4ff;}
        .stButton > button{border-radius:14px;border:1px solid var(--line);background:rgba(255,255,255,.03);color:#e6eef8;font-weight:600;min-height:2.6rem;}
        .stButton > button:hover{border-color:rgba(0,212,170,.24);background:rgba(255,255,255,.045);color:#ffffff;box-shadow:none;}
        .stTabs [data-baseweb="tab-list"]{gap:.36rem;background:rgba(255,255,255,.025);padding:.28rem;border-radius:14px;border:1px solid var(--line);}
        .stTabs [data-baseweb="tab"]{border-radius:10px;color:#aebed2;padding:.42rem .82rem;font-size:.92rem;}
        .stTabs [aria-selected="true"]{background:rgba(0,212,170,.12);color:#fff !important;}
        .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"], .stTextInput input, .stNumberInput input{background:rgba(255,255,255,.03);border-radius:14px;border:1px solid var(--line);color:#eef6ff;}
        .stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div {background:transparent;}
        .stMultiSelect [data-baseweb="tag"]{background:rgba(0,212,170,.10) !important;border:1px solid rgba(0,212,170,.16) !important;border-radius:999px !important;padding:.1rem .35rem !important;color:#e8fbff !important;}
        .stMultiSelect [data-baseweb="tag"] span{color:#d9ebff !important;font-weight:500;}
        .stMultiSelect [data-baseweb="tag"] svg{fill:#7ce8d2 !important;}
        .stRadio [role="radiogroup"]{gap:.55rem;}
        .stRadio [data-baseweb="radio"]{background:rgba(255,255,255,.025);border:1px solid var(--line);padding:.5rem .75rem;border-radius:999px;}
        .stRadio [data-baseweb="radio"] *{color:#eef6ff !important;}
        .stRadio [data-baseweb="radio"][aria-checked="true"]{background:rgba(0,212,170,.12);border-color:rgba(0,212,170,.32);}
        .streamlit-expanderHeader, [data-testid="stExpander"] details, [data-testid="stExpander"] summary{background:rgba(16,26,40,.92) !important;border:1px solid rgba(255,255,255,.08) !important;border-radius:14px !important;}
        .streamlit-expanderHeader{padding:.15rem .2rem;}
        .streamlit-expanderContent, [data-testid="stExpander"] details > div{background:rgba(12,20,32,.72) !important;border:1px solid rgba(148,163,184,.08);border-top:none;border-radius:0 0 14px 14px;padding-top:.55rem;}
        .stSlider [data-baseweb="slider"]{padding-top:.45rem;}
        .stSlider [role="slider"]{background:#7dd3fc !important;box-shadow:0 0 0 4px rgba(125,211,252,.15);}
        .stSlider [data-baseweb="thumb-value"]{color:#d9ebff;}
        .stMarkdown hr{border-color:rgba(148,163,184,.08);}
        pre, .stCodeBlock{background:rgba(8,15,26,.84) !important;border:1px solid rgba(148,163,184,.10);border-radius:16px !important;}
        [data-testid="stDataFrame"], div[data-testid="stTable"] {background:transparent !important;border:none !important;}
        [data-testid="stDataFrame"] [data-testid="stHeader"], [data-testid="stDataFrame"] canvas {background:transparent !important;}
        .table-shell{background:linear-gradient(180deg, rgba(16,25,39,.90), rgba(11,18,28,.96));border:1px solid var(--line);border-radius:18px;padding:.4rem;overflow:auto;}
        .table-shell table{width:100%;border-collapse:collapse;font-size:.93rem;color:#dbe7f5;}
        .table-shell thead th{position:sticky;top:0;background:rgba(15,23,36,.98);color:#89a0be;font-weight:600;text-align:left;padding:.85rem .95rem;border-bottom:1px solid rgba(148,163,184,.14);}
        .table-shell tbody td{padding:.82rem .95rem;border-bottom:1px solid rgba(148,163,184,.08);}
        .table-shell tbody tr:hover{background:rgba(125,211,252,.04);}
        .table-caption{font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;color:#afc4da;margin-bottom:.55rem;}
        .page-title{font-size:2rem;font-weight:700;color:#f5fbff;text-align:center;margin:.2rem 0 .35rem 0;}
        .page-subtitle{font-size:.98rem;color:#d4e2f3;text-align:center;max-width:760px;margin:0 auto .9rem auto;line-height:1.7;}
        .brand-shell{display:flex;align-items:center;gap:.85rem;padding:.25rem 0 1rem 0;}
        .brand-logo{position:relative;width:42px;height:42px;border-radius:14px;background:radial-gradient(circle at 30% 30%, rgba(94,234,212,.28), rgba(0,212,170,.10) 45%, rgba(0,212,170,.02) 70%), linear-gradient(135deg, rgba(13,34,52,.96), rgba(17,37,57,.92));border:1px solid rgba(0,212,170,.26);box-shadow:0 8px 22px rgba(0,0,0,.18);}
        .brand-logo::before{content:"";position:absolute;inset:9px;border-radius:10px;border:2px solid rgba(94,234,212,.72);}
        .brand-logo::after{content:"";position:absolute;width:8px;height:8px;border-radius:999px;background:#00d4aa;right:8px;top:8px;box-shadow:0 0 0 4px rgba(0,212,170,.12);}
        .brand-title{font-size:1.08rem;font-weight:800;color:#f5fbff;line-height:1.1;letter-spacing:.01em;}
        .brand-subtitle{font-size:.78rem;color:#b9c9dc;margin-top:.15rem;line-height:1.45;}
        .plotly-notifier{display:none;}
        .risk-high{color:#ffd7d7;font-weight:700;}.risk-mid{color:#ffe7bf;font-weight:700;}.risk-low{color:#dffff4;font-weight:700;}
        .risk-pill{display:inline-flex;align-items:center;gap:.4rem;padding:.38rem .72rem;border-radius:999px;font-weight:700;font-size:.9rem;border:1px solid transparent;}
        .risk-pill-high{background:rgba(255,107,107,.14);border-color:rgba(255,107,107,.28);color:#ffdede;}
        .risk-pill-mid{background:rgba(255,184,77,.14);border-color:rgba(255,184,77,.28);color:#ffe8c5;}
        .risk-pill-low{background:rgba(0,212,170,.14);border-color:rgba(0,212,170,.28);color:#dcfff5;}
        .stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label, .stTextInput label, .stNumberInput label, .stMetric label {color:#f2f7ff !important;font-weight:700 !important;}
        .stSelectbox [data-baseweb="select"] span, .stMultiSelect [data-baseweb="select"] span, .stSelectbox [data-baseweb="select"] div, .stMultiSelect [data-baseweb="select"] div {color:#eef6ff !important;}
        .stSelectbox [data-baseweb="select"] input::placeholder, .stMultiSelect [data-baseweb="select"] input::placeholder {color:#b9c9dc !important;opacity:1 !important;}
        .streamlit-expanderHeader p, .streamlit-expanderHeader span, [data-testid="stExpander"] summary *, [data-testid="stExpander"] label * {color:#f1f7ff !important;font-weight:700 !important;}
        .stCaption, [data-testid="stCaptionContainer"] {color:#c4d5e8 !important;}
        [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {color:#f4f8ff !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def themed_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None, fill="transparent")
        .configure_axis(
            labelColor="#8ea0b8",
            titleColor="#c7d2e1",
            gridColor="rgba(148,163,184,.12)",
            domainColor="rgba(148,163,184,.14)",
            tickColor="rgba(148,163,184,.14)",
        )
        .configure_legend(labelColor="#c7d2e1", titleColor="#c7d2e1", orient="top")
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-help'>{help_text}</div></div>", unsafe_allow_html=True)


def section_title(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    subtitle_html = f"<div class='page-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='page-title'>{title}</div>{subtitle_html}",
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class='brand-shell'>
            <div class='brand-logo'></div>
            <div>
                <div class='brand-title'>知行镜</div>
                <div class='brand-subtitle'>Student Insight Platform<br/>学生行为分析与干预平台</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_row(items: list[str]) -> None:
    html = "".join(f"<span class='badge'>{item}</span>" for item in items if item)
    st.markdown(html or "<span class='badge'>暂无</span>", unsafe_allow_html=True)


def render_table(df: pd.DataFrame, caption: str = "") -> None:
    if df is None or df.empty:
        st.info("暂无数据。")
        return
    html = df.fillna("-").to_html(index=False, border=0, escape=False)
    prefix = f"<div class='table-caption'>{caption}</div>" if caption else ""
    st.markdown(f"{prefix}<div class='table-shell'>{html}</div>", unsafe_allow_html=True)


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


def risk_class(label: str) -> str:
    label = str(label)
    if "高风险" in label:
        return "risk-high"
    if "中风险" in label:
        return "risk-mid"
    return "risk-low"


def risk_pill(label: str) -> str:
    text = str(label or "未知")
    if "高风险" in text:
        cls = "risk-pill risk-pill-high"
    elif "中风险" in text:
        cls = "risk-pill risk-pill-mid"
    else:
        cls = "risk-pill risk-pill-low"
    return f"<span class='{cls}'>{text}</span>"


def ensure_state(master_df: pd.DataFrame) -> None:
    if "selected_student" not in st.session_state:
        st.session_state.selected_student = master_df["student_id"].iloc[0] if not master_df.empty else ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def choose_student(student_id: str) -> None:
    st.session_state.selected_student = student_id


def get_filtered_df(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return master_df
    st.sidebar.markdown("### 全局筛选")
    risks = sorted(master_df["risk_level"].dropna().astype(str).unique().tolist())
    dims = sorted(master_df["dominant_dimension_display"].dropna().astype(str).unique().tolist())
    patterns = sorted(master_df["pattern_label"].dropna().astype(str).unique().tolist())
    maps = sorted(master_df["dominant_map_display"].dropna().astype(str).unique().tolist())
    st.sidebar.caption("不选即默认全量，避免标签堆满侧栏。")
    with st.sidebar.expander("风险等级", expanded=True):
        pick_risks = st.multiselect("风险等级", risks, default=[], placeholder="全部风险等级", label_visibility="collapsed")
    with st.sidebar.expander("主导维度", expanded=False):
        pick_dims = st.multiselect("主导维度", dims, default=[], placeholder="全部主导维度", label_visibility="collapsed")
    with st.sidebar.expander("行为模式", expanded=False):
        pick_patterns = st.multiselect("行为模式", patterns, default=[], placeholder="全部行为模式", label_visibility="collapsed")
    with st.sidebar.expander("主导机制", expanded=False):
        pick_maps = st.multiselect("主导机制", maps, default=[], placeholder="全部主导机制", label_visibility="collapsed")
    floor = st.sidebar.slider("综合风险下限", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    risk_scope = pick_risks or risks
    dim_scope = pick_dims or dims
    pattern_scope = pick_patterns or patterns
    map_scope = pick_maps or maps
    df = master_df.copy()
    df = df[df["risk_level"].isin(risk_scope)]
    df = df[df["dominant_dimension_display"].isin(dim_scope)]
    df = df[df["pattern_label"].isin(pattern_scope)]
    df = df[df["dominant_map_display"].isin(map_scope)]
    df = df[df["total_risk"].fillna(0) >= floor]
    st.sidebar.markdown(
        f"<div class='quick-card' style='padding:.85rem;margin-top:.8rem;'>"
        f"<div style='font-size:.76rem;color:#89a0be;text-transform:uppercase;letter-spacing:.05em;'>当前筛选结果</div>"
        f"<div style='font-size:1.5rem;font-weight:700;color:#f3f8ff;margin-top:.35rem;'>{len(df)}</div>"
        f"<div style='color:#89a0be;margin-top:.2rem;'>名学生进入当前视图</div></div>",
        unsafe_allow_html=True,
    )
    return df.reset_index(drop=True)


def get_student_row(df: pd.DataFrame, student_id: str) -> dict[str, Any]:
    if df.empty or student_id not in df["student_id"].values:
        return {}
    return df[df["student_id"] == student_id].iloc[0].to_dict()


def parse_student_id_from_text(text: str, student_ids: list[str]) -> str | None:
    for student_id in student_ids:
        if student_id and student_id in text:
            return student_id
    match = re.search(r"\b[a-zA-Z0-9]{8,20}\b", text)
    if match and match.group(0) in student_ids:
        return match.group(0)
    return None


def render_story_block(title: str, body: str, tone: str = "default") -> None:
    accent = "#38bdf8" if tone == "warm" else "#4fd1c5"
    st.markdown(
        f"<div style='background:linear-gradient(160deg, rgba(255,255,255,.045), rgba(255,255,255,.02));"
        f"border:1px solid rgba(255,255,255,.07);border-left:4px solid {accent};border-radius:16px;padding:1rem 1rem 1rem 1.05rem;margin-bottom:.8rem;'>"
        f"<div style='font-size:.82rem;color:#9fb3d2;letter-spacing:.04em;text-transform:uppercase;margin-bottom:.45rem;'>{title}</div>"
        f"<div style='color:#e8f1ff;line-height:1.8;'>{body}</div></div>",
        unsafe_allow_html=True,
    )


def build_student_brief(row: dict[str, Any], intervention: dict[str, Any]) -> dict[str, str]:
    risk_level = str(row.get("risk_level", "未知"))
    dimension = str(row.get("dominant_dimension_display", "未知"))
    mechanism = str(row.get("dominant_map_display", "未知"))
    pattern = str(row.get("pattern_label", "未知"))
    features = [row.get("life_shap_top1", ""), row.get("study_shap_top1", ""), row.get("sport_shap_top1", "")]
    features = [item for item in features if item]
    diagnosis = f"该生当前处于{risk_level}状态，主要风险由{dimension}维度驱动，并表现为{pattern}。从机制归因看，{mechanism}是更核心的解释路径。"
    evidence = f"当前最关键的行为线索集中在{'、'.join(features[:3]) if features else '暂无显著特征'}，综合风险分为 {fmt_num(row.get('total_risk'), 3)}。"
    action = intervention.get("intervention_text", "暂无个性化干预建议。")
    return {"diagnosis": diagnosis, "evidence": evidence, "action": action}


def build_agent_steps(prompt: str, student_id: str | None, filtered_df: pd.DataFrame, data: dict[str, Any]) -> tuple[str, str, list[dict[str, str]]]:
    report_map = data["report_map"]
    intervention_map = data["intervention_map"]
    pattern_df = data["pattern_df"]
    agent_name = classify_agent(prompt)
    steps: list[dict[str, str]] = [
        {"agent": "OrchestratorAgent", "title": "任务分解", "content": f"识别到的问题类型为 `{agent_name}`，开始分派到对应分析角色。"}
    ]

    if any(key in prompt for key in ["整体", "全校", "统计", "群体", "模式"]):
        overview = f"当前筛选范围内共有 {len(filtered_df)} 名学生，高风险占比 {fmt_pct((filtered_df['risk_level'] == '高风险').mean())}。"
        mechanism = filtered_df["dominant_map_display"].mode().iat[0] if not filtered_df.empty and not filtered_df["dominant_map_display"].mode().empty else "未知"
        dimension = filtered_df["dominant_dimension_display"].mode().iat[0] if not filtered_df.empty and not filtered_df["dominant_dimension_display"].mode().empty else "未知"
        steps.append({"agent": "RiskAgent", "title": "群体风险扫描", "content": overview + f" 主导维度以 {dimension} 为主。"})
        top = ""
        if not pattern_df.empty:
            row = pattern_df.sort_values("student_count", ascending=False).iloc[0]
            top = f"当前人数最多的模式是 {row['pattern_label']}，共 {int(row['student_count'])} 人，占比 {fmt_pct(row['ratio'])}。"
        steps.append({"agent": "ReportAgent", "title": "群体摘要生成", "content": f"主导机制以 {mechanism} 为主。{top}"})
        final_answer = overview + f"主导机制以 {mechanism} 为主。{top}"
        return agent_name, final_answer, steps

    if student_id and student_id in filtered_df["student_id"].values:
        row = get_student_row(filtered_df, student_id)
        report = report_map.get(student_id, {})
        intervention = intervention_map.get(student_id, {})
        brief = build_student_brief(row, intervention)
        steps.append({"agent": "RiskAgent", "title": "个体风险识别", "content": brief["diagnosis"]})
        steps.append({"agent": "BehaviorAgent", "title": "行为证据抽取", "content": brief["evidence"]})
        steps.append({"agent": "MechanismAgent", "title": "机制归因", "content": f"主导机制为 {row.get('dominant_map_display', '未知')}，模式标签为 {row.get('pattern_label', '未知')}。"})
        if any(key in prompt for key in ["干预", "建议", "方案"]):
            steps.append({"agent": "InterventionAgent", "title": "干预生成", "content": brief["action"]})
            final_answer = brief["action"]
        elif any(key in prompt for key in ["解释", "为什么", "原因", "归因"]):
            final_answer = brief["diagnosis"] + brief["evidence"]
        else:
            report_text = str(report.get("summary") or row.get("profile_text", "暂无该学生的详细报告。"))
            steps.append({"agent": "ReportAgent", "title": "报告汇总", "content": report_text})
            final_answer = report_text
        return agent_name, final_answer, steps

    fallback = "我可以帮你查询学生画像、解释风险原因、总结群体结构，或者生成干预建议。你可以直接输入学生 ID，或者问我“全校整体情况”。"
    steps.append({"agent": "ReportAgent", "title": "回退说明", "content": fallback})
    return "OrchestratorAgent", fallback, steps


def render_overview(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    group_profile = data["group_profile"]
    pattern_df = data["pattern_df"]
    page_header("学智 · A14 学生行为分析系统", "")
    st.markdown(
        "<div class='hero-card'><div style='display:flex;gap:.45rem;flex-wrap:wrap;justify-content:center;'><span class='badge'>风险融合</span><span class='badge'>SHAP / MAP</span><span class='badge'>画像报告</span><span class='badge'>干预闭环</span></div></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("当前样本数", str(len(filtered_df)), "应用筛选后的学生数量")
    with c2:
        metric_card("高风险人数", str(int((filtered_df["risk_level"] == "高风险").sum())), "当前视角")
    with c3:
        metric_card("平均综合风险", fmt_num(filtered_df["total_risk"].mean(), 3), "越高风险越强")
    with c4:
        metric_card("行为模式数", str(filtered_df["pattern_label"].nunique()), "当前筛选下出现的模式")
    with c5:
        metric_card("主导机制", filtered_df["dominant_map_display"].mode().iat[0] if not filtered_df.empty and not filtered_df["dominant_map_display"].mode().empty else "未知", "当前视角最高频机制")

    left, right = st.columns([1.2, 0.8])
    with left:
        section_title("风险层级分布")
        risk_df = filtered_df.groupby("risk_level", as_index=False).agg(count=("student_id", "count"))
        if not risk_df.empty:
            risk_df["ratio"] = risk_df["count"] / max(risk_df["count"].sum(), 1)
            chart = alt.Chart(risk_df).mark_arc(innerRadius=58).encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("risk_level:N", scale=alt.Scale(domain=list(RISK_COLORS.keys()), range=list(RISK_COLORS.values()))),
                tooltip=["risk_level", "count", alt.Tooltip("ratio:Q", format=".1%")],
            ).properties(height=320)
            st.altair_chart(themed_chart(chart), use_container_width=True)

        section_title("主导维度对比")
        dim_df = filtered_df.groupby("dominant_dimension_display", as_index=False).agg(student_count=("student_id", "count"), avg_total_risk=("total_risk", "mean"))
        if not dim_df.empty:
            chart = alt.Chart(dim_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("dominant_dimension_display:N", title="主导维度"),
                y=alt.Y("student_count:Q", title="人数"),
                color=alt.Color("avg_total_risk:Q", scale=alt.Scale(range=["#4fd1c5", "#7aa2f7"]), legend=None),
                tooltip=["dominant_dimension_display", "student_count", alt.Tooltip("avg_total_risk:Q", format=".3f")],
            ).properties(height=320)
            st.altair_chart(themed_chart(chart), use_container_width=True)

    with right:
        section_title("重点关注学生")
        for _, row in filtered_df.head(6).iterrows():
            st.markdown(
                "<div class='quick-card'>"
                f"<div><strong>{row['student_id']}</strong> · <span class='{risk_class(row['risk_level'])}'>{row['risk_level']}</span></div>"
                f"<div style='color:#8fa6c8;margin-top:6px;'>模式: {row['pattern_label']} · 维度: {row['dominant_dimension_display']} · 机制: {row['dominant_map_display']}</div>"
                f"<div style='color:#5e7698;margin-top:6px;'>综合风险 {fmt_num(row['total_risk'], 3)}</div></div>",
                unsafe_allow_html=True,
            )
            if st.button(f"查看 {row['student_id']}", key=f"overview_pick_{row['student_id']}", use_container_width=True):
                choose_student(row["student_id"])
        section_title("标签摘要")
        badge_row([f"{k}: {fmt_pct(v)}" for k, v in group_profile.get("dominant_map_ratio", {}).items()])
        badge_row([f"{k}: {fmt_pct(v)}" for k, v in group_profile.get("dominant_dimension_ratio", {}).items()])

    section_title("行为模式风险比较")
    if not pattern_df.empty:
        view = pattern_df[pattern_df["pattern_label"].isin(filtered_df["pattern_label"].unique())]
        chart = alt.Chart(view).mark_circle(size=220).encode(
            x=alt.X("student_count:Q", title="人数"),
            y=alt.Y("avg_total_risk:Q", title="平均综合风险"),
            size=alt.Size("ratio:Q", scale=alt.Scale(range=[250, 1800]), legend=None),
            color=alt.Color("main_MAP:N", scale=alt.Scale(range=["#4fd1c5", "#60a5fa", "#93c5fd"])),
            tooltip=["pattern_label", "student_count", alt.Tooltip("ratio:Q", format=".1%"), "main_dimension", "main_MAP"],
        ).properties(height=360)
        st.altair_chart(themed_chart(chart), use_container_width=True)


def render_group_profile(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    pattern_df = data["pattern_df"]
    page_header("学生群体画像分析", "")
    section_title("群体画像")
    left_ctrl, right_ctrl = st.columns([0.8, 1.2])
    with left_ctrl:
        selected_pattern = st.selectbox("聚焦模式", ["全部模式"] + sorted(filtered_df["pattern_label"].dropna().astype(str).unique().tolist()))
    with right_ctrl:
        sort_mode = st.radio("排序指标", ["人数", "平均风险"], horizontal=True)
    subset = filtered_df if selected_pattern == "全部模式" else filtered_df[filtered_df["pattern_label"] == selected_pattern]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("模式内人数", str(len(subset)), "当前聚焦群体")
    with c2:
        metric_card("高风险占比", fmt_pct((subset["risk_level"] == "高风险").mean() if not subset.empty else None), "群体内部")
    with c3:
        metric_card("主导机制", subset["dominant_map_display"].mode().iat[0] if not subset.empty and not subset["dominant_map_display"].mode().empty else "未知", "群体内最高频机制")

    summary_left, summary_right = st.columns([1.1, 0.9])
    top_pattern = subset["pattern_label"].mode().iat[0] if not subset.empty and not subset["pattern_label"].mode().empty else "未知"
    top_dimension = subset["dominant_dimension_display"].mode().iat[0] if not subset.empty and not subset["dominant_dimension_display"].mode().empty else "未知"
    with summary_left:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        render_story_block(
            "群体结论",
            f"当前聚焦群体以 {top_pattern} 为主，风险主要集中在 {top_dimension} 维度，"
            f"主导机制更偏向 {subset['dominant_map_display'].mode().iat[0] if not subset.empty and not subset['dominant_map_display'].mode().empty else '未知'}。"
            f"这一页优先回答“这群学生现在最需要被怎么理解”。",
        )
        badge_row(
            [
                f"当前模式: {selected_pattern}",
                f"样本量: {len(subset)}",
                f"高风险占比: {fmt_pct((subset['risk_level'] == '高风险').mean() if not subset.empty else None)}",
                f"主导机制: {subset['dominant_map_display'].mode().iat[0] if not subset.empty and not subset['dominant_map_display'].mode().empty else '未知'}",
            ]
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with summary_right:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        render_story_block("群体特征", f"主导模式为 {top_pattern}，主导维度为 {top_dimension}。")
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        section_title("风险结构")
        compare_df = pd.DataFrame(
            [{"维度": "生活风险", "均值": safe_float(subset["life_risk"].mean()) or 0.0},
             {"维度": "学习风险", "均值": safe_float(subset["study_risk"].mean()) or 0.0},
             {"维度": "运动风险", "均值": safe_float(subset["sport_risk"].mean()) or 0.0}]
        )
        chart = alt.Chart(compare_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("维度:N", title="画像维度"), y=alt.Y("均值:Q", scale=alt.Scale(domain=[0, 1]), title="平均风险"),
            color=alt.Color("维度:N", scale=alt.Scale(range=["#4fd1c5", "#60a5fa", "#93c5fd"]), legend=None),
            tooltip=["维度", alt.Tooltip("均值:Q", format=".3f")],
        ).properties(height=330)
        st.altair_chart(themed_chart(chart), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        section_title("模式统计")
        if not pattern_df.empty:
            sort_col = "student_count" if sort_mode == "人数" else "avg_total_risk"
            view_df = pattern_df.sort_values(sort_col, ascending=False).copy()
            view_df["ratio"] = view_df["ratio"].map(fmt_pct)
            view_df["avg_total_risk"] = view_df["avg_total_risk"].map(lambda x: fmt_num(x, 3))
            render_table(view_df, "")
        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns([0.95, 1.05])
    with lower_left:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        section_title("模式画像卡")
        focus_cards = subset.sort_values("total_risk", ascending=False).head(4)
        for _, row in focus_cards.iterrows():
            st.markdown(
                "<div class='quick-card'>"
                f"<div style='display:flex;justify-content:space-between;gap:.5rem;align-items:flex-start;'><strong>{row['student_id']}</strong>"
                f"<span class='{risk_class(row['risk_level'])}'>{row['risk_level']}</span></div>"
                f"<div style='color:#9fb3d2;margin-top:.45rem;line-height:1.7;'>"
                f"{row['pattern_label']} · {row['dominant_dimension_display']}主导 · {row['dominant_map_display']}机制</div>"
                f"<div style='color:#6f88a6;margin-top:.45rem;'>综合风险 {fmt_num(row['total_risk'], 3)}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with lower_right:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        section_title("群体样本")
        cols = ["student_id", "risk_level", "dominant_dimension_display", "pattern_label", "dominant_map_display", "total_risk"]
        sample = subset[cols].head(15).copy()
        sample["total_risk"] = sample["total_risk"].map(lambda x: fmt_num(x, 3))
        render_table(sample, "")
        st.markdown("</div>", unsafe_allow_html=True)


def render_student_detail(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    report_map = data["report_map"]
    intervention_map = data["intervention_map"]
    if filtered_df.empty:
        st.warning("当前筛选下没有学生数据。")
        return
    page_header("学生个体详情分析", "")
    ids = filtered_df["student_id"].astype(str).tolist()
    if st.session_state.selected_student not in ids:
        st.session_state.selected_student = ids[0]

    bar1, bar2, bar3 = st.columns([1.4, 0.8, 0.8])
    with bar1:
        selected = st.selectbox("学生检索", ids, index=ids.index(st.session_state.selected_student))
        st.session_state.selected_student = selected
    idx = ids.index(st.session_state.selected_student)
    with bar2:
        if st.button("上一位", use_container_width=True, disabled=idx == 0):
            choose_student(ids[idx - 1])
    with bar3:
        if st.button("下一位", use_container_width=True, disabled=idx == len(ids) - 1):
            choose_student(ids[idx + 1])

    row = get_student_row(filtered_df, st.session_state.selected_student)
    report = report_map.get(st.session_state.selected_student, {})
    intervention = intervention_map.get(st.session_state.selected_student, {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-label'>风险状态</div><div style='margin-top:.55rem;'>{risk_pill(str(row.get('risk_level', '未知')))}</div>", unsafe_allow_html=True)
    with c2:
        metric_card("综合风险", fmt_num(row.get("total_risk"), 3))
    with c3:
        metric_card("主导维度", str(row.get("dominant_dimension_display", "未知")))
    with c4:
        metric_card("主导机制", str(row.get("dominant_map_display", "未知")))

    brief = build_student_brief(row, intervention)
    st.markdown(f"<div class='narrative-card'>{brief['diagnosis']}</div>", unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["报告摘要", "证据与解释", "干预方案", "Agent 报告"])
    with tab1:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            render_story_block("结论先行", brief["diagnosis"])
            render_story_block("关键证据", brief["evidence"])
            badge_row([f"模式: {row.get('pattern_label', '未知')}", f"维度: {row.get('dominant_dimension_display', '未知')}", f"机制: {row.get('dominant_map_display', '未知')}", f"优先级: {intervention.get('priority', '-')}"])
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            chart_df = pd.DataFrame([{"维度": "生活风险", "数值": safe_float(row.get("life_risk")) or 0.0}, {"维度": "学习风险", "数值": safe_float(row.get("study_risk")) or 0.0}, {"维度": "运动风险", "数值": safe_float(row.get("sport_risk")) or 0.0}])
            chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("维度:N", title="风险维度"), y=alt.Y("数值:Q", scale=alt.Scale(domain=[0, 1]), title="风险分数"),
                color=alt.Color("维度:N", scale=alt.Scale(range=["#4fd1c5", "#60a5fa", "#93c5fd"]), legend=None),
                tooltip=["维度", alt.Tooltip("数值:Q", format=".3f")],
            ).properties(height=300)
            st.altair_chart(themed_chart(chart), use_container_width=True)
    with tab2:
        left, right = st.columns(2)
        with left:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            render_story_block("行为特征", "以下特征来自 SHAP 解释结果，是当前学生最值得关注的行为线索。", "warm")
            badge_row([row.get("life_shap_top1", ""), row.get("life_shap_top2", ""), row.get("life_shap_top3", "")])
            badge_row([row.get("study_shap_top1", ""), row.get("study_shap_top2", ""), row.get("study_shap_top3", "")])
            badge_row([row.get("sport_shap_top1", ""), row.get("sport_shap_top2", ""), row.get("sport_shap_top3", "")])
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            render_story_block("机制解释", f"从 MAP 结果看，该生更偏向于 {row.get('dominant_map_display', '未知')} 机制，需要围绕这一主导机制组织后续支持。")
            map_df = pd.DataFrame([{"机制": "动机", "分值": safe_float(row.get("M_score")) or 0.0}, {"机制": "能力", "分值": safe_float(row.get("A_score")) or 0.0}, {"机制": "提示", "分值": safe_float(row.get("P_score")) or 0.0}])
            chart = alt.Chart(map_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("机制:N", title="MAP 机制"), y=alt.Y("分值:Q", title="机制分值"),
                color=alt.Color("机制:N", scale=alt.Scale(range=["#4fd1c5", "#60a5fa", "#93c5fd"]), legend=None),
                tooltip=["机制", alt.Tooltip("分值:Q", format=".3f")],
            ).properties(height=300)
            st.altair_chart(themed_chart(chart), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with tab3:
        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            render_story_block("干预目标", f"当前建议优先围绕 {row.get('dominant_dimension_display', '未知')} 维度展开，优先级为 {intervention.get('priority', '-') }。")
            render_story_block("执行建议", intervention.get('intervention_text', '暂无个性化干预建议。'), "warm")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            badge_row([f"干预类型: {intervention.get('intervention_type', '未知')}", f"主导机制: {row.get('dominant_map_display', '未知')}", f"风险层级: {row.get('risk_level', '未知')}"])
            similar = filtered_df[(filtered_df["pattern_label"] == row.get("pattern_label")) & (filtered_df["student_id"] != row.get("student_id"))].head(4)
            for _, item in similar.iterrows():
                st.markdown(
                    "<div class='quick-card'>"
                    f"<div><strong>{item['student_id']}</strong></div>"
                    f"<div style='color:#8fa6c8;margin-top:6px;'>同属 {item['pattern_label']} · 综合风险 {fmt_num(item['total_risk'], 3)}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    with tab4:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        if report:
            if report.get("summary"):
                render_story_block("Agent 汇总摘要", str(report["summary"]))
            if report.get("narrative"):
                render_story_block("叙事化画像", str(report["narrative"]), "warm")
            st.code(json.dumps(report, ensure_ascii=False, indent=2), language="json")
        else:
            st.info("当前学生暂无多 Agent 报告。")
        st.markdown("</div>", unsafe_allow_html=True)


def render_risk_alert(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    intervention_map = data["intervention_map"]
    page_header("风险预警中心", "")
    threshold = st.slider("风险阈值", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    alerts = filtered_df[filtered_df["total_risk"].fillna(0) >= threshold].copy().sort_values("total_risk", ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("预警人数", str(len(alerts)), "高于当前阈值")
    with c2:
        metric_card("极高风险", str(int((alerts["total_risk"].fillna(0) >= 0.8).sum())), "综合风险 ≥ 0.8")
    with c3:
        metric_card("生活主导", str(int((alerts["dominant_dimension_display"] == "生活").sum())), "预警群体内部")
    with c4:
        metric_card("能力主导", str(int((alerts["dominant_map_display"] == "能力").sum())), "预警群体内部")

    tab1, tab2, tab3 = st.tabs(["预警列表", "模式切片", "干预清单"])
    with tab1:
        cols = ["student_id", "risk_level", "dominant_dimension_display", "pattern_label", "dominant_map_display", "total_risk"]
        preview = alerts[cols].head(80).copy()
        preview["total_risk"] = preview["total_risk"].map(lambda x: fmt_num(x, 3))
        render_table(preview, "预警学生")
    with tab2:
        if not alerts.empty:
            chart_df = alerts.groupby(["pattern_label", "dominant_map_display"], as_index=False).agg(student_count=("student_id", "count"), avg_total_risk=("total_risk", "mean"))
            chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
                x=alt.X("avg_total_risk:Q", title="平均综合风险"), y=alt.Y("pattern_label:N", sort="-x", title="行为模式"),
                color=alt.Color("dominant_map_display:N", scale=alt.Scale(range=["#4fd1c5", "#60a5fa", "#93c5fd"])),
                tooltip=["pattern_label", "dominant_map_display", "student_count", alt.Tooltip("avg_total_risk:Q", format=".3f")],
            ).properties(height=340)
            st.altair_chart(themed_chart(chart), use_container_width=True)
    with tab3:
        for student_id in alerts["student_id"].tolist()[:20]:
            row = alerts[alerts["student_id"] == student_id].iloc[0]
            text = intervention_map.get(student_id, {}).get("intervention_text", "暂无干预建议")
            st.markdown("<div class='quick-card'>" + f"<div><strong>{student_id}</strong> · <span class='{risk_class(row['risk_level'])}'>{row['risk_level']}</span></div><div style='color:#8fa6c8;margin-top:6px;'>模式: {row['pattern_label']} · 机制: {row['dominant_map_display']}</div><div style='margin-top:8px;color:#dce8ff;'>{text}</div></div>", unsafe_allow_html=True)


def classify_agent(prompt: str) -> str:
    if any(key in prompt for key in ["解释", "为什么", "原因", "归因"]):
        return "BehaviorAgent + MechanismAgent"
    if any(key in prompt for key in ["干预", "建议", "方案"]):
        return "InterventionAgent"
    if any(key in prompt for key in ["整体", "全校", "统计", "群体", "模式"]):
        return "RiskAgent + ReportAgent"
    return "RiskAgent + ReportAgent"


def local_agent_reply(prompt: str, student_id: str | None, filtered_df: pd.DataFrame, data: dict[str, Any]) -> tuple[str, str]:
    agent_name, answer, _ = build_agent_steps(prompt, student_id, filtered_df, data)
    return agent_name, answer


def render_agent_chat(filtered_df: pd.DataFrame, data: dict[str, Any]) -> None:
    ids = filtered_df["student_id"].astype(str).tolist()
    page_header("智能对话助手", "")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        quick_cols = st.columns(2)
        prompts = ["全校整体风险情况怎么样？", "哪个群体最值得优先关注？", "请解释这个学生为什么高风险", "给我一条具体干预建议"]
        for i, text in enumerate(prompts):
            if quick_cols[i % 2].button(text, key=f"quick_agent_{i}", use_container_width=True):
                st.session_state.pending_prompt = text
        selected = st.selectbox("对话默认学生", ["不指定学生"] + ids[:300])
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
        prompt = st.chat_input("输入问题，例如：请解释 pjwtqxbj965 为什么被判为高风险")
        if st.session_state.get("pending_prompt"):
            prompt = st.session_state.pop("pending_prompt")
        if prompt:
            student_id = None if selected == "不指定学生" else selected
            student_id = parse_student_id_from_text(prompt, ids) or student_id
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            agent_name, answer, steps = build_agent_steps(prompt, student_id, filtered_df, data)
            with st.chat_message("assistant"):
                st.markdown(f"**当前响应 Agent**: `{agent_name}`")
                for step in steps:
                    st.markdown(
                        "<div class='quick-card'>"
                        f"<div style='font-size:.8rem;color:#8cf4d8;letter-spacing:.05em;text-transform:uppercase;'>{step['agent']}</div>"
                        f"<div style='font-weight:700;color:#f4f8ff;margin-top:4px;'>{step['title']}</div>"
                        f"<div style='color:#cfe0fb;line-height:1.72;margin-top:6px;'>{step['content']}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                render_story_block("最终回答", answer, "warm")
            st.session_state.chat_history.append({"role": "assistant", "content": f"[{agent_name}] {answer}"})
            st.session_state.last_agent_name = agent_name
            st.session_state.last_agent_answer = answer
            st.session_state.last_agent_student = student_id or ""
            st.session_state.last_agent_steps = steps
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown(f"**上次调用 Agent**: `{st.session_state.get('last_agent_name', '无')}`")
        if st.session_state.get("last_agent_student"):
            st.markdown(f"**当前绑定学生**: {st.session_state['last_agent_student']}")
        badge_row(["RiskAgent", "BehaviorAgent", "MechanismAgent", "InterventionAgent", "ReportAgent"])
        st.markdown("<div class='narrative-card'>这个页面不只是聊天框，而是多 Agent 协作结果层的展示入口。界面上直接显示当前响应 Agent，答辩时可以很自然地解释“多 Agent 体现在哪”。</div>", unsafe_allow_html=True)
        if st.session_state.get("last_agent_steps"):
            st.markdown("**最近一次调度轨迹**")
            for step in st.session_state["last_agent_steps"]:
                st.markdown(f"- `{step['agent']}`: {step['title']}")
        if st.session_state.get("last_agent_answer"):
            st.markdown(f"<div class='narrative-card'>{st.session_state['last_agent_answer']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_demo_chain(data: dict[str, Any]) -> None:
    page_header("Demo 链路展示", "")
    chain = data["demo_case"].get("chain", {})
    steps = [("原始风险", chain.get("submodel_risk", {})), ("融合结果", chain.get("fusion", {})), ("SHAP 行为", chain.get("shap", {})), ("MAP 机制", chain.get("map", {})), ("模式归类", {"pattern": chain.get("pattern", "-")}), ("干预输出", {"intervention_text": chain.get("intervention_text", "-")})]
    cols = st.columns(len(steps))
    for col, (title, payload) in zip(cols, steps):
        with col:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='narrative-card'>{chain.get('profile_text', '暂无单学生案例摘要。')}</div>", unsafe_allow_html=True)


st.set_page_config(page_title="学智 - A14 学生行为分析系统", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
inject_css()
data = prepare_data()
master_df = data["master_df"]
ensure_state(master_df)
render_sidebar_brand()
filtered_df = get_filtered_df(master_df)
st.sidebar.markdown("---")
page = st.sidebar.radio("功能导航", ["📊 总览仪表盘", "👥 群体画像", "📋 学生详情", "⚠️ 风险预警", "💬 智能对话", "🧪 Demo 链路"])
st.sidebar.markdown("---")
st.sidebar.markdown("🔒 数据安全: 学生 ID 已脱敏处理\n\n📊 数据来源: 多源异构校园行为数据")

if page == "📊 总览仪表盘":
    render_overview(filtered_df, data)
elif page == "👥 群体画像":
    render_group_profile(filtered_df, data)
elif page == "📋 学生详情":
    render_student_detail(filtered_df, data)
elif page == "⚠️ 风险预警":
    render_risk_alert(filtered_df, data)
elif page == "💬 智能对话":
    render_agent_chat(filtered_df, data)
else:
    render_demo_chain(data)
