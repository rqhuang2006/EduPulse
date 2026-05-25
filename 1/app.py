from __future__ import annotations

import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(site.getusersitepackages())
site.addsitedir(str(ROOT / ".python_packages"))
site.addsitedir(str(ROOT / "streamlit_runtime"))

import streamlit as st

from src.frontend.components import inject_css, render_app_chrome, render_sidebar
from src.frontend.data import prepare_data
from src.frontend.pages import render_current_page
from src.frontend.state import ensure_state


st.set_page_config(
    page_title="知行镜 - A14 学生行为分析系统",
    page_icon="镜",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
data = prepare_data("pattern_remap_v9")
ensure_state(data["master_df"])
render_app_chrome()
sidebar_context = render_sidebar(data)
render_current_page(sidebar_context["page"], sidebar_context["filtered_df"], data)
