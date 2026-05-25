from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from .data import data_source_to_cn, decision_to_cn, dimension_to_cn, fmt_num, fmt_pct, map_to_cn, prettify_dataframe_for_display
from .state import apply_filters, get_pages


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;800&family=Source+Serif+4:wght@600;700&display=swap');
        :root {
            --bg: #f6efe5;
            --panel: rgba(255, 250, 244, 0.95);
            --ink: #2e241d;
            --muted: #7f6a5b;
            --line: rgba(112, 84, 64, 0.15);
            --accent: #c4673c;
            --accent-strong: #e97f4e;
            --green: #4f7d57;
            --amber: #c18b2b;
            --red: #b84b42;
            --navy: #385a73;
            --sidebar-bg: #32251c;
            --sidebar-panel: rgba(255,255,255,0.04);
            --sidebar-width: 21rem;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.9), transparent 30%),
                radial-gradient(circle at top right, rgba(224, 197, 173, 0.35), transparent 25%),
                linear-gradient(180deg, #f8f1e8 0%, #f4eadf 55%, #efe3d6 100%);
            color: var(--ink);
            font-family: "Noto Sans SC", "Segoe UI", sans-serif;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
            position: fixed !important;
            inset: 0 0 auto 0 !important;
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
            z-index: 1200 !important;
        }
        [data-testid="stHeader"] > div {
            overflow: visible !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        [data-testid="stToolbarActions"],
        .stAppDeployButton {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        [data-testid="stDecoration"],
        #MainMenu {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2f231b 0%, #38281d 100%);
            color: #f7eee5;
            border-right: 1px solid rgba(255,255,255,0.06);
            min-width: var(--sidebar-width) !important;
            max-width: var(--sidebar-width) !important;
            width: var(--sidebar-width) !important;
            transform: none !important;
            margin-left: 0 !important;
            visibility: visible !important;
            pointer-events: auto !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: var(--sidebar-width) !important;
            max-width: var(--sidebar-width) !important;
            width: var(--sidebar-width) !important;
            transform: none !important;
            margin-left: 0 !important;
            visibility: visible !important;
            pointer-events: auto !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.18rem !important;
            padding-bottom: 1.25rem !important;
        }
        [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 22px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            margin-bottom: 0.9rem;
        }
        [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.15rem 0.2rem;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] {
            color: #f7eee5 !important;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p {
            color: rgba(247, 238, 229, 0.78) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] > div {
            gap: 0.35rem !important;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] label {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 0.42rem;
        }
        [data-testid="stSidebar"] [role="radio"] {
            padding: 0.72rem 0.9rem !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            background: rgba(255,255,255,0.02) !important;
            transition: all 0.15s ease;
        }
        [data-testid="stSidebar"] [role="radio"]:hover {
            background: rgba(255,255,255,0.05) !important;
            border-color: rgba(255,255,255,0.12) !important;
        }
        [data-testid="stSidebar"] details {
            background: rgba(255,255,255,0.02) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 18px !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"] details summary {
            background: linear-gradient(180deg, rgba(86, 60, 43, 0.96), rgba(67, 46, 32, 0.96)) !important;
            color: #f7eee5 !important;
            border-radius: 18px !important;
            padding: 0.78rem 0.95rem !important;
        }
        [data-testid="stSidebar"] details[open] summary {
            border-bottom: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 18px 18px 0 0 !important;
        }
        [data-testid="stSidebar"] details summary:hover {
            background: linear-gradient(180deg, rgba(98, 68, 49, 0.98), rgba(74, 51, 36, 0.98)) !important;
        }
        [data-testid="stSidebar"] details summary span,
        [data-testid="stSidebar"] details summary p,
        [data-testid="stSidebar"] details summary svg {
            color: #f7eee5 !important;
            fill: #f7eee5 !important;
        }
        [data-testid="stSidebar"] details > div {
            background: rgba(58, 40, 28, 0.54) !important;
            padding-top: 0.18rem !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(255, 249, 242, 0.96) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 16px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.25) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #5c4537 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #5c4537 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: rgba(88, 61, 44, 0.12) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(112, 84, 64, 0.18) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] span,
        [data-testid="stSidebar"] [data-baseweb="tag"] svg {
            color: #5c4537 !important;
            fill: #5c4537 !important;
        }
        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
            background: linear-gradient(135deg, rgba(233,127,78,0.16), rgba(196,103,60,0.1)) !important;
            border-color: rgba(233,127,78,0.38) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] [role="radio"] > div:first-child {
            display: none !important;
        }
        [data-testid="stSidebar"] [role="radio"] p {
            color: #f7eee5 !important;
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            margin: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            color: #f7eee5 !important;
            font-weight: 700 !important;
        }
        .block-container {
            padding-top: 1.46rem;
            padding-bottom: 2rem;
            padding-left: 3.2rem;
            padding-right: 2.2rem;
            max-width: 1440px;
        }
        [data-testid="stVerticalBlock"] {
            gap: 1rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            margin-bottom: 0.95rem;
        }
        [data-testid="column"] > div {
            padding-bottom: 0.45rem;
        }
        .app-chrome {
            display: none;
        }
        .st-key-chrome_role_switch {
            position: fixed;
            top: 0.82rem;
            right: 1.15rem;
            width: 154px;
            z-index: 1000;
        }
        .st-key-chrome_role_switch [data-baseweb="select"] > div {
            background: rgba(255, 250, 244, 0.88);
            border: 1px solid rgba(112, 84, 64, 0.12);
            border-radius: 999px;
            box-shadow: 0 8px 22px rgba(80, 55, 34, 0.08);
            min-height: 38px;
        }
        .st-key-chrome_role_switch [data-baseweb="select"] input {
            color: #3b2d24 !important;
            font-size: 0.86rem !important;
            font-weight: 700 !important;
        }
        .st-key-chrome_role_switch svg {
            color: #7a614f !important;
        }
        .sidebar-brand {
            padding: 0 0 0.02rem 0;
            margin-top: -0.86rem;
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.95rem;
        }
        .product-logo {
            width: 58px;
            height: 58px;
            border-radius: 20px;
            position: relative;
            background:
                radial-gradient(circle at 35% 30%, rgba(255,255,255,0.22), transparent 32%),
                linear-gradient(145deg, #4b3427, #8f6544);
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.18),
                0 10px 24px rgba(0,0,0,0.22);
            overflow: hidden;
            flex: 0 0 auto;
        }
        .product-logo::before {
            content: "";
            position: absolute;
            width: 28px;
            height: 28px;
            border: 3px solid rgba(255,248,242,0.92);
            border-radius: 50%;
            left: 12px;
            top: 12px;
        }
        .product-logo::after {
            content: "";
            position: absolute;
            width: 20px;
            height: 4px;
            background: rgba(255,248,242,0.92);
            border-radius: 999px;
            right: 8px;
            bottom: 13px;
            transform: rotate(45deg);
            transform-origin: center;
        }
        .brand-eyebrow {
            color: rgba(247, 238, 229, 0.62);
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }
        .brand-title {
            color: #fff7f0;
            font-family: "Source Serif 4", serif;
            font-size: 1.38rem;
            font-weight: 700;
            line-height: 1;
            margin: 0;
        }
        .brand-subtitle {
            margin-top: 0.38rem;
            color: rgba(247, 238, 229, 0.78);
            font-size: 0.88rem;
            line-height: 1.6;
        }
        .sidebar-section-label {
            color: rgba(247, 238, 229, 0.72);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0 0 0.55rem 0.1rem;
        }
        .sidebar-section-copy {
            color: rgba(247, 238, 229, 0.72);
            font-size: 0.86rem;
            line-height: 1.55;
            margin: 0.05rem 0 0.8rem 0.1rem;
        }
        .hero-shell {
            background: linear-gradient(135deg, rgba(58, 42, 30, 0.96), rgba(102, 71, 45, 0.94));
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 1.02rem 1.2rem;
            color: #fff7f0;
            box-shadow: 0 16px 46px rgba(78, 49, 28, 0.13);
            position: relative;
            overflow: hidden;
            margin-top: 0.16rem;
            margin-bottom: 1.18rem;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -42px -62px auto;
            width: 168px;
            height: 168px;
            background: radial-gradient(circle, rgba(255, 213, 158, 0.28), transparent 72%);
        }
        .hero-shell.compact {
            padding: 1rem 1.2rem;
            border-radius: 24px;
            margin-top: 0.18rem;
            margin-bottom: 1.18rem;
        }
        .hero-shell.compact::after {
            width: 160px;
            height: 160px;
            inset: auto -40px -60px auto;
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            font-size: 0.76rem;
            letter-spacing: 0.04em;
        }
        .hero-title {
            margin-top: 0.4rem;
            font-family: "Source Serif 4", serif;
            font-size: 1.42rem;
            font-weight: 700;
            line-height: 1.12;
        }
        .hero-shell.compact .hero-title {
            margin-top: 0.42rem;
            font-size: 1.38rem;
        }
        .hero-copy {
            margin-top: 0.34rem;
            color: rgba(255,247,240,0.86);
            line-height: 1.62;
            font-size: 0.95rem;
        }
        .hero-shell.compact .hero-copy {
            margin-top: 0.35rem;
            line-height: 1.68;
            font-size: 0.95rem;
        }
        .metric-tile {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 30px rgba(80, 55, 34, 0.08);
            min-height: 122px;
        }
        .assistant-gap {
            height: 0.9rem;
        }
        .assistant-panel-head {
            background:
                radial-gradient(circle at 84% 18%, rgba(255, 204, 145, 0.18), transparent 16%),
                radial-gradient(circle at 16% 18%, rgba(255, 255, 255, 0.08), transparent 22%),
                radial-gradient(circle at 72% 78%, rgba(120, 165, 196, 0.12), transparent 24%),
                linear-gradient(118deg, rgba(29, 38, 46, 0.99) 0%, rgba(40, 58, 68, 0.98) 34%, rgba(54, 74, 86, 0.97) 70%, rgba(82, 95, 107, 0.94) 100%);
            border: 1px solid rgba(143, 169, 186, 0.2);
            border-radius: 28px;
            padding: calc(1.32rem + 7px) 1.28rem calc(1.18rem + 7px) 1.28rem;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.14),
                inset 0 -18px 30px rgba(20, 28, 35, 0.22),
                0 20px 42px rgba(31, 42, 49, 0.2);
            margin-bottom: 0.92rem;
            position: relative;
            overflow: hidden;
        }
        .assistant-panel-head::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            background: linear-gradient(180deg, rgba(255, 210, 154, 0.96) 0%, rgba(118, 173, 208, 0.86) 100%);
        }
        .assistant-panel-head::after {
            content: "";
            position: absolute;
            inset: auto -34px -36px auto;
            width: 168px;
            height: 168px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(122, 174, 206, 0.16), transparent 72%);
        }
        .assistant-panel-head::selection {
            background: rgba(255, 210, 154, 0.25);
        }
        .assistant-panel-head .assistant-strip-title::after {
            content: "";
            display: block;
            width: 88px;
            height: 3px;
            margin-top: 0.55rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255, 214, 161, 0.96), rgba(128, 184, 218, 0.78));
        }
        .assistant-panel-head .assistant-strip-title,
        .assistant-panel-head .assistant-strip-copy,
        .assistant-panel-head .assistant-kicker,
        .assistant-panel-head .assistant-meta-row {
            position: relative;
            z-index: 1;
        }
        .assistant-panel-ornament {
            position: absolute;
            right: 2.4rem;
            top: 50%;
            width: 246px;
            height: 170px;
            transform: translateY(-50%);
            pointer-events: none;
            z-index: 0;
            opacity: 0.8;
            -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.52) 12%, #000 34%);
            mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.52) 12%, #000 34%);
        }
        .assistant-orbit,
        .assistant-node,
        .assistant-glow,
        .assistant-core-ring {
            position: absolute;
            display: block;
        }
        .assistant-core-ring {
            width: 122px;
            height: 122px;
            right: 36px;
            top: 23px;
            border-radius: 50%;
            border: 1px solid rgba(168, 206, 232, 0.24);
            box-shadow:
                inset 0 0 0 1px rgba(255, 224, 190, 0.08),
                inset 0 -18px 26px rgba(31, 50, 64, 0.16),
                0 0 0 10px rgba(255,255,255,0.015);
            background:
                radial-gradient(circle at 34% 30%, rgba(255, 245, 231, 0.16), transparent 22%),
                radial-gradient(circle at 68% 70%, rgba(110, 170, 208, 0.18), transparent 34%),
                radial-gradient(circle at 52% 52%, rgba(255,255,255,0.04), transparent 58%),
                linear-gradient(135deg, rgba(255, 202, 145, 0.16) 0%, rgba(255, 222, 191, 0.06) 32%, rgba(128, 184, 218, 0.08) 72%, rgba(88, 148, 188, 0.18) 100%);
        }
        .assistant-core-ring.ring-inner {
            width: 82px;
            height: 82px;
            right: 56px;
            top: 43px;
            border-color: rgba(255, 219, 178, 0.22);
            box-shadow:
                inset 0 0 20px rgba(98, 150, 186, 0.1),
                inset 0 10px 18px rgba(255, 214, 169, 0.08),
                0 0 0 1px rgba(255,255,255,0.02);
            background:
                radial-gradient(circle at 35% 34%, rgba(255, 240, 220, 0.18), transparent 22%),
                radial-gradient(circle at 64% 68%, rgba(120, 182, 220, 0.18), transparent 34%),
                linear-gradient(140deg, rgba(255, 196, 136, 0.14) 0%, rgba(255,255,255,0.02) 42%, rgba(119, 177, 214, 0.14) 100%);
        }
        .assistant-core-ring.ring-highlight {
            width: 96px;
            height: 96px;
            right: 49px;
            top: 36px;
            border-color: rgba(196, 225, 245, 0.04) rgba(196, 225, 245, 0.18) rgba(255, 215, 171, 0.08) rgba(255, 215, 171, 0.04);
            border-style: solid;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.015);
            transform: rotate(-10deg);
        }
        .assistant-orbit {
            border-radius: 999px;
            border: 1px solid rgba(173, 205, 227, 0.16);
            background: transparent;
        }
        .assistant-orbit.orbit-a {
            inset: 10px 2px 18px 88px;
            border-color: rgba(154, 196, 224, 0.14);
        }
        .assistant-orbit.orbit-b {
            inset: 26px 20px 34px 116px;
            border-color: rgba(255, 212, 163, 0.1);
            transform: rotate(-8deg);
        }
        .assistant-node {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            box-shadow: 0 0 0 5px rgba(255,255,255,0.02);
        }
        .assistant-node.node-a {
            right: 42px;
            top: 42px;
            background: rgba(255, 212, 163, 0.82);
        }
        .assistant-node.node-b {
            right: 126px;
            bottom: 40px;
            background: rgba(151, 201, 233, 0.74);
        }
        .assistant-glow {
            border-radius: 50%;
            filter: blur(3px);
        }
        .assistant-glow.glow-a {
            width: 138px;
            height: 138px;
            right: 26px;
            top: 12px;
            background: radial-gradient(circle, rgba(111, 175, 217, 0.12), transparent 70%);
        }
        .assistant-glow.glow-b {
            width: 110px;
            height: 110px;
            right: 88px;
            bottom: 16px;
            background: radial-gradient(circle, rgba(255, 200, 142, 0.1), transparent 72%);
        }
        .assistant-panel-head .assistant-meta-row::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: -0.55rem;
            height: 1px;
            background: linear-gradient(90deg, rgba(255, 221, 184, 0.18), rgba(128, 184, 218, 0.18), rgba(255,255,255,0));
        }
        .assistant-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.34rem 0.8rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(255, 238, 216, 0.12), rgba(125, 176, 207, 0.14));
            color: #f3e5d3;
            font-size: 0.76rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.88rem;
            border: 1px solid rgba(198, 218, 232, 0.14);
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.1),
                0 8px 18px rgba(23, 31, 38, 0.16);
        }
        .assistant-strip-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #f4f7fa;
            margin-bottom: 0.36rem;
            letter-spacing: 0.01em;
        }
        .assistant-strip-copy {
            color: rgba(231, 237, 242, 0.82);
            line-height: 1.72;
            font-size: 0.96rem;
            margin-bottom: 0;
            max-width: 46rem;
        }
        .assistant-meta-row {
            display: flex;
            align-items: center;
            gap: 0.56rem;
            flex-wrap: wrap;
            margin-top: 0.95rem;
        }
        .assistant-meta-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.34rem 0.72rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(255, 239, 219, 0.08), rgba(125, 176, 207, 0.12));
            border: 1px solid rgba(194, 216, 231, 0.1);
            color: #e8d8c3;
            font-size: 0.8rem;
            font-weight: 700;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        @media (max-width: 1100px) {
            .assistant-panel-ornament {
                opacity: 0.34;
                right: 0.8rem;
                transform: translateY(-50%) scale(0.84);
                transform-origin: right center;
            }
        }
        @media (max-width: 860px) {
            .assistant-panel-ornament {
                display: none;
            }
        }
        .assistant-toolbar-head {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 0.62rem;
            margin: 0.1rem 0 0.82rem 0;
            flex-wrap: wrap;
        }
        .assistant-subhead-shell {
            height: 52px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }
        .assistant-subhead-shell,
        .assistant-subhead-shell > div,
        .assistant-subhead-shell > div > div {
            height: 52px !important;
            display: flex !important;
            align-items: center !important;
        }
        .assistant-subhead-shell > div {
            margin: 0 !important;
            width: 100%;
        }
        .assistant-subhead-shell p {
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            height: 52px !important;
        }
        .assistant-subhead {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            height: 52px;
            min-height: 52px;
            padding: 0 0.92rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(206, 153, 110, 0.18), rgba(242, 219, 195, 0.38));
            color: #935631;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin: 0;
            border: 1px solid rgba(189, 135, 96, 0.18);
            white-space: nowrap;
            line-height: 1;
            transform: translateY(-7px);
        }
        .assistant-toolbar-copy {
            color: #8b6c58;
            font-size: 0.86rem;
            line-height: 1;
            font-weight: 700;
        }
        .assistant-answer-label {
            display: inline-flex;
            align-items: center;
            margin: 1rem 0 0.72rem 0;
            padding: 0.32rem 0.74rem;
            border-radius: 999px;
            background: rgba(110, 76, 52, 0.08);
            color: #6d4932;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .st-key-dashboard_assistant_student,
        .st-key-dashboard_assistant_prompt,
        .st-key-dashboard_assistant_button {
            height: 56px;
        }
        .st-key-dashboard_assistant_student > div,
        .st-key-dashboard_assistant_prompt > div,
        .st-key-dashboard_assistant_button > div {
            margin: 0 !important;
            width: 100%;
            height: 56px;
        }
        .st-key-dashboard_assistant_button button {
            background: linear-gradient(135deg, #c8693c, #9e5330) !important;
            color: #fff8f2 !important;
            border: none !important;
            border-radius: 16px !important;
            min-height: 56px !important;
            height: 56px !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            padding: 0 1rem !important;
            box-shadow: 0 14px 26px rgba(157, 83, 48, 0.18) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .st-key-dashboard_assistant_button button:hover {
            background: linear-gradient(135deg, #d97545, #af5b34) !important;
        }
        .st-key-dashboard_assistant_student [data-baseweb="select"] > div {
            border-radius: 16px !important;
            border: 1px solid rgba(164, 111, 77, 0.14) !important;
            background:
                linear-gradient(180deg, rgba(255, 253, 250, 0.99), rgba(250, 243, 236, 0.98)) !important;
            min-height: 56px !important;
            height: 56px !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.75),
                0 8px 20px rgba(80, 55, 34, 0.05);
            display: flex !important;
            align-items: center !important;
            padding: 0 0.9rem !important;
        }
        .st-key-dashboard_assistant_student [data-baseweb="select"] {
            width: 100% !important;
        }
        .st-key-dashboard_assistant_student [data-baseweb="select"] > div > div {
            display: flex !important;
            align-items: center !important;
            min-height: 56px !important;
            height: 56px !important;
            padding: 0 !important;
        }
        .st-key-dashboard_assistant_student [data-baseweb="select"] > div > div > div,
        .st-key-dashboard_assistant_student [data-baseweb="select"] > div > div > div > div {
            display: flex !important;
            align-items: center !important;
            min-height: 56px !important;
            height: 56px !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        .st-key-dashboard_assistant_student [data-baseweb="select"] span,
        .st-key-dashboard_assistant_student [data-baseweb="select"] input {
            line-height: 1.35 !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            height: 56px !important;
            line-height: 56px !important;
        }
        .st-key-dashboard_assistant_prompt > div,
        .st-key-dashboard_assistant_prompt [data-testid="stTextInputRootElement"] {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            height: 56px !important;
            display: flex !important;
            align-items: center !important;
        }
        .st-key-dashboard_assistant_prompt [data-baseweb="base-input"] {
            border-radius: 16px !important;
            border: 1px solid rgba(165, 111, 78, 0.16) !important;
            background:
                linear-gradient(180deg, rgba(255, 253, 250, 0.99), rgba(251, 245, 238, 0.98)) !important;
            min-height: 56px !important;
            height: 56px !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.78),
                0 8px 20px rgba(80, 55, 34, 0.05) !important;
            overflow: hidden !important;
            align-items: center !important;
            padding: 0 1rem !important;
            width: 100% !important;
        }
        .st-key-dashboard_assistant_prompt [data-baseweb="base-input"] > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            min-height: 56px !important;
            height: 56px !important;
            display: flex !important;
            align-items: center !important;
        }
        .st-key-dashboard_assistant_prompt input {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            min-height: 56px !important;
            height: 56px !important;
            padding: 0 !important;
            outline: none !important;
            line-height: 1.35 !important;
            margin: 0 !important;
            display: block !important;
        }
        .st-key-dashboard_assistant_prompt input::placeholder {
            color: #8c7665 !important;
        }
        .st-key-dashboard_quick_prompt_0 button,
        .st-key-dashboard_quick_prompt_1 button,
        .st-key-dashboard_quick_prompt_2 button,
        .st-key-dashboard_quick_prompt_3 button {
            background:
                linear-gradient(180deg, rgba(255, 252, 248, 0.99), rgba(250, 243, 235, 0.98)) !important;
            border: 1px solid rgba(176, 123, 86, 0.15) !important;
            border-radius: 18px !important;
            min-height: 52px !important;
            color: #473429 !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            line-height: 1 !important;
            letter-spacing: 0.01em !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.82),
                0 10px 22px rgba(80, 55, 34, 0.05) !important;
            text-align: center !important;
            padding: 0 0.55rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        .st-key-dashboard_quick_prompt_0 button p,
        .st-key-dashboard_quick_prompt_1 button p,
        .st-key-dashboard_quick_prompt_2 button p,
        .st-key-dashboard_quick_prompt_3 button p,
        .st-key-dashboard_quick_prompt_0 button span,
        .st-key-dashboard_quick_prompt_1 button span,
        .st-key-dashboard_quick_prompt_2 button span,
        .st-key-dashboard_quick_prompt_3 button span {
            font-family: "Source Serif 4", "Noto Sans SC", serif !important;
            font-size: 0.86rem !important;
            font-weight: 700 !important;
            line-height: 1.05 !important;
            letter-spacing: 0.01em !important;
            color: #4a3427 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        .st-key-dashboard_quick_prompt_0 button:hover,
        .st-key-dashboard_quick_prompt_1 button:hover,
        .st-key-dashboard_quick_prompt_2 button:hover,
        .st-key-dashboard_quick_prompt_3 button:hover {
            border-color: rgba(176, 123, 86, 0.3) !important;
            background:
                linear-gradient(180deg, rgba(255, 249, 242, 0.99), rgba(247, 236, 224, 0.98)) !important;
        }
        .metric-label {
            color: #6f5c4d;
            font-size: 0.82rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .metric-value {
            margin-top: 0.55rem;
            color: var(--ink);
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.05;
        }
        .metric-help {
            margin-top: 0.55rem;
            color: #6d5a4b;
            line-height: 1.6;
            font-size: 0.92rem;
        }
        .metric-inline-wrap {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 30px rgba(80, 55, 34, 0.08);
            min-height: 122px;
        }
        .metric-inline-label {
            color: #6f5c4d;
            font-size: 0.82rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .metric-inline-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }
        .metric-inline-value {
            color: var(--ink);
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.05;
        }
        .metric-inline-help {
            margin-top: 0.5rem;
            color: #6d5a4b;
            line-height: 1.6;
            font-size: 0.92rem;
        }
        .student-case-metric {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.9rem 0.95rem;
            box-shadow: 0 10px 26px rgba(80, 55, 34, 0.07);
            min-height: 104px;
        }
        .student-case-metric-label {
            color: #6f5c4d;
            font-size: 0.76rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .student-case-metric-value {
            margin-top: 0.42rem;
            color: var(--ink);
            font-size: 1.1rem;
            font-weight: 800;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .student-case-metric-value.numeric {
            font-size: 1.35rem;
            letter-spacing: 0.01em;
        }
        .student-case-metric-help {
            margin-top: 0.38rem;
            color: #6d5a4b;
            line-height: 1.45;
            font-size: 0.84rem;
        }
        .student-case-chip-row {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .compact-stat-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 22px rgba(80, 55, 34, 0.06);
            min-height: 108px;
        }
        .compact-stat-title {
            color: #6f5c4d;
            font-size: 0.76rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .compact-stat-value {
            margin-top: 0.45rem;
            color: var(--ink);
            font-size: 1.22rem;
            font-weight: 800;
            line-height: 1.22;
            white-space: normal;
            word-break: break-word;
        }
        .compact-stat-help {
            margin-top: 0.42rem;
            color: #6d5a4b;
            line-height: 1.45;
            font-size: 0.84rem;
        }
        .flow-card {
            background:
                radial-gradient(circle at top right, rgba(214, 166, 120, 0.14), transparent 28%),
                linear-gradient(180deg, rgba(255, 251, 246, 0.98), rgba(248, 239, 229, 0.96));
            border: 1px solid rgba(146, 105, 78, 0.16);
            border-radius: 22px;
            padding: 0.95rem 1rem;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.8),
                0 12px 28px rgba(119, 83, 56, 0.06);
            min-height: 100%;
        }
        .flow-step {
            color: #9a6848;
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }
        .flow-title {
            color: #32251d;
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }
        .flow-copy {
            color: #6f5b4c;
            font-size: 0.9rem;
            line-height: 1.65;
        }
        .section-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #34271f;
            margin-bottom: 0.8rem;
        }
        .narrative {
            color: #3a2d24;
            line-height: 1.8;
            background: rgba(255,255,255,0.72);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(112, 84, 64, 0.12);
        }
        .requirements-card {
            background:
                radial-gradient(circle at top right, rgba(214, 166, 120, 0.18), transparent 30%),
                linear-gradient(180deg, rgba(255, 251, 246, 0.98), rgba(248, 239, 229, 0.96));
            border: 1px solid rgba(146, 105, 78, 0.16);
            border-radius: 24px;
            padding: 1rem 1rem 0.95rem;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.82),
                0 14px 32px rgba(119, 83, 56, 0.07);
            min-height: 100%;
        }
        .requirements-card-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.7rem;
        }
        .requirements-kicker {
            color: #9a6848;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }
        .requirements-card-title {
            color: #33261e;
            font-size: 1.24rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .requirements-card-copy {
            color: #6b594b;
            font-size: 0.9rem;
            line-height: 1.6;
            margin-bottom: 0.7rem;
        }
        .requirements-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-bottom: 0.85rem;
        }
        .requirements-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.3rem 0.72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(146, 105, 78, 0.12);
            color: #6f5443;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .requirements-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.68rem;
        }
        .requirements-metric {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(146, 105, 78, 0.1);
            border-radius: 16px;
            padding: 0.72rem 0.78rem;
        }
        .requirements-metric-label {
            color: #7b624f;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
        }
        .requirements-metric-value {
            color: #2f241d;
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .requirements-note {
            margin-top: 0.85rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(146, 105, 78, 0.12);
            color: #6f5b4c;
            font-size: 0.88rem;
            line-height: 1.62;
        }
        .requirements-inline-time {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            color: #7a644f;
            font-size: 0.8rem;
            font-weight: 500;
            line-height: 1.4;
            white-space: nowrap;
            padding: 0 0.15rem 0.05rem 0.15rem;
            margin-top: 0.28rem;
        }
        .status-chip, .risk-chip, .map-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }
        .status-chip { background: rgba(56, 90, 115, 0.12); color: var(--navy); }
        .risk-high { background: rgba(184, 75, 66, 0.14); color: var(--red); }
        .risk-mid { background: rgba(193, 139, 43, 0.16); color: var(--amber); }
        .risk-low { background: rgba(79, 125, 87, 0.14); color: var(--green); }
        .map-chip { background: rgba(185, 92, 52, 0.12); color: var(--accent); }
        .source-note {
            color: rgba(246, 237, 228, 0.9);
            font-size: 0.88rem;
            line-height: 1.72;
            padding-top: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        .divider-space {
            height: 1.15rem;
        }
        .page-section-gap {
            height: 1.3rem;
        }
        .app-data-table-shell {
            background:
                linear-gradient(180deg, rgba(255, 252, 248, 0.98), rgba(252, 245, 238, 0.96));
            border: 1px solid rgba(146, 105, 78, 0.18);
            border-radius: 22px;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.82),
                0 12px 28px rgba(119, 83, 56, 0.07);
            padding: 0.35rem;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            overflow: hidden;
            margin-bottom: 0.55rem;
            outline: none !important;
        }
        .app-data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            table-layout: fixed;
            color: #35281f;
            font-size: 0.95rem;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            border-radius: 18px;
            overflow: hidden;
            background: rgba(255, 252, 247, 0.94);
        }
        .app-data-table thead th {
            background: linear-gradient(180deg, rgba(239, 228, 216, 0.96), rgba(232, 217, 203, 0.94));
            color: #6c4d39;
            font-weight: 800;
            letter-spacing: 0.01em;
            text-align: left;
            padding: 0.84rem 0.72rem;
            border-bottom: 1px solid rgba(146, 105, 78, 0.16);
            border-right: 1px solid rgba(146, 105, 78, 0.08);
            white-space: normal;
            overflow-wrap: anywhere;
            box-shadow: none !important;
            background-clip: padding-box;
        }
        .app-data-table thead th:first-child {
            border-top-left-radius: 18px;
            border-left: none !important;
        }
        .app-data-table thead th:last-child {
            border-top-right-radius: 18px;
            border-right: none;
        }
        .app-data-table tbody td {
            background: rgba(255, 252, 247, 0.88);
            padding: 0.86rem 0.72rem;
            border-bottom: 1px solid rgba(146, 105, 78, 0.11);
            border-right: 1px solid rgba(146, 105, 78, 0.06);
            vertical-align: middle;
            word-break: break-word;
            overflow-wrap: anywhere;
            transition: background 0.16s ease, color 0.16s ease;
            box-shadow: none !important;
            background-clip: padding-box;
        }
        .app-data-table tbody tr:nth-child(even) td {
            background: rgba(250, 242, 234, 0.94);
        }
        .app-data-table tbody tr:hover td {
            background: rgba(243, 230, 216, 0.98);
            color: #2f241d;
        }
        .app-data-table tbody td:first-child {
            font-weight: 700;
            color: #4c372a;
            border-left: none !important;
        }
        .app-data-table tbody td:last-child {
            border-right: none;
        }
        .app-data-table tbody tr:last-child td {
            border-bottom: none;
        }
        .app-data-table tbody tr:last-child td:first-child {
            border-bottom-left-radius: 18px;
        }
        .app-data-table tbody tr:last-child td:last-child {
            border-bottom-right-radius: 18px;
        }
        .app-data-table tbody tr:last-child td {
            border-bottom: none !important;
        }
        .app-data-table-shell table,
        .app-data-table-shell colgroup,
        .app-data-table-shell thead,
        .app-data-table-shell tbody,
        .app-data-table-shell tr,
        .app-data-table-shell th,
        .app-data-table-shell td {
            outline: none !important;
            box-shadow: none !important;
        }
        .table-card-breath {
            height: 0.45rem;
        }
        [data-testid="stAltairChart"] {
            background:
                linear-gradient(180deg, rgba(255, 251, 246, 0.96), rgba(250, 243, 235, 0.94));
            border: 1px solid rgba(146, 105, 78, 0.14);
            border-radius: 22px;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.8),
                0 10px 24px rgba(119, 83, 56, 0.06);
            padding: 0.8rem 0.9rem 0.45rem;
            overflow: hidden;
        }
        [data-testid="stAltairChart"] > div,
        [data-testid="stAltairChart"] .vega-embed,
        [data-testid="stAltairChart"] canvas,
        [data-testid="stAltairChart"] svg {
            border-radius: 18px !important;
            overflow: hidden !important;
        }
        [data-testid="stAltairChart"] .vega-bindings,
        [data-testid="stAltairChart"] .vega-actions {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def themed_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.properties(
            background="#fbf5ee",
            padding={"left": 10, "right": 10, "top": 10, "bottom": 4},
        )
        .configure(background="#fbf5ee")
        .configure_view(
            strokeOpacity=0,
            fill="#fbf5ee",
        )
        .configure_axis(
            labelColor="#6a5343",
            labelFontSize=12,
            titleColor="#7b604d",
            titleFontSize=13,
            titleFontWeight=700,
            gridColor="rgba(146,105,78,0.14)",
            domainColor="rgba(146,105,78,0.18)",
            tickColor="rgba(146,105,78,0.18)",
        )
        .configure_header(
            labelColor="#6a5343",
            titleColor="#7b604d",
        )
    )


def section_title(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def render_app_chrome() -> str:
    if "view_role" not in st.session_state or not st.session_state.view_role:
        st.session_state.view_role = "教师/管理端"

    st.markdown("<div class='app-chrome'><div class='role-switch-shell'>", unsafe_allow_html=True)
    selected = st.selectbox(
        "角色切换",
        ["教师/管理端", "学生自助端"],
        index=0 if st.session_state.view_role == "教师/管理端" else 1,
        label_visibility="collapsed",
        key="chrome_role_switch",
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state.view_role = selected
    return selected


def page_header(title: str, subtitle: str, badge: str = "", compact: bool = False) -> None:
    subtitle_html = f"<div class='hero-copy'>{subtitle}</div>" if subtitle else ""
    badge_html = f"<div class='hero-kicker'>{badge}</div>" if badge else ""
    shell_class = "hero-shell compact" if compact else "hero-shell compact"
    st.markdown(
        f"<div class='{shell_class}'>"
        f"{badge_html}"
        f"<div class='hero-title'>{title}</div>"
        f"{subtitle_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        "<div class='metric-tile'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-help'>{help_text}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def narrative_card(text: str) -> None:
    st.markdown(f"<div class='narrative'>{text}</div>", unsafe_allow_html=True)


def render_table(df: pd.DataFrame, caption: str = "") -> None:
    if caption:
        section_title(caption)
    if df.empty:
        st.info("当前没有可展示的数据。")
        return
    display_df = prettify_dataframe_for_display(df)
    table_html = display_df.to_html(index=False, classes="app-data-table", border=0, escape=True)
    st.markdown(f"<div class='app-data-table-shell'>{table_html}</div>", unsafe_allow_html=True)


def risk_chip(label: Any) -> str:
    text = str(label or "未知")
    if "高风险" in text:
        cls = "risk-chip risk-high"
    elif "中风险" in text:
        cls = "risk-chip risk-mid"
    else:
        cls = "risk-chip risk-low"
    return f"<span class='{cls}'>{text}</span>"


def map_chip(label: Any) -> str:
    return f"<span class='map-chip'>{map_to_cn(label)}</span>"


def status_chip(label: Any) -> str:
    return f"<span class='status-chip'>{decision_to_cn(label)}</span>"


def _build_source_lines(data: dict[str, Any]) -> list[str]:
    bundle_meta = data.get("frontend_bundle", {}) if isinstance(data.get("frontend_bundle", {}), dict) else {}
    bundle_counts = bundle_meta.get("counts", {}) if isinstance(bundle_meta.get("counts", {}), dict) else {}
    harness = data.get("harness", {}) if isinstance(data.get("harness", {}), dict) else {}
    lines = [
        "学生数据已脱敏处理",
        f"数据来源：{data_source_to_cn(data.get('data_source'))}",
    ]
    if bundle_meta:
        lines.append(f"Bundle 学生数：{bundle_counts.get('students', '-')}")
        if bundle_meta.get("generated_at"):
            lines.append(f"更新时间：{bundle_meta.get('generated_at')}")
        if bundle_meta.get("built_from_run_id"):
            lines.append(f"Run ID：{bundle_meta.get('built_from_run_id')}")
    if harness.get("run_record_path"):
        lines.append(f"当前记录：{Path(str(harness.get('run_record_path'))).name}")
    return lines


def render_sidebar(data: dict[str, Any]) -> dict[str, Any]:
    master_df = data["master_df"]
    view_role = st.session_state.view_role

    with st.sidebar.container(border=True):
        st.markdown(
            "<div class='sidebar-brand'>"
            "<div class='brand-lockup'>"
            "<div class='product-logo'></div>"
            "<div>"
            "<div class='brand-eyebrow'>A14 Student Insight</div>"
            "<div class='brand-title'>知行镜</div>"
            "<div class='brand-subtitle'>学生行为分析与干预工作台</div>"
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    pages = get_pages(view_role)
    page_label_map: dict[str, str] = {}
    if view_role == "教师/管理端" and len(pages) >= 3:
        page_label_map[pages[2]] = "动态建模展示"
    if st.session_state.current_page not in pages:
        st.session_state.current_page = pages[0]
        st.session_state.sidebar_nav_radio = st.session_state.current_page
    pending_page = st.session_state.get("pending_page_nav")
    if pending_page in pages:
        st.session_state.sidebar_nav_radio = pending_page
        st.session_state.current_page = pending_page
        st.session_state.pending_page_nav = None
    elif st.session_state.get("sidebar_nav_radio") not in pages:
        st.session_state.sidebar_nav_radio = st.session_state.current_page

    page_options = pages[:]
    current_index = pages.index(st.session_state.get("sidebar_nav_radio", st.session_state.current_page))

    with st.sidebar.container(border=True):
        st.markdown("<div class='sidebar-section-label'>工作区</div>", unsafe_allow_html=True)
        picked = st.radio(
            "页面选择",
            page_options,
            index=current_index,
            format_func=lambda item: page_label_map.get(item, item),
            label_visibility="collapsed",
            key="sidebar_nav_radio",
        )
    selected_page = pages[page_options.index(picked)]
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page

    filtered_df = master_df.copy()
    if view_role == "教师/管理端" and not master_df.empty:
        with st.sidebar.container(border=True):
            st.markdown("<div class='sidebar-section-label'>全局筛选</div>", unsafe_allow_html=True)
            with st.expander("风险等级", expanded=True):
                risk_levels = st.multiselect(
                    "风险等级",
                    options=sorted(master_df["risk_level"].dropna().astype(str).unique().tolist()),
                    default=[],
                    label_visibility="collapsed",
                )
            with st.expander("主导维度", expanded=False):
                dimensions = st.multiselect(
                    "主导维度",
                    options=sorted(master_df["dominant_dimension"].dropna().astype(str).unique().tolist()),
                    default=[],
                    format_func=dimension_to_cn,
                    label_visibility="collapsed",
                )
            with st.expander("行为模式", expanded=False):
                patterns = st.multiselect(
                    "行为模式",
                    options=sorted(master_df["pattern_label"].dropna().astype(str).unique().tolist()),
                    default=[],
                    label_visibility="collapsed",
                )
            min_total_risk = st.slider("综合风险下限", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        filtered_df = apply_filters(master_df, risk_levels, dimensions, patterns, min_total_risk)


    student_options = filtered_df.get("student_id", pd.Series(dtype=str)).astype(str).tolist() if not filtered_df.empty else []
    if student_options:
        pending_student = st.session_state.get("pending_selected_student")
        if pending_student in student_options:
            st.session_state.selected_student = pending_student
            st.session_state.pending_selected_student = None
        if st.session_state.selected_student not in student_options:
            st.session_state.selected_student = student_options[0]
        with st.sidebar.container(border=True):
            st.markdown("<div class='sidebar-section-label'>学生焦点</div>", unsafe_allow_html=True)
            st.selectbox(
                "当前学生",
                student_options,
                index=student_options.index(st.session_state.selected_student),
                key="selected_student",
                label_visibility="collapsed",
            )
    return {
        "view_role": view_role,
        "page": st.session_state.current_page,
        "filtered_df": filtered_df,
    }


def summary_cards(filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        st.info("筛选后暂无学生。")
        return
    total_students = len(filtered_df)
    high_risk = int((filtered_df["risk_level"] == "高风险").sum())
    avg_risk = filtered_df["total_risk"].fillna(0).mean()
    top_dimension = filtered_df["dominant_dimension"].mode().iloc[0] if not filtered_df["dominant_dimension"].mode().empty else "unknown"

    cols = st.columns(4)
    with cols[0]:
        metric_card("学生总数", str(total_students), "当前筛选结果")
    with cols[1]:
        metric_card("高风险人数", str(high_risk), f"占比 {fmt_pct(high_risk / total_students if total_students else 0)}")
    with cols[2]:
        metric_card("平均综合风险", fmt_num(avg_risk, 3), "以主表 total_risk 计算")
    with cols[3]:
        metric_card("主导维度", dimension_to_cn(top_dimension), "当前最常见风险来源")
