"""Netflix-inspired dark theme for the M3DocRAG explorer.

Palette and component styles live here so app.py stays about behaviour.
Streamlit's internal DOM changes between releases, so selectors that target
Streamlit's own testids are treated as progressive enhancement - every
custom component below carries its own styling and stays legible if a
Streamlit-internal selector stops matching.
"""

import streamlit as st

RED = "#E50914"
RED_DARK = "#B20710"
GREEN = "#46D369"  # Netflix's "match score" green
BG = "#141414"
CARD = "#181818"
CARD_HI = "#232323"
TEXT = "#FFFFFF"
MUTED = "#B3B3B3"

MODALITY_COLORS = {"text": "#4A9EFF", "table": GREEN, "image": "#F5B50A"}
MODALITY_ICONS = {"text": "📝", "table": "📊", "image": "🖼️"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}}
.stApp {{ background: {BG}; }}

/* the sidebar is gone - all controls live in the top bar */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{ padding-top: 1.2rem !important; max-width: 1500px; }}

/* ---------- masthead ---------- */
.nf-header {{
    display: flex; align-items: baseline; gap: 14px;
    padding: 4px 0 14px 0; flex-wrap: wrap;
}}
.nf-wordmark {{
    font-size: 2.1rem; font-weight: 900; letter-spacing: -0.055em;
    color: {RED}; text-transform: uppercase; line-height: 1;
    text-shadow: 0 2px 14px rgba(229,9,20,0.35);
}}
.nf-wordmark span {{ color: {TEXT}; }}
.nf-tagline {{ color: {MUTED}; font-size: 0.9rem; font-weight: 500; }}

/* ---------- control bar ---------- */
.nf-controlbar {{
    background: linear-gradient(180deg, {CARD_HI} 0%, {CARD} 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 6px 14px 2px 14px; margin-bottom: 16px;
}}
.nf-ctl-label {{
    color: {MUTED}; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: -6px;
}}

/* ---------- stat cards ---------- */
.nf-stat {{
    background: {CARD}; border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid var(--accent, {RED});
    border-radius: 8px; padding: 14px 18px; height: 100%;
    transition: transform .15s ease, background .15s ease;
}}
.nf-stat:hover {{ transform: translateY(-2px); background: {CARD_HI}; }}
.nf-stat-label {{
    color: {MUTED}; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.13em; text-transform: uppercase;
}}
.nf-stat-value {{
    color: {TEXT}; font-size: 1.9rem; font-weight: 800;
    letter-spacing: -0.03em; line-height: 1.15;
}}

/* ---------- tabs as top nav ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px; background: transparent; border-bottom: 1px solid rgba(255,255,255,0.10);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 6px 6px 0 0; padding: 9px 20px;
    color: {MUTED}; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.02em;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT}; background: rgba(255,255,255,0.05); }}
.stTabs [aria-selected="true"] {{
    color: {TEXT} !important; background: rgba(229,9,20,0.14) !important;
    box-shadow: inset 0 -3px 0 {RED};
}}
.stTabs [data-baseweb="tab-highlight"] {{ background: {RED}; }}

/* ---------- badges ---------- */
.badge {{
    display: inline-block; padding: 3px 11px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; color: #fff; margin-right: 6px; margin-bottom: 4px;
}}

/* ---------- detail panel ---------- */
.nf-detail {{
    background: linear-gradient(135deg, {CARD_HI} 0%, {CARD} 55%);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 22px 26px; margin-top: 6px;
}}
.nf-question {{
    color: {TEXT}; font-size: 1.5rem; font-weight: 800;
    letter-spacing: -0.02em; line-height: 1.3; margin-bottom: 12px;
}}
.nf-answer-label {{
    color: {MUTED}; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 6px;
}}
.answer-box {{
    padding: 13px 16px; border-radius: 8px; font-size: 1rem;
    font-weight: 600; color: {TEXT}; min-height: 3rem;
    border-left: 4px solid; background: rgba(255,255,255,0.045);
}}
.gold-box {{ border-color: {GREEN}; }}
.pred-correct {{ border-color: {GREEN}; }}
.pred-wrong {{ border-color: {RED}; }}

.nf-section {{
    color: {TEXT}; font-size: 1.05rem; font-weight: 800;
    letter-spacing: 0.01em; margin: 22px 0 4px 0;
}}
.nf-note {{ color: {MUTED}; font-size: 0.83rem; line-height: 1.55; }}
.small-mono {{
    font-family: 'SF Mono', Menlo, monospace; font-size: 0.76rem; color: {MUTED};
}}

/* ---------- page cards ---------- */
.nf-pagecard {{
    background: {CARD}; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;
}}
.nf-pagecard.hit {{ border-left: 4px solid {GREEN}; }}
.nf-pagecard.miss {{ border-left: 4px solid {RED}; }}

/* ---------- widgets ---------- */
div[data-testid="stDataFrame"] {{ border: 1px solid rgba(255,255,255,0.09); border-radius: 8px; }}
div[data-baseweb="select"] > div {{
    background: {CARD_HI} !important; border-color: rgba(255,255,255,0.14) !important;
}}
.stTextInput input {{
    background: {CARD_HI} !important; border-color: rgba(255,255,255,0.14) !important;
    color: {TEXT} !important;
}}
div[data-testid="stExpander"] {{
    background: {CARD}; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
}}
.stProgress > div > div > div > div {{ background: {RED}; }}
hr {{ border-color: rgba(255,255,255,0.09); }}
#MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


def modality_badge(modality):
    color = MODALITY_COLORS.get(modality, "#6b7280")
    icon = MODALITY_ICONS.get(modality, "❓")
    return badge(f"{icon} {modality}", color)


def masthead(subtitle: str):
    st.markdown(
        f"""
        <div class="nf-header">
            <div class="nf-wordmark">M3Doc<span>RAG</span></div>
            <div class="nf-tagline">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, accent: str = RED):
    st.markdown(
        f"""
        <div class="nf-stat" style="--accent:{accent}">
            <div class="nf-stat-label">{label}</div>
            <div class="nf-stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
