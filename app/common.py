"""Shared helpers for all Streamlit pages."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.llm.client import LLM
from src.llm.sql_guard import engine
from src.utils.config import REPORTS, FIGURES, AS_OF_DATE


def setup(title: str, icon: str = "🎬"):
    st.set_page_config(page_title=f"{title} | Customer Intelligence", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")
    with st.sidebar:
        st.markdown("**Netflix Customer Intelligence Platform**")
        st.caption(f"Data as of {AS_OF_DATE}. Independent portfolio project modeled on a subscription-streaming business; synthetic data.")
        llm = get_llm()
        st.info("AI mode: **LLM connected**" if llm.available else "AI mode: **offline** (rule-based). Set `ANTHROPIC_API_KEY` for free-form AI answers.")


@st.cache_resource
def get_llm():
    return LLM()


@st.cache_data(show_spinner=False, ttl=600)
def q(sql: str, **params) -> pd.DataFrame:
    with engine().connect() as c:
        return pd.read_sql(text(sql), c, params=params or None)


@st.cache_data(show_spinner=False)
def model_report() -> dict:
    return json.loads((REPORTS / "model_report.json").read_text())


def figure(name: str):
    p = FIGURES / name
    if p.exists():
        st.image(str(p), use_container_width=True)


RED, DARK, GREY = "#E50914", "#221F1F", "#B3B3B3"
TIER_COLORS = {"Low": "#2A9D5C", "Medium": "#F4A261", "High": RED}
