import streamlit as st
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import setup, get_llm

setup("Automated Reports", "📄")
from src.llm.reports import generate

kind = st.radio("Period", ["weekly", "monthly", "quarterly"], horizontal=True, index=1)
if st.button("Generate report", type="primary"):
    st.session_state.report = generate(kind, get_llm())
rep = st.session_state.get("report")
if rep:
    md = rep.markdown()
    st.markdown(md)
    st.download_button("Download (.md)", md, file_name=f"{rep.kind}_report_{rep.label.replace(' ', '_')}.md", mime="text/markdown")
else:
    st.caption("Facts are computed with SQL first; the interpretation is written afterwards and checked so it cannot cite numbers the SQL did not return.")
