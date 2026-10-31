import streamlit as st
import plotly.express as px
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import setup, get_llm

setup("AI Analytics Copilot", "🤖")
from src.llm.copilot import Copilot


@st.cache_resource
def copilot():
    return Copilot(get_llm())


if "chat" not in st.session_state:
    st.session_state.chat = []
EX = ["What is the overall churn rate?", "Which acquisition channel has the highest churn?", "What are customers saying about payment failures?",
      "Why did churn increase in India in May 2026?", "What did the analysis find about churn drivers?", "How reliable is the churn model?", "Explain CUST000702", "What should we do to retain CUST000702?", "Generate a monthly report"]
cols = st.columns(3)
for i, e in enumerate(EX):
    if cols[i % 3].button(e, key=f"ex{i}", use_container_width=True):
        st.session_state.pending = e
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("resp"):
            r = m["resp"]
            st.caption(f"Capability: **{r.capability}**")
            if r.sql:
                with st.expander("Generated SQL"):
                    st.code(r.sql, language="sql")
            if r.df is not None and len(r.df):
                st.dataframe(r.df, use_container_width=True, hide_index=True)
                if r.chart:
                    fn = px.line if r.chart["kind"] == "line" else px.bar
                    st.plotly_chart(fn(r.df, x=r.chart["x"], y=r.chart["y"]), use_container_width=True)
            if r.steps:
                with st.expander("Investigation steps (every number came from these queries)"):
                    for n, s in r.steps:
                        st.markdown(f"**{n}**"); st.code(s, language="sql")
            for n in r.notes:
                st.caption(n)
prompt = st.chat_input("Ask about customers, churn, revenue, complaints...") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.chat.append({"role": "user", "text": prompt})
    with st.spinner("Thinking..."):
        resp = copilot().ask(prompt)
    st.session_state.chat.append({"role": "assistant", "text": resp.text, "resp": resp})
    st.rerun()
