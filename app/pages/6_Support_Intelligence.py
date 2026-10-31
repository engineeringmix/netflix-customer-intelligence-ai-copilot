import plotly.express as px
import streamlit as st
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import setup, q, get_llm, RED

setup("Support Intelligence", "🎧")
from src.llm.rag import RagStore


@st.cache_resource
def store():
    return RagStore()


st.subheader("Complaint search")
query = st.text_input("Ask about complaints or feedback", "What are customers saying about payment failures?")
if query:
    r = store().ask(query, get_llm())
    st.markdown(r.summary)
    st.caption(f"Based on {r.n_records:,} of {r.n_total:,} records. Search is TF-IDF/LSA over FAISS; complaint text in this dataset is highly repetitive (about 45 distinct texts), so results are theme counts rather than nuanced semantics.")
    if r.sources is not None and len(r.sources):
        st.dataframe(r.sources, hide_index=True, use_container_width=True)
a, b = st.columns(2)
cat = q("""SELECT issue_category, COUNT(*) AS tickets, ROUND(AVG(customer_satisfaction_score),2) AS avg_csat, ROUND(AVG(resolution_time_hours),1) AS avg_resolution_h
           FROM support_tickets GROUP BY 1 ORDER BY tickets DESC""")
a.plotly_chart(px.bar(cat, x="issue_category", y="tickets", color="avg_csat", title="Complaint volume (colour = avg CSAT)", color_continuous_scale="RdYlGn"), use_container_width=True)
tr = q("""SELECT DATE_TRUNC('quarter', ticket_date)::date AS quarter, COUNT(*) AS tickets FROM support_tickets WHERE ticket_date IS NOT NULL GROUP BY 1 ORDER BY 1 DESC LIMIT 16""")
b.plotly_chart(px.line(tr.sort_values("quarter"), x="quarter", y="tickets", markers=True, title="Tickets per quarter"), use_container_width=True)
st.subheader("Sentiment: the provided label vs the star rating")
s = q("SELECT rating, sentiment_label, COUNT(*) AS n FROM customer_feedback GROUP BY 1, 2")
st.plotly_chart(px.bar(s, x="rating", y="n", color="sentiment_label", barmode="stack", title="Sentiment label by rating (labels disagree with ratings ~5% of the time)"), use_container_width=True)
m = q("SELECT COUNT(*) FILTER (WHERE sentiment_mismatch) AS bad, COUNT(*) AS total FROM customer_feedback").iloc[0]
st.caption(f"{int(m.bad)} of {int(m.total)} labels contradict the rating; the platform uses the rating-derived label (`sentiment_validated`) for analysis.")
