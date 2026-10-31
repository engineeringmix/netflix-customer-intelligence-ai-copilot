import plotly.express as px
import streamlit as st
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import setup, q, RED

setup("Customer Analytics", "👥")
opts = q("SELECT DISTINCT country FROM customers ORDER BY 1").country.tolist()
f1, f2, f3 = st.columns(3)
country = f1.multiselect("Country", opts)
plan = f2.multiselect("Current plan", ["Basic", "Standard", "Premium"])
seg = f3.multiselect("Segment", q("SELECT DISTINCT segment_name FROM customer_scores ORDER BY 1").segment_name.tolist())
df = q("SELECT * FROM v_customer_360")
if country: df = df[df.country.isin(country)]
if plan: df = df[df.current_plan.isin(plan)]
if seg: df = df[df.segment_name.isin(seg)]
st.write(f"**{len(df):,} customers** match; churn rate **{100 * df.churned.mean():.1f}%**" if len(df) else "No customers match.")
if len(df):
    t1, t2, t3, t4 = st.tabs(["Demographics", "Engagement", "Subscription behaviour", "Segments"])
    with t1:
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(df, x="age", nbins=30, title="Age", color_discrete_sequence=[RED]), use_container_width=True)
        b.plotly_chart(px.bar(df.country.value_counts().head(12).reset_index(), x="count", y="country", orientation="h", title="Top countries", color_discrete_sequence=[RED]), use_container_width=True)
        st.plotly_chart(px.bar(df.acquisition_channel.value_counts().reset_index(), x="acquisition_channel", y="count", title="Acquisition channel", color_discrete_sequence=[RED]), use_container_width=True)
    with t2:
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(df, x="engagement_score", color="churned", nbins=30, barmode="overlay", opacity=.6, title="Engagement score: churned vs retained"), use_container_width=True)
        b.plotly_chart(px.box(df, x="primary_device", y="avg_watch_time_per_session", title="Avg minutes per session by main device", color_discrete_sequence=[RED]), use_container_width=True)
        st.plotly_chart(px.box(df, x="churned", y="days_since_last_activity", title="Days since last activity: retained (False) vs churned (True)", color_discrete_sequence=[RED]), use_container_width=True)
    with t3:
        a, b = st.columns(2)
        g = df.groupby("current_plan").agg(customers=("customer_id", "size"), churn=("churned", "mean"), auto_renew=("auto_renew", "mean"), avg_clv=("customer_lifetime_value", "mean")).round(3).reset_index()
        a.dataframe(g, hide_index=True, use_container_width=True)
        b.plotly_chart(px.bar(g, x="current_plan", y="churn", title="Churn rate by plan", color_discrete_sequence=[RED]), use_container_width=True)
        st.caption(f"Customers who changed plan at least once: {int((df.plan_changes > 0).sum()):,} ({100 * (df.plan_changes > 0).mean():.1f}%).")
    with t4:
        s = df.groupby("segment_name").agg(customers=("customer_id", "size"), churn=("churned", "mean"), avg_engagement=("engagement_score", "mean"), avg_clv=("customer_lifetime_value", "mean")).round(2).reset_index()
        st.dataframe(s, hide_index=True, use_container_width=True)
