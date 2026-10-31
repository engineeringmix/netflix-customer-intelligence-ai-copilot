"""Page 1: Executive Dashboard.  Run: streamlit run app/dashboard.py"""
import plotly.express as px
import streamlit as st

from common import setup, q, RED, TIER_COLORS

setup("Executive Dashboard")
k = q("""SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE NOT churned) AS active, ROUND(100.0*AVG(churned::int),1) AS churn_rate,
                COUNT(*) FILTER (WHERE NOT churned AND risk_tier='High') AS high_risk,
                ROUND(SUM(plan_price*churn_risk_score) FILTER (WHERE NOT churned AND risk_tier='High')::numeric,0) AS at_risk_rev
         FROM v_customer_360""").iloc[0]
rev = q("SELECT ROUND(SUM(amount) FILTER (WHERE payment_status='Success'),0) AS total, ROUND(SUM(amount) FILTER (WHERE payment_status='Success' AND payment_date >= DATE '2026-05-01'),0) AS may FROM payments").iloc[0]
csat = q("SELECT ROUND(AVG(customer_satisfaction_score),2) AS v FROM support_tickets").v[0]
c = st.columns(6)
c[0].metric("Total customers", f"{int(k.total):,}")
c[1].metric("Active customers", f"{int(k.active):,}")
c[2].metric("Churn rate", f"{k.churn_rate}%")
c[3].metric("Revenue (all-time)", f"${int(rev.total):,}", f"May-26: ${int(rev.may):,}", delta_color="off")
c[4].metric("High-risk customers", f"{int(k.high_risk):,}", f"~${int(k.at_risk_rev):,}/mo at risk", delta_color="off")
c[5].metric("Avg support CSAT", f"{csat}/5")
st.caption("Billing data ends 2026-05-31. 'High risk' = model churn probability >= 40% among active customers.")

a, b = st.columns(2)
m = q("""SELECT DATE_TRUNC('month', churn_date)::date AS month, COUNT(*) AS cancellations FROM churn_labels WHERE churned GROUP BY 1 ORDER BY 1""")
a.plotly_chart(px.bar(m, x="month", y="cancellations", title="Cancellations per month", color_discrete_sequence=[RED]), use_container_width=True)
r = q("""SELECT sp.plan_name, ROUND(SUM(p.amount)) AS revenue FROM payments p JOIN subscriptions s USING (subscription_id) JOIN subscription_plans sp USING (plan_id)
         WHERE p.payment_status='Success' GROUP BY 1""")
b.plotly_chart(px.pie(r, names="plan_name", values="revenue", title="Revenue mix by plan", color_discrete_sequence=[RED, "#221F1F", "#B3B3B3"]), use_container_width=True)
a, b = st.columns(2)
t = q("SELECT risk_tier, COUNT(*) AS customers, ROUND(100*AVG(churned::int),1) AS actual_churn_pct FROM v_customer_360 GROUP BY 1")
a.plotly_chart(px.bar(t, x="risk_tier", y="customers", color="risk_tier", color_discrete_map=TIER_COLORS, category_orders={"risk_tier": ["Low", "Medium", "High"]},
                      title="Customers by model risk tier"), use_container_width=True)
b.dataframe(t.rename(columns={"actual_churn_pct": "observed churn %"}), use_container_width=True, hide_index=True)
b.caption("Model quality in one line: High-tier customers churn about 3x as often as Low-tier. The model ranks risk, it does not predict individuals reliably (ROC-AUC about 0.66).")
