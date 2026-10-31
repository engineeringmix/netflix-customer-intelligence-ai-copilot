import pandas as pd
import plotly.express as px
import streamlit as st
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import setup, q, get_llm, model_report, figure, RED, TIER_COLORS

setup("Churn Prediction", "🔮")
from src.llm.explainer import explain_customer, recommend, contributions, get_customer, HIDE

top = q("SELECT customer_id, churn_risk_score, risk_tier, current_plan FROM v_customer_360 WHERE NOT churned ORDER BY churn_risk_score DESC LIMIT 200")
c1, c2 = st.columns([1, 2])
pick = c1.selectbox("Pick from the 200 highest-risk active customers", top.customer_id, format_func=lambda i: f"{i}  ({top.set_index('customer_id').loc[i, 'churn_risk_score']:.0%})")
typed = c2.text_input("...or type any customer id (e.g. CUST000123)")
cid = (typed or pick).strip().upper()
e = explain_customer(cid, get_llm())
if e is None:
    st.error(f"Customer {cid} not found.")
else:
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{e.probability:.0%}", "portfolio average 25%", delta_color="off")
    b.metric("Risk tier", e.tier)
    c.metric("Plan / segment", f"{e.row.current_plan} / {e.row.segment_name}")
    if e.row.churned:
        st.warning("This customer has already churned (historical label).")
    st.subheader("Why?")
    st.write(e.narrative)
    contrib = contributions(e.row)
    contrib = contrib[~contrib.index.isin(HIDE)]
    show = pd.concat([contrib.head(6), contrib.tail(6)]).reset_index()
    show.columns = ["feature", "effect on risk (log-odds)"]
    show["direction"] = show["effect on risk (log-odds)"].apply(lambda v: "raises risk" if v > 0 else "lowers risk")
    st.plotly_chart(px.bar(show, x="effect on risk (log-odds)", y="feature", color="direction", orientation="h", color_discrete_map={"raises risk": RED, "lowers risk": "#2A9D5C"},
                           title="Top model drivers for this customer (exact TreeSHAP)"), use_container_width=True)
    st.subheader("Recommended retention action")
    r = recommend(cid, get_llm())
    st.markdown(f"**{r.priority} priority: {r.action}**")
    for w in r.rationale:
        st.markdown(f"- {w}")
    if r.message:
        st.info(r.message)
with st.expander("How good is this model? (read before trusting any single score)"):
    rep = model_report()
    st.dataframe(pd.DataFrame(rep["churn_comparison"]).round(3), hide_index=True, use_container_width=True)
    d = rep["churn_decile_lift"]
    st.write(f"Top-decile customers churn at **{d['top_decile_churn_rate']:.0%}** vs a base rate of {d['base_rate']:.0%} (lift {d['top_decile_lift']:.1f}x); bottom decile {d['bottom_decile_churn_rate']:.0%}. "
             f"ROC-AUC is about 0.65: useful for ranking who to contact first, weak for predicting any one person.")
    st.caption(rep.get("churn_selection_rule", ""))
    figure("model_churn_curves.png")
