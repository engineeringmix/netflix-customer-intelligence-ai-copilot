import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import setup, q, model_report, figure

setup("Customer Segmentation", "🧩")
from src.utils.config import MODELS

seg = joblib.load(MODELS / "segmentation.joblib")
df = q("SELECT * FROM v_customer_360")
X = df[seg["cols"]].copy()
X["avg_customer_satisfaction"] = X.avg_customer_satisfaction.fillna(X.avg_customer_satisfaction.median())
X["avg_watch_time_per_session"] = X.avg_watch_time_per_session.fillna(0)
import numpy as np
for c in ["login_frequency", "failed_payment_rate", "support_ticket_count"]:
    X[c] = np.log1p(X[c])
Z = seg["scaler"].transform(X)
xy = PCA(2, random_state=42).fit_transform(Z)
p = pd.DataFrame(xy, columns=["PC1", "PC2"]).assign(segment=df.segment_name.values)
st.plotly_chart(px.scatter(p.sample(3000, random_state=1), x="PC1", y="PC2", color="segment", opacity=.6, title="Customers projected to 2D (PCA of the clustering features)"), use_container_width=True)
st.warning(f"Honest read: silhouette score is only **{model_report()['segmentation']['silhouette_at_k']:.2f}** (near 0 = overlapping). "
           "The data has no sharply separated groups, so these segments are a useful way to slice the base, not natural clusters. The overlap in the chart above is real.")
prof = df.groupby("segment_name").agg(customers=("customer_id", "size"), avg_engagement=("engagement_score", "mean"), avg_minutes_per_session=("avg_watch_time_per_session", "mean"),
                                      avg_tenure_years=("customer_tenure_days", lambda s: s.mean() / 365), avg_plan_price=("plan_price", "mean"),
                                      failed_payment_rate=("failed_payment_rate", "mean"), tickets=("support_ticket_count", "mean"), avg_clv=("customer_lifetime_value", "mean"),
                                      observed_churn=("churned", "mean")).round(2).reset_index()
st.dataframe(prof, hide_index=True, use_container_width=True)
for name, d in seg["descriptions"].items():
    st.markdown(f"**{name}**: {d}")
with st.expander("How K was chosen"):
    figure("model_segmentation_k.png")
    st.caption("Silhouette is flat (about 0.12) for every K from 2 to 10, so K = 5 was chosen for business usability, not because the data demands it.")
