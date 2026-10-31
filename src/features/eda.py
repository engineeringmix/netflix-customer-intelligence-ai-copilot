"""24 EDA charts -> reports/figures/eda_*.png.  Run: python -m src.features.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import PROCESSED, FIGURES

sns.set_theme(style="whitegrid", palette="deep")
RED = "#E50914"


def R(n, **kw):
    return pd.read_csv(PROCESSED / f"{n}.csv", **kw)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"eda_{name}.png", dpi=110)
    plt.close()


def bar_rate(df, col, name, title, order=None, rot=0, top=None):
    g = df.groupby(col).churned.agg(["mean", "size"]).reset_index()
    if top:
        g = g.sort_values("mean", ascending=False).head(top)
    g = g.sort_values("mean", ascending=False) if order is None else g.set_index(col).loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=g, x=col, y="mean", color=RED, ax=ax)
    ax.axhline(df.churned.mean(), ls="--", c="k", lw=1, label="overall")
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m, f"{m:.0%}\nn={n}", ha="center", va="bottom", fontsize=7)
    ax.set_title(title); ax.set_ylabel("Churn rate"); ax.set_xlabel(""); ax.legend(); plt.xticks(rotation=rot, ha="right" if rot else "center")
    ax.set_ylim(0, g["mean"].max() * 1.3); save(name)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    f = R("customer_features"); sc = R("customer_scores"); f = f.merge(sc, on="customer_id")
    cust = R("customers", parse_dates=["registration_date"]); view = R("viewing_activity", parse_dates=["viewing_date"])
    pay = R("payments", parse_dates=["payment_date"]); tix = R("support_tickets", parse_dates=["ticket_date"])
    subs = R("subscriptions", parse_dates=["subscription_start_date", "cancellation_date"]); plans = R("subscription_plans")
    fb = R("customer_feedback"); content = R("content"); churn = R("churn_labels", parse_dates=["churn_date"])
    # 01 country
    fig, ax = plt.subplots(figsize=(8, 5)); c = cust.country.value_counts().head(12)
    sns.barplot(x=c.values, y=c.index, color=RED, ax=ax); ax.set_title("Customers by country (top 12)"); save("01_customers_by_country")
    # 02 age
    fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(cust.age, bins=30, color=RED, ax=ax); ax.set_title("Age distribution (after cleaning)"); save("02_age_distribution")
    # 03 channel
    fig, ax = plt.subplots(figsize=(8, 4)); c = cust.acquisition_channel.value_counts()
    sns.barplot(x=c.values, y=c.index, color=RED, ax=ax); ax.set_title("Customers by acquisition channel"); save("03_acquisition_channel")
    # 04 plan distribution + 05 revenue mix
    cur = subs.sort_values("subscription_start_date").groupby("customer_id").tail(1).merge(plans[["plan_id", "plan_name"]], on="plan_id")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cur.plan_name.value_counts().plot.pie(autopct="%1.0f%%", ax=ax[0], colors=["#E50914", "#221f1f", "#b20710"]); ax[0].set_ylabel(""); ax[0].set_title("Plan distribution (current plan)")
    pp = pay[pay.payment_status == "Success"].merge(subs[["subscription_id", "plan_id"]], on="subscription_id").merge(plans[["plan_id", "plan_name"]], on="plan_id")
    rev = pp.groupby("plan_name").amount.sum(); rev.plot.pie(autopct="%1.0f%%", ax=ax[1], colors=["#E50914", "#221f1f", "#b20710"]); ax[1].set_ylabel(""); ax[1].set_title("Revenue mix (successful payments)")
    save("04_05_plan_and_revenue_mix")
    # 06 cancellations over time
    m = churn[churn.churned].groupby(churn.churn_date.dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(9, 4)); m.plot(kind="bar", color=RED, ax=ax); ax.set_title("Cancellations per month"); ax.set_xlabel(""); save("06_cancellations_over_time")
    # 07 churn by plan
    bar_rate(f, "current_plan", "07_churn_by_plan", "Churn rate by current plan", order=["Basic", "Standard", "Premium"])
    # 08 watch time over time
    v = view.groupby(view.viewing_date.dt.to_period("M")).agg(minutes=("watch_duration_minutes", "sum"), customers=("customer_id", "nunique"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)); v.minutes.plot(ax=ax[0], marker="o", color=RED); ax[0].set_title("Total watch minutes per month")
    v.customers.plot(ax=ax[1], marker="o", color="#221f1f"); ax[1].set_title("Monthly active customers"); save("08_watch_time_trend")
    # 09 watch duration dist
    fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(view.watch_duration_minutes, bins=50, color=RED, ax=ax); ax.set_title("Watch duration per session (outliers capped)"); save("09_watch_duration_distribution")
    # 10 completion by content type
    vc = view.merge(content[["content_id", "content_type"]], on="content_id")
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(data=vc, x="content_type", y="completion_percentage", color="#f4a6a8", ax=ax); ax.set_title("Completion % by content type"); save("10_completion_by_content_type")
    # 11 device
    fig, ax = plt.subplots(figsize=(7, 4)); d = view.groupby("device_type").watch_duration_minutes.sum().sort_values()
    d.plot.barh(color=RED, ax=ax); ax.set_title("Total watch minutes by device (cleaned categories)"); save("11_device_usage")
    # 12 churn by country
    bar_rate(f[f.country != "Unknown"], "country", "12_churn_by_country", "Churn rate by country (top 10)", rot=45, top=10)
    # 13 tenure bucket
    f["tenure_bucket"] = pd.cut(f.customer_tenure_days, [0, 365, 730, 1460, 3000], labels=["<1y", "1-2y", "2-4y", "4y+"])
    bar_rate(f, "tenure_bucket", "13_churn_by_tenure", "Churn rate by tenure", order=["<1y", "1-2y", "2-4y", "4y+"])
    # 14 engagement tier
    f["engagement_tier"] = pd.qcut(f.engagement_score, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    bar_rate(f, "engagement_tier", "14_churn_by_engagement", "Churn rate by engagement-score quartile", order=["Q1 low", "Q2", "Q3", "Q4 high"])
    # 15 payment failures
    f["failed_bucket"] = pd.cut(f.failed_payment_count, [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    bar_rate(f, "failed_bucket", "15_churn_by_payment_failures", "Churn rate by # failed payments", order=["0", "1", "2", "3+"])
    # 16 ticket volume
    f["ticket_bucket"] = pd.cut(f.support_ticket_count, [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    bar_rate(f, "ticket_bucket", "16_churn_by_ticket_volume", "Churn rate by # support tickets", order=["0", "1", "2", "3+"])
    # 17 ticket trend
    t = tix.dropna(subset=["ticket_date"]); tm = t.groupby(t.ticket_date.dt.to_period("Q")).size()
    fig, ax = plt.subplots(figsize=(9, 4)); tm.tail(16).plot(ax=ax, marker="o", color=RED); ax.set_title("Support tickets per quarter (valid dates only)"); save("17_ticket_volume_trend")
    # 18 resolution time
    fig, ax = plt.subplots(figsize=(8, 4)); r = tix.groupby("issue_category").resolution_time_hours.mean().sort_values()
    r.plot.barh(color=RED, ax=ax); ax.set_title("Average resolution time (hours) by category"); save("18_resolution_time_by_category")
    # 19 complaint category vs CSAT
    g = tix.groupby("issue_category").agg(n=("ticket_id", "size"), csat=("customer_satisfaction_score", "mean")).sort_values("n")
    fig, ax = plt.subplots(figsize=(8, 4)); ax.barh(g.index, g.n, color=RED); ax.set_title("Complaint volume (bars) with avg CSAT (labels)")
    for i, (n, c) in enumerate(zip(g.n, g.csat)): ax.text(n, i, f" CSAT {c:.2f}", va="center", fontsize=8)
    save("19_complaints_vs_csat")
    # 20 churn reasons
    fig, ax = plt.subplots(figsize=(8, 4.5)); cr = churn.churn_reason.value_counts().sort_values()
    cr.plot.barh(color=RED, ax=ax); ax.set_title("Churn reasons"); save("20_churn_reasons")
    # 21 failed payment reasons
    fig, ax = plt.subplots(figsize=(8, 4)); fr = pay.failed_payment_reason.value_counts().sort_values()
    fr.plot.barh(color="#221f1f", ax=ax); ax.set_title("Failed payment reasons"); save("21_failed_payment_reasons")
    # 22 label noise
    fig, ax = plt.subplots(figsize=(7, 4)); ct = pd.crosstab(fb.rating, fb.sentiment_label, normalize="index")[["Negative", "Neutral", "Positive"]]
    ct.plot.bar(stacked=True, ax=ax, color=["#E50914", "#999", "#2a9d5c"]); ax.set_title("Sentiment label by star rating (label noise check)"); ax.set_ylabel("share"); save("22_sentiment_vs_rating")
    # 23 correlation
    cols = ["customer_tenure_days", "engagement_score", "watch_min_last_3m", "monthly_watch_time_trend", "days_since_last_activity", "login_frequency",
            "failed_payment_rate", "support_ticket_count", "avg_customer_satisfaction", "plan_price", "customer_lifetime_value", "churn_risk_score", "churned"]
    fig, ax = plt.subplots(figsize=(9, 7.5)); sns.heatmap(f[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 7}, ax=ax); ax.set_title("Feature correlations"); save("23_correlation_heatmap")
    # 24 risk distribution
    fig, ax = plt.subplots(figsize=(8, 4)); sns.kdeplot(data=f, x="churn_risk_score", hue="churned", common_norm=False, fill=True, ax=ax); ax.set_title("Model churn-risk score: churners vs retained (out-of-fold)"); save("24_risk_score_distribution")
    print(len(list(FIGURES.glob("eda_*.png"))), "EDA figures")


if __name__ == "__main__":
    main()
