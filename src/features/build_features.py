"""Customer-level feature table (one row per customer), measured as of AS_OF_DATE.

Run: python -m src.features.build_features  ->  data/processed/customer_features.csv
"""
import numpy as np
import pandas as pd

from src.utils.config import PROCESSED, AS_OF_DATE

ASOF = pd.Timestamp(AS_OF_DATE)


def _read(name, **kw):
    return pd.read_csv(PROCESSED / f"{name}.csv", **kw)


def _trend(minutes_by_month: pd.DataFrame) -> pd.Series:
    """Slope (minutes / month) of a straight line through the last 6 monthly totals (zero-filled)."""
    x = np.arange(minutes_by_month.shape[1], dtype=float)
    x = x - x.mean()
    y = minutes_by_month.values.astype(float)
    return pd.Series((y - y.mean(axis=1, keepdims=True)) @ x / (x @ x), index=minutes_by_month.index)


def build() -> pd.DataFrame:
    cust = _read("customers", parse_dates=["registration_date"]).set_index("customer_id")
    subs = _read("subscriptions", parse_dates=["subscription_start_date"])
    plans = _read("subscription_plans").set_index("plan_id")
    view = _read("viewing_activity", parse_dates=["viewing_date"])
    pay = _read("payments", parse_dates=["payment_date"])
    tix = _read("support_tickets")
    fb = _read("customer_feedback", parse_dates=["feedback_date"])
    content = _read("content").set_index("content_id")
    churn = _read("churn_labels").set_index("customer_id")

    f = pd.DataFrame(index=cust.index)
    # --- demographics / account -------------------------------------------------------------
    f["age"] = cust.age
    f["gender"] = cust.gender
    f["country"] = cust.country
    f["acquisition_channel"] = cust.acquisition_channel
    f["customer_segment"] = cust.customer_segment
    f["customer_tenure_days"] = (ASOF - cust.registration_date).dt.days.clip(lower=0)

    # --- subscription -------------------------------------------------------------------------
    cur = subs.sort_values("subscription_start_date").groupby("customer_id").tail(1).set_index("customer_id")
    f["current_plan"] = cur.plan_id.map(plans.plan_name)
    f["plan_price"] = cur.plan_id.map(plans.monthly_price)
    f["auto_renew"] = cur.auto_renew.astype(int)
    f["plan_changes"] = subs.groupby("customer_id").size().sub(1).reindex(f.index).fillna(0)

    # --- engagement ----------------------------------------------------------------------------
    g = view.groupby("customer_id")
    f["total_sessions"] = g.size().reindex(f.index).fillna(0)
    f["total_watch_minutes"] = g.watch_duration_minutes.sum().reindex(f.index).fillna(0)
    f["avg_watch_time_per_session"] = g.watch_duration_minutes.mean().reindex(f.index)
    f["avg_completion_pct"] = g.completion_percentage.mean().reindex(f.index)
    last = g.viewing_date.max().reindex(f.index)
    f["days_since_last_activity"] = (ASOF - last).dt.days.fillna(365)
    view["month"] = view.viewing_date.dt.to_period("M")
    months = pd.period_range(end=ASOF.to_period("M"), periods=6, freq="M")
    mm = (view[view.month.isin(months)].pivot_table(index="customer_id", columns="month",
                                                   values="watch_duration_minutes", aggfunc="sum")
          .reindex(index=f.index, columns=months).fillna(0))
    f["monthly_watch_time_trend"] = _trend(mm)
    f["watch_min_last_3m"] = mm.iloc[:, 3:].sum(axis=1)
    f["watch_min_prev_3m"] = mm.iloc[:, :3].sum(axis=1)
    f["watch_trend_log_ratio"] = np.log1p(f.watch_min_last_3m) - np.log1p(f.watch_min_prev_3m)  # <0 = declining
    active_months = view.groupby("customer_id").month.nunique().reindex(f.index).fillna(0)
    f["login_frequency"] = (f.total_sessions / active_months.replace(0, np.nan)).fillna(0)
    f["distinct_titles"] = g.content_id.nunique().reindex(f.index).fillna(0)
    f["distinct_devices"] = g.device_type.nunique().reindex(f.index).fillna(0)
    top_dev = g.device_type.agg(lambda s: s.mode().iloc[0]).reindex(f.index)
    f["primary_device"] = top_dev.fillna("None")
    view = view.join(content[["genre"]], on="content_id")
    f["distinct_genres"] = view.groupby("customer_id").genre.nunique().reindex(f.index).fillna(0)

    # --- billing -------------------------------------------------------------------------------
    pg = pay.groupby("customer_id")
    f["payment_attempts"] = pg.size().reindex(f.index).fillna(0)
    f["failed_payment_count"] = pay[pay.payment_status == "Failed"].groupby("customer_id").size().reindex(f.index).fillna(0)
    f["failed_payment_rate"] = (f.failed_payment_count / f.payment_attempts.replace(0, np.nan)).fillna(0)
    recent = pay[(pay.payment_status == "Failed") & (pay.payment_date > ASOF - pd.Timedelta(days=90))]
    f["failed_payments_90d"] = recent.groupby("customer_id").size().reindex(f.index).fillna(0)
    f["payment_methods_used"] = pg.payment_method.nunique().reindex(f.index).fillna(0)
    f["customer_lifetime_value"] = pay[pay.payment_status == "Success"].groupby("customer_id").amount.sum().reindex(f.index).fillna(0)

    # --- support & feedback --------------------------------------------------------------------
    tg = tix.groupby("customer_id")
    f["support_ticket_count"] = tg.size().reindex(f.index).fillna(0)
    f["avg_customer_satisfaction"] = tg.customer_satisfaction_score.mean().reindex(f.index)
    f["escalated_tickets"] = tix[tix.ticket_status == "Escalated"].groupby("customer_id").size().reindex(f.index).fillna(0)
    f["pending_tickets"] = tix[tix.ticket_status == "Pending"].groupby("customer_id").size().reindex(f.index).fillna(0)
    f["avg_resolution_hours"] = tg.resolution_time_hours.mean().reindex(f.index)
    f["billing_ticket_count"] = tix[tix.issue_category == "Billing & Payments"].groupby("customer_id").size().reindex(f.index).fillna(0)
    fg = fb.groupby("customer_id")
    f["avg_feedback_rating"] = fg.rating.mean().reindex(f.index)
    f["negative_feedback_count"] = fb[fb.rating <= 2].groupby("customer_id").size().reindex(f.index).fillna(0)

    # --- composite engagement score (0-100): recency 40%, frequency 30%, watch depth 30% ----------
    rec = 1 - f.days_since_last_activity.clip(upper=90) / 90
    freq = f.total_sessions.rank(pct=True)
    depth = f.avg_watch_time_per_session.fillna(0).rank(pct=True)
    f["engagement_score"] = (100 * (0.4 * rec + 0.3 * freq + 0.3 * depth)).round(1)

    f["churned"] = churn.churned.reindex(f.index).astype(int)
    f.index.name = "customer_id"
    return f.reset_index()


def main():
    df = build()
    df.to_csv(PROCESSED / "customer_features.csv", index=False)
    print(df.shape)
    print(df.isna().sum()[lambda s: s > 0])
    print(df.describe().T[["mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
