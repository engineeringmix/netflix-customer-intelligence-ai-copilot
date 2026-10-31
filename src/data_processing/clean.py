"""Data cleaning pipeline: raw CSVs -> data/processed/*.csv + a cleaning log.

Run:  python -m src.data_processing.clean
Every decision is recorded in reports/cleaning_log.json so it can be audited.
"""
import json
import numpy as np
import pandas as pd

from src.utils.config import RAW, PROCESSED, REPORTS

COUNTRY_MAP = {
    "united states": "United States", "usa": "United States", "us": "United States", "u.s.a": "United States",
    "india": "India", "in": "India", "bharat": "India",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom",
    "canada": "Canada", "ca": "Canada", "brazil": "Brazil", "br": "Brazil",
    "germany": "Germany", "de": "Germany", "france": "France", "fr": "France",
    "mexico": "Mexico", "mx": "Mexico", "japan": "Japan", "jp": "Japan",
    "philippines": "Philippines", "ph": "Philippines", "indonesia": "Indonesia", "id": "Indonesia",
    "italy": "Italy", "it": "Italy", "nigeria": "Nigeria", "ng": "Nigeria",
    "spain": "Spain", "es": "Spain", "south africa": "South Africa", "za": "South Africa",
    "australia": "Australia", "au": "Australia", "south korea": "South Korea", "kr": "South Korea", "s. korea": "South Korea",
    "argentina": "Argentina", "ar": "Argentina", "poland": "Poland", "pl": "Poland",
    "united arab emirates": "United Arab Emirates", "uae": "United Arab Emirates", "u.a.e": "United Arab Emirates",
}
DEVICE_MAP = {
    "smart tv": "Smart TV", "smarttv": "Smart TV", "tv": "Smart TV",
    "mobile": "Mobile", "smartphone": "Mobile", "phone": "Mobile",
    "tablet": "Tablet", "ipad": "Tablet",
    "laptop": "Laptop", "desktop": "Laptop", "pc": "Laptop",
    "gaming console": "Gaming Console", "console": "Gaming Console", "playstation/xbox": "Gaming Console",
}


def std_country(s: pd.Series) -> pd.Series:
    key = s.astype("string").str.strip().str.lower()
    out = key.map(COUNTRY_MAP)
    unmapped = key[out.isna() & key.notna()].unique()
    assert len(unmapped) == 0, f"Unmapped country values: {unmapped}"
    return out


def clean_customers(log):
    df = pd.read_csv(RAW / "customers.csv")
    log["customers_raw_rows"] = len(df)
    df = df.drop_duplicates()
    assert df.customer_id.is_unique, "duplicates were not exact copies"
    log["customers_exact_duplicates_removed"] = log["customers_raw_rows"] - len(df)

    df["country"] = std_country(df["country"])
    # Missing country: fill from the customer's most common login country (92% agreement when both exist).
    v = pd.read_csv(RAW / "viewing_activity.csv", usecols=["customer_id", "login_location"])
    v["login_location"] = std_country(v["login_location"])
    modal = v.groupby("customer_id").login_location.agg(lambda x: x.mode().iloc[0])
    miss = df.country.isna()
    df.loc[miss, "country"] = df.loc[miss, "customer_id"].map(modal)
    log["country_missing_before"] = int(miss.sum())
    log["country_filled_from_login_location"] = int(miss.sum() - df.country.isna().sum())
    df["country"] = df["country"].fillna("Unknown")

    df["is_age_outlier"] = (df.age < 13) | (df.age > 100)
    log["age_outliers_set_to_null"] = int(df.is_age_outlier.sum())
    df.loc[df.is_age_outlier, "age"] = np.nan
    df["age_was_missing"] = df.age.isna()
    seg_med = df.groupby("customer_segment").age.transform("median")
    df["age"] = df["age"].fillna(seg_med).round().astype(int)
    log["age_imputed_with_segment_median"] = int(df.age_was_missing.sum())

    mode_lang = df.dropna(subset=["preferred_language"]).groupby("country").preferred_language.agg(lambda x: x.mode().iloc[0])
    miss = df.preferred_language.isna()
    df.loc[miss, "preferred_language"] = df.loc[miss, "country"].map(mode_lang)
    df["preferred_language"] = df["preferred_language"].fillna("Unknown")
    log["language_missing_filled_by_country_mode"] = int(miss.sum())
    df["city"] = df["city"].fillna("Unknown")
    df["state_or_region"] = None  # shipped 100% blank
    df["registration_date"] = pd.to_datetime(df.registration_date)
    return df.sort_values("customer_id").reset_index(drop=True)


def clean_viewing(log, customers):
    df = pd.read_csv(RAW / "viewing_activity.csv", parse_dates=["viewing_date"])
    df["login_location"] = std_country(df["login_location"])
    raw_dev = df.device_type.astype("string").str.strip().str.lower()
    df["device_type"] = raw_dev.map(DEVICE_MAP)
    assert df.device_type[raw_dev.notna()].notna().all()
    log["device_missing"] = int(df.device_type.isna().sum())
    modal = df.dropna(subset=["device_type"]).groupby("customer_id").device_type.agg(lambda x: x.mode().iloc[0])
    miss = df.device_type.isna()
    df.loc[miss, "device_type"] = df.loc[miss, "customer_id"].map(modal)
    df["device_type"] = df["device_type"].fillna("Unknown")

    # Outliers: values of 500-900 minutes are clearly injected (99th pct = 108). Cap at the 99.5th
    # percentile of the NON-outlier values and keep a flag.
    for col in ["watch_duration_minutes", "session_duration_minutes"]:
        out = df[col] > 300
        cap = df.loc[~out, col].quantile(0.995)
        df[f"is_{col.split('_')[0]}_outlier"] = out
        log[f"{col}_outliers_capped"] = int(out.sum())
        df.loc[out, col] = int(round(cap))
    # session length must be >= watch time
    bad = df.session_duration_minutes < df.watch_duration_minutes
    log["sessions_shorter_than_watch_time_fixed"] = int(bad.sum())
    df.loc[bad, "session_duration_minutes"] = df.loc[bad, "watch_duration_minutes"]

    c = pd.read_csv(RAW / "content.csv")[["content_id", "content_type"]]
    df = df.merge(c, on="content_id", how="left")
    log["completion_missing"] = int(df.completion_percentage.isna().sum())
    df["completion_percentage"] = df.completion_percentage.fillna(
        df.groupby("content_type").completion_percentage.transform("median")).round(2)
    return df.drop(columns="content_type")


def clean_payments_and_link(log, subs):
    p = pd.read_csv(RAW / "payments.csv", parse_dates=["payment_date"]).drop(columns="subscription_id")
    p["amount_raw"] = p["amount"]
    # Resolve subscription_id: latest subscription of the same customer that started on/before the payment date.
    s = subs.sort_values("subscription_start_date")[["customer_id", "subscription_id", "subscription_start_date", "monthly_price"]]
    m = pd.merge_asof(p.sort_values("payment_date"), s, left_on="payment_date", right_on="subscription_start_date", by="customer_id")
    m = m.sort_values("payment_id").reset_index(drop=True)
    log["payments_linked_to_subscription"] = int(m.subscription_id.notna().sum())
    log["payments_unlinked"] = int(m.subscription_id.isna().sum())
    # Anomalous amounts: should equal the subscription price (99.0% do). Everything else = proration/refund/entry error.
    m["is_amount_anomaly"] = (m.amount - m.monthly_price).abs() > 0.01
    log["payment_amount_anomalies_replaced_with_sub_price"] = int(m.is_amount_anomaly.sum())
    m.loc[m.is_amount_anomaly, "amount"] = m.loc[m.is_amount_anomaly, "monthly_price"]
    return m.drop(columns=["subscription_start_date", "monthly_price"])


def clean_tickets(log):
    t = pd.read_csv(RAW / "support_tickets.csv")
    d = pd.to_datetime(t.ticket_date, errors="coerce")
    log["tickets_invalid_dates_set_to_null"] = int(d.isna().sum())
    t["ticket_date_valid"] = d.notna()
    t["ticket_date"] = d
    # CSAT / resolution blank for open tickets = expected, NOT imputed.
    t["customer_satisfaction_score"] = t.customer_satisfaction_score.astype("Int64")
    return t


def clean_feedback(log):
    f = pd.read_csv(RAW / "customer_feedback.csv", parse_dates=["feedback_date"])
    f["sentiment_validated"] = np.select([f.rating >= 4, f.rating == 3], ["Positive", "Neutral"], "Negative")
    f["sentiment_mismatch"] = f.sentiment_label != f.sentiment_validated
    log["feedback_sentiment_labels_disagreeing_with_rating"] = int(f.sentiment_mismatch.sum())
    return f


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    log = {}
    customers = clean_customers(log)
    plans = pd.read_csv(RAW / "subscription_plans.csv")
    subs = pd.read_csv(RAW / "subscriptions.csv", parse_dates=["subscription_start_date", "subscription_end_date", "cancellation_date"])
    merged = subs.merge(plans[["plan_id", "monthly_price"]], on="plan_id", suffixes=("", "_list"))
    log["subscriptions_price_differs_from_plan_list_price"] = int((merged.monthly_price != merged.monthly_price_list).sum())
    content = pd.read_csv(RAW / "content.csv")
    churn = pd.read_csv(RAW / "churn_labels.csv", parse_dates=["churn_date"])
    churn["churned"] = churn.churned.astype(bool)

    out = {
        "customers": customers, "subscription_plans": plans, "subscriptions": subs, "content": content,
        "viewing_activity": clean_viewing(log, customers), "payments": clean_payments_and_link(log, subs),
        "support_tickets": clean_tickets(log), "customer_feedback": clean_feedback(log), "churn_labels": churn,
    }
    for name, df in out.items():
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
        log[f"rows_{name}"] = len(df)
    # referential integrity
    ids = set(customers.customer_id)
    for t in ["subscriptions", "viewing_activity", "payments", "support_tickets", "customer_feedback", "churn_labels"]:
        assert set(out[t].customer_id) <= ids, f"orphans in {t}"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
