import pandas as pd
from sqlalchemy import create_engine, text
from src.utils.config import DB_URL, PROCESSED


def scalar(sql):
    with create_engine(DB_URL).connect() as c:
        return c.execute(text(sql)).scalar()


def test_row_counts():
    for t, n in {"customers": 8000, "subscriptions": 8875, "viewing_activity": 66422, "payments": 110735, "support_tickets": 6120,
                 "customer_feedback": 5046, "churn_labels": 8000, "content": 500, "subscription_plans": 3}.items():
        assert scalar(f"SELECT COUNT(*) FROM {t}") == n, t


def test_cleaning_invariants():
    assert scalar("SELECT COUNT(*) FROM customers WHERE country IS NULL") == 0
    assert scalar("SELECT COUNT(DISTINCT country) FROM customers WHERE country <> 'Unknown'") == 20
    assert scalar("SELECT COUNT(DISTINCT device_type) FROM viewing_activity") <= 6
    assert scalar("SELECT COUNT(*) FROM viewing_activity WHERE watch_duration_minutes > 300") == 0
    assert scalar("SELECT COUNT(*) FROM customers WHERE age < 13 OR age > 100") == 0
    assert scalar("SELECT COUNT(*) FROM payments WHERE subscription_id IS NULL") == 0
    assert scalar("SELECT COUNT(*) FROM support_tickets WHERE ticket_date IS NULL") == 61


def test_payments_inside_subscription_window():
    bad = scalar("""SELECT COUNT(*) FROM payments p JOIN subscriptions s USING (subscription_id)
                    WHERE p.payment_date < s.subscription_start_date OR (s.subscription_end_date IS NOT NULL AND p.payment_date > s.subscription_end_date)""")
    assert bad == 0


def test_ml_tables_consistent():
    assert scalar("SELECT COUNT(*) FROM customer_scores") == 8000
    assert scalar("SELECT COUNT(*) FROM customer_scores WHERE churn_risk_score NOT BETWEEN 0 AND 1") == 0
    f = pd.read_csv(PROCESSED / "customer_features.csv")
    assert f.customer_id.is_unique and len(f) == 8000


def test_no_target_leakage_columns_in_churn_model():
    from src.models.train import CAT, CHURN_NUM
    banned = {"churned", "customer_lifetime_value", "churn_reason", "churn_date", "cancellation_date", "cancellation_reason", "subscription_status", "gender"}
    assert not banned & set(CAT + CHURN_NUM)
