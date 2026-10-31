"""Publish ML outputs to PostgreSQL so SQL, the copilot and the app all read the same numbers.
Creates: customer_features, customer_scores (churn_risk_score, segment, predicted_clv), plus the view v_customer_360.
"""
import pandas as pd
from sqlalchemy import create_engine, text

from src.utils.config import DB_URL, PROCESSED


def main():
    eng = create_engine(DB_URL)
    f = pd.read_csv(PROCESSED / "customer_features.csv").drop(columns=["churned"])
    s = pd.read_csv(PROCESSED / "customer_scores.csv")
    with eng.begin() as c:
        c.execute(text("DROP VIEW IF EXISTS v_customer_360"))
    f.to_sql("customer_features", eng, if_exists="replace", index=False, chunksize=2000)
    s.to_sql("customer_scores", eng, if_exists="replace", index=False)
    with eng.begin() as c:
        c.execute(text("ALTER TABLE customer_features ADD PRIMARY KEY (customer_id)"))
        c.execute(text("ALTER TABLE customer_scores ADD PRIMARY KEY (customer_id)"))
        c.execute(text("""CREATE VIEW v_customer_360 AS
            SELECT f.*, s.churn_risk_score, s.risk_tier, s.segment_name, s.predicted_clv, cl.churned, cl.churn_reason
            FROM customer_features f JOIN customer_scores s USING (customer_id) JOIN churn_labels cl USING (customer_id)"""))
    with eng.begin() as c:   # read-only role used by the copilot (idempotent)
        c.execute(text("DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='ci_readonly') THEN CREATE ROLE ci_readonly LOGIN PASSWORD 'ci_readonly'; END IF; END $$"))
        c.execute(text("GRANT USAGE ON SCHEMA public TO ci_readonly"))
        c.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO ci_readonly"))
        c.execute(text("ALTER ROLE ci_readonly SET statement_timeout = '15s'"))
        c.execute(text("ALTER ROLE ci_readonly SET default_transaction_read_only = on"))
    print("customer_features:", len(f), "customer_scores:", len(s))


if __name__ == "__main__":
    main()
