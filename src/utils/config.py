"""Central configuration: paths, DB URL, dataset as-of date."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
LOG_DIR = ROOT / "reports"

# The dataset's viewing window ends 2026-06-30. All "recency" metrics are measured
# against this date (NOT today's date), otherwise everything looks stale.
AS_OF_DATE = "2026-06-30"

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://ci:ci@localhost:5432/netflix_ci")

TABLE_LOAD_ORDER = [
    "customers", "subscription_plans", "subscriptions", "content",
    "viewing_activity", "payments", "support_tickets", "customer_feedback", "churn_labels",
]

# Copilot connects with a SELECT-only role (defence in depth on top of the SQL validator).
DB_URL_RO = os.getenv("DATABASE_URL_RO", "postgresql+psycopg2://ci_readonly:ci_readonly@localhost:5432/netflix_ci")
QUERY_LOG = REPORTS / "copilot_query_log.jsonl"
COPILOT_MODEL = os.getenv("COPILOT_MODEL", "claude-sonnet-5-5")
MAX_ROWS = int(os.getenv("COPILOT_MAX_ROWS", "500"))
QUERY_TIMEOUT_S = int(os.getenv("COPILOT_TIMEOUT_S", "15"))
