# Netflix Customer Intelligence, Churn Prediction & AI Analytics Copilot

An independent, portfolio-grade analytics-engineering project modeled on a subscription-streaming business (synthetic data):
PostgreSQL warehouse -> 37 SQL analyses -> feature engineering -> three ML models -> a seven-capability AI copilot -> a 7-page Streamlit app.

> **Read this first (honest status).** Everything below was run and tested end to end **except the live-LLM path**: no API key was available while building, so LLM-written answers were tested only with a scripted mock model. Without a key the copilot runs in a rule-based *offline mode* (prepared analyses, theme counting, templated explanations). The churn model is **weak (ROC-AUC about 0.65)** because the dataset is deliberately noisy; it ranks customers usefully but cannot call individuals reliably. Details in `docs/PROJECT_REPORT.md`.

## Quick start

```bash
pip install -r requirements.txt
# 1) PostgreSQL 14+ running, then (once):
#    CREATE USER ci WITH PASSWORD 'ci' SUPERUSER; CREATE DATABASE netflix_ci OWNER ci;
#    (the read-only role `ci_readonly` is created and granted automatically by load_ml_tables.py)
./run_all.sh                       # clean -> load -> features -> models -> SQL -> charts -> notebooks -> tests (~6 min)
streamlit run app/dashboard.py     # the 7-page app (pages appear in the sidebar)
export ANTHROPIC_API_KEY=...       # optional, enables free-form AI answers
```

## What is where

| Path | What |
|---|---|
| `data/raw/` | the 9 CSVs exactly as provided |
| `data/processed/` | cleaned tables, `customer_features.csv` (41 cols), `customer_scores.csv` |
| `database/schema.sql`, `load_data.py` | PostgreSQL schema (PKs, FKs, 16 indexes) and COPY loader |
| `database/sql_business_questions.sql` | the 37 queries, all tested (`reports/sql_run_summary.csv`, results in `reports/sql_results/`) |
| `src/data_processing/clean.py` | every cleaning decision, logged to `reports/cleaning_log.json` |
| `src/features/` | `build_features.py` (customer table), `eda.py` (24 charts) |
| `src/models/train.py` | churn, segmentation, CLV; writes `models/` and `reports/model_report.json` |
| `src/llm/` | the copilot: `sql_guard`, `text2sql`, `rag`, `explainer` (+ retention), `reports`, `rca`, `grounding`, `copilot` (router) |
| `app/` | Streamlit: `dashboard.py` + `pages/2..7` |
| `notebooks/` | 01 cleaning, 02 EDA, 03 features, 04 churn, 05 segmentation, 06 CLV (executed, outputs saved) |
| `tests/` | 44 pytest tests (security, grounding, pipeline integrity, every app page renders) |
| `docs/PROJECT_REPORT.md` | final report, findings and limitations |

## The seven copilot capabilities

1. **Text-to-SQL**: LLM writes SQL -> `sql_guard` (SELECT only; INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE and 17 more keywords and patterns blocked; every table and column checked against the live schema; one statement only) -> executed by a **SELECT-only database role** in a read-only transaction with a 15 s timeout and 500-row cap -> every query logged to `reports/copilot_query_log.jsonl`.
2. **RAG** over ticket + feedback text (FAISS; TF-IDF/LSA embeddings, no download needed). Summaries state how many records they rest on.
3. **Prediction explainer**: exact TreeSHAP contributions from XGBoost turned into plain language. Demographics are never shown as "reasons".
4. **Report generator** (weekly/monthly/quarterly): facts from SQL are printed separately from the interpretation; data-coverage warnings are added automatically.
5. **Root-cause agent**: fixed investigation sequence, every number from a logged query, a significance test, and it tells you when the premise ("churn increased") is false.
6. **Retention recommender**: auditable rules (risk x value x billing x support) decide the action; the LLM only drafts wording.

7. **Analysis insights** (added): the copilot knows this project's own findings (churn drivers, model quality, segments, support, data limits). They are recomputed from the data and model outputs by `python -m src.llm.insights` (saved to `reports/analysis_insights.json`), answer questions like "what drives churn?" or "how reliable is the model?", and are attached as caveats to other answers: SQL answers about revenue note that billing ends 2026-05-31, customer explanations and retention advice show the observed churn rate of the customer's risk tier, and reports include a "known limits" section.

All LLM prose passes through `grounding.unverified_numbers`: if a number is not in the query results, the text is rejected and a deterministic summary is shown instead.

## Known data quirks (you will meet them)

- Viewing data covers 2025-07-01..2026-06-30; **payments and registrations end 2026-05-31**; churn dates start 2025-10-01. So June has no billing, Jul-Sep 2025 show 0% churn, and "as-of date" is fixed at 2026-06-30 (not today).
- `subscriptions.monthly_price` never changes between a customer's two rows, and 35% of "Plan Changed" rows keep the same plan. Real price changes exist only as tier changes (Q29, Q34).
- Ticket and feedback text has only ~45 distinct strings, so RAG is theme counting, not rich semantic search.
- Churn happens inside the viewing window and 36% of churners still have sessions after churning, so features cannot be strictly "pre-churn". Treat model results as associations.
