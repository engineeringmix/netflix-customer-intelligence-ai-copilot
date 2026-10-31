# Final Project Report: Netflix Customer Intelligence, Churn Prediction & AI Analytics Copilot

Independent portfolio project modeled on a subscription-streaming business. Data is synthetic. Data as-of date: 2026-06-30.

## 1. Summary

| Item | Result |
|---|---|
| Database | PostgreSQL 16, 9 source tables + 2 ML tables + 1 view, PKs/FKs, 16 indexes; 66,422 viewing and 110,735 payment rows |
| SQL | 37/37 business questions run and validated (`reports/sql_run_summary.csv`) |
| EDA | 24 charts (`reports/figures/eda_*.png`) |
| Churn model | XGBoost, test ROC-AUC 0.65, CV AUC 0.661 +/- 0.014; top-risk decile churns at 45% vs 25% base rate |
| Segmentation | K-Means, K = 5, silhouette 0.12 (no natural clusters) |
| CLV model | Random Forest, test R2 0.98, MAE about $10 (largely arithmetic, see 6.3) |
| Copilot | 7 capabilities (incl. analysis insights), SQL security layer, number-grounding guard |
| App | 7 Streamlit pages, all render in automated tests |
| Tests | 44 pytest tests, all passing |

**Not verified:** LLM-written answers with a real model. No API key was available; that path was tested only with a scripted mock. Offline (rule-based) mode is what was exercised on real data.

## 2. Business questions and what the data says

- **Overall churn is 24.9%** (1,991 of 8,000).
- **Where churn differs is small.** Plan: Basic 26.3%, Standard 24.6%, Premium 22.5%. Channel: 23.6% (Partner Bundle) to 25.9% (Paid Social). Country: South Korea 29.3% and Germany 27.9% highest. With 200-400 customers per country these gaps are mostly within normal variation. **Do not build a strategy on geography or channel.**
- **Behaviour is where the signal is**, and it is modest: the best single feature (watch minutes in the last 3 months) has univariate AUC 0.61; days since last activity 0.60; failed-payment rate 0.58. Customers with auto-renew off churn at 28.7% vs 24.0%.
- **Stated reasons** (churned customers): Too expensive 365 (18%), Found alternative platform 284 (14%), Not enough content 217 (11%), No longer needed 212 (11%).
- **Plan migration (Q34):** 875 customers changed plan; 35% of those "changes" keep the same plan, 26% are upgrades, 39% downgrades. Churn: upgraders 27.7%, downgraders 26.5%, same-tier 27.9%: no meaningful difference at these sample sizes.
- **Value concentration (Q26):** top spend quartile averages $298 lifetime, bottom $74.
- **Revenue (Q02):** Standard $596k, Premium $480k, Basic $320k all-time successful payments.

## 3. Data cleaning (`reports/cleaning_log.json`)

48 exact duplicate customers removed; 83 country spellings -> 20; 409 missing countries filled from the customer's modal login country; 21 impossible ages nulled and 342 ages imputed by segment median; 664 missing + 24 spellings of device -> 5 canonical (missing filled by the customer's modal device); 531 missing completion % filled by content-type median; 161 watch/session outliers (500-900 min) capped; 6,068 sessions shorter than their watch time fixed; 1,112 anomalous payment amounts replaced with the subscription price (raw kept); 61 invalid ticket dates set to NULL; 110,735/110,735 payments linked to a subscription by customer and date (every payment falls inside its subscription window, tested); sentiment labels re-derived from ratings (263 disagreed). Blank CSAT/resolution on open tickets was **left blank deliberately**.

## 4. Models

### 4.1 Churn (classification)
Four candidates, 5-fold CV on an 80/20 stratified split, class imbalance handled (`class_weight` / `scale_pos_weight`):

| Model | CV AUC | Test AUC | Precision @0.5 | Recall @0.5 |
|---|---|---|---|---|
| Random Forest | 0.662 | 0.651 | 0.366 | 0.520 |
| XGBoost | 0.661 | 0.652 | 0.349 | 0.593 |
| Gradient Boosting | 0.659 | 0.658 | 0.546 | 0.090 |
| Logistic Regression | 0.658 | 0.655 | 0.344 | 0.606 |

**All four are statistically tied.** XGBoost was selected by an explicit rule (within one CV standard deviation of the best) because exact TreeSHAP gives per-customer explanations from the same model that produces the score. Raw probabilities are distorted by class weighting, so scores are isotonic-calibrated on out-of-fold predictions; every customer's stored score comes from a model that never saw that customer's label.

**Precision/recall tradeoff.** Under my *assumed* economics (offer costs $5; 30% of contacted churners accept and are worth about $80 -> $24 expected), contacting pays off when P(churn) > 21%. So tiers are Low < 21%, Medium 21-40%, High >= 40%. Observed churn in those tiers: 15%, 28%, 44% (3,519 / 3,234 / 1,247 customers). The profit-optimal cut-off favours recall (about 57%) over precision (about 37%). **The $5/$24 figures are assumptions; replace them with real campaign data.**

**What this model is.** A ranking tool: contacting the top decile reaches churners at 1.8x the base rate. It is not a reliable individual predictor.

### 4.2 Segmentation
Silhouette is 0.117-0.127 for every K from 2 to 10 and the elbow is flat: **the data has no real clusters.** K = 5 was chosen for usability. Names (assigned by deterministic centroid rules): Premium Loyalists, Disengaged Drifters (lowest engagement, highest churn), Support-Heavy Regulars, New & Highly Engaged, Budget Basic Subscribers.

### 4.3 Customer lifetime value (regression)
Random Forest R2 0.98, MAE $10.1, RMSE $13.7 (mean CLV $175); XGBoost nearly identical; Linear Regression R2 0.83. **This is mostly arithmetic**: CLV is about months subscribed x plan price, both of which are features, and payments are capped at 15 per customer. Do not present R2 0.98 as a modelling achievement.

## 5. The copilot

Security (verified by tests): only SELECT; 23 blocked keywords or patterns; multiple statements rejected; unknown tables/columns rejected using the live schema; all 37 prepared queries pass the validator; execution via a SELECT-only role in a read-only transaction (a DELETE through that role fails, tested); 15 s timeout; 500-row cap; every question/SQL/status logged.

Hallucination control: LLM prose is checked by `grounding.unverified_numbers`; one with a number absent from the query results is discarded (tested with a deliberately lying mock). The root-cause agent answers "why did churn increase in X?" with a fixed query sequence and a two-proportion test; in the sample runs, month-to-month changes in single countries were within normal noise and the agent says so, including when the premise is false.

Analysis insights: the copilot answers questions about this project's own findings and attaches them as caveats elsewhere (billing coverage on revenue answers, tier track record on customer scores, a limits section in reports). The findings are computed from the data and model outputs, not typed in.

Retention rules combine risk tier, value tier (CLV >= 75th percentile or Premium plan), billing failures, open/escalated tickets, auto-renew, segment, and plan. Service recovery is placed before any discount for unhappy customers.

## 6. Limitations (what a reviewer should challenge)

1. **Leakage / time structure.** Churn dates fall inside the observation window and 36% of churners still have sessions after churning, so features are not strictly "before churn". Metrics measure association with the label, not a true forward-looking forecast. A production design would snapshot features at a fixed date before an outcome window.
2. **Weak signal.** AUC 0.65 is the honest ceiling here; no feature exceeds 0.61 alone.
3. **Offline copilot is limited.** Without an LLM key, free-form SQL falls back to the 37 prepared analyses; questions outside them are refused rather than guessed.
4. **RAG is shallow on this data.** About 45 distinct complaint strings exist across 11,166 records, and embeddings are TF-IDF/LSA with a lexical gate, not neural. It counts themes well; it does not discover nuance.
5. **Dataset coverage gaps.** Payments/registrations end 2026-05-31, so June revenue and new-customer counts are zero; churn starts Oct 2025 (Jul-Sep show 0%); Q20 cohort retention is inflated for old cohorts for the same reason.
6. **Pricing field inconsistency** (`subscriptions.monthly_price` constant across plan changes) means Q29 treats tier changes as the price event.
7. **Not delivered:** optional Model 4 (CSAT prediction), SHAP library (native TreeSHAP used), Docker, conversation memory, demo video, live-LLM validation.

## 7. Resume wording that matches what was actually built

- Built a PostgreSQL warehouse (9 source tables + ML tables, FKs, 16 indexes), cleaned seven classes of data defects, and answered 37 analytical SQL questions (window functions, CTEs, cohorts) over about 200,000 records.
- Trained and honestly evaluated churn (XGBoost, calibrated, ROC-AUC 0.65, top-decile lift 1.8x), K-Means segmentation and CLV regression models; documented that the data has weak predictive signal.
- Built a six-capability LLM analytics copilot (text-to-SQL with a validation layer and read-only DB role, FAISS RAG, TreeSHAP explainer, report generator, root-cause agent) with an automated check that blocks numbers not returned by SQL; shipped as a 7-page Streamlit app with 39 automated tests.

Avoid on a resume: "production-grade", "improved churn identification" (nothing was measured against a baseline process), and R2 0.98 as a headline.
