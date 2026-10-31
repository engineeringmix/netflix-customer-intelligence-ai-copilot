"""Generates and executes the six notebooks (thin, readable wrappers over src/). Run: python scripts/build_notebooks.py"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
HEAD = "import sys, json; sys.path.insert(0, '..')\nimport pandas as pd, numpy as np\nfrom IPython.display import Image, display, Markdown\nfrom src.utils.config import *\npd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)"


def make(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(HEAD)]
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells.insert(0, nb.cells.pop(0))
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
    nbf.write(nb, NB / name)
    print("built", name)


make("01_data_cleaning.ipynb", [
 ("md", "# 01 Data cleaning\nEvery cleaning decision is made in `src/data_processing/clean.py` and logged. This notebook shows the **problem -> fix** evidence for each item in the PRD checklist (section 9.1)."),
 ("code", "from src.data_processing import clean\nlog = {}\nraw = pd.read_csv(RAW/'customers.csv'); cl = pd.read_csv(PROCESSED/'customers.csv')\nprint('customers raw:', len(raw), '-> clean:', len(cl), '| exact duplicates removed:', len(raw)-len(raw.drop_duplicates()))\nprint('country spellings raw:', raw.country.nunique(), '-> clean:', cl.country.nunique())\nprint(raw.country.value_counts().head(8).to_dict())"),
 ("md", "## Country: 83 spellings -> 20 canonical countries\n`INDIA / india / IN / Bharat` all become `India`. 409 missing countries were **filled from each customer's most common login country** (the two agree 92% of the time where both exist), not guessed."),
 ("code", "print(json.dumps(json.load(open(REPORTS/'cleaning_log.json')), indent=1))"),
 ("md", "## Outliers and missing values: strategy per column\n| Column | Problem | Strategy | Why |\n|---|---|---|---|\n| age | 322 missing, 21 impossible (2, 5, 130, 145) | outliers -> null, then impute **segment median**; flag columns kept | students are younger than corporate users, so a global median would bias |\n| country | 409 missing | modal login country | behavioural evidence beats a constant |\n| city | 484 missing | 'Unknown' | no reliable source |\n| preferred_language | 254 missing | modal language of the customer's country | |\n| device_type | 664 missing + 24 spellings | canonical set of 5; missing -> customer's modal device | |\n| completion_percentage | 531 missing | median by content type | |\n| watch/session minutes | 161 values of 500-900 min | capped at the 99.5th pct of normal values; flag kept | genuine binges do not last 15 hours |\n| session < watch time | 6,068 rows (9%) | session raised to watch time | a session cannot be shorter than what was watched |\n| payments.amount | 1,112 anomalies (0, x2, x0.5, x11...) | replaced by the subscription's price; raw kept in `amount_raw` | 99% of payments equal the subscription price |\n| ticket_date | 61 invalid ('2025-13-45') | set to NULL, never guessed | excluded from time analyses |\n| CSAT / resolution time blank | 1,528 open tickets | **left blank on purpose** | blank = not yet resolved, not missing data |\n| sentiment_label | ~5% contradict the star rating | `sentiment_validated` derived from rating (>=4 pos, 3 neu, <=2 neg) | original kept |"),
 ("code", "v = pd.read_csv(RAW/'viewing_activity.csv'); vc = pd.read_csv(PROCESSED/'viewing_activity.csv')\nprint('device values raw:', v.device_type.nunique(), '-> clean:', vc.device_type.nunique())\nprint('max watch minutes raw:', v.watch_duration_minutes.max(), '-> clean:', vc.watch_duration_minutes.max())\np = pd.read_csv(PROCESSED/'payments.csv')\nprint('payments linked to a subscription:', p.subscription_id.notna().mean())\nprint(p[p.is_amount_anomaly][['payment_id','amount_raw','amount']].head())"),
 ("md", "## Finding that changes how the data must be read\n- `subscriptions.monthly_price` **never differs between a customer's two rows**, so it is not the price at the time of each segment (the dictionary says it is). Real price changes only exist as plan changes (see SQL Q29).\n- Churn dates (Oct-25 to Jun-26) sit **inside** the viewing window, and 36% of churners still have sessions after their churn date. Features therefore cannot be made perfectly 'before churn'. See the final report, section Limitations."),
])

make("02_exploratory_analysis.ipynb", [
 ("md", "# 02 Exploratory analysis (24 charts)\nCharts are produced by `src/features/eda.py` and saved in `reports/figures/`."),
 ("code", "from pathlib import Path\nfor p in sorted(FIGURES.glob('eda_*.png')):\n    display(Markdown(f'**{p.stem}**')); display(Image(filename=str(p), width=760))"),
 ("md", "## What the charts say (read with the model caveats in mind)\n- Churn is only mildly different across countries (25-29%), plans (Premium 22% vs Basic 26%) and channels. **Geography and channel are weak levers.**\n- The strongest raw signals are behavioural: lower recent watch time, longer inactivity, failed payments.\n- Auto-renew OFF customers churn at 29% vs 24% with it on.\n- Stated sentiment labels are noisy: ~5% contradict the star rating."),
])

make("03_feature_engineering.ipynb", [
 ("md", "# 03 Feature engineering\n`src/features/build_features.py` builds one row per customer as of 2026-06-30 (not today's date)."),
 ("code", "f = pd.read_csv(PROCESSED/'customer_features.csv'); print(f.shape); f.head(3).T"),
 ("md", "## Required features (PRD 9.2) and 11 extras discovered during EDA\nRequired: customer_tenure_days, avg_watch_time_per_session, monthly_watch_time_trend, days_since_last_activity, login_frequency, support_ticket_count, avg_customer_satisfaction, failed_payment_count/rate, plan_changes, engagement_score, customer_lifetime_value (+ churn_risk_score added after modelling).\n\nExtras: `watch_trend_log_ratio`, `watch_min_last_3m`, `watch_min_prev_3m`, `failed_payments_90d`, `avg_completion_pct`, `distinct_titles`, `distinct_genres`, `distinct_devices`, `escalated_tickets`, `pending_tickets`, `billing_ticket_count`, `negative_feedback_count`, `avg_feedback_rating`, `payment_methods_used`."),
 ("code", "from sklearn.metrics import roc_auc_score\nrows = []\nfor c in f.select_dtypes('number').columns:\n    if c in ('churned',): continue\n    a = roc_auc_score(f.churned, f[c].fillna(f[c].median())); rows.append((c, round(max(a, 1-a), 3), 'higher = more churn' if a > .5 else 'higher = less churn'))\npd.DataFrame(rows, columns=['feature','univariate_AUC','direction']).sort_values('univariate_AUC', ascending=False).head(15)"),
 ("md", "**Reading this honestly:** the single best feature reaches AUC 0.61. No feature is a strong predictor, which is why the model caps out near 0.66. The data generator added a lot of noise, as the dictionary warned."),
 ("code", "print(f.isna().sum()[lambda s: s>0])  # NaNs are left for the model's imputer (median + missing-indicator), not hidden"),
])

make("04_churn_prediction.ipynb", [
 ("md", "# 04 Churn prediction\nTraining code: `src/models/train.py`. Features exclude CLV, churn fields, cancellation fields and gender (see tests/test_pipeline.py)."),
 ("code", "rep = json.load(open(REPORTS/'model_report.json'))\ndisplay(pd.DataFrame(rep['churn_comparison']).round(3))\nprint(rep['churn_selection_rule'])"),
 ("code", "display(Image(filename=str(FIGURES/'model_churn_curves.png'))); display(Image(filename=str(FIGURES/'model_churn_confusion.png'), width=380))"),
 ("code", "print(json.dumps(rep['churn_thresholds'], indent=1)); print(rep['churn_decile_lift']); print(rep['risk_tier_counts'], rep['risk_tier_churn_rates'])"),
 ("md", "## Model choice (written justification)\n**All four candidates are statistically tied** (CV ROC-AUC 0.658-0.662, fold std 0.014-0.019). Picking 'the winner' by the 4th decimal would be fooling ourselves. XGBoost was chosen because it is within one standard deviation of the best and gives exact per-customer TreeSHAP explanations, so the score and its explanation come from the same model.\n\n**Precision/recall in a retention-offer context.** A wasted offer costs money (assumed $5); a missed churner costs the revenue we could have kept (assumed ~$24 expected: 30% accept x ~$80 retained). Contacting a customer pays off when P(churn) > 5/24 = 21%. That is why the tiers are Low < 21%, Medium 21-40%, High >= 40%, and why the profit-optimal threshold favours **recall (~57%) over precision (~37%)** (at the default 0.5 cut-off the XGBoost figures are recall 59%, precision 35%): it is cheap to be wrong about who to offer, expensive to miss them. **These dollar figures are my assumptions, not your data. Change them in `src/models/train.py`.**\n\n**What the model can and cannot do.** ROC-AUC 0.65 is weak. It is useful for *ranking* (top decile churns 45% vs 25% base, bottom decile 8.5%) and poor for calling any single person. Probabilities are isotonic-calibrated on out-of-fold scores."),
 ("code", "import joblib, xgboost as xgb\nfrom src.models.train import CAT, CHURN_NUM\nb = joblib.load(MODELS/'churn_xgb_explainer.joblib'); pipe = b['model']\nf = pd.read_csv(PROCESSED/'customer_features.csv').sample(1500, random_state=1)\nXt = pipe.named_steps['pre'].transform(f[CAT+CHURN_NUM]); names = [n.split('__',1)[1] for n in pipe.named_steps['pre'].get_feature_names_out()]\nc = pipe.named_steps['m'].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]\npd.Series(np.abs(c).mean(0), index=names).sort_values(ascending=False).head(15).round(3)"),
])

make("05_customer_segmentation.ipynb", [
 ("md", "# 05 Customer segmentation (K-Means)"),
 ("code", "rep = json.load(open(REPORTS/'model_report.json'))['segmentation_k_search']; display(pd.DataFrame(rep)); display(Image(filename=str(FIGURES/'model_segmentation_k.png')))"),
 ("md", "**Elbow and silhouette are both flat.** Silhouette is 0.12 for every K from 2 to 10 and the elbow has no clear bend. The honest conclusion: **the customers do not form natural clusters.** K=5 was picked for business usability. These segments are a way to slice the base, not a discovery."),
 ("code", "s = pd.read_csv(PROCESSED/'customer_scores.csv').merge(pd.read_csv(PROCESSED/'customer_features.csv'), on='customer_id')\ns.groupby('segment_name').agg(customers=('customer_id','size'), engagement=('engagement_score','mean'), minutes_per_session=('avg_watch_time_per_session','mean'), tenure_years=('customer_tenure_days', lambda x: x.mean()/365), plan_price=('plan_price','mean'), tickets=('support_ticket_count','mean'), avg_clv=('customer_lifetime_value','mean'), churn=('churned','mean')).round(2)"),
 ("code", "d = json.load(open(REPORTS/'model_report.json'))['segmentation']['descriptions']\nfor k, v in d.items(): display(Markdown(f'**{k}**: {v}'))"),
])

make("06_clv_prediction.ipynb", [
 ("md", "# 06 Customer lifetime value (regression)\nTarget = sum of successful payments to date. Features contain **no payment-derived columns**."),
 ("code", "rep = json.load(open(REPORTS/'model_report.json')); display(pd.DataFrame(rep['clv_comparison']).round(3)); print('CLV mean/std:', round(rep['clv_target_mean'],1), round(rep['clv_target_std'],1)); display(Image(filename=str(FIGURES/'model_clv_actual_vs_pred.png'), width=420))"),
 ("md", "## Read this before putting R² = 0.98 on a resume\nThe tree models explain 98% of variance, but **that is mostly arithmetic, not insight**: CLV is roughly *months subscribed x plan price*, and tenure and plan price are features. Linear regression already gets R² 0.83 and the trees mostly learn the cap (payments are limited to 15 per customer) and plan-price steps. The model is fine for filling in value for the retention rules; it is not a discovery. MAE ~ $10 on a mean of ~$175."),
])
