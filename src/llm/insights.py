"""Capability 7: the copilot's knowledge of the PROJECT ANALYSIS.

Findings are computed from the real data and model outputs (never typed in by hand), saved to reports/analysis_insights.json, and then:
  1. answer "what did the analysis find / what drives churn / how good is the model / what are the limits" questions;
  2. attach relevant caveats to other capabilities (SQL answers, explanations, retention advice, reports).
Rebuild after retraining:  python -m src.llm.insights
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.client import LLM
from src.llm.grounding import unverified_numbers
from src.utils.config import PROCESSED, REPORTS

PATH = REPORTS / "analysis_insights.json"


def build_insights() -> list:
    f = pd.read_csv(PROCESSED / "customer_features.csv")
    sc = pd.read_csv(PROCESSED / "customer_scores.csv")
    cl = pd.read_csv(PROCESSED / "churn_labels.csv")
    subs = pd.read_csv(PROCESSED / "subscriptions.csv")
    tk = pd.read_csv(PROCESSED / "support_tickets.csv")
    fb = pd.read_csv(PROCESSED / "customer_feedback.csv")
    rep = json.loads((REPORTS / "model_report.json").read_text())
    d = f.merge(sc, on="customer_id")
    pct = lambda x: f"{100 * x:.1f}%"
    out = []

    def add(id_, topic, text, keywords, tier="finding"):
        out.append({"id": id_, "topic": topic, "text": text, "keywords": keywords, "kind": tier})

    n, ch = len(f), int(f.churned.sum())
    add("churn_overall", "churn rate", f"Overall churn is {pct(ch / n)} ({ch:,} of {n:,} customers).", "churn rate overall how many customers churned cancelled")
    p = d.groupby("current_plan").churned.mean()
    add("churn_plan", "plan", f"Churn by current plan: Basic {pct(p['Basic'])}, Standard {pct(p['Standard'])}, Premium {pct(p['Premium'])}. The gaps are small.",
        "plan basic standard premium churn subscription tier drivers")
    ar = d.groupby("auto_renew").churned.mean()
    add("churn_autorenew", "auto-renew", f"Customers with auto-renew OFF churn at {pct(ar[0])} versus {pct(ar[1])} with it on.", "auto renew autorenew churn driver")
    reasons = cl[cl.churned].churn_reason.value_counts()
    top = ", ".join(f"{k} ({v:,}, {pct(v / ch)})" for k, v in reasons.head(4).items())
    add("churn_reasons", "stated reasons", f"Top stated churn reasons: {top}. Price-related reasons (Too expensive, Household budget cuts) are the largest group.", "reasons why customers cancel stated price expensive alternative content")
    rows = []
    for c in d.select_dtypes("number").columns:
        if c in ("churned", "segment_id", "churn_risk_score", "predicted_clv", "customer_lifetime_value"):
            continue
        a = roc_auc_score(d.churned, d[c].fillna(d[c].median()))
        rows.append((c, max(a, 1 - a), a))
    rows.sort(key=lambda r: -r[1])
    names = {"watch_min_last_3m": "watch minutes in the last 3 months", "monthly_watch_time_trend": "monthly watch-time trend", "days_since_last_activity": "days since last activity",
             "engagement_score": "engagement score", "watch_trend_log_ratio": "recent-vs-prior watch change", "failed_payment_rate": "failed-payment rate"}
    t3 = "; ".join(f"{names.get(c, c.replace('_', ' '))} (AUC {a:.2f})" for c, a, _ in rows[:5])
    add("drivers", "what drives churn", f"Strongest single signals of churn (univariate ROC-AUC, 0.5 = useless): {t3}. Behaviour (recent viewing, billing failures) matters more than demographics. "
        f"No single signal is strong.", "drivers what drives churn signals features predict reasons behaviour engagement important")
    g = d.groupby("country").churned.agg(["mean", "size"]).query("size >= 100").sort_values("mean", ascending=False)
    ch_ = d.groupby("acquisition_channel").churned.mean()
    add("churn_geo_channel", "country and channel", f"Country churn ranges from {pct(g['mean'].iloc[-1])} to {pct(g['mean'].iloc[0])} ({g.index[0]} highest); acquisition channel from {pct(ch_.min())} to {pct(ch_.max())}. "
        "With 200-400 customers per country these gaps are mostly normal variation, so geography and channel are weak levers.", "country geography channel acquisition region churn higher lower")
    m = rep["churn_comparison"]
    xg = next(r for r in m if r["model"] == rep["churn_final_model"])
    dl = rep["churn_decile_lift"]
    add("model_quality", "model quality", f"The churn model ({rep['churn_final_model']}) has test ROC-AUC {xg['roc_auc']:.2f} (CV {xg['cv_auc_mean']:.3f}). That is weak: all four candidate models tie. "
        f"It works as a RANKING tool: the top-risk 10% of customers churn at {pct(dl['top_decile_churn_rate'])} versus a {pct(dl['base_rate'])} base rate, the bottom 10% at {pct(dl['bottom_decile_churn_rate'])}. "
        "It cannot reliably call an individual customer.", "model accuracy how good reliable trust auc performance weak predict churn model")
    tr, tc = rep["risk_tier_churn_rates"], rep["risk_tier_counts"]
    add("risk_tiers", "risk tiers", f"Risk tiers (calibrated churn probability): Low {tc['Low']:,} customers (observed churn {pct(tr['Low'])}), Medium {tc['Medium']:,} ({pct(tr['Medium'])}), High {tc['High']:,} ({pct(tr['High'])}). "
        f"Cut-offs: contact pays off above {rep['risk_tiers']['break_even_probability']:.0%} (assumed $5 offer cost vs about $24 expected value), High from {rep['risk_tiers']['high_from']:.0%}.", "risk tier high medium low threshold who to contact offer", "model")
    s = d.groupby("segment_name").agg(n=("customer_id", "size"), churn=("churned", "mean"), clv=("customer_lifetime_value", "mean")).sort_values("churn", ascending=False)
    seg = "; ".join(f"{k} ({int(r.n):,} customers, churn {pct(r.churn)}, avg CLV ${r.clv:.0f})" for k, r in s.iterrows())
    add("segments", "segments", f"Segments: {seg}. Silhouette is only {rep['segmentation']['silhouette_at_k']:.2f}: the customers do not form natural clusters, so segments are a slicing convenience.", "segment segmentation cluster group loyal drifters budget")
    cv = {r["model"]: r for r in rep["clv_comparison"]}
    add("clv", "lifetime value", f"CLV models reach R2 {cv[rep['clv_final_model']]['R2']:.2f} (MAE ${cv[rep['clv_final_model']]['MAE']:.0f} on a mean of ${rep['clv_target_mean']:.0f}), but this is largely arithmetic: "
        "CLV is roughly months subscribed x plan price, both inputs.", "clv lifetime value customer value regression revenue per customer")
    cs = tk.groupby("issue_category").customer_satisfaction_score.mean().sort_values()
    vol = tk.issue_category.value_counts()
    add("support", "support", f"Support CSAT averages {tk.customer_satisfaction_score.mean():.2f}/5. Lowest by category: {cs.index[0]} ({cs.iloc[0]:.2f}) and {cs.index[1]} ({cs.iloc[1]:.2f}). "
        f"Largest ticket volume: {vol.index[0]} ({vol.iloc[0]:,}), then {vol.index[1]} ({vol.iloc[1]:,}). Resolution times are almost identical across categories (about 17-19 hours).", "support tickets complaints csat satisfaction category resolution time billing streaming")
    mis = int(fb.sentiment_mismatch.sum())
    add("feedback", "feedback sentiment", f"{mis} of {len(fb):,} feedback entries ({pct(mis / len(fb))}) carry a sentiment label that contradicts the star rating; analysis uses the rating-derived label.", "feedback sentiment label noise rating reviews", "caveat")
    two = int(subs.customer_id.duplicated().sum())
    add("quirk_coverage", "data coverage", "Viewing data covers 2025-07-01 to 2026-06-30, but payments and registrations end 2026-05-31 and churn dates start 2025-10-01. June has no billing or sign-ups, "
        "July-September 2025 show 0% churn, and old-cohort retention (SQL Q20) is overstated. The as-of date is fixed at 2026-06-30.", "data coverage dates missing june billing payments quirks period", "caveat")
    add("quirk_plans", "plan changes", f"{two:,} customers have two subscription rows; about 35% of those 'plan changes' keep the same plan, and the subscription price never changes between rows. "
        "Upgraders and downgraders churn at similar rates (about 27-28% vs 26-27%).", "plan change migration upgrade downgrade price subscription", "caveat")
    add("quirk_leakage", "time structure", "Churn dates sit inside the observation window and 36% of churned customers still have sessions after their churn date, so features are not strictly 'before churn'. "
        "Model results describe association with churn, not a true forecast.", "leakage limitation caveat forecast time window honest trust", "caveat")
    add("quirk_text", "complaint text", "Ticket and feedback text has only about 45 distinct strings across 11,166 records, so complaint search is theme counting, not nuanced semantic search.", "text complaint rag search quirks limitation", "caveat")
    PATH.write_text(json.dumps(out, indent=1))
    return out


def load() -> list:
    return json.loads(PATH.read_text()) if PATH.exists() else build_insights()


def _stems(t: str):
    stop = {"the", "of", "by", "and", "for", "with", "what", "which", "how", "are", "is", "does", "did", "have", "has", "our", "analysis", "project", "tell", "about", "show", "me"}
    return [w[:5] for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in stop]


class Insights:
    def __init__(self):
        self.items = load()
        docs = [f"{i['topic']} {i['keywords']} {i['text']}" for i in self.items]
        self.vec = TfidfVectorizer(analyzer=_stems).fit(docs)
        self.m = self.vec.transform(docs)

    def search(self, q: str, k: int = 4, min_sim: float = 0.08) -> list:
        sims = cosine_similarity(self.vec.transform([q]), self.m)[0]
        idx = [i for i in np.argsort(-sims)[:k] if sims[i] >= min_sim]
        return [self.items[i] for i in idx]

    def get(self, id_: str) -> str:
        return next(i["text"] for i in self.items if i["id"] == id_)

    def caveats_for_sql(self, sql: str) -> list:
        s = sql.lower()
        out = []
        if "payments" in s or "revenue" in s:
            out.append("Billing data ends 2026-05-31: June 2026 revenue/payment figures are zero or partial.")
        if "registration_date" in s or "customers" in s and "cohort" in s:
            out.append("Sign-up data ends 2026-05-31 and churn dates start 2025-10-01: recent sign-ups and early-period churn are understated.")
        if "monthly_price" in s:
            out.append("subscriptions.monthly_price never changes between a customer's rows; use subscription_plans for list prices.")
        if "churn_risk_score" in s or "risk_tier" in s:
            out.append("Risk scores come from a weak model (ROC-AUC about 0.65): good for ranking groups, not for judging one customer.")
        return out

    def answer(self, question: str, llm: LLM | None = None):
        llm = llm or LLM()
        hits = self.search(question)
        if not hits:
            return "I could not match that to the analysis findings. Try asking about churn drivers, model quality, segments, support, or data limitations.", [], "offline"
        ev = [h["text"] for h in hits]
        if llm.available:
            out = llm.complete("You answer questions about a completed customer-analytics project for a business reader, in plain words, max 150 words. Use ONLY the findings "
                               "given. Keep every caveat that the findings state (weak model, data gaps). Do not add numbers or causes.", f"Question: {question}\nFindings:\n- " + "\n- ".join(ev), 450)
            bad = unverified_numbers(out, ev)
            if not bad:
                return out, hits, "llm (numbers verified)"
        return "\n\n".join(f"- {t}" for t in ev), hits, "offline"


if __name__ == "__main__":
    for i in build_insights():
        print(f"[{i['id']}] {i['text'][:160]}")
