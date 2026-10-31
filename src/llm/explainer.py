"""Capability 3 (ML prediction explainer) and Capability 6 (retention recommendation)."""
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

from src.llm.client import LLM
from src.llm.sql_guard import engine
from src.utils.config import MODELS

_EX = {}


def _bundle():
    if "b" not in _EX:
        _EX["b"] = joblib.load(MODELS / "churn_xgb_explainer.joblib")
    return _EX["b"]


def tier_context(tier: str) -> str:
    """What the analysis observed for this risk tier (from reports/model_report.json), so a score is always shown with its track record."""
    import json
    from src.utils.config import REPORTS
    rep = json.loads((REPORTS / "model_report.json").read_text())
    r, n = rep["risk_tier_churn_rates"].get(tier), rep["risk_tier_counts"].get(tier)
    if r is None:
        return ""
    return (f"In the analysis, {n:,} customers sit in the {tier} tier and {r:.0%} of them actually churned (portfolio average "
            f"{rep['churn_decile_lift']['base_rate']:.0%}). The model is weak for individuals (ROC-AUC about 0.65): treat this as a ranking, not a verdict.")


def get_customer(customer_id: str) -> pd.Series | None:
    with engine().connect() as c:
        df = pd.read_sql(text("SELECT * FROM v_customer_360 WHERE customer_id = :id"), c, params={"id": customer_id.strip().upper()})
    return None if df.empty else df.iloc[0]


def _phrase(feat: str, r: pd.Series) -> str:
    v = r.get(feat)
    P = {
        "days_since_last_activity": lambda: f"last watched {int(v)} days ago",
        "watch_min_last_3m": lambda: f"watched {int(v)} minutes in the last 3 months",
        "watch_min_prev_3m": lambda: f"watched {int(v)} minutes in the 3 months before that",
        "monthly_watch_time_trend": lambda: f"monthly watch time is trending {'up' if v > 0 else 'down'} ({v:+.1f} min/month)",
        "watch_trend_log_ratio": lambda: f"watch time {'fell' if v < 0 else 'rose'} by about {abs(1 - np.exp(v)) * 100:.0f}% versus the prior 3 months",
        "engagement_score": lambda: f"overall engagement score is {v:.0f}/100",
        "failed_payment_rate": lambda: f"{v:.0%} of payment attempts failed",
        "failed_payment_count": lambda: f"{int(v)} failed payment(s) on record",
        "failed_payments_90d": lambda: f"{int(v)} failed payment(s) in the last 90 days",
        "support_ticket_count": lambda: f"raised {int(v)} support ticket(s)",
        "avg_customer_satisfaction": lambda: ("no rated support tickets" if pd.isna(v) else f"average support satisfaction is {v:.1f}/5"),
        "customer_tenure_days": lambda: f"has been a customer for {v / 365:.1f} years",
        "plan_price": lambda: f"is on a ${v:.2f}/month plan",
        "auto_renew": lambda: "auto-renew is " + ("on" if v == 1 else "OFF"),
        "login_frequency": lambda: f"averages {v:.1f} sessions per active month",
        "total_watch_minutes": lambda: f"{int(v)} total minutes watched",
        "total_sessions": lambda: f"{int(v)} viewing sessions",
        "distinct_titles": lambda: f"watched {int(v)} different titles",
        "distinct_genres": lambda: f"watched {int(v)} genres",
        "avg_feedback_rating": lambda: ("no feedback ratings" if pd.isna(v) else f"average feedback rating {v:.1f}/5"),
        "avg_completion_pct": lambda: f"completes about {v:.0f}% of what they start",
        "avg_resolution_hours": lambda: ("no resolved tickets" if pd.isna(v) else f"support tickets take {v:.0f}h to resolve on average"),
        "billing_ticket_count": lambda: f"{int(v)} billing-related ticket(s)",
        "escalated_tickets": lambda: f"{int(v)} escalated ticket(s)",
        "pending_tickets": lambda: f"{int(v)} unresolved ticket(s)",
        "plan_changes": lambda: f"{int(v)} plan change(s)",
        "distinct_devices": lambda: f"uses {int(v)} device types",
        "payment_methods_used": lambda: f"has used {int(v)} payment methods",
        "negative_feedback_count": lambda: f"{int(v)} negative feedback entr{'y' if v == 1 else 'ies'}",
    }
    return P.get(feat, lambda: f"{feat.replace('_', ' ')} = {v}")()


@dataclass
class Explanation:
    customer_id: str
    probability: float
    tier: str
    drivers_up: list = field(default_factory=list)
    drivers_down: list = field(default_factory=list)
    narrative: str = ""
    row: pd.Series | None = None
    mode: str = "offline"


def contributions(r: pd.Series) -> pd.Series:
    """Per-feature SHAP-style contributions (log-odds) from XGBoost's native pred_contribs (exact TreeSHAP)."""
    b = _bundle()
    X = pd.DataFrame([r])[b["cat"] + b["num"]].copy()
    for c in b["num"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    pipe = b["model"]
    Xt = pipe.named_steps["pre"].transform(X)
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    booster = pipe.named_steps["m"].get_booster()
    c = booster.predict(xgb.DMatrix(Xt), pred_contribs=True)[0][:-1]
    s = pd.Series(c, index=[n.split("__", 1)[1] for n in names])
    base = {}                                    # roll one-hot / missing-indicator columns up to their parent feature
    for n, v in s.items():
        parent = n
        for cat in b["cat"]:
            if n.startswith(cat + "_"):
                parent = cat
        if n.startswith("missingindicator_"):
            parent = n.replace("missingindicator_", "")
        base[parent] = base.get(parent, 0) + v
    return pd.Series(base).sort_values()


HIDE = {"age", "country", "gender"}   # demographics are never shown as "reasons" in customer-facing explanations


def explain_customer(customer_id: str, llm: LLM | None = None, top: int = 4) -> Explanation | None:
    llm = llm or LLM()
    r = get_customer(customer_id)
    if r is None:
        return None
    contrib = contributions(r)
    up = [(f, float(v), _phrase(f, r)) for f, v in contrib[::-1].items() if v > 0.02 and f not in HIDE][:top]
    down = [(f, float(v), _phrase(f, r)) for f, v in contrib.items() if v < -0.02 and f not in HIDE][:top]
    e = Explanation(r.customer_id, float(r.churn_risk_score), r.risk_tier, up, down, row=r)
    facts = (f"Customer {e.customer_id} ({r.current_plan} plan, {r.segment_name}). Calibrated churn probability {e.probability:.0%} ({e.tier} risk; "
             f"portfolio average 25%).\nFactors raising risk: " + "; ".join(p for _, _, p in up) +
             ".\nFactors lowering risk: " + "; ".join(p for _, _, p in down) + ".")
    ctx = tier_context(e.tier)
    facts += "\nTrack record: " + ctx
    if llm.available:
        e.narrative = llm.complete("You explain a churn-model result to a retention manager in 3-5 sentences. Use ONLY the facts given; do not add numbers. "
                                   "Say that the model finds associations, not proven causes.", facts, 400)
        e.mode = "llm"
    else:
        e.narrative = (f"{e.customer_id} is flagged **{e.tier} risk** with a {e.probability:.0%} estimated chance of churning (average customer: 25%). "
                       f"Pushing risk up: {'; '.join(p for _, _, p in up) or 'nothing notable'}. "
                       f"Holding risk down: {'; '.join(p for _, _, p in down) or 'nothing notable'}. "
                       "These are statistical associations in the model, not proven causes. " + ctx)
    return e


# ------------------------------------------------------------------ Capability 6: retention strategy
@dataclass
class Recommendation:
    customer_id: str
    action: str
    priority: str
    rationale: list
    message: str = ""
    row: pd.Series | None = None


def value_tier(r: pd.Series, p75_clv: float) -> str:
    return "High" if (r.customer_lifetime_value >= p75_clv or r.current_plan == "Premium") else "Standard"


def recommend(customer_id: str, llm: LLM | None = None) -> Recommendation | None:
    """Rules decide the action (auditable); the LLM only drafts the message wording."""
    llm = llm or LLM()
    r = get_customer(customer_id)
    if r is None:
        return None
    with engine().connect() as c:
        p75 = float(pd.read_sql(text("SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY customer_lifetime_value) AS p FROM customer_features"), c).p[0])
    vt, tier, why = value_tier(r, p75), r.risk_tier, []
    if bool(r.churned):
        return Recommendation(r.customer_id, "Win-back campaign (customer already churned)", "Medium", [f"Churn reason on file: {r.churn_reason}."], row=r)
    why.append(f"Risk tier {tier} (churn probability {r.churn_risk_score:.0%}); value tier {vt} (CLV ${r.customer_lifetime_value:.0f}, {r.current_plan} plan).")
    if tier == "High" and vt == "High":
        action, prio = "Immediate personalised outreach (account/CS call + tailored offer)", "Urgent"
    elif tier == "High":
        action, prio = "Automated promotional campaign (discount or bundle offer)", "High"
    elif tier == "Medium" and vt == "High":
        action, prio = "Proactive engagement: personalised content picks + satisfaction check-in", "Medium"
    elif tier == "Medium":
        action, prio = "Light-touch nudge (recommendations email)", "Low"
    else:
        action, prio = "No action: monitor", "None"
    if r.failed_payments_90d >= 1 or r.failed_payment_rate >= 0.15:
        why.append(f"Billing friction ({r.failed_payment_rate:.0%} failed attempts): send a payment-method update reminder before any discount.")
    if r.pending_tickets > 0 or r.escalated_tickets > 0 or (pd.notna(r.avg_customer_satisfaction) and r.avg_customer_satisfaction <= 2.5):
        why.append("Open/escalated ticket or low support satisfaction: service recovery must come BEFORE a commercial offer.")
    if r.auto_renew == 0 and tier != "Low":
        why.append("Auto-renew is off: offer an incentive to switch it on.")
    if r.segment_name == "Disengaged Drifters":
        why.append("Segment 'Disengaged Drifters': lead with content re-engagement, not price.")
    if r.current_plan == "Basic" and tier != "Low":
        why.append("Basic-plan customers are price-sensitive: an annual-plan or bundle offer fits better than a one-off discount.")
    if tier != "Low":
        why.append(tier_context(tier))
    rec = Recommendation(r.customer_id, action, prio, why, row=r)
    if prio != "None" and llm.available:
        rec.message = llm.complete("Draft a short, warm retention message (max 70 words) from the streaming service. Do not mention churn risk, scores or "
                                   "that a model flagged them. Do not promise discounts that are not listed.", f"Customer first name: {r.get('first_name', 'there')}\nAction: {action}\nContext: {' '.join(why)}", 200)
    return rec
