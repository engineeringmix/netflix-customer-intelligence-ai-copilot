"""Capability 5: root-cause analysis agent. A FIXED investigation sequence; every number comes from a logged SQL query.
Sequence: churn delta -> engagement MoM -> payment failures -> support signals -> plan mix of churners -> churn reasons -> significance test."""
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.llm.client import LLM
from src.llm.grounding import unverified_numbers
from src.llm.sql_guard import run_safe
from src.utils.config import AS_OF_DATE
from src.data_processing.clean import COUNTRY_MAP

COUNTRIES = sorted(set(COUNTRY_MAP.values()))
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])}


def parse_question(q: str):
    ql = q.lower()
    region = next((c for c in COUNTRIES if c.lower() in ql), None)
    if region is None:
        for k, v in COUNTRY_MAP.items():
            if len(k) <= 3 and re.search(rf"\b{re.escape(k)}\b", ql):
                region = v
                break
    asof = pd.Timestamp(AS_OF_DATE)
    last_full = (asof + pd.Timedelta(days=1)).to_period("M") - 1          # latest complete month
    month = last_full
    m = re.search(r"(20\d\d)-(\d\d)", q)
    if m:
        month = pd.Period(f"{m.group(1)}-{m.group(2)}")
    else:
        for name, n in MONTHS.items():
            if re.search(rf"\b{name}\b", ql):
                y = re.search(r"\b(20\d\d)\b", q)
                year = int(y.group(1)) if y else (last_full.year if n <= last_full.month else last_full.year - 1)
                month = pd.Period(year=year, month=n, freq="M")
                break
    return region, month


@dataclass
class RcaResult:
    question: str
    region: str | None
    month: str
    evidence: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)       # (step name, sql)
    narrative: str = ""
    verdict: str = ""
    mode: str = "offline"
    warnings: list = field(default_factory=list)


def _q(sql, question, steps, name):
    r = run_safe(sql, question)
    steps.append((name, r.sql or sql))
    if not r.ok:
        raise RuntimeError(f"{name}: {r.error}")
    return pd.DataFrame(r.rows, columns=r.columns)


def investigate(question: str, llm: LLM | None = None) -> RcaResult:
    llm = llm or LLM()
    region, month = parse_question(question)
    pm = month - 1
    ms, me = month.start_time.date(), (month + 1).start_time.date()
    ps = pm.start_time.date()
    reg = f"AND c.country = '{region}'" if region else ""
    res = RcaResult(question, region, str(month))
    ev, S = {}, res.steps
    # 1. churn delta
    def churn_for(start, end):
        d = _q(f"""SELECT COUNT(*) FILTER (WHERE c.registration_date < DATE '{start}' AND (cl.churn_date IS NULL OR cl.churn_date >= DATE '{start}')) AS active_at_start,
                          COUNT(*) FILTER (WHERE cl.churned AND cl.churn_date >= DATE '{start}' AND cl.churn_date < DATE '{end}') AS churned
                   FROM customers c JOIN churn_labels cl USING (customer_id) WHERE TRUE {reg}""", question, S, f"churn {start}")
        return int(d.active_at_start[0]), int(d.churned[0])
    a1, c1 = churn_for(ms, me)
    a0, c0 = churn_for(ps, ms)
    r1, r0 = c1 / a1 if a1 else np.nan, c0 / a0 if a0 else np.nan
    ev["churn"] = {"month": str(month), "prior_month": str(pm), "churned": c1, "active_at_start": a1, "churn_rate_pct": round(100 * r1, 2),
                   "prior_churned": c0, "prior_active_at_start": a0, "prior_churn_rate_pct": round(100 * r0, 2),
                   "change_pct_points": round(100 * (r1 - r0), 2)}
    pool = (c1 + c0) / (a1 + a0) if (a1 + a0) else 0
    se = np.sqrt(pool * (1 - pool) * (1 / a1 + 1 / a0)) if a1 and a0 and 0 < pool < 1 else np.nan
    p = float(2 * (1 - norm.cdf(abs((r1 - r0) / se)))) if se == se and se > 0 else np.nan
    ev["churn"]["p_value"] = round(p, 3) if p == p else None
    res.verdict = ("increase" if r1 > r0 else "decrease" if r1 < r0 else "no change") + (" (statistically distinguishable)" if p == p and p < 0.05 else " (within normal month-to-month noise)")
    # 2. engagement
    def eng(start, end):
        d = _q(f"""SELECT COUNT(v.viewing_id) AS sessions, COALESCE(SUM(v.watch_duration_minutes),0) AS minutes, COUNT(DISTINCT v.customer_id) AS viewers,
                          ROUND(AVG(v.watch_duration_minutes),1) AS avg_min_per_session
                   FROM viewing_activity v JOIN customers c USING (customer_id)
                   WHERE v.viewing_date >= DATE '{start}' AND v.viewing_date < DATE '{end}' {reg}""", question, S, f"engagement {start}")
        return {k: (None if pd.isna(v) else float(v)) for k, v in d.iloc[0].items()}
    ev["engagement"] = {"month": eng(ms, me), "prior_month": eng(ps, ms)}
    # 3. payments
    def pay(start, end):
        d = _q(f"""SELECT COUNT(*) AS attempts, COUNT(*) FILTER (WHERE p.payment_status='Failed') AS failed
                   FROM payments p JOIN customers c USING (customer_id) WHERE p.payment_date >= DATE '{start}' AND p.payment_date < DATE '{end}' {reg}""", question, S, f"payments {start}")
        a, f = int(d.attempts[0]), int(d.failed[0])
        return {"attempts": a, "failed": f, "failed_rate_pct": round(100 * f / a, 2) if a else None}
    ev["payments"] = {"month": pay(ms, me), "prior_month": pay(ps, ms)}
    if ev["payments"]["month"]["attempts"] == 0:
        res.warnings.append(f"No billing data exists for {month} (payments end 2026-05-31): payment signals cannot be assessed for this month.")
    # 4. support
    def sup(start, end):
        d = _q(f"""SELECT t.issue_category AS category, COUNT(*) AS tickets, ROUND(AVG(t.customer_satisfaction_score),2) AS avg_csat
                   FROM support_tickets t JOIN customers c USING (customer_id)
                   WHERE t.ticket_date >= DATE '{start}' AND t.ticket_date < DATE '{end}' {reg} GROUP BY t.issue_category ORDER BY tickets DESC""", question, S, f"support {start}")
        return {"tickets": int(d.tickets.sum()) if len(d) else 0, "top_categories": d.head(3).values.tolist()}
    ev["support"] = {"month": sup(ms, me), "prior_month": sup(ps, ms)}
    # 5. plan mix of churners
    d = _q(f"""WITH cur AS (SELECT DISTINCT ON (customer_id) customer_id, plan_id FROM subscriptions ORDER BY customer_id, subscription_start_date DESC)
               SELECT sp.plan_name, COUNT(*) FILTER (WHERE cl.churn_date >= DATE '{ms}' AND cl.churn_date < DATE '{me}') AS churned_month,
                      COUNT(*) FILTER (WHERE cl.churn_date >= DATE '{ps}' AND cl.churn_date < DATE '{ms}') AS churned_prior
               FROM churn_labels cl JOIN customers c USING (customer_id) JOIN cur USING (customer_id) JOIN subscription_plans sp USING (plan_id)
               WHERE cl.churned {reg} GROUP BY sp.plan_name ORDER BY sp.plan_name""", question, S, "plan mix")
    ev["plan_mix_of_churners"] = d.values.tolist()
    # 6. reasons
    d = _q(f"""SELECT cl.churn_reason, COUNT(*) AS n FROM churn_labels cl JOIN customers c USING (customer_id)
               WHERE cl.churned AND cl.churn_date >= DATE '{ms}' AND cl.churn_date < DATE '{me}' {reg} GROUP BY 1 ORDER BY n DESC LIMIT 3""", question, S, "churn reasons")
    ev["top_churn_reasons"] = d.values.tolist()
    res.evidence = ev

    template = _template(res)
    if llm.available:
        out = llm.complete("You are a careful analyst. Write a root-cause explanation (max 150 words) for a manager using ONLY the evidence JSON. "
                           "Every number you write must appear in the evidence. Do NOT claim causation beyond what the evidence supports; if the change is "
                           "within normal noise, say so. If a signal has no data, say so.", f"Question: {question}\nEvidence: {ev}\nVerdict: {res.verdict}", 500)
        bad = unverified_numbers(out, ev)
        if bad:
            res.warnings.append(f"LLM text contained numbers not found in query results ({', '.join(bad[:5])}); showing the deterministic summary instead.")
            res.narrative, res.mode = template, "offline (LLM output rejected by number check)"
        else:
            res.narrative, res.mode = out, "llm (numbers verified)"
    else:
        res.narrative = template
    return res


def _pct(a, b):
    return None if not b else round(100 * (a - b) / b, 1)


def _template(res: RcaResult) -> str:
    ev, c = res.evidence, res.evidence["churn"]
    where = res.region or "the whole base"
    asked_increase = bool(re.search(r"increas|rise|rose|spike|jump|went up|higher|grew", res.question.lower()))
    premise = ""
    if asked_increase and c["change_pct_points"] <= 0:
        premise = f"**The premise does not hold:** churn in {where} did not increase in {c['month']}. "
    lines = [premise + f"**{where}, {c['month']}:** {c['churned']} customers churned out of {c['active_at_start']} active at the start of the month "
             f"({c['churn_rate_pct']}%), versus {c['prior_churned']} of {c['prior_active_at_start']} ({c['prior_churn_rate_pct']}%) in {c['prior_month']}: "
             f"a change of {c['change_pct_points']} percentage points, a {res.verdict}" + (f" (p = {c['p_value']})." if c["p_value"] is not None else ".")]
    e1, e0 = ev["engagement"]["month"], ev["engagement"]["prior_month"]
    if e1["sessions"] is not None and e0["sessions"]:
        lines.append(f"- Engagement: {int(e1['sessions'])} sessions vs {int(e0['sessions'])} the month before; {int(e1['viewers'])} vs {int(e0['viewers'])} distinct viewers.")
    p1, p0 = ev["payments"]["month"], ev["payments"]["prior_month"]
    lines.append(f"- Payments: {p1['failed_rate_pct']}% failed vs {p0['failed_rate_pct']}% the month before." if p1["attempts"] else "- Payments: no billing data for this month, so payment failures cannot be assessed.")
    s1, s0 = ev["support"]["month"], ev["support"]["prior_month"]
    top = ", ".join(f"{c_[0]} ({c_[1]})" for c_ in s1["top_categories"]) or "none"
    lines.append(f"- Support: {s1['tickets']} tickets vs {s0['tickets']} the month before; top categories: {top}.")
    if ev["top_churn_reasons"]:
        lines.append("- Stated churn reasons: " + ", ".join(f"{r} ({n})" for r, n in ev["top_churn_reasons"]) + ".")
    lines.append("These are correlations from the data, not proof of cause.")
    return "\n".join(lines)
