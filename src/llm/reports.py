"""Capability 4: automated business report. Facts (SQL) are computed first and shown separately from AI interpretation."""
from dataclasses import dataclass, field

import pandas as pd

from src.llm.client import LLM
from src.llm.grounding import unverified_numbers
from src.llm.sql_guard import run_safe
from src.utils.config import AS_OF_DATE, REPORTS

ASOF = pd.Timestamp(AS_OF_DATE)


def period_bounds(kind: str):
    """(start, end_exclusive, label) for the period ending at the as-of date; kind in weekly|monthly|quarterly."""
    end = (ASOF + pd.Timedelta(days=1))
    if kind == "weekly":
        return end - pd.Timedelta(days=7), end, f"Week ending {ASOF.date()}"
    if kind == "monthly":
        p = ASOF.to_period("M")
        return p.start_time, (p + 1).start_time, str(p)
    if kind == "quarterly":
        p = ASOF.to_period("Q")
        return p.start_time, (p + 1).start_time, str(p)
    raise ValueError("kind must be weekly, monthly or quarterly")


def _one(sql, name):
    r = run_safe(sql, f"report:{name}")
    if not r.ok:
        raise RuntimeError(f"{name}: {r.error}")
    return pd.DataFrame(r.rows, columns=r.columns)


def metrics_for(start, end) -> dict:
    s, e = start.date(), end.date()
    m = {}
    d = _one(f"""SELECT ROUND(COALESCE(SUM(amount) FILTER (WHERE payment_status='Success'),0),2) AS revenue, COUNT(*) AS payment_attempts,
                        COUNT(*) FILTER (WHERE payment_status='Failed') AS failed_payments FROM payments WHERE payment_date >= DATE '{s}' AND payment_date < DATE '{e}'""", "billing")
    m.update({k: float(v) for k, v in d.iloc[0].items()})
    m["failed_payment_rate_pct"] = round(100 * m["failed_payments"] / m["payment_attempts"], 2) if m["payment_attempts"] else None
    d = _one(f"""SELECT COUNT(*) FILTER (WHERE c.registration_date >= DATE '{s}' AND c.registration_date < DATE '{e}') AS new_customers,
                        COUNT(*) FILTER (WHERE cl.churned AND cl.churn_date >= DATE '{s}' AND cl.churn_date < DATE '{e}') AS churned_customers,
                        COUNT(*) FILTER (WHERE c.registration_date < DATE '{s}' AND (cl.churn_date IS NULL OR cl.churn_date >= DATE '{s}')) AS active_at_start,
                        COUNT(*) FILTER (WHERE c.registration_date < DATE '{e}' AND (cl.churn_date IS NULL OR cl.churn_date >= DATE '{e}')) AS active_at_end
                 FROM customers c JOIN churn_labels cl USING (customer_id)""", "customers")
    m.update({k: int(v) for k, v in d.iloc[0].items()})
    m["churn_rate_pct"] = round(100 * m["churned_customers"] / m["active_at_start"], 2) if m["active_at_start"] else None
    d = _one(f"""SELECT COUNT(*) AS sessions, COUNT(DISTINCT customer_id) AS active_viewers, ROUND(AVG(watch_duration_minutes),1) AS avg_minutes_per_session
                 FROM viewing_activity WHERE viewing_date >= DATE '{s}' AND viewing_date < DATE '{e}'""", "engagement")
    m.update({k: (None if pd.isna(v) else float(v)) for k, v in d.iloc[0].items()})
    d = _one(f"""SELECT COUNT(*) AS tickets, ROUND(AVG(customer_satisfaction_score),2) AS avg_csat FROM support_tickets WHERE ticket_date >= DATE '{s}' AND ticket_date < DATE '{e}'""", "support")
    m.update({k: (None if pd.isna(v) else float(v)) for k, v in d.iloc[0].items()})
    d = _one(f"""SELECT issue_subcategory, COUNT(*) AS n FROM support_tickets WHERE ticket_date >= DATE '{s}' AND ticket_date < DATE '{e}' GROUP BY 1 ORDER BY n DESC LIMIT 3""", "complaints")
    m["top_complaints"] = d.values.tolist()
    return m


def _limits() -> str:
    from src.llm.insights import Insights
    ins = Insights()
    return "- " + "\n- ".join(ins.get(i) for i in ("model_quality", "quirk_coverage", "quirk_leakage"))


@dataclass
class Report:
    kind: str
    label: str
    facts: dict
    prior: dict
    interpretation: str = ""
    warnings: list = field(default_factory=list)
    mode: str = "offline"
    snapshot: dict = field(default_factory=dict)

    def markdown(self) -> str:
        f, p = self.facts, self.prior
        def row(name, key, fmt="{:,.0f}"):
            a, b = f.get(key), p.get(key)
            fa = "n/a" if a is None else fmt.format(a)
            fb = "n/a" if b is None else fmt.format(b)
            ch = "" if (a is None or not b) else f"{(a - b) / b * 100:+.1f}%"
            return f"| {name} | {fa} | {fb} | {ch} |"
        rows = [row("Revenue (successful payments, $)", "revenue", "{:,.2f}"), row("Failed payment rate (%)", "failed_payment_rate_pct", "{:.2f}"),
                row("New customers", "new_customers"), row("Churned customers", "churned_customers"), row("Churn rate (%)", "churn_rate_pct", "{:.2f}"),
                row("Active customers at period end", "active_at_end"), row("Active viewers", "active_viewers"), row("Sessions", "sessions"),
                row("Avg minutes per session", "avg_minutes_per_session", "{:.1f}"), row("Support tickets", "tickets"), row("Average CSAT (1-5)", "avg_csat", "{:.2f}")]
        top = ", ".join(f"{n} ({c})" for n, c in f["top_complaints"]) or "none"
        snap = self.snapshot
        warn = "".join(f"\n> **Data note:** {w}" for w in self.warnings)
        return (f"# {self.kind.title()} business report: {self.label}\n{warn}\n\n## Calculated facts (from SQL)\n\n| Metric | This period | Prior period | Change |\n|---|---|---|---|\n"
                + "\n".join(rows) + f"\n\nTop complaints this period: {top}.\n\nCurrent snapshot: {int(snap.get('high_risk', 0)):,} active customers are High churn risk "
                f"(model score), representing about ${snap.get('at_risk_revenue', 0):,.0f} of expected monthly revenue at risk.\n\n"
                f"## Known limits of the analysis behind these numbers\n\n{_limits()}\n\n"
                f"## AI interpretation ({'LLM; numbers verified against the facts above' if self.mode.startswith('llm') else 'rule-based, no LLM configured'})\n\n{self.interpretation}\n")


def generate(kind: str = "monthly", llm: LLM | None = None, save: bool = True) -> Report:
    llm = llm or LLM()
    s, e, label = period_bounds(kind)
    span = e - s
    facts, prior = metrics_for(s, e), metrics_for(s - span, s)
    snap = _one("""SELECT COUNT(*) FILTER (WHERE risk_tier='High') AS high_risk,
                          ROUND(SUM(plan_price * churn_risk_score) FILTER (WHERE risk_tier='High')::numeric,0) AS at_risk_revenue
                   FROM v_customer_360 WHERE NOT churned""", "snapshot").iloc[0].to_dict()
    rep = Report(kind, label, facts, prior, snapshot={k: float(v) for k, v in snap.items()})
    cov = _one("SELECT (SELECT MAX(payment_date) FROM payments) AS pay_end, (SELECT MAX(registration_date) FROM customers) AS reg_end", "coverage").iloc[0]
    last_day = (e - pd.Timedelta(days=1)).date()
    if cov.pay_end < last_day:
        rep.warnings.append(f"Billing data ends {cov.pay_end}, so revenue and payment metrics are {'zero' if facts['payment_attempts'] == 0 else 'partial'} for this period.")
    if cov.reg_end < last_day:
        rep.warnings.append(f"Registration data ends {cov.reg_end}, so 'new customers' is understated for this period.")
    if llm.available:
        out = llm.complete("Write an executive summary for a streaming-service leadership team: sections 'Executive summary', 'Key findings', 'Risks', "
                           "'Recommendations'. Use ONLY the numbers given. Label correlations as such, do not claim causes. Mention data caveats.",
                           f"Period: {label}\nFacts: {facts}\nPrior period: {prior}\nSnapshot: {rep.snapshot}\nWarnings: {rep.warnings}", 900)
        bad = unverified_numbers(out, [facts, prior, rep.snapshot])
        if bad:
            rep.warnings.append(f"LLM text had unverifiable numbers ({', '.join(bad[:5])}); rule-based summary shown instead.")
        else:
            rep.interpretation, rep.mode = out, "llm"
    if not rep.interpretation:
        def d(k):
            a, b = facts.get(k), prior.get(k)
            return None if a is None or not b else (a - b) / b * 100
        pts = []
        cr, pr = facts["churn_rate_pct"], prior["churn_rate_pct"]
        if cr is not None and pr is not None:
            pts.append(f"Churn was {cr}% versus {pr}% in the prior period ({'worse' if cr > pr else 'better' if cr < pr else 'flat'}).")
        if facts["payment_attempts"]:
            pts.append(f"Revenue was ${facts['revenue']:,.2f} with {facts['failed_payment_rate_pct']}% of payment attempts failing.")
        if facts["active_viewers"] is not None and prior["active_viewers"]:
            pts.append(f"Active viewers: {int(facts['active_viewers']):,} vs {int(prior['active_viewers']):,}.")
        if facts["top_complaints"]:
            pts.append("Most common complaint: " + facts["top_complaints"][0][0] + f" ({facts['top_complaints'][0][1]} tickets).")
        pts.append(f"{int(rep.snapshot['high_risk']):,} active customers are currently High risk.")
        rep.interpretation = "- " + "\n- ".join(pts) + "\n\nRecommendation (rule-based): prioritise outreach to High-risk, high-value customers and clear open billing/support issues first."
    if save:
        out_dir = REPORTS / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{kind}_{label.replace(' ', '_')}.md").write_text(rep.markdown())
    return rep
