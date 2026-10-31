"""Capability 1: natural language -> SQL -> validation -> execution -> plain-language answer."""
import re
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.database.run_queries import load_queries
from src.llm.client import LLM
from src.llm.sql_guard import get_schema, run_safe, engine
from src.utils.config import AS_OF_DATE

CONVENTIONS = f"""Business conventions (follow them):
- The data's as-of date is {AS_OF_DATE}. NEVER use CURRENT_DATE/NOW(). Viewing data covers 2025-07-01..{AS_OF_DATE}; payments end 2026-05-31.
- Active customer = churn_labels.churned = FALSE. Current plan = the customer's latest subscriptions row (DISTINCT ON customer_id ORDER BY subscription_start_date DESC).
- Revenue / lifetime spend = SUM(payments.amount) WHERE payment_status = 'Success'.
- customer_scores / v_customer_360 hold ML outputs: churn_risk_score (calibrated churn probability 0-1), risk_tier (Low/Medium/High), segment_name, predicted_clv.
- v_customer_360 = one row per customer with engineered features + scores + churn label. Prefer it for customer-level questions.
- PostgreSQL syntax only. Return ONE SELECT statement. Add LIMIT for row-heavy outputs. Round money/percentages."""


def schema_prompt() -> str:
    lines = [f"{t}({', '.join(cols)})" for t, cols in get_schema().items()]
    return "Tables (PostgreSQL):\n" + "\n".join(lines) + "\n\n" + CONVENTIONS


FEW_SHOT = """Examples:
Q: How many active customers by country?
SQL: SELECT c.country, COUNT(*) AS active_customers FROM customers c JOIN churn_labels cl USING (customer_id) WHERE cl.churned = FALSE GROUP BY c.country ORDER BY active_customers DESC
Q: Which plan has the highest churn rate?
SQL: WITH cur AS (SELECT DISTINCT ON (customer_id) customer_id, plan_id FROM subscriptions ORDER BY customer_id, subscription_start_date DESC) SELECT sp.plan_name, ROUND(100.0*AVG(cl.churned::int),2) AS churn_rate_pct FROM cur JOIN subscription_plans sp USING (plan_id) JOIN churn_labels cl USING (customer_id) GROUP BY sp.plan_name ORDER BY churn_rate_pct DESC
Q: Show 10 high-risk customers with the largest predicted lifetime value
SQL: SELECT customer_id, current_plan, churn_risk_score, predicted_clv FROM v_customer_360 WHERE risk_tier = 'High' AND churned = FALSE ORDER BY predicted_clv DESC LIMIT 10"""


@dataclass
class Answer:
    question: str
    sql: str = ""
    df: pd.DataFrame | None = None
    answer: str = ""
    ok: bool = False
    error: str = ""
    mode: str = "llm"
    chart: dict | None = None
    notes: list = field(default_factory=list)


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    s = m.group(1) if m else text
    s = re.sub(r"^\s*SQL:\s*", "", s.strip(), flags=re.I)
    return s.strip()


def suggest_chart(df: pd.DataFrame):
    if df is None or df.empty or df.shape[1] < 2 or len(df) < 2:
        return None
    x, y = df.columns[0], df.columns[1]
    if not pd.api.types.is_numeric_dtype(df[y]):
        return None
    kind = "line" if ("date" in x.lower() or "month" in x.lower() or pd.api.types.is_datetime64_any_dtype(df[x])) else "bar"
    return {"kind": kind, "x": x, "y": y}


def _stem_tokens(text: str):
    stop = {"the", "of", "by", "and", "for", "with", "per", "each", "what", "which", "how", "many", "are", "is", "has", "have", "does"}
    return [w[:5] for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in stop]


class _Retriever:
    """Offline fallback: match the question to the closest of the 37 tested business queries."""
    def __init__(self):
        self.qs = load_queries()
        self.vec = TfidfVectorizer(analyzer=_stem_tokens).fit([t for _, t, _ in self.qs])
        self.m = self.vec.transform([t for _, t, _ in self.qs])

    def best(self, question):
        sims = cosine_similarity(self.vec.transform([question]), self.m)[0]
        i = int(sims.argmax())
        return self.qs[i], float(sims[i])


_RETR = None


def df_to_markdown(df: pd.DataFrame, n=30) -> str:
    return df.head(n).to_csv(index=False)


def ask_sql(question: str, llm: LLM | None = None) -> Answer:
    global _RETR
    llm = llm or LLM()
    ans = Answer(question=question)
    if not llm.available:
        _RETR = _RETR or _Retriever()
        (qid, title, sql), sim = _RETR.best(question)
        if sim < 0.40:
            ans.error = ("No LLM key is configured and the question does not match any of the 37 prepared analyses. "
                         "Set ANTHROPIC_API_KEY to enable free-form questions.")
            ans.mode = "offline"
            return ans
        ans.mode = "offline"
        ans.notes.append(f"Offline mode: matched prepared analysis {qid} ('{title}'), similarity {sim:.2f}.")
        sql = sql
    else:
        system = ("You are a careful PostgreSQL analyst. Write ONE read-only SELECT query that answers the question, using only the tables and "
                  "columns listed. Output only the SQL.\n\n" + schema_prompt() + "\n\n" + FEW_SHOT)
        sql = extract_sql(llm.complete(system, question, max_tokens=900))
    res = run_safe(sql, question)
    if not res.ok and llm.available:       # one repair attempt with the error fed back
        fix = extract_sql(llm.complete(system, f"{question}\n\nYour previous SQL failed.\nSQL: {sql}\nError: {res.error}\nReturn corrected SQL only.", 900))
        ans.notes.append(f"First attempt failed ({res.error[:120]}); retried.")
        res = run_safe(fix, question)
    ans.sql = res.sql
    if not res.ok:
        ans.error = res.error
        return ans
    ans.df = pd.DataFrame(res.rows, columns=res.columns)
    ans.ok, ans.chart = True, suggest_chart(ans.df)
    if res.truncated:
        ans.notes.append("Result truncated to the row cap.")
    if ans.df.empty:
        ans.answer = "The query ran successfully but returned no rows."
    elif llm.available:
        ans.answer = llm.complete("You explain query results to a business manager in 2-4 plain sentences. Use ONLY numbers that appear in the result. "
                                  "If the result is truncated or empty, say so. Do not speculate about causes.",
                                  f"Question: {question}\nSQL: {ans.sql}\nResult ({len(ans.df)} rows, first 30 shown):\n{df_to_markdown(ans.df)}", 500)
    else:
        ans.answer = f"Returned {len(ans.df)} row(s). Top row: " + ", ".join(f"{c}={v}" for c, v in ans.df.iloc[0].items())
    return ans
