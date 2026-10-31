"""Router: one entry point that picks the right capability for a question. All six capabilities share the same DB, models and index."""
import re
from dataclasses import dataclass, field

from src.llm.client import LLM

CUST = re.compile(r"\bCUST\d{6}\b", re.I)


def route(q: str) -> str:
    ql = q.lower()
    if re.search(r"\b(report|summary of the (week|month|quarter)|weekly|monthly|quarterly)\b", ql) and re.search(r"(generate|write|draft|create|give|produce|report)", ql):
        return "report"
    if CUST.search(q):
        return "retention" if re.search(r"(offer|retain|retention|recommend|what should we do|action|outreach)", ql) else "explain"
    if re.search(r"\bwhy\b.*\b(churn|cancel)", ql) and re.search(r"(increase|increased|rise|rose|spike|jump|went up|grew|growth|higher|drop|decrease|fell|declin)", ql):
        return "rca"
    if re.search(r"(key |main )?(findings?|insights?)\b|what (did|does) the (analysis|project|data) (find|show|say)|what (drives|drive|causes) churn|(churn )?drivers\b|"
                 r"how (good|accurate|reliable|trustworthy) is (the|our) (churn )?model|can (i|we) trust|limitations?|caveats?|summar\w* (of )?the (analysis|project)|what should we focus|segments?\b.*(fragile|valuable|risky|best|worst|biggest|churn)|(fragile|valuable)\b.*segments?\b", ql) and not re.search(r"\bwhy did\b", ql):
        return "insights"
    if re.search(r"(what are|what do|what is) (customers|users|people).*(saying|complain|think|feel)|complain|feedback|reviews?\b|themes?|sentiment", ql) and not re.search(r"\b(count|how many|average|avg|top \d+|rate)\b", ql):
        return "rag"
    return "sql"


@dataclass
class CopilotResponse:
    capability: str
    text: str
    sql: str = ""
    df: object = None
    chart: dict | None = None
    sources: object = None
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    ok: bool = True


class Copilot:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()
        self._rag = None
        self._ins = None

    @property
    def insights(self):
        from src.llm.insights import Insights
        self._ins = self._ins or Insights()
        return self._ins

    def ask(self, q: str) -> CopilotResponse:
        cap = route(q)
        if cap == "sql":
            from src.llm.text2sql import ask_sql
            a = ask_sql(q, self.llm)
            notes = a.notes + (["Analysis caveat: " + c for c in self.insights.caveats_for_sql(a.sql)] if a.sql else [])
            return CopilotResponse("text-to-sql", a.answer or a.error, a.sql, a.df, a.chart, None, [], notes, a.ok)
        if cap == "insights":
            text, hits, mode = self.insights.answer(q, self.llm)
            return CopilotResponse("analysis-insights", text, notes=[f"Based on {len(hits)} analysis finding(s) (mode: {mode}). Findings are recomputed from the data by `python -m src.llm.insights`."])
        if cap == "rag":
            from src.llm.rag import RagStore
            self._rag = self._rag or RagStore()
            r = self._rag.ask(q, self.llm)
            return CopilotResponse("rag", r.summary, df=r.sources, notes=[f"Based on {r.n_records:,} of {r.n_total:,} records."], sources=r.sources)
        if cap in ("explain", "retention"):
            from src.llm.explainer import explain_customer, recommend
            cid = CUST.search(q).group(0).upper()
            if cap == "explain":
                e = explain_customer(cid, self.llm)
                return CopilotResponse("explainer", e.narrative if e else f"Customer {cid} not found.", ok=e is not None)
            r = recommend(cid, self.llm)
            if r is None:
                return CopilotResponse("retention", f"Customer {cid} not found.", ok=False)
            txt = f"**{r.priority} priority: {r.action}**\n\n" + "\n".join(f"- {w}" for w in r.rationale) + (f"\n\nDraft message:\n> {r.message}" if r.message else "")
            return CopilotResponse("retention", txt)
        if cap == "rca":
            from src.llm.rca import investigate
            r = investigate(q, self.llm)
            return CopilotResponse("root-cause", r.narrative, steps=r.steps, notes=r.warnings + [f"Mode: {r.mode}"])
        from src.llm.reports import generate
        kind = "quarterly" if "quarter" in q.lower() else "weekly" if "week" in q.lower() else "monthly"
        rep = generate(kind, self.llm)
        return CopilotResponse("report", rep.markdown(), notes=rep.warnings)
