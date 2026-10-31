import pandas as pd
from src.llm.client import MockLLM
from src.llm.copilot import route
from src.llm.grounding import unverified_numbers
from src.llm.text2sql import ask_sql, extract_sql
from src.llm.rca import parse_question, investigate


def test_router():
    assert route("What is the churn rate by plan?") == "sql"
    assert route("What are customers saying about buffering?") == "rag"
    assert route("Explain CUST000010") == "explain"
    assert route("What should we offer CUST000010 to retain them?") == "retention"
    assert route("Why did churn increase in India last month?") == "rca"
    assert route("Generate a quarterly report") == "report"


def test_text2sql_repairs_then_answers():
    m = MockLLM(["SELECT nonexistent_col FROM customers", "```sql\nSELECT COUNT(*) AS n FROM customers\n```", "There are 8000 customers."])
    a = ask_sql("how many customers?", m)
    assert a.ok and int(a.df.n[0]) == 8000 and any("retried" in n for n in a.notes)


def test_text2sql_blocks_malicious_llm_output():
    a = ask_sql("anything", MockLLM(["DROP TABLE customers", "DELETE FROM customers"]))
    assert not a.ok and "SELECT" in a.error


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("SQL: SELECT 2") == "SELECT 2"


def test_grounding_catches_invented_numbers():
    ev = {"churned": 31, "rate": 0.0297, "n": [1044]}
    assert unverified_numbers("31 customers churned of 1,044 (2.97%)", ev) == []
    assert unverified_numbers("churn rose to 45%", ev) == ["45%"]


def test_rca_rejects_llm_that_invents_numbers():
    liar = MockLLM(lambda s, u: "Churn jumped to 77.3% because of 9,999 failed payments.")
    r = investigate("Why did churn increase in India in May 2026?", liar)
    assert "77.3" not in r.narrative and "rejected" in r.mode


def test_rca_parse():
    reg, mon = parse_question("why did churn increase in Germany in March 2026")
    assert reg == "Germany" and str(mon) == "2026-03"
    reg, mon = parse_question("why did churn go up in the US last month")
    assert reg == "United States" and str(mon) == "2026-06"


def test_rca_flags_false_premise_or_confirms():
    r = investigate("Why did churn increase in Germany last month?")
    assert ("premise does not hold" in r.narrative) == (r.evidence["churn"]["change_pct_points"] <= 0)


def test_insights_route_and_answers_come_from_findings():
    from src.llm.copilot import Copilot
    assert route("What did the analysis find about churn drivers?") == "insights"
    assert route("How reliable is the churn model?") == "insights"
    assert route("Why did churn increase in India last month?") == "rca"      # must not be hijacked
    r = Copilot().ask("How reliable is the churn model?")
    assert r.capability == "analysis-insights" and "ROC-AUC" in r.text and "ranking" in r.text.lower()
    r = Copilot().ask("What drives churn?")
    assert "watch" in r.text.lower()


def test_insights_numbers_match_model_report():
    import json
    from src.llm.insights import build_insights
    from src.utils.config import REPORTS
    rep = json.loads((REPORTS / "model_report.json").read_text())
    txt = " ".join(i["text"] for i in build_insights() if i["id"] == "risk_tiers")
    assert f"{rep['risk_tier_counts']['High']:,}" in txt


def test_sql_answers_carry_data_caveats():
    from src.llm.copilot import Copilot
    r = Copilot().ask("what was total revenue by plan?")
    assert any("Billing data ends" in n for n in r.notes)


def test_explanation_includes_track_record():
    from src.llm.explainer import explain_customer
    e = explain_customer("CUST000702")
    assert "actually churned" in e.narrative


def test_segment_question_goes_to_insights_and_weak_sql_match_is_refused():
    from src.llm.copilot import Copilot
    from src.llm.text2sql import ask_sql
    assert route("Which segment is the most fragile?") == "insights"
    assert "Disengaged Drifters" in Copilot().ask("Which segment is the most fragile?").text
    assert not ask_sql("bake me a cake").ok
