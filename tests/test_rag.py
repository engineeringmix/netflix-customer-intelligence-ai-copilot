from src.llm.rag import RagStore


def test_rag_finds_payment_theme_and_cites_count():
    r = RagStore().ask("what are customers saying about payment failures?")
    assert r.n_records > 0 and any("payment" in t[0].lower() for t in r.top_themes) and str(f"{r.n_records:,}") in r.summary


def test_rag_returns_nothing_for_unrelated_query():
    assert RagStore().ask("xylophone zebra").n_records == 0
