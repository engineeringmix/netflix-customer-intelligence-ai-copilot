"""Capability 2: RAG over support-ticket descriptions and customer-feedback text.

Vector store = FAISS. Embeddings = TF-IDF + truncated SVD (LSA), which needs no model download and works offline.
To use neural embeddings, replace `Embedder` with an API/sentence-transformer encoder; nothing else changes.
"""
import re
from collections import Counter
from dataclasses import dataclass, field

import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.llm.client import LLM
from src.utils.config import PROCESSED, MODELS

INDEX_DIR = MODELS / "rag"
_STOP = {"what", "that", "this", "with", "about", "customers", "customer", "saying", "have", "from", "they", "their", "when", "which", "there", "been", "were", "does", "into", "show", "tell", "problems", "issues"}


def load_documents() -> pd.DataFrame:
    t = pd.read_csv(PROCESSED / "support_tickets.csv", parse_dates=["ticket_date"])
    f = pd.read_csv(PROCESSED / "customer_feedback.csv", parse_dates=["feedback_date"])
    a = pd.DataFrame({"doc_id": t.ticket_id, "source": "support_ticket", "customer_id": t.customer_id, "date": t.ticket_date,
                      "text": t.ticket_description, "category": t.issue_category, "subcategory": t.issue_subcategory,
                      "rating": t.customer_satisfaction_score.astype(float)})
    b = pd.DataFrame({"doc_id": f.feedback_id, "source": "feedback", "customer_id": f.customer_id, "date": f.feedback_date,
                      "text": f.feedback_text, "category": f.sentiment_validated, "subcategory": "", "rating": f.rating.astype(float)})
    return pd.concat([a, b], ignore_index=True)


def _augment(d: pd.DataFrame) -> pd.Series:
    """Text that gets embedded: category + subcategory + the complaint itself (so 'payment failures' finds Billing > Failed payment)."""
    return (d.category.fillna("") + " " + d.subcategory.fillna("") + ". " + d.text).str.strip()


def build_index():
    """Index UNIQUE texts (templated complaints repeat thousands of times); records are mapped back by text."""
    docs = load_documents()
    uniq = docs.groupby(["source", "text"], as_index=False).agg(category=("category", lambda s: s.mode().iloc[0]),
                                                                 subcategory=("subcategory", lambda s: s.mode().iloc[0]), records=("doc_id", "size"))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)   # char n-grams: failed ~ failure ~ failing
    X = vec.fit_transform(_augment(uniq))
    n_comp = max(2, min(128, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_comp, random_state=42)
    E = normalize(svd.fit_transform(X)).astype("float32")
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    joblib.dump({"vec": vec, "svd": svd}, INDEX_DIR / "embedder.joblib")
    docs.to_pickle(INDEX_DIR / "docs.pkl")
    uniq.to_pickle(INDEX_DIR / "unique.pkl")
    return len(docs), len(uniq)


@dataclass
class RagAnswer:
    question: str
    summary: str
    n_records: int
    n_total: int
    top_themes: list = field(default_factory=list)
    sources: pd.DataFrame | None = None
    mode: str = "offline"


class RagStore:
    def __init__(self):
        if not (INDEX_DIR / "unique.pkl").exists():
            build_index()
        self.index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
        emb = joblib.load(INDEX_DIR / "embedder.joblib")
        self.vec, self.svd = emb["vec"], emb["svd"]
        self.docs = pd.read_pickle(INDEX_DIR / "docs.pkl")
        self.uniq = pd.read_pickle(INDEX_DIR / "unique.pkl")

    def search(self, query: str, max_unique: int = 8, min_sim: float = 0.45, rel: float = 0.75) -> pd.DataFrame:
        """Returns every RECORD whose text is among the best-matching unique texts (similarity attached)."""
        q = normalize(self.svd.transform(self.vec.transform([query]))).astype("float32")
        sims, ids = self.index.search(q, min(max_unique, self.index.ntotal))
        sims, ids = sims[0], ids[0]
        keep = [(i, s) for i, s in zip(ids, sims) if s >= min_sim and s >= rel * sims[0]]
        # Lexical gate: LSA similarity alone is not discriminative on a tiny, templated vocabulary, so a hit must also share
        # at least one content word (matched on a 4-letter stem: failed ~ failure) with the question.
        stems = {w[:4] for w in re.findall(r"[a-z]{4,}", query.lower()) if w not in _STOP}
        def shares(i):
            u = self.uniq.iloc[i]
            words = {w[:4] for w in re.findall(r"[a-z]{4,}", _augment(pd.DataFrame([u])).iloc[0].lower())}
            return bool(stems & words)
        keep = [(i, s) for i, s in keep if shares(i)]
        if not keep:
            return self.docs.iloc[0:0].assign(similarity=[])
        parts = []
        for i, s in keep:
            u = self.uniq.iloc[i]
            parts.append(self.docs[(self.docs.text == u.text) & (self.docs.source == u.source)].assign(similarity=float(s)))
        return pd.concat(parts, ignore_index=True)

    def ask(self, question: str, llm: LLM | None = None) -> RagAnswer:
        llm = llm or LLM()
        hits = self.search(question)
        n = len(hits)
        if n == 0:
            return RagAnswer(question, "No records were similar enough to this question to summarise. Try different wording.", 0, len(self.docs))
        tk = hits[hits.source == "support_ticket"]
        themes = tk.subcategory.value_counts().head(6).rename("records").reset_index().values.tolist()
        fbk = hits[hits.source == "feedback"]
        stats = (f"{n:,} of {len(self.docs):,} records matched ({len(tk):,} support tickets, {len(fbk):,} feedback entries); "
                 f"mean CSAT/rating of matches {hits.rating.mean():.2f}.")
        sample = hits.sort_values("similarity", ascending=False).drop_duplicates("text").head(10)
        if llm.available:
            ctx = "\n".join(f"[{r.source}|{r.category}|{r.subcategory}] x{(hits.text == r.text).sum()}: {r.text}" for r in sample.itertuples())
            summary = llm.complete("You summarise customer complaints for a support lead. Use ONLY the supplied records. Name the 2-4 main themes, "
                                   "quote briefly, and state how many records the summary is based on (given). Never invent counts.",
                                   f"Question: {question}\nRetrieval stats: {stats}\nTicket subcategory counts: {themes}\nDistinct complaint texts (xN = number of records):\n{ctx}", 600)
            mode = "llm"
        else:
            quotes = "\n".join(f'- "{r.text}" (x{(hits.text == r.text).sum():,})' for r in sample.head(4).itertuples())
            th = ", ".join(f"{s} ({c:,})" for s, c in themes) or "n/a (feedback only)"
            summary = f"Based on {stats}\n\nTicket themes: {th}.\n\nMost representative comments:\n{quotes}"
            mode = "offline"
        src = sample[["doc_id", "source", "category", "subcategory", "date", "rating", "text", "similarity"]]
        return RagAnswer(question, summary, n, len(self.docs), themes, src, mode)
