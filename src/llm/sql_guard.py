"""Validation layer for LLM-generated SQL. Non-negotiable rules from the PRD:
  1. SELECT only (WITH ... SELECT allowed)      2. INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE (and friends) blocked BEFORE execution
  3. every table and column checked against the real schema     4. timeout + row cap enforced     5. every query logged
"""
import json
import re
import time
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlalchemy import create_engine, text

from src.utils.config import DB_URL_RO, DB_URL, MAX_ROWS, QUERY_TIMEOUT_S, QUERY_LOG

BLOCKED = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|merge|call|do|vacuum|"
                     r"pg_sleep|pg_read_file|pg_ls_dir|lo_import|lo_export|dblink|set\s+role|set\s+session|execute)\b", re.I)
_SCHEMA_CACHE = {}


class SQLRejected(Exception):
    pass


def get_schema(engine=None) -> dict:
    """{table: [columns]} for the public schema (tables + views)."""
    if _SCHEMA_CACHE:
        return _SCHEMA_CACHE
    eng = engine or create_engine(DB_URL_RO)
    with eng.connect() as c:
        rows = c.execute(text("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name, ordinal_position")).fetchall()
    for t, col in rows:
        _SCHEMA_CACHE.setdefault(t, []).append(col)
    return _SCHEMA_CACHE


def validate(sql: str, schema: dict) -> str:
    """Returns cleaned SQL or raises SQLRejected."""
    s = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.I | re.M)
    s = re.sub(r"--[^\n]*|/\*.*?\*/", " ", s, flags=re.S).strip().rstrip(";").strip()
    if not s:
        raise SQLRejected("Empty query.")
    stripped = re.sub(r"'(?:[^']|'')*'", "''", s)          # ignore string literals when scanning for keywords
    stripped = re.sub(r"--.*?$|/\*.*?\*/", " ", stripped, flags=re.M | re.S)
    if ";" in stripped:
        raise SQLRejected("Multiple statements are not allowed.")
    m = BLOCKED.search(stripped)
    if m:
        raise SQLRejected(f"Blocked keyword: {m.group(1).upper()}. Only SELECT is permitted.")
    try:
        tree = sqlglot.parse_one(s, read="postgres")
    except Exception as e:  # noqa: BLE001
        raise SQLRejected(f"SQL could not be parsed: {str(e)[:150]}")
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)) and not (isinstance(tree, exp.With)):
        raise SQLRejected("Only SELECT statements are permitted.")
    if tree.find(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create, exp.Command):
        raise SQLRejected("Only SELECT statements are permitted.")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    real = {}                                   # alias -> set of real tables bound to that alias anywhere in the query
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if not name or name in ctes:
            continue
        if name not in schema:
            raise SQLRejected(f"Unknown table: {t.name}. Known tables: {', '.join(sorted(schema))}")
        real.setdefault((t.alias or t.name).lower(), set()).add(name)
    derived = set(ctes)                          # CTE names, subquery / function aliases and their column aliases
    for a in tree.find_all(exp.TableAlias):
        par = a.parent
        if isinstance(par, exp.Table) and par.name.lower() in schema:
            continue                             # alias of a REAL table: its columns ARE checked
        if a.name:
            derived.add(a.name.lower())
        derived |= {c.name.lower() for c in a.args.get("columns") or []}
    select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    allowed_cols = {c for tbs in real.values() for tb in tbs for c in schema[tb]}
    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if name == "*":
            continue
        q = col.table.lower() if col.table else ""
        if q:
            if q in derived:                     # columns of derived tables are not in the base schema; skip
                continue
            if q not in real:
                raise SQLRejected(f"Unknown table alias: {col.table}")
            if not any(name in schema[tb] for tb in real[q]):
                tb = sorted(real[q])[0]
                raise SQLRejected(f"Unknown column {col.table}.{col.name} (table {tb} has: {', '.join(schema[tb])})")
        elif name not in allowed_cols and name not in select_aliases and name not in derived:
            raise SQLRejected(f"Unknown column: {col.name}")
    return s


@dataclass
class QueryResult:
    sql: str
    ok: bool
    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    error: str = ""
    truncated: bool = False
    seconds: float = 0.0


def log_query(question: str, sql: str, status: str, detail: str = "", rows: int = 0):
    QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QUERY_LOG, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": question, "sql": sql, "status": status,
                            "detail": detail, "rows": rows}) + "\n")


_ENGINE = {}


def engine():
    if "e" not in _ENGINE:
        _ENGINE["e"] = create_engine(DB_URL_RO, pool_pre_ping=True)
    return _ENGINE["e"]


def run_safe(sql: str, question: str = "", max_rows: int = MAX_ROWS, timeout_s: int = QUERY_TIMEOUT_S) -> QueryResult:
    """validate -> execute (read-only txn, statement_timeout, row cap) -> log."""
    t0 = time.time()
    try:
        clean = validate(sql, get_schema(engine()))
    except SQLRejected as e:
        log_query(question, sql, "REJECTED", str(e))
        return QueryResult(sql=sql, ok=False, error=str(e))
    wrapped = f"SELECT * FROM ({clean}) AS q LIMIT {max_rows + 1}"
    try:
        with engine().connect() as c:
            c.execute(text("SET TRANSACTION READ ONLY"))
            c.execute(text(f"SET LOCAL statement_timeout = {int(timeout_s) * 1000}"))
            res = c.execute(text(wrapped))
            cols, rows = list(res.keys()), [list(r) for r in res.fetchall()]
            c.rollback()
    except Exception as e:  # noqa: BLE001
        msg = str(getattr(e, "orig", e))[:300]
        log_query(question, clean, "ERROR", msg)
        return QueryResult(sql=clean, ok=False, error=msg, seconds=time.time() - t0)
    trunc = len(rows) > max_rows
    rows = rows[:max_rows]
    log_query(question, clean, "OK" + (" (truncated)" if trunc else ""), rows=len(rows))
    return QueryResult(sql=clean, ok=True, rows=rows, columns=cols, truncated=trunc, seconds=time.time() - t0)
