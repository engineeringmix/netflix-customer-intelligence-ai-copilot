"""Run all 37 business queries against PostgreSQL, save results and a summary.

Usage: python -m src.database.run_queries
"""
import re
import time

import pandas as pd
from sqlalchemy import create_engine, text

from src.utils.config import DB_URL, ROOT, REPORTS


def load_queries():
    sql = (ROOT / "database" / "sql_business_questions.sql").read_text()
    parts = re.split(r"^-- (Q\d{2}) \| (.*)$", sql, flags=re.M)
    out = []
    for i in range(1, len(parts), 3):
        body = parts[i + 2]
        body = re.split(r"^-- #{5,}", body, flags=re.M)[0]  # strip section banners
        out.append((parts[i], parts[i + 1].strip(), body.strip()))
    return out


def main():
    eng = create_engine(DB_URL)
    res_dir = REPORTS / "sql_results"
    res_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid, title, sql in load_queries():
        t0 = time.time()
        try:
            with eng.connect() as c:
                df = pd.read_sql(text(sql), c)
            df.head(1000).to_csv(res_dir / f"{qid}.csv", index=False)
            rows.append((qid, title, "OK", len(df), round(time.time() - t0, 2)))
        except Exception as e:  # noqa: BLE001
            rows.append((qid, title, f"ERROR: {str(e)[:120]}", 0, round(time.time() - t0, 2)))
    summary = pd.DataFrame(rows, columns=["query", "title", "status", "rows", "seconds"])
    summary.to_csv(REPORTS / "sql_run_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\n{(summary.status == 'OK').sum()}/{len(summary)} queries OK")


if __name__ == "__main__":
    main()
