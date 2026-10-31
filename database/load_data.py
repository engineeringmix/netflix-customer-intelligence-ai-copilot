"""Create schema and bulk-load cleaned CSVs into PostgreSQL using COPY.

Usage: python database/load_data.py        (needs data/processed/ from src.data_processing.clean)
"""
import io
import sys
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import DB_URL, PROCESSED, TABLE_LOAD_ORDER, ROOT  # noqa: E402


def dsn():
    # postgresql+psycopg2://user:pw@host:port/db -> libpq URI
    return DB_URL.replace("+psycopg2", "")


def main():
    conn = psycopg2.connect(dsn())
    cur = conn.cursor()
    cur.execute((ROOT / "database" / "schema.sql").read_text())
    for t in TABLE_LOAD_ORDER:
        df = pd.read_csv(PROCESSED / f"{t}.csv")
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (t,))
        cols = [r[0] for r in cur.fetchall()]
        df = df[cols]
        for c in df.columns:  # booleans / ints with NaN -> proper text for COPY
            if df[c].dtype == "object" and set(df[c].dropna().unique()) <= {True, False, "True", "False"}:
                df[c] = df[c].map({True: "t", "True": "t", False: "f", "False": "f"})
            elif str(df[c].dtype) in ("float64", "Float64") and df[c].dropna().apply(float.is_integer).all() and c not in ("amount", "amount_raw", "monthly_price", "resolution_time_hours", "completion_percentage"):
                df[c] = df[c].astype("Int64")
        df = df.replace({True: "t", False: "f"})
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="\\N")
        buf.seek(0)
        cur.copy_expert(f"COPY {t} ({','.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)
        print(f"loaded {t}: {len(df)} rows")
    conn.commit()
    cur.execute("ANALYZE")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
