import pytest
from src.llm.sql_guard import validate, get_schema, SQLRejected, run_safe

SCHEMA = get_schema()


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers", "DELETE FROM customers", "UPDATE customers SET age = 1", "INSERT INTO customers (customer_id) VALUES ('x')",
    "ALTER TABLE customers ADD COLUMN x int", "TRUNCATE payments", "SELECT 1; DROP TABLE customers", "SELECT pg_sleep(60)",
    "CREATE TABLE t AS SELECT 1", "GRANT ALL ON customers TO public", "COPY customers TO '/tmp/x'"])
def test_blocked(sql):
    with pytest.raises(SQLRejected):
        validate(sql, SCHEMA)


def test_unknown_table_and_column():
    with pytest.raises(SQLRejected, match="Unknown table"):
        validate("SELECT * FROM secrets", SCHEMA)
    with pytest.raises(SQLRejected, match="Unknown column"):
        validate("SELECT salary FROM customers", SCHEMA)
    with pytest.raises(SQLRejected, match="Unknown column"):
        validate("SELECT c.salary FROM customers c", SCHEMA)


def test_valid_queries_pass():
    validate("SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC", SCHEMA)
    validate("WITH a AS (SELECT customer_id FROM churn_labels WHERE churned) SELECT COUNT(*) FROM a", SCHEMA)


def test_keyword_inside_string_literal_is_fine():
    validate("SELECT customer_id FROM customers WHERE city = 'Drop Zone'", SCHEMA)


def test_row_cap_and_logging(tmp_path, monkeypatch):
    r = run_safe("SELECT * FROM payments", question="cap test", max_rows=7)
    assert r.ok and len(r.rows) == 7 and r.truncated


def test_db_role_is_read_only():
    from sqlalchemy import text
    from src.llm.sql_guard import engine
    with engine().connect() as c:
        with pytest.raises(Exception):
            c.execute(text("DELETE FROM customers WHERE customer_id = 'CUST000001'"))


def test_all_37_business_queries_pass_validator():
    from src.database.run_queries import load_queries
    qs = load_queries()
    assert len(qs) == 37
    for qid, _, sql in qs:
        validate(sql, SCHEMA)
