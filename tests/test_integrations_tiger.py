"""Tiger Data layer tests. The database connection is a MOCK recording SQL; no Postgres/TimescaleDB is contacted, so
schema syntax and performance against the real service are NOT verified by these tests."""
from pathlib import Path

import pandas as pd
import pytest

from roadwatch.integrations import tiger_data as td

ROOT = Path(__file__).resolve().parents[1]
real = pytest.mark.skipif(not (ROOT / "data" / "processed" / "stage5" / "worklists").exists(), reason="real outputs absent")


class FakeCur:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def executemany(self, sql, rows): self.log.append(("many", sql, len(rows)))


class FakeConn:
    def __init__(self): self.log = []
    def execute(self, sql, params=None):
        self.log.append(("exec", sql, params))
        return types_cur(self)
    def cursor(self): return FakeCur(self.log)


def types_cur(conn):
    import types
    return types.SimpleNamespace(fetchall=lambda: [])


def test_schema_has_no_sensitive_columns_and_uses_documented_syntax():
    sql = td.SCHEMA_FILE.read_text(encoding="utf-8")
    assert "timescaledb.hypertable" in sql and "timescaledb.partition_column" in sql and "timescaledb.continuous" in sql
    assert "add_continuous_aggregate_policy" in sql and "IF NOT EXISTS" in sql
    for bad in ("descriptor", "street", "address", "complaint", "latitude", "longitude", "unique_key"):
        assert bad not in sql.lower().replace("-- contains no complaint text, addresses or street names", "")


def test_upsert_is_idempotent_by_construction():
    df = pd.DataFrame({"week_start": [pd.Timestamp("2026-03-16", tz="UTC")], "policy": ["p"], "block_id": [1], "rank": [1], "score": [0.2]})
    s = td.upsert_sql("rw_block_scores", df)
    assert "ON CONFLICT (week_start, policy, block_id) DO UPDATE SET rank = EXCLUDED.rank, score = EXCLUDED.score" in s
    assert "%s, %s, %s, %s, %s" in s and "VALUES" in s


def test_sync_batches_and_migrate_runs_each_statement():
    c = FakeConn()
    ex = {"rw_weekly_reports": pd.DataFrame({"week_start": pd.to_datetime(["2026-03-02", "2026-03-09"], utc=True), "reports": [800, 900], "week_status": ["observable"] * 2})}
    assert td.sync(c, ex, batch=1) == {"rw_weekly_reports": 2}
    assert [x for x in c.log if x[0] == "many"][0][2] == 1 and len([x for x in c.log if x[0] == "many"]) == 2
    c2 = FakeConn()
    td.migrate(c2)
    assert len(c2.log) == len(td.statements()) == 10 and all("%" not in x[1] for x in c2.log)
    kinds = " ".join(x[1][:30] for x in c2.log)
    assert kinds.count("CREATE TABLE IF NOT EXISTS") == 5 and "CREATE MATERIALIZED VIEW" in kinds


def test_queries_are_parameterised_and_bounded():
    c = FakeConn()
    s = td.TigerSeries(c)
    s.rank_history(52878, limit=10**9)
    s.weekly_reports("2026-01-01", "2026-06-01'; DROP TABLE x;--")
    for _, sql, params in c.log:
        assert "DROP" not in sql and "%s" in sql and "LIMIT %s" in sql
    assert c.log[0][2][-1] == td.MAX_ROWS and c.log[0][2][0] == 52878          # limit capped, id bound as a parameter
    assert "DROP" in c.log[1][2][1]                                                # injection attempt travels only as a bound value


@real
def test_local_fallback_answers_same_queries_from_saved_outputs():
    ex = td.build_exports()
    assert set(ex) == set(td.TABLE_KEYS)
    for t, keys in td.TABLE_KEYS.items():
        assert not ex[t].duplicated(keys).any(), t                                 # primary keys are unique -> sync cannot conflict
    assert (ex["rw_block_scores"]["rank"] <= td.TOP_N).all() and len(ex["rw_weekly_reports"]) > 300
    ls = td.LocalSeries(ex)
    mar = ls.weekly_reports("2026-03-01", "2026-04-01")
    assert len(mar) == 5 and mar["reports"].max() > 800                            # the March 2026 surge is visible
    top = ex["rw_block_scores"].query("rank == 1").iloc[0]
    h = ls.rank_history(int(top["block_id"]))
    assert len(h) >= 1 and (h["rank"] >= 1).all()
    assert set(ls.monthly_metrics(td.POLICY.replace("primary__LGBM", "Strict single-block LightGBM (top-K)"), "1pct").columns) >= {"capture_any"}
    assert len(ls.monthly_reports()) > 60
