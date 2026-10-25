"""Optional time-series serving layer on Tiger Data (PostgreSQL/TimescaleDB), with a local pandas fallback.

Canonical data remain the archived snapshots and Parquet outputs. `build_exports` derives bounded tables from SAVED outputs
(no raw 311 text, no addresses). `sync` loads them idempotently (INSERT ... ON CONFLICT DO UPDATE). Queries are
parameterised, indexed and LIMIT-bounded. `LocalSeries` answers the same queries from the same exports without any database,
so the temporal view works with the integration off.
Connection: TIGER_DATABASE_URL (a standard postgres:// URL; Tiger Cloud requires sslmode=require), never logged.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .. import dashboard_data as dd
from ..config import Paths
from . import config

SCHEMA_FILE = Path(__file__).with_name("tiger_schema.sql")
MAX_ROWS = 2000
TOP_N = 500
POLICY = "primary__LGBM"
CAP_LABEL = {0.005: "0.5pct", 0.01: "1pct", 0.02: "2pct"}


# ------------------------------------------------------------------ exports from saved outputs
def build_exports(paths: Paths | None = None, top_n: int = TOP_N) -> dict[str, pd.DataFrame]:
    paths = paths or Paths()
    L = dd.Locations(paths)
    fr, _, _ = dd.volume_series(paths)
    weekly = pd.DataFrame({"week_start": pd.to_datetime(fr["origin"], utc=True), "reports": fr["y"].astype(int), "week_status": fr["week_status"]})
    weekly = weekly[weekly["week_status"] == "observable"].reset_index(drop=True)
    # scores: top-N blocks per week from the saved strict single-block predictions
    p = pq.read_table(L.s4 / "predictions" / f"{POLICY}.parquet", columns=["block_id", "origin", "score"]).to_pandas()
    p["origin"] = pd.to_datetime(p["origin"])
    p = p.sort_values(["origin", "score", "block_id"], ascending=[True, False, True])
    p["rank"] = p.groupby("origin").cumcount() + 1
    scores = p[p["rank"] <= top_n].rename(columns={"origin": "week_start"}).assign(policy=POLICY)
    scores["week_start"] = pd.to_datetime(scores["week_start"], utc=True)
    scores = scores[["week_start", "policy", "block_id", "rank", "score"]].reset_index(drop=True)
    sel = []
    for cap, tag in CAP_LABEL.items():
        w = pq.read_table(L.s5 / "worklists" / dd.LISTS["strict_topk"][1].format(cap=dd.CAP_NAME[cap]),
                          columns=["forecast_week_start", "block_id", "rank", "priority"]).to_pandas()
        sel.append(w.rename(columns={"forecast_week_start": "week_start"}).assign(policy="strict_topk", capacity=tag))
    selections = pd.concat(sel, ignore_index=True)
    selections["week_start"] = pd.to_datetime(selections["week_start"], utc=True)
    selections = selections[["week_start", "policy", "capacity", "block_id", "rank", "priority"]]
    m = dd.policy_monthly(paths)
    metrics = pd.DataFrame({"month": pd.to_datetime(m["month"] + "-01", utc=True), "policy": m["policy"],
                            "capacity": m["capacity"].map(CAP_LABEL), "reports_total": m["reports_total"].astype(int),
                            "capture_any": m["capture_any_candidate"], "capture_assigned": m["capture_assigned_block"]})
    return {"rw_weekly_reports": weekly, "rw_block_scores": scores, "rw_selections": selections, "rw_monthly_metrics": metrics}


# ------------------------------------------------------------------ SQL generation
TABLE_KEYS = {"rw_weekly_reports": ["week_start"], "rw_block_scores": ["week_start", "policy", "block_id"],
              "rw_selections": ["week_start", "policy", "capacity", "block_id"], "rw_monthly_metrics": ["month", "policy", "capacity"]}


def upsert_sql(table: str, df: pd.DataFrame) -> str:
    cols = list(df.columns)
    keys = TABLE_KEYS[table]
    upd = [c for c in cols if c not in keys]
    return (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) ON CONFLICT ({', '.join(keys)}) "
            f"DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in upd)}")


def _rows(df: pd.DataFrame):
    return [tuple(None if pd.isna(v) else (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v.item() if hasattr(v, 'item') else v)
                  for v in r) for r in df.itertuples(index=False, name=None)]


def connect(url: str | None = None):
    import psycopg                                                   # lazy; only when explicitly requested
    return psycopg.connect(url or config.secret("TIGER_DATABASE_URL"), connect_timeout=10, autocommit=True)


def migrate(conn) -> None:
    for stmt in statements():
        conn.execute(stmt)


def statements() -> list[str]:
    """Split the migration into statements (comment lines removed first, so a comment never hides a statement)."""
    text = "\n".join(l for l in SCHEMA_FILE.read_text(encoding="utf-8").splitlines() if not l.strip().startswith("--"))
    return [s.strip() for s in text.split(";") if s.strip()]


def sync(conn, exports: dict[str, pd.DataFrame], batch: int = 5000) -> dict[str, int]:
    """Idempotent: re-running leaves identical rows; changed values are updated."""
    done = {}
    for table, df in exports.items():
        sql = upsert_sql(table, df)
        rows = _rows(df)
        with conn.cursor() as cur:
            for i in range(0, len(rows), batch):
                cur.executemany(sql, rows[i:i + batch])
        done[table] = len(rows)
    return done


# ------------------------------------------------------------------ bounded queries (database)
def _limit(n: int) -> int:
    return max(1, min(int(n), MAX_ROWS))


class TigerSeries:
    def __init__(self, conn):
        self.conn = conn

    def weekly_reports(self, start: str, end: str, limit: int = MAX_ROWS) -> pd.DataFrame:
        cur = self.conn.execute("SELECT week_start, reports FROM rw_weekly_reports WHERE week_start >= %s AND week_start < %s "
                                "ORDER BY week_start LIMIT %s", (start, end, _limit(limit)))
        return pd.DataFrame(cur.fetchall(), columns=["week_start", "reports"])

    def monthly_reports(self, limit: int = 200) -> pd.DataFrame:
        cur = self.conn.execute("SELECT bucket, reports, peak_week_reports, weeks FROM rw_monthly_reports ORDER BY bucket LIMIT %s", (_limit(limit),))
        return pd.DataFrame(cur.fetchall(), columns=["bucket", "reports", "peak_week_reports", "weeks"])

    def rank_history(self, block_id: int, policy: str = POLICY, limit: int = 200) -> pd.DataFrame:
        cur = self.conn.execute("SELECT week_start, rank, score FROM rw_block_scores WHERE block_id = %s AND policy = %s "
                                "ORDER BY week_start LIMIT %s", (int(block_id), policy, _limit(limit)))
        return pd.DataFrame(cur.fetchall(), columns=["week_start", "rank", "score"])

    def monthly_metrics(self, policy: str, capacity: str, limit: int = 200) -> pd.DataFrame:
        cur = self.conn.execute("SELECT month, reports_total, capture_any, capture_assigned FROM rw_monthly_metrics "
                                "WHERE policy = %s AND capacity = %s ORDER BY month LIMIT %s", (policy, capacity, _limit(limit)))
        return pd.DataFrame(cur.fetchall(), columns=["month", "reports_total", "capture_any", "capture_assigned"])


# ------------------------------------------------------------------ local fallback: same queries, same exports
class LocalSeries:
    def __init__(self, exports: dict[str, pd.DataFrame] | None = None, paths: Paths | None = None):
        self.x = exports if exports is not None else build_exports(paths)

    def weekly_reports(self, start: str, end: str, limit: int = MAX_ROWS) -> pd.DataFrame:
        w = self.x["rw_weekly_reports"]
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        return w[(w["week_start"] >= s) & (w["week_start"] < e)].sort_values("week_start").head(_limit(limit))[["week_start", "reports"]].reset_index(drop=True)

    def monthly_reports(self, limit: int = 200) -> pd.DataFrame:
        w = self.x["rw_weekly_reports"].copy()
        w["bucket"] = w["week_start"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
        g = w.groupby("bucket").agg(reports=("reports", "sum"), peak_week_reports=("reports", "max"), weeks=("reports", "size")).reset_index()
        return g.sort_values("bucket").head(_limit(limit)).reset_index(drop=True)

    def rank_history(self, block_id: int, policy: str = POLICY, limit: int = 200) -> pd.DataFrame:
        s = self.x["rw_block_scores"]
        return s[(s["block_id"] == int(block_id)) & (s["policy"] == policy)].sort_values("week_start").head(_limit(limit))[["week_start", "rank", "score"]].reset_index(drop=True)

    def monthly_metrics(self, policy: str, capacity: str, limit: int = 200) -> pd.DataFrame:
        m = self.x["rw_monthly_metrics"]
        return m[(m["policy"] == policy) & (m["capacity"] == capacity)].sort_values("month").head(_limit(limit))[
            ["month", "reports_total", "capture_any", "capture_assigned"]].reset_index(drop=True)


def measure(series, calls: dict, repeats: int = 20) -> dict:
    """Median and p95 latency (ms) of representative dashboard queries; reports what it measured, nothing else."""
    out = {}
    for name, fn in calls.items():
        t = []
        for _ in range(repeats):
            a = time.perf_counter()
            fn(series)
            t.append((time.perf_counter() - a) * 1000)
        t.sort()
        out[name] = {"median_ms": round(t[len(t) // 2], 2), "p95_ms": round(t[int(len(t) * 0.95) - 1], 2), "runs": repeats}
    return out
