-- RoadWatch time-series serving layer for Tiger Data (PostgreSQL + TimescaleDB). Idempotent: safe to run repeatedly.
-- Canonical data stay in the archived snapshots and Parquet files; these tables are a derived, rebuildable serving copy.
-- Contains NO complaint text, addresses or street names: only block ids, ranks, scores, counts and metrics.
-- Hypertable syntax per the current Tiger Data docs: CREATE TABLE ... WITH (timescaledb.hypertable, timescaledb.partition_column=...).
-- Continuous aggregates cannot be created inside a transaction block; run this file with autocommit (roadwatch does).

CREATE TABLE IF NOT EXISTS rw_schema_version (version INT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());

-- Brooklyn weekly pothole reports (all test and history weeks)
CREATE TABLE IF NOT EXISTS rw_weekly_reports (
  week_start  TIMESTAMPTZ NOT NULL,
  reports     INT NOT NULL,
  week_status TEXT NOT NULL,
  PRIMARY KEY (week_start)
) WITH (timescaledb.hypertable, timescaledb.partition_column = 'week_start', timescaledb.chunk_interval = '1 year');

-- Saved ranking scores: top-N blocks per week per policy (a bounded subset, not all 23,448 blocks)
CREATE TABLE IF NOT EXISTS rw_block_scores (
  week_start TIMESTAMPTZ NOT NULL,
  policy     TEXT NOT NULL,
  block_id   BIGINT NOT NULL,
  rank       INT NOT NULL,
  score      DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (week_start, policy, block_id)
) WITH (timescaledb.hypertable, timescaledb.partition_column = 'week_start', timescaledb.chunk_interval = '6 months');
CREATE INDEX IF NOT EXISTS rw_block_scores_block ON rw_block_scores (block_id, week_start DESC);

-- Inspection selections (the saved worklists): which block was listed, when, under which policy and capacity
CREATE TABLE IF NOT EXISTS rw_selections (
  week_start TIMESTAMPTZ NOT NULL,
  policy     TEXT NOT NULL,
  capacity   TEXT NOT NULL,
  block_id   BIGINT NOT NULL,
  rank       INT NOT NULL,
  priority   TEXT NOT NULL,
  PRIMARY KEY (week_start, policy, capacity, block_id)
) WITH (timescaledb.hypertable, timescaledb.partition_column = 'week_start', timescaledb.chunk_interval = '6 months');
CREATE INDEX IF NOT EXISTS rw_selections_block ON rw_selections (block_id, week_start DESC);

-- Model evaluation over time (monthly capture per policy and capacity, from the saved evaluation outputs)
CREATE TABLE IF NOT EXISTS rw_monthly_metrics (
  month           TIMESTAMPTZ NOT NULL,
  policy          TEXT NOT NULL,
  capacity        TEXT NOT NULL,
  reports_total   INT NOT NULL,
  capture_any     DOUBLE PRECISION NOT NULL,
  capture_assigned DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (month, policy, capacity)
) WITH (timescaledb.hypertable, timescaledb.partition_column = 'month', timescaledb.chunk_interval = '1 year');

-- Continuous aggregate: monthly report volume, with its maximum week (surge detection)
CREATE MATERIALIZED VIEW IF NOT EXISTS rw_monthly_reports
WITH (timescaledb.continuous) AS
SELECT time_bucket(INTERVAL '1 month', week_start) AS bucket,
       sum(reports) AS reports, max(reports) AS peak_week_reports, count(*) AS weeks
FROM rw_weekly_reports
GROUP BY bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('rw_monthly_reports',
  start_offset => INTERVAL '3 years', end_offset => INTERVAL '1 day', schedule_interval => INTERVAL '1 day',
  if_not_exists => true);

INSERT INTO rw_schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
