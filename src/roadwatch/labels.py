"""Weekly block-level report-risk labels with explicit observability.

Calendar: origin t = Monday 00:00 America/New_York; horizon = [t, t + 7 calendar days) in local wall-clock time.
311 `created_date` is a floating NYC wall-clock timestamp (Stage 2), so a report belongs to the horizon whose local
Monday-to-Monday window contains it - no UTC conversion is involved. DST changes (Sunday 02:00 local) fall inside a
horizon, never on its boundary, so a horizon spans 167, 168 or 169 elapsed UTC hours.

Label per (block, origin):
  1        >= 1 qualifying report assigned to the block in the horizon (always, even in degraded weeks)
  0        no assigned report AND the horizon is observable AND the block is not an unresolved candidate
  <NA>     unknown: horizon incomplete (data cutoff), horizon affected by unlocatable reports, or (confident-only
           variant) the block is a candidate of an ambiguous report. Unknown is never converted to 0.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
WEEK_STATUSES = ["observable", "unlocated_reports", "incomplete_final"]
LABEL_STATUSES = ["positive", "negative", "unknown"]
UNKNOWN_REASONS = ["none", "unlocated_reports", "incomplete_final", "ambiguous_candidate"]


@dataclass(frozen=True)
class LabelParams:
    embargo_days: int = 2                 # 311 availability lag (config.AVAILABILITY_LAGS['sr311'])
    hidden_share_threshold: float = 0.05  # max share of potentially-Brooklyn reports that are unlocatable


def week_origin(ts_local: pd.Series) -> pd.Series:
    """Monday (date) of the local week containing each naive local timestamp."""
    d = ts_local.dt.normalize()
    return (d - pd.to_timedelta(d.dt.day_of_week, unit="D")).dt.date


def first_monday_on_or_after(d: dt.date) -> dt.date:
    return d + dt.timedelta(days=(7 - d.weekday()) % 7)


def utc_hours(origin: dt.date) -> float:
    a = dt.datetime.combine(origin, dt.time(0), NY)
    b = dt.datetime.combine(origin + dt.timedelta(days=7), dt.time(0), NY)
    return (b.astimezone(dt.UTC) - a.astimezone(dt.UTC)).total_seconds() / 3600


def build_calendar(data_start: dt.date, data_end_exclusive: dt.date, embargo_days: int) -> pd.DataFrame:
    """Origins from the first Monday on/after data_start while the horizon starts before data_end_exclusive.
    A horizon is data-complete only if t + 7 <= data_end_exclusive - embargo_days (reports need time to publish)."""
    observable_until = data_end_exclusive - dt.timedelta(days=embargo_days)
    origins, t = [], first_monday_on_or_after(data_start)
    while t < data_end_exclusive:
        origins.append(t)
        t += dt.timedelta(days=7)
    w = pd.DataFrame({"origin": origins})
    w["horizon_end"] = [o + dt.timedelta(days=7) for o in origins]
    w["utc_hours"] = [utc_hours(o) for o in origins]
    w["data_complete"] = [e <= observable_until for e in w["horizon_end"]]
    w.attrs["observable_until"] = observable_until.isoformat()
    return w


def assign_week_status(weeks: pd.DataFrame, located_per_week: pd.Series, hidden_per_week: pd.Series,
                       threshold: float) -> pd.DataFrame:
    """hidden_share = H / (H + L): H = potentially-Brooklyn reports that cannot be located, L = Brooklyn reports
    known to be in Brooklyn (matched or not). Upper bound: every H report is assumed to be in Brooklyn."""
    w = weeks.copy()
    w["located_brooklyn_reports"] = w["origin"].map(located_per_week).fillna(0).astype(int)
    w["hidden_potential_brooklyn"] = w["origin"].map(hidden_per_week).fillna(0).astype(int)
    tot = w["located_brooklyn_reports"] + w["hidden_potential_brooklyn"]
    w["hidden_share_upper_bound"] = np.where(tot > 0, w["hidden_potential_brooklyn"] / tot.clip(lower=1), 0.0)
    status = np.where(~w["data_complete"], "incomplete_final",
                      np.where(w["hidden_share_upper_bound"] > threshold, "unlocated_reports", "observable"))
    w["week_status"] = pd.Categorical(status, categories=WEEK_STATUSES)
    return w


def build_panel(block_ids: np.ndarray, weeks: pd.DataFrame, positives: pd.DataFrame,
                ambiguous_candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    """positives: rows (block_id, origin, report_id) - one row per report per block (dedup applied here).
    ambiguous_candidates: rows (block_id, origin) whose label is unknown unless positive (confident-only variant)."""
    block_ids = np.asarray(block_ids, dtype="int64")
    bpos = pd.Series(np.arange(len(block_ids)), index=block_ids)
    wpos = pd.Series(np.arange(len(weeks)), index=pd.Index(weeks["origin"]))
    nb, nw = len(block_ids), len(weeks)
    pos = positives.drop_duplicates(["block_id", "origin", "report_id"])
    pos = pos[pos["block_id"].isin(bpos.index) & pos["origin"].isin(wpos.index)]
    counts = np.zeros((nw, nb), dtype=np.int32)
    np.add.at(counts, (wpos[pos["origin"]].to_numpy(), bpos[pos["block_id"]].to_numpy()), 1)
    wstat = weeks["week_status"].astype(str).to_numpy()
    reason = np.full((nw, nb), "none", dtype=object)
    reason[wstat == "unlocated_reports", :] = "unlocated_reports"
    reason[wstat == "incomplete_final", :] = "incomplete_final"
    if ambiguous_candidates is not None and len(ambiguous_candidates):
        ac = ambiguous_candidates[ambiguous_candidates["block_id"].isin(bpos.index) & ambiguous_candidates["origin"].isin(wpos.index)]
        wi, bi = wpos[ac["origin"]].to_numpy(), bpos[ac["block_id"]].to_numpy()
        cur = reason[wi, bi]
        reason[wi, bi] = np.where(cur == "none", "ambiguous_candidate", cur)
    positive = counts > 0
    reason[positive] = "none"
    label = np.where(positive, 1, np.where(reason == "none", 0, -1)).astype(np.int8)
    status = np.where(label == 1, "positive", np.where(label == 0, "negative", "unknown"))
    panel = pd.DataFrame({
        "block_id": np.tile(block_ids, nw),
        "origin": np.repeat(weeks["origin"].to_numpy(), nb),
        "label": pd.arrays.IntegerArray(label.ravel().clip(0).astype(np.int8), label.ravel() < 0),
        "label_status": pd.Categorical(status.ravel(), categories=LABEL_STATUSES),
        "unknown_reason": pd.Categorical(reason.ravel(), categories=UNKNOWN_REASONS),
        "n_reports": counts.ravel().astype(np.int16),
        "week_status": pd.Categorical(np.repeat(wstat, nb), categories=WEEK_STATUSES),
    })
    return panel
