"""As-of block-week features from 311 water and sewer COMPLAINTS (not confirmed leaks or flooding).

Spatial join (EPSG:2263, feet): each complaint is attributed to the NEAREST block within NEAR_FT, and, separately, to every
block within AREA_FT (wider-area context). Complaints without coordinates (1.3%) cannot be placed and are dropped (documented).
Availability rule: identical to the production 311 history features, a complaint is visible at forecast Monday t if it was
created before t - 2 days (created day <= t - 3 days); the window [t-2d-W, t-2d) is half-open. Counts are true zeros: the
source covers the whole period (both 311 taxonomies are mapped, see ingest.py).

Features, per physical group g in GROUPS:
  inf_<g>_n30 / n90 / n365   complaints attributed to the block in the window (count)
  inf_<g>_area365            complaints within AREA_FT of the block in the last 365 days (count; a complaint counts once per block)
  inf_<g>_days_since         days between the latest attributed complaint and the data cutoff (t - 2 days), capped at 365
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from ..config import Paths
from .ingest import GROUPS

NEAR_FT = 100.0       # ~30 m: a complaint sits on, or immediately beside, the block
AREA_FT = 300.0       # ~90 m: wider neighbourhood
LAG_DAYS = 2          # production 311 availability lag
WINDOWS = (30, 90, 365)
CAP = 365
EPOCH = dt.date(2019, 1, 1)


def feature_names() -> list[str]:
    out = []
    for g in GROUPS:
        out += [f"inf_{g}_n{w}" for w in WINDOWS] + [f"inf_{g}_area365", f"inf_{g}_days_since"]
    return out


def definitions() -> list[dict]:
    d = []
    for g in GROUPS:
        d += [{"feature": f"inf_{g}_n{w}", "units": "complaints", "window": f"[t-2d-{w}d, t-2d)", "aggregation": f"nearest block within {NEAR_FT:.0f} ft",
               "source": "NYC 311 erm2-nwe9 (water/sewer complaint types)", "availability": "created_date < t - 2 days"} for w in WINDOWS]
        d += [{"feature": f"inf_{g}_area365", "units": "complaints", "window": "[t-2d-365d, t-2d)", "aggregation": f"all blocks within {AREA_FT:.0f} ft",
               "source": "same", "availability": "created_date < t - 2 days"},
              {"feature": f"inf_{g}_days_since", "units": "days before the data cutoff t-2d (cap 365)", "window": "[t-2d-365d, t-2d)", "aggregation": f"nearest block within {NEAR_FT:.0f} ft",
               "source": "same", "availability": "created_date < t - 2 days"}]
    return d


def load_blocks(paths: Paths):
    b = pd.read_parquet(paths.processed / "stage3" / "blocks.parquet", columns=["block_id", "geometry"]).sort_values("block_id").reset_index(drop=True)
    return b["block_id"].to_numpy(), shapely.from_wkb(b["geometry"].to_numpy())


def attribute(geoms, x: np.ndarray, y: np.ndarray):
    """Return (near_block_idx per event or -1, area pairs (event_idx, block_idx))."""
    tree = STRtree(geoms)
    pts = shapely.points(x, y)
    idx = tree.query_nearest(pts, max_distance=NEAR_FT, all_matches=False)
    near = np.full(len(x), -1, dtype=np.int64)
    near[idx[0]] = idx[1]
    ev, bl = tree.query(pts, predicate="dwithin", distance=AREA_FT)
    return near, np.vstack([ev, bl])


def _day(ts: pd.Series) -> np.ndarray:
    return ((ts.dt.normalize() - pd.Timestamp(EPOCH)).dt.days).to_numpy(np.int64)


def _count_and_last(keys: np.ndarray, blocks: np.ndarray, hi_day: int, lo_day: int, span: int):
    """keys sorted = block*span+day. count of events per block with lo_day <= day <= hi_day, and last event day <= hi_day (or -1)."""
    if keys.size == 0:                       # a group without any event: true zeros, no previous event
        return np.zeros(len(blocks), np.int64), np.full(len(blocks), -1, np.int64)
    base = blocks * span
    c = np.searchsorted(keys, base + hi_day, side="right") - np.searchsorted(keys, base + lo_day, side="left")
    j = np.searchsorted(keys, base + hi_day, side="right") - 1
    ok = (j >= 0)
    last = np.where(ok, keys[np.clip(j, 0, None)] - base, -1)
    last = np.where((last >= 0) & (last <= hi_day) & ok & (keys[np.clip(j, 0, None)] // span == blocks), last, -1)
    return c, last


def build(paths: Paths, events: pd.DataFrame, origins: list[dt.date]) -> pd.DataFrame:
    """events: columns created (datetime), x, y, group. Returns block_id x origin feature table."""
    ids, geoms = load_blocks(paths)
    ev = events[events["group"].notna() & events["x"].notna() & events["y"].notna()].reset_index(drop=True)
    near, area = attribute(geoms, ev["x"].to_numpy(float), ev["y"].to_numpy(float))
    day = _day(ev["created"])
    span = int(max(day.max() if len(day) else 0, (max(origins) - EPOCH).days) + 10)
    nb = len(ids)
    blocks = np.arange(nb, dtype=np.int64)
    out = {"block_id": np.tile(ids, len(origins)), "origin": np.repeat(np.array(origins, dtype="datetime64[D]"), nb)}
    cols = {n: np.empty(nb * len(origins), np.float32) for n in feature_names()}
    for g in GROUPS:
        m = (ev["group"].to_numpy() == g)
        sel = m & (near >= 0)
        k_near = np.sort(near[sel] * span + day[sel])
        a_ev, a_bl = area[0], area[1]
        am = m[a_ev]
        k_area = np.sort(a_bl[am] * span + day[a_ev[am]])
        for oi, t in enumerate(origins):
            hi = (t - EPOCH).days - LAG_DAYS - 1                 # last visible created day: t-3
            sl = slice(oi * nb, (oi + 1) * nb)
            for w in WINDOWS:
                c, last = _count_and_last(k_near, blocks, hi, hi - w + 1, span)
                cols[f"inf_{g}_n{w}"][sl] = c
                if w == CAP:
                    ds = np.where(last >= 0, hi + 1 - last, CAP)             # days between creation day and the window end
                    cols[f"inf_{g}_days_since"][sl] = np.minimum(ds, CAP)
            ca, _ = _count_and_last(k_area, blocks, hi, hi - 365 + 1, span)
            cols[f"inf_{g}_area365"][sl] = ca
    out.update(cols)
    df = pd.DataFrame(out)
    df["origin"] = pd.to_datetime(df["origin"]).dt.date
    return df
