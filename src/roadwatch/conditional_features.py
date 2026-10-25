"""EXPERIMENTAL, ASSUMPTION-DEPENDENT features from conditional DOT sources (never part of the primary model).

Availability delays are ASSUMPTIONS (config.AVAILABILITY_LAGS; publication dates are unverifiable, audit §11a.4):
  pavement ratings     inspection date        <= t - 45 days
  pothole work orders  rptclosed date         <= t - 35 days   (closed-only dataset; open orders invisible)
  in-house resurfacing paving end date        <= t - 21 days   (in-house crews only; contract paving missing)
Only event dates are used - never current statuses, repair outcomes or retrospectively updated fields.
Block mapping: source geometry midpoint -> nearest eligible block within 30 ft, preferring the block whose CSCL b5sc
equals the first 6 digits of the source OFT code (audit §4: 95.1% same-code match for pavement).
"""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .features import DAY0, EventIndex, FeatureDef, _d

LAG_PAVEMENT, LAG_WORKORDER, LAG_PAVING = 45, 35, 21
EXPERIMENTAL = ["pav_rating_latest", "pav_rating_age_days", "pav_rating_missing", "wo_closed_365d",
                "wo_days_since_last_closed", "paving_days_since_last", "paving_never_in_record"]


def map_to_blocks(geoms: gpd.GeoSeries, oft: pd.Series, blocks: gpd.GeoDataFrame, max_ft: float = 30.0) -> np.ndarray:
    """Return block positions (or -1) for each source geometry."""
    g = geoms.to_crs("EPSG:2263").to_numpy()
    ok = ~pd.isna(g)
    ok[ok] = ~shapely.is_empty(g[ok])
    mid = np.full(len(g), None, dtype=object)
    tid = np.full(len(g), -1)
    tid[ok] = shapely.get_type_id(g[ok])
    lines = ok & np.isin(tid, [1, 5])            # LineString / MultiLineString -> midpoint along the line
    pts = ok & (tid == 0)                        # Point (e.g. intersection resurfacing) -> itself
    other = ok & ~lines & ~pts                   # anything else -> centroid
    if lines.any():
        mid[lines] = shapely.line_interpolate_point(g[lines], 0.5, normalized=True)
    mid[pts] = g[pts]
    if other.any():
        mid[other] = shapely.centroid(g[other])
    tree = STRtree(blocks.geometry.to_numpy())
    b5 = blocks["b5sc"].to_numpy()
    out = np.full(len(g), -1, dtype=np.int64)
    idx = np.flatnonzero(ok)
    qi, bi = tree.query(mid[idx], predicate="dwithin", distance=max_ft)
    dist = shapely.distance(mid[idx][qi], blocks.geometry.to_numpy()[bi])
    cand = pd.DataFrame({"q": idx[qi], "b": bi, "d": dist})
    code = oft.astype(str).str[:6].to_numpy()
    cand["same"] = b5[cand["b"]] == code[cand["q"]]
    cand = cand.sort_values(["q", "same", "d"], ascending=[True, False, True]).drop_duplicates("q")
    out[cand["q"].to_numpy()] = cand["b"].to_numpy()
    return out


class LastValueIndex:
    """Latest value strictly before a cutoff day, per block."""

    def __init__(self, block_pos, day, value, n):
        key = block_pos.astype(np.int64) * 100_000 + day.astype(np.int64)
        o = np.argsort(key, kind="stable")
        self.key, self.val, self.bp = key[o], np.asarray(value, float)[o], np.arange(n, dtype=np.int64)

    def last(self, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
        i = np.searchsorted(self.key, self.bp * 100_000 + cutoff, side="left") - 1
        ok = (i >= 0) & (self.key[np.clip(i, 0, None)] // 100_000 == self.bp)
        v, age = np.full(len(self.bp), np.nan), np.full(len(self.bp), np.nan)
        v[ok] = self.val[i[ok]]
        age[ok] = cutoff - self.key[i[ok]] % 100_000
        return v, age


class ConditionalFeatureBuilder:
    def __init__(self, blocks: gpd.GeoDataFrame, pavement: gpd.GeoDataFrame, workorders: gpd.GeoDataFrame,
                 paving: gpd.GeoDataFrame):
        self.blocks = blocks.sort_values("block_id").reset_index(drop=True)
        n = len(self.blocks)
        self.n = n
        self.mapping_stats = {}
        p = pavement[pavement["rating_value"].notna() & pavement["inspection_local"].notna()]
        pb = map_to_blocks(p.geometry, p["oftcode"], self.blocks)
        self.mapping_stats["pavement_rated_rows"] = (len(p), int((pb >= 0).sum()))
        keep = pb >= 0
        self.pav = LastValueIndex(pb[keep], (p["inspection_local"][keep] - pd.Timestamp(DAY0)).dt.days.to_numpy(),
                                  p["rating_value"][keep].to_numpy(), n)
        w = workorders[workorders["rptclosed_date"].notna()]
        wb = map_to_blocks(w.geometry, w["oft"], self.blocks)
        self.mapping_stats["workorders_closed_rows"] = (len(w), int((wb >= 0).sum()))
        keep = wb >= 0
        wd = (w["rptclosed_date"][keep] - pd.Timestamp(DAY0)).dt.days.to_numpy()
        valid = wd >= -3650
        self.wo = EventIndex(wb[keep][valid], wd[valid] + 3650, np.ones(int(valid.sum())), n)   # shift: pre-2020 dates
        r = paving[paving["location_actual_paving_end_date_local"].notna()]
        rb = map_to_blocks(r.geometry, r["oft_code"], self.blocks)
        self.mapping_stats["paving_rows_with_end_date"] = (len(r), int((rb >= 0).sum()))
        keep = rb >= 0
        rd = (r["location_actual_paving_end_date_local"][keep] - pd.Timestamp(DAY0)).dt.days.to_numpy() + 3650
        self.pv = EventIndex(rb[keep], rd, np.ones(int(keep.sum())), n)

    def build(self, origins: list[dt.date]) -> pd.DataFrame:
        frames = []
        for t in origins:
            v, age = self.pav.last(_d(t) - LAG_PAVEMENT + 1)     # inspection day <= t - 45 d (inclusive)
            wo_c = _d(t) - LAG_WORKORDER + 3650 + 1          # inclusive of the day exactly at t - 35 d
            pv_c = _d(t) - LAG_PAVING + 3650 + 1
            ds_wo = self.wo.days_since_last(wo_c)
            ds_pv = self.pv.days_since_last(pv_c)
            frames.append(pd.DataFrame({
                "block_id": self.blocks["block_id"].to_numpy(), "origin": t,
                "pav_rating_latest": v.astype(np.float32), "pav_rating_age_days": age.astype(np.float32),
                "pav_rating_missing": np.isnan(v).astype(np.float32),
                "wo_closed_365d": self.wo.window(wo_c, 365).astype(np.float32),
                "wo_days_since_last_closed": np.where(np.isnan(ds_wo), 3650, ds_wo).astype(np.float32),
                "paving_days_since_last": np.where(np.isnan(ds_pv), 7300, ds_pv).astype(np.float32),
                "paving_never_in_record": np.isnan(ds_pv).astype(np.float32)}))
        return pd.concat(frames, ignore_index=True)


def experimental_definitions() -> list[FeatureDef]:
    src = "ASSUMED availability"
    return [
        FeatureDef("pav_rating_latest", "experimental", "latest", "rating 1-10", "6yyb-pb25", f"{src}: inspection <= t-45d", "latest rating (0 = not rated excluded)"),
        FeatureDef("pav_rating_age_days", "experimental", "latest", "days", "6yyb-pb25", f"{src}: inspection <= t-45d", "age of latest rating at t-45d"),
        FeatureDef("pav_rating_missing", "experimental", "latest", "0/1", "6yyb-pb25", f"{src}", "no available rating"),
        FeatureDef("wo_closed_365d", "experimental", "365 d", "work orders", "x9wy-ing4", f"{src}: rptclosed <= t-35d", "closed pothole work orders"),
        FeatureDef("wo_days_since_last_closed", "experimental", "all", "days (3650 = none)", "x9wy-ing4", f"{src}: rptclosed <= t-35d", "recency"),
        FeatureDef("paving_days_since_last", "experimental", "all", "days (7300 = none)", "ffaf-8mrv", f"{src}: paving end <= t-21d", "in-house paving recency"),
        FeatureDef("paving_never_in_record", "experimental", "all", "0/1", "ffaf-8mrv", f"{src}", "no in-house paving on record (contract paving missing!)"),
    ]
