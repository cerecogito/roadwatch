"""Leakage-resistant, as-of feature generation for block-week forecasts.

Prediction timestamp: t = forecast origin, Monday 00:00 America/New_York (labels.py calendar).
Availability rules (config.AVAILABILITY_LAGS):
  311 reports   created_date (local) <  t - 2 days            -> cutoff day c311 = t - 2 days (exclusive)
  NOAA daily    observation DATE      <= t - 5 days            (observed ~4-day lag + 1 day margin)
  label weeks   a past week w is "matured" only if w + 7 days + 2 days <= t, i.e. w <= t - 14 days
Static CSCL attributes are a 2026 snapshot (anachronism documented, not leakage of outcomes).

Historical report attribution is FIXED for every target experiment (it never depends on the target variant):
  frac  each included report adds 1/k to each of its k candidate blocks (k = 1 for unique matches)
  any   each included report adds 1 to every candidate block
  uniq  only uniquely matched reports add 1
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely

DAY0 = dt.date(2020, 1, 1)
KEY = 100_000                      # > number of days in the archive
LAG_311_DAYS = 2
LAG_NOAA_DAYS = 5
CAP_DAYS_SINCE = 365


@dataclass(frozen=True)
class FeatureDef:
    name: str
    group: str        # history | calendar | weather | static | experimental
    window: str
    units: str
    source: str
    availability: str
    definition: str


def _d(x) -> int:
    return (x - DAY0).days


class EventIndex:
    """Weighted events keyed by (block position, day). Window sums and recency via binary search - no dense matrix."""

    def __init__(self, block_pos: np.ndarray, day: np.ndarray, weight: np.ndarray, n_blocks: int,
                 half_lives: tuple[int, ...] = ()):
        key = block_pos.astype(np.int64) * KEY + day.astype(np.int64)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        w = weight[order].astype(np.float64)
        self.cw = np.concatenate([[0.0], np.cumsum(w)])
        self.d = day[order].astype(np.float64)
        self.blk = self.key // KEY
        # segmented (per-block) cumulative sums of w * 2^(day/h): no cross-block prefix, so no cancellation error
        self.ewm = {}
        for h in half_lives:
            g = pd.Series(w * np.exp(self.d * np.log(2) / h)).groupby(self.blk).cumsum().to_numpy()
            self.ewm[h] = g
        self.bp = np.arange(n_blocks, dtype=np.int64)

    def _idx(self, day: int) -> np.ndarray:
        return np.searchsorted(self.key, self.bp * KEY + day, side="left")

    def window(self, cutoff: int, days: int) -> np.ndarray:
        """Sum of weights with cutoff - days <= day < cutoff."""
        return self.cw[self._idx(cutoff)] - self.cw[self._idx(cutoff - days)]

    def decayed(self, cutoff: int, h: int) -> np.ndarray:
        """Sum of w * 0.5 ** ((cutoff - day) / h) over all events with day < cutoff."""
        j = self._idx(cutoff) - 1
        ok = (j >= 0) & (self.blk[np.clip(j, 0, None)] == self.bp) if len(self.key) else np.zeros(len(self.bp), bool)
        out = np.zeros(len(self.bp))
        out[ok] = self.ewm[h][j[ok]] * np.exp(-cutoff * np.log(2) / h)
        return out

    def days_since_last(self, cutoff: int) -> np.ndarray:
        """Days between the latest event strictly before cutoff and cutoff; NaN if none."""
        out = np.full(len(self.bp), np.nan)
        if not len(self.key):
            return out
        i = self._idx(cutoff) - 1
        ok = (i >= 0) & (self.key[np.clip(i, 0, None)] // KEY == self.bp)
        out[ok] = cutoff - (self.key[i[ok]] % KEY)
        return out


class FeatureBuilder:
    def __init__(self, assignments: pd.DataFrame, weeks: pd.DataFrame, blocks: pd.DataFrame, noaa: pd.DataFrame):
        self.blocks = blocks.sort_values("block_id").reset_index(drop=True)
        self.block_ids = self.blocks["block_id"].astype("int64").to_numpy()
        self.n = len(self.block_ids)
        pos = pd.Series(np.arange(self.n), index=self.block_ids)
        a = assignments[assignments["included_in_labels"]].copy()
        a["day"] = (pd.to_datetime(a["created_date_local"]).dt.normalize() - pd.Timestamp(DAY0)).dt.days
        ex = a[["report_id", "day", "match_status", "candidate_block_ids", "n_candidates"]].explode("candidate_block_ids")
        ex = ex[ex["candidate_block_ids"].isin(pos.index)]
        bp = pos[ex["candidate_block_ids"].astype("int64")].to_numpy()
        day = ex["day"].to_numpy()
        k = ex["n_candidates"].to_numpy().astype(float)
        uniq = (ex["match_status"] == "matched").to_numpy()
        self.frac = EventIndex(bp, day, 1.0 / k, self.n, half_lives=(7, 30, 90))
        self.any = EventIndex(bp, day, np.ones(len(bp)), self.n)
        self.uniq = EventIndex(bp[uniq], day[uniq], np.ones(int(uniq.sum())), self.n)
        self.report_days = np.sort(a["day"].to_numpy())           # Brooklyn-wide activity (unique reports)
        # weekly any-candidate positives for observed positive-week rates
        weeks = weeks.sort_values("origin").reset_index(drop=True)
        self.week_origins = pd.to_datetime(weeks["origin"]).dt.date.to_numpy()
        self.week_idx = {w: i for i, w in enumerate(self.week_origins)}
        self.week_observable = (weeks["week_status"].astype(str) == "observable").to_numpy()
        self.week_unlocated = (weeks["week_status"].astype(str) == "unlocated_reports").to_numpy()
        wk =np.array([_d(w) for w in self.week_origins])
        widx = np.searchsorted(wk, day, side="right") - 1
        valid = widx >= 0
        pw = pd.DataFrame({"bp": bp[valid], "w": widx[valid]}).drop_duplicates()
        obs_w = self.week_observable[pw["w"].to_numpy()].astype(float)
        self.posweek = EventIndex(pw["bp"].to_numpy(), pw["w"].to_numpy(), obs_w, self.n)
        self.week_start_day = wk
        # neighbours: blocks sharing a centerline node; 2,000 ft grid cells on block midpoints
        nf, nt = self.blocks["node_from"].to_numpy(), self.blocks["node_to"].to_numpy()
        inc = sp.csr_matrix((np.ones(2 * self.n), (np.r_[nf, nt], np.r_[np.arange(self.n), np.arange(self.n)])))
        adj = (inc.T @ inc).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        self.adj = (adj > 0).astype(np.float32)
        mid = shapely.line_interpolate_point(self.blocks.geometry.to_numpy(), 0.5, normalized=True)
        cx, cy = (shapely.get_x(mid) // 2000).astype(int), (shapely.get_y(mid) // 2000).astype(int)
        self.cell = pd.factorize(pd.Series(cx).astype(str) + "_" + pd.Series(cy).astype(str))[0]
        self.n_adjacent = np.asarray(self.adj.sum(axis=1)).ravel()
        self.static = self._static()
        self.noaa = noaa.sort_values("date").reset_index(drop=True)
        self.noaa_day = (self.noaa["date"] - pd.Timestamp(DAY0)).dt.days.to_numpy()

    # ---------------- static ----------------
    def _static(self) -> pd.DataFrame:
        b = self.blocks
        st =pd.DataFrame({
            "length_m": b["length_m"].astype("float32"),
            "street_width_ft": b["streetwidth_num"].astype("float32"),
            "travel_lanes": b["number_travel_lanes_num"].astype("float32"),
            "parking_lanes": b["number_park_lanes_num"].astype("float32"),
            "total_lanes": b["number_total_lanes_num"].astype("float32"),
            "posted_speed_mph": b["posted_speed_num"].astype("float32"),
            "truck_route": b["truck_route_type"].notna().astype("float32"),
            "bike_lane": b["bike_lane"].notna().astype("float32"),
            "one_way": b["trafdir"].isin(["FT", "TF"]).astype("float32"),
            "n_adjacent_blocks": self.n_adjacent.astype("float32"),
        })
        typ = b["post_type"].fillna(b["pre_type"]).fillna("NONE")
        top = ["ST", "AVE", "RD", "PL", "BLVD", "PKWY", "CT", "DR"]
        st["street_type_code"] = pd.Categorical(np.where(typ.isin(top), typ, "OTHER"), categories=top + ["OTHER"]).codes.astype("float32")
        sp_ = b["snow_priority"].fillna("NONE")
        st["snow_priority_code"] = pd.Categorical(sp_, categories=sorted(sp_.unique())).codes.astype("float32")
        return st

    # ---------------- per-origin computation ----------------
    def history(self, t: dt.date) -> dict[str, np.ndarray]:
        c = _d(t) - LAG_311_DAYS
        f = {}
        for w in (7, 30, 90, 365):
            f[f"rep_frac_{w}d"] = self.frac.window(c, w)
        f["rep_any_30d"], f["rep_any_365d"] = self.any.window(c, 30), self.any.window(c, 365)
        f["rep_unique_365d"] = self.uniq.window(c, 365)
        for h in (7, 30, 90):
            f[f"rep_decay_hl{h}d"] = self.frac.decayed(c, h)
        # fixed 365-day horizon: identical meaning for every origin from 2021-01-04 (archive starts 2020-01-01);
        # an unbounded "never reported" would drift with archive length (non-stationary)
        ds = self.any.days_since_last(c)
        none = np.isnan(ds) | (ds > CAP_DAYS_SINCE)
        f["no_report_in_365d"] = none.astype(float)
        f["days_since_last_report"] = np.where(none, CAP_DAYS_SINCE, ds)
        f["rep_rate_change_90v365"] = f["rep_frac_90d"] / 90 - f["rep_frac_365d"] / 365
        # observed positive-week rate over the 52 most recent MATURED weeks (w <= t - 14 d), unknown weeks excluded
        i_last = self.week_idx.get(t - dt.timedelta(days=14))
        if i_last is None:
            i_last = int(np.searchsorted(self.week_start_day, _d(t) - 14, side="right") - 1)
        lo = max(i_last - 51, 0)
        n_obs = int(self.week_observable[lo:i_last + 1].sum()) if i_last >= 0 else 0
        pos_obs = self.posweek.window(i_last + 1, 52) if i_last >= 0 else np.zeros(self.n)
        f["obs_weeks_52"] = np.full(self.n, n_obs, float)
        f["pos_week_rate_52"] = pos_obs / n_obs if n_obs else np.full(self.n, np.nan)
        f["unknown_weeks_52"] = np.full(self.n, (i_last + 1 - lo) - n_obs if i_last >= 0 else 0, float)
        # lookback windows touching unlocated-report weeks (time-level; never treated as zero activity)
        win_lo = c - 365
        touch = (self.week_start_day + 7 > win_lo) & (self.week_start_day < c) & self.week_unlocated
        f["unlocated_weeks_in_365d"] = np.full(self.n, int(touch.sum()), float)
        f["history_window_compromised"] = (f["unlocated_weeks_in_365d"] > 0).astype(float)
        f["history_days_available"] = np.full(self.n, min(c - 0, 365), float)
        for w in (30, 365):
            f[f"nbr_frac_{w}d"] = self.adj @ f[f"rep_frac_{w}d"]
        for w in (90, 365):
            tot = np.bincount(self.cell, weights=f[f"rep_frac_{w}d"])
            f[f"cell_frac_{w}d"] = tot[self.cell] - f[f"rep_frac_{w}d"]
        for w in (7, 28):
            n = np.searchsorted(self.report_days, c) - np.searchsorted(self.report_days, c - w)
            f[f"bk_reports_{w}d"] = np.full(self.n, n, float)
        return f

    def calendar(self, t: dt.date) -> dict[str, np.ndarray]:
        woy = t.isocalendar()[1]
        return {"woy_sin": np.full(self.n, np.sin(2 * np.pi * woy / 52.18)),
                "woy_cos": np.full(self.n, np.cos(2 * np.pi * woy / 52.18))}

    def weather(self, t: dt.date) -> dict[str, np.ndarray]:
        last = _d(t) - LAG_NOAA_DAYS                       # inclusive last available observation day
        m = self.noaa_day <= last
        nd = self.noaa_day[m]
        tmin = self.noaa["tmin_value"].to_numpy()[m]
        tmax = self.noaa["tmax_value"].to_numpy()[m]
        prcp = self.noaa["prcp_value"].to_numpy()[m]
        snow = self.noaa["snow_value"].to_numpy()[m]

        def win(days):
            return nd > last - days

        out = {}
        w7, w14, w30, w60 = win(7), win(14), win(30), win(60)
        out["tmin_min_7d"] = np.nanmin(tmin[w7]) if w7.any() else np.nan
        out["tmin_mean_7d"] = np.nanmean(tmin[w7]) if w7.any() else np.nan
        out["tmax_mean_7d"] = np.nanmean(tmax[w7]) if w7.any() else np.nan
        out["tmin_mean_30d"] = np.nanmean(tmin[w30]) if w30.any() else np.nan
        fz, ft = tmin < 0, (tmin < 0) & (tmax > 0)
        for nm, w in (("14d", w14), ("30d", w30), ("60d", w60)):
            out[f"freeze_days_{nm}"] = float(fz[w].sum())
            out[f"freeze_thaw_days_{nm}"] = float(ft[w].sum())
        for nm, w in (("7d", w7), ("14d", w14), ("30d", w30)):
            out[f"prcp_mm_{nm}"] = float(np.nansum(prcp[w]))
        out["wet_days_14d"] = float((prcp[w14] >= 1.0).sum())
        run = 0
        for v in prcp[::-1]:
            if v >= 1.0:
                run += 1
            else:
                break
        out["consecutive_wet_days"] = float(run)
        out["snow_mm_30d"] = float(np.nansum(snow[w30]))
        out["freeze_thaw_x_prcp_30d"] = out["freeze_thaw_days_30d"] * out["prcp_mm_30d"]
        out["weather_days_available_30d"] = float(w30.sum())
        return {k: np.full(self.n, v, float) for k, v in out.items()}

    def build(self, origins: list[dt.date]) -> pd.DataFrame:
        frames = []
        for t in origins:
            cols = {"block_id": self.block_ids, "origin": np.full(self.n, np.datetime64(t, "D"))}
            for part in (self.history(t), self.calendar(t), self.weather(t)):
                cols.update({k: np.asarray(v, dtype=np.float32) for k, v in part.items()})
            df = pd.DataFrame(cols)
            frames.append(pd.concat([df, self.static], axis=1))
        out = pd.concat(frames, ignore_index=True)
        out["origin"] = pd.to_datetime(out["origin"]).dt.date
        return out


HIST = ["rep_frac_7d", "rep_frac_30d", "rep_frac_90d", "rep_frac_365d", "rep_any_30d", "rep_any_365d", "rep_unique_365d",
        "rep_decay_hl7d", "rep_decay_hl30d", "rep_decay_hl90d", "no_report_in_365d", "days_since_last_report",
        "rep_rate_change_90v365", "obs_weeks_52", "pos_week_rate_52", "unknown_weeks_52", "unlocated_weeks_in_365d",
        "history_window_compromised", "nbr_frac_30d", "nbr_frac_365d", "cell_frac_90d", "cell_frac_365d",
        "bk_reports_7d", "bk_reports_28d"]
CAL = ["woy_sin", "woy_cos"]
WEATHER = ["tmin_min_7d", "tmin_mean_7d", "tmax_mean_7d", "tmin_mean_30d", "freeze_days_14d", "freeze_thaw_days_14d",
           "freeze_days_30d", "freeze_thaw_days_30d", "freeze_days_60d", "freeze_thaw_days_60d", "prcp_mm_7d", "prcp_mm_14d",
           "prcp_mm_30d", "wet_days_14d", "consecutive_wet_days", "snow_mm_30d", "freeze_thaw_x_prcp_30d"]
STATIC = ["length_m", "street_width_ft", "travel_lanes", "parking_lanes", "total_lanes", "posted_speed_mph", "truck_route",
          "bike_lane", "one_way", "n_adjacent_blocks", "street_type_code", "snow_priority_code"]
CATEGORICAL = ["street_type_code", "snow_priority_code"]
DIAGNOSTIC_ONLY = ["history_days_available", "weather_days_available_30d"]   # coverage indicators, not model inputs

FEATURE_SETS = {
    "history": HIST + CAL,
    "history_weather": HIST + CAL + WEATHER,
    "history_static": HIST + CAL + STATIC,
    "core": HIST + CAL + WEATHER + STATIC,
}


def feature_definitions() -> list[FeatureDef]:
    A311 = "reports with created_date < t - 2 days (2-day 311 lag)"
    ANOAA = "NOAA DATE <= t - 5 days (observed ~4-day lag + 1 day)"
    defs = []
    for w in (7, 30, 90, 365):
        defs.append(FeatureDef(f"rep_frac_{w}d", "history", f"[t-2d-{w}d, t-2d)", "reports (fractional)", "311 + Stage-3 candidates",
                               A311, "sum over reports of 1/k for each of the report's k candidate blocks"))
    defs += [
        FeatureDef("rep_any_30d", "history", "[t-2d-30d, t-2d)", "reports", "311 + Stage-3 candidates", A311, "reports listing the block as any candidate"),
        FeatureDef("rep_any_365d", "history", "[t-2d-365d, t-2d)", "reports", "311 + Stage-3 candidates", A311, "as rep_any_30d"),
        FeatureDef("rep_unique_365d", "history", "[t-2d-365d, t-2d)", "reports", "311 + Stage-3 matching", A311, "uniquely matched reports only"),
        *[FeatureDef(f"rep_decay_hl{h}d", "history", "all history before t-2d", "decayed reports", "311", A311,
                     f"sum of 1/k * 0.5^(age/{h} days)") for h in (7, 30, 90)],
        FeatureDef("no_report_in_365d", "history", "[t-2d-365d, t-2d)", "0/1", "311", A311, "no candidate report in the last 365 days"),
        FeatureDef("days_since_last_report", "history", "[t-2d-365d, t-2d)", "days (capped 365)", "311", A311, "days since latest candidate report"),
        FeatureDef("rep_rate_change_90v365", "history", "90 vs 365 days", "reports/day", "311", A311, "rep_frac_90d/90 - rep_frac_365d/365"),
        FeatureDef("obs_weeks_52", "history", "52 matured weeks (w <= t-14d)", "weeks", "Stage-3 label_weeks", "week matured: w+7d+2d <= t",
                   "observable weeks in window (unknown weeks excluded)"),
        FeatureDef("pos_week_rate_52", "history", "52 matured weeks", "share", "311 + label_weeks", "week matured",
                   "observable weeks with >= 1 candidate report / observable weeks; unknown weeks in neither numerator nor denominator"),
        FeatureDef("unknown_weeks_52", "history", "52 matured weeks", "weeks", "label_weeks", "week matured", "non-observable weeks in window"),
        FeatureDef("unlocated_weeks_in_365d", "history", "[t-2d-365d, t-2d)", "weeks", "label_weeks", A311,
                   "unlocated-report weeks overlapping the 365-day window (history possibly deflated)"),
        FeatureDef("history_window_compromised", "history", "365 days", "0/1", "label_weeks", A311, "unlocated_weeks_in_365d > 0"),
        *[FeatureDef(f"nbr_frac_{w}d", "history", f"{w} days", "reports (fractional)", "311 + CSCL topology", A311,
                     "sum of rep_frac over blocks sharing a centerline node") for w in (30, 365)],
        *[FeatureDef(f"cell_frac_{w}d", "history", f"{w} days", "reports (fractional)", "311 + 2,000 ft grid", A311,
                     "sum of rep_frac over other blocks whose midpoint is in the same 2,000 ft cell") for w in (90, 365)],
        *[FeatureDef(f"bk_reports_{w}d", "history", f"{w} days", "reports", "311", A311, "Brooklyn-wide included reports (time-level)") for w in (7, 28)],
        FeatureDef("woy_sin", "calendar", "t", "-", "calendar", "known", "sin(2*pi*ISO week/52.18)"),
        FeatureDef("woy_cos", "calendar", "t", "-", "calendar", "known", "cos(2*pi*ISO week/52.18)"),
    ]
    wdefs = {"tmin_min_7d": "min TMIN, degC", "tmin_mean_7d": "mean TMIN, degC", "tmax_mean_7d": "mean TMAX, degC",
             "tmin_mean_30d": "mean TMIN, degC", "freeze_days_14d": "days TMIN<0", "freeze_thaw_days_14d": "days TMIN<0<TMAX (proxy)",
             "freeze_days_30d": "days TMIN<0", "freeze_thaw_days_30d": "days TMIN<0<TMAX (proxy)", "freeze_days_60d": "days TMIN<0",
             "freeze_thaw_days_60d": "days TMIN<0<TMAX (proxy)", "prcp_mm_7d": "sum PRCP, mm", "prcp_mm_14d": "sum PRCP, mm",
             "prcp_mm_30d": "sum PRCP, mm", "wet_days_14d": "days PRCP>=1 mm", "consecutive_wet_days": "trailing run of PRCP>=1 mm days",
             "snow_mm_30d": "sum SNOW, mm", "freeze_thaw_x_prcp_30d": "freeze_thaw_days_30d * prcp_mm_30d"}
    for k, v in wdefs.items():
        defs.append(FeatureDef(k, "weather", k.split("_")[-1], v.split(",")[-1].strip(), "NOAA GHCN-D USW00094728 (one station)", ANOAA, v))
    sdefs = {"length_m": "projected block length (EPSG:2263) in m", "street_width_ft": "CSCL streetwidth", "travel_lanes": "CSCL number_travel_lanes",
             "parking_lanes": "CSCL number_park_lanes", "total_lanes": "CSCL number_total_lanes", "posted_speed_mph": "CSCL posted_speed",
             "truck_route": "CSCL truck_route_type present", "bike_lane": "CSCL bike_lane present", "one_way": "CSCL trafdir FT/TF",
             "n_adjacent_blocks": "eligible blocks sharing a node", "street_type_code": "CSCL post/pre type (8 classes + OTHER)",
             "snow_priority_code": "CSCL snow_priority category"}
    for k, v in sdefs.items():
        defs.append(FeatureDef(k, "static", "static", "-", "CSCL centerline 2026 snapshot", "static (anachronism documented)", v))
    return defs
