"""Data access and business logic for the RoadWatch dashboard (no Streamlit imports - testable without a UI).

All numbers shown by the dashboard come from saved Stage 3-5 outputs. The one-off `build()` step precomputes
(1) block geometry paths for the map and (2) a unified policy summary that scores the SAVED Stage-5 selections under
both report-capture definitions. Nothing here retrains, retunes or changes the Stage 4 models or Stage 5 policy.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

from . import inspection as insp
from .config import Paths
from .storage import read_json, sha256_file, utc_iso, write_json

D = dt.date
STRESS = (D(2026, 3, 2), D(2026, 3, 30))
CAPACITIES = (0.005, 0.01, 0.02)
CAP_NAME = {0.005: "0.5pct", 0.01: "1pct", 0.02: "2pct"}
POLICIES = [  # (label, saved selection stem, own-target labels)
    ("Stage-5 decision: multi-block LightGBM + non-adjacent", "multi_block__LGBM__{cap}__nonadjacent", "multi_block"),
    ("Plain top-K multi-block LightGBM", "multi_block__LGBM__{cap}", "multi_block"),
    ("Strict single-block LightGBM (top-K)", "primary__LGBM__{cap}", "primary"),
    ("Strict single-block LightGBM + non-adjacent", "primary__LGBM__{cap}__nonadjacent", "primary"),
    ("Historical-rate baseline", "multi_block__B1_hist_rate__{cap}", "multi_block"),
    ("Recency baseline", "multi_block__B2_recency__{cap}", "multi_block"),
]
RECOMMENDED = POLICIES[0][0]   # kept name for compatibility: the Stage-5 decided policy (conditional, see Stage-7 audit)
# Inspection lists the dashboard can show (Stage 7). key -> (label, worklist file pattern, ranking predictions, non_adjacent)
LISTS = {
    "multi_nonadj": ("Stage-5 decision: multi-block LightGBM + non-adjacent",
                     "multi_block__LGBM__{cap}.parquet", "multi_block__LGBM", True),
    "strict_topk": ("Strict single-block LightGBM, top-K (best under conservative expected capture)",
                    "primary__LGBM__{cap}__topk.parquet", "primary__LGBM", False),
    "strict_nonadj": ("Strict single-block LightGBM + non-adjacent (if inspecting a block covers its end intersections)",
                      "primary__LGBM__{cap}__nonadjacent.parquet", "primary__LGBM", True),
}


class MissingOutput(Exception):
    """A required saved output is absent; `hint` names the command that produces it."""

    def __init__(self, path: Path, hint: str):
        super().__init__(f"missing {path} - run: {hint}")
        self.path, self.hint = path, hint


def _need(p: Path, hint: str) -> Path:
    if not p.exists():
        raise MissingOutput(p, hint)
    return p


@dataclass(frozen=True)
class Locations:
    paths: Paths

    @property
    def s3(self): return self.paths.processed / "stage3"

    @property
    def s4(self): return self.paths.processed / "stage4"

    @property
    def s5(self): return self.paths.processed / "stage5"

    @property
    def out5(self): return self.paths.root / "outputs" / "stage5"

    @property
    def s6(self): return self.paths.processed / "stage6"

    @property
    def out6(self): return self.paths.root / "outputs" / "stage6"


# ============================== one-off precomputation ==============================
def build(paths: Paths, n_boot: int = 1000, seed: int = 23) -> dict:
    from .stage5 import _capture, reports
    L = Locations(paths)
    L.s6.mkdir(parents=True, exist_ok=True)
    L.out6.mkdir(parents=True, exist_ok=True)
    # 1. map geometry: WGS84 coordinate paths per block (computed once, not on every interaction)
    b = pd.read_parquet(_need(L.s3 / "blocks.parquet", "python -I -m roadwatch stage3"),
                        columns=["block_id", "full_street_name", "l_zip", "r_zip", "length_m", "geometry_wgs84_wkb"])
    geoms = shapely.from_wkb(b["geometry_wgs84_wkb"].to_numpy())
    paths_ = [np.round(np.asarray(g.coords if g.geom_type == "LineString" else shapely.line_merge(g).coords)[:, :2], 6).tolist()
              for g in geoms]
    bp = pd.DataFrame({"block_id": b["block_id"], "street": b["full_street_name"].str.replace(r"\s+", " ", regex=True),
                       "zip": b["l_zip"].fillna(b["r_zip"]).fillna("unknown"), "length_m": b["length_m"].round(1),
                       "path": paths_})
    bp.to_parquet(L.s6 / "block_paths.parquet", index=False)
    # 2. unified policy summary from SAVED selections, scored under both capture definitions
    rp_any, rp_asg = reports(paths, "multi_block"), reports(paths, "primary")
    labels = {v: pd.read_parquet(_need(L.s4 / "predictions" / f"{v}__LGBM.parquet", "stage4 test"),
                                 columns=["block_id", "origin", "label"]) for v in ("multi_block", "primary")}
    for v in labels:
        labels[v]["origin"] = pd.to_datetime(labels[v]["origin"]).dt.date
    rng = np.random.default_rng(seed)
    rows, monthly = [], []
    for label, stem, target in POLICIES:
        for cap in CAPACITIES:
            f = _need(L.s5 / "selections" / f"{stem.format(cap=CAP_NAME[cap])}.parquet", "python -I -m roadwatch stage5 all")
            sel = pd.read_parquet(f)
            sel["origin"] = pd.to_datetime(sel["origin"]).dt.date
            a, s = _capture(sel, rp_any), _capture(sel, rp_asg)
            w = insp.weekly_outcomes(sel, labels[target], rp_any if target == "multi_block" else rp_asg).set_index("origin")
            for period, mask in (("test", None), ("march_2026", STRESS)):
                aa, ss, ww = (a, s, w) if mask is None else tuple(x[(x.index >= mask[0]) & (x.index <= mask[1])] for x in (a, s, w))
                rows.append({
                    "policy": label, "capacity": cap, "period": period, "weeks": len(ww), "k_per_week": int(w["selected"].iloc[0]),
                    "inspections": int(ww["selected"].sum()), "reports_total": int(aa["total"].sum()),
                    "captured_any_candidate": int(aa["captured"].sum()),
                    "capture_any_candidate": aa["captured"].sum() / aa["total"].sum(),
                    "capture_any_ci95": insp.bootstrap_ratio(aa["captured"].to_numpy(float), aa["total"].to_numpy(float), rng, n_boot),
                    "captured_assigned_block": int(ss["captured"].sum()),
                    "capture_assigned_block": ss["captured"].sum() / ss["total"].sum(),
                    "capture_assigned_ci95": insp.bootstrap_ratio(ss["captured"].to_numpy(float), ss["total"].to_numpy(float), rng, n_boot),
                    "precision_own_target": ww["hits"].sum() / ww["labelled_selected"].sum(),
                    "precision_ci95": insp.bootstrap_ratio(ww["hits"].to_numpy(float), ww["labelled_selected"].to_numpy(float), rng, n_boot),
                    "precision_target": target,
                    "reports_per_100_inspections_any": 100 * aa["captured"].sum() / ww["selected"].sum()})
            mon = a.assign(m=pd.to_datetime(pd.Series(a.index, index=a.index)).dt.to_period("M").astype(str)).groupby("m").sum()
            mon_s = s.assign(m=pd.to_datetime(pd.Series(s.index, index=s.index)).dt.to_period("M").astype(str)).groupby("m").sum()
            for m_, r in mon.iterrows():
                monthly.append({"policy": label, "capacity": cap, "month": m_, "reports_total": int(r["total"]),
                                "capture_any_candidate": r["captured"] / r["total"],
                                "capture_assigned_block": mon_s.loc[m_, "captured"] / mon_s.loc[m_, "total"]})
    summ = pd.DataFrame(rows)
    for c in ("capture_any_ci95", "capture_assigned_ci95", "precision_ci95"):
        summ[c] = summ[c].map(lambda x: json.dumps([round(v, 5) for v in x]))
    summ.to_csv(L.out6 / "policy_summary.csv", index=False)
    pd.DataFrame(monthly).to_csv(L.out6 / "policy_monthly.csv", index=False)
    man = {"generated_utc": utc_iso(), "bootstrap": {"resamples": n_boot, "seed": seed, "unit": "test week"},
           "inputs_sha256": {p.name: sha256_file(p) for p in sorted((L.s5 / "worklists").glob("*.parquet"))},
           "outputs": ["data/processed/stage6/block_paths.parquet", "outputs/stage6/policy_summary.csv", "outputs/stage6/policy_monthly.csv"]}
    write_json(L.s6 / "dashboard_manifest.json", man)
    return {"blocks": len(bp), "summary_rows": len(summ), "monthly_rows": len(monthly)}


# ============================== loaders ==============================
def config(paths: Paths) -> dict:
    return read_json(_need(paths.root / "config" / "stage5.json", "config/stage5.json is part of the repository"))


def available_weeks(paths: Paths) -> list[D]:
    f = _need(Locations(paths).s5 / "worklists" / "multi_block__LGBM__1pct.parquet", "python -I -m roadwatch stage5 worklists")
    w = pq.read_table(f, columns=["forecast_week_start"]).column(0).to_pandas()
    return sorted(pd.to_datetime(w).dt.date.unique())


def resolve_week(requested, weeks: list[D]) -> tuple[D, str | None]:
    """Return a valid forecast week and an optional user-facing message (invalid / unavailable -> latest week)."""
    if not weeks:
        raise MissingOutput(Path("worklists"), "python -I -m roadwatch stage5 worklists")
    if requested in (None, ""):
        return weeks[-1], None
    try:
        d = requested if isinstance(requested, D) else D.fromisoformat(str(requested))
    except ValueError:
        return weeks[-1], f"'{requested}' is not a valid date; showing the latest available week {weeks[-1]}."
    if d in weeks:
        return d, None
    return weeks[-1], (f"No saved prediction for {d}; predictions exist for {weeks[0]} to {weeks[-1]} (Mondays). "
                       f"Showing {weeks[-1]}.")


def default_list(paths: Paths) -> str:
    key = config(paths).get("dashboard_default_list", "multi_nonadj")
    return key if key in LISTS else "multi_nonadj"


def worklist(paths: Paths, week: D, capacity: float, list_key: str = "multi_nonadj") -> pd.DataFrame:
    """Saved worklist for one week, capacity and inspection list (LISTS), ordered by rank."""
    if capacity not in CAP_NAME:
        raise ValueError(f"capacity must be one of {list(CAP_NAME)}")
    if list_key not in LISTS:
        raise ValueError(f"list must be one of {list(LISTS)}")
    hint = "python -I -m roadwatch stage5 worklists" if list_key == "multi_nonadj" else "python -I -m roadwatch stage5 alternative-worklists"
    f = _need(Locations(paths).s5 / "worklists" / LISTS[list_key][1].format(cap=CAP_NAME[capacity]), hint)
    w = pq.read_table(f, filters=[("forecast_week_start", "=", week)]).to_pandas()
    return w.sort_values("rank").reset_index(drop=True)


def custom_worklist(paths: Paths, week: D, k: int, list_key: str = "multi_nonadj") -> pd.DataFrame:
    """Manually configured capacity (K slots): same engine and rule as the chosen list, computed on demand for one week."""
    from .stage5 import build_worklists
    _, _, pol, nonadj = LISTS[list_key]
    wl, _ = build_worklists(paths, pol, int(k), origins=[week], non_adjacent=nonadj)
    return wl.sort_values("rank").reset_index(drop=True)


def capture_definitions(paths: Paths) -> tuple[pd.DataFrame, dict]:
    L = Locations(paths)
    t = pd.read_csv(_need(L.out5 / "capture_definitions.csv", "python -I -m roadwatch stage5 capture-definitions"))
    meta = read_json(_need(L.out5 / "capture_definitions.json", "python -I -m roadwatch stage5 capture-definitions"))
    return t, meta


def capture_table(t: pd.DataFrame, capacity: float, period: str) -> pd.DataFrame:
    """Display table: capture under the four definitions with 95% CIs (numbers verbatim from capture_definitions.csv)."""
    s = t[(t["capacity"] == capacity) & (t["period"] == period)]
    out = {"List": s["list"].to_numpy()}
    names = {"any_candidate": "Any candidate block", "assigned_block": "Assigned block", "fractional": "Fractional (expected)",
             "intersection_node": "Intersection-node coverage"}
    for k, n in names.items():
        out[n] = [f"{100 * v:.1f}% ({100 * c[0]:.1f}-{100 * c[1]:.1f})" for v, c in zip(s[k], s[f"{k}_ci95"].map(json.loads))]
    return pd.DataFrame(out)


def week_scores(paths: Paths, week: D, policy: str = "multi_block__LGBM") -> pd.DataFrame:
    """All eligible blocks' scores for a week with a deterministic within-week rank and percentile (1 = highest risk)."""
    f = _need(Locations(paths).s4 / "predictions" / f"{policy}.parquet", "python -I -m roadwatch stage4 test")
    p = pq.read_table(f, columns=["block_id", "origin", "score"], filters=[("origin", "=", week)]).to_pandas()
    if p.empty:
        raise MissingOutput(f, f"no saved prediction rows for {week}")
    r = insp.rank_week(p)
    r["risk_percentile"] = 100 * (1 - (r["rank"] - 1) / len(r))
    return r[["block_id", "score", "rank", "risk_percentile"]]


def block_paths(paths: Paths) -> pd.DataFrame:
    """Block coordinate paths as plain Python lists [[lon, lat], ...] (Parquet returns nested numpy arrays, which the
    deck.gl JSON serialiser does not render as coordinates)."""
    bp = pd.read_parquet(_need(Locations(paths).s6 / "block_paths.parquet", "python -I -m roadwatch stage6 build"))
    bp["path"] = [[[float(x), float(y)] for x, y in p] for p in bp["path"]]
    return bp


def policy_summary(paths: Paths) -> pd.DataFrame:
    s = pd.read_csv(_need(Locations(paths).out6 / "policy_summary.csv", "python -I -m roadwatch stage6 build"))
    for c in ("capture_any_ci95", "capture_assigned_ci95", "precision_ci95"):
        s[c] = s[c].map(json.loads)
    return s


def policy_monthly(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(_need(Locations(paths).out6 / "policy_monthly.csv", "python -I -m roadwatch stage6 build"))


def volume_series(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    L = Locations(paths)
    fr = pd.read_parquet(_need(L.s5 / "volume_frame.parquet", "python -I -m roadwatch stage5 volume"),
                         columns=["origin", "y", "week_status"])
    te = pd.read_parquet(_need(L.s5 / "volume_test_predictions.parquet", "python -I -m roadwatch stage5 volume"))
    res = read_json(_need(L.out5 / "volume_forecast_results.json", "python -I -m roadwatch stage5 volume"))
    return fr, te, res


def evidence(paths: Paths) -> dict:
    """Headline evaluation numbers, read verbatim from saved Stage 4/5 outputs."""
    root = paths.root / "outputs"
    t4 = pd.read_csv(_need(root / "stage4" / "test_results.csv", "python -I -m roadwatch stage4 evaluate"))
    boot = read_json(_need(root / "stage4" / "bootstrap_comparisons.json", "python -I -m roadwatch stage4 uncertainty"))
    dec = read_json(_need(root / "stage5" / "decided_policy_test_summary.json", "python -I -m roadwatch stage5 policy-test"))
    pd_ = read_json(_need(Locations(paths).s5 / "policy_design.json", "python -I -m roadwatch stage5 policy-design"))
    keep = t4[(t4["group"] == "main") & t4["variant"].isin(["multi_block", "primary"]) &
              t4["model"].isin(["B1_hist_rate", "B2_recency", "LR", "LGBM"])]
    return {"stage4_table": keep, "bootstrap": boot["results"], "decided": dec, "policy_design": pd_}


def provenance(paths: Paths) -> pd.DataFrame:
    """Latest finalized raw snapshot per source (retrieval time, rows, observed range) from the immutable manifests."""
    idx = paths.raw / "MANIFEST.jsonl"
    if not idx.exists():
        return pd.DataFrame()
    latest = {}
    for line in idx.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("snapshot_id"):
            latest[e["source_key"]] = e
    rows = []
    for key, e in sorted(latest.items()):
        meta = paths.processed / f"{key}.meta.json"
        rep = sorted((paths.root / "outputs" / "data_quality").glob(f"{key}__{e['snapshot_id']}.json"))
        r = read_json(rep[0]) if rep else {}
        rows.append({"source": key, "snapshot_id": e["snapshot_id"], "retrieved_utc": r.get("retrieval_completed_utc"),
                     "role": r.get("role"), "rows": (r.get("totals") or {}).get("rows_after_dedup"),
                     "observed_range": " -> ".join(str(x)[:10] for x in ((r.get("observed_date_range") or {}).get("min"),
                                                                           (r.get("observed_date_range") or {}).get("max")) if x),
                     "validation": e.get("validation_status"), "processed_table": meta.exists()})
    return pd.DataFrame(rows)


# ============================== view helpers (pure) ==============================
def filter_worklist(wl: pd.DataFrame, query: str = "", priorities: list[str] | None = None) -> pd.DataFrame:
    """Case-insensitive search over street name, ZIP and block id; optional priority filter. Order (rank) preserved."""
    out = wl
    if priorities:
        out = out[out["priority"].isin(priorities)]
    q = (query or "").strip().lower()
    if q:
        hay = (out["full_street_name"].fillna("").str.replace(r"\s+", " ", regex=True).str.lower() + " " +
               out["zip"].astype(str) + " " + out["block_id"].astype(str))
        out = out[hay.str.contains(q, regex=False)]
    return out.sort_values("rank").reset_index(drop=True)


def policy_table(summary: pd.DataFrame, capacity: float, period: str = "test") -> pd.DataFrame:
    """Display table for one capacity/period; numbers are taken verbatim from policy_summary.csv."""
    s = summary[(summary["capacity"] == capacity) & (summary["period"] == period)].copy()
    order = {p[0]: i for i, p in enumerate(POLICIES)}
    s = s.sort_values("policy", key=lambda x: x.map(order))
    fmt = lambda v, ci: f"{100 * v:.1f}% ({100 * ci[0]:.1f}-{100 * ci[1]:.1f})"  # noqa: E731
    return pd.DataFrame({
        "Policy": s["policy"].to_numpy(),
        "Slots / week": s["k_per_week"].to_numpy(),
        "Reports captured (any candidate block)": [f"{c:,} / {t:,}" for c, t in zip(s["captured_any_candidate"], s["reports_total"])],
        "Capture, any candidate (95% CI)": [fmt(v, c) for v, c in zip(s["capture_any_candidate"], s["capture_any_ci95"])],
        "Capture, assigned block (95% CI)": [fmt(v, c) for v, c in zip(s["capture_assigned_block"], s["capture_assigned_ci95"])],
        "Precision vs own target (95% CI)": [f"{v:.3f} ({c[0]:.3f}-{c[1]:.3f}) [{t}]" for v, c, t in
                                             zip(s["precision_own_target"], s["precision_ci95"], s["precision_target"])],
        "Reports per 100 inspections": s["reports_per_100_inspections_any"].round(1).to_numpy(),
    })


# ============================== exports (deterministic) ==============================
EXPORT_COLUMNS = ["forecast_week_start", "priority", "rank", "block_id", "full_street_name", "zip", "model_score_relative",
                  "reports_365d", "reports_30d", "days_since_last_report", "adjacent_block_reports_365d", "length_m",
                  "travel_lanes", "street_width_ft", "truck_route_type", "bike_lane", "rationale", "model_drivers",
                  "recommended_action", "capacity_scenario", "selection_rule", "model_version", "mid_lat", "mid_lon"]


def to_csv_bytes(wl: pd.DataFrame) -> bytes:
    return wl.sort_values("rank")[EXPORT_COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")


def to_geojson_bytes(wl: pd.DataFrame) -> bytes:
    feats = []
    for _, r in wl.sort_values("rank").iterrows():
        props = {c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else str(r[c]) if isinstance(r[c], D) else r[c]))
                 for c in EXPORT_COLUMNS}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": json.loads(shapely.to_geojson(shapely.from_wkt(r["geometry_wkt_wgs84"])))})
    return json.dumps({"type": "FeatureCollection", "features": feats}, sort_keys=True, default=str).encode("utf-8")
