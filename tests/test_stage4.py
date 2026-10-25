"""Stage 4: as-of features, leakage guards, walk-forward windows, calibration, evaluation.
SYNTHETIC TEST FIXTURES only (network from tests/stage3_fixtures.py, invented reports and weather)."""
import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from roadwatch import evaluation as ev
from roadwatch.blocks import build_network
from roadwatch.conditional_features import ConditionalFeatureBuilder
from roadwatch.features import DIAGNOSTIC_ONLY, FEATURE_SETS, FeatureBuilder
from roadwatch.modeling import (Platt, SelectionLeakError, det_uniform, guard_selection_data, negative_sample,
                                retrain_schedule, windows_for)
from roadwatch.stage4 import MAIN_VARIANTS, STRESS, experiments
from stage3_fixtures import centerline

D = dt.date
T = D(2025, 3, 3)          # a Monday prediction origin


def _blocks():
    b = build_network(centerline()).blocks.copy()
    for c, v in {"streetwidth_num": 30.0, "number_travel_lanes_num": 2.0, "number_park_lanes_num": 2.0,
                 "number_total_lanes_num": 4.0, "posted_speed_num": 25.0, "truck_route_type": None, "bike_lane": None,
                 "trafdir": "TW", "snow_priority": "C", "b5sc": "000000", "post_type": "ST", "pre_type": None}.items():
        b[c] = v
    return b


def _weeks(unknown=()):
    o = [D(2024, 1, 1) + dt.timedelta(weeks=i) for i in range(80)]
    return pd.DataFrame({"origin": o, "week_status": ["unlocated_reports" if x in unknown else "observable" for x in o]})


def _noaa(override=None):
    d = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    n = pd.DataFrame({"date": d, "tmin_value": 5.0, "tmax_value": 12.0, "prcp_value": 0.0, "snow_value": 0.0})
    for day, col, val in (override or []):
        n.loc[n["date"] == pd.Timestamp(day), col] = val
    return n


def _assign(events):
    """events: (report_id, created 'YYYY-mm-dd HH:MM:SS', [candidate block ids], status)"""
    cols = ["report_id", "created_date_local", "candidate_block_ids", "n_candidates", "match_status", "included_in_labels"]
    df = pd.DataFrame([{"report_id": r, "created_date_local": pd.Timestamp(ts), "candidate_block_ids": c,
                        "n_candidates": len(c), "match_status": s, "included_in_labels": True} for r, ts, c, s in events],
                      columns=cols)
    df["created_date_local"] = pd.to_datetime(df["created_date_local"])
    df["included_in_labels"] = df["included_in_labels"].astype(bool)
    return df


def _fb(events, unknown=(), noaa=None):
    return FeatureBuilder(_assign(events), _weeks(unknown), _blocks(), noaa if noaa is not None else _noaa())


def _row(df, block):
    return df[df["block_id"] == block].iloc[0]


# ---------------- 311 as-of cutoff and windows ----------------
def test_311_two_day_cutoff_and_window_boundaries():
    ev_ = [("in", "2025-02-28 23:59:59", [101], "matched"),      # Friday: created < t-2d -> available
           ("out", "2025-03-01 00:00:00", [101], "matched"),     # Saturday 00:00 = t-2d -> NOT available
           ("edge_in", "2025-02-22 00:00:00", [101], "matched"),  # c-7 days -> inside 7-day window
           ("edge_out", "2025-02-21 23:00:00", [101], "matched")]  # c-8 days -> outside 7-day window
    r = _row(_fb(ev_).build([T]), 101)
    assert r["rep_frac_7d"] == 2 and r["rep_frac_30d"] == 3
    assert r["days_since_last_report"] == 1


def test_future_records_never_change_past_features():
    base = [("a", "2025-02-10 10:00:00", [101], "matched"), ("b", "2025-01-15 10:00:00", [101, 102], "ambiguous")]
    future = base + [("f1", "2025-03-01 08:00:00", [101], "matched"), ("f2", "2025-03-05 08:00:00", [102], "matched"),
                     ("f3", "2025-06-01 08:00:00", [301], "matched")]
    a = _fb(base).build([T]).drop(columns=["bk_reports_7d", "bk_reports_28d"])
    b = _fb(future).build([T]).drop(columns=["bk_reports_7d", "bk_reports_28d"])
    pd.testing.assert_frame_equal(a, b)
    assert _fb(base).build([T])["bk_reports_28d"].iloc[0] == _fb(future).build([T])["bk_reports_28d"].iloc[0]


def test_fractional_any_unique_attribution_and_decay():
    ev_ = [("u", "2025-02-20 10:00:00", [101], "matched"), ("amb", "2025-02-20 11:00:00", [101, 102], "ambiguous")]
    f = _fb(ev_).build([T])
    r1, r2 = _row(f, 101), _row(f, 102)
    assert r1["rep_frac_30d"] == pytest.approx(1.5) and r2["rep_frac_30d"] == pytest.approx(0.5)
    assert r1["rep_any_30d"] == 2 and r1["rep_unique_365d"] == 1 and r2["rep_unique_365d"] == 0
    age = (D(2025, 3, 1) - D(2025, 2, 20)).days                       # cutoff day minus event day
    assert r1["rep_decay_hl7d"] == pytest.approx(1.5 * 0.5 ** (age / 7), rel=1e-6)


def test_unknown_weeks_are_not_zero_and_unmatured_weeks_excluded():
    unknown = {D(2025, 1, 13)}
    ev_ = [("w_unknown", "2025-01-14 09:00:00", [101], "matched"),     # positive in an UNKNOWN week -> not in numerator
           ("w_obs", "2025-01-21 09:00:00", [101], "matched"),          # observable week -> counted
           ("w_recent", "2025-02-24 09:00:00", [101], "matched")]       # week of t-7: not matured -> not in rate
    r = _row(_fb(ev_, unknown).build([T]), 101)
    assert r["unknown_weeks_52"] == 1
    assert r["obs_weeks_52"] == 51                                       # 52 matured weeks minus 1 unknown
    assert r["pos_week_rate_52"] == pytest.approx(1 / 51)
    assert r["unlocated_weeks_in_365d"] == 1 and r["history_window_compromised"] == 1
    assert r["rep_frac_30d"] == 1                                        # raw counts still see the matured+recent report


def test_weather_lag_excludes_recent_observations():
    t4 = T - dt.timedelta(days=4)
    t5 = T - dt.timedelta(days=5)
    base = _fb([], noaa=_noaa()).build([T]).iloc[0]
    late = _fb([], noaa=_noaa([(t4, "tmin_value", -30.0), (t4, "prcp_value", 99.0)])).build([T]).iloc[0]
    assert late["tmin_min_7d"] == base["tmin_min_7d"] and late["prcp_mm_7d"] == base["prcp_mm_7d"]   # t-4 not yet published
    avail = _fb([], noaa=_noaa([(t5, "tmin_value", -3.0), (t5, "prcp_value", 4.0)])).build([T]).iloc[0]
    assert avail["tmin_min_7d"] == -3.0 and avail["prcp_mm_7d"] == 4.0 and avail["freeze_thaw_days_14d"] == 1


def test_training_and_inference_features_identical():
    ev_ = [("a", "2025-02-10 10:00:00", [101], "matched"), ("b", "2025-01-15 10:00:00", [101, 102], "ambiguous")]
    fb = _fb(ev_)
    both = fb.build([T - dt.timedelta(weeks=1), T])
    single = fb.build([T])
    pd.testing.assert_frame_equal(both[both["origin"] == T].reset_index(drop=True), single)


# ---------------- conditional sources ----------------
def test_conditional_sources_respect_assumed_lags():
    blocks = _blocks()
    to4326 = lambda g: gpd.GeoSeries(g, crs="EPSG:2263").to_crs("EPSG:4326")  # noqa: E731
    geom = to4326([LineString([(0, 10), (0, 290)])] * 2)
    pav = gpd.GeoDataFrame({"oftcode": ["000000000000000000"] * 2, "rating_value": [6.0, 3.0],
                            "inspection_local": pd.to_datetime([T - dt.timedelta(days=45), T - dt.timedelta(days=44)])},
                           geometry=geom, crs="EPSG:4326")
    wo = gpd.GeoDataFrame({"oft": ["0"] * 2, "rptclosed_date": pd.to_datetime([T - dt.timedelta(days=35), T - dt.timedelta(days=34)])},
                          geometry=geom, crs="EPSG:4326")
    pv = gpd.GeoDataFrame({"oft_code": ["0"], "location_actual_paving_end_date_local": pd.to_datetime([T - dt.timedelta(days=20)])},
                          geometry=gpd.GeoSeries([LineString([(0, 10), (0, 290)])], crs="EPSG:2263"), crs="EPSG:2263")
    r = ConditionalFeatureBuilder(blocks, pav, wo, pv).build([T])
    r = r[r["block_id"] == 101].iloc[0]
    assert r["pav_rating_latest"] == 6.0          # t-45 available, t-44 (rating 3.0) not yet
    assert r["wo_closed_365d"] == 1               # closed t-35 counts, t-34 does not
    assert r["paving_never_in_record"] == 1       # paving end t-20 is inside the 21-day delay


# ---------------- modelling protocol ----------------
def test_walk_forward_windows_and_schedule():
    sched = retrain_schedule(D(2025, 7, 7), D(2026, 10, 12))
    assert sched[0] == (D(2025, 7, 7), D(2025, 8, 3)) and sched[-1][1] == D(2026, 10, 12)
    assert all(r.weekday() == 0 and r.day <= 7 for r, _ in sched) and len(sched) == 16
    for r, _ in sched:
        w = windows_for(r)
        assert w["fit_before"] == w["holdout_from"] < w["last_matured_origin"] < r
        assert w["last_matured_origin"] + dt.timedelta(days=7 + 2) <= r       # horizon + 2-day lag matured by r


def test_selection_guard_blocks_test_outcomes():
    guard_selection_data([D(2025, 6, 30)])
    with pytest.raises(SelectionLeakError):
        guard_selection_data([D(2025, 6, 30), D(2025, 7, 7)])


def test_negative_sampling_is_deterministic_and_unbiased_by_weight():
    b = np.arange(200_000) % 23448
    o = pd.Series(pd.to_datetime("2024-01-01") + pd.to_timedelta((np.arange(200_000) // 23448) * 7, unit="D"))
    u1, u2 = det_uniform(b, o), det_uniform(b, o)
    assert np.array_equal(u1, u2) and (u1 >= 0).all() and (u1 < 1).all() and abs(u1.mean() - 0.5) < 0.01
    y = (u1 > 0.99).astype(int)
    keep, w = negative_sample(y, det_uniform(b + 7, o), 0.1)
    assert keep[y == 1].all()
    assert (w[keep] * (1 - y[keep])).sum() == pytest.approx((1 - y).sum(), rel=0.03)


def test_platt_preserves_ranking_and_fixes_miscalibration():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.001, 0.05, 200_000)
    y = (rng.uniform(size=p_true.size) < p_true).astype(int)
    s = np.clip(p_true * 4, 0, 0.999)                       # 4x over-confident scores
    cal = Platt("logit").fit(s, y).predict(s)
    assert np.array_equal(np.argsort(s, kind="stable"), np.argsort(cal, kind="stable"))
    assert ((cal - y) ** 2).mean() < ((s - y) ** 2).mean()
    assert abs(cal.mean() - y.mean()) < 0.002


def test_platt_with_sparse_or_absent_positives_is_constant_not_extrapolated():
    s = np.linspace(0, 1, 1000)
    assert np.all(Platt("logit").fit(s, np.zeros(1000, int)).predict(s) == 0.0)        # no positives in holdout
    y = np.zeros(1000, int)
    y[-1] = 1
    p = Platt("logit").fit(s, y).predict(s)
    assert (np.diff(p) >= -1e-12).all() and 0 <= p.min() and p.max() < 1                # single positive: monotone, bounded


def test_negative_sample_identical_across_variants_except_for_labels():
    b = np.arange(50_000) % 23448
    o = pd.Series(pd.to_datetime("2024-01-01") + pd.to_timedelta((np.arange(50_000) // 23448) * 7, unit="D"))
    u = det_uniform(b, o)
    y_a = (u > 0.995).astype(int)
    y_b = ((u > 0.990) & (u < 0.996)).astype(int)
    ka, _ = negative_sample(y_a, u, 0.1)
    kb, _ = negative_sample(y_b, u, 0.1)
    neg_a, neg_b = ka & (y_a == 0), kb & (y_b == 0)
    both_neg = (y_a == 0) & (y_b == 0)
    assert np.array_equal(neg_a[both_neg], neg_b[both_neg])     # same negatives where both variants say negative


def test_persisted_models_reproduce_predictions(tmp_path):
    import joblib
    import lightgbm as lgb
    from roadwatch.modeling import ModelSpec, fit_model, predict_raw
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(size=4000), "street_type_code": rng.integers(0, 3, 4000).astype("float32")})
    X["y"] = (rng.uniform(size=4000) < X["a"] * 0.1).astype(float)
    X["_w"] = 1.0
    for spec in (ModelSpec("LR", {"C": 0.1}), ModelSpec("LGBM", {"num_leaves": 7, "min_data_in_leaf": 50, "max_rounds": 30})):
        fr = fit_model(spec, ["a", "street_type_code"], X.iloc[:3000], X.iloc[3000:], "y")
        p1 = predict_raw(spec, fr, X, ["a", "street_type_code"])
        if spec.name == "LR":
            joblib.dump(fr.model, tmp_path / "m.joblib")
            p2 = joblib.load(tmp_path / "m.joblib").predict_proba(X[["a", "street_type_code"]])[:, 1]
        else:
            fr.model.save_model(str(tmp_path / "m.txt"), num_iteration=fr.model.best_iteration)
            p2 = lgb.Booster(model_file=str(tmp_path / "m.txt")).predict(X[["a", "street_type_code"]])
        assert np.allclose(p1, p2)
        assert np.allclose(predict_raw(spec, fr, X, ["a", "street_type_code"]), p1)     # stable on repeat


def test_topk_ties_are_deterministic_and_unknown_excluded():
    p = pd.DataFrame({"block_id": np.arange(100), "origin": [D(2025, 7, 7)] * 100, "score": np.r_[np.ones(50), np.zeros(50)],
                      "prob": 0.01, "label": np.r_[np.ones(5), np.full(45, np.nan), np.zeros(50)]})
    a = ev.topk_select(p, 0.1, 100)
    b = ev.topk_select(p.sample(frac=1, random_state=3), 0.1, 100)
    assert sorted(a["block_id"]) == sorted(b["block_id"]) and len(a) == 10
    m = ev.capacity_metrics(p, None, 100, shares=(0.5,))["top_50pct"]
    assert m["selected_labelled"] == 5 and m["precision"] == 1.0          # 45 unknown labels are not negatives


def test_experiment_inputs_never_include_labels_or_ids():
    sel = {"chosen": {v: {"B0_global": {"params": {}, "feature_set": None},
                          "B1_hist_rate": {"params": {"prior_weeks": 10}, "feature_set": None},
                          "B2_recency": {"params": {"half_life": 30}, "feature_set": None},
                          "LR": {"params": {"C": 0.1}, "feature_set": "core"},
                          "LGBM": {"params": {"num_leaves": 15, "min_data_in_leaf": 200}, "feature_set": "history_static"}}
                      for v in MAIN_VARIANTS}}
    for e in experiments(sel):
        assert not any(f.startswith("y_") or f in ("block_id", "origin") or f in DIAGNOSTIC_ONLY for f in e["features"])
        assert e["variant"] in MAIN_VARIANTS + ["no_supplementary"]   # the variant only selects the target column
    lg = [e for e in experiments(sel) if e["id"] == "multi_block__LGBM"][0]
    assert lg["features"] == FEATURE_SETS["history_static"]           # frozen selection drives the test experiment
    for fs in FEATURE_SETS.values():
        assert not any(f.startswith("y_") for f in fs)
    ids = [e["id"] for e in experiments(sel)]
    assert len(ids) == len(set(ids))


def test_march_2026_stress_window():
    assert STRESS == (D(2026, 3, 2), D(2026, 3, 30)) and all(d.weekday() == 0 for d in STRESS)
