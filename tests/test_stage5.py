"""Stage 5 decision engine and volume forecast. SYNTHETIC TEST FIXTURES only (invented scores, labels, reports)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from roadwatch import inspection as insp
from roadwatch import volume as vol

D = dt.date
W1, W2 = D(2025, 7, 7), D(2025, 7, 14)


def scores(n=100, origin=W1, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    s = np.round(rng.uniform(size=n), 1) if ties else rng.uniform(size=n)
    return pd.DataFrame({"block_id": np.arange(1000, 1000 + n), "origin": origin, "score": s})


def test_capacity_is_enforced_exactly_with_stage4_rounding():
    for share, k in ((0.005, 118), (0.01, 235), (0.02, 469)):
        assert insp.capacity_k(share, 23448) == k
    ws = insp.select_week(scores(1000), 0.01, 23448)
    assert len(ws.selected) == 235 or ws.shortfall == 235 - len(ws.selected)
    ws = insp.select_week(scores(1000), 37, 23448)                    # explicit integer capacity
    assert len(ws.selected) == 37 and ws.k == 37 and ws.selected["block_id"].is_unique
    assert list(ws.selected["rank"]) == list(range(1, 38))
    with pytest.raises(insp.SelectionError):
        insp.capacity_k(1.5, 100)


def test_shortfall_reported_when_too_few_valid_blocks():
    ws = insp.select_week(scores(10), 50, 23448)
    assert len(ws.selected) == 10 and ws.shortfall == 40


def test_selection_is_deterministic_and_ties_order_independent():
    s = scores(500, ties=True)                                         # many exact ties (scores rounded to 0.1)
    a = insp.select_week(s, 60, 23448).selected
    b = insp.select_week(s.sample(frac=1, random_state=9), 60, 23448).selected
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert a["score"].is_monotonic_decreasing


def test_missing_and_invalid_scores_excluded_and_reported():
    s = scores(50)
    s.loc[[0, 1, 2], "score"] = [np.nan, np.inf, -np.inf]
    ws = insp.select_week(s, 50, 23448)
    assert len(ws.excluded) == 3 and set(ws.excluded["reason"]) == {"invalid_or_missing_score"}
    assert not ws.selected["block_id"].isin(s.loc[[0, 1, 2], "block_id"]).any() and ws.shortfall == 3


def test_duplicate_blocks_rejected_not_double_selected():
    s = pd.concat([scores(20), scores(20).head(1)])
    with pytest.raises(insp.SelectionError, match="more than once"):
        insp.select_week(s, 5, 23448)
    with pytest.raises(insp.SelectionError, match="single forecast origin"):
        insp.select_week(pd.concat([scores(5, W1), scores(5, W2)]), 2, 23448)


def test_selection_ignores_labels_so_future_outcomes_cannot_leak():
    s = scores(300)
    a = insp.select_week(s.assign(label=1.0), 30, 23448).selected
    b = insp.select_week(s.assign(label=np.where(np.arange(300) % 2, 1.0, np.nan)), 30, 23448).selected
    assert a["block_id"].tolist() == b["block_id"].tolist()


def test_unique_report_counting_with_multi_candidate_reports():
    sel = pd.DataFrame({"block_id": [1, 2], "origin": [W1, W1]})
    labels = pd.DataFrame({"block_id": [1, 2, 3, 4], "origin": [W1] * 4, "label": [1, 1, 1, np.nan]})
    reports = pd.DataFrame({"block_id": [1, 2, 3, 3], "origin": [W1] * 4, "report_id": ["r1", "r1", "r2", "r3"]})
    w = insp.weekly_outcomes(sel, labels, reports).iloc[0]                  # r1 sits on blocks 1 and 2: counted once
    assert (w["reports_captured"], w["reports_total"], w["hits"], w["positives"], w["labelled_selected"]) == (1, 3, 2, 3, 2)
    s = insp.summarize(insp.weekly_outcomes(sel, labels, reports))
    assert s["report_capture_rate"] == pytest.approx(1 / 3) and s["precision"] == 1.0 and s["recall"] == pytest.approx(2 / 3)


def test_zip_diversification_respects_quotas_and_capacity():
    s = scores(400)
    s["zip"] = np.where(s["score"] > 0.5, "A", np.where(s.index % 2, "B", "C"))  # high scores concentrated in ZIP A
    share = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    ws = insp.select_week(s, 60, 23448, zip_share=share, max_multiple=1.5)
    quota = int(np.ceil(1.5 * 60 / 3))
    assert len(ws.selected) == 60 and ws.selected["block_id"].is_unique
    assert ws.selected["zip"].value_counts().max() <= quota
    un = insp.select_week(s, 60, 23448).selected
    assert (un["zip"] == "A").sum() == 60                                     # unconstrained baseline unchanged


def test_non_adjacent_rule_never_selects_blocks_sharing_a_node():
    s = scores(200)
    rng = np.random.default_rng(5)
    nodes = pd.DataFrame({"node_from": rng.integers(0, 120, 200), "node_to": rng.integers(0, 120, 200)},
                         index=s["block_id"].to_numpy())
    ws = insp.select_week(s, 25, 23448, nodes=nodes)
    sel_nodes = np.r_[nodes.loc[ws.selected["block_id"], "node_from"], nodes.loc[ws.selected["block_id"], "node_to"]]
    for b in ws.selected["block_id"]:                                       # no other selected block touches b's nodes
        others = ws.selected["block_id"][ws.selected["block_id"] != b]
        assert not set(nodes.loc[b]).intersection(set(nodes.loc[others].to_numpy().ravel()))
    assert len(ws.selected) <= 25 and ws.selected["block_id"].is_unique
    assert ws.selected["score"].is_monotonic_decreasing                       # still follows the ranking
    a = insp.select_week(s.sample(frac=1, random_state=2), 25, 23448, nodes=nodes).selected
    assert a["block_id"].tolist() == ws.selected["block_id"].tolist()          # deterministic


def test_paired_bootstrap_ci_sign():
    rng = np.random.default_rng(0)
    num_a, num_b, den = np.full(50, 30.0), np.full(50, 20.0), np.full(50, 100.0)
    lo, hi = insp.bootstrap_ratio(num_a, den, rng, 200, num_b, den)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


# ---------------- volume forecast ----------------
def _volume_fixture():
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(3)
    per_day = rng.poisson(20 + 10 * np.sin(2 * np.pi * days.day_of_year / 365), len(days))
    rep = pd.Series(np.repeat(days.date, per_day))
    origins = pd.date_range("2024-01-01", "2024-12-23", freq="7D").date
    y = [int(((rep >= o) & (rep < o + dt.timedelta(days=7))).sum()) for o in origins]
    weeks = pd.DataFrame({"origin": origins, "located_brooklyn_reports": y, "week_status": "observable"})
    noaa = pd.DataFrame({"date": days, "tmin_value": 3.0, "tmax_value": 9.0, "prcp_value": 1.0, "snow_value": 0.0})
    return rep, weeks, noaa


def test_volume_features_are_as_of_and_do_not_see_target_week():
    rep, weeks, noaa = _volume_fixture()
    f = vol.build_frame(rep, weeks, noaa)
    o = weeks["origin"].iloc[10]
    late = pd.concat([rep, pd.Series([o + dt.timedelta(days=1)] * 500, dtype=object),          # inside the target week
                      pd.Series([o - dt.timedelta(days=1)] * 500, dtype=object)])               # inside the 2-day lag
    g = vol.build_frame(late, weeks, noaa)
    cols = vol.FEATURES["recent_seasonal_weather"]
    pd.testing.assert_series_equal(f.loc[10, cols], g.loc[10, cols], check_names=False)
    noaa2 = noaa.copy()
    noaa2.loc[noaa2["date"] == pd.Timestamp(o - dt.timedelta(days=4)), "tmin_value"] = -20.0   # not yet published at t
    assert vol.build_frame(rep, weeks, noaa2).loc[10, "tmin_7"] == f.loc[10, "tmin_7"]


def test_volume_baselines_and_errors():
    rep, weeks, noaa = _volume_fixture()
    f = vol.build_frame(rep, weeks, noaa)
    assert np.array_equal(vol.baseline_forecast(f, "naive_last7"), f["r7"].to_numpy(float))
    e = vol.errors(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert e["mae"] == 2.0 and e["bias_mean_pred_minus_actual"] == 0.0
    m = vol.fit_glm(f.iloc[:40], vol.FEATURES["recent_seasonal"], 0.01)
    lo, hi = vol.interval(m, f.iloc[:40], vol.FEATURES["recent_seasonal"], (0.1, 0.9), 26)
    assert 0 < lo <= 1 <= hi
