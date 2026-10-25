"""Stage 6 dashboard: filters, capacity selection, displayed metrics, worklist/map/export consistency.
Unit tests use SYNTHETIC fixtures; integration tests read the real saved outputs and are skipped if absent."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roadwatch import dashboard_data as dd
from roadwatch import dashboard_map as dm
from roadwatch.config import Paths

D = dt.date
ROOT = Path(__file__).resolve().parents[1]
REAL = Paths(ROOT)
HAVE_REAL = (REAL.processed / "stage6" / "block_paths.parquet").exists() and (REAL.processed / "stage5" / "worklists").exists()
real = pytest.mark.skipif(not HAVE_REAL, reason="real Stage 5/6 outputs not generated")
WEEKS = [D(2025, 7, 7), D(2025, 7, 14), D(2025, 7, 21)]


def synth_worklist(n=6, week=D(2025, 7, 14)):  # SYNTHETIC
    return pd.DataFrame({
        "forecast_week_start": week, "priority": ["P1", "P1", "P2", "P2", "P3", "P3"][:n], "rank": np.arange(1, n + 1),
        "block_id": np.arange(100, 100 + n), "full_street_name": ["E  18 ST", "AVE N", "5 AVE", "KINGS HWY", "AVE N", "W 1 ST"][:n],
        "zip": ["11226", "11230", "11220", "11223", "11230", "11223"][:n], "model_score_relative": np.linspace(0.4, 0.1, n),
        "reports_365d": [9, 8, 7, 6, 5, 4][:n], "reports_30d": 1, "days_since_last_report": 3.0, "adjacent_block_reports_365d": 2.0,
        "length_m": 80.0, "travel_lanes": 2.0, "street_width_ft": 34.0, "truck_route_type": None, "bike_lane": None,
        "rationale": "r", "model_drivers": "d", "recommended_action": "INSPECT", "capacity_scenario": "1pct",
        "selection_rule": "top-K + non-adjacent", "model_version": "v", "mid_lat": 40.6, "mid_lon": -73.95,
        "geometry_wkt_wgs84": [f"LINESTRING (-73.9{i} 40.6, -73.9{i} 40.61)" for i in range(n)]})


# ---------------- pure helpers ----------------
def test_resolve_week_valid_invalid_unavailable():
    assert dd.resolve_week(None, WEEKS) == (WEEKS[-1], None)
    assert dd.resolve_week("2025-07-14", WEEKS) == (D(2025, 7, 14), None)
    w, msg = dd.resolve_week("not-a-date", WEEKS)
    assert w == WEEKS[-1] and "not a valid date" in msg
    w, msg = dd.resolve_week("2030-01-07", WEEKS)
    assert w == WEEKS[-1] and "No saved prediction" in msg
    with pytest.raises(dd.MissingOutput):
        dd.resolve_week(None, [])


def test_filter_worklist_search_priority_and_order():
    wl = synth_worklist()
    assert dd.filter_worklist(wl, "ave n")["rank"].tolist() == [2, 5]           # whitespace/case-insensitive, rank order kept
    assert dd.filter_worklist(wl, "e 18 st")["block_id"].tolist() == [100]       # repeated internal spaces normalised
    assert dd.filter_worklist(wl, "11223", ["P3"])["block_id"].tolist() == [105]
    assert dd.filter_worklist(wl, "", None)["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert dd.filter_worklist(wl, "zzz").empty


def test_map_selected_layer_matches_worklist_and_threshold():
    wl = synth_worklist()
    paths = pd.DataFrame({"block_id": np.arange(100, 120), "street": "S", "zip": "11223",
                          "path": [[[-73.9, 40.6], [-73.9, 40.61]]] * 20})
    scores = pd.DataFrame({"block_id": np.arange(100, 120), "score": np.linspace(1, 0, 20), "rank": np.arange(1, 21),
                           "risk_percentile": 100 * (1 - np.arange(20) / 20)})
    bg, sel = dm.layer_frames(paths, scores, wl, min_percentile=80)
    assert sel["block_id"].tolist() == wl["block_id"].tolist() and sel["rank"].tolist() == wl["rank"].tolist()
    assert (bg["risk_percentile"] >= 80).all() and len(bg) == 5
    assert dm.band_of(99.5)[0] == "Top 1%" and dm.band_of(10)[0] == "Remaining 80%"
    assert dm.deck(bg, sel).to_json()                                             # serialisable spec


def test_exports_are_deterministic_and_consistent():
    wl = synth_worklist()
    c1, c2 = dd.to_csv_bytes(wl), dd.to_csv_bytes(wl.sample(frac=1, random_state=4))
    g1, g2 = dd.to_geojson_bytes(wl), dd.to_geojson_bytes(wl.sample(frac=1, random_state=5))
    assert c1 == c2 and g1 == g2                                                  # independent of row order
    csv = pd.read_csv(pd.io.common.BytesIO(c1))
    gj = json.loads(g1)
    assert csv["block_id"].tolist() == [f["properties"]["block_id"] for f in gj["features"]] == wl["block_id"].tolist()
    assert all(f["geometry"]["type"] == "LineString" for f in gj["features"])


def test_policy_table_formats_source_numbers_verbatim():
    rows = []
    for i, (label, _, tgt) in enumerate(dd.POLICIES):
        rows.append({"policy": label, "capacity": 0.01, "period": "test", "k_per_week": 235, "captured_any_candidate": 100 + i,
                     "reports_total": 1000, "capture_any_candidate": (100 + i) / 1000, "capture_any_ci95": [0.09, 0.11],
                     "captured_assigned_block": 80, "capture_assigned_block": 0.08, "capture_assigned_ci95": [0.07, 0.09],
                     "precision_own_target": 0.12, "precision_ci95": [0.1, 0.14], "precision_target": tgt,
                     "reports_per_100_inspections_any": 17.44})
    t = dd.policy_table(pd.DataFrame(rows), 0.01)
    assert t["Policy"].tolist() == [p[0] for p in dd.POLICIES]                   # fixed display order
    assert t.loc[0, "Capture, any candidate (95% CI)"] == "10.0% (9.0-11.0)"
    assert t.loc[2, "Precision vs own target (95% CI)"].endswith("[primary]")


# ---------------- real saved outputs ----------------
@real
def test_real_capacity_selection_row_counts():
    week = dd.available_weeks(REAL)[-1]
    for cap, k in ((0.005, 118), (0.01, 235), (0.02, 469)):
        wl = dd.worklist(REAL, week, cap)
        assert len(wl) == k and wl["block_id"].is_unique and wl["rank"].tolist() == list(range(1, k + 1))
    with pytest.raises(ValueError):
        dd.worklist(REAL, week, 0.03)


@real
def test_real_saved_worklist_equals_engine_recomputation():
    from roadwatch.stage5 import build_worklists
    week = dd.available_weeks(REAL)[10]
    saved = dd.worklist(REAL, week, 0.01)
    fresh, excl = build_worklists(REAL, "multi_block__LGBM", 0.01, origins=[week])
    assert len(excl) == 0
    assert saved["block_id"].tolist() == fresh.sort_values("rank")["block_id"].tolist()


@real
def test_real_custom_capacity_uses_engine():
    week = dd.available_weeks(REAL)[-1]
    wl = dd.custom_worklist(REAL, week, 40)
    assert len(wl) == 40 and wl["selection_rule"].iloc[0].startswith("top-K + non-adjacent")
    assert wl["block_id"].tolist() == dd.worklist(REAL, week, 0.01)["block_id"].head(40).tolist()  # same ranking prefix


@real
def test_real_displayed_metrics_match_stage5_outputs():
    s = dd.policy_summary(REAL)
    dec = json.loads((ROOT / "outputs" / "stage5" / "decided_policy_test_summary.json").read_text())["results"]
    r = s[(s["policy"] == dd.RECOMMENDED) & (s["period"] == "test")].set_index("capacity")
    for cap, name in ((0.005, "0.5pct"), (0.01, "1pct"), (0.02, "2pct")):
        assert r.loc[cap, "captured_any_candidate"] == dec[f"{name}|overall"]["reports_captured"]
        assert r.loc[cap, "precision_own_target"] == pytest.approx(dec[f"{name}|overall"]["precision"])
    pc = pd.read_csv(ROOT / "outputs" / "stage5" / "policy_comparison.csv")
    top = pc[(pc["policy"] == "multi_block__LGBM") & (pc["capacity"] == 0.01) & (pc["period"] == "overall")].iloc[0]
    row = s[(s["policy"] == "Plain top-K multi-block LightGBM") & (s["capacity"] == 0.01) & (s["period"] == "test")].iloc[0]
    assert row["capture_any_candidate"] == pytest.approx(top["report_capture_rate"])
    assert row["precision_own_target"] == pytest.approx(top["precision"])


@real
def test_real_map_matches_worklist_and_scores_rank_consistently():
    week = dd.available_weeks(REAL)[-1]
    wl = dd.worklist(REAL, week, 0.01)
    sc = dd.week_scores(REAL, week)
    bg, sel = dm.layer_frames(dd.block_paths(REAL), sc, wl, 80)
    assert sel["block_id"].tolist() == wl["block_id"].tolist() and sel["path"].notna().all()
    assert len(sc) == 23448 and sc["rank"].is_unique
    s = sc.set_index("block_id").loc[wl["block_id"], "score"].to_numpy()
    assert np.all(np.diff(s) <= 1e-12)                                            # list follows the score ranking


# ---------------- headless app interactions (Streamlit AppTest) ----------------
@real
def test_app_interactions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    metric = {m.label: m.value for m in at.metric}
    assert metric["Blocks to inspect"] == "235"                                    # default 1%, not set by volume forecast
    cap_radio = [r for r in at.radio if r.label.startswith("Inspection capacity")][0]
    cap_radio.set_value("2% (469 blocks)").run()
    assert {m.label: m.value for m in at.metric}["Blocks to inspect"] == "469" and not at.exception
    week_box = [s for s in at.selectbox if s.label.startswith("Forecast week")][0]
    week_box.set_value(dd.available_weeks(REAL)[0]).run()
    assert not at.exception and {m.label: m.value for m in at.metric}["Forecast week starting"] == "07 Jul 2025"
    at.text_input[0].set_value("avenue that does not exist").run()
    assert not at.exception and any("0 of 469 listed blocks match" in c.value for c in at.caption)
    view = [r for r in at.radio if r.label == "View"][0]
    view.set_value("Decision-policy comparison").run()
    assert not at.exception
    period = [r for r in at.radio if r.label == "Period"][0]
    period.set_value("March 2026 surge (5 weeks)").run()
    assert not at.exception and len(at.dataframe) >= 2
    shown = [d.value for d in at.dataframe if "Policy" in d.value.columns][0]
    assert shown["Policy"].iloc[0].startswith("Stage-5 decision") and shown["Slots / week"].iloc[0] == 469
    for v in ("Weekly volume & scenarios", "Evidence & limitations", "Data provenance"):
        [r for r in at.radio if r.label == "View"][0].set_value(v).run()
        assert not at.exception, v
    [r for r in at.radio if r.label == "View"][0].set_value("Weekly volume & scenarios").run()
    assert any("did not reliably outperform simple baselines" in e.value for e in at.error)


@real
def test_app_custom_capacity_and_deep_link():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.query_params["week"] = "2026-03-16"
    at.query_params["view"] = "map"
    at.run()
    assert {m.label: m.value for m in at.metric}["Forecast week starting"] == "16 Mar 2026"
    [r for r in at.radio if r.label.startswith("Inspection capacity")][0].set_value("Custom number of blocks").run()
    at.number_input[0].set_value(40).run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Blocks to inspect"] == "40" and m["Capacity scenario"] == "40 slots"


@real
def test_app_handles_invalid_week_query_param():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.query_params["week"] = "2031-02-30"
    at.run()
    assert not at.exception
    assert any("not a valid date" in i.value for i in at.info)
