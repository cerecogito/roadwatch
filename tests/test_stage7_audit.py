"""Stage 7 regression tests for audit findings. Unit tests use SYNTHETIC fixtures; integration tests read real saved outputs."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roadwatch import dashboard_data as dd
from roadwatch import inspection as insp
from roadwatch.config import Paths
from roadwatch.stage5 import _capture

D = dt.date
W1, W2, W3 = D(2025, 7, 7), D(2025, 7, 14), D(2025, 7, 21)
ROOT = Path(__file__).resolve().parents[1]
REAL = Paths(ROOT)
real = pytest.mark.skipif(not (ROOT / "outputs" / "stage5" / "capture_definitions.csv").exists(), reason="real outputs absent")


def test_capture_keeps_zero_capture_weeks_in_sorted_order():
    """Finding L (Stage 7): pd.concat moved zero-capture weeks to the end, misaligning paired bootstraps."""
    rp = pd.DataFrame({"block_id": [1, 2, 3], "origin": [W1, W2, W3], "report_id": ["a", "b", "c"]})
    sel = pd.DataFrame({"block_id": [1, 3], "origin": [W1, W3]})          # nothing captured in W2
    c = _capture(sel, rp)
    assert list(c.index) == [W1, W2, W3] and c["captured"].tolist() == [1, 0, 1] and c["total"].tolist() == [1, 1, 1]


def _rf():  # SYNTHETIC report frame
    nodes = pd.DataFrame({"node_from": [10, 10, 11, 20, 21], "node_to": [11, 12, 13, 21, 22]}, index=[1, 2, 3, 4, 5])
    a = pd.DataFrame({
        "origin": [W1] * 4, "report_id": ["u", "int", "bf", "far"],
        "candidate_block_ids": [[1], [1, 2], [4, 5], [1, 4]], "assigned_block_id": [1, 2, 5, 4],
        "match_status": ["matched", "ambiguous", "ambiguous", "ambiguous"],
        "method": ["coord_name", "intersection_node", "blockface_one_sided", "intersection_multi_node"],
        "included_in_labels": True})
    return insp.report_frame(a, nodes)


def test_report_frame_intersection_credit_only_for_intersection_reports():
    rf = _rf().set_index("report_id")
    assert bool(rf.loc["int", "shared_node"])                 # intersection, candidates meet at node 10
    assert not bool(rf.loc["bf", "shared_node"])              # blockface alternatives share node 21 but are NOT intersections
    assert not bool(rf.loc["far", "shared_node"])             # intersection report but candidates share no node
    assert not bool(rf.loc["u", "shared_node"])


def test_capture_by_definition_values():
    rf = _rf()
    sel = pd.DataFrame({"block_id": [1, 4], "origin": [W1, W1]})
    w = insp.capture_by_definition(sel, rf).iloc[0]
    # u: captured (1) everywhere; int: any 1, assigned 0 (assigned 2), frac 0.5, node 1; bf: any 1, assigned 0, frac 0.5,
    # node 0.5 (fractional, not intersection); far: any 1, assigned 1 (assigned 4), frac 1.0, node 1.0 (fractional = 2/2)
    assert w["any_candidate"] == 4 and w["assigned_block"] == 2
    assert w["fractional"] == pytest.approx(1 + 0.5 + 0.5 + 1.0)
    assert w["intersection_node"] == pytest.approx(1 + 1 + 0.5 + 1.0)
    assert w["reports_total"] == 4


def test_missing_audit_outputs_raise_actionable_error(tmp_path):
    """Empty root: the dashboard loaders raise MissingOutput naming the command that builds the file."""
    p = Paths(tmp_path)
    with pytest.raises(dd.MissingOutput, match="capture-definitions"):
        dd.capture_definitions(p)
    with pytest.raises(dd.MissingOutput, match="alternative-worklists"):
        dd.worklist(p, W1, 0.01, "strict_topk")
    with pytest.raises(ValueError):
        dd.worklist(p, W1, 0.01, "no_such_list")


@real
def test_saved_alternative_lists_equal_saved_selections():
    week = dd.available_weeks(REAL)[20]
    for key, stem in (("strict_topk", "primary__LGBM__1pct"), ("strict_nonadj", "primary__LGBM__1pct__nonadjacent"),
                      ("multi_nonadj", "multi_block__LGBM__1pct__nonadjacent")):
        wl = dd.worklist(REAL, week, 0.01, key)
        sel = pd.read_parquet(ROOT / "data" / "processed" / "stage5" / "selections" / f"{stem}.parquet")
        sel = sel[pd.to_datetime(sel["origin"]).dt.date == week].sort_values("rank")
        assert wl["block_id"].tolist() == sel["block_id"].tolist(), key
    assert set(dd.worklist(REAL, week, 0.01, "strict_topk")["selection_rule"]) == {"top-K"}
    assert set(dd.worklist(REAL, week, 0.01, "strict_nonadj")["selection_rule"]) == {"top-K + non-adjacent"}


@real
def test_capture_definitions_match_saved_stage5_counts():
    t, meta = dd.capture_definitions(REAL)
    x = t[(t["period"] == "test") & (t["capacity"] == 0.01)].set_index("list")
    dec = x.loc["Stage-5 decision: multi-block LightGBM + non-adjacent"]
    assert dec["any_candidate"] == pytest.approx(3125 / 16855) and dec["reports"] == 16855
    assert meta["test_report_composition"]["reports"] == 16855
    s = dd.policy_summary(REAL)
    row = s[(s["policy"] == "Strict single-block LightGBM (top-K)") & (s["capacity"] == 0.01) & (s["period"] == "test")].iloc[0]
    assert x.loc["Strict single-block LightGBM (top-K)", "assigned_block"] == pytest.approx(row["capture_assigned_block"])


@real
def test_corrected_capacity_guidance_uses_adopted_list():
    import json
    g = json.loads((ROOT / "outputs" / "stage5" / "capacity_guidance.json").read_text())
    assert g["adopted_list"]["list"].endswith("__nonadjacent")
    assert g["adopted_list"]["fixed_middle"]["reports_captured"] == 3125     # equals the decided policy's 1% result


@real
def test_app_list_selector_and_policy_view():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    lst = [r for r in at.radio if r.label == "Inspection list"][0]
    lst.set_value("strict_topk").run()
    assert not at.exception and {m.label: m.value for m in at.metric}["Blocks to inspect"] == "235"
    assert any("selection rule: **top-K**" in c.value for c in at.caption)
    [r for r in at.radio if r.label == "View"][0].set_value("Decision-policy comparison").run()
    assert not at.exception
    assert any("depends on what counts as 'captured'" in w.value for w in at.warning)
    cols = at.dataframe[0].value.columns.tolist()
    assert cols == ["List", "Any candidate block", "Assigned block", "Fractional (expected)", "Intersection-node coverage"]
