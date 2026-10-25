"""Stage 8 regression tests: dashboard default list = strict single-block top-K, consistent with saved Stage 5/7 evidence.
Integration tests read real saved outputs and are skipped when those are absent."""
import json
from pathlib import Path

import pandas as pd
import pytest

from roadwatch import dashboard_data as dd
from roadwatch.config import Paths

ROOT = Path(__file__).resolve().parents[1]
REAL = Paths(ROOT)
real = pytest.mark.skipif(not (ROOT / "data" / "processed" / "stage5" / "worklists").exists(), reason="real outputs absent")
SEL = ROOT / "data" / "processed" / "stage5" / "selections"


def test_config_default_is_strict_topk_and_other_lists_remain():
    cfg = json.loads((ROOT / "config" / "stage5.json").read_text(encoding="utf-8"))
    assert cfg["dashboard_default_list"] == "strict_topk"
    assert dd.default_list(REAL) == "strict_topk"
    assert set(dd.LISTS) == {"strict_topk", "multi_nonadj", "strict_nonadj"}     # comparisons stay selectable
    assert dd.LISTS["strict_topk"][2] == "primary__LGBM" and dd.LISTS["strict_topk"][3] is False


def test_cli_worklist_rule_maps_to_non_adjacent_override(monkeypatch, tmp_path):
    """--rule topk / nonadjacent / config must reach build_worklists as False / True / None (SYNTHETIC: no data read)."""
    from roadwatch import cli, stage5
    seen = []
    monkeypatch.setattr(stage5, "single_worklist",
                        lambda paths, week, cap, pol, div, rule: seen.append((str(week), cap, pol, rule)) or [])
    for rule in ("topk", "nonadjacent", "config"):
        assert cli.main(["--root", str(tmp_path), "stage5", "worklist", "--week", "2026-03-16", "--policy", "primary__LGBM",
                         "--rule", rule]) == 0
    assert [s[3] for s in seen] == ["topk", "nonadjacent", "config"]


def test_single_worklist_rule_reaches_build_worklists(monkeypatch, tmp_path):
    from roadwatch import stage5
    captured = {}
    monkeypatch.setattr(stage5, "cfg_load", lambda p: {"primary_policy": "x"})
    monkeypatch.setattr(stage5, "dirs", lambda p: (tmp_path, tmp_path))

    def fake_build(paths, pol, cap, origins=None, diversify=False, non_adjacent=None):
        captured.setdefault("na", []).append(non_adjacent)
        raise RuntimeError("stop")       # only the routing is under test
    monkeypatch.setattr(stage5, "build_worklists", fake_build)
    for rule in ("topk", "nonadjacent", "config"):
        with pytest.raises(RuntimeError):
            stage5.single_worklist(REAL, None, 0.01, "primary__LGBM", False, rule)
    assert captured["na"] == [False, True, None]


@real
def test_default_worklist_and_exports_equal_saved_strict_selection():
    week = dd.available_weeks(REAL)[-1]
    for cap, k, stem in ((0.005, 118, "primary__LGBM__0.5pct"), (0.01, 235, "primary__LGBM__1pct"), (0.02, 469, "primary__LGBM__2pct")):
        wl = dd.worklist(REAL, week, cap, dd.default_list(REAL))
        sel = pd.read_parquet(SEL / f"{stem}.parquet")
        sel = sel[pd.to_datetime(sel["origin"]).dt.date == week].sort_values("rank")
        assert len(wl) == k and wl["block_id"].tolist() == sel["block_id"].tolist()
        assert set(wl["selection_rule"]) == {"top-K"} and wl["model_version"].str.startswith("primary__LGBM").all()
        gj = json.loads(dd.to_geojson_bytes(wl))
        assert [f["properties"]["block_id"] for f in gj["features"]] == sel["block_id"].tolist()
        csv = pd.read_csv(pd.io.common.BytesIO(dd.to_csv_bytes(wl)))
        assert csv["block_id"].tolist() == sel["block_id"].tolist()


@real
def test_app_opens_on_strict_list_with_consistent_metrics():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.query_params["week"] = "2026-03-16"
    at.run()
    assert not at.exception
    assert [r for r in at.radio if r.label == "Inspection list"][0].value == "strict_topk"
    m = {x.label: x.value for x in at.metric}
    assert m["Blocks to inspect"] == "235"
    assert any("selection rule: **top-K**" in c.value for c in at.caption)
    assert any("single-block target" in i.value for i in at.info)
    # evidence view shows the strict list's saved Stage-7 capture figures verbatim
    [r for r in at.radio if r.label == "View"][0].set_value("Evidence & limitations").run()
    assert not at.exception
    t, _ = dd.capture_definitions(REAL)
    x = t[(t["period"] == "test") & (t["capacity"] == 0.01)].set_index("list").loc["Strict single-block LightGBM (top-K)"]
    m = {x_.label: x_.value for x_ in at.metric}
    assert m["Strict single-block top-K at 1%"] == f"{100 * x['any_candidate']:.1f}% / {100 * x['fractional']:.1f}%"
