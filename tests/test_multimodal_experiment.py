"""Experiment harness and context tests. Unit tests use SYNTHETIC data; integration tests read the real saved artefacts."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roadwatch.config import Paths
from roadwatch.multimodal import context as ctx
from roadwatch.multimodal import experiment as ex

ROOT = Path(__file__).resolve().parents[1]
real = pytest.mark.skipif(not (ROOT / "data" / "processed" / "multimodal" / "infra_features.parquet").exists(), reason="experimental tables not built")


def test_paired_ci_is_paired_and_deterministic():
    rng = np.random.default_rng(0)
    base_n = rng.integers(5, 15, 60).astype(float)
    den = np.full(60, 100.0)
    same = ex.paired_ci(base_n, den, base_n, den)
    assert same == [0.0, 0.0]                                   # identical candidates -> difference exactly zero in every resample
    better = ex.paired_ci(base_n + 3, den, base_n, den)
    assert better[0] > 0.02 and better == ex.paired_ci(base_n + 3, den, base_n, den)   # a constant +3% shows up; reproducible


def test_feature_list_keeps_production_features_first_and_never_duplicates():
    f0 = ex.feature_list([])
    f = ex.feature_list(["infra", "terrain"])
    assert f[:len(f0)] == f0 and len(f) == len(set(f)) == len(f0) + len(ex.GROUP_FEATURES["infra"]) + len(ex.GROUP_FEATURES["terrain"])
    assert set(ex.SETS) == {"F0", "F1", "F2", "F3", "F4", "F5"} and ex.SETS["F0"] == []


def test_merge_preserves_rows_and_never_zero_fills(monkeypatch):
    class E:
        dyn = pd.DataFrame({"block_id": [1, 1], "origin": pd.to_datetime(["2024-01-01", "2024-01-08"]), "inf_drain_flood_n30": [2.0, 3.0]})
        static = pd.DataFrame({"block_id": [1], "terrain_elev_mean_ft": [10.0]})
    d = ex.MMData.__new__(ex.MMData)
    d.extras, d.groups = E(), ["infra", "terrain"]
    left = pd.DataFrame({"block_id": [1, 1, 2], "origin": [dt.date(2024, 1, 1), dt.date(2024, 1, 8), dt.date(2024, 1, 8)]})
    out = d._merge(left)
    assert len(out) == 3 and out["inf_drain_flood_n30"].tolist()[:2] == [2.0, 3.0]
    assert np.isnan(out["inf_drain_flood_n30"].iloc[2]) and np.isnan(out["terrain_elev_mean_ft"].iloc[2])    # unknown block stays missing, not 0


def test_unavailable_source_is_reported_not_faked(tmp_path):
    e = ex.Extras(Paths(tmp_path))                                 # empty processed dir: nothing built
    assert not e.available("infra") and not e.available("terrain") and not e.available("displacement") and not e.available("permits")


def test_context_is_silent_when_tables_absent(tmp_path):
    assert ctx.infra_context(Paths(tmp_path), 1, dt.date(2026, 3, 16)) is None
    assert ctx.experiment_results(Paths(tmp_path)) is None


@real
def test_saved_infra_features_cover_all_blocks_and_origins_without_nans():
    f = pd.read_parquet(ROOT / "data" / "processed" / "multimodal" / "infra_features.parquet")
    assert f["block_id"].nunique() == 23448 and f["origin"].nunique() == 302
    assert not f.isna().any().any() and (f.filter(like="_days_since") <= 365).all().all() and (f.filter(like="_n") >= 0).all().all()
    ctxt = ctx.infra_context(Paths(ROOT), 52878, dt.date(2026, 3, 16))
    assert ctxt["groups"]["water_main_leak"]["on_block_365d"] >= 1 and "not used by the ranking model" in ctxt["note"]


@real
def test_harness_f0_reproduces_production_predictions_exactly():
    f = ROOT / "data" / "processed" / "multimodal" / "predictions" / "test_lgbm_F0.parquet"
    if not f.exists():
        pytest.skip("test-period F0 not run")
    mine = pd.read_parquet(f)
    prod = pd.read_parquet(ROOT / "data" / "processed" / "stage4" / "predictions" / "primary__LGBM.parquet")
    m = mine.merge(prod, on=["block_id", "origin"], suffixes=("_m", "_p"))
    assert len(m) == len(mine) == len(prod) and float((m["score_m"] - m["score_p"]).abs().max()) == 0.0


@real
def test_new_assistant_tools_return_saved_figures():
    from roadwatch.integrations.evidence_tools import EvidenceTools
    t = EvidenceTools()
    c = t.call("get_block_context", {"block_id": 52878, "week": "2026-03-16"})
    assert c["data"]["groups"]["drain_flood"]["on_block_365d"] == 4 and c["sources"]
    r = t.call("get_multimodal_results", {"stage": "validation"})
    assert "F3" in r["data"]["feature_sets"] and "no demonstrated gain" in r["data"]["meaning"]
    assert "error" in t.call("get_block_context", {"block_id": 52878, "week": "2031-01-01"})
