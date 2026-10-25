"""Weekly calendar, label logic, observability and reproducibility. SYNTHETIC data only."""
import datetime as dt
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from roadwatch import stage3
from roadwatch.config import Paths
from roadwatch.labels import assign_week_status, build_calendar, build_panel, utc_hours, week_origin
from roadwatch.stage3_diagnostics import annual_rate_drift
from roadwatch.storage import sha256_file, write_json
from stage3_fixtures import centerline

D = dt.date


def test_week_origin_boundaries():
    s = pd.Series(pd.to_datetime(["2025-01-06 00:00:00", "2025-01-12 23:59:59", "2025-01-13 00:00:00"]))
    assert week_origin(s).tolist() == [D(2025, 1, 6), D(2025, 1, 6), D(2025, 1, 13)]


def test_dst_weeks_have_167_or_169_utc_hours():
    assert utc_hours(D(2025, 3, 3)) == 167      # contains 2025-03-09 spring-forward
    assert utc_hours(D(2025, 10, 27)) == 169    # contains 2025-11-02 fall-back
    assert utc_hours(D(2025, 1, 6)) == 168


def test_calendar_and_incomplete_final_horizons():
    w = build_calendar(D(2025, 1, 1), D(2025, 2, 6), embargo_days=2)   # data through 2025-02-05, observable to 02-04
    assert w["origin"].iloc[0] == D(2025, 1, 6) and w["origin"].iloc[-1] == D(2025, 2, 3)
    assert w.set_index("origin")["data_complete"].to_dict() == {
        D(2025, 1, 6): True, D(2025, 1, 13): True, D(2025, 1, 20): True, D(2025, 1, 27): True, D(2025, 2, 3): False}


def _weeks(statuses):
    w = pd.DataFrame({"origin": [D(2025, 1, 6) + dt.timedelta(days=7 * i) for i in range(len(statuses))]})
    w["week_status"] = pd.Categorical(statuses, categories=["observable", "unlocated_reports", "incomplete_final"])
    return w


def test_positive_negative_unknown_logic_and_no_double_counting():
    weeks = _weeks(["observable", "unlocated_reports", "incomplete_final"])
    pos = pd.DataFrame({"block_id": [1, 1, 1, 2], "origin": [D(2025, 1, 6), D(2025, 1, 6), D(2025, 1, 13), D(2025, 1, 20)],
                        "report_id": ["a", "a", "b", "c"]})   # report 'a' duplicated -> counted once
    p = build_panel(np.array([1, 2, 3]), weeks, pos).set_index(["block_id", "origin"])
    assert p.loc[(1, D(2025, 1, 6)), "n_reports"] == 1 and p.loc[(1, D(2025, 1, 6)), "label"] == 1
    assert p.loc[(3, D(2025, 1, 6)), "label"] == 0                                  # observable, no report
    assert p.loc[(1, D(2025, 1, 13)), "label"] == 1                                 # positive survives degraded week
    assert pd.isna(p.loc[(3, D(2025, 1, 13)), "label"])                             # unknown, NOT zero
    assert p.loc[(3, D(2025, 1, 13)), "unknown_reason"] == "unlocated_reports"
    assert p.loc[(2, D(2025, 1, 20)), "label"] == 1 and pd.isna(p.loc[(1, D(2025, 1, 20)), "label"])
    assert p.loc[(1, D(2025, 1, 20)), "unknown_reason"] == "incomplete_final"
    assert (p.loc[p["label_status"] == "unknown", "label"].isna()).all()
    assert not (p.loc[p["label_status"] == "unknown", "n_reports"] > 0).any()


def test_ambiguous_candidates_unknown_in_confident_variant():
    weeks = _weeks(["observable"])
    pos = pd.DataFrame({"block_id": [1], "origin": [D(2025, 1, 6)], "report_id": ["a"]})
    cand = pd.DataFrame({"block_id": [1, 2], "origin": [D(2025, 1, 6)] * 2})
    p = build_panel(np.array([1, 2, 3]), weeks, pos, cand).set_index("block_id")
    assert p.loc[1, "label"] == 1 and pd.isna(p.loc[2, "label"]) and p.loc[2, "unknown_reason"] == "ambiguous_candidate"
    assert p.loc[3, "label"] == 0


def test_week_status_threshold():
    w = build_calendar(D(2025, 1, 6), D(2025, 2, 10), 2)
    located = pd.Series({D(2025, 1, 6): 100, D(2025, 1, 13): 100})
    hidden = pd.Series({D(2025, 1, 6): 5, D(2025, 1, 13): 6})     # shares 4.8% and 5.7%
    st = assign_week_status(w, located, hidden, 0.05).set_index("origin")["week_status"].astype(str)
    assert st[D(2025, 1, 6)] == "observable" and st[D(2025, 1, 13)] == "unlocated_reports"


def test_annual_rate_drift_detection():
    rates = pd.Series({2021: 99.0, 2022: 98.8, 2023: 92.0, 2024: 98.9})
    flags = annual_rate_drift(rates, tol_pp=3.0)
    assert [f["year"] for f in flags] == [2023]


# ---------------- end-to-end on a synthetic processed layout ----------------
def _write_fixture(root):
    paths = Paths(root)
    proc = paths.processed
    proc.mkdir(parents=True)
    centerline().to_parquet(proc / "centerline.parquet")
    snap = root / "data" / "raw" / "sr311" / "SYN"
    snap.mkdir(parents=True)
    write_json(snap / "_state.json", {"requested_start": "2025-01-01", "requested_end_exclusive": "2025-02-06"})
    for k in ("centerline", "sr311", "sr311_unassigned"):
        write_json(proc / f"{k}.meta.json", {"snapshot_id": "SYN", "raw_snapshot": "data/raw/sr311/SYN", "sha256": "x"})

    def rows(specs, borough):
        out = []
        for rid, created, at, street, c1, c2, i1, i2, x, y in specs:
            out.append({"unique_key": rid, "created_date": created, "created_date_local": pd.Timestamp(created),
                        "complaint_type": "Street Condition", "descriptor": "Pothole", "address_type": at,
                        "street_name": street, "incident_address": street, "cross_street_1": c1, "cross_street_2": c2,
                        "intersection_street_1": i1, "intersection_street_2": i2, "incident_zip": "11299",
                        "borough": borough, "x_coordinate_state_plane_num": x, "y_coordinate_state_plane_num": y,
                        "latitude_num": None, "longitude_num": None, "status": "Closed", "closed_date_local": pd.NaT,
                        "resolution_description": None})
        return pd.DataFrame(out)
    prim = rows([("p1", "2025-01-07T09:00:00.000", "ADDRESS", "1 AVENUE", "EAST 1 STREET", "EAST 2 STREET", None, None, 2.0, 100.0),
                 ("p2", "2025-01-13T00:00:00.000", "BLOCKFACE", "2 AVENUE", "EAST 1 STREET", "EAST 2 STREET", None, None, None, None),
                 ("p3", "2025-01-21T08:00:00.000", "INTERSECTION", None, None, None, "1 AVENUE", "EAST 2 STREET", 0.0, 300.0),
                 ("p4", "2025-01-28T08:00:00.000", "BLOCKFACE", "1 AVENUE", "EAST 1 STREET", "EAST 2 STREET", None, None, None, None)],
                "BROOKLYN")
    supp = rows([("u1", "2025-01-28T10:00:00.000", "INTERSECTION", None, None, None, None, None, None, None)] * 1, None)
    # make week of 2025-01-27 degraded: 1 hidden vs 1 located report -> 50% > 5%
    prim.to_parquet(proc / "sr311.parquet")
    supp.to_parquet(proc / "sr311_unassigned.parquet")
    return paths


def test_stage3_end_to_end_is_reproducible(tmp_path):
    p1, p2 = _write_fixture(tmp_path / "a"), _write_fixture(tmp_path / "b")
    m1 = stage3.run(p1, expected_blocks=None)
    m2 = stage3.run(p2, expected_blocks=None)
    for name in ("labels_primary.parquet", "labels_alt_tiebreak.parquet", "labels_multi_block.parquet",
                 "labels_confident_only.parquet", "report_assignments.parquet", "label_weeks.parquet"):
        a = pd.read_parquet(p1.processed / "stage3" / name)
        b = pd.read_parquet(p2.processed / "stage3" / name)
        pd.testing.assert_frame_equal(a, b)
    lab = pd.read_parquet(p1.processed / "stage3" / "labels_primary.parquet").set_index(["block_id", "origin"])
    assert lab.loc[(101, D(2025, 1, 6)), "label"] == 1          # p1 (Tue) in week of Jan 6
    assert lab.loc[(201, D(2025, 1, 13)), "label"] == 1         # p2 at Monday 00:00 -> starts week of Jan 13
    assert lab.loc[(201, D(2025, 1, 6)), "label"] == 0          # ...and is excluded from the previous week
    assert lab.loc[(101, D(2025, 1, 27)), "label"] == 1         # positive survives the degraded week
    assert pd.isna(lab.loc[(201, D(2025, 1, 27)), "label"])     # degraded week: unknown, not zero
    assert lab.xs(D(2025, 2, 3), level="origin")["label_status"].astype(str).eq("unknown").all()   # incomplete final
    w = pd.read_parquet(p1.processed / "stage3" / "label_weeks.parquet").set_index("origin")
    assert w.loc[D(2025, 1, 27), "week_status"] == "unlocated_reports"
    mb = pd.read_parquet(p1.processed / "stage3" / "labels_multi_block.parquet").set_index(["block_id", "origin"])
    assert all(mb.loc[(b, D(2025, 1, 20)), "label"] == 1 for b in (101, 102, 302))   # intersection -> all candidates
    assert m1["label_summaries"] == m2["label_summaries"]
