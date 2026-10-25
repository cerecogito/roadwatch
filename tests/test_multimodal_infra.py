"""Multimodal infrastructure features: spatial attribution, temporal cutoff, taxonomy mapping, missing-source behaviour.
Uses small SYNTHETIC blocks and events (labelled as such); no network."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import shapely

from roadwatch.multimodal import infra_features as inf
from roadwatch.multimodal import ingest

D = dt.date


def test_taxonomy_old_and_new_map_to_same_groups():
    assert ingest.group_of("Possible Water Main Break (Use Comments) (WA1)", "Water System") == "water_main_leak"
    assert ingest.group_of("Water Main Break", "Water Maintenance") == "water_main_leak"
    assert ingest.group_of("Catch Basin Clogged/Flooding (Use Comments) (SC)", "Sewer") == ingest.group_of("Catch Basin Clogged", "Sewer Maintenance") == "drain_flood"
    assert ingest.group_of("Hydrant Running (WC3)", "Water System") == ingest.group_of("Fire Hydrant Running", "Water Maintenance") == "hydrant_water"
    assert ingest.group_of("Sewer Backup (Use Comments) (SA)", "Sewer") is None            # private plumbing: deliberately ignored
    assert ingest.group_of("Dirty Water", "Water Maintenance") is None
    assert ingest.group_of(None, None) is None


def test_attribution_uses_nearest_block_within_100ft_and_area_within_300ft():
    geoms = np.array([shapely.LineString([(0, 0), (1000, 0)]), shapely.LineString([(0, 250), (1000, 250)])])
    near, area = inf.attribute(geoms, np.array([500.0, 500.0, 500.0]), np.array([40.0, 125.0, 5000.0]))
    assert near.tolist() == [0, -1, -1]                       # 40 ft -> block 0; 125 ft from both -> too far; far away -> none
    pairs = set(zip(area[0].tolist(), area[1].tolist()))
    assert (0, 0) in pairs and (1, 0) in pairs and (1, 1) in pairs and not any(e == 2 for e, _ in pairs)
    assert (0, 1) in pairs is False or True                    # block 1 is 210 ft from event 0 -> within 300 ft too
    assert (0, 1) in pairs


@pytest.fixture
def small(monkeypatch):
    geoms = np.array([shapely.LineString([(0, 0), (1000, 0)]), shapely.LineString([(0, 2000), (1000, 2000)])])
    monkeypatch.setattr(inf, "load_blocks", lambda paths: (np.array([11, 22]), geoms))
    return geoms


def _events(rows):
    return pd.DataFrame([{"created": pd.Timestamp(c), "x": x, "y": y, "group": g} for c, x, y, g in rows])


def test_temporal_cutoff_is_t_minus_2_days_and_window_is_half_open(small):
    t = D(2024, 3, 11)                                     # a Monday
    ev = _events([("2024-03-08 23:59", 500, 10, "drain_flood"),   # day t-3: visible
                  ("2024-03-09 00:01", 500, 10, "drain_flood"),   # day t-2: NOT visible (inside the 2-day lag)
                  ("2024-03-10 12:00", 500, 10, "drain_flood"),   # t-1: not visible
                  ("2024-03-11 08:00", 500, 10, "drain_flood"),   # on/after t: never visible
                  ("2023-03-09 10:00", 500, 10, "drain_flood")])  # 368 days before t -> outside the 365-day window
    f = inf.build(None, ev, [t]).set_index("block_id")
    assert f.loc[11, "inf_drain_flood_n30"] == 1 and f.loc[11, "inf_drain_flood_n365"] == 1
    assert f.loc[11, "inf_drain_flood_days_since"] == 1    # measured from the data cutoff t-2d, like production ("last report N days before cutoff")
    assert f.loc[22, "inf_drain_flood_n365"] == 0 and f.loc[22, "inf_drain_flood_days_since"] == inf.CAP    # no event in the window -> cap, not zero days
    assert f.loc[11, "inf_hydrant_water_n365"] == 0


def test_later_events_never_change_earlier_features(small):
    base = [("2024-02-01 10:00", 500, 10, "water_main_leak")]
    t = D(2024, 3, 11)
    a = inf.build(None, _events(base), [t])
    b = inf.build(None, _events(base + [("2024-06-01 10:00", 500, 10, "water_main_leak"), ("2025-01-01 10:00", 500, 10, "drain_flood")]), [t])
    pd.testing.assert_frame_equal(a, b)                       # events after the forecast date cannot leak in


def test_events_without_coordinates_or_group_are_dropped_not_zero_filled(small):
    ev = _events([("2024-03-01 10:00", np.nan, np.nan, "drain_flood"), ("2024-03-01 10:00", 500, 10, None)])
    f = inf.build(None, ev, [D(2024, 3, 11)])
    counts = [c for c in inf.feature_names() if not c.endswith("days_since")]
    assert f[counts].sum().sum() == 0 and len(f) == 2 and f["inf_drain_flood_days_since"].eq(inf.CAP).all()


def test_feature_definitions_cover_every_feature_with_units_and_availability():
    d = {x["feature"]: x for x in inf.definitions()}
    assert set(d) == set(inf.feature_names()) and all(x["units"] and x["availability"] for x in d.values())
