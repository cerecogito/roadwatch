"""Block index, name normalisation and matching. SYNTHETIC fixtures only (tests/stage3_fixtures.py)."""
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString

from roadwatch.blocks import BlockIndexError, build_network, validate_blocks
from roadwatch.matching import Matcher, MatchParams, derive_aliases, hashed_pick
from roadwatch.names import normalize_street, street_key
from roadwatch.stage3 import classify_supplementary
from stage3_fixtures import centerline, report, reports


@pytest.fixture(scope="module")
def net():
    return build_network(centerline())


@pytest.fixture(scope="module")
def matcher(net):
    return Matcher(net, MatchParams())


def one(matcher, **kw):
    return matcher.match(reports([report("r1", **kw)])).iloc[0]


# ---------------- block index ----------------
def test_block_index_ids_crs_and_eligibility(net):
    b = net.blocks
    assert b.crs.to_epsg() == 2263
    assert sorted(b["block_id"]) == sorted(i for i, *_ , rw in [(s[0], s[3]) for s in __import__("stage3_fixtures").SEGMENTS] if rw == 1)
    assert 103 not in set(b["block_id"]) and 900 not in set(b["block_id"])   # bridge and highway are not blocks
    assert validate_blocks(b, expected_count=len(b))[0]["status"] == "PASS"


def test_block_index_from_wgs84_is_projected():
    gdf = centerline().to_crs("EPSG:4326")
    assert build_network(gdf).blocks.crs.to_epsg() == 2263


def test_validate_blocks_detects_duplicates_invalid_geometry_coverage_and_crs(net):
    b = net.blocks.copy()
    dup = b.copy()
    dup.loc[1, "block_id"] = dup.loc[0, "block_id"]
    with pytest.raises(BlockIndexError, match="duplicated"):
        validate_blocks(dup, None)
    bad = b.copy()
    bad.loc[0, "geometry"] = LineString([(0, 0), (0, 0)])  # degenerate line -> invalid
    with pytest.raises(BlockIndexError, match="invalid block geometry"):
        validate_blocks(bad, None)
    with pytest.raises(BlockIndexError, match="coverage changed"):
        validate_blocks(b, expected_count=len(b) + 5)
    with pytest.raises(BlockIndexError, match="EPSG:2263"):
        validate_blocks(b.to_crs("EPSG:4326"), None)


# ---------------- names ----------------
@pytest.mark.parametrize("raw,expected", [
    ("EAST   18 STREET", "E 18 ST"), ("east 18th street", "E 18 ST"), ("ST MARK'S AVENUE", "ST MARKS AVE"),
    ("FORT HAMILTON PARKWAY", "FT HAMILTON PKWY"), ("PROSPECT PARK SOUTHWEST", "PROSPECT PARK SW"),
    ("AVENUE N", "AVE N"), ("  W  8 ST ", "W 8 ST")])
def test_normalize_street(raw, expected):
    assert normalize_street(raw) == expected


def test_street_key_is_space_insensitive_and_null_safe():
    assert street_key("MC DONALD AVENUE") == street_key("MCDONALD AVE") == "MCDONALDAVE"
    assert street_key("DE KALB AVENUE") == street_key("DEKALB AVE")
    assert street_key(None) is None and street_key("") is None


# ---------------- coordinate matching ----------------
def test_address_point_snaps_to_same_street_block(matcher):
    r = one(matcher, address_type="ADDRESS", street="1 AVENUE", c1="EAST 1 STREET", c2="EAST 2 STREET", x=3, y=150)
    assert (r.status, r.method, r.assigned_block_id, r.confidence) == ("matched", "coord_name", 101, "high")
    assert r.assigned_distance_ft == pytest.approx(3.0)


def test_corner_tie_resolved_by_cross_streets(matcher):
    r = one(matcher, address_type="ADDRESS", street="1 AVENUE", c1="EAST 2 STREET", c2="EAST 3 STREET", x=0, y=300)
    assert (r.status, r.method, r.assigned_block_id) == ("matched", "coord_name_cross", 102)


def test_unresolvable_tie_is_flagged_and_deterministic(matcher):
    kw = dict(address_type="ADDRESS", street="1 AVENUE", c1="NOWHERE ST", c2="NOWHERE ST", x=0, y=300)
    r = one(matcher, **kw)
    assert r.status == "ambiguous" and r.candidate_block_ids == [101, 102] and r.assigned_block_id in (101, 102)
    df = reports([report(f"r{i}", **kw) for i in range(30)])
    a = matcher.match(df)["assigned_block_id"].tolist()
    b = matcher.match(df.iloc[::-1])["assigned_block_id"].tolist()[::-1]   # input order must not matter
    assert a == b and set(a) == {101, 102}                                 # both candidates used across reports


def test_hashed_pick_ignores_candidate_order():
    c = [5, 9, 2, 7]
    assert hashed_pick("x", c, "s") == hashed_pick("x", list(reversed(c)), "s") == hashed_pick("x", sorted(c), "s")


def test_point_beyond_snap_distance_falls_back_to_names(matcher):
    r = one(matcher, address_type="ADDRESS", street="1 AVENUE", c1="EAST 1 STREET", c2="EAST 2 STREET", x=200, y=150)
    assert r.method == "blockface_names" and r.assigned_block_id == 101 and r.name_corroborated == False  # noqa: E712
    assert r.reason == "coordinates_not_corroborating" and r.confidence == "medium"


# ---------------- street-name matching ----------------
def test_blockface_unique_chain_and_divided_road(matcher):
    r = one(matcher, address_type="BLOCKFACE", street="1 AVENUE", c1="EAST 2 STREET", c2="EAST 1 STREET")
    assert (r.status, r.method, r.assigned_block_id) == ("matched", "blockface_names", 101)
    r = one(matcher, address_type="BLOCKFACE", street="1 AVENUE", c1="EAST 3 STREET", c2="EAST 4 STREET")
    assert (r.status, r.method, r.candidate_block_ids) == ("matched", "blockface_chain", [104])   # through the bridge
    r = one(matcher, address_type="BLOCKFACE", street="DIV ROAD", c1="A STREET", c2="B STREET")
    assert r.status == "ambiguous" and r.candidate_block_ids == [401, 402]


def test_unknown_and_out_of_universe_streets(matcher):
    assert one(matcher, address_type="BLOCKFACE", street="NO SUCH AVE", c1="A ST", c2="B ST").reason == "street_not_found"
    r = one(matcher, address_type="BLOCKFACE", street="BELT PARKWAY", c1="EAST 4 STREET", c2="1 AVENUE")
    assert r.status == "out_of_universe"


# ---------------- intersections ----------------
def test_intersection_candidates_and_order_invariance(matcher):
    a = one(matcher, address_type="INTERSECTION", i1="1 AVENUE", i2="EAST 2 STREET", x=0.5, y=300)
    b = one(matcher, address_type="INTERSECTION", i1="EAST 2 STREET", i2="1 AVENUE", x=0.5, y=300)
    assert a.status == "ambiguous" and a.candidate_block_ids == [101, 102, 302] and a.name_corroborated == True  # noqa: E712
    assert b.candidate_block_ids == a.candidate_block_ids and b.assigned_block_id == a.assigned_block_id
    c = one(matcher, address_type="INTERSECTION", i1="1 AVENUE", i2="EAST 2 STREET")   # no coordinates
    assert c.candidate_block_ids == [101, 102, 302] and c.name_corroborated is None


def test_intersection_coordinate_fallback_when_names_unknown(matcher):
    r = one(matcher, address_type="INTERSECTION", i1="LOCAL NAME A", i2="LOCAL NAME B", x=1000, y=1)
    assert r.method == "intersection_coord" and r.confidence == "ambiguous" and set(r.candidate_block_ids) == {201, 301}


# ---------------- aliases and supplementary ----------------
def test_alias_requires_enough_coordinate_evidence(net):
    rows = [report(f"a{i}", "ADDRESS", street="OLD NAME AVENUE", x=1000, y=100 + i) for i in range(5)]
    tbl = derive_aliases(reports(rows), net)
    assert tbl.loc[0, "report_key"] == "OLDNAMEAVE" and tbl.loc[0, "cscl_key"] == "2AVE" and bool(tbl.loc[0, "accepted"])
    tbl4 = derive_aliases(reports(rows[:4]), net)
    assert not tbl4["accepted"].any()


def test_supplementary_classification(matcher):
    rows = [report("s1", "BLOCKFACE", c1="EAST 1 STREET", c2="EAST 2 STREET"),          # cross-only, compatible
            report("s2", "BLOCKFACE", c1="QUEENS BOULEVARD", c2="SOMEWHERE ST"),         # cross-only, incompatible
            report("s3", "INTERSECTION"),                                               # nothing
            report("s4", "ADDRESS", street="1 AVENUE", c1="EAST 1 STREET", c2="EAST 2 STREET", x=2, y=100),
            report("s5", "BLOCKFACE", street="2 AVENUE", c1="EAST 1 STREET", c2="EAST 2 STREET", zip_="11299"),
            report("s6", "BLOCKFACE", street="2 AVENUE", c1="EAST 1 STREET", c2="EAST 2 STREET", zip_="10001")]
    rep = reports(rows)
    cls = classify_supplementary(rep, matcher.match(rep), matcher, {201: {"11299"}, 101: {"11299"}})
    assert cls.tolist() == ["cross_only_brooklyn_compatible", "cross_only_brooklyn_incompatible",
                            "unlocatable_no_information", "accepted_coord_and_name", "accepted_names_and_zip",
                            "matched_in_brooklyn_without_borough_evidence"]
