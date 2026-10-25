"""De-duplication, timestamp handling, schema checks, partition coverage. SYNTHETIC data only."""
import datetime as dt

import pandas as pd

from roadwatch import sources
from roadwatch.normalize import dedupe, frame_from_records, normalize, parse_floating
from roadwatch.sources import SOURCES
from roadwatch.validate import common_checks, validate


def test_dedupe_removes_exact_duplicates_and_keeps_latest_conflicting_version():
    df = pd.DataFrame([
        {"unique_key": "1", "status": "Open", ":updated_at": "2026-01-01"},
        {"unique_key": "1", "status": "Open", ":updated_at": "2026-01-01"},     # exact duplicate
        {"unique_key": "2", "status": "Open", ":updated_at": "2026-01-01"},
        {"unique_key": "2", "status": "Closed", ":updated_at": "2026-02-01"},   # newer version of the same SR
        {"unique_key": "3", "status": "Open", ":updated_at": "2026-01-01"},
    ])
    out, stats = dedupe(df, "unique_key")
    assert sorted(out["unique_key"]) == ["1", "2", "3"]
    assert out.set_index("unique_key").loc["2", "status"] == "Closed"
    assert stats["exact_duplicates_removed"] == 1 and stats["ids_with_conflicting_versions"] == 1


def test_dedupe_never_merges_distinct_reports_with_identical_content():
    df = pd.DataFrame([{"unique_key": "10", "street_name": "SYNTHETIC AVENUE"},
                       {"unique_key": "11", "street_name": "SYNTHETIC AVENUE"}])
    out, stats = dedupe(df, "unique_key")
    assert len(out) == 2 and stats["exact_duplicates_removed"] == 0


def test_floating_timestamps_are_not_shifted_including_dst_edges():
    s = pd.Series(["2025-11-02T01:30:00.000",   # ambiguous local hour (DST ends)
                   "2025-03-09T02:30:00.000",   # non-existent local hour (DST starts)
                   "2025-01-01T23:59:59",
                   "2025-01-01T00:00:00Z",      # carries an offset -> must be flagged, not silently converted
                   None])
    parsed, bad = parse_floating(s)
    assert parsed[0] == pd.Timestamp("2025-11-02 01:30:00")
    assert parsed[1] == pd.Timestamp("2025-03-09 02:30:00")
    assert parsed[2] == pd.Timestamp("2025-01-01 23:59:59")
    assert pd.isna(parsed[3]) and pd.isna(parsed[4]) and bad == 1
    assert parsed.dt.tz is None


def test_frame_from_records_restores_omitted_null_fields_and_serialises_nested_values():
    recs = [{":id": "a", "unique_key": "1", "location": {"latitude": "40.6", "longitude": "-73.9"}}, {":id": "b", "unique_key": "2"}]
    df = frame_from_records(recs, ["unique_key", "location", "street_name"])
    assert "street_name" in df.columns and df["street_name"].isna().all()
    assert df.loc[0, "location"] == '{"latitude":"40.6","longitude":"-73.9"}'


def _ctx(columns, observed, state=None):
    return {"state": state or {"status": "downloaded", "partition_results": {"x": "ok"},
                                "coverage_check": {"passed": True}},
            "dataset_metadata": {"columns": [{"fieldName": c, "dataTypeName": "text"} for c in columns],
                                 "schema_fingerprint": "f"},
            "observed_columns": observed,
            "dedup": {"id_field": "unique_key", "rows_in": 1, "exact_duplicates_removed": 0,
                      "ids_with_conflicting_versions": 0, "null_ids": 0}}


def test_missing_required_column_fails_and_unexpected_column_warns():
    spec = SOURCES["sr311"]
    published = [f for f in spec.required_fields if f != "cross_street_2"]
    checks = common_checks(spec, pd.DataFrame({"unique_key": ["1"]}),
                           _ctx(published, published + ["brand_new_field"]))
    by = {c["check"]: c for c in checks}
    assert by["schema_required_fields"]["status"] == "FAIL" and "cross_street_2" in by["schema_required_fields"]["detail"]
    assert by["schema_unexpected_columns"]["status"] == "WARN"


def test_incomplete_extract_fails_validation():
    spec = SOURCES["sr311"]
    ctx = _ctx(list(spec.required_fields), [], {"status": "in_progress", "partition_results": {"2020-01": "failed"},
                                                  "coverage_check": None})
    by = {c["check"]: c for c in common_checks(spec, pd.DataFrame({"unique_key": ["1"]}), ctx)}
    assert by["extract_complete"]["status"] == "FAIL" and by["coverage_total"]["status"] == "FAIL"


def test_monthly_partitions_cover_range_without_gaps():
    spec = SOURCES["sr311"]
    parts = spec.partitioner(spec, dt.date(2020, 1, 1), dt.date(2026, 10, 24))
    assert parts[0].where.startswith("created_date >= '2020-01-01T00:00:00'")
    assert parts[-2].where.endswith("created_date < '2026-10-24T00:00:00'") and parts[-1].name == "null-date"
    bounds = [(p.where.split("'")[1], p.where.split("'")[3]) for p in parts[:-1]]
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))  # contiguous
    assert len(parts) == 82 + 1


def test_open_year_buckets_include_before_after_and_null():
    spec = SOURCES["workorders"]
    names = [p.name for p in spec.partitioner(spec, None, dt.date(2026, 10, 24))]
    assert names[0] == "pre-2010" and names[-2:] == ["2027-on", "null-date"]
    assert "rptdate >= '2015/01/01' AND rptdate < '2016/01/01'" in [p.where for p in spec.partitioner(spec, None, dt.date(2026, 10, 24))]


def test_noaa_partitions_end_inclusive_day_before_exclusive_end():
    spec = SOURCES["noaa"]
    parts = spec.partitioner(spec, dt.date(2019, 10, 1), dt.date(2026, 10, 24))
    assert parts[0].start == dt.date(2019, 10, 1) and parts[-1].end_inclusive == dt.date(2026, 10, 23)


SYNTH_NOAA_CSV = (  # SYNTHETIC weather rows for parser tests
    '"STATION","NAME","LATITUDE","LONGITUDE","ELEVATION","DATE","PRCP","PRCP_ATTRIBUTES","SNOW","SNOW_ATTRIBUTES",'
    '"SNWD","SNWD_ATTRIBUTES","TMAX","TMAX_ATTRIBUTES","TMIN","TMIN_ATTRIBUTES"\n'
    '"USW00094728","NY CITY CENTRAL PARK, NY US","40.77898","-73.96925","42.7","2026-01-01","1.5",",,W,2400","0.0",",,W,2400","0.0",",,W,2400","3.3",",,W","-2.1",",,W"\n'
    '"USW00094728","NY CITY CENTRAL PARK, NY US","40.77898","-73.96925","42.7","2026-01-02","0.0","T,,1,2400","","","0.0",",,1,2400","4.4",",X,1","-1.0",",,1"\n'
    '"USW00094728","NY CITY CENTRAL PARK, NY US","40.77898","-73.96925","42.7","2026-01-02","0.0","T,,1,2400","","","0.0",",,1,2400","4.4",",X,1","-1.0",",,1"\n'
    '"USW00094728","NY CITY CENTRAL PARK, NY US","40.77898","-73.96925","42.7","2026-01-04","0.0",",,1,2400","0.0",",,1,2400","5.0",",,1","0.5",",,1"\n')


def test_noaa_flags_split_and_duplicate_and_missing_dates_detected():
    from roadwatch.noaa import parse_csv
    spec = SOURCES["noaa"]
    raw = parse_csv(SYNTH_NOAA_CSV.encode(), spec)
    ctx = {"state": {"status": "downloaded", "partition_results": {"2026": "ok"}, "requested_start": "2026-01-01",
                     "requested_end_exclusive": "2026-01-06"}}
    table, stats = normalize(spec, raw, ctx)
    assert stats["exact_duplicates_removed"] == 1
    assert table.loc[table["DATE"] == "2026-01-02", "prcp_mflag"].item() == "T"
    assert table.loc[table["DATE"] == "2026-01-02", "tmax_qflag"].item() == "X"
    assert pd.isna(table.loc[table["DATE"] == "2026-01-02", "snow_value"].item())
    ctx["dedup"] = stats
    by = {c["check"]: c for c in validate(spec, table, ctx)}
    assert by["date_unique"]["status"] == "PASS"
    assert by["date_coverage"]["status"] == "WARN" and "2026-01-03" in by["date_coverage"]["value"]["missing_interior"]
    assert by["quality_flags"]["status"] == "WARN"


def test_pavement_zero_is_not_rated_not_poor():
    spec = SOURCES["pavement"]
    geom = {"type": "MultiLineString", "coordinates": [[[-73.95, 40.65], [-73.951, 40.651]]]}
    recs = [{":id": "s1", "oftcode": "3" * 18, "systemrating": "0", "nonratingreason": "Construction",
             "inspection": "2024-05-01T00:00:00.000", "the_geom": geom},
            {":id": "s2", "oftcode": "3" * 18, "systemrating": "6.59", "inspection": "2024-05-01T00:00:00.000", "the_geom": geom}]
    cols = ["the_geom", "oftcode", "systemrating", "nonratingreason", "inspection", "locationgeometry_stlength", "boroughname"]
    ctx = {"dataset_metadata": {"columns": [{"fieldName": c, "dataTypeName": "text"} for c in cols]}}
    gdf, _ = normalize(spec, recs, ctx)
    assert gdf["is_not_rated"].tolist() == [True, False]
    assert pd.isna(gdf.loc[0, "rating_value"]) and gdf.loc[1, "rating_value"] == 6.59
    assert gdf.loc[0, "systemrating"] == "0"  # original value preserved
