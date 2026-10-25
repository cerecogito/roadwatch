"""Source-specific validation. Every check returns PASS / WARN / FAIL / INFO with a human-readable detail.

FAIL blocks promotion of the processed table; WARN is reported but does not block; INFO is a diagnostic.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import shapely

from .config import AUDIT_REFERENCE, NOAA_STATION, NYC_LATLON_BOUNDS, NYC_STATE_PLANE_BOUNDS
from .sources import BOROUGHS, NOAA_ELEMENTS, NOAA_UNITS, SourceSpec

SYSTEM_FIELDS = {":id", ":created_at", ":updated_at", ":version"}


def check(name: str, status: str, detail: str, value=None) -> dict:
    return {"check": name, "status": status, "detail": detail, "value": value}


def compare_to_reference(name: str, observed: int, reference: int, warn_rel: float = 0.01, fail_rel: float = 0.10,
                         what: str = "rows") -> dict:
    rel = abs(observed - reference) / reference if reference else float("inf")
    status = "PASS" if rel <= warn_rel else "WARN" if rel <= fail_rel else "FAIL"
    return check(name, status, f"{what}: observed {observed:,} vs audit reference {reference:,} ({rel:.2%} difference)",
                 {"observed": observed, "reference": reference})


def summarize(checks: list[dict]) -> dict:
    counts = {s: sum(c["status"] == s for c in checks) for s in ("PASS", "WARN", "FAIL", "INFO")}
    status = "fail" if counts["FAIL"] else "warn" if counts["WARN"] else "pass"
    return {"status": status, "counts": counts}


def _missingness(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if c != "geometry"]
    return {c: round(float(df[c].isna().mean()), 4) for c in cols}


# ---------------- common checks ----------------
def common_checks(spec: SourceSpec, df: pd.DataFrame, ctx: dict) -> list[dict]:
    out = []
    state = ctx["state"]
    bad_parts = [k for k, v in state.get("partition_results", {}).items() if v != "ok"]
    out.append(check("extract_complete", "PASS" if state["status"] in ("downloaded", "finalized") and not bad_parts else "FAIL",
                     f"snapshot status={state['status']}; partitions not ok: {bad_parts or 'none'}"))
    if spec.kind == "socrata":
        cov = state.get("coverage_check") or {}
        out.append(check("coverage_total", "PASS" if cov.get("passed") else "FAIL",
                         f"server total {cov.get('server_total_at_check')} vs sum of verified partitions "
                         f"{cov.get('sum_of_partitions')} under: {cov.get('where')}", cov))
        published = {c["fieldName"] for c in ctx["dataset_metadata"]["columns"]}
        missing = [f for f in spec.required_fields if f not in published]
        out.append(check("schema_required_fields", "FAIL" if missing else "PASS",
                         f"required fields missing from published schema: {missing or 'none'}"))
        observed = set(ctx.get("observed_columns", []))
        unexpected = sorted(c for c in observed - published if not c.startswith(":"))
        out.append(check("schema_unexpected_columns", "WARN" if unexpected else "PASS",
                         f"row keys not in published schema: {unexpected or 'none'}"))
        system = sorted(c for c in observed if c.startswith(":") and c not in SYSTEM_FIELDS)
        if system:
            out.append(check("socrata_computed_columns", "INFO",
                             "portal-generated ':@computed_region_*' columns returned by $select=:* (kept, not used)", system))
        fp, prev = ctx["dataset_metadata"]["schema_fingerprint"], ctx.get("previous_schema_fingerprint")
        out.append(check("schema_fingerprint", "WARN" if prev and prev != fp else "INFO",
                         f"fingerprint {fp}" + (f" (previous finalized snapshot: {prev})" if prev else " (no previous snapshot)"), fp))
    else:
        missing = [f for f in spec.required_fields if f not in ctx.get("observed_columns", [])]
        out.append(check("schema_required_fields", "FAIL" if missing else "PASS", f"missing columns: {missing or 'none'}"))
    out.append(check("nonempty", "PASS" if len(df) else "FAIL", f"{len(df):,} rows after de-duplication"))
    d = ctx["dedup"]
    ok = d["ids_with_conflicting_versions"] == 0 and d["null_ids"] == 0 and df[spec.id_field].is_unique
    out.append(check("id_uniqueness", "PASS" if ok else "WARN",
                     f"id={d['id_field']}: rows in {d['rows_in']:,}, exact duplicates removed {d['exact_duplicates_removed']}, "
                     f"ids with conflicting versions {d['ids_with_conflicting_versions']}, null ids {d['null_ids']}", d))
    bad_ts = {k: v for k, v in (ctx.get("nonconforming_timestamps") or {}).items() if v}
    if ctx.get("nonconforming_timestamps") is not None:
        out.append(check("timestamps_floating_format", "FAIL" if bad_ts else "PASS",
                         "timestamps that are not offset-free ISO values (would indicate a time-zone/schema change): "
                         f"{bad_ts or 'none'}"))
    out.append(check("missingness", "INFO", "share of nulls per column", _missingness(df)))
    return out


def geometry_checks(gdf, bounds_latlon: bool = True) -> list[dict]:
    geom = gdf.geometry
    missing = int(geom.isna().sum())
    present = geom[geom.notna()]
    arr = present.to_numpy()
    empty = int(shapely.is_empty(arr).sum()) if len(arr) else 0
    invalid = int((~shapely.is_valid(arr)).sum()) if len(arr) else 0
    types = present.geom_type.value_counts().to_dict()
    out = [check("geometry_present", "PASS" if missing == 0 else "WARN", f"{missing} rows without geometry", missing),
           check("geometry_nonempty_valid", "PASS" if empty == 0 and invalid == 0 else "WARN",
                 f"empty {empty}, invalid (OGC) {invalid}, types {types}", {"empty": empty, "invalid": invalid, "types": types}),
           check("geometry_crs", "INFO", f"CRS {gdf.crs.to_string() if gdf.crs else None}")]
    if len(present):
        minx, miny, maxx, maxy = present.total_bounds
        if bounds_latlon:
            (la0, la1), (lo0, lo1) = NYC_LATLON_BOUNDS["lat"], NYC_LATLON_BOUNDS["lon"]
            ok = la0 <= miny and maxy <= la1 and lo0 <= minx and maxx <= lo1
        else:
            (x0, x1), (y0, y1) = NYC_STATE_PLANE_BOUNDS["x"], NYC_STATE_PLANE_BOUNDS["y"]
            ok = x0 <= minx and maxx <= x1 and y0 <= miny and maxy <= y1
        out.append(check("geometry_within_nyc", "PASS" if ok else "WARN",
                         f"total bounds {[round(v, 5) for v in (minx, miny, maxx, maxy)]}"))
    return out


def _year_counts(s: pd.Series) -> dict:
    return {int(k): int(v) for k, v in s.dt.year.value_counts().sort_index().items() if not pd.isna(k)}


def _range(s: pd.Series) -> dict:
    s = s.dropna()
    return {"min": str(s.min()) if len(s) else None, "max": str(s.max()) if len(s) else None}


# ---------------- source-specific ----------------
def _sr311(spec: SourceSpec, df: pd.DataFrame, ctx: dict) -> list[dict]:
    out = []
    state = ctx["state"]
    start, end = pd.Timestamp(state["requested_start"]), pd.Timestamp(state["requested_end_exclusive"])
    ct_ok = (df["complaint_type"] == "Street Condition").all()
    if spec.key == "sr311":
        b_ok = (df["borough"] == "BROOKLYN").all()
    else:
        b_ok = (~df["borough"].isin(BOROUGHS)).all()
    out.append(check("scope_filter", "PASS" if ct_ok and b_ok else "FAIL",
                     f"complaint_type all 'Street Condition': {ct_ok}; borough constraint satisfied: {b_ok}"))
    c = df["created_date_local"]
    outside = int(((c < start) | (c >= end)).sum())
    out.append(check("created_in_requested_range", "PASS" if outside == 0 and c.notna().all() else "FAIL",
                     f"{outside} rows outside [{start.date()}, {end.date()}); null created_date {int(c.isna().sum())}"))
    out.append(check("observed_created_range", "INFO", "observed created_date range (local wall-clock)", _range(c)))
    pot = df[df["descriptor"] == "Pothole"]
    if spec.key == "sr311":
        ref = AUDIT_REFERENCE["sr311_bk_pothole_by_year"]
        obs = _year_counts(pot["created_date_local"])
        for y, r in ref.items():
            if start <= pd.Timestamp(y, 1, 1) and pd.Timestamp(y + 1, 1, 1) <= end:
                out.append(compare_to_reference(f"pothole_count_{y}_vs_audit", obs.get(y, 0), r, 0.01, 0.05, "Pothole SRs"))
        if state["requested_start"] == "2020-01-01" and state["requested_end_exclusive"] == "2026-10-24":
            out.append(compare_to_reference("total_vs_stage2_probe", len(df),
                                            AUDIT_REFERENCE["sr311_bk_street_condition_2020_to_20261024"], 0.01, 0.05))
    else:
        if state["requested_start"] == "2020-01-01" and state["requested_end_exclusive"] == "2026-10-24":
            out.append(compare_to_reference("total_vs_stage2_probe", len(df),
                                            AUDIT_REFERENCE["sr311_unassigned_borough_street_condition_since_2020"], 0.02, 0.10))
        out.append(check("unassigned_borough_values", "INFO", "borough values",
                         df["borough"].fillna("<null>").value_counts().to_dict()))
    out.append(check("descriptor_counts", "INFO", "rows per descriptor", df["descriptor"].fillna("<null>").value_counts().to_dict()))
    lat, lon = df["latitude_num"], df["longitude_num"]
    has = lat.notna() & lon.notna()
    (la0, la1), (lo0, lo1) = NYC_LATLON_BOUNDS["lat"], NYC_LATLON_BOUNDS["lon"]
    bad_ll = int((has & ~(lat.between(la0, la1) & lon.between(lo0, lo1))).sum())
    x, y = df["x_coordinate_state_plane_num"], df["y_coordinate_state_plane_num"]
    hasxy = x.notna() & y.notna()
    (x0, x1), (y0, y1) = NYC_STATE_PLANE_BOUNDS["x"], NYC_STATE_PLANE_BOUNDS["y"]
    bad_xy = int((hasxy & ~(x.between(x0, x1) & y.between(y0, y1))).sum())
    out.append(check("coordinates_valid", "PASS" if bad_ll == 0 and bad_xy == 0 else "WARN",
                     f"lat/lon present {int(has.sum()):,} (outside NYC: {bad_ll}); state-plane XY present {int(hasxy.sum()):,} "
                     f"(outside NYC: {bad_xy}); lat/lon present without XY: {int((has & ~hasxy).sum())}"))
    miss = (pot.assign(y=pot["created_date_local"].dt.year, no_coords=pot["latitude_num"].isna())
               .groupby(["y", "address_type"], dropna=False)["no_coords"].agg(["size", "sum"]))
    out.append(check("pothole_missing_coords_by_year_address_type", "INFO",
                     "Pothole SRs: (rows, rows without lat/lon) by year and address_type",
                     {f"{int(k[0])}|{k[1]}": [int(v["size"]), int(v["sum"])] for k, v in miss.iterrows()}))
    loc_text = pot["street_name"].notna() | pot["intersection_street_1"].notna()
    out.append(check("pothole_location_text_or_coords", "PASS" if (loc_text | pot["latitude_num"].notna()).all() else "WARN",
                     f"Pothole SRs with neither street text nor coordinates: {int((~loc_text & pot['latitude_num'].isna()).sum())}"))
    closed_before = int((df["closed_date_local"] < df["created_date_local"]).sum())
    out.append(check("closed_before_created", "PASS" if closed_before == 0 else "WARN", f"{closed_before} rows", closed_before))
    hours = c.dt.hour.value_counts(normalize=True).sort_index()
    out.append(check("created_hour_of_day_profile", "INFO",
                     "share of SRs by hour of created_date (daytime peak is consistent with local wall-clock time)",
                     {int(h): round(float(v), 4) for h, v in hours.items()}))
    last = c.dt.normalize().value_counts().sort_index().tail(10)
    out.append(check("recent_daily_counts", "INFO", "rows per day for the last 10 days present (publication lag evidence)",
                     {str(k.date()): int(v) for k, v in last.items()}))
    out.append(check("channel_type_values", "INFO", "open_data_channel_type values (audit: UNKNOWN for recent potholes)",
                     df["open_data_channel_type"].fillna("<null>").value_counts().to_dict()))
    return out


def _centerline(spec, gdf, ctx):
    out = [compare_to_reference("row_count_vs_audit", len(gdf), AUDIT_REFERENCE["centerline_bk_rows"], 0.0, 0.02),
           check("scope_filter", "PASS" if (gdf["boroughcode"] == "3").all() else "FAIL", "all boroughcode == '3'")]
    out += geometry_checks(gdf)
    elig = gdf[gdf["is_eligible_block"]]
    out.append(compare_to_reference("eligible_blocks_vs_audit", len(elig), AUDIT_REFERENCE["centerline_bk_eligible_blocks"],
                                    0.0, 0.02, "constructed street blocks (rw_type=1, status=2)"))
    out.append(check("physicalid_unique_among_eligible", "PASS" if elig["physicalid"].is_unique else "WARN",
                     f"{int(elig['physicalid'].duplicated().sum())} duplicated physicalid among eligible blocks; "
                     f"{int(elig['physicalid'].isna().sum())} null"))
    out.append(check("globalid_unique", "PASS" if gdf["globalid"].is_unique else "WARN",
                     f"{int(gdf['globalid'].duplicated().sum())} duplicated globalid"))
    q = elig["length_m_projected"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1)
    out.append(check("eligible_block_length_m", "INFO", "projected (EPSG:2263) length quantiles, metres",
                     {f"p{int(k * 100)}": float(v) for k, v in q.items()} | {"total_km": round(float(elig['length_m_projected'].sum()) / 1000, 1)}))
    ratio = (elig["segmentlength_num"] / (elig["length_m_projected"] / 0.3048)).median()
    out.append(check("segmentlength_is_feet", "PASS" if abs(ratio - 1) < 0.01 else "WARN",
                     f"median segmentlength / projected length in feet = {ratio:.4f}"))
    out.append(check("rw_type_status_counts", "INFO", "rw_type x status",
                     gdf.groupby(["rw_type", "status"]).size().rename(lambda k: f"{k}").to_dict()))
    return out


def _pavement(spec, gdf, ctx):
    r = gdf["systemrating_num"]
    out = [compare_to_reference("row_count_vs_audit", len(gdf), AUDIT_REFERENCE["pavement_bk_rows"], 0.0, 0.05),
           check("scope_filter", "PASS" if (gdf["boroughname"] == "Brooklyn").all() else "FAIL", "all boroughname == 'Brooklyn'"),
           check("rating_range", "PASS" if r.dropna().between(0, 10).all() else "FAIL",
                 f"systemrating range {r.min()}..{r.max()}; unparseable/null {int(r.isna().sum())}")]
    zero = gdf["is_not_rated"]
    with_reason = gdf["nonratingreason"].notna()
    out.append(check("zero_means_not_rated", "PASS" if (with_reason[zero]).all() else "WARN",
                     f"rating==0 rows {int(zero.sum()):,}; of which with NonRatingReason {int((zero & with_reason).sum()):,}; "
                     f"rating>0 rows that also carry a NonRatingReason {int((~zero & with_reason & r.notna()).sum())}"))
    out.append(check("inspection_years", "INFO", "rows per inspection year", _year_counts(gdf["inspection_local"])))
    out.append(check("inspection_date_only", "PASS" if (gdf["inspection_local"].dropna().dt.normalize() == gdf["inspection_local"].dropna()).all() else "WARN",
                     "inspection values carry no time-of-day"))
    lens = gdf["oftcode"].str.len().value_counts().to_dict()
    out.append(check("oftcode_length", "PASS" if set(lens) <= {18} else "WARN", f"oftcode lengths {lens}"))
    out += geometry_checks(gdf)
    out.append(check("publication_date_unverified", "INFO",
                     "Socrata :created_at dates (row load dates, NOT rating publication dates)",
                     gdf[":created_at"].str[:10].value_counts().head(5).to_dict()))
    return out


def _workorders(spec, gdf, ctx):
    out = [compare_to_reference("row_count_vs_audit", len(gdf), AUDIT_REFERENCE["workorders_bk_rows"], 0.01, 0.05),
           check("scope_filter", "PASS" if (gdf["boro"] == "B").all() else "FAIL", "all boro == 'B'")]
    bad_r = int((gdf["rptdate"].notna() & gdf["rptdate_date"].isna()).sum())
    bad_c = int((gdf["rptclosed"].notna() & gdf["rptclosed_date"].isna()).sum())
    out.append(check("text_dates_parse", "PASS" if bad_r == bad_c == 0 else "WARN",
                     f"unparseable rptdate {bad_r}, rptclosed {bad_c} (format 'YYYY/MM/DD HH:MM:SS')"))
    neg = int((gdf["rptclosed_date"] < gdf["rptdate_date"]).sum())
    out.append(check("closed_before_reported", "PASS" if neg == 0 else "WARN", f"{neg} rows with rptclosed < rptdate"))
    out.append(check("defnum_unique", "PASS" if gdf["defnum"].is_unique else "WARN",
                     f"{int(gdf['defnum'].duplicated().sum())} duplicated defnum"))
    out.append(check("source_counts", "INFO", "source codes (CTZ=Citizen, YRD=Yard per metadata PDF)",
                     gdf["source"].fillna("<null>").value_counts().to_dict()))
    out.append(check("rptdate_years", "INFO", "rows per report year", _year_counts(gdf["rptdate_date"])))
    out.append(check("latest_dates", "INFO", "latest rptdate / rptclosed (publication delay evidence)",
                     {"rptdate": str(gdf["rptdate_date"].max()), "rptclosed": str(gdf["rptclosed_date"].max())}))
    out += geometry_checks(gdf)
    return out


def _resurf_inhouse(spec, gdf, ctx):
    pe = gdf["location_actual_paving_end_date_local"]
    out = [compare_to_reference("row_count_vs_audit", len(gdf), AUDIT_REFERENCE["resurfacing_inhouse_bk_rows"], 0.01, 0.05),
           check("scope_filter", "PASS" if (gdf["borough_code"] == "B").all() else "FAIL", "all borough_code == 'B'"),
           check("paving_end_years", "INFO", "rows per paving end year (null = not paved / unknown)",
                 _year_counts(pe) | {"null": int(pe.isna().sum())}),
           check("location_status_counts", "INFO", "location_status", gdf["location_status"].fillna("<null>").value_counts().to_dict()),
           check("project_type_counts", "INFO", "project_type", gdf["project_type"].fillna("<null>").value_counts().to_dict())]
    wkt_present = gdf["location_wkt"].notna()
    unparsed = int((wkt_present & gdf.geometry.isna()).sum())
    out.append(check("wkt_parse", "PASS" if unparsed == 0 else "WARN", f"{unparsed} unparseable location_wkt of {int(wkt_present.sum()):,}"))
    out += geometry_checks(gdf, bounds_latlon=False)
    return out


def _resurf_schedule(spec, gdf, ctx):
    d = gdf["date_local"]
    out = [compare_to_reference("row_count_vs_audit", len(gdf), AUDIT_REFERENCE["resurfacing_schedule_bk_rows"], 0.01, 0.05),
           check("scope_filter", "PASS" if (gdf["boroughname"] == "BROOKLYN").all() else "FAIL", "all boroughname == 'BROOKLYN'"),
           check("date_years", "INFO", "rows per scheduled year", _year_counts(d)),
           check("future_dated_rows", "INFO", "rows dated on/after the retrieval date (never usable in backtests)",
                 int(gdf["is_on_or_after_retrieval_date"].sum())),
           check("crewtype_worktype", "INFO", "crewtype x worktype",
                 gdf.groupby(["crewtype", "worktype"], dropna=False).size().rename(lambda k: f"{k}").to_dict())]
    wkt_present = gdf["location_wkt"].notna()
    unparsed = int((wkt_present & gdf.geometry.isna()).sum())
    out.append(check("wkt_parse", "PASS" if unparsed == 0 else "WARN", f"{unparsed} unparseable location_wkt of {int(wkt_present.sum()):,}"))
    out += geometry_checks(gdf, bounds_latlon=False)
    return out


def _noaa(spec, df, ctx):
    out = []
    st = df[["station_id", "station_name", "station_latitude", "station_longitude", "station_elevation_m"]].drop_duplicates()
    ids_ok = list(st["station_id"].unique()) == [NOAA_STATION["id"]]
    meta_ok = len(st) == 1 and st.iloc[0]["station_name"] == NOAA_STATION["name"] and \
        abs(st.iloc[0]["station_latitude"] - NOAA_STATION["latitude"]) < 1e-4 and \
        abs(st.iloc[0]["station_longitude"] - NOAA_STATION["longitude"]) < 1e-4
    out.append(check("station_identity", "PASS" if ids_ok else "FAIL", f"station ids {list(st['station_id'].unique())}"))
    out.append(check("station_metadata", "PASS" if meta_ok else "WARN", "station name/location as returned by NCEI",
                     st.astype(str).to_dict("records")))
    out.append(check("date_unique", "PASS" if df["DATE"].is_unique else "FAIL", f"{int(df['DATE'].duplicated().sum())} duplicate dates"))
    state = ctx["state"]
    start = dt.date.fromisoformat(state["requested_start"])
    last = dt.date.fromisoformat(state["requested_end_exclusive"]) - dt.timedelta(days=1)
    all_days = pd.date_range(start, last, freq="D")
    missing = all_days.difference(df["date"])
    max_date = df["date"].max().date()
    trailing = (last - max_date).days
    interior = [d for d in missing if d.date() < max_date]
    out.append(check("date_coverage", "PASS" if not interior else "WARN",
                     f"requested {len(all_days)} days {start}..{last}; present {df['DATE'].nunique()}; "
                     f"missing interior days {len(interior)}; trailing days not yet published {trailing}",
                     {"missing_interior": [str(d.date()) for d in interior][:50], "latest_date": str(max_date)}))
    out.append(check("reporting_lag_days", "INFO", f"latest observation {max_date}; retrieval date "
                     f"{state['requested_end_exclusive']} -> lag {(last - max_date).days + 1} days", (last - max_date).days + 1))
    ranges = {"tmax_value": (-35, 45), "tmin_value": (-35, 45), "prcp_value": (0, 400), "snow_value": (0, 1500), "snwd_value": (0, 2000)}
    bad = {k: int((~df[k].dropna().between(*v)).sum()) for k, v in ranges.items()}
    inverted = int((df["tmin_value"] > df["tmax_value"]).sum())
    out.append(check("value_ranges", "PASS" if not any(bad.values()) and inverted == 0 else "WARN",
                     f"out-of-range counts {bad}; TMIN > TMAX on {inverted} days"))
    miss = {el: int(df[f"{el.lower()}_value"].isna().sum()) for el in NOAA_ELEMENTS}
    core_bad = any(miss[e] > 0.01 * len(df) for e in ("TMAX", "TMIN", "PRCP"))
    out.append(check("element_missingness", "WARN" if core_bad else "PASS", f"missing values per element over {len(df)} days: {miss}", miss))
    qf = {el: df[f"{el.lower()}_qflag"].replace("", np.nan).dropna().value_counts().to_dict() for el in NOAA_ELEMENTS}
    out.append(check("quality_flags", "WARN" if any(qf.values()) else "PASS",
                     f"non-blank GHCN-D QFLAG values (failed NOAA QC) per element: {qf}", qf))
    sf = df.assign(y=df["date"].dt.year).groupby(["y", "tmax_sflag"]).size()
    out.append(check("source_flags_by_year", "INFO", "TMAX source flag by year (W=ISD WBAN/ASOS, 1=NWS CF6 per readme.txt)",
                     {f"{k[0]}|{k[1]}": int(v) for k, v in sf.items()}))
    trace = {el: int((df[f"{el.lower()}_mflag"] == "T").sum()) for el in ("PRCP", "SNOW")}
    out.append(check("trace_flags", "INFO", "MFLAG 'T' counts (trace, per readme.txt)", trace))
    out.append(check("units", "INFO", "units as requested from NCEI (units=metric)", NOAA_UNITS))
    return out


SPECIFIC = {"sr311": _sr311, "sr311_unassigned": _sr311, "centerline": _centerline, "pavement": _pavement,
            "workorders": _workorders, "resurfacing_inhouse": _resurf_inhouse, "resurfacing_schedule": _resurf_schedule,
            "noaa": _noaa}


def validate(spec: SourceSpec, table: pd.DataFrame, ctx: dict) -> list[dict]:
    checks = common_checks(spec, table, ctx)
    if len(table):
        checks += SPECIFIC[spec.key](spec, table, ctx)
    return checks
