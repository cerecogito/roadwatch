"""Turn archived raw records into typed tables while preserving every original field.

Conventions
- Original source columns are kept verbatim as strings (Socrata JSON delivers every value as a string).
- Parsed columns are added with suffixes: `_local` (naive NYC wall-clock datetime from a floating timestamp),
  `_num` (float), `_date` (date-only values).
- Socrata timestamps such as `created_date` have no UTC offset ("floating"). They are parsed without any time-zone
  conversion; the wall-clock values are exactly those published.
"""
from __future__ import annotations

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .sources import NOAA_ELEMENTS, SourceSpec

FLOATING_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?$")
FT_TO_M = 0.3048


def _canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, (dict, list)) else v


def frame_from_records(records: list[dict], expected_columns: list[str]) -> pd.DataFrame:
    """Socrata omits null fields from JSON rows, so absent keys are added back as nulls for every published column."""
    df = pd.DataFrame.from_records(records)
    for c in expected_columns:
        if c not in df.columns:
            df[c] = None
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (dict, list))).any():
            df[c] = df[c].map(_canonical)
    return df


def dedupe(df: pd.DataFrame, id_col: str, updated_col: str = ":updated_at") -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows; if one id has differing contents keep the most recently updated version and count it.
    Distinct ids are never merged."""
    n_in = len(df)
    exact = df.duplicated(keep="first")
    out = df[~exact]
    dup_mask = out[id_col].duplicated(keep=False) & out[id_col].notna()
    conflicting = int(out.loc[dup_mask, id_col].nunique())
    if conflicting:
        order = out[updated_col] if updated_col in out.columns else pd.Series(range(len(out)), index=out.index)
        out = out.assign(_ord=order).sort_values("_ord", kind="stable").drop_duplicates(id_col, keep="last").drop(columns="_ord")
    stats = {"id_field": id_col, "rows_in": n_in, "exact_duplicates_removed": int(exact.sum()),
             "ids_with_conflicting_versions": conflicting, "rows_out": len(out),
             "null_ids": int(out[id_col].isna().sum())}
    return out.reset_index(drop=True), stats


def parse_floating(s: pd.Series) -> tuple[pd.Series, int]:
    """Parse offset-free ISO timestamps without shifting them. Returns (parsed, n_nonconforming)."""
    present = s.notna()
    ok = s.where(present).astype("object").map(lambda v: bool(FLOATING_TS.match(v)) if isinstance(v, str) else False)
    parsed = pd.to_datetime(s.where(ok), format="ISO8601", errors="coerce")
    return parsed, int((present & ~ok).sum())


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def geometry_from_geojson(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), None, dtype=object)
    mask = s.notna().to_numpy()
    if mask.any():
        out[mask] = shapely.from_geojson(s[mask].to_numpy(), on_invalid="ignore")
    return out


def geometry_from_wkt(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), None, dtype=object)
    mask = s.notna().to_numpy()
    if mask.any():
        out[mask] = shapely.from_wkt(s[mask].to_numpy(), on_invalid="ignore")
    return out


# ---------------- per-source normalisers ----------------
def _sr311(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    ctx["nonconforming_timestamps"] = {}
    for c in ("created_date", "closed_date", "due_date", "resolution_action_updated_date"):
        if c in df.columns:
            df[f"{c}_local"], bad = parse_floating(df[c])
            ctx["nonconforming_timestamps"][c] = bad
    for c in ("latitude", "longitude", "x_coordinate_state_plane", "y_coordinate_state_plane"):
        df[f"{c}_num"] = to_num(df[c])
    return df


def _centerline(df: pd.DataFrame, ctx: dict) -> gpd.GeoDataFrame:
    for c in ("physicalid", "rw_type", "status", "segmentlength", "shape_length", "streetwidth",
              "number_travel_lanes", "number_park_lanes", "number_total_lanes", "posted_speed"):
        if c in df.columns:
            df[f"{c}_num"] = to_num(df[c])
    df["is_eligible_block"] = (df["rw_type_num"] == 1) & (df["status_num"] == 2)
    gdf = gpd.GeoDataFrame(df.drop(columns=["the_geom"]), geometry=geometry_from_geojson(df["the_geom"]), crs="EPSG:4326")
    gdf["length_m_projected"] = gdf.geometry.to_crs("EPSG:2263").length * FT_TO_M
    return gdf


def _pavement(df: pd.DataFrame, ctx: dict) -> gpd.GeoDataFrame:
    df["systemrating_num"] = to_num(df["systemrating"])
    df["is_not_rated"] = df["systemrating_num"] == 0          # audit §4: 0 means not rated
    df["rating_value"] = df["systemrating_num"].where(df["systemrating_num"] > 0)
    df["inspection_local"], bad = parse_floating(df["inspection"])
    ctx["nonconforming_timestamps"] = {"inspection": bad}
    df["locationgeometry_stlength_ft_num"] = to_num(df["locationgeometry_stlength"])
    return gpd.GeoDataFrame(df.drop(columns=["the_geom"]), geometry=geometry_from_geojson(df["the_geom"]), crs="EPSG:4326")


def _workorders(df: pd.DataFrame, ctx: dict) -> gpd.GeoDataFrame:
    for c in ("rptdate", "rptclosed"):
        df[f"{c}_date"] = pd.to_datetime(df[c], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    df["shape_leng_ft_num"] = to_num(df.get("shape_leng"))
    return gpd.GeoDataFrame(df.drop(columns=["the_geom"]), geometry=geometry_from_geojson(df["the_geom"]), crs="EPSG:4326")


def _resurf_inhouse(df: pd.DataFrame, ctx: dict) -> gpd.GeoDataFrame:
    ctx["nonconforming_timestamps"] = {}
    for c in ("location_actual_milling_start_date", "location_actual_milling_end_date", "location_actual_paving_start_date",
              "location_actual_paving_end_date", "location_actual_protect_until"):
        if c in df.columns:
            df[f"{c}_local"], ctx["nonconforming_timestamps"][c] = parse_floating(df[c])
    return gpd.GeoDataFrame(df, geometry=geometry_from_wkt(df["location_wkt"]), crs="EPSG:2263")


def _resurf_schedule(df: pd.DataFrame, ctx: dict) -> gpd.GeoDataFrame:
    df["date_local"], bad = parse_floating(df["date"])
    ctx["nonconforming_timestamps"] = {"date": bad}
    df["is_on_or_after_retrieval_date"] = df["date_local"] >= pd.Timestamp(ctx["requested_end_exclusive"])
    return gpd.GeoDataFrame(df, geometry=geometry_from_wkt(df["location_wkt"]), crs="EPSG:2263")


def _noaa(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    out = pd.DataFrame({"station_id": df["STATION"], "station_name": df.get("NAME"),
                        "station_latitude": to_num(df.get("LATITUDE")), "station_longitude": to_num(df.get("LONGITUDE")),
                        "station_elevation_m": to_num(df.get("ELEVATION")), "DATE": df["DATE"],
                        "date": pd.to_datetime(df["DATE"], format="%Y-%m-%d")})
    for el in NOAA_ELEMENTS:
        raw = df[el] if el in df.columns else pd.Series([""] * len(df))
        out[el] = raw
        out[f"{el.lower()}_value"] = to_num(raw.replace("", np.nan))
        attrs = (df[f"{el}_ATTRIBUTES"] if f"{el}_ATTRIBUTES" in df.columns else pd.Series([""] * len(df))).str.split(",")
        part = lambda i: attrs.map(lambda a: a[i] if isinstance(a, list) and len(a) > i else "")  # noqa: E731
        out[f"{el.lower()}_mflag"] = part(0)
        out[f"{el.lower()}_qflag"] = part(1)
        out[f"{el.lower()}_sflag"] = part(2)
        out[f"{el.lower()}_attr4_unverified"] = part(3)  # 4th attribute is not described in readme.txt
    return out


NORMALIZERS = {"sr311": _sr311, "sr311_unassigned": _sr311, "centerline": _centerline, "pavement": _pavement,
               "workorders": _workorders, "resurfacing_inhouse": _resurf_inhouse,
               "resurfacing_schedule": _resurf_schedule, "noaa": _noaa}


def normalize(spec: SourceSpec, records, ctx: dict):
    """Returns (table, dedup_stats). `records` is a list of dicts (Socrata) or a DataFrame (NOAA)."""
    if spec.kind == "socrata":
        cols = [c["fieldName"] for c in ctx["dataset_metadata"]["columns"]]
        df = frame_from_records(records, cols)
        ctx["observed_columns"] = sorted(set().union(*(r.keys() for r in records))) if records else []
    else:
        df = records.copy()
        ctx["observed_columns"] = list(df.columns)
    df, stats = dedupe(df, spec.id_field)
    return NORMALIZERS[spec.key](df, ctx), stats
