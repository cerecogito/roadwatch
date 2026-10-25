"""Source specifications for the approved Brooklyn-only pipeline.

Dataset IDs, field names, filters and roles come from the data audit. Each source is
split into bounded partitions so that downloads are resumable and every partition can be count-verified.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

from .config import NCEI_DATA_ENDPOINT, NOAA_STATION, SOCRATA_DOMAIN

BOROUGHS = ("BROOKLYN", "QUEENS", "MANHATTAN", "BRONX", "STATEN ISLAND")


@dataclass(frozen=True)
class Partition:
    name: str
    where: str | None = None             # Socrata: clause ANDed with the base filter
    start: dt.date | None = None         # NOAA: inclusive
    end_inclusive: dt.date | None = None


@dataclass(frozen=True)
class SourceSpec:
    key: str
    title: str
    kind: str                            # "socrata" | "noaa"
    role: str                            # primary | reference | primary_weather | conditional | supplementary
    identifier: str                      # Socrata dataset id or NOAA station id
    endpoint: str
    base_where: str | None
    id_field: str                        # identifier used for de-duplication
    required_fields: tuple[str, ...]     # must exist in the published schema
    date_field: str | None               # field used for partitioning / observed range
    default_start: dt.date | None        # None = full history (open-ended buckets)
    partitioner: Callable[["SourceSpec", dt.date | None, dt.date], list[Partition]] = field(repr=False)
    coverage: Callable[["SourceSpec", dt.date | None, dt.date], str | None] = field(repr=False, default=lambda s, a, b: None)
    page_size: int = 10_000
    fitness: str = ""
    caveats: tuple[str, ...] = ()


# ---------------- partition builders ----------------
def _ts(d: dt.date) -> str:
    return f"'{d:%Y-%m-%d}T00:00:00'"


def monthly(spec: SourceSpec, start: dt.date | None, end: dt.date) -> list[Partition]:
    f, parts = spec.date_field, []
    cur = dt.date(start.year, start.month, 1)
    while cur < end:
        nxt = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        lo, hi = max(cur, start), min(nxt, end)
        if lo < hi:
            parts.append(Partition(f"{lo:%Y-%m}", f"{f} >= {_ts(lo)} AND {f} < {_ts(hi)}"))
        cur = nxt
    parts.append(Partition("null-date", f"{f} IS NULL"))
    return parts


def yearly_within(spec: SourceSpec, start: dt.date | None, end: dt.date) -> list[Partition]:
    f, parts = spec.date_field, []
    for y in range(start.year, end.year + 1):
        lo, hi = max(dt.date(y, 1, 1), start), min(dt.date(y + 1, 1, 1), end)
        if lo < hi:
            parts.append(Partition(f"{y}", f"{f} >= {_ts(lo)} AND {f} < {_ts(hi)}"))
    parts.append(Partition("null-date", f"{f} IS NULL"))
    return parts


def date_range_coverage(spec: SourceSpec, start: dt.date | None, end: dt.date) -> str:
    f = spec.date_field
    return f"(({f} >= {_ts(start)} AND {f} < {_ts(end)}) OR {f} IS NULL)"


def open_year_buckets(first_year: int, extra_years: int = 0, text_dates: bool = False):
    """Full-history partitioning: pre-<first>, one bucket per year, <last+1>-on, null. Covers every row."""
    def lit(y: int) -> str:
        return f"'{y}/01/01'" if text_dates else f"'{y}-01-01T00:00:00'"

    def build(spec: SourceSpec, start: dt.date | None, end: dt.date) -> list[Partition]:
        f, last = spec.date_field, end.year + extra_years
        parts = [Partition(f"pre-{first_year}", f"{f} < {lit(first_year)}")]
        parts += [Partition(f"{y}", f"{f} >= {lit(y)} AND {f} < {lit(y + 1)}") for y in range(first_year, last + 1)]
        parts += [Partition(f"{last + 1}-on", f"{f} >= {lit(last + 1)}"), Partition("null-date", f"{f} IS NULL")]
        return parts
    return build


def single(spec: SourceSpec, start: dt.date | None, end: dt.date) -> list[Partition]:
    return [Partition("all", "1=1")]


def noaa_years(spec: SourceSpec, start: dt.date | None, end: dt.date) -> list[Partition]:
    parts, last = [], end - dt.timedelta(days=1)
    for y in range(start.year, last.year + 1):
        lo, hi = max(dt.date(y, 1, 1), start), min(dt.date(y, 12, 31), last)
        if lo <= hi:
            parts.append(Partition(f"{y}", start=lo, end_inclusive=hi))
    return parts


# ---------------- specifications ----------------
SR311_FIELDS = ("unique_key", "created_date", "closed_date", "agency", "complaint_type", "descriptor", "descriptor_2",
                "location_type", "incident_zip", "incident_address", "street_name", "cross_street_1", "cross_street_2",
                "intersection_street_1", "intersection_street_2", "address_type", "city", "landmark", "status",
                "resolution_description", "resolution_action_updated_date", "community_board", "bbl", "borough",
                "x_coordinate_state_plane", "y_coordinate_state_plane", "latitude", "longitude", "location",
                "open_data_channel_type", "bridge_highway_name", "road_ramp")

SOURCES: dict[str, SourceSpec] = {}


def _add(spec: SourceSpec) -> None:
    SOURCES[spec.key] = spec


_add(SourceSpec(
    key="sr311", title="NYC 311 Service Requests - Street Condition, Brooklyn", kind="socrata", role="primary",
    identifier="erm2-nwe9", endpoint=f"{SOCRATA_DOMAIN}/resource/erm2-nwe9.json",
    base_where="complaint_type='Street Condition' AND borough='BROOKLYN'", id_field="unique_key",
    required_fields=SR311_FIELDS, date_field="created_date", default_start=dt.date(2020, 1, 1),
    partitioner=monthly, coverage=date_range_coverage,
    fitness="Primary label source (public pothole REPORTS, not confirmed defects).",
    caveats=("Reports are not verified physical failures (audit §2.6).",
             "created_date is a floating timestamp (no offset); interpreted as NYC local wall-clock time.",
             "BLOCKFACE reports have no coordinates since 2022; location text fields must be kept.",
             "The most recent ~2 days are incomplete because of the ~1-day publication lag.",
             "2020 is retrieved as look-back history for 2021 forecast origins (technical plan §2)."),
))
_add(SourceSpec(
    key="sr311_unassigned", title="NYC 311 Street Condition with missing/'Unspecified' borough (citywide)",
    kind="socrata", role="supplementary", identifier="erm2-nwe9", endpoint=f"{SOCRATA_DOMAIN}/resource/erm2-nwe9.json",
    base_where="complaint_type='Street Condition' AND (borough IS NULL OR borough NOT IN "
               "('" + "','".join(BOROUGHS) + "'))",
    id_field="unique_key", required_fields=SR311_FIELDS, date_field="created_date",
    default_start=dt.date(2020, 1, 1), partitioner=yearly_within, coverage=date_range_coverage,
    fitness="Held so Brooklyn reports with a missing borough are not silently dropped; borough assignment is a later-stage task.",
    caveats=("Contains records from all boroughs; NOT part of the Brooklyn table until assigned by location.",),
))
_add(SourceSpec(
    key="centerline", title="NYC Street Centerline (CSCL), Brooklyn", kind="socrata", role="reference",
    identifier="inkn-q76z", endpoint=f"{SOCRATA_DOMAIN}/resource/inkn-q76z.json", base_where="boroughcode='3'",
    id_field=":id", required_fields=("the_geom", "physicalid", "b5sc", "rw_type", "status", "boroughcode",
                                     "full_street_name", "segmentlength", "streetwidth", "number_travel_lanes",
                                     "trafdir", "l_blockfaceid", "r_blockfaceid", "globalid"),
    date_field=None, default_start=None, partitioner=single,
    fitness="Geographic reference: block universe (rw_type=1, status=2) and geometry.",
    caveats=("Current snapshot; attributes may post-date historical forecast origins.",
             "SHAPE__Length / shape_length is in Web-Mercator units; compute length from projected geometry.",
             "No LION SegmentID in this dataset; joins to other sources are spatial or OFT-based (later stage)."),
))
_add(SourceSpec(
    key="pavement", title="Street Pavement Ratings, Brooklyn", kind="socrata", role="conditional",
    identifier="6yyb-pb25", endpoint=f"{SOCRATA_DOMAIN}/resource/6yyb-pb25.json", base_where="boroughname='Brooklyn'",
    id_field=":id", required_fields=("the_geom", "oftcode", "onstreetna", "fromstreet", "tostreetna", "systemrating",
                                     "nonratingreason", "inspection", "locationgeometry_stlength"),
    date_field="inspection", default_start=None, partitioner=open_year_buckets(2018),
    fitness="Conditional/optional feature source; core pipeline must run without it.",
    caveats=("systemrating 0 means NOT RATED (always paired with a NonRatingReason), not 'very poor'.",
             "Publication date of individual ratings is unverified; 45-day availability lag is an ASSUMPTION.",
             "Dataset rebuilt 2025-11-10 (fields removed/added); Socrata :created_at reflects bulk reloads, not rating publication.",
             "nonratingreason is omitted from rows where it is null (Socrata JSON omits nulls)."),
))
_add(SourceSpec(
    key="workorders", title="Street Pothole Work Orders - Closed, Brooklyn", kind="socrata", role="conditional",
    identifier="x9wy-ing4", endpoint=f"{SOCRATA_DOMAIN}/resource/x9wy-ing4.json", base_where="boro='B'",
    id_field=":id", required_fields=("the_geom", "defnum", "initby", "oft", "onprimname", "frmprimnam", "toprimname",
                                     "boro", "source", "rptdate", "rptclosed"),
    date_field="rptdate", default_start=None, partitioner=open_year_buckets(2010, text_dates=True),
    fitness="Supplementary: repair-history feature and secondary evaluation labels only.",
    caveats=("Closed work orders only; open orders are invisible.",
             "Does not cover all defects or all repair crews; a missing work order is not proof of no repair.",
             "rptdate/rptclosed are TEXT 'YYYY/MM/DD 00:00:00' (date-only).",
             "35-day availability lag is an ASSUMPTION (monthly manual publication)."),
))
_add(SourceSpec(
    key="resurfacing_inhouse", title="DOT In-house Street Resurfacing Projects, Brooklyn", kind="socrata",
    role="conditional", identifier="ffaf-8mrv", endpoint=f"{SOCRATA_DOMAIN}/resource/ffaf-8mrv.json",
    base_where="borough_code='B'", id_field=":id",
    required_fields=("oft_code", "project_type", "borough_code", "location_on_street", "location_from_street",
                     "location_to_street", "project_id", "project_status", "location_actual_paving_end_date",
                     "location_actual_milling_end_date", "location_status", "location_segment_id", "location_wkt"),
    date_field="location_actual_paving_end_date", default_start=None, partitioner=open_year_buckets(2010),
    fitness="Supplementary: days-since-paving feature (in-house crews only).",
    caveats=("In-house crews only; contract paving is missing.",
             "21-day availability lag is an ASSUMPTION.",
             "location_wkt is in NY State Plane feet per the dataset dictionary."),
))
_add(SourceSpec(
    key="resurfacing_schedule", title="Street Resurfacing Schedule, Brooklyn", kind="socrata", role="conditional",
    identifier="xnfm-u3k5", endpoint=f"{SOCRATA_DOMAIN}/resource/xnfm-u3k5.json", base_where="boroughname='BROOKLYN'",
    id_field=":id", required_fields=("workscheduleprojectlocationid", "date", "onstreetname", "fromstreetname",
                                     "tostreetname", "worktype", "crewtype", "oftcode", "location_segment_id",
                                     "location_wkt"),
    date_field="date", default_start=None, partitioner=open_year_buckets(2020, extra_years=1),
    fitness="Supplementary: past-dated planned work only (schedules are subject to change).",
    caveats=("Planned, not confirmed, work; future-dated rows must never be used in backtests.",
             "Historical publication snapshots are unavailable.",
             "location_wkt CRS is not stated in this dataset's metadata; coordinate magnitudes match NY State Plane feet."),
))
_add(SourceSpec(
    key="noaa", title="NOAA GHCN-Daily, Central Park (USW00094728)", kind="noaa", role="primary_weather",
    identifier=NOAA_STATION["id"], endpoint=NCEI_DATA_ENDPOINT, base_where=None, id_field="DATE",
    required_fields=("STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION", "DATE", "PRCP", "PRCP_ATTRIBUTES",
                     "SNOW", "SNOW_ATTRIBUTES", "SNWD", "SNWD_ATTRIBUTES", "TMAX", "TMAX_ATTRIBUTES",
                     "TMIN", "TMIN_ATTRIBUTES"),
    date_field="DATE", default_start=dt.date(2019, 10, 1), partitioner=noaa_years,
    fitness="Primary weather source: one citywide daily series (not street-level).",
    caveats=("Single station in Manhattan; not Brooklyn street-level weather.",
             "Daily TMAX/TMIN only: any freeze-thaw-DAY measure derived later is an approximation, not hourly crossings.",
             "Recent values carry source flag '1' (NWS CF6) while older values carry 'W' (ISD); recent values may be revised.",
             "~4-day reporting lag observed on 2026-10-24."),
))

NOAA_ELEMENTS = ("PRCP", "SNOW", "SNWD", "TMAX", "TMIN")
NOAA_UNITS = {"PRCP": "mm", "SNOW": "mm", "SNWD": "mm", "TMAX": "degC", "TMIN": "degC"}
