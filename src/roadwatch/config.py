"""Paths, endpoints and audit-derived reference values.

Reference counts below were measured during the Stage-1 audit and the Stage-2 pre-implementation probes. They are used only as plausibility checks, never as data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NYC_TZ = "America/New_York"

SOCRATA_DOMAIN = "https://data.cityofnewyork.us"
NCEI_DATA_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1"

# NYC bounding boxes used for coordinate validity checks.
NYC_LATLON_BOUNDS = {"lat": (40.47, 40.93), "lon": (-74.27, -73.68)}
NYC_STATE_PLANE_BOUNDS = {"x": (909_000, 1_068_000), "y": (110_000, 275_000)}  # EPSG:2263, US ft

# Availability lags (days) to be enforced in later stages. Only the 311 and NOAA lags were observed.
AVAILABILITY_LAGS = {
    "sr311": {"days": 2, "status": "observed (~1 day, audit §2.5); 2-day embargo chosen"},
    "noaa": {"days": 5, "status": "observed (~4 days on 2026-10-24); 5-day embargo chosen"},
    "pavement": {"days": 45, "status": "ASSUMPTION - publication date of individual ratings is unverified"},
    "workorders": {"days": 35, "status": "ASSUMPTION - monthly manual publication; not verified per record"},
    "resurfacing_inhouse": {"days": 21, "status": "ASSUMPTION - stated 2-week publishing cadence; not verified per record"},
    "resurfacing_schedule": {"days": None, "status": "Planned dates; past-dated entries only, treated as noisy"},
    "centerline": {"days": None, "status": "Static current snapshot (anachronism flagged)"},
}

AUDIT_REFERENCE = {
    # erm2-nwe9, borough='BROOKLYN', descriptor='Pothole', counts by created year
    "sr311_bk_pothole_by_year": {2020: 8360, 2021: 11129, 2022: 11836, 2023: 9098, 2024: 8528, 2025: 8542},
    # Stage-2 probes on 2026-10-24
    "sr311_bk_street_condition_2020_to_20261024": 134_727,
    "sr311_unassigned_borough_street_condition_since_2020": 5_768,
    "centerline_bk_rows": 27_673,
    "centerline_bk_eligible_blocks": 23_448,  # rw_type=1 & status=2 (join_tests_brooklyn_2025.txt)
    "pavement_bk_rows": 135_776,
    "workorders_bk_rows": 110_830,
    "resurfacing_inhouse_bk_rows": 88_921,
    "resurfacing_schedule_bk_rows": 108_587,
}

NOAA_STATION = {
    "id": "USW00094728",
    "name": "NY CITY CENTRAL PARK, NY US",
    "latitude": 40.77898,
    "longitude": -73.96925,
    "elevation_m": 42.7,
}


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def quality(self) -> Path:
        return self.root / "outputs" / "data_quality"

    @property
    def manifest_index(self) -> Path:
        return self.raw / "MANIFEST.jsonl"

    @property
    def local_registry(self) -> Path:
        return self.raw / "local_originals" / "registry.json"
