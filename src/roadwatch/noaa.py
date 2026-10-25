"""NOAA NCEI Access Data Service: GHCN-Daily observations for one station, one CSV per calendar-year partition."""
from __future__ import annotations

import datetime as dt
import io
import shutil
from pathlib import Path

import pandas as pd

from .http import InvalidResponse, RetryingClient
from .sources import NOAA_ELEMENTS, Partition, SourceSpec
from .storage import read_gz, rename_dir, utc_iso, write_gz, write_json


def request_params(spec: SourceSpec, part: Partition) -> dict:
    return {"dataset": "daily-summaries", "stations": spec.identifier, "startDate": part.start.isoformat(),
            "endDate": part.end_inclusive.isoformat(), "dataTypes": ",".join(NOAA_ELEMENTS), "units": "metric",
            "format": "csv", "includeAttributes": "true", "includeStationName": "true", "includeStationLocation": "1"}


def parse_csv(raw: bytes, spec: SourceSpec, part: Partition | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as e:  # noqa: BLE001 - any parser failure means the body is unusable
        raise InvalidResponse(f"NOAA body is not parseable CSV: {e}") from e
    if len(df) == 0:
        return df
    missing_station_cols = [c for c in ("STATION", "DATE") if c not in df.columns]
    if missing_station_cols:
        raise InvalidResponse(f"NOAA CSV missing columns {missing_station_cols}")
    dates = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise InvalidResponse("NOAA CSV has unparseable DATE values")
    if part is not None and (dates.min().date() < part.start or dates.max().date() > part.end_inclusive):
        raise InvalidResponse("NOAA CSV contains dates outside the requested range")
    return df


def fetch_partition(client: RetryingClient, spec: SourceSpec, part: Partition, part_dir: Path, **_) -> dict:
    tmp = part_dir.with_name(part_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    params = request_params(spec, part)
    resp, df = client.request(spec.endpoint, params, parse=lambda r: parse_csv(r.content, spec, part))
    sha = write_gz(tmp / "page_00000.csv.gz", resp.content)
    requested_days = (part.end_inclusive - part.start).days + 1
    got = set(df["DATE"]) if len(df) else set()
    all_days = {(part.start + dt.timedelta(days=i)).isoformat() for i in range(requested_days)}
    record = {"partition": part.name, "request_params": params, "requested_days": requested_days,
              "fetched_rows": len(df), "unique_ids": len(got), "missing_dates": sorted(all_days - got),
              "pages": [{"file": "page_00000.csv.gz", "rows": len(df), "sha256_uncompressed": sha}],
              "attempts": 1, "completed_utc": utc_iso(), "notes": []}
    write_json(tmp / "_SUCCESS.json", record)
    if part_dir.exists():
        shutil.rmtree(part_dir)
    rename_dir(tmp, part_dir)
    return record


def load_partition_frame(part_dir: Path, spec: SourceSpec) -> pd.DataFrame:
    return parse_csv(read_gz(part_dir / "page_00000.csv.gz"), spec)
