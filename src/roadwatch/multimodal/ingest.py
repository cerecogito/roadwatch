"""Bounded, reproducible extracts of 311 water and sewer complaints (NYC Open Data erm2-nwe9, Brooklyn, 2020 onward).

Pages by :id, accepts the extract only when rows fetched == unique ids == server count(*), archives every response byte for
byte (gzip + sha256) under data/raw/multimodal/infra311/<utc>/, and writes data/processed/multimodal/infra311.parquet.
These are COMPLAINTS (a caller's report), not confirmed leaks or flooding. Only creation fields are used: created_date,
type, descriptor and location are fixed when the request is made; status/closure fields (later revised) are never read.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests

from ..config import Paths

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
START = "2020-01-01"
WHERE = (f"borough='BROOKLYN' AND complaint_type in('Sewer','Water System','Water Maintenance','Sewer Maintenance') AND created_date>='{START}T00:00:00'")
FIELDS = "unique_key,created_date,complaint_type,descriptor,x_coordinate_state_plane,y_coordinate_state_plane"
PAGE = 50_000

# descriptor code (the "(WA1)" suffix) -> physical group. Codes outside these groups are ignored on purpose (e.g. private
# plumbing backups, water quality, billing).
GROUP_BY_CODE = {"WA1": "water_main_leak", "WA2": "water_main_leak",
                 "WC3": "hydrant_water", "WA4": "hydrant_water", "WC1": "hydrant_water",
                 "SC": "drain_flood", "SC1": "drain_flood", "SJ": "drain_flood", "SA1": "drain_flood", "SA3": "drain_flood"}
# NYC replaced the "Sewer"/"Water System" 311 complaint types with "Sewer Maintenance"/"Water Maintenance" (old types end
# 2026-07-29). Both taxonomies map to the same three physical groups so the signal stays continuous across the change.
GROUP_BY_NEW = {"Water Main Break": "water_main_leak", "Leak": "water_main_leak",
                "Fire Hydrant Running": "hydrant_water", "Fire Hydrant Running Full": "hydrant_water", "Fire Hydrant Leaking": "hydrant_water",
                "Catch Basin Clogged": "drain_flood", "Flooding on Street": "drain_flood", "Flooding on Highway": "drain_flood",
                "Manhole Overflow": "drain_flood", "Catch Basin Sunken": "drain_flood", "Manhole Cover Missing": "drain_flood",
                "Catch Basin Grate Missing": "drain_flood"}
GROUPS = ["water_main_leak", "hydrant_water", "drain_flood"]


def group_of(descriptor: str | None, complaint_type: str | None = None) -> str | None:
    if complaint_type in ("Water Maintenance", "Sewer Maintenance"):
        return GROUP_BY_NEW.get(descriptor or "")
    m = re.search(r"\(([A-Z]{1,4}\d?)\)\s*$", descriptor or "")
    return GROUP_BY_CODE.get(m.group(1)) if m else None


def _get(session, params):
    r = session.get(URL, params=params, timeout=90)
    r.raise_for_status()
    return r.content


def fetch(paths: Paths | None = None, session=None) -> dict:
    paths = paths or Paths()
    s = session or requests.Session()
    s.headers["User-Agent"] = "RoadWatch-multimodal/0.1 (research)"
    count = int(json.loads(_get(s, {"$select": "count(*) as n", "$where": WHERE}))[0]["n"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw = paths.raw / "multimodal" / "infra311" / stamp
    raw.mkdir(parents=True, exist_ok=True)
    rows, pages = [], []
    for i, off in enumerate(range(0, count + PAGE, PAGE)):
        body = _get(s, {"$select": FIELDS, "$where": WHERE, "$order": ":id", "$limit": PAGE, "$offset": off})
        data = json.loads(body)
        if not data:
            break
        f = raw / f"page_{i:03d}.json.gz"
        f.write_bytes(gzip.compress(body))
        pages.append({"file": f.name, "rows": len(data), "sha256_raw_bytes": hashlib.sha256(body).hexdigest()})
        rows += data
    df = pd.DataFrame(rows)
    if len(df) != count or df["unique_key"].nunique() != count:
        raise RuntimeError(f"extract incomplete: fetched {len(df)}, unique {df['unique_key'].nunique()}, server count {count}")
    man = {"source": "NYC Open Data erm2-nwe9", "where": WHERE, "fields": FIELDS, "server_count": count, "retrieved_utc": stamp, "pages": pages}
    (raw / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return {"count": count, "manifest": str(raw / "manifest.json"), "df": normalise(df)}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"unique_key": df["unique_key"].astype(str), "created": pd.to_datetime(df["created_date"]),
                        "complaint_type": df["complaint_type"], "descriptor": df["descriptor"],
                        "x": pd.to_numeric(df.get("x_coordinate_state_plane"), errors="coerce"),
                        "y": pd.to_numeric(df.get("y_coordinate_state_plane"), errors="coerce")})
    out["group"] = [group_of(d, c) for d, c in zip(out["descriptor"], out["complaint_type"])]
    return out


def save(paths: Paths, df: pd.DataFrame) -> Path:
    d = paths.processed / "multimodal"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "infra311.parquet"
    df.to_parquet(f, index=False)
    return f
