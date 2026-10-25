"""Register the originally downloaded files in place (checksummed, never moved or modified) with their audit status."""
from __future__ import annotations

import datetime as dt

from .config import Paths
from .storage import append_jsonl, read_json, sha256_file, utc_iso, write_json

LOCAL_ORIGINALS = [
    {"file": "311_Service_Requests_from_2020_to_Present_20261024.csv", "dataset": "erm2-nwe9 (export)",
     "extract_status": "truncated", "pipeline_use": "EXCLUDED as historical source",
     "reason": "Export ends in a Socrata JSON error body ('Internal error', status 500) at line 109,646; "
               "valid rows cover only 2026-10-13 13:32:12 -> 2026-10-23 02:46:57.",
     "rows_verified": 109_644, "evidence": "data audit"},
    {"file": "Automated_Traffic_Volume_Counts_20261024.csv", "dataset": "7ym2-wayt",
     "extract_status": "complete", "pipeline_use": "EXCLUDED (rejected source)",
     "reason": "Only 146 of 23,448 Brooklyn blocks (0.62%) have a count since 2022; no imputation of exposure.",
     "rows_verified": 1_875_154, "evidence": "data audit"},
    {"file": "open-meteo-40.70N74.00W32m.csv", "dataset": "Open-Meteo export (40.70N, 74.00W, 32 m)",
     "extract_status": "incomplete", "pipeline_use": "EXCLUDED (NOAA GHCN-D is the weather source)",
     "reason": "precipitation (mm) is null in 59,592/59,592 rows; series ends 2026-10-18 19:00.",
     "rows_verified": 59_592, "evidence": "data audit"},
    {"file": "Centerline_20261024.csv", "dataset": "inkn-q76z", "extract_status": "complete",
     "pipeline_use": "Preserved original; pipeline ingests the same dataset via API (Brooklyn filter) for consistent "
                     "API field names and provenance",
     "reason": "Row count equals API count (122,331) at audit time.", "rows_verified": 122_331,
     "evidence": "data audit"},
    {"file": "Street_Pavement_Ratings_20261024.csv", "dataset": "6yyb-pb25", "extract_status": "complete",
     "pipeline_use": "Preserved original; pipeline ingests the same dataset via API (Brooklyn filter)",
     "reason": "Row count equals API count (524,928) at audit time.", "rows_verified": 524_928,
     "evidence": "data audit"},
]

EXCLUDED_WITHOUT_FILE = [
    {"source": "MTA Bus Time API (bustime.mta.info)", "pipeline_use": "EXCLUDED",
     "reason": "Real-time only; historical positions cannot be reconstructed for backtesting (audit §7b). "
               "Key kept in .env for possible forward collection; not used."},
]


def register(paths: Paths) -> dict:
    entries = []
    for item in LOCAL_ORIGINALS:
        p = paths.root / item["file"]
        if not p.exists():
            entries.append({**item, "present": False})
            continue
        st = p.stat()
        entries.append({**item, "present": True, "bytes": st.st_size, "sha256": sha256_file(p),
                        "modified_local": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                        "location": "project root (left in place, not moved or modified)"})
    registry = {"registered_utc": utc_iso(), "files": entries, "excluded_sources_without_file": EXCLUDED_WITHOUT_FILE}
    target = paths.local_registry
    previous = read_json(target) if target.exists() else None
    same = previous and [(e["file"], e.get("sha256")) for e in previous["files"]] == [(e["file"], e.get("sha256")) for e in entries]
    if not same:
        if previous:  # keep the old registry instead of overwriting it
            target.rename(target.with_name(f"registry_{previous['registered_utc'].replace(':', '')}.json"))
        write_json(target, registry)
        append_jsonl(paths.manifest_index, {"source_key": "local_originals", "manifest": str(target.relative_to(paths.root)),
                                            "manifest_sha256": sha256_file(target), "finalized_utc": registry["registered_utc"]})
        return registry
    return previous
