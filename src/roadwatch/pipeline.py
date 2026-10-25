"""Orchestration: download -> build (normalise + validate) -> finalise manifest -> promote processed table.

Promotion rule: data/processed/<key>.parquet is replaced only when the new snapshot downloaded completely and has no
FAIL checks. A failed or partial refresh therefore never overwrites a valid existing table.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import snapshot as snap_mod
from .config import AVAILABILITY_LAGS, Paths
from .http import RetryingClient
from .normalize import normalize
from .sources import SourceSpec
from .storage import read_json, sha256_file, utc_iso, write_json
from .validate import summarize, validate

log = logging.getLogger(__name__)


@dataclass
class BuildOutcome:
    source_key: str
    snapshot_id: str
    validation_status: str
    promoted: bool
    rows: int
    report_path: Path


def _previous_fingerprint(paths: Paths, spec: SourceSpec, current: Path) -> str | None:
    for p in reversed(snap_mod.list_snapshots(paths, spec)):
        if p != current and (p / "dataset_metadata.json").exists() and (p / "manifest.json").exists():
            return read_json(p / "dataset_metadata.json")["schema_fingerprint"]
    return None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def _write_table(table, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    table.to_parquet(tmp, index=False)
    os.replace(tmp, target)
    return sha256_file(target)


def build(spec: SourceSpec, paths: Paths, snap: Path, *, finalize_manifest: bool) -> BuildOutcome:
    state = read_json(snap / "_state.json")
    ctx = {"state": state, "requested_end_exclusive": state["requested_end_exclusive"]}
    if spec.kind == "socrata":
        ctx["dataset_metadata"] = read_json(snap / "dataset_metadata.json")
        ctx["previous_schema_fingerprint"] = _previous_fingerprint(paths, spec, snap)
    records = snap_mod.load_records(spec, snap)
    raw_rows = len(records)
    table, dedup = normalize(spec, records, ctx)
    ctx["dedup"] = dedup
    checks = _jsonable(validate(spec, table, ctx))
    summary = summarize(checks)
    snapshot_id = state["snapshot_id"]

    promoted, processed_info = False, None
    target = paths.processed / f"{spec.key}.parquet"
    if summary["status"] != "fail":
        sha = _write_table(table, target)
        processed_info = {"path": str(target.relative_to(paths.root)), "sha256": sha, "rows": len(table)}
        write_json(paths.processed / f"{spec.key}.meta.json", {
            "source_key": spec.key, "snapshot_id": snapshot_id, "built_utc": utc_iso(), "rows": len(table),
            "sha256": sha, "validation_status": summary["status"], "role": spec.role,
            "raw_snapshot": str(snap.relative_to(paths.root))})
        promoted = True
    else:
        log.error("%s %s: validation FAILED; existing processed table left untouched", spec.key, snapshot_id)

    date_range = None
    if spec.date_field:
        col = {"sr311": "created_date_local", "sr311_unassigned": "created_date_local", "pavement": "inspection_local",
               "workorders": "rptdate_date", "resurfacing_inhouse": "location_actual_paving_end_date_local",
               "resurfacing_schedule": "date_local", "noaa": "date"}[spec.key]
        s = table[col].dropna() if len(table) else pd.Series(dtype="datetime64[ns]")
        date_range = {"field": spec.date_field, "min": str(s.min()) if len(s) else None, "max": str(s.max()) if len(s) else None}

    cov_ok = (state.get("coverage_check") or {}).get("passed", spec.kind != "socrata")
    extract_status = "complete" if state["status"] in ("downloaded", "finalized") and cov_ok else "partial"
    report = {
        "source_key": spec.key, "title": spec.title, "role": spec.role, "kind": spec.kind,
        "identifier": spec.identifier, "endpoint": spec.endpoint, "snapshot_id": snapshot_id,
        "retrieval_started_utc": state["retrieval_started_utc"], "retrieval_completed_utc": state.get("retrieval_completed_utc"),
        "query": {"base_where": spec.base_where, "select": ":*, *" if spec.kind == "socrata" else None,
                  "order": ":id" if spec.kind == "socrata" else None, "page_size": state["page_size"],
                  "requested_start": state.get("requested_start"), "requested_end_exclusive": state["requested_end_exclusive"],
                  "partitioning": f"{len(snap_mod.partitions_for(spec, state))} partitions on {spec.date_field or 'none'}"},
        "app_token_used": state.get("app_token_used", False),
        "dataset_metadata": {k: v for k, v in ctx.get("dataset_metadata", {}).items() if k != "columns"} or None,
        "observed_date_range": date_range,
        "totals": {"raw_rows": raw_rows, "rows_after_dedup": len(table), "dedup": dedup},
        "format": {"raw": "gzip of exact API response bytes (JSON pages for Socrata, CSV for NOAA); sha256 of uncompressed bytes",
                   "processed": "Parquet/GeoParquet; original string fields kept, parsed columns added"},
        "processed": processed_info, "extract_status": extract_status, "fitness": spec.fitness,
        "caveats": list(spec.caveats), "availability_lag": AVAILABILITY_LAGS.get(spec.key.replace("_unassigned", "")),
        "validation": {**summary, "checks": checks},
    }
    paths.quality.mkdir(parents=True, exist_ok=True)
    report_path = paths.quality / f"{spec.key}__{snapshot_id}.json"
    write_json(report_path, report)
    if finalize_manifest and not (snap / "manifest.json").exists():
        snap_mod.finalize(spec, paths, snap, report)
    return BuildOutcome(spec.key, snapshot_id, summary["status"], promoted, len(table), report_path)


def ingest(spec: SourceSpec, paths: Paths, client: RetryingClient, **kw) -> BuildOutcome | snap_mod.SnapshotResult:
    result = snap_mod.download(spec, paths, client, **kw)
    if not result.downloaded:
        log.error("%s: download incomplete (failed partitions %s); snapshot kept for resume: %s",
                  spec.key, result.failed_partitions, result.snapshot_dir)
        return result
    return build(spec, paths, result.snapshot_dir, finalize_manifest=True)


def rebuild(spec: SourceSpec, paths: Paths, snapshot_id: str | None = None) -> BuildOutcome:
    """Reproduce the processed table from an archived snapshot without any network access."""
    if snapshot_id:
        snap = paths.raw / spec.key / snapshot_id
    else:
        snap = snap_mod.latest_finalized(paths, spec)
        if snap is None:
            raise snap_mod.SnapshotError(f"no finalized snapshot for {spec.key}")
    return build(spec, paths, snap, finalize_manifest=False)
