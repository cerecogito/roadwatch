"""Snapshot lifecycle: create or resume, download partitions, cross-check coverage, finalise an immutable manifest.

Layout:  data/raw/<source>/<snapshot_id>/
             _state.json                 mutable while downloading
             dataset_metadata.json       published schema at retrieval time (Socrata only)
             partitions/<name>/          verified pages + _SUCCESS.json
             partitions/<name>.partial/  in-flight download (discarded on resume)
             manifest.json               written exactly once, after validation (immutable)
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from . import noaa, socrata
from .config import NYC_TZ, Paths
from .http import HttpError, RetryingClient
from .sources import Partition, SourceSpec
from .storage import append_jsonl, read_json, sha256_file, utc_iso, utc_now, write_json

log = logging.getLogger(__name__)


class SnapshotError(Exception):
    pass


@dataclass
class SnapshotResult:
    snapshot_dir: Path
    downloaded: bool
    failed_partitions: list[str]


def default_end() -> dt.date:
    """Exclusive end of the requested range: today's date in New York (records for today are not yet published)."""
    return utc_now().astimezone(ZoneInfo(NYC_TZ)).date()


def partitions_for(spec: SourceSpec, state: dict) -> list[Partition]:
    start = dt.date.fromisoformat(state["requested_start"]) if state.get("requested_start") else None
    end = dt.date.fromisoformat(state["requested_end_exclusive"])
    return spec.partitioner(spec, start, end)


def list_snapshots(paths: Paths, spec: SourceSpec) -> list[Path]:
    d = paths.raw / spec.key
    return sorted(p for p in d.iterdir() if p.is_dir() and (p / "_state.json").exists()) if d.exists() else []


def latest_finalized(paths: Paths, spec: SourceSpec) -> Path | None:
    done = [p for p in list_snapshots(paths, spec) if (p / "manifest.json").exists()]
    return done[-1] if done else None


def _resumable(paths: Paths, spec: SourceSpec) -> Path | None:
    open_ = [p for p in list_snapshots(paths, spec) if not (p / "manifest.json").exists()]
    return open_[-1] if open_ else None


def download(spec: SourceSpec, paths: Paths, client: RetryingClient, *, start: dt.date | None = None,
             end: dt.date | None = None, page_size: int | None = None, new: bool = False,
             token_used: bool = False) -> SnapshotResult:
    snap = None if new else _resumable(paths, spec)
    if snap is not None:
        state = read_json(snap / "_state.json")
        if (start and state.get("requested_start") != start.isoformat()) or \
           (end and state["requested_end_exclusive"] != end.isoformat()):
            raise SnapshotError(f"{snap} is an unfinished snapshot for a different range; use --new to start another")
        log.info("resuming %s", snap)
    else:
        start = start or spec.default_start
        end = end or default_end()
        snap_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
        snap = paths.raw / spec.key / snap_id
        snap.mkdir(parents=True, exist_ok=False)
        state = {"source_key": spec.key, "snapshot_id": snap_id, "status": "in_progress",
                 "retrieval_started_utc": utc_iso(), "requested_start": start.isoformat() if start else None,
                 "requested_end_exclusive": end.isoformat(), "page_size": page_size or spec.page_size,
                 "app_token_used": token_used, "partition_results": {}, "errors": []}
        if spec.kind == "socrata":
            write_json(snap / "dataset_metadata.json", socrata.fetch_metadata(client, spec.identifier))
        write_json(snap / "_state.json", state)

    failed: list[str] = []
    for part in partitions_for(spec, state):
        pdir = snap / "partitions" / part.name
        if (pdir / "_SUCCESS.json").exists():
            continue
        try:
            if spec.kind == "socrata":
                rec = socrata.fetch_partition(client, spec, part, pdir, page_size=state["page_size"])
            else:
                rec = noaa.fetch_partition(client, spec, part, pdir)
            log.info("%s/%s: %d rows in %d page(s)", spec.key, part.name, rec["fetched_rows"], len(rec["pages"]))
        except (HttpError, socrata.IncompleteExtractError) as e:
            failed.append(part.name)
            state["errors"].append({"partition": part.name, "at_utc": utc_iso(), "error": str(e)[:500]})
            log.error("%s/%s failed: %s", spec.key, part.name, e)
        state["partition_results"][part.name] = "failed" if part.name in failed else "ok"
        write_json(snap / "_state.json", state)

    if failed:
        state["status"] = "in_progress"
        write_json(snap / "_state.json", state)
        return SnapshotResult(snap, False, failed)

    if spec.kind == "socrata" and "coverage_check" not in state:
        state["coverage_check"] = coverage_check(spec, client, snap, state)
    state["status"] = "downloaded"
    state["retrieval_completed_utc"] = state.get("retrieval_completed_utc") or utc_iso()
    write_json(snap / "_state.json", state)
    return SnapshotResult(snap, True, [])


def coverage_check(spec: SourceSpec, client: RetryingClient, snap: Path, state: dict) -> dict:
    """Does the union of partitions cover every row matching the base filter? Compares the server total with the
    sum of verified partition counts, so a gap in the partition design cannot pass silently."""
    start = dt.date.fromisoformat(state["requested_start"]) if state.get("requested_start") else None
    end = dt.date.fromisoformat(state["requested_end_exclusive"])
    cov = spec.coverage(spec, start, end)
    where = spec.base_where if cov is None else f"({spec.base_where}) AND {cov}"
    total = socrata.count_rows(client, spec, where)
    fetched = sum(read_json(snap / "partitions" / p.name / "_SUCCESS.json")["fetched_rows"]
                  for p in partitions_for(spec, state))
    return {"where": where, "server_total_at_check": total, "sum_of_partitions": fetched,
            "difference": total - fetched, "checked_utc": utc_iso(), "passed": total == fetched}


def load_records(spec: SourceSpec, snap: Path):
    state = read_json(snap / "_state.json")
    if state["status"] not in ("downloaded", "finalized"):
        raise SnapshotError(f"{snap} is not a completed download (status={state['status']}); refusing to build")
    if spec.kind == "socrata":
        rows: list[dict] = []
        for part in partitions_for(spec, state):
            rows.extend(socrata.load_partition_rows(snap / "partitions" / part.name))
        return rows
    import pandas as pd
    frames = [noaa.load_partition_frame(snap / "partitions" / p.name, spec) for p in partitions_for(spec, state)]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def finalize(spec: SourceSpec, paths: Paths, snap: Path, manifest: dict) -> Path:
    """Write manifest.json once and append a line to the global index. Refuses to overwrite."""
    target = snap / "manifest.json"
    if target.exists():
        raise SnapshotError(f"{target} already exists; manifests are immutable")
    state = read_json(snap / "_state.json")
    partitions = []
    for part in partitions_for(spec, state):
        rec = read_json(snap / "partitions" / part.name / "_SUCCESS.json")
        partitions.append(rec)
    manifest = {"manifest_version": 1, **manifest, "partitions": partitions,
                "coverage_check": state.get("coverage_check"), "retrieval_errors_resolved": state.get("errors", [])}
    write_json(target, manifest)
    state["status"] = "finalized"
    write_json(snap / "_state.json", state)
    append_jsonl(paths.manifest_index, {
        "source_key": spec.key, "snapshot_id": state["snapshot_id"], "manifest": str(target.relative_to(paths.root)),
        "manifest_sha256": sha256_file(target), "extract_status": manifest.get("extract_status"),
        "validation_status": manifest.get("validation", {}).get("status"), "finalized_utc": utc_iso()})
    return target
