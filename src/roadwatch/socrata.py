"""Socrata (NYC Open Data) access: metadata, counts and count-verified, paginated partition downloads."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .config import SOCRATA_DOMAIN
from .http import InvalidResponse, RetryingClient
from .sources import Partition, SourceSpec
from .storage import read_gz, rename_dir, schema_fingerprint, utc_iso, write_gz, write_json

log = logging.getLogger(__name__)

SELECT = ":*, *"      # system fields (:id, :created_at, :updated_at, :version) plus every published column
ORDER = ":id"         # stable ordering for offset pagination


class IncompleteExtractError(Exception):
    pass


def parse_json_list(resp) -> list:
    try:
        data = json.loads(resp.content)
    except ValueError as e:
        raise InvalidResponse(f"body is not valid JSON (possibly truncated): {e}") from e
    if isinstance(data, dict) and (data.get("error") or "message" in data):
        raise InvalidResponse(f"error object returned with HTTP 200: {str(data)[:200]}")
    if not isinstance(data, list):
        raise InvalidResponse(f"expected a JSON array, got {type(data).__name__}")
    return data


def fetch_metadata(client: RetryingClient, dataset_id: str) -> dict:
    def parse(resp):
        try:
            m = json.loads(resp.content)
        except ValueError as e:
            raise InvalidResponse(str(e)) from e
        if not isinstance(m, dict) or "columns" not in m:
            raise InvalidResponse("metadata without 'columns'")
        return m
    _, m = client.request(f"{SOCRATA_DOMAIN}/api/views/{dataset_id}.json", parse=parse)
    cols = [(c["fieldName"], c["dataTypeName"]) for c in m["columns"] if not c["fieldName"].startswith(":")]
    return {"dataset_id": dataset_id, "name": m.get("name"), "rowsUpdatedAt": m.get("rowsUpdatedAt"),
            "columns": [{"fieldName": n, "dataTypeName": t} for n, t in cols],
            "schema_fingerprint": schema_fingerprint(cols)}


def count_rows(client: RetryingClient, spec: SourceSpec, where: str) -> int:
    def parse(resp):
        rows = parse_json_list(resp)
        if len(rows) != 1 or "n" not in rows[0]:
            raise InvalidResponse(f"unexpected count response: {rows!r:.200}")
        return int(rows[0]["n"])
    _, n = client.request(spec.endpoint, {"$select": "count(*) AS n", "$where": where}, parse=parse)
    return n


def partition_where(spec: SourceSpec, part: Partition) -> str:
    return f"({spec.base_where}) AND ({part.where})"


def fetch_partition(client: RetryingClient, spec: SourceSpec, part: Partition, part_dir: Path,
                    page_size: int, max_attempts: int = 3) -> dict:
    """Download one partition page by page into `<part_dir>.partial`, verify it against a server-side count, then
    atomically rename it to `part_dir` with a _SUCCESS.json record. Raises IncompleteExtractError if it never verifies."""
    where = partition_where(spec, part)
    tmp = part_dir.with_name(part_dir.name + ".partial")
    notes: list[str] = []
    for attempt in range(1, max_attempts + 1):
        expected = count_rows(client, spec, where)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        pages, ids, dup_ids, offset, i = [], set(), 0, 0, 0
        while True:
            params = {"$select": SELECT, "$where": where, "$order": ORDER, "$limit": page_size, "$offset": offset}
            resp, rows = client.request(spec.endpoint, params, parse=parse_json_list)
            fname = f"page_{i:05d}.json.gz"
            sha = write_gz(tmp / fname, resp.content)
            for r in rows:
                rid = r.get(":id")
                if rid is None:
                    raise IncompleteExtractError(f"{spec.key}/{part.name}: row without :id system field")
                if rid in ids:
                    dup_ids += 1
                ids.add(rid)
            pages.append({"file": fname, "offset": offset, "rows": len(rows), "sha256_uncompressed": sha})
            i += 1
            if len(rows) < page_size:
                break
            offset += page_size
        fetched = sum(p["rows"] for p in pages)
        ok = dup_ids == 0 and len(ids) == fetched == expected
        if not ok and dup_ids == 0 and len(ids) == fetched:
            recount = count_rows(client, spec, where)
            if recount == fetched:
                notes.append(f"server count changed during fetch ({expected} -> {recount}); accepted re-count")
                expected, ok = recount, True
        if ok:
            record = {"partition": part.name, "where": where, "expected_count": expected, "fetched_rows": fetched,
                      "unique_ids": len(ids), "pages": pages, "page_size": page_size, "attempts": attempt,
                      "completed_utc": utc_iso(), "notes": notes}
            write_json(tmp / "_SUCCESS.json", record)
            if part_dir.exists():  # stale directory without _SUCCESS from an older interrupted run
                shutil.rmtree(part_dir)
            rename_dir(tmp, part_dir)
            return record
        msg = (f"{spec.key}/{part.name} attempt {attempt}: expected {expected}, fetched {fetched}, "
               f"unique {len(ids)}, duplicate ids across pages {dup_ids}")
        notes.append(msg)
        log.warning(msg)
    raise IncompleteExtractError("; ".join(notes))


def load_partition_rows(part_dir: Path) -> list[dict]:
    record = json.loads((part_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for p in record["pages"]:
        rows.extend(json.loads(read_gz(part_dir / p["file"])))
    if len(rows) != record["fetched_rows"]:
        raise IncompleteExtractError(f"{part_dir}: archived pages hold {len(rows)} rows, record says {record['fetched_rows']}")
    return rows
