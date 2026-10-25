"""Atomic file writes, checksums and the append-only manifest index."""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def utc_iso(ts: dt.datetime | None = None) -> str:
    return (ts or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary sibling then rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=False, default=str).encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_gz(path: Path, raw: bytes) -> str:
    """Store the exact source bytes gzip-compressed. Returns the sha256 of the *uncompressed* source bytes."""
    atomic_write_bytes(path, gzip.compress(raw, compresslevel=6, mtime=0))
    return sha256_bytes(raw)


def read_gz(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def schema_fingerprint(columns: list[tuple[str, str]]) -> str:
    text = "\n".join(f"{n}:{t}" for n, t in sorted(columns))
    return sha256_bytes(text.encode("utf-8"))[:16]
