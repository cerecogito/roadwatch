"""Shared test helpers.

SYNTHETIC TEST FIXTURES: every row served by FakeSocrata/FakeSession below is invented for unit tests only.
None of it is NYC data and none of it may be used for any claim about data coverage or model performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.config import Paths  # noqa: E402
from roadwatch.http import RetryingClient  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes | str = b"", headers: dict | None = None):
        self.status_code = status_code
        self.content = content.encode() if isinstance(content, str) else content
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class FakeSession:
    """Plays back a scripted list of responses/exceptions, or delegates to a handler(url, params)."""

    def __init__(self, script=None, handler=None):
        self.script = list(script or [])
        self.handler = handler
        self.calls: list[tuple[str, dict, dict]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        item = self.handler(url, params or {}) if self.handler else self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSocrata:
    """SYNTHETIC: a minimal Socrata-like server keyed by the exact $where string."""

    def __init__(self, data: dict[str, list[dict]], count_override: dict[str, int] | None = None,
                 fail_where: dict[str, int] | None = None):
        self.data = data
        self.count_override = count_override or {}
        self.fail_where = fail_where or {}

    def __call__(self, url, params):
        where = params.get("$where")
        if where in self.fail_where:
            return FakeResponse(self.fail_where[where], '{"message": "synthetic failure"}')
        if url.endswith(".json") and "/api/views/" in url:
            return FakeResponse(200, json.dumps({"name": "synthetic", "rowsUpdatedAt": 0, "columns": [
                {"fieldName": "id", "dataTypeName": "text"}, {"fieldName": "value", "dataTypeName": "text"}]}))
        if params.get("$select") == "count(*) AS n":
            if where in self.count_override:
                n = self.count_override[where]
            elif where in self.data:
                n = len(self.data[where])
            else:  # coverage check queries the base filter alone: total = all rows across partitions
                n = sum(len(v) for v in self.data.values())
            return FakeResponse(200, json.dumps([{"n": str(n)}]))
        rows = self.data.get(where, [])
        off, lim = int(params.get("$offset", 0)), int(params.get("$limit", 1000))
        return FakeResponse(200, json.dumps(rows[off:off + lim]))


def synthetic_rows(n: int, prefix: str = "r") -> list[dict]:
    return [{":id": f"{prefix}-{i:04d}", ":updated_at": "2026-01-01T00:00:00Z", "id": f"{prefix}{i}", "value": str(i)}
            for i in range(n)]


@pytest.fixture
def paths(tmp_path) -> Paths:
    return Paths(tmp_path)


@pytest.fixture
def no_sleep():
    waits: list[float] = []
    return waits


def make_client(session, waits=None, **kw) -> RetryingClient:
    return RetryingClient(session=session, sleep=(waits.append if waits is not None else (lambda s: None)), **kw)
