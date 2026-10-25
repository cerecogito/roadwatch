"""Paginated, count-verified partition downloads. SYNTHETIC data only."""
import json

import pytest

from conftest import FakeSession, FakeSocrata, make_client, synthetic_rows
from roadwatch.socrata import IncompleteExtractError, fetch_partition, load_partition_rows, partition_where
from roadwatch.sources import Partition, SourceSpec, single

SPEC = SourceSpec(key="synthetic", title="SYNTHETIC", kind="socrata", role="primary", identifier="test-0000",
                  endpoint="https://example.invalid/resource/test-0000.json", base_where="x=1", id_field="id",
                  required_fields=("id",), date_field=None, default_start=None, partitioner=single)
PART = Partition("p1", "p=1")
WHERE = partition_where(SPEC, PART)


def _data_calls(session):
    return [c for c in session.calls if c[1].get("$select") == ":*, *"]


def test_multiple_pages_are_all_retrieved(tmp_path):
    rows = synthetic_rows(25)
    s = FakeSession(handler=FakeSocrata({WHERE: rows}))
    rec = fetch_partition(make_client(s), SPEC, PART, tmp_path / "p1", page_size=10)
    assert rec["fetched_rows"] == 25 and rec["unique_ids"] == 25 and len(rec["pages"]) == 3
    assert [c[1]["$offset"] for c in _data_calls(s)] == [0, 10, 20]
    assert all(c[1]["$order"] == ":id" for c in _data_calls(s))
    assert load_partition_rows(tmp_path / "p1") == rows
    assert not (tmp_path / "p1.partial").exists()


def test_exact_multiple_of_page_size_requests_one_empty_page(tmp_path):
    s = FakeSession(handler=FakeSocrata({WHERE: synthetic_rows(20)}))
    rec = fetch_partition(make_client(s), SPEC, PART, tmp_path / "p1", page_size=10)
    assert rec["fetched_rows"] == 20 and [p["rows"] for p in rec["pages"]] == [10, 10, 0]


def test_empty_partition_is_valid(tmp_path):
    s = FakeSession(handler=FakeSocrata({WHERE: []}))
    rec = fetch_partition(make_client(s), SPEC, PART, tmp_path / "p1", page_size=10)
    assert rec["fetched_rows"] == 0 and rec["expected_count"] == 0


def test_count_mismatch_never_marks_success(tmp_path):
    server = FakeSocrata({WHERE: synthetic_rows(25)}, count_override={WHERE: 26})  # server claims a row we never receive
    s = FakeSession(handler=server)
    with pytest.raises(IncompleteExtractError):
        fetch_partition(make_client(s), SPEC, PART, tmp_path / "p1", page_size=10, max_attempts=2)
    assert not (tmp_path / "p1").exists()
    assert not (tmp_path / "p1.partial" / "_SUCCESS.json").exists()


def test_offset_drift_duplicates_are_detected(tmp_path):
    rows = synthetic_rows(15)

    class Drifting(FakeSocrata):
        def __call__(self, url, params):
            resp = super().__call__(url, params)
            if params.get("$offset") == 10:  # SYNTHETIC: a row inserted upstream shifts the second page by one
                from conftest import FakeResponse
                return FakeResponse(200, json.dumps(rows[9:14]))
            return resp

    s = FakeSession(handler=Drifting({WHERE: rows}))
    with pytest.raises(IncompleteExtractError, match="duplicate ids"):
        fetch_partition(make_client(s), SPEC, PART, tmp_path / "p1", page_size=10, max_attempts=1)


def test_count_change_during_fetch_is_accepted_only_if_recount_matches(tmp_path):
    rows = synthetic_rows(12)
    counts = iter([11, 12])  # SYNTHETIC: a late-published row arrives between the first count and the download

    class Growing(FakeSocrata):
        def __call__(self, url, params):
            if params.get("$select") == "count(*) AS n":
                from conftest import FakeResponse
                return FakeResponse(200, json.dumps([{"n": str(next(counts))}]))
            return super().__call__(url, params)

    rec = fetch_partition(make_client(FakeSession(handler=Growing({WHERE: rows}))), SPEC, PART, tmp_path / "p1", page_size=10)
    assert rec["fetched_rows"] == 12 and rec["expected_count"] == 12 and rec["notes"]
