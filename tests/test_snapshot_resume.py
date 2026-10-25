"""Snapshot resume, immutability and safe promotion. SYNTHETIC data only."""
import datetime as dt

import pandas as pd
import pytest

from conftest import FakeSession, FakeSocrata, make_client, synthetic_rows
from roadwatch import pipeline, snapshot
from roadwatch.socrata import partition_where
from roadwatch.sources import Partition, SourceSpec
from roadwatch.storage import read_json, sha256_file


def _two_parts(spec, start, end):
    return [Partition("a", "p='a'"), Partition("b", "p='b'")]


SPEC = SourceSpec(key="synthetic", title="SYNTHETIC", kind="socrata", role="primary", identifier="test-0000",
                  endpoint="https://example.invalid/resource/test-0000.json", base_where="x=1", id_field="id",
                  required_fields=("id",), date_field=None, default_start=None, partitioner=_two_parts)
WA = partition_where(SPEC, Partition("a", "p='a'"))
WB = partition_where(SPEC, Partition("b", "p='b'"))
END = dt.date(2026, 10, 24)


def _data():
    return {WA: synthetic_rows(12, "a"), WB: synthetic_rows(7, "b")}


def test_interrupted_download_resumes_without_refetching(paths):
    s1 = FakeSession(handler=FakeSocrata(_data(), fail_where={WB: 400}))
    r1 = snapshot.download(SPEC, paths, make_client(s1), end=END, page_size=5)
    assert not r1.downloaded and r1.failed_partitions == ["b"]
    state = read_json(r1.snapshot_dir / "_state.json")
    assert state["status"] == "in_progress" and state["partition_results"] == {"a": "ok", "b": "failed"}
    a_page_sha = sha256_file(r1.snapshot_dir / "partitions" / "a" / "page_00000.json.gz")

    s2 = FakeSession(handler=FakeSocrata(_data()))
    r2 = snapshot.download(SPEC, paths, make_client(s2), page_size=5)
    assert r2.downloaded and r2.snapshot_dir == r1.snapshot_dir
    assert not any(c[1].get("$where") == WA for c in s2.calls)       # partition a was not fetched again
    assert sha256_file(r1.snapshot_dir / "partitions" / "a" / "page_00000.json.gz") == a_page_sha
    state = read_json(r2.snapshot_dir / "_state.json")
    assert state["status"] == "downloaded" and state["coverage_check"]["passed"]
    assert len(snapshot.load_records(SPEC, r2.snapshot_dir)) == 19


def test_partial_directory_is_discarded_on_resume(paths):
    r1 = snapshot.download(SPEC, paths, make_client(FakeSession(handler=FakeSocrata(_data(), fail_where={WB: 400}))),
                           end=END, page_size=5)
    junk = r1.snapshot_dir / "partitions" / "b.partial"
    junk.mkdir(parents=True)
    (junk / "page_00000.json.gz").write_bytes(b"not gzip - leftover from a crash")
    r2 = snapshot.download(SPEC, paths, make_client(FakeSession(handler=FakeSocrata(_data()))), page_size=5)
    assert r2.downloaded and not junk.exists()
    assert len(snapshot.load_records(SPEC, r2.snapshot_dir)) == 19


def test_coverage_gap_is_reported(paths):
    server = FakeSocrata(_data())
    server.count_override["x=1"] = 20  # SYNTHETIC: the base filter matches a row no partition captured
    r = snapshot.download(SPEC, paths, make_client(FakeSession(handler=server)), end=END, page_size=5)
    cov = read_json(r.snapshot_dir / "_state.json")["coverage_check"]
    assert cov["passed"] is False and cov["difference"] == 1


def test_incomplete_snapshot_cannot_be_built(paths):
    r = snapshot.download(SPEC, paths, make_client(FakeSession(handler=FakeSocrata(_data(), fail_where={WB: 400}))),
                          end=END, page_size=5)
    with pytest.raises(snapshot.SnapshotError):
        snapshot.load_records(SPEC, r.snapshot_dir)


def test_manifest_is_written_once(paths):
    r = snapshot.download(SPEC, paths, make_client(FakeSession(handler=FakeSocrata(_data()))), end=END, page_size=5)
    snapshot.finalize(SPEC, paths, r.snapshot_dir, {"extract_status": "complete", "validation": {"status": "pass"}})
    with pytest.raises(snapshot.SnapshotError):
        snapshot.finalize(SPEC, paths, r.snapshot_dir, {"extract_status": "complete"})
    assert paths.manifest_index.read_text().count('"source_key": "synthetic"') == 1


def test_failed_validation_does_not_replace_existing_processed_table(paths, monkeypatch):
    r = snapshot.download(SPEC, paths, make_client(FakeSession(handler=FakeSocrata(_data()))), end=END, page_size=5)
    target = paths.processed / "synthetic.parquet"
    target.parent.mkdir(parents=True)
    pd.DataFrame({"previous": ["valid table"]}).to_parquet(target)
    before = sha256_file(target)
    monkeypatch.setattr(pipeline, "normalize", lambda spec, recs, ctx: (pd.DataFrame(recs), {"rows_in": len(recs)}))
    monkeypatch.setattr(pipeline, "validate", lambda spec, t, ctx: [{"check": "x", "status": "FAIL", "detail": "", "value": None}])
    out = pipeline.build(SPEC, paths, r.snapshot_dir, finalize_manifest=True)
    assert out.validation_status == "fail" and not out.promoted
    assert sha256_file(target) == before
    manifest = read_json(r.snapshot_dir / "manifest.json")
    assert manifest["validation"]["status"] == "fail" and manifest["processed"] is None
