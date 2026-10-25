"""Snowflake evidence tests. HTTP is a MOCK; no Snowflake account is contacted, so the real SQL API / Cortex behaviour is NOT verified."""
import types
from pathlib import Path

import pandas as pd
import pytest

from roadwatch.integrations import config
from roadwatch.integrations import snowflake_evidence as se

ROOT = Path(__file__).resolve().parents[1]
real = pytest.mark.skipif(not (ROOT / "outputs" / "stage5" / "capture_definitions.csv").exists(), reason="real outputs absent")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for v in ("ROADWATCH_ENABLE_SNOWFLAKE", "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_PAT", "SNOWFLAKE_CORTEX_MODEL"):
        monkeypatch.delenv(v, raising=False)


def test_only_approved_templates_and_enumerated_parameters():
    with pytest.raises(ValueError):
        se.check_request("DROP TABLE x", {})
    with pytest.raises(ValueError):
        se.check_request("capture_by_period", {"period": "test'; DROP TABLE x;--", "capacity": "1pct"})
    with pytest.raises(ValueError):
        se.check_request("capture_by_period", {"period": "test"})
    assert se.check_request("capture_by_period", {"period": "test", "capacity": "1pct"})
    assert all(sql.count("?") == len(names) and sql.lstrip().upper().startswith("SELECT") for sql, names, _ in se.TEMPLATES.values())


def test_number_verification_catches_invented_figures():
    rows = pd.DataFrame({"value_pct": [13.5], "ci_lo_pct": [12.5], "ci_hi_pct": [14.5], "weeks": [67], "reports": [16855]})
    assert se.verify_numbers("Fractional capture was 13.5% (12.5-14.5) over 67 weeks and 16,855 reports.", rows) == []
    assert se.verify_numbers("Capture was 15.2%.", rows) == ["15.2"]


def test_loader_sql_is_parameterised():
    s = se.load_sql()
    assert len(s) == 3 and s[1].count("?") == 1 and s[2].count("?") == 2 and "PARSE_JSON(?)" in s[2]


@real
def test_curated_table_is_versioned_and_local_brief_matches_saved_figures():
    df = se.build_curated()
    assert df["evidence_id"].is_unique and {"capture", "model"} <= set(df["family"]) <= {"capture", "model", "multimodal"}
    assert se.evidence_version(df) == se.evidence_version(se.build_curated())              # deterministic version
    out = se.local_brief(df, "capture_by_period", {"period": "test", "capacity": "1pct"})
    assert "13.5% (12.5-14.5)" in out["brief"] and out["unverified_numbers"] == [] and out["mode"] == "local"
    assert out["provenance"]["template"] == "capture_by_period" and out["provenance"]["evidence_version"]
    m = se.local_brief(df, "model_vs_baselines", {"target": "primary", "period": "overall"})
    assert "LGBM: precision@1% 9.5%" in m["brief"] and "must not be compared" in m["brief"]


def _post_factory(cortex_text, calls):
    def post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        if url.endswith("/api/v2/statements"):
            data = {"resultSetMetaData": {"rowType": [{"name": n} for n in ("LIST_NAME", "METRIC", "VALUE_PCT", "CI_LO_PCT", "CI_HI_PCT", "WEEKS", "REPORTS")]},
                    "data": [["Strict single-block LightGBM (top-K)", "fractional", "13.5", "12.5", "14.5", "67", "16855"]]}
        else:
            data = {"choices": [{"message": {"content": cortex_text}}], "usage": {"total_tokens": 99}}
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    return post


@real
def test_remote_path_request_shape_cache_and_unverified_flag(monkeypatch, tmp_path):
    df = se.build_curated()
    monkeypatch.setenv("ROADWATCH_ENABLE_SNOWFLAKE", "1")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "org-acct")
    monkeypatch.setenv("SNOWFLAKE_PAT", "p" * 12)
    calls = []
    c = se.SnowflakeEvidence(post=_post_factory("Fractional capture (strict single-block target) was 13.5%, CI 12.5-14.5.", calls), cache_dir=tmp_path)
    out = c.brief(df, "capture_by_period", {"period": "test", "capacity": "1pct"})
    assert out["mode"] == "snowflake-cortex" and out["unverified_numbers"] == [] and out["provenance"]["model"] == se.DEFAULT_MODEL
    sql_call, llm_call = calls
    assert sql_call[0] == "https://org-acct.snowflakecomputing.com/api/v2/statements" and sql_call[1]["bindings"]["1"] == {"type": "TEXT", "value": "test"}
    assert "?" in sql_call[1]["statement"] and "test" not in sql_call[1]["statement"]          # values travel as bindings, not SQL text
    assert llm_call[0].endswith("/api/v2/cortex/v1/chat/completions") and llm_call[2]["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
    assert "p" * 12 not in str(out)
    again = c.brief(df, "capture_by_period", {"period": "test", "capacity": "1pct"})
    assert "cache" in again["note"] and len(calls) == 2
    bad = se.SnowflakeEvidence(post=_post_factory("Capture was 99.9%.", []), cache_dir=tmp_path / "b").brief(df, "capture_by_period", {"period": "validation", "capacity": "1pct"})
    assert bad["unverified_numbers"] == ["99.9"] and "unverified" in bad["note"]
    assert se.SnowflakeEvidence(post=_post_factory("x", []), max_requests=0, cache_dir=tmp_path / "c").brief(df, "capture_by_period", {"period": "test", "capacity": "2pct"})["mode"] == "local"


@real
def test_disabled_and_failing_remote_fall_back_to_local(monkeypatch, tmp_path):
    df = se.build_curated()
    out = se.SnowflakeEvidence(cache_dir=tmp_path).brief(df, "capture_by_period", {"period": "test", "capacity": "1pct"})
    assert out["mode"] == "local" and "disabled" in out["note"]
    monkeypatch.setenv("ROADWATCH_ENABLE_SNOWFLAKE", "1")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "a")
    monkeypatch.setenv("SNOWFLAKE_PAT", "p" * 9)
    boom = se.SnowflakeEvidence(post=lambda *a, **k: (_ for _ in ()).throw(TimeoutError()), cache_dir=tmp_path / "z")
    assert "failed" in boom.brief(df, "capture_by_period", {"period": "test", "capacity": "1pct"})["note"]


@real
def test_multimodal_brief_uses_the_curated_experiment_rows():
    df = se.build_curated()
    if "multimodal" not in set(df["family"]):
        pytest.skip("multimodal comparison not generated")
    out = se.local_brief(df, "multimodal_vs_baseline", {"period": "validation_multimodal"})
    assert "F0" in out["brief"] and "F3" in out["brief"] and "no demonstrated gain" in out["brief"] and out["unverified_numbers"] == []
    with pytest.raises(ValueError):
        se.check_request("multimodal_vs_baseline", {"period": "anything else"})
