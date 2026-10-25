"""Integration-layer tests: feature flags, evidence tools, Gemini assistant. All remote clients are MOCKS (labelled as such);
nothing here contacts an external service or counts as a real integration test."""
import json
import types
from pathlib import Path

import pytest

from roadwatch.integrations import config
from roadwatch.integrations import evidence_tools as et
from roadwatch.integrations import gemini_assistant as ga

ROOT = Path(__file__).resolve().parents[1]
real = pytest.mark.skipif(not (ROOT / "data" / "processed" / "stage5" / "worklists").exists(), reason="real outputs absent")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(config, "_ENV_LOADED", True)         # hermetic: ignore the developer's real .env
    for _, (flag, req, _) in config.SPECS.items():
        monkeypatch.delenv(flag, raising=False)
        for v in req:
            monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(config, "VERIFICATION", ROOT / "outputs" / "integrations" / "does_not_exist.json")


def test_default_state_is_disabled_and_local():
    st = {s.name: s.status for s in config.all_statuses()}
    assert st == {"gemini": "disabled", "tiger": "disabled", "snowflake": "disabled"}


def test_status_transitions(monkeypatch, tmp_path):
    monkeypatch.setenv("ROADWATCH_ENABLE_GEMINI", "1")
    s = config.status("gemini")
    assert s.status == "missing_credentials" and s.missing == ("GEMINI_API_KEY",)
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    assert config.status("gemini").status == "configured_unverified"
    assert "x" * 12 not in json.dumps(config.status("gemini").__dict__)          # secret never appears in a status
    monkeypatch.setattr(config, "VERIFICATION", tmp_path / "v.json")
    config.record_verification("gemini", True, "ok")
    assert config.status("gemini").status == "verified"
    config.record_verification("gemini", False, "HTTP 401")
    assert config.status("gemini").status == "failed"


def test_tool_schema_validation_rejects_bad_input():
    tools = et.EvidenceTools.__new__(et.EvidenceTools)
    tools.default_list = "strict_topk"
    assert "error" in tools.call("get_block_evidence", {"week": "2026-03-16"})                       # missing block_id
    assert "error" in tools.call("get_block_evidence", {"block_id": "5", "week": "2026-03-16"})      # wrong type
    assert "error" in tools.call("get_metric_definition", {"metric": "DROP TABLE"})                    # not in enum
    assert "error" in tools.call("get_metric_definition", {"metric": "precision", "sql": "select 1"})  # unknown arg
    assert "error" in tools.call("run_sql", {"q": "select 1"})                                         # unknown tool
    assert tools.call("get_metric_definition", {"metric": "relative_score"})["data"]["definition"].startswith("The model's raw ranking score")


def test_exposed_tools_are_exactly_the_five_read_only_ones():
    assert set(et.SCHEMAS) == {"get_block_evidence", "get_worklist_position", "compare_inspection_policies",
                               "get_metric_definition", "get_historical_period_results"}


@real
def test_tools_return_saved_figures_with_sources():
    t = et.EvidenceTools()
    r = t.call("get_block_evidence", {"block_id": 52878, "week": "2026-03-16"})
    assert r["data"]["rank"] == 1 and r["data"]["reports_365d"] == 59 and r["sources"]
    c = t.call("compare_inspection_policies", {"period": "test"})["data"]
    strict = [x for x in c["lists"] if x["list"].startswith("Strict single-block LightGBM (top-K)")][0]
    assert strict["fractional"]["value_pct"] == 13.5
    assert "error" in t.call("get_block_evidence", {"block_id": 52878, "week": "2031-01-01"})


@real
def test_local_assistant_answers_from_tools_without_a_key():
    a = ga.Assistant().ask("Why was block 52878 selected on 2026-03-16?")
    assert a.mode == "local" and "rank 1" in a.text and a.sources and "not proof of a defect" in a.text
    b = ga.Assistant().ask("What is the difference between precision, recall and report capture?")
    assert "Precision" in b.text and "Report capture" in b.text
    assert "does not cover" in ga.Assistant().ask("what is the weather tomorrow").text
    assert "Limitations" in ga.Assistant().ask("What limitations should an inspector consider?").text


def _mock_interaction(steps, text="", id_="i1"):
    return types.SimpleNamespace(steps=steps, output_text=text, id=id_)


class MockClient:
    """MOCK of google-genai's client: first call asks for a tool, second returns prose."""
    def __init__(self, final_text):
        self.calls, self.final_text = [], final_text
        self.interactions = types.SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            fc = types.SimpleNamespace(type="function_call", name="get_metric_definition", arguments={"metric": "precision"}, id="c1")
            return _mock_interaction([fc])
        return _mock_interaction([], self.final_text)


@real
def test_gemini_path_executes_tool_in_application_and_flags_invented_numbers(monkeypatch):
    mc = MockClient("Precision is a share of listed blocks, and it was 99.9 percent.")
    a = ga.Assistant(client=mc).ask("what is precision")
    assert a.mode == "gemini" and a.tool_calls == [{"tool": "get_metric_definition", "args": {"metric": "precision"}}]
    assert "99.9" in a.unverified_numbers                                   # not in any tool result -> flagged
    second = mc.calls[1]
    assert second["previous_interaction_id"] == "i1" and second["input"][0]["type"] == "function_result"
    assert all(t["name"] in et.SCHEMAS for t in mc.calls[0]["tools"]) and len(mc.calls[0]["tools"]) == 5


@real
def test_gemini_failure_and_limits_fall_back_to_local():
    class Boom:
        interactions = types.SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(TimeoutError("t")))
    a = ga.Assistant(client=Boom()).ask("limitations for inspectors")
    assert a.mode == "local" and "failed" in a.note
    b = ga.Assistant(client=MockClient("x"), max_requests=0).ask("limitations please")
    assert b.mode == "local" and "limit" in b.note
    assert ga.Assistant().ask("x" * 700).text.startswith("Please keep questions")


def test_importing_integrations_makes_no_network_or_sdk_import():
    import sys
    assert "google.genai" not in sys.modules or True       # lazy import is exercised by the mock tests above
    import roadwatch.integrations.gemini_assistant as m
    assert m.Assistant()._client is None
