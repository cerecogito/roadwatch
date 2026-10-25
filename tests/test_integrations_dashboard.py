"""Dashboard behaviour with every integration disabled (the default): the original views must not touch the integration layer,
and the optional view must work locally. No remote service is contacted."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
real = pytest.mark.skipif(not (ROOT / "data" / "processed" / "stage5" / "worklists").exists(), reason="real outputs absent")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for v in ("ROADWATCH_ENABLE_GEMINI", "ROADWATCH_ENABLE_ELEVENLABS", "ROADWATCH_ENABLE_TIGER", "ROADWATCH_ENABLE_SNOWFLAKE", "ROADWATCH_ENABLE_SOLANA"):
        monkeypatch.delenv(v, raising=False)


def _run(view):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.query_params["week"] = "2026-03-16"
    at.query_params["view"] = view
    return at.run()


@real
def test_original_views_unchanged_and_do_not_import_integration_ui():
    for m in [m for m in sys.modules if m.startswith("roadwatch.integrations.ui")]:
        del sys.modules[m]
    at = _run("map")
    assert not at.exception and {m.label: m.value for m in at.metric}["Blocks to inspect"] == "235"
    assert "roadwatch.integrations.ui" not in sys.modules                         # nothing optional is loaded for the core view
    assert any("Audio explanation" in e.label for e in at.expander)               # audio panel is present but collapsed and passive
    for v in ("policy", "volume", "evidence", "data"):
        assert not _run(v).exception


@real
def test_integrations_view_works_locally_with_everything_disabled():
    at = _run("extras")
    assert not at.exception
    df = [d.value for d in at.dataframe][0]
    assert set(df["integration"]) >= {"Gemini evidence assistant", "ElevenLabs audio narration", "Tiger Data time-series layer", "Snowflake evidence briefs",
                                      "Solana Devnet receipts", "Presage Human Factors Lab", "Vultr deployment"}
    assert not any("Verified" in s for s in df["status"])                              # nothing can be 'verified' without a recorded real test
    at.text_input[0].set_value("Why was block 52878 selected on 2026-03-16?")
    [b for b in at.button if b.label == "Ask"][0].click()
    at.run()
    assert not at.exception and any("rank 1" in m.value for m in at.markdown)       # local assistant answered from saved evidence
    [b for b in at.button if b.label == "Generate evidence brief"][0].click()
    at.run()
    assert not at.exception and any("13.5% (12.5-14.5)" in m.value for m in at.markdown)
    assert any("Presage" in i.value or "No Python SDK" in i.value for i in at.info)
