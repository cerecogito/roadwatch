"""ElevenLabs boundary tests. The HTTP layer is a MOCK; no real API call is made."""
import types

import pytest
import requests

from roadwatch.integrations import config
from roadwatch.integrations import elevenlabs_tts as el

EV = {"on_worklist": True, "rank": 1, "street": "17 AVE", "zip": "11204", "priority": "P1", "week": "2026-03-16", "reports_365d": 59,
      "reports_30d": 18, "length_m": 79, "travel_lanes": 2.0, "block_id": 52878}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for v in ("ROADWATCH_ENABLE_ELEVENLABS", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "ELEVENLABS_MODEL_ID"):
        monkeypatch.delenv(v, raising=False)


def test_narration_text_uses_only_verified_fields_and_disclaims():
    t = el.block_narration(EV)
    assert "Rank 1: 17 AVE" in t and "59 311 pothole reports" in t and "not an independent prediction" in t and "not a repair order" in t
    assert len(t) < el.MAX_CHARS


def test_disabled_and_missing_key_never_call_api():
    calls = []
    n = el.Narrator(post=lambda *a, **k: calls.append(1))
    audio, msg = n.synthesize("hello")
    assert audio is None and "disabled" in msg and not calls
    import os
    os.environ["ROADWATCH_ENABLE_ELEVENLABS"] = "1"
    audio, msg = n.synthesize("hello")
    assert audio is None and "missing" in msg and not calls


def test_request_shape_cache_and_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("ROADWATCH_ENABLE_ELEVENLABS", "1")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k" * 10)
    seen = []

    def ok(url, **kw):
        seen.append((url, kw))
        return types.SimpleNamespace(status_code=200, content=b"ID3mock")
    n = el.Narrator(post=ok, cache_dir=tmp_path)
    a1, m1 = n.synthesize(el.block_narration(EV))
    a2, m2 = n.synthesize(el.block_narration(EV))
    assert a1 == a2 == b"ID3mock" and n.api_calls == 1 and "cache" in m2          # second call served from cache
    url, kw = seen[0]
    assert url.endswith("/v1/text-to-speech/" + el.DEFAULT_VOICE) and kw["headers"]["xi-api-key"] == "k" * 10
    assert kw["json"]["model_id"] == el.DEFAULT_MODEL and kw["params"] == {"output_format": "mp3_44100_128"} and kw["timeout"] == 30
    assert "k" * 10 not in m1 + m2
    bad = el.Narrator(post=lambda *a, **k: types.SimpleNamespace(status_code=401, content=b""), cache_dir=tmp_path / "x")
    assert bad.synthesize("different text")[0] is None
    boom = el.Narrator(post=lambda *a, **k: (_ for _ in ()).throw(requests.ConnectTimeout()), cache_dir=tmp_path / "y")
    assert "unavailable" in boom.synthesize("another text")[1]
    assert el.Narrator(post=ok, cache_dir=tmp_path / "z").synthesize("x" * 1000)[0] is None
