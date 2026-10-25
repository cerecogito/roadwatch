"""Feature flags, credential detection and integration status (no network access).

Status values (never inferred from a mock; "verified" only comes from a recorded real smoke test):
  verified               a real call to the external service succeeded (outputs/integrations/verification.json)
  configured_unverified  enabled and credentials present, but no recorded real smoke test
  disabled               feature flag off (the default)
  missing_credentials    enabled but required environment variables are absent
  failed                 the last real smoke test failed
  unsupported            cannot be implemented in this environment (see note)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERIFICATION = ROOT / "outputs" / "integrations" / "verification.json"

# name -> (flag variable, required env vars, label)
SPECS = {
    "gemini": ("ROADWATCH_ENABLE_GEMINI", ("GEMINI_API_KEY",), "Gemini evidence assistant"),
    "tiger": ("ROADWATCH_ENABLE_TIGER", ("TIGER_DATABASE_URL",), "Tiger Data time-series layer"),
    "snowflake": ("ROADWATCH_ENABLE_SNOWFLAKE", ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_PAT"), "Snowflake evidence briefs"),
}
TRUE = {"1", "true", "yes", "on"}


@dataclass
class Status:
    name: str
    label: str
    status: str
    detail: str
    missing: tuple[str, ...] = ()
    verified_at: str | None = None
    extras: dict = field(default_factory=dict)


_ENV_LOADED = False


def _load_env() -> None:
    """Read the git-ignored .env once (existing environment variables win). Values are never logged."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        _ENV_LOADED = True
        from ..env import load_dotenv
        load_dotenv(ROOT / ".env")


def flag(name: str) -> bool:
    _load_env()
    return os.environ.get(SPECS[name][0], "").strip().lower() in TRUE


def missing_credentials(name: str) -> tuple[str, ...]:
    _load_env()
    return tuple(v for v in SPECS[name][1] if not os.environ.get(v, "").strip())


def secret(var: str) -> str | None:
    """Read a credential from the environment. The value must never be logged or written anywhere."""
    _load_env()
    return os.environ.get(var, "").strip() or None


def verification_record(name: str) -> dict | None:
    try:
        return json.loads(VERIFICATION.read_text(encoding="utf-8")).get(name)
    except (OSError, ValueError):
        return None


def record_verification(name: str, ok: bool, detail: str, evidence: dict | None = None) -> None:
    """Called ONLY by scripts that performed a real external call. Stores no credentials."""
    import datetime as dt
    data = {}
    if VERIFICATION.exists():
        data = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    data[name] = {"ok": ok, "detail": detail, "at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "evidence": evidence or {}}
    VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
    VERIFICATION.write_text(json.dumps(data, indent=1), encoding="utf-8")


def status(name: str) -> Status:
    _, req, label = SPECS[name]
    if not flag(name):
        return Status(name, label, "disabled", f"Set {SPECS[name][0]}=1 to enable.")
    miss = missing_credentials(name)
    if miss:
        return Status(name, label, "missing_credentials", "Missing: " + ", ".join(miss), missing=miss)
    rec = verification_record(name)
    if rec and rec.get("ok"):
        return Status(name, label, "verified", rec["detail"], verified_at=rec["at_utc"])
    if rec and not rec.get("ok"):
        return Status(name, label, "failed", rec["detail"], verified_at=rec["at_utc"])
    return Status(name, label, "configured_unverified", "Credentials present; no real smoke test recorded yet.")


def all_statuses() -> list[Status]:
    return [status(n) for n in SPECS]
