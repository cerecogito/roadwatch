"""Secrets handling. Values are read from the process environment or a git-ignored .env file and never logged."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> list[str]:
    """Load KEY=VALUE lines into os.environ without overriding existing variables. Returns the key names loaded."""
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def socrata_app_token() -> str | None:
    token = os.environ.get("SOCRATA_APP_TOKEN", "").strip()
    return token or None
