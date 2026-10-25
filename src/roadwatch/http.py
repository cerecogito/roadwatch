"""HTTP client with bounded retries for rate limits, server errors, timeouts and unparseable (partial) bodies.

Headers (which may carry an app token) are never logged.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class HttpError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class InvalidResponse(Exception):
    """The server answered 200 but the body is not what the endpoint promises (e.g. truncated JSON)."""


class RetriesExhausted(HttpError):
    pass


def _snippet(text: str, n: int = 300) -> str:
    return " ".join(text[:n].split())


class RetryingClient:
    def __init__(self, session: Any = None, headers: dict | None = None, max_attempts: int = 6,
                 backoff_base: float = 2.0, backoff_max: float = 120.0, timeout: tuple[float, float] = (15, 300),
                 sleep: Callable[[float], None] = time.sleep):
        self.session = session or requests.Session()
        self._headers = dict(headers or {})
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self.sleep = sleep
        self.request_count = 0

    def _backoff(self, attempt: int) -> float:
        return min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1))

    def _retry_after(self, resp) -> float | None:
        value = (getattr(resp, "headers", None) or {}).get("Retry-After")
        try:
            return min(float(value), self.backoff_max * 2) if value is not None else None
        except ValueError:
            return None

    def request(self, url: str, params: dict | None = None, parse: Callable[[Any], Any] | None = None):
        """GET with retries. `parse(resp)` may raise InvalidResponse to trigger a retry. Returns (resp, parsed)."""
        last_err: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            wait = self._backoff(attempt)
            try:
                self.request_count += 1
                resp = self.session.get(url, params=params, headers=self._headers, timeout=self.timeout)
            except requests.RequestException as e:  # timeouts, connection resets, chunked-encoding errors
                last_err = e
            else:
                if resp.status_code == 200:
                    if parse is None:
                        return resp, None
                    try:
                        return resp, parse(resp)
                    except InvalidResponse as e:
                        last_err = e
                elif resp.status_code in RETRYABLE_STATUS:
                    last_err = HttpError(f"HTTP {resp.status_code}: {_snippet(resp.text)}", resp.status_code)
                    wait = self._retry_after(resp) or wait
                else:
                    raise HttpError(f"HTTP {resp.status_code} (not retryable) for {url}: {_snippet(resp.text)}",
                                    resp.status_code)
            if attempt < self.max_attempts:
                log.warning("attempt %d/%d failed for %s params=%s: %s; retrying in %.0fs",
                            attempt, self.max_attempts, url, params, last_err, wait)
                self.sleep(wait)
        raise RetriesExhausted(f"gave up after {self.max_attempts} attempts for {url}: {last_err}") from last_err
