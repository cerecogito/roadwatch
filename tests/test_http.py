"""Retry handling. SYNTHETIC responses only."""
import logging

import pytest
import requests

from conftest import FakeResponse, FakeSession, make_client
from roadwatch.http import HttpError, RetriesExhausted
from roadwatch.socrata import parse_json_list


def test_retries_server_error_then_succeeds():
    waits = []
    s = FakeSession([FakeResponse(503, "busy"), FakeResponse(200, "[]")])
    resp, rows = make_client(s, waits).request("u", {}, parse=parse_json_list)
    assert rows == [] and len(s.calls) == 2 and waits == [2.0]


def test_respects_retry_after_on_429():
    waits = []
    s = FakeSession([FakeResponse(429, "slow down", {"Retry-After": "7"}), FakeResponse(200, "[]")])
    make_client(s, waits).request("u", {}, parse=parse_json_list)
    assert waits == [7.0]


def test_timeout_and_connection_errors_are_retried():
    s = FakeSession([requests.Timeout("t"), requests.ConnectionError("c"), FakeResponse(200, "[1]")])
    _, rows = make_client(s).request("u", {}, parse=parse_json_list)
    assert rows == [1] and len(s.calls) == 3


def test_non_retryable_status_raises_immediately():
    s = FakeSession([FakeResponse(400, '{"message":"bad query"}')])
    with pytest.raises(HttpError) as e:
        make_client(s).request("u", {})
    assert e.value.status == 400 and len(s.calls) == 1


def test_truncated_json_body_is_retried():
    s = FakeSession([FakeResponse(200, '[{"a": 1}, {"a"'), FakeResponse(200, '[{"a": 1}]')])
    _, rows = make_client(s).request("u", {}, parse=parse_json_list)
    assert rows == [{"a": 1}] and len(s.calls) == 2


def test_error_object_with_http_200_is_retried_then_gives_up():
    body = '{"error": true, "message": "Internal error", "status": 500}'
    s = FakeSession([FakeResponse(200, body)] * 3)
    with pytest.raises(RetriesExhausted):
        make_client(s, max_attempts=3).request("u", {}, parse=parse_json_list)
    assert len(s.calls) == 3


def test_secret_header_never_logged(caplog):
    token = "SYNTHETIC-SECRET-TOKEN-123"
    s = FakeSession([FakeResponse(500, "oops"), FakeResponse(200, "[]")])
    client = make_client(s, headers={"X-App-Token": token})
    with caplog.at_level(logging.DEBUG):
        client.request("u", {"$where": "x=1"}, parse=parse_json_list)
    assert s.calls[0][2]["X-App-Token"] == token  # sent to the server...
    assert token not in caplog.text               # ...but never written to logs
