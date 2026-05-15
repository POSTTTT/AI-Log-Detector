"""Tests for scripts/bootstrap_kibana.py.

Uses a tiny in-memory fake of ``requests.Session`` so the suite has zero
network dependencies and runs in milliseconds.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

# Make the in-tree script importable.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bootstrap_kibana  # noqa: E402


# ---------------------------------------------------------------------------
# Fake requests.Session
# ---------------------------------------------------------------------------

@dataclass
class FakeResponse:
    status_code: int = 200
    body: Any = field(default_factory=dict)
    text: str = ""

    def json(self) -> Any:
        return self.body


@dataclass
class FakeSession:
    """Routes GET/POST to user-supplied handlers; records every call (incl. kwargs)."""

    handlers: dict[tuple[str, str], Callable[[Any], FakeResponse]] = field(default_factory=dict)
    calls: list[tuple[str, str, dict]] = field(default_factory=list)

    def _dispatch(self, method: str, url: str, **kwargs) -> FakeResponse:
        for (m, path), handler in self.handlers.items():
            if m == method and url.endswith(path):
                self.calls.append((method, url, kwargs))
                return handler(kwargs)
        raise AssertionError(f"No handler for {method} {url}")

    def get(self, url: str, **kwargs) -> FakeResponse:
        return self._dispatch("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> FakeResponse:
        return self._dispatch("POST", url, **kwargs)


# ---------------------------------------------------------------------------
# wait_for_kibana
# ---------------------------------------------------------------------------

def test_wait_for_kibana_succeeds_when_available(monkeypatch):
    sess = FakeSession()
    sess.handlers[("GET", "/api/status")] = lambda _: FakeResponse(
        status_code=200,
        body={"status": {"overall": {"level": "available"}}},
    )
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", lambda _: None)
    elapsed = bootstrap_kibana.wait_for_kibana(
        "http://kibana:5601", timeout=5, poll_interval=0.01, session=sess
    )
    assert elapsed >= 0.0
    assert any(c[0] == "GET" and c[1].endswith("/api/status") for c in sess.calls)


def test_wait_for_kibana_polls_until_available(monkeypatch):
    sess = FakeSession()
    states = iter(["degraded", "degraded", "available"])
    sess.handlers[("GET", "/api/status")] = lambda _: FakeResponse(
        status_code=200,
        body={"status": {"overall": {"level": next(states)}}},
    )
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", lambda _: None)
    bootstrap_kibana.wait_for_kibana(
        "http://kibana:5601", timeout=5, poll_interval=0.01, session=sess
    )
    status_calls = [c for c in sess.calls if c[1].endswith("/api/status")]
    # Three GET /api/status calls — two "degraded" plus the one that returned available.
    assert len(status_calls) == 3


def test_wait_for_kibana_times_out(monkeypatch):
    sess = FakeSession()
    sess.handlers[("GET", "/api/status")] = lambda _: FakeResponse(
        status_code=200,
        body={"status": {"overall": {"level": "unavailable"}}},
    )
    # Advance the clock manually so we hit the timeout deterministically.
    t = [0.0]
    monkeypatch.setattr(bootstrap_kibana.time, "monotonic", lambda: t[0])

    def fake_sleep(_):
        t[0] += 1.0
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", fake_sleep)

    with pytest.raises(bootstrap_kibana.KibanaError, match="not ready"):
        bootstrap_kibana.wait_for_kibana(
            "http://kibana:5601", timeout=3, poll_interval=0.01, session=sess
        )


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def _ready_status_handler(_):
    return FakeResponse(200, {"status": {"overall": {"level": "available"}}})


def _existing_data_views_handler(titles: list[str]):
    def handler(_):
        return FakeResponse(
            200, {"data_view": [{"id": f"id-{i}", "title": t} for i, t in enumerate(titles)]}
        )
    return handler


def test_bootstrap_creates_missing_data_views(monkeypatch):
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", lambda _: None)
    posted: list[dict] = []

    def post_handler(kwargs):
        posted.append(json.loads(kwargs["data"]))
        return FakeResponse(200, {"data_view": {"id": "new-id"}})

    sess = FakeSession()
    sess.handlers[("GET", "/api/status")] = _ready_status_handler
    sess.handlers[("GET", "/api/data_views")] = _existing_data_views_handler([])
    sess.handlers[("POST", "/api/data_views/data_view")] = post_handler

    result = bootstrap_kibana.bootstrap("http://kibana:5601", timeout=5, session=sess)
    assert sorted(result.created) == ["logs-raw-*", "logs-scored"]
    assert result.already_existed == []
    assert len(posted) == 2
    titles = {p["data_view"]["title"] for p in posted}
    assert titles == {"logs-raw-*", "logs-scored"}
    # Both POSTs must carry the kbn-xsrf header — Kibana rejects writes without it.
    post_calls = [c for c in sess.calls if c[0] == "POST"]
    assert len(post_calls) == 2
    for _, _, kwargs in post_calls:
        assert kwargs["headers"]["kbn-xsrf"] == "true"
    # Each created data view uses @timestamp as the time field.
    for body in posted:
        assert body["data_view"]["timeFieldName"] == "@timestamp"


def test_bootstrap_skips_existing(monkeypatch):
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", lambda _: None)
    sess = FakeSession()
    sess.handlers[("GET", "/api/status")] = _ready_status_handler
    sess.handlers[("GET", "/api/data_views")] = _existing_data_views_handler(
        ["logs-raw-*", "logs-scored"]
    )

    def post_handler(_):
        raise AssertionError("POST should not be called when both data views already exist")
    sess.handlers[("POST", "/api/data_views/data_view")] = post_handler

    result = bootstrap_kibana.bootstrap("http://kibana:5601", timeout=5, session=sess)
    assert result.created == []
    assert sorted(result.already_existed) == ["logs-raw-*", "logs-scored"]


def test_bootstrap_creates_only_the_missing_one(monkeypatch):
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", lambda _: None)
    posted: list[dict] = []

    def post_handler(kwargs):
        posted.append(json.loads(kwargs["data"]))
        return FakeResponse(200, {"data_view": {"id": "new-id"}})

    sess = FakeSession()
    sess.handlers[("GET", "/api/status")] = _ready_status_handler
    sess.handlers[("GET", "/api/data_views")] = _existing_data_views_handler(["logs-raw-*"])
    sess.handlers[("POST", "/api/data_views/data_view")] = post_handler

    result = bootstrap_kibana.bootstrap("http://kibana:5601", timeout=5, session=sess)
    assert result.already_existed == ["logs-raw-*"]
    assert result.created == ["logs-scored"]
    assert len(posted) == 1
    assert posted[0]["data_view"]["title"] == "logs-scored"


def test_bootstrap_raises_on_data_view_creation_failure(monkeypatch):
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", lambda _: None)
    sess = FakeSession()
    sess.handlers[("GET", "/api/status")] = _ready_status_handler
    sess.handlers[("GET", "/api/data_views")] = _existing_data_views_handler([])
    sess.handlers[("POST", "/api/data_views/data_view")] = lambda _: FakeResponse(
        status_code=500, body={}, text="internal error"
    )

    with pytest.raises(bootstrap_kibana.KibanaError, match="HTTP 500"):
        bootstrap_kibana.bootstrap("http://kibana:5601", timeout=5, session=sess)


def test_bootstrap_raises_on_list_failure(monkeypatch):
    monkeypatch.setattr(bootstrap_kibana.time, "sleep", lambda _: None)
    sess = FakeSession()
    sess.handlers[("GET", "/api/status")] = _ready_status_handler
    sess.handlers[("GET", "/api/data_views")] = lambda _: FakeResponse(
        status_code=403, body={}, text="forbidden"
    )

    with pytest.raises(bootstrap_kibana.KibanaError, match="HTTP 403"):
        bootstrap_kibana.bootstrap("http://kibana:5601", timeout=5, session=sess)
