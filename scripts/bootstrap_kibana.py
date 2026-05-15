"""Bootstrap Kibana for the AI-Log-Detector demo.

Waits for Kibana to come up after ``docker compose up``, then creates two
data views (formerly "index patterns"):

* ``logs-raw``     -> matches the ``logs-raw-*`` indices Filebeat writes to
* ``logs-scored``  -> matches the ``logs-scored`` index the streaming scorer writes to

Both data views use ``@timestamp`` as the time field, so Kibana's Discover
and Lens UIs immediately understand them as time-series data.

The script is idempotent — re-running it is a no-op if the data views
already exist. It is also import-friendly: ``main()`` returns an exit code
so it can be called from a test or from ``scripts/run_demo.py``.

Usage:
    python scripts/bootstrap_kibana.py
    python scripts/bootstrap_kibana.py --kibana-url http://localhost:5601 --timeout 180
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests

# (data view title pattern, friendly name)
DATA_VIEWS: list[tuple[str, str]] = [
    ("logs-raw-*", "logs-raw"),
    ("logs-scored", "logs-scored"),
]


@dataclass
class BootstrapResult:
    waited_seconds: float
    created: list[str]
    already_existed: list[str]


class KibanaError(RuntimeError):
    """Raised when Kibana returns an unexpected response."""


def wait_for_kibana(
    base_url: str,
    *,
    timeout: float = 180.0,
    poll_interval: float = 3.0,
    session: requests.Session | None = None,
) -> float:
    """Poll ``GET /api/status`` until Kibana reports overall level=available.

    Returns the number of seconds spent waiting. Raises ``KibanaError`` if
    the timeout elapses first.
    """
    s = session or requests.Session()
    start = time.monotonic()
    last_state = "unknown"
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise KibanaError(
                f"Kibana not ready after {timeout:.0f}s (last state: {last_state!r})"
            )
        try:
            resp = s.get(f"{base_url}/api/status", timeout=5)
        except requests.RequestException as exc:
            last_state = f"connection error: {exc.__class__.__name__}"
            time.sleep(poll_interval)
            continue
        if resp.status_code == 200:
            body = resp.json()
            level = (
                body.get("status", {})
                .get("overall", {})
                .get("level")
            ) or body.get("status", {}).get("overall", {}).get("state")
            last_state = str(level)
            if level == "available":
                return elapsed
        else:
            last_state = f"HTTP {resp.status_code}"
        time.sleep(poll_interval)


def _list_data_view_titles(
    base_url: str, session: requests.Session
) -> set[str]:
    resp = session.get(f"{base_url}/api/data_views", timeout=10)
    if resp.status_code != 200:
        raise KibanaError(
            f"Could not list data views: HTTP {resp.status_code} - {resp.text[:200]}"
        )
    body = resp.json()
    return {dv.get("title") for dv in body.get("data_view", []) if dv.get("title")}


def _create_data_view(
    base_url: str, *, title: str, name: str, session: requests.Session
) -> dict[str, Any]:
    payload = {
        "data_view": {
            "title": title,
            "name": name,
            "timeFieldName": "@timestamp",
        },
        "override": False,
    }
    resp = session.post(
        f"{base_url}/api/data_views/data_view",
        headers={"kbn-xsrf": "true", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=15,
    )
    if resp.status_code not in (200, 201):
        raise KibanaError(
            f"Failed to create data view {title!r}: "
            f"HTTP {resp.status_code} - {resp.text[:300]}"
        )
    return resp.json()


def bootstrap(
    base_url: str,
    *,
    timeout: float = 180.0,
    session: requests.Session | None = None,
) -> BootstrapResult:
    """Wait for Kibana and ensure all DATA_VIEWS exist. Idempotent."""
    s = session or requests.Session()
    waited = wait_for_kibana(base_url, timeout=timeout, session=s)

    existing = _list_data_view_titles(base_url, s)
    created: list[str] = []
    already: list[str] = []
    for title, name in DATA_VIEWS:
        if title in existing:
            already.append(title)
            continue
        _create_data_view(base_url, title=title, name=name, session=s)
        created.append(title)

    return BootstrapResult(
        waited_seconds=waited, created=created, already_existed=already
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kibana-url", default="http://localhost:5601")
    ap.add_argument(
        "--timeout", type=float, default=180.0,
        help="Seconds to wait for Kibana to become available.",
    )
    args = ap.parse_args(argv)

    print(f"[kibana] waiting for {args.kibana_url} (up to {args.timeout:.0f}s)...")
    try:
        result = bootstrap(args.kibana_url, timeout=args.timeout)
    except KibanaError as exc:
        print(f"[kibana] error: {exc}", file=sys.stderr)
        return 2

    print(f"[kibana] ready after {result.waited_seconds:.1f}s")
    for title in result.created:
        print(f"[kibana] created data view: {title}")
    for title in result.already_existed:
        print(f"[kibana] data view already exists: {title}")
    print()
    print("Open Kibana:")
    print(f"  {args.kibana_url}/app/discover                 (Discover view)")
    print(f"  {args.kibana_url}/app/management/kibana/dataViews  (Data view manager)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
