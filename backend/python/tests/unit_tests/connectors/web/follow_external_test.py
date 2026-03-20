"""
Focused crawl test: external redirect handling for follow_external flag.

Scenario
--------
A local FastAPI test server exposes:
  GET /start -> 302 redirect to https://www.google.com

This test intentionally uses the real fetch stack
(`fetch_url_with_fallback`) to verify redirect behavior end-to-end.

Expected behaviour
------------------
- follow_external=False:
    - Redirect crossing to google.com is skipped.
    - No visited URL is recorded and no record is produced.
- follow_external=True:
    - Redirect is accepted and processed.
    - One page is visited/recorded (crawl is capped to 1 page for determinism).
    - All produced records have weburl on a single domain (www.google.com).
"""

import asyncio
from typing import Any, Callable, Dict
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.responses import RedirectResponse


_GOOGLE_ROOT = "https://www.google.com"


def make_test_app() -> FastAPI:
    """Return FastAPI app with one route that redirects to google.com."""
    app = FastAPI()

    @app.get("/start")
    async def start_route():  # type: ignore[return]
        return RedirectResponse(url=_GOOGLE_ROOT, status_code=302)

    return app


@pytest.fixture(scope="function")
def test_app() -> FastAPI:
    return make_test_app()


@pytest.mark.parametrize(
    "follow_external, expected_visited, expected_record_count",
    [
        (False, 1, 0),
        (True, 2, 1),
    ],
)
def test_external_redirect_respects_follow_external(
    test_server_base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    patch_crawl_sleep,
    follow_external: bool,
    expected_visited: int,
    expected_record_count: int,
) -> None:
    asyncio.run(
        _run_external_redirect_case(
            base_url=test_server_base_url,
            web_connector_harness=web_connector_harness,
            build_sync_config=build_sync_config,
            follow_external=follow_external,
            expected_visited=expected_visited,
            expected_record_count=expected_record_count,
        )
    )


async def _run_external_redirect_case(
    base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    follow_external: bool,
    expected_visited: int,
    expected_record_count: int,
) -> None:
    start_url = f"{base_url}/start"
    web_harness = web_connector_harness
    test_web_connector = web_harness.connector
    web_harness.mock_config_service.get_config.return_value = build_sync_config(
        url=start_url,
        max_pages=1,
        follow_external=follow_external,
        restrict_to_start_path=False,
    )

    try:
        await test_web_connector.init()
        await test_web_connector._crawl_recursive(start_url, depth=0)

        assert len(test_web_connector.visited_urls) == expected_visited, (
            f"Expected {expected_visited} visited URLs with follow_external={follow_external}, "
            f"got {len(test_web_connector.visited_urls)}: {test_web_connector.visited_urls}"
        )

        assert len(web_harness.collected) == expected_record_count, (
            f"Expected {expected_record_count} records with follow_external={follow_external}, "
            f"got {len(web_harness.collected)}"
        )

        if follow_external:
            web_domains = {
                urlparse(record.weburl).netloc.lower()
                for record, _perms in web_harness.collected
            }
            assert web_domains == {"www.google.com"}, (
                f"Expected all record.weburl domains to be www.google.com, got {web_domains}"
            )
    finally:
        await test_web_connector.cleanup()
