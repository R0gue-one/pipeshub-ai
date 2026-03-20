"""
Focused crawl test: url_should_contain filtering for WebConnector.

Scenario
--------
A FastAPI test server exposes:
  GET /start          -> HTML linking to /allowed/page and /blocked/page
  GET /allowed/page   -> HTML page
  GET /blocked/page   -> HTML page

Expected behaviour
------------------
- url_should_contain=["allowed"]:
    - Start URL is still processed (special-case allow for configured start URL).
    - /allowed/page is processed.
    - /blocked/page is skipped by URL substring filter.
- url_should_contain=[]:
    - All three pages are processed.
"""

import asyncio
from typing import Any, Callable, Dict
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from unittest.mock import AsyncMock, patch


_START_HTML = """\
<!DOCTYPE html>
<html>
  <head><title>Start</title></head>
  <body>
    <a href="/allowed/page">Allowed</a>
    <a href="/blocked/page">Blocked</a>
  </body>
</html>
"""

_ALLOWED_HTML = """\
<!DOCTYPE html>
<html>
  <head><title>Allowed</title></head>
  <body><p>Allowed content.</p></body>
</html>
"""

_BLOCKED_HTML = """\
<!DOCTYPE html>
<html>
  <head><title>Blocked</title></head>
  <body><p>Blocked content.</p></body>
</html>
"""


def make_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/start")
    async def start_route():  # type: ignore[return]
        return HTMLResponse(content=_START_HTML)

    @app.get("/allowed/page")
    async def allowed_page():  # type: ignore[return]
        return HTMLResponse(content=_ALLOWED_HTML)

    @app.get("/blocked/page")
    async def blocked_page():  # type: ignore[return]
        return HTMLResponse(content=_BLOCKED_HTML)

    return app


@pytest.fixture(scope="function")
def test_app() -> FastAPI:
    return make_test_app()


@pytest.mark.parametrize(
    "url_should_contain, expected_visited, expected_record_count, expected_paths",
    [
        (["allowed"], 3, 2, {"/start", "/allowed/page"}),
        ([], 3, 3, {"/start", "/allowed/page", "/blocked/page"}),
    ],
)
def test_url_should_contain_filters_non_matching_urls(
    test_server_base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    patch_crawl_sleep,
    url_should_contain: list[str],
    expected_visited: int,
    expected_record_count: int,
    expected_paths: set[str],
) -> None:
    asyncio.run(
        _run_url_should_contain_case(
            base_url=test_server_base_url,
            web_connector_harness=web_connector_harness,
            build_sync_config=build_sync_config,
            url_should_contain=url_should_contain,
            expected_visited=expected_visited,
            expected_record_count=expected_record_count,
            expected_paths=expected_paths,
        )
    )


async def _run_url_should_contain_case(
    base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    url_should_contain: list[str],
    expected_visited: int,
    expected_record_count: int,
    expected_paths: set[str],
) -> None:
    start_url = f"{base_url}/start"
    web_harness = web_connector_harness
    test_web_connector = web_harness.connector
    web_harness.mock_config_service.get_config.return_value = build_sync_config(
        url=start_url,
        depth=2,
        max_pages=10,
        follow_external=False,
        restrict_to_start_path=False,
        url_should_contain=url_should_contain,
    )

    try:
        with patch.object(
            test_web_connector,
            "_ensure_parent_records_exist",
            new=AsyncMock(return_value=None),
        ):
            await test_web_connector.init()
            await test_web_connector._crawl_recursive(start_url, depth=0)

        assert len(test_web_connector.visited_urls) == expected_visited, (
            f"Expected {expected_visited} visited URLs with url_should_contain="
            f"{url_should_contain}, got {len(test_web_connector.visited_urls)}: "
            f"{test_web_connector.visited_urls}"
        )

        assert len(web_harness.collected) == expected_record_count, (
            f"Expected {expected_record_count} records with url_should_contain="
            f"{url_should_contain}, got {len(web_harness.collected)}"
        )

        record_paths = {urlparse(record.weburl).path for record, _perms in web_harness.collected}
        assert record_paths == expected_paths, (
            f"Expected record paths {expected_paths}, got {record_paths}"
        )
    finally:
        await test_web_connector.cleanup()
