"""
Focused crawl test: restrict_to_start_path behavior for WebConnector.

Scenario
--------
A FastAPI test server exposes:
  GET /docs/        -> HTML linking to /docs/inside and /outside
  GET /docs/inside  -> HTML page
  GET /outside      -> HTML page

Expected behaviour
------------------
- restrict_to_start_path=True:
    - Only URLs under the start path prefix (/docs/) are crawled.
    - /outside is skipped.
- restrict_to_start_path=False:
    - Both in-path and out-of-path links are crawled.
"""

import asyncio
from typing import Any, Callable, Dict
from unittest.mock import AsyncMock, patch
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


_DOCS_HTML = """\
<!DOCTYPE html>
<html>
  <head><title>Docs Root</title></head>
  <body>
    <a href="/docs/inside">Inside</a>
    <a href="/outside">Outside</a>
  </body>
</html>
"""

_INSIDE_HTML = """\
<!DOCTYPE html>
<html>
  <head><title>Inside</title></head>
  <body><p>Inside docs path.</p></body>
</html>
"""

_OUTSIDE_HTML = """\
<!DOCTYPE html>
<html>
  <head><title>Outside</title></head>
  <body><p>Outside docs path.</p></body>
</html>
"""


def make_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/docs/")
    async def docs_root():  # type: ignore[return]
        return HTMLResponse(content=_DOCS_HTML)

    @app.get("/docs/inside")
    async def docs_inside():  # type: ignore[return]
        return HTMLResponse(content=_INSIDE_HTML)

    @app.get("/outside")
    async def outside():  # type: ignore[return]
        return HTMLResponse(content=_OUTSIDE_HTML)

    return app


@pytest.fixture(scope="function")
def test_app() -> FastAPI:
    return make_test_app()


@pytest.mark.parametrize(
    "restrict_to_start_path, expected_visited, expected_record_count, expected_paths",
    [
        (True, 2, 2, {"/docs/", "/docs/inside"}),
        (False, 3, 3, {"/docs/", "/docs/inside", "/outside"}),
    ],
)
def test_restrict_to_start_path_filters_links(
    test_server_base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    patch_crawl_sleep,
    restrict_to_start_path: bool,
    expected_visited: int,
    expected_record_count: int,
    expected_paths: set[str],
) -> None:
    asyncio.run(
        _run_restrict_to_start_path_case(
            base_url=test_server_base_url,
            web_connector_harness=web_connector_harness,
            build_sync_config=build_sync_config,
            restrict_to_start_path=restrict_to_start_path,
            expected_visited=expected_visited,
            expected_record_count=expected_record_count,
            expected_paths=expected_paths,
        )
    )


async def _run_restrict_to_start_path_case(
    base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    restrict_to_start_path: bool,
    expected_visited: int,
    expected_record_count: int,
    expected_paths: set[str],
) -> None:
    start_url = f"{base_url}/docs/"
    web_harness = web_connector_harness
    test_web_connector = web_harness.connector
    web_harness.mock_config_service.get_config.return_value = build_sync_config(
        url=start_url,
        depth=2,
        max_pages=10,
        follow_external=False,
        restrict_to_start_path=restrict_to_start_path,
        url_should_contain=[],
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
            f"Expected {expected_visited} visited URLs with restrict_to_start_path="
            f"{restrict_to_start_path}, got {len(test_web_connector.visited_urls)}: "
            f"{test_web_connector.visited_urls}"
        )

        assert len(web_harness.collected) == expected_record_count, (
            f"Expected {expected_record_count} records with restrict_to_start_path="
            f"{restrict_to_start_path}, got {len(web_harness.collected)}"
        )

        record_paths = {urlparse(record.weburl).path for record, _perms in web_harness.collected}
        assert record_paths == expected_paths, (
            f"Expected record paths {expected_paths}, got {record_paths}"
        )
    finally:
        await test_web_connector.cleanup()
