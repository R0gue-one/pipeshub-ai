"""
Focused crawl test: HTTP redirect deduplication for WebConnector.

Scenario
--------
A FastAPI test server exposes two routes:
  GET /start  → 302 redirect to /target
  GET /target → HTML page that links back to itself (<a href="/target">)

Expected behaviour
------------------
- The crawler visits 2 distinct URLs (/start and /target).
- Both fetches resolve to the same final URL (/target), so both produce a
  record with the same external_record_id (http://127.0.0.1:{port}/target/).
- Only 1 unique external_record_id is produced across all on_new_records calls.

No real infrastructure (database, Kafka, ArangoDB, Redis …) is required.
Every DB/messaging call is intercepted by an AsyncMock.
"""

import asyncio
from typing import Any, Callable, Dict

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

# ---------------------------------------------------------------------------
# Test HTML served at /target
# Contains a self-referencing link so the crawler discovers /target as a link
# to follow (depth+1), triggering the second HTTP fetch.
# ---------------------------------------------------------------------------
_TARGET_HTML = """\
<!DOCTYPE html>
<html>
  <head><title>Target Page</title></head>
  <body>
    <h1>Target</h1>
    <p>Some content here.</p>
    <a href="/target">Self link</a>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# Test FastAPI application
# ---------------------------------------------------------------------------

def make_test_app() -> FastAPI:
    """Return a FastAPI app with /start (redirect) and /target (HTML page)."""
    app = FastAPI()

    @app.get("/start")
    async def start_route():  # type: ignore[return]
        return RedirectResponse(url="/target", status_code=302)

    @app.get("/target")
    async def target_route():  # type: ignore[return]
        return HTMLResponse(content=_TARGET_HTML)

    return app


@pytest.fixture(scope="function")
def test_app() -> FastAPI:
    return make_test_app()


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def test_redirect_crawl_deduplicates_to_single_record(
    test_server_base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    patch_crawl_sleep,
) -> None:
    """
    2 pages are crawled (/start → redirect → /target, then /target directly),
    but both fetches produce the same external_record_id, so only 1 unique
    record identity exists across all on_new_records calls.

    No database or external services are required: all DataSourceEntitiesProcessor
    methods are replaced by AsyncMock objects.
    """
    asyncio.run(
        _run_redirect_test(
            base_url=test_server_base_url,
            web_connector_harness=web_connector_harness,
            build_sync_config=build_sync_config,
        )
    )


async def _run_redirect_test(base_url: str, web_connector_harness, build_sync_config: Callable[..., Dict[str, Any]]) -> None:
    start_url = f"{base_url}/start"
    expected_external_id = f"{base_url}/target/"
    web_harness = web_connector_harness
    test_web_connector = web_harness.connector
    web_harness.mock_config_service.get_config.return_value = build_sync_config(
        url=start_url,
        follow_external=False,
        # Must be False: if True, start_path_prefix = "/start/" and the
        # "/target" link would be filtered out before it could be queued.
        restrict_to_start_path=False,
    )

    try:
        await test_web_connector.init()
        await test_web_connector._crawl_recursive(start_url, depth=0)

        # ------------------------------------------------------------ assertions

        # 1. Both /start and /target were visited (2 distinct crawl entries)
        assert len(test_web_connector.visited_urls) == 2, (
            f"Expected 2 visited URLs, got {len(test_web_connector.visited_urls)}: "
            f"{test_web_connector.visited_urls}"
        )

        # 2. At least one record was collected
        assert web_harness.collected, "on_new_records was never called — no records were produced"

        # 3. All records share a single external_record_id (the redirect destination)
        external_ids = {record.external_record_id for record, _perms in web_harness.collected}
        assert len(external_ids) == 1, (
            f"Expected exactly 1 unique external_record_id, got {len(external_ids)}: "
            f"{external_ids}"
        )

        # 4. That unique id points to /target/ (trailing-slash normalised)
        assert expected_external_id in external_ids, (
            f"Expected external_record_id '{expected_external_id}', "
            f"but got: {external_ids}"
        )
    finally:
        await test_web_connector.cleanup()
