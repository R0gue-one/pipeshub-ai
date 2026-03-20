import asyncio
from typing import Any, Callable, Dict
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from unittest.mock import AsyncMock, patch

from conftest import WebConnectorHarness 

def make_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/foo/bar/baz")
    async def foo_bar_route():  # type: ignore[return]
        return HTMLResponse(status_code=200)

    return app


@pytest.fixture(scope="function")
def test_app() -> FastAPI:
    return make_test_app()

@pytest.mark.parametrize(
    "expected_record_count, expected_paths",
    [
        (3, {"/foo/bar/baz/", "/foo/bar/", "/foo/"}),
    ],
)
def test_ensure_parent_records(
    test_server_base_url: str,
    web_connector_harness: WebConnectorHarness,
    build_sync_config: Callable[..., Dict[str, Any]],
    patch_crawl_sleep,
    expected_record_count: int,
    expected_paths: set[str],
) -> None:
    asyncio.run(
        _run_ensure_parent_records_case(
            base_url=test_server_base_url,
            web_connector_harness=web_connector_harness,
            build_sync_config=build_sync_config,
            expected_record_count=expected_record_count,
            expected_paths=expected_paths,
        )
    )


async def _run_ensure_parent_records_case(
    base_url: str,
    web_connector_harness: WebConnectorHarness,
    build_sync_config: Callable[..., Dict[str, Any]],
    expected_record_count: int,
    expected_paths: set[str],
) -> None:
    start_url = f"{base_url}/foo/bar/baz"
    test_web_connector = web_connector_harness.connector

    web_connector_harness.mock_config_service.get_config.return_value = build_sync_config(
        url=start_url,
        depth=2,
        max_pages=10,
    )
    fake_db: dict[str, Any] = {}

    async def _capture_and_store(records):
        # keep existing assertions behavior
        web_connector_harness.collected.extend(records)
        for record, _perms in records:
            fake_db[record.external_record_id] = record

    async def _lookup_record_by_external_id(*, connector_id, external_record_id):
        record = fake_db.get(external_record_id)
        if record:
            return record
        # support legacy no-trailing-slash lookup path too
        if external_record_id.endswith("/"):
            return fake_db.get(external_record_id.rstrip("/"))
        return fake_db.get(external_record_id + "/")

    web_connector_harness.mock_dep.on_new_records.side_effect = _capture_and_store
    web_connector_harness.mock_dep.get_record_by_external_id.side_effect = _lookup_record_by_external_id

    try:
        await test_web_connector.init()
        await test_web_connector._crawl_recursive(start_url, depth=0)

        assert len(web_connector_harness.collected) == expected_record_count, (
            f"Expected {expected_record_count} records with url_should_contain="
            f"got {len(web_connector_harness.collected)}"
        )

        record_paths = {urlparse(r.external_record_id).path for r, _ in web_connector_harness.collected}
        assert record_paths == expected_paths, (
            f"Expected record paths {expected_paths}, got {record_paths}"
        )
    finally:
        await test_web_connector.cleanup()

