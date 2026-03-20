import asyncio
from typing import Any, Callable, Dict

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config.constants.arangodb import Connectors, OriginTypes
from app.models.entities import FileRecord, RecordType


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
    """Return a FastAPI app with /target (HTML page)."""
    app = FastAPI()

    @app.get("/target")
    async def target_route():  # type: ignore[return]
        return HTMLResponse(content=_TARGET_HTML)

    return app


@pytest.fixture(scope="function")
def test_app() -> FastAPI:
    return make_test_app()


def test_stream_record(
    test_server_base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
    patch_crawl_sleep,
) -> None:
    """
    Content is streamed from the /target page.
    stream_record should return a StreamingResponse with non-empty response_content.
    """
    asyncio.run(
        _run_stream_record_test(
            base_url=test_server_base_url,
            web_connector_harness=web_connector_harness,
            build_sync_config=build_sync_config,
        )
    )


async def _run_stream_record_test(
    base_url: str,
    web_connector_harness,
    build_sync_config: Callable[..., Dict[str, Any]],
) -> None:
    start_url = f"{base_url}/target"
    test_web_connector = web_connector_harness.connector
    web_connector_harness.mock_config_service.get_config.return_value = build_sync_config(
        url=start_url,
    )

    record = FileRecord(
        id="target_record_id",
        org_id="test_org_id",
        record_name="Test",
        record_type=RecordType.WEBPAGE,
        external_record_id=start_url,
        version=1,
        origin=OriginTypes.CONNECTOR,
        connector_name=Connectors.WEB,
        connector_id="test-connector-id",
        weburl=start_url,
        mime_type="text/html",
        is_file=False,
    )

    try:
        await test_web_connector.init()
        response = await test_web_connector.stream_record(record)

        # 1. stream_record should return a StreamingResponse
        assert isinstance(response, StreamingResponse), (
            f"Expected StreamingResponse, got {type(response)}"
        )

        # 2. response_content should be present (non-empty body)
        body_chunks = []
        async for chunk in response.body_iterator:
            body_chunks.append(
                chunk.encode("utf-8") if isinstance(chunk, str) else chunk
            )
        body = b"".join(body_chunks)
        assert len(body) > 0, "Expected non-empty response content from stream_record"
    finally:
        await test_web_connector.cleanup()
