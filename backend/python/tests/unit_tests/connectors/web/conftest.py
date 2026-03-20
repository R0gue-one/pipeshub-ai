import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import uvicorn
from fastapi import FastAPI

from app.connectors.sources.web.connector import WebConnector


def _get_free_port() -> int:
    """Return a free TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    """Block until TCP port accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Server on {host}:{port} did not start within {timeout}s")


@pytest.fixture(scope="function")
def test_server_base_url(test_app: FastAPI):
    """Run the provided FastAPI app on an ephemeral local uvicorn server."""
    port = _get_free_port()
    config = uvicorn.Config(test_app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_port("127.0.0.1", port)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@dataclass
class WebConnectorHarness:
    connector: WebConnector
    mock_config_service: AsyncMock
    mock_dep: AsyncMock
    mock_store: MagicMock
    collected: List[Tuple[Any, Any]]


@pytest.fixture(scope="function")
def web_connector_harness() -> WebConnectorHarness:
    """Create a WebConnector plus mocked dependencies and record capture list."""
    mock_config_service = AsyncMock()
    mock_dep = AsyncMock()
    mock_dep.org_id = "test-org"
    mock_store = MagicMock()

    collected: List[Tuple[Any, Any]] = []
    fake_db: dict[str, Any] = {}

    async def _capture_and_store(records):
        # keep existing assertions behavior
        collected.extend(records)
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

    mock_dep.on_new_records.side_effect = _capture_and_store
    mock_dep.get_record_by_external_id.side_effect = _lookup_record_by_external_id

    connector = WebConnector(
        logger=logging.getLogger("test.web_connector"),
        data_entities_processor=mock_dep,
        data_store_provider=mock_store,
        config_service=mock_config_service,
        connector_id="test-connector-id",
    )

    return WebConnectorHarness(
        connector=connector,
        mock_config_service=mock_config_service,
        mock_dep=mock_dep,
        mock_store=mock_store,
        collected=collected,
    )


@pytest.fixture(scope="function")
def build_sync_config() -> Callable[..., Dict[str, Any]]:
    """Return a helper that builds the standard web connector sync config payload."""

    def _build(url: str, **overrides: Any) -> Dict[str, Any]:
        sync: Dict[str, Any] = {
            "url": url,
            "type": "recursive",
            "depth": 2,
            "max_pages": 10,
            "max_size_mb": 10,
            "follow_external": False,
            "restrict_to_start_path": False,
            "url_should_contain": [],
        }
        sync.update(overrides)
        return {"sync": sync}

    return _build


@pytest.fixture(scope="function")
def patch_crawl_sleep():
    """Patch asyncio.sleep to remove crawl delays in tests."""
    with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        yield
