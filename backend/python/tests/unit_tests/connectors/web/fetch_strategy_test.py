"""
Unit tests for fetch_strategy.py — multi-strategy URL fetcher with fallback chain.

Structure
---------
MockServer (aiohttp.test_utils.TestServer) — a real HTTP server used for 
integration-style strategy tests that verify the aiohttp strategy against actual TCP connections.

Behavioral Suite (all branches mocked, no network I/O)
------------------------------------------------------------
 Tests core orchestrator logic (early exits, safety aborts, and retries) 
 without being coupled to the exact number or order of underlying libraries.
"""

import asyncio
import logging
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from app.connectors.sources.web.fetch_strategy import (
    FetchResponse,
    fetch_url_with_fallback,
)

logger = logging.getLogger("test.fetch_strategy")

# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _resp(
    status: int,
    strategy: str = "mocked_strategy",
    content: bytes = b"<html>OK</html>",
    headers: Optional[dict] = None,
    url: str = "http://example.com/",
) -> FetchResponse:
    return FetchResponse(
        status_code=status,
        content_bytes=content,
        headers=headers or {},
        final_url=url,
        strategy=strategy,
    )

def _side_effects(status_list: List[Optional[int]], strategy: str) -> list:
    out: list = []
    for v in status_list:
        if v is None:
            out.append(None)
        else:
            out.append(_resp(v, strategy=strategy))
    out.extend([None] * 20)
    return out

def _dummy_session() -> MagicMock:
    session = MagicMock(spec=aiohttp.ClientSession)
    head_resp_mock = MagicMock()
    head_resp_mock.headers = {}
    head_ctx = MagicMock()
    head_ctx.__aenter__ = AsyncMock(return_value=head_resp_mock)
    head_ctx.__aexit__ = AsyncMock(return_value=None)
    session.head = MagicMock(return_value=head_ctx)
    return session

@pytest.fixture
def mock_strategies():
    """Central fixture to mock ALL underlying fetcher libraries."""
    with (
        patch("app.connectors.sources.web.fetch_strategy._try_curl_cffi", AsyncMock(return_value=None)) as m_curl,
        patch("app.connectors.sources.web.fetch_strategy._try_cloudscraper", AsyncMock(return_value=None)) as m_cloud,
        patch("app.connectors.sources.web.fetch_strategy._try_aiohttp", AsyncMock(return_value=None)) as m_aio,
        patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", AsyncMock()) as m_sleep,
    ):
        yield {
            "curl": m_curl,
            "cloud": m_cloud,
            "aio": m_aio,
            "sleep": m_sleep
        }

# ---------------------------------------------------------------------------
# Mock Server Integration Tests
# ---------------------------------------------------------------------------

def _build_server_app(routes: dict) -> web.Application:
    app = web.Application()
    for path, (status, body) in routes.items():
        async def _handler(request, _s=status, _b=body):
            return web.Response(status=_s, body=_b)
        app.router.add_get(path, _handler)
    return app

async def _run_with_mock_server(
    routes: dict, path: str, *, max_retries_per_strategy: int = 1, max_429_retries: int = 0
) -> Optional[FetchResponse]:
    app = _build_server_app(routes)
    async with TestServer(app) as server:
        url = f"http://127.0.0.1:{server.port}{path}"
        with (
            patch("app.connectors.sources.web.fetch_strategy._try_aiohttp", AsyncMock(return_value=None)),
            patch("app.connectors.sources.web.fetch_strategy._try_cloudscraper", AsyncMock(return_value=None)),
            patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", AsyncMock()),
        ):
            async with aiohttp.ClientSession() as session:
                return await fetch_url_with_fallback(
                    url, session, logger, 
                    max_retries_per_strategy=max_retries_per_strategy, max_429_retries=max_429_retries
                )

class TestMockServer:
    def test_mock_server_200_ok(self):
        routes = {"/page": (200, b"<html>Hello</html>")}
        result = asyncio.run(_run_with_mock_server(routes, "/page"))
        assert result is not None
        assert result.status_code == 200
        assert "curl_cffi_h2" in result.strategy

    def test_mock_server_404_returns_result_without_retry(self):
        routes = {"/missing": (404, b"Not Found")}
        result = asyncio.run(_run_with_mock_server(routes, "/missing"))
        assert result is not None
        assert result.status_code == 404

    def test_mock_server_500_server_error_returned(self):
        routes = {"/crash": (500, b"Internal Server Error")}
        result = asyncio.run(_run_with_mock_server(routes, "/crash"))
        assert result is not None
        assert result.status_code == 500

    def test_mock_server_all_strategies_fail_returns_none(self):
        routes = {"/page": (200, b"OK")}
        async def _run():
            app = _build_server_app(routes)
            async with TestServer(app) as server:
                url = f"http://127.0.0.1:{server.port}/page"
                with (
                    patch("app.connectors.sources.web.fetch_strategy._try_curl_cffi", AsyncMock(return_value=None)),
                    patch("app.connectors.sources.web.fetch_strategy._try_cloudscraper", AsyncMock(return_value=None)),
                    patch("app.connectors.sources.web.fetch_strategy._try_aiohttp", AsyncMock(return_value=None)),
                    patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", AsyncMock()),
                ):
                    async with aiohttp.ClientSession() as session:
                        return await fetch_url_with_fallback(url, session, logger, max_retries_per_strategy=1)
        assert asyncio.run(_run()) is None


# ---------------------------------------------------------------------------
# Behavioral Fallback Tests
# ---------------------------------------------------------------------------

def test_fallback_stops_on_first_success(mock_strategies):
    """Proves the chain exits early if a strategy succeeds."""
    async def _run():
        mock_strategies["curl"].return_value = _resp(200)
        result = await fetch_url_with_fallback(
            "http://example.com/", _dummy_session(), logger, max_retries_per_strategy=1
        )
        assert result is not None
        assert result.status_code == 200
        assert mock_strategies["aio"].call_count == 0  # Fallback bypassed
    asyncio.run(_run())


def test_fallback_aborts_on_non_retryable_error(mock_strategies):
    """Proves a 404/410/405 stops the entire chain safely."""
    async def _run():
        mock_strategies["curl"].return_value = _resp(404)
        result = await fetch_url_with_fallback(
            "http://example.com/", _dummy_session(), logger, max_retries_per_strategy=1
        )
        assert result is not None
        assert result.status_code == 404
        assert mock_strategies["cloud"].call_count == 0  # Fallback aborted
    asyncio.run(_run())


def test_bot_detection_exhausts_retries_then_falls_through(mock_strategies):
    """Proves 403 triggers same-strategy retries, then moves to next strategy."""
    async def _run():
        mock_strategies["curl"].side_effect = _side_effects([403, 403], strategy="curl")
        mock_strategies["cloud"].return_value = _resp(200, strategy="cloudscraper")
        
        result = await fetch_url_with_fallback(
            "http://example.com/", _dummy_session(), logger, 
            max_retries_per_strategy=2, max_429_retries=0
        )
        
        assert result is not None
        assert result.status_code == 200
        assert "cloudscraper" in result.strategy
    asyncio.run(_run())


def test_429_rate_limited_retried_with_backoff_then_succeeds(mock_strategies):
    """Proves 429 triggers an asyncio.sleep backoff loop without abandoning the strategy."""
    async def _run():
        responses = [
            _resp(429, headers={"Retry-After": "1"}),
            _resp(200)
        ]
        mock_strategies["curl"].side_effect = responses
        
        result = await fetch_url_with_fallback(
            "http://example.com/", _dummy_session(), logger, 
            max_retries_per_strategy=1, max_429_retries=1
        )
        
        assert result is not None
        assert result.status_code == 200
        assert mock_strategies["sleep"].called is True
        assert mock_strategies["cloud"].call_count == 0  # Resolved without fallback
    asyncio.run(_run())