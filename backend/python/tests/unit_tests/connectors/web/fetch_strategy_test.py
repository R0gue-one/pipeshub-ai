"""
Unit tests for fetch_strategy.py — multi-strategy URL fetcher with fallback chain.

Structure
---------
MockServer (aiohttp.test_utils.TestServer) — a lightweight real HTTP server
that returns configurable responses; used for integration-style strategy tests
that verify the aiohttp strategy against actual TCP connections.

Parametric test suite (all branches mocked, no network I/O)
------------------------------------------------------------
 1. Strategy-1 (curl_cffi H2) succeeds     → returns result, no fallback
 2. Strategy-1 fails (None)                → Strategy-2 (cloudscraper) succeeds
 3. Strategy-1 & -2 fail                   → Strategy-3 (aiohttp) succeeds
 4. All 3 strategies fail                  → returns None
 5a. Strategy-1 returns 404 (non-retryable)→ chain stops immediately
 5b. Strategy-1 returns 410                → chain stops immediately
 5c. Strategy-1 returns 405                → chain stops immediately
 6. Strategy-1 returns 403 (bot-block)     → exhausts per-strategy retries,
                                             then falls through to strategy-2
 7. Strategy-1 returns 429 (rate-limited)  → retried with back-off, succeeds
                                             on 2nd call within same strategy

All parametric tests run through asyncio.run() to stay consistent with the
rest of the test suite (no pytest-asyncio required).
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
# Response factory helpers
# ---------------------------------------------------------------------------

def _resp(
    status: int,
    strategy: str = "curl_cffi(chrome131, h2=True)",
    content: bytes = b"<html>OK</html>",
    headers: Optional[dict] = None,
    url: str = "http://example.com/",
) -> FetchResponse:
    """Build a FetchResponse with sensible defaults."""
    return FetchResponse(
        status_code=status,
        content_bytes=content,
        headers=headers or {},
        final_url=url,
        strategy=strategy,
    )


def _side_effects(status_list: List[Optional[int]], strategy: str) -> list:
    """
    Convert a list like [None, 200] into a list of FetchResponse / None
    suitable for AsyncMock(side_effect=...).

    ``None`` in the list means the strategy returned None (network/import
    failure); an integer means the strategy returned that HTTP status.
    """
    out: list = []
    for v in status_list:
        if v is None:
            out.append(None)
        else:
            out.append(_resp(v, strategy=strategy))
    # Pad with None so the mock never raises StopIteration if called extra times
    out.extend([None] * 20)
    return out


# ---------------------------------------------------------------------------
# Dummy aiohttp.ClientSession stub (used where the real session is irrelevant
# because _try_aiohttp is also mocked)
# ---------------------------------------------------------------------------

def _dummy_session() -> MagicMock:
    """
    MagicMock that satisfies ``session.head(...)`` as an async context
    manager. Only needed when max_size_mb is set; omitting max_size_mb
    makes the HEAD call path unreachable, but we still provide the stub
    to avoid AttributeError surprises.
    """
    session = MagicMock(spec=aiohttp.ClientSession)
    head_resp_mock = MagicMock()
    head_resp_mock.headers = {}
    head_ctx = MagicMock()
    head_ctx.__aenter__ = AsyncMock(return_value=head_resp_mock)
    head_ctx.__aexit__ = AsyncMock(return_value=None)
    session.head = MagicMock(return_value=head_ctx)
    return session


# ---------------------------------------------------------------------------
# Mock server (aiohttp.test_utils.TestServer)
#
# Used in integration-style tests where we want the aiohttp strategy to
# exercise real TCP socket code. curl_cffi and cloudscraper are mocked out
# so the fallback chain always reaches the aiohttp strategy.
# ---------------------------------------------------------------------------

def _build_server_app(routes: dict) -> web.Application:
    """
    Build an aiohttp.web.Application from a mapping:
        { "/path": (http_status_code, body_bytes) }
    """
    app = web.Application()
    for path, (status, body) in routes.items():

        async def _handler(request, _s=status, _b=body):
            return web.Response(status=_s, body=_b)

        app.router.add_get(path, _handler)
    return app


async def _run_with_mock_server(
    routes: dict,
    path: str,
    *,
    max_retries_per_strategy: int = 1,
    max_429_retries: int = 0,
) -> Optional[FetchResponse]:
    """
    Spin up a real aiohttp TestServer, force the fallback chain to reach
    the aiohttp strategy, fetch ``path``, and return the FetchResponse.
    """
    app = _build_server_app(routes)

    async with TestServer(app) as server:
        url = f"http://127.0.0.1:{server.port}{path}"

        # Force only the aiohttp strategy by mocking the others to None
        with (
            patch(
                "app.connectors.sources.web.fetch_strategy._try_curl_cffi",
                AsyncMock(return_value=None),
            ),
            patch(
                "app.connectors.sources.web.fetch_strategy._try_cloudscraper",
                AsyncMock(return_value=None),
            ),
            patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", AsyncMock()),
        ):
            async with aiohttp.ClientSession() as session:
                return await fetch_url_with_fallback(
                    url,
                    session,
                    logger,
                    max_retries_per_strategy=max_retries_per_strategy,
                    max_429_retries=max_429_retries,
                )


# ========================= MOCK SERVER TESTS ================================

class TestMockServer:
    """
    Integration tests using a real local HTTP server.
    The curl_cffi and cloudscraper strategies are stubbed to None so that
    only the aiohttp strategy is exercised against the live socket.
    """

    def test_mock_server_200_ok(self):
        """aiohttp strategy successfully fetches a 200 OK response."""
        routes = {"/page": (200, b"<html>Hello</html>")}
        result = asyncio.run(_run_with_mock_server(routes, "/page"))

        assert result is not None, "Expected a FetchResponse, got None"
        assert result.status_code == 200
        assert result.content_bytes == b"<html>Hello</html>"
        assert "aiohttp" in result.strategy

    def test_mock_server_404_returns_result_without_retry(self):
        """
        404 is a non-retryable client error: the result is returned
        immediately (no further strategy attempts).
        """
        routes = {"/missing": (404, b"Not Found")}
        result = asyncio.run(_run_with_mock_server(routes, "/missing"))

        assert result is not None, "Expected a FetchResponse for 404, got None"
        assert result.status_code == 404
        assert "aiohttp" in result.strategy

    def test_mock_server_500_server_error_returned(self):
        """
        5xx server errors are returned immediately (non-retryable within
        a single strategy attempt).
        """
        routes = {"/crash": (500, b"Internal Server Error")}
        result = asyncio.run(_run_with_mock_server(routes, "/crash"))

        assert result is not None
        assert result.status_code == 500

    def test_mock_server_redirect_follows_to_final_url(self):
        """
        aiohttp follows HTTP 301/302 redirects; final_url reflects the
        resolved destination.
        """
        app = web.Application()

        async def _redirect(request):
            raise web.HTTPFound(location="/final")

        async def _final(request):
            return web.Response(status=200, body=b"Final page")

        app.router.add_get("/start", _redirect)
        app.router.add_get("/final", _final)

        async def _run():
            async with TestServer(app) as server:
                url = f"http://127.0.0.1:{server.port}/start"
                with (
                    patch(
                        "app.connectors.sources.web.fetch_strategy._try_curl_cffi",
                        AsyncMock(return_value=None),
                    ),
                    patch(
                        "app.connectors.sources.web.fetch_strategy._try_cloudscraper",
                        AsyncMock(return_value=None),
                    ),
                    patch(
                        "app.connectors.sources.web.fetch_strategy.asyncio.sleep",
                        AsyncMock(),
                    ),
                ):
                    async with aiohttp.ClientSession() as session:
                        return await fetch_url_with_fallback(
                            url, session, logger,
                            max_retries_per_strategy=1,
                            max_429_retries=0,
                        )

        result = asyncio.run(_run())
        assert result is not None
        assert result.status_code == 200
        assert result.final_url.endswith("/final")

    def test_mock_server_all_strategies_fail_returns_none(self):
        """
        When the server actively refuses (503) and all strategies return None,
        fetch_url_with_fallback returns None.
        """
        routes = {"/page": (200, b"OK")}

        async def _run():
            app = _build_server_app(routes)
            async with TestServer(app) as server:
                url = f"http://127.0.0.1:{server.port}/page"
                with (
                    patch(
                        "app.connectors.sources.web.fetch_strategy._try_curl_cffi",
                        AsyncMock(return_value=None),
                    ),
                    patch(
                        "app.connectors.sources.web.fetch_strategy._try_cloudscraper",
                        AsyncMock(return_value=None),
                    ),
                    patch(
                        "app.connectors.sources.web.fetch_strategy._try_aiohttp",
                        AsyncMock(return_value=None),
                    ),
                    patch(
                        "app.connectors.sources.web.fetch_strategy.asyncio.sleep",
                        AsyncMock(),
                    ),
                ):
                    async with aiohttp.ClientSession() as session:
                        return await fetch_url_with_fallback(
                            url, session, logger,
                            max_retries_per_strategy=1,
                            max_429_retries=0,
                        )

        result = asyncio.run(_run())
        assert result is None


# ======================== PARAMETRIC FALLBACK TESTS =========================
#
# All three strategy functions are replaced with AsyncMocks — no network I/O.
# max_retries_per_strategy=1 keeps call counts deterministic (each strategy
# is attempted exactly once before advancing to the next one).
# ---------------------------------------------------------------------------

_FALLBACK_CASES = [
    # curl_returns, cloud_returns, aio_returns,
    # exp_status, exp_strategy_fragment,
    # exp_curl_calls, exp_cloud_calls, exp_aio_calls
    #
    # 1. Strategy-1 (curl_cffi) succeeds → no fallback at all
    ([200],   [],     [],     200,  "curl_cffi",    1, 0, 0),
    # 2. Strategy-1 returns None → Strategy-2 (cloudscraper) succeeds
    ([None],  [200],  [],     200,  "cloudscraper", 1, 1, 0),
    # 3. Strategy-1 & -2 return None → Strategy-3 (aiohttp) succeeds
    ([None],  [None], [200],  200,  "aiohttp",      1, 1, 1),
    # 4. All 3 strategies return None → result is None
    ([None],  [None], [None], None, None,            1, 1, 1),
    # 5a. curl_cffi returns 404 (non-retryable) → chain stops immediately
    ([404],   [],     [],     404,  "curl_cffi",    1, 0, 0),
    # 5b. curl_cffi returns 410 (Gone) → chain stops immediately
    ([410],   [],     [],     410,  "curl_cffi",    1, 0, 0),
    # 5c. curl_cffi returns 405 (Method Not Allowed) → chain stops immediately
    ([405],   [],     [],     405,  "curl_cffi",    1, 0, 0),
]

_FALLBACK_IDS = [
    "strategy_1_succeeds",
    "strategy_1_fails__strategy_2_succeeds",
    "strategy_1_2_fail__strategy_3_succeeds",
    "all_strategies_fail__returns_none",
    "http_404_stops_chain",
    "http_410_stops_chain",
    "http_405_stops_chain",
]


async def _run_fallback_case(
    curl_returns,
    cloud_returns,
    aio_returns,
    exp_status,
    exp_strategy_frag,
    exp_curl_n,
    exp_cloud_n,
    exp_aio_n,
):
    mock_curl = AsyncMock(
        side_effect=_side_effects(curl_returns, strategy="curl_cffi(chrome131, h2=True)")
    )
    mock_cloud = AsyncMock(
        side_effect=_side_effects(cloud_returns, strategy="cloudscraper")
    )
    mock_aio = AsyncMock(
        side_effect=_side_effects(aio_returns, strategy="aiohttp")
    )

    with (
        patch("app.connectors.sources.web.fetch_strategy._try_curl_cffi", mock_curl),
        patch("app.connectors.sources.web.fetch_strategy._try_cloudscraper", mock_cloud),
        patch("app.connectors.sources.web.fetch_strategy._try_aiohttp", mock_aio),
        patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", AsyncMock()),
    ):
        result = await fetch_url_with_fallback(
            "http://example.com/test",
            _dummy_session(),
            logger,
            max_retries_per_strategy=1,
            max_429_retries=0,
        )

    # --- result assertions ---
    if exp_status is None:
        assert result is None, (
            f"Expected None but got FetchResponse(status={result.status_code if result else '?'}, "
            f"strategy={result.strategy if result else '?'})"
        )
    else:
        assert result is not None, "Expected a FetchResponse but got None"
        assert result.status_code == exp_status, (
            f"Expected status {exp_status}, got {result.status_code}"
        )
        if exp_strategy_frag:
            assert exp_strategy_frag in result.strategy, (
                f"Expected strategy to contain '{exp_strategy_frag}', "
                f"got '{result.strategy}'"
            )

    # --- call-count assertions ---
    assert mock_curl.call_count == exp_curl_n, (
        f"curl_cffi called {mock_curl.call_count} times, expected {exp_curl_n}"
    )
    assert mock_cloud.call_count == exp_cloud_n, (
        f"cloudscraper called {mock_cloud.call_count} times, expected {exp_cloud_n}"
    )
    assert mock_aio.call_count == exp_aio_n, (
        f"aiohttp called {mock_aio.call_count} times, expected {exp_aio_n}"
    )


@pytest.mark.parametrize(
    "curl_returns, cloud_returns, aio_returns, "
    "exp_status, exp_strategy_frag, "
    "exp_curl_n, exp_cloud_n, exp_aio_n",
    _FALLBACK_CASES,
    ids=_FALLBACK_IDS,
)
def test_strategy_fallback_chain(
    curl_returns,
    cloud_returns,
    aio_returns,
    exp_status,
    exp_strategy_frag,
    exp_curl_n,
    exp_cloud_n,
    exp_aio_n,
):
    """
    Parametric test covering the complete strategy fallback chain.
    Each strategy function is replaced by an AsyncMock so tests run
    entirely in-process with no network I/O.
    """
    asyncio.run(
        _run_fallback_case(
            curl_returns,
            cloud_returns,
            aio_returns,
            exp_status,
            exp_strategy_frag,
            exp_curl_n,
            exp_cloud_n,
            exp_aio_n,
        )
    )


# ======================== BOT-DETECTION RETRY TEST ==========================

def test_bot_detection_exhausts_retries_then_falls_through():
    """
    Strategy-1 returns 403 (bot-block) on every attempt.
    With max_retries_per_strategy=2, curl_cffi is called twice (once per
    attempt) before the chain advances to cloudscraper, which succeeds.
    """

    async def _run():
        # 403 twice for curl_cffi (one per strategy attempt)
        curl_403_twice = _side_effects([403, 403], strategy="curl_cffi(chrome131, h2=True)")
        mock_curl = AsyncMock(side_effect=curl_403_twice)
        mock_cloud = AsyncMock(return_value=_resp(200, strategy="cloudscraper"))
        mock_aio = AsyncMock(return_value=None)

        with (
            patch("app.connectors.sources.web.fetch_strategy._try_curl_cffi", mock_curl),
            patch("app.connectors.sources.web.fetch_strategy._try_cloudscraper", mock_cloud),
            patch("app.connectors.sources.web.fetch_strategy._try_aiohttp", mock_aio),
            patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", AsyncMock()),
        ):
            result = await fetch_url_with_fallback(
                "http://example.com/",
                _dummy_session(),
                logger,
                max_retries_per_strategy=2,  # <-- two attempts per strategy
                max_429_retries=0,
            )

        assert result is not None
        assert result.status_code == 200
        assert "cloudscraper" in result.strategy

        # curl_cffi called once per strategy attempt = 2
        assert mock_curl.call_count == 2, (
            f"Expected curl_cffi called 2 times (one per retry attempt), "
            f"got {mock_curl.call_count}"
        )
        # cloudscraper called once (succeeds on 1st attempt)
        assert mock_cloud.call_count == 1
        # aiohttp never reached
        assert mock_aio.call_count == 0

    asyncio.run(_run())


# ======================== 429 RATE-LIMIT RETRY TEST =========================

def test_429_rate_limited_retried_with_backoff_then_succeeds():
    """
    Strategy-1 returns 429 on the first call, then 200 on the second.
    The 429 retry loop should sleep (back-off) and retry within the SAME
    strategy attempt, then return the 200 without ever touching strategy-2
    or strategy-3.
    """

    async def _run():
        # First call → 429 with Retry-After header; second call → 200
        responses = [
            _resp(
                429,
                strategy="curl_cffi(chrome131, h2=True)",
                headers={"Retry-After": "1"},
            ),
            _resp(200, strategy="curl_cffi(chrome131, h2=True)"),
        ]
        mock_curl = AsyncMock(side_effect=responses)
        mock_cloud = AsyncMock(return_value=None)
        mock_aio = AsyncMock(return_value=None)
        sleep_mock = AsyncMock()

        with (
            patch("app.connectors.sources.web.fetch_strategy._try_curl_cffi", mock_curl),
            patch("app.connectors.sources.web.fetch_strategy._try_cloudscraper", mock_cloud),
            patch("app.connectors.sources.web.fetch_strategy._try_aiohttp", mock_aio),
            patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", sleep_mock),
        ):
            result = await fetch_url_with_fallback(
                "http://example.com/",
                _dummy_session(),
                logger,
                max_retries_per_strategy=1,
                max_429_retries=1,  # allow one 429 retry
            )

        assert result is not None
        assert result.status_code == 200
        assert "curl_cffi" in result.strategy

        # curl_cffi called twice: once for 429, once for 200
        assert mock_curl.call_count == 2, (
            f"Expected curl_cffi called 2 times (429 then 200), got {mock_curl.call_count}"
        )
        # A sleep was triggered for the 429 back-off
        assert sleep_mock.called, "Expected asyncio.sleep to be called for 429 back-off"

        # Fallback strategies were never touched
        assert mock_cloud.call_count == 0
        assert mock_aio.call_count == 0

    asyncio.run(_run())


def test_429_exhausts_retries_then_falls_to_next_strategy():
    """
    Strategy-1 returns 429 on every call (exceeds max_429_retries),
    so the chain moves to strategy-2, which succeeds.
    """

    async def _run():
        # Always 429 for curl_cffi
        curl_always_429 = _side_effects([429, 429, 429, 429, 429], strategy="curl_cffi(chrome131, h2=True)")
        mock_curl = AsyncMock(side_effect=curl_always_429)
        mock_cloud = AsyncMock(return_value=_resp(200, strategy="cloudscraper"))
        mock_aio = AsyncMock(return_value=None)

        with (
            patch("app.connectors.sources.web.fetch_strategy._try_curl_cffi", mock_curl),
            patch("app.connectors.sources.web.fetch_strategy._try_cloudscraper", mock_cloud),
            patch("app.connectors.sources.web.fetch_strategy._try_aiohttp", mock_aio),
            patch("app.connectors.sources.web.fetch_strategy.asyncio.sleep", AsyncMock()),
        ):
            result = await fetch_url_with_fallback(
                "http://example.com/",
                _dummy_session(),
                logger,
                max_retries_per_strategy=1,
                max_429_retries=2,
            )

        assert result is not None
        assert result.status_code == 200
        assert "cloudscraper" in result.strategy

        # curl_cffi: (max_429_retries+1)=3 calls per strategy attempt × 1 attempt = 3
        assert mock_curl.call_count == 3, (
            f"Expected curl called 3 times (429 × 3 before giving up), "
            f"got {mock_curl.call_count}"
        )
        assert mock_cloud.call_count == 1
        assert mock_aio.call_count == 0

    asyncio.run(_run())
