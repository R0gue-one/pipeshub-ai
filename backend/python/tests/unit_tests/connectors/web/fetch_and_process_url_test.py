"""
Unit tests for WebConnector._fetch_and_process_url – retry_urls population.

Scenarios covered
-----------------
1. fetch_url_with_fallback returns None for a *new* URL
   → retry_urls gets a fresh entry: retries=0, synthetic status_code=408.

2. fetch_url_with_fallback returns a retryable HTTP status for a *new* URL
   → retry_urls gets a fresh entry: retries=0, status_code=<actual HTTP code>.

3. fetch_url_with_fallback returns a retryable HTTP status for a URL *already*
   in retry_urls
   → existing entry is updated: retries incremented by 1, latest HTTP
     status_code stored.

All three cases also assert that:
  - the method returns None (no RecordUpdate is produced),
  - the entry url matches the normalised form of the requested URL,
  - the entry status is Status.PENDING.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.connectors.sources.web.connector import (
    RETRYABLE_STATUS_CODES,
    RetryUrl,
    Status,
    WebConnector,
)
from app.connectors.sources.web.fetch_strategy import FetchResponse
from conftest import WebConnectorHarness

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_URL = "http://example.com/page"

# _normalize_url strips trailing slashes from path segments and drops fragments;
# for _URL the normalised form is identical.
_NORM_URL = "http://example.com/page"

# Full dotted path used by patch() – must match the name imported inside connector.py.
_FETCH_PATCH = "app.connectors.sources.web.connector.fetch_url_with_fallback"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fetch_response(status: int, url: str = _URL) -> FetchResponse:
    """Return a minimal FetchResponse with the given status code."""
    return FetchResponse(
        status_code=status,
        content_bytes=b"",
        headers={},
        final_url=url,
        strategy="mock",
    )


def _call_fetch_and_process_url(
    harness: WebConnectorHarness,
    url: str = _URL,
    depth: int = 1,
    referer: str | None = "http://example.com/",
):
    """Invoke _fetch_and_process_url synchronously."""
    return asyncio.run(
        harness.connector._fetch_and_process_url(url, depth, referer)
    )


# ---------------------------------------------------------------------------
# Case 1: fetch returns None → new entry created
# ---------------------------------------------------------------------------


def test_fetch_none_new_url_creates_retry_entry(
    web_connector_harness: WebConnectorHarness,
) -> None:
    """
    When all fetch strategies are exhausted (None returned) and the URL has
    never been seen before, a new RetryUrl is inserted into retry_urls with
    retries=0 and the synthetic HTTP 408 (request timeout) status code.
    """
    connector = web_connector_harness.connector
    connector.session = MagicMock()  # prevent early-exit due to session=None

    with patch(_FETCH_PATCH, new=AsyncMock(return_value=None)):
        result = _call_fetch_and_process_url(web_connector_harness)

    assert result is None, "_fetch_and_process_url must return None when fetch fails"

    entry = connector.retry_urls.get(_NORM_URL)
    assert entry is not None, f"{_NORM_URL!r} must appear in retry_urls after a None fetch result"
    assert entry.url == _NORM_URL, f"entry.url mismatch: {entry.url!r}"
    assert entry.status == Status.PENDING, f"status must be PENDING, got {entry.status!r}"
    assert entry.status_code == 408, (
        f"New entry (None result) must use synthetic timeout code 408, got {entry.status_code}"
    )
    assert entry.retries == 0, (
        f"First failure must record retries=0, got {entry.retries}"
    )


# ---------------------------------------------------------------------------
# Case 2: retryable HTTP status → new entry created
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("retryable_code", [403])
def test_retryable_status_new_url_creates_retry_entry(
    web_connector_harness: WebConnectorHarness,
    retryable_code: int,
) -> None:
    """
    When fetch succeeds but returns a retryable HTTP status (e.g. 429, 503)
    and the URL is not yet in retry_urls, a new entry is created with
    retries=0 and status_code equal to the actual HTTP response code.
    """
    connector = web_connector_harness.connector
    connector.session = MagicMock()

    with patch(_FETCH_PATCH, new=AsyncMock(return_value=_make_fetch_response(retryable_code))):
        result = _call_fetch_and_process_url(web_connector_harness)

    assert result is None, (
        f"_fetch_and_process_url must return None for retryable status {retryable_code}"
    )

    entry = connector.retry_urls.get(_NORM_URL)
    assert entry is not None, (
        f"{_NORM_URL!r} must appear in retry_urls for HTTP {retryable_code}"
    )
    assert entry.url == _NORM_URL
    assert entry.status == Status.PENDING
    assert entry.status_code == retryable_code, (
        f"status_code must match the HTTP response ({retryable_code}), got {entry.status_code}"
    )
    assert entry.retries == 0, (
        f"First failure must record retries=0, got {entry.retries}"
    )


# ---------------------------------------------------------------------------
# Case 3: retryable HTTP status → existing entry updated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("retryable_code", sorted([403]))
def test_retryable_status_existing_url_increments_retries(
    web_connector_harness: WebConnectorHarness,
    retryable_code: int,
) -> None:
    """
    When fetch returns a retryable HTTP status and the URL already exists in
    retry_urls, the entry is updated: retries is incremented by 1 and
    status_code is replaced with the latest HTTP response code.
    """
    connector = web_connector_harness.connector
    connector.session = MagicMock()

    prior_retries = 1
    connector.retry_urls[_NORM_URL] = RetryUrl(
        url=_NORM_URL,
        status=Status.PENDING,
        status_code=429,  # previous code – may differ from retryable_code
        retries=prior_retries,
        last_attempted=0,
        depth=1,
        referer=None,
    )

    with patch(_FETCH_PATCH, new=AsyncMock(return_value=_make_fetch_response(retryable_code))):
        result = _call_fetch_and_process_url(web_connector_harness)

    assert result is None

    updated = connector.retry_urls.get(_NORM_URL)
    assert updated is not None, f"{_NORM_URL!r} must still be in retry_urls after update"
    assert updated.retries == prior_retries + 1, (
        f"retries must be incremented from {prior_retries} to {prior_retries + 1}, "
        f"got {updated.retries}"
    )
    assert updated.status_code == retryable_code, (
        f"status_code must be updated to the latest response code ({retryable_code}), "
        f"got {updated.status_code}"
    )
    assert updated.status == Status.PENDING
