"""
Unit tests for WebConnector.process_retry_urls – failed placeholder records.

Scenario
--------
Two RetryUrl entries are pre-seeded in connector.retry_urls with
``retries == MAX_RETRIES`` (i.e. fully exhausted).

Phase 1 – recursive crawl
    A crawl is kicked off against a dummy start URL that returns HTTP 404
    (non-retryable).  After the start URL is processed the main queue is
    empty.  The re-enqueue logic inside _crawl_recursive_generator finds no
    candidates with ``retries < MAX_RETRIES`` and breaks immediately, leaving
    the two exhausted entries in retry_urls untouched and unvisited.

Phase 2 – process_retry_urls
    process_retry_urls() iterates over the retry_urls snapshot and calls
    _create_failed_placeholder_record for each entry, which emits a
    RecordUpdate with is_new=True and indexing_status=FAILED.

Assertions
----------
- Neither exhausted URL appears in visited_urls after the crawl.
- Both exhausted URLs remain in retry_urls after the crawl.
- After process_retry_urls() exactly 2 records with
  indexing_status == ProgressStatus.FAILED.value have been collected.
- The FAILED records correspond to the two pre-seeded URLs.
"""

import asyncio
from typing import Any, Callable, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.constants.arangodb import ProgressStatus
from app.connectors.sources.web.connector import (
    MAX_RETRIES,
    RetryUrl,
    Status,
)
from app.connectors.sources.web.fetch_strategy import FetchResponse
from conftest import WebConnectorHarness

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_START_URL = "http://example.com/"
_FAILED_URL_A = "http://example.com/page-a"
_FAILED_URL_B = "http://example.com/page-b"

_FETCH_PATCH = "app.connectors.sources.web.connector.fetch_url_with_fallback"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_404_response(url: str = _START_URL) -> FetchResponse:
    """Return a 404 FetchResponse.

    404 is not in RETRYABLE_STATUS_CODES so the URL will NOT be added to
    retry_urls – it simply produces no RecordUpdate and returns None.
    """
    return FetchResponse(
        status_code=404,
        content_bytes=b"",
        headers={},
        final_url=url,
        strategy="mock",
    )


def _seed_exhausted_retry_urls(connector) -> None:
    """Pre-populate retry_urls with two fully-exhausted entries.

    Both entries have ``retries == MAX_RETRIES`` so they will be skipped
    by the re-enqueue logic inside _crawl_recursive_generator and left
    for process_retry_urls() to handle.
    """
    connector.retry_urls[_FAILED_URL_A] = RetryUrl(
        url=_FAILED_URL_A,
        status=Status.PENDING,
        status_code=503,
        retries=MAX_RETRIES,
        last_attempted=0,
        depth=1,
        referer=_START_URL,
    )
    connector.retry_urls[_FAILED_URL_B] = RetryUrl(
        url=_FAILED_URL_B,
        status=Status.PENDING,
        status_code=429,
        retries=MAX_RETRIES,
        last_attempted=0,
        depth=1,
        referer=_START_URL,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_exhausted_retry_urls_produce_failed_placeholders(
    web_connector_harness: WebConnectorHarness,
    patch_crawl_sleep,
) -> None:
    """
    Full flow:
      1. Two exhausted retry entries are seeded before the crawl.
      2. _crawl_recursive runs; start URL returns 404 (no retry entry created).
         Re-enqueue logic finds only exhausted candidates → breaks.
      3. Neither exhausted URL is visited during the crawl.
      4. process_retry_urls() produces exactly 2 FAILED placeholder records.
    """
    asyncio.run(_run_scenario(web_connector_harness))


async def _run_scenario(harness: WebConnectorHarness) -> None:
    connector = harness.connector

    # Configure the minimum state that _crawl_recursive needs.
    # (Mirrors what init() does; avoids creating a real aiohttp session.)
    connector.url = _START_URL
    connector.session = MagicMock()

    # ── Step 1: seed exhausted retry entries ─────────────────────────────────
    _seed_exhausted_retry_urls(connector)

    assert len(connector.retry_urls) == 2, "pre-condition: 2 retry entries must be seeded"

    # ── Step 2: run the recursive crawl ──────────────────────────────────────
    # The start URL returns 404 (non-retryable):
    #   • _fetch_and_process_url returns None without adding to retry_urls.
    #   • Queue empties; re-enqueue finds no candidates (both entries are
    #     at MAX_RETRIES); generator breaks.
    with patch(_FETCH_PATCH, new=AsyncMock(return_value=_make_404_response())):
        await connector._crawl_recursive(_START_URL, depth=0)

    # ── Step 3: assert crawl did NOT touch the exhausted entries ─────────────
    norm_a = connector._normalize_url(_FAILED_URL_A)
    norm_b = connector._normalize_url(_FAILED_URL_B)

    assert norm_a not in connector.visited_urls, (
        f"{_FAILED_URL_A!r} must NOT be visited during the crawl phase; "
        f"visited_urls={connector.visited_urls}"
    )
    assert norm_b not in connector.visited_urls, (
        f"{_FAILED_URL_B!r} must NOT be visited during the crawl phase; "
        f"visited_urls={connector.visited_urls}"
    )

    assert norm_a in connector.retry_urls, (
        f"{_FAILED_URL_A!r} must still be in retry_urls after crawl"
    )
    assert norm_b in connector.retry_urls, (
        f"{_FAILED_URL_B!r} must still be in retry_urls after crawl"
    )

    # ── Step 4: process_retry_urls → FAILED placeholders ─────────────────────
    await connector.process_retry_urls()

    # ── Step 5: assertions on collected records ───────────────────────────────
    failed_records = [
        (record, perms)
        for record, perms in harness.collected
    ]

    assert len(failed_records) == 2, (
        f"Expected exactly 2 FAILED placeholder records after process_retry_urls(), "
        f"got {len(failed_records)}. "
        f"All collected ({len(harness.collected)} total): "
        f"{[(r.weburl, r.indexing_status) for r, _ in harness.collected]}"
    )

    failed_weburls = {record.weburl for record, _ in failed_records}

    assert _FAILED_URL_A in failed_weburls, (
        f"Expected a FAILED placeholder for {_FAILED_URL_A!r}; "
        f"got weburls: {failed_weburls}"
    )
    assert _FAILED_URL_B in failed_weburls, (
        f"Expected a FAILED placeholder for {_FAILED_URL_B!r}; "
        f"got weburls: {failed_weburls}"
    )
