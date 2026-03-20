"""
Unit tests for WebConnector._extract_links_from_content (line 1183).

Strategy
--------
We pass raw HTML bytes directly to _extract_links_from_content, so no
FastAPI server or network I/O is required.  The web_connector_harness
fixture from conftest.py supplies a fully-wired WebConnector; we then
configure the per-test attributes (follow_external, restrict_to_start_path,
start_path_prefix) through _configure_connector() before each call.

Link types covered
------------------
  1. javascript:void(0)          -> rejected  (non-http/s scheme)
  2. mailto:hi@test.com          -> rejected  (non-http/s scheme)
  3. Link with .jpg extension    -> rejected  (media extension skip-list)
  4. Link with .PNG extension    -> rejected  (case-insensitive extension check)
  5. Link with .svg extension    -> rejected  (media extension skip-list)
  6. External domain link        -> rejected  when follow_external=False
                                   accepted   when follow_external=True
  7. rel="nofollow" + valid URL  -> ACCEPTED  (connector does not respect nofollow)
  8. Fragment-only href (#sec)   -> rejected  (parsed.fragment is truthy)
  9. Absolute URL with #fragment -> rejected  (parsed.fragment is truthy)
 10. Link outside start-path     -> rejected  when restrict_to_start_path=True
                                   accepted   when restrict_to_start_path=False
 11. Valid same-domain link      -> accepted
 12. Relative link               -> resolved to absolute & accepted
"""

import asyncio
import logging
from typing import Any, Callable, Dict
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest

from conftest import WebConnectorHarness

# ---------------------------------------------------------------------------
# Shared HTML fixture – one page with every edge-case link type
# ---------------------------------------------------------------------------

_BASE_URL = "http://example.com/docs/"

_NO_URL_HTML = b"""\
<!DOCTYPE html>
<html>
<head><title>No links here</title></head>
<body><p>No links here.</p></body>
</html>
"""

_MIXED_HTML = b"""\
<!DOCTYPE html>
<html>
<head><title>Link extraction test page</title></head>
<body>
  <!-- 1. javascript: scheme - must be rejected -->
  <a href="javascript:void(0)">JS void</a>

  <!-- 2. mailto: scheme - must be rejected -->
  <a href="mailto:hi@test.com">Email</a>

  <!-- 3. Image extension .jpg - must be rejected -->
  <a href="http://example.com/images/photo.jpg">JPEG image</a>

  <!-- 4. Image extension .PNG - must be rejected (case-insensitive) -->
  <a href="http://example.com/images/banner.PNG">PNG image</a>

  <!-- 5. Image extension .svg - must be rejected -->
  <a href="/assets/logo.svg">SVG icon</a>

  <!-- 6. External domain - rejected when follow_external=False -->
  <a href="https://external.org/page">External site</a>

  <!-- 7. rel="nofollow" but otherwise valid - connector does NOT honour nofollow -->
  <a href="/docs/nofollow-page" rel="nofollow">Nofollow link</a>

  <!-- 8. Fragment-only href - must be rejected -->
  <a href="#section-2">In-page anchor</a>

  <!-- 9. Absolute URL with fragment - must be rejected -->
  <a href="http://example.com/docs/page#top">Page with fragment</a>

  <!-- 10. URL outside the start path /docs/ - rejected when restrict_to_start_path=True -->
  <a href="/about">About page</a>

  <!-- 11. Valid same-domain link - must be accepted -->
  <a href="http://example.com/docs/guide">Docs guide</a>

  <!-- 12. Relative link - resolved to absolute, must be accepted -->
  <a href="tutorial">Relative tutorial</a>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_connector(
    harness: WebConnectorHarness,
    *,
    follow_external: bool = False,
    restrict_to_start_path: bool = False,
    start_url: str = _BASE_URL,
) -> None:
    """
    Apply per-test crawl settings directly on the harness connector.
    Mirrors what WebConnector.init() does when it reads config.
    """
    c = harness.connector
    c.follow_external = follow_external
    c.restrict_to_start_path = restrict_to_start_path
    c.url = start_url
    # Derive the prefix the same way init() does
    c.start_path_prefix = urlparse(start_url).path.rstrip("/") + "/"


def _extract(harness: WebConnectorHarness, html: bytes, base_url: str = _BASE_URL) -> list:
    """Call _extract_links_from_content synchronously with the supplied HTML bytes."""
    file_record = MagicMock()
    file_record.weburl = base_url
    return asyncio.run(
        harness.connector._extract_links_from_content(base_url, html, file_record)
    )


# ---------------------------------------------------------------------------
# Tests: links that are ALWAYS rejected regardless of settings
# ---------------------------------------------------------------------------

def test_javascript_scheme_rejected(web_connector_harness: WebConnectorHarness) -> None:
    """javascript:void(0) has a non-http/s scheme and must never appear."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert not any("javascript" in lnk for lnk in links), (
        "javascript: links must be rejected; got: " + str(links)
    )


def test_mailto_scheme_rejected(web_connector_harness: WebConnectorHarness) -> None:
    """mailto: has a non-http/s scheme and must never appear."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert not any("mailto" in lnk for lnk in links), (
        "mailto: links must be rejected; got: " + str(links)
    )


def test_jpg_extension_rejected(web_connector_harness: WebConnectorHarness) -> None:
    """.jpg is in the media-extension skip-list."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert not any(".jpg" in lnk.lower() for lnk in links), (
        ".jpg links must be rejected; got: " + str(links)
    )


def test_png_extension_rejected(web_connector_harness: WebConnectorHarness) -> None:
    """.PNG (uppercase) is matched case-insensitively against the skip-list."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert not any(".png" in lnk.lower() for lnk in links), (
        ".png links must be rejected; got: " + str(links)
    )


def test_svg_extension_rejected(web_connector_harness: WebConnectorHarness) -> None:
    """.svg is in the media-extension skip-list."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert not any(".svg" in lnk.lower() for lnk in links), (
        ".svg links must be rejected; got: " + str(links)
    )


def test_fragment_only_href_rejected(web_connector_harness: WebConnectorHarness) -> None:
    """href='#section-2' resolves to a URL whose fragment is non-empty."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert not any("#" in lnk for lnk in links), (
        "Fragment links must be rejected; got: " + str(links)
    )


def test_absolute_url_with_fragment_rejected(web_connector_harness: WebConnectorHarness) -> None:
    """http://example.com/docs/page#top carries a fragment and must be rejected."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert "http://example.com/docs/page#top" not in links, (
        "Absolute URL with #fragment must be rejected; got: " + str(links)
    )


# ---------------------------------------------------------------------------
# Test: rel="nofollow" is NOT honoured by the connector
# ---------------------------------------------------------------------------

def test_nofollow_link_is_accepted(web_connector_harness: WebConnectorHarness) -> None:
    """
    rel="nofollow" is purely advisory.  The connector does not read the rel
    attribute, so the link must appear in the output like any other valid URL.
    """
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert "http://example.com/docs/nofollow-page" in links, (
        "nofollow link must NOT be filtered; got: " + str(links)
    )


# ---------------------------------------------------------------------------
# Tests: external-domain links controlled by follow_external flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "follow_external, should_contain",
    [
        (False, False),
        (True, True),
    ],
    ids=["follow_external=False rejects external", "follow_external=True accepts external"],
)
def test_external_domain_link(
    web_connector_harness: WebConnectorHarness,
    follow_external: bool,
    should_contain: bool,
) -> None:
    _configure_connector(web_connector_harness, follow_external=follow_external)
    links = _extract(web_connector_harness, _MIXED_HTML)
    external_url = "https://external.org/page"
    if should_contain:
        assert external_url in links, (
            f"External link must be accepted when follow_external={follow_external}; got: {links}"
        )
    else:
        assert external_url not in links, (
            f"External link must be rejected when follow_external={follow_external}; got: {links}"
        )


# ---------------------------------------------------------------------------
# Tests: restrict_to_start_path controls whether /about is accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "restrict_to_start_path, should_contain",
    [
        (True, False),
        (False, True),
    ],
    ids=["restrict=True rejects /about", "restrict=False accepts /about"],
)
def test_restrict_to_start_path(
    web_connector_harness: WebConnectorHarness,
    restrict_to_start_path: bool,
    should_contain: bool,
) -> None:
    _configure_connector(
        web_connector_harness, restrict_to_start_path=restrict_to_start_path
    )
    links = _extract(web_connector_harness, _MIXED_HTML)
    about_url = "http://example.com/about"
    if should_contain:
        assert about_url in links, (
            f"/about must be accepted when restrict_to_start_path={restrict_to_start_path}; got: {links}"
        )
    else:
        assert about_url not in links, (
            f"/about must be rejected when restrict_to_start_path={restrict_to_start_path}; got: {links}"
        )


# ---------------------------------------------------------------------------
# Tests: valid links are always included
# ---------------------------------------------------------------------------

def test_absolute_same_domain_link_accepted(web_connector_harness: WebConnectorHarness) -> None:
    """http://example.com/docs/guide is a plain valid same-domain link."""
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert "http://example.com/docs/guide" in links, (
        "Valid same-domain link must be accepted; got: " + str(links)
    )


def test_relative_link_resolved_and_accepted(web_connector_harness: WebConnectorHarness) -> None:
    """
    href='tutorial' resolves against http://example.com/docs/
    to http://example.com/docs/tutorial, which must appear in the output.
    """
    _configure_connector(web_connector_harness)
    links = _extract(web_connector_harness, _MIXED_HTML)
    assert "http://example.com/docs/tutorial" in links, (
        "Relative link must be resolved to absolute and accepted; got: " + str(links)
    )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

def test_none_html_bytes_no_session_returns_empty(
    web_connector_harness: WebConnectorHarness,
) -> None:
    """When html_bytes is None and session is None, return []."""
    _configure_connector(web_connector_harness)
    web_connector_harness.connector.session = None
    file_record = MagicMock()
    file_record.weburl = _BASE_URL
    links = asyncio.run(
        web_connector_harness.connector._extract_links_from_content(
            _BASE_URL, None, file_record
        )
    )
    assert links == [], "Without html_bytes or a session, result must be []; got: " + str(links)


def test_html_with_no_anchors_returns_empty(
    web_connector_harness: WebConnectorHarness,
) -> None:
    """A page with no <a> tags yields no links."""
    _configure_connector(web_connector_harness)
    links = _extract(
        web_connector_harness,
        _NO_URL_HTML,
    )
    assert links == [], "HTML with no anchors must yield []; got: " + str(links)


def test_empty_bytes_no_session_returns_empty(
    web_connector_harness: WebConnectorHarness,
) -> None:
    """Empty byte string is falsy, treated as missing; no session means []."""
    _configure_connector(web_connector_harness)
    web_connector_harness.connector.session = None
    file_record = MagicMock()
    file_record.weburl = _BASE_URL
    links = asyncio.run(
        web_connector_harness.connector._extract_links_from_content(
            _BASE_URL, b"", file_record
        )
    )
    assert links == [], "Empty bytes with no session must yield []; got: " + str(links)


# ---------------------------------------------------------------------------
# Snapshot test: exact set of accepted URLs with default settings
# ---------------------------------------------------------------------------

def test_exact_accepted_urls_snapshot(web_connector_harness: WebConnectorHarness) -> None:
    """
    Regression guard: with follow_external=False and restrict_to_start_path=False
    the extracted set must match exactly – no more, no less.
    """
    _configure_connector(
        web_connector_harness,
        follow_external=False,
        restrict_to_start_path=False,
    )
    links = _extract(web_connector_harness, _MIXED_HTML)
    link_set = set(links)

    expected = {
        "http://example.com/docs/nofollow-page",  # nofollow is ignored
        "http://example.com/about",               # outside /docs/ but restrict=False
        "http://example.com/docs/guide",          # plain valid absolute link
        "http://example.com/docs/tutorial",       # relative -> resolved to absolute
    }

    should_be_absent = {
        "https://external.org/page",              # external, follow_external=False
        "http://example.com/images/photo.jpg",    # .jpg extension
        "http://example.com/images/banner.PNG",   # .PNG extension
        "http://example.com/assets/logo.svg",     # .svg extension (resolved from /assets/logo.svg)
        "http://example.com/docs/page#top",       # has fragment
    }

    for url in expected:
        assert url in link_set, (
            f"Expected URL missing from results: {url!r}\nGot: {link_set}"
        )

    for url in should_be_absent:
        assert url not in link_set, (
            f"URL that should be rejected appeared in results: {url!r}\nGot: {link_set}"
        )

    assert link_set == expected, (
        f"Unexpected extra URLs in results.\nExpected: {expected}\nActual:   {link_set}"
    )
