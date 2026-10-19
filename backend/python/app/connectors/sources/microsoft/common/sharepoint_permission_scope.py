"""Which items of a SharePoint list (a document library, a OneDrive) carry their own permissions.

Graph marks no driveItem permission as inherited, so only SharePoint REST
(``HasUniqueRoleAssignments``) can tell a uniquely permissioned item from one
that inherits. SharePoint REST refuses app-only tokens from a client secret: the
app has to sign in with a certificate, and hold the SharePoint (not Graph)
``Sites.Read.All`` application permission.
"""

import asyncio
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from http import HTTPStatus
from urllib.parse import urlparse

import httpx
from aiolimiter import AsyncLimiter

GUID_PATTERN = re.compile(r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})')

SHAREPOINT_REST_HEADERS = {
    'Accept': 'application/json;odata=verbose',
    'Content-Type': 'application/json;odata=verbose',
}
SHAREPOINT_REST_RETRY_STATUSES = (HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.SERVICE_UNAVAILABLE)
SHAREPOINT_REST_MAX_ATTEMPTS = 3
SHAREPOINT_REST_MAX_RETRY_AFTER_S = 30
SHAREPOINT_ITEMS_PAGE_SIZE = 5000


@dataclass
class ListPermissionScope:
    """Which parts of a SharePoint list carry their own permissions."""
    web_url: str
    list_url: str
    has_unique_role_assignments: bool
    # Lower-cased list item UniqueId -> list item Id, for items with unique role assignments.
    unique_items: dict[str, int]


def etag_unique_id(e_tag: object) -> str | None:
    """The list item UniqueId in a drive item eTag, ``"{GUID},n"``, lower-cased."""
    if isinstance(e_tag, str):
        match = GUID_PATTERN.search(e_tag)
        if match:
            return match.group(1).lower()
    return None


def drive_item_unique_id(item: object) -> str | None:
    """The SharePoint list item UniqueId of a drive item, lower-cased."""
    sharepoint_ids = getattr(item, 'sharepoint_ids', None)
    unique_id = getattr(sharepoint_ids, 'list_item_unique_id', None) if sharepoint_ids else None
    if isinstance(unique_id, str) and unique_id:
        return unique_id.lower()
    return etag_unique_id(getattr(item, 'e_tag', None))


class SharePointPermissionScopeReader:
    """Reads ``ListPermissionScope`` over SharePoint REST.

    ``get_token`` takes a resource such as ``https://contoso-my.sharepoint.com``
    and returns a bearer token for it.
    """

    def __init__(self, get_token: Callable[[str], Awaitable[str]], rate_limiter: AsyncLimiter) -> None:
        self._get_token = get_token
        self._rate_limiter = rate_limiter

    async def read_list_scope(self, web_url: str, list_url: str) -> ListPermissionScope:
        """Raises when any page can't be read, so a failed read can't pass for "nothing is unique"."""
        list_body: dict = {}
        async for body in self._pages(f"{list_url}?$select=Id,HasUniqueRoleAssignments"):
            list_body = body

        unique_items: dict[str, int] = {}
        items_url = f"{list_url}/items?$select=Id,UniqueId,HasUniqueRoleAssignments&$top={SHAREPOINT_ITEMS_PAGE_SIZE}"
        async for body in self._pages(items_url, follow_next=True):
            for item in body.get('results', []):
                unique_id = item.get('UniqueId')
                if item.get('HasUniqueRoleAssignments') is True and isinstance(unique_id, str):
                    unique_items[unique_id.lower()] = item.get('Id')

        return ListPermissionScope(
            web_url=web_url,
            list_url=list_url,
            has_unique_role_assignments=list_body.get('HasUniqueRoleAssignments') is True,
            unique_items=unique_items,
        )

    async def _pages(self, url: str, *, follow_next: bool = False) -> AsyncGenerator[dict, None]:
        access_token = await self._get_token(f"https://{urlparse(url).netloc}")
        headers = {**SHAREPOINT_REST_HEADERS, 'Authorization': f'Bearer {access_token}'}

        next_url: str | None = url
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            while next_url:
                payload = (await self._get(http_client, next_url, headers)).json()
                body = payload.get('d', {})
                yield body
                next_url = (body.get('__next') or payload.get('odata.nextLink')) if follow_next else None

    async def _get(self, http_client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> httpx.Response:
        attempt = 1
        while True:
            async with self._rate_limiter:
                response = await http_client.get(url, headers=headers)
            if response.status_code == HTTPStatus.OK:
                return response
            if response.status_code not in SHAREPOINT_REST_RETRY_STATUSES or attempt >= SHAREPOINT_REST_MAX_ATTEMPTS:
                raise RuntimeError(f"SharePoint REST answered {response.status_code} for {url}")
            retry_after = response.headers.get('Retry-After', '')
            delay = int(retry_after) if retry_after.isdigit() else 2 ** attempt
            await asyncio.sleep(min(delay, SHAREPOINT_REST_MAX_RETRY_AFTER_S))
            attempt += 1
