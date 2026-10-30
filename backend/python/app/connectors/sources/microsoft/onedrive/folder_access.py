"""What each folder of a OneDrive let people do when it was last read, to tell when its sharing changed.

One job: keep a digest of each folder's access list and say whether a list read now differs. Graph's delta lists a
folder whose sharing changed, but a removal is reported on the folder alone, not on the files inside it, and the
folder's `shared` facet stays as it was while anyone else still has access (live probe of a real OneDrive,
2026-10-29). Files that store their own access list are rewritten only when this says the folder's changed.

Stored in a sync point as JSON text: a graph property holds scalars, not maps.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

STATE_FIELD = "folderAccess"


def access_digest(permissions: Iterable[Any]) -> str:
    grants = sorted({
        (str(getattr(p.entity_type, "value", p.entity_type)), p.external_id or "", (p.email or "").lower(),
         str(getattr(p.type, "value", p.type)))
        for p in permissions
    })
    return hashlib.sha256(json.dumps(grants).encode()).hexdigest()[:24]


class FolderAccess:
    def __init__(self, read: Callable[[], Awaitable[dict[str, Any] | None]],
                 write: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._read = read
        self._write = write
        self._digests: dict[str, str] = {}
        self._dirty = False

    async def load(self) -> None:
        stored = (await self._read() or {}).get(STATE_FIELD)
        try:
            loaded = json.loads(stored) if stored else {}
        except (TypeError, ValueError):
            loaded = {}
        self._digests = loaded if isinstance(loaded, dict) else {}

    def changed(self, folder_id: str, permissions: Iterable[Any]) -> bool | None:
        """Whether the folder's access differs from when it was last remembered. None when it never was (a new
        folder, or one stored before this existed)."""
        before = self._digests.get(folder_id)
        return None if before is None else before != access_digest(permissions)

    def remember(self, folder_id: str, permissions: Iterable[Any]) -> None:
        """Called once what depends on the folder's access has been brought up to date: a folder remembered
        before that would not be seen as changed again."""
        digest = access_digest(permissions)
        if self._digests.get(folder_id) != digest:
            self._digests[folder_id] = digest
            self._dirty = True

    async def save(self) -> None:
        if self._dirty:
            await self._write({STATE_FIELD: json.dumps(self._digests, separators=(",", ":"))})
            self._dirty = False
