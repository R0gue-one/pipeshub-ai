"""Folders that users added to their OneDrive with "Add shortcut to My files", so what a colleague who is not
synced shared with them is synced too.

One job: remember which shared folders the synced users hold shortcuts to, and when each was last looked for and
read. The folders themselves are read by the connector.

A shortcut lives in the holder's OneDrive, but Graph leaves it out of that drive's delta, of its folders' children
and of its list's delta, for an app token; only the drive's list items show it, with a `remoteItem` naming the
folder (live probe of a real tenant, 2026-10-30; Microsoft Q&A 761174 and onedrive-api-docs #1427 and #1781
report the same for the delta and the children). Listing a drive's items costs a call per few hundred items, so a
drive is looked at once per CHECK_INTERVAL_MS, and on every full sync. A shared folder has no delta of its own
either, so it is listed whole at the same pace.

Stored in a sync point as JSON text: a graph property holds scalars, not maps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

STATE_FIELD = "shortcuts"
CHECK_INTERVAL_MS = 60 * 60 * 1000
# Only folders in someone's OneDrive. A SharePoint library ("documentLibrary") is the SharePoint connector's.
FOLLOWED_DRIVE_TYPES = frozenset({"business"})


@dataclass(frozen=True)
class SharedFolder:
    drive_id: str
    item_id: str

    @property
    def key(self) -> str:
        return f"{self.drive_id}|{self.item_id}"

    @classmethod
    def from_key(cls, key: str) -> SharedFolder:
        drive_id, _, item_id = key.partition("|")
        return cls(drive_id, item_id)


class ShortcutRegistry:
    def __init__(self, read: Callable[[], Awaitable[dict[str, Any] | None]],
                 write: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._read = read
        self._write = write
        # Folder key -> the users holding a shortcut to it, and when it was last read whole.
        self._folders: dict[str, dict[str, Any]] = {}
        # User id -> when their OneDrive was last looked at for shortcuts.
        self._checked: dict[str, int] = {}
        self._dirty = False

    async def load(self) -> None:
        stored = (await self._read() or {}).get(STATE_FIELD)
        try:
            loaded = json.loads(stored) if stored else {}
        except (TypeError, ValueError):
            loaded = {}
        loaded = loaded if isinstance(loaded, dict) else {}
        self._folders = dict(loaded.get("folders") or {})
        self._checked = dict(loaded.get("checked") or {})
        self._dirty = False

    def check_due(self, user_id: str, now_ms: int) -> bool:
        checked = self._checked.get(user_id)
        return checked is None or now_ms - checked >= CHECK_INTERVAL_MS

    def found(self, user_id: str, folders: Iterable[SharedFolder], now_ms: int) -> None:
        """What a complete look at the user's OneDrive found: their shortcuts are exactly these now."""
        keys = {f.key for f in folders}
        for key, entry in self._folders.items():
            holders = set(entry["holders"])
            if (user_id in holders) != (key in keys):
                entry["holders"] = sorted(holders ^ {user_id})
        for key in keys - set(self._folders):
            self._folders[key] = {"holders": [user_id], "readAt": None}
        self._checked[user_id] = now_ms
        self._dirty = True

    def keep_holders(self, user_ids: set[str]) -> None:
        """Users not synced any more hold nothing: their shortcuts no longer bring a folder in."""
        for entry in self._folders.values():
            kept = [u for u in entry["holders"] if u in user_ids]
            if kept != entry["holders"]:
                entry["holders"] = kept
                self._dirty = True
        for user_id in set(self._checked) - user_ids:
            del self._checked[user_id]
            self._dirty = True

    def unheld(self) -> list[SharedFolder]:
        return [SharedFolder.from_key(k) for k, e in sorted(self._folders.items()) if not e["holders"]]

    def held(self) -> list[SharedFolder]:
        return [SharedFolder.from_key(k) for k, e in sorted(self._folders.items()) if e["holders"]]

    def read_due(self, folder: SharedFolder, now_ms: int) -> bool:
        read_at = self._folders.get(folder.key, {}).get("readAt")
        return read_at is None or now_ms - int(read_at) >= CHECK_INTERVAL_MS

    def read(self, folder: SharedFolder, now_ms: int) -> None:
        if folder.key in self._folders:
            self._folders[folder.key]["readAt"] = now_ms
            self._dirty = True

    def forget(self, folder: SharedFolder) -> None:
        if self._folders.pop(folder.key, None) is not None:
            self._dirty = True

    def folders_in(self, drive_id: str) -> list[SharedFolder]:
        return [f for f in map(SharedFolder.from_key, sorted(self._folders)) if f.drive_id == drive_id]

    async def save(self) -> None:
        if self._dirty:
            await self._write({STATE_FIELD: json.dumps({"folders": self._folders, "checked": self._checked},
                                                       sort_keys=True, separators=(",", ":"))})
            self._dirty = False
