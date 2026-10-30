"""Which OneDrive each synced user has, so a drive that leaves the source can be found and removed.

Users are listed by id and their drive is asked for by user (/users/{id}/drive). Once a user leaves the tenant,
or their OneDrive is deleted, nothing in a later listing names the drive any more; its records would stay
searchable for ever. The drive id seen on each successful read is kept here, in one sync point.

A connector that synced before this was kept has no entry for a drive until its next successful read of it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

ReadState = Callable[[], Awaitable[dict[str, Any] | None]]
WriteState = Callable[[dict[str, Any]], Awaitable[None]]


class DriveRegistry:
    def __init__(self, read_state: ReadState, write_state: WriteState) -> None:
        self._read_state = read_state
        self._write_state = write_state
        self._drives: dict[str, str] = {}
        self._dirty = False

    async def load(self) -> None:
        stored = (await self._read_state() or {}).get("drives") or {}
        # Kept as JSON text: a Neo4j property can't hold a map.
        self._drives = dict(json.loads(stored) if isinstance(stored, str) else stored)
        self._dirty = False

    def remember(self, user_id: str, drive_id: str) -> None:
        if self._drives.get(user_id) != drive_id:
            self._drives[user_id] = drive_id
            self._dirty = True

    def drive_of(self, user_id: str) -> str | None:
        return self._drives.get(user_id)

    def users(self) -> set[str]:
        return set(self._drives)

    def forget(self, user_id: str) -> None:
        if self._drives.pop(user_id, None) is not None:
            self._dirty = True

    async def save(self) -> None:
        if self._dirty:
            await self._write_state({"drives": json.dumps(self._drives, sort_keys=True)})
            self._dirty = False
