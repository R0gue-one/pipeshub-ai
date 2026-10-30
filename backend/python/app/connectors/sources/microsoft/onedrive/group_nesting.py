"""Which groups contain which, so a change inside a nested group reaches the groups around it.

Each group is stored with its transitive users (SharePoint opens a share with a security group to the members of
the groups nested in it). Graph's group delta reports a membership change on the group whose direct members
changed (https://learn.microsoft.com/en-us/graph/api/group-delta: members@delta lists direct members), not on
the groups that contain it, and a deleted group can no longer be asked what contained it. So the containment
seen when each outer group was read is kept, and the outer groups of a changed or deleted group are read again.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

ReadState = Callable[[], Awaitable[dict[str, Any] | None]]
WriteState = Callable[[dict[str, Any]], Awaitable[None]]

STATE_FIELD = "nestedIn"


class GroupNesting:
    def __init__(self, read_state: ReadState, write_state: WriteState) -> None:
        self._read_state = read_state
        self._write_state = write_state
        # inner group id -> ids of the groups it is nested in (at any depth)
        self._nested_in: dict[str, set[str]] = {}
        self._dirty = False

    async def load(self) -> None:
        stored = (await self._read_state() or {}).get(STATE_FIELD) or {}
        # Kept as JSON text: a Neo4j property can't hold a map.
        if isinstance(stored, str):
            stored = json.loads(stored)
        self._nested_in = {inner: set(outers) for inner, outers in stored.items()}
        self._dirty = False

    def record(self, outer_id: str, inner_ids: set[str]) -> None:
        """What a fresh read of `outer_id`'s transitive members found nested in it, replacing what was known."""
        for inner, outers in list(self._nested_in.items()):
            if outer_id in outers and inner not in inner_ids:
                outers.discard(outer_id)
                if not outers:
                    del self._nested_in[inner]
                self._dirty = True
        for inner in inner_ids:
            outers = self._nested_in.setdefault(inner, set())
            if outer_id not in outers:
                outers.add(outer_id)
                self._dirty = True

    def outer_groups(self, group_id: str) -> set[str]:
        return set(self._nested_in.get(group_id, ()))

    def forget(self, group_id: str) -> None:
        """A deleted group: it contains nothing and is in nothing any more."""
        changed = self._nested_in.pop(group_id, None) is not None
        for outers in self._nested_in.values():
            if group_id in outers:
                outers.discard(group_id)
                changed = True
        self._dirty = self._dirty or changed

    async def save(self) -> None:
        if self._dirty:
            await self._write_state({STATE_FIELD: json.dumps(
                {inner: sorted(outers) for inner, outers in self._nested_in.items()}, sort_keys=True)})
            self._dirty = False
