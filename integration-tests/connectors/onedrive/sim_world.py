# pyright: ignore-file

"""OneDrive against the connectors-simulator: users' OneDrives the test owns, and who may read what.

The simulator (connectors-simulator's ``sharepoint`` source, which also answers Graph's OneDrive routes) holds
one tenant: users, groups, and each user's OneDrive as SharePoint keeps it, a personal site with one library.
The PipesHub container reaches it under Microsoft's host names through the TLS front of
deployment/docker-compose/docker-compose.integration.sharepoint-sim.yml; this process drives it through its
control API (``SHAREPOINT_SIM_URL``).

The connector syncs the OneDrives of tenant users who have a PipesHub account. Each test loads a tenant,
creates a OneDrive connector, syncs, changes the tenant and syncs again. After a sync the simulator's oracle
(``/_sim/sharepoint/expected``, computed from the tenant alone) says who may read each file; each probed
person, logged in, opens every record of the connector (``GET /api/v1/knowledgeBase/record/:id``). The
differences: LEAK, DENIED, MISSING, STALE, EXTRA, DUPLICATE, UNSETTLED, SIM_UNMODELED, SIM_SPEC (meanings in
connectors/sharepoint/sim_world.py). Folder records are not compared; a locked OneDrive is left out (nobody can
open it now, and it may be unlocked).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
import requests
from connectors.onedrive.sim_tenants import ONEDRIVE_PEOPLE
from connectors.sharepoint.sim_world import (  # the simulator client and helpers are the same service's
    OPEN_CONCURRENCY,
    SETTLE_POLL_SEC,
    SETTLE_TIMEOUT_SEC,
    SYNC_POLL_SEC,
    SYNC_TIMEOUT_SEC,
    FOLDER_MIME,
    SharePointSim,
    _account,
    client_certificate,
    fault,
    succeeded,
)
from helper.record_access import record_access_status
from helper.second_user import NO_ACCESS_STATUSES, SecondUser
from pipeshub_client import (  # type: ignore[import-not-found]
    PipeshubClient,
    PipeshubClientError,
)

from app.config.constants.arangodb import AppStatus

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterable

    from helper.graph_provider import GraphProviderProtocol

    from app.models.entities import Record

logger = logging.getLogger("onedrive-sim")

CONNECTOR_TYPE = "OneDrive"

__all__ = ["OneDriveWorld", "fault", "succeeded", "user_path", "onedrive_sim_people", "onedrive_world"]


def external_id(item: dict[str, Any]) -> str:
    """The externalRecordId the connector gives the item's record: the drive item id."""
    return item["id"]


def user_path(world: "OneDriveWorld", key: str) -> str:
    """A regex for the Graph paths of one user's OneDrive (by user id or drive id), for a fault on it alone."""
    user = next(u for u in world.expected["users"] if u["key"] == key)
    drive = re.escape(user["onedrive"]["drive_id"]) if user.get("onedrive") else "^$"
    return f"(/users/{re.escape(user['id'])}/drive|/drives/{drive})"


_RECORDS_CYPHER = (
    "MATCH (r:Record {connectorId: $c}) RETURN r.id AS id, r.externalRecordId AS externalRecordId, "
    "r.recordName AS recordName, r.recordType AS recordType, r.mimeType AS mimeType, "
    "r.isDeleted AS isDeleted, r.isPlaceholder AS isPlaceholder, r.externalParentId AS externalParentId, "
    "r.version AS version"
)
_RECORDS_AQL = (
    "FOR r IN records FILTER r.connectorId == @c RETURN {id: r._key, externalRecordId: r.externalRecordId, "
    "recordName: r.recordName, recordType: r.recordType, mimeType: r.mimeType, isDeleted: r.isDeleted, "
    "isPlaceholder: r.isPlaceholder, externalParentId: r.externalParentId, version: r.version}"
)


class OneDriveWorld:
    """One test's tenant, its connector, and the comparison between them."""

    def __init__(self, sim: SharePointSim, client: PipeshubClient, graph_provider: GraphProviderProtocol,
                 people: dict[str, SecondUser]) -> None:
        self.sim = sim
        self.client = client
        self.graph = graph_provider
        self.people = people
        self.connector_id: str | None = None
        self.instance_name = ""
        self.expected: dict[str, Any] = {}
        # The PipesHub accounts compared: fixed when the tenant is loaded, so a person disabled or deleted
        # in the source later is still asked, and must then be refused everything.
        self.probed: dict[str, SecondUser] = {}
        self.scope_extensions: set[str] | None = None
        self._fault_rules: list[dict[str, Any]] = []

    # ---- set-up ---------------------------------------------------------------------------------------

    async def start(self, tenant: dict[str, Any], *, behaviours: dict[str, Any] | None = None,
                    filters: dict[str, Any] | None = None, guests: bool = False,
                    enable: bool = True) -> str | None:
        """Load the tenant, create a OneDrive connector for it and, with ``enable``, run its first sync.
        Returns the first sync's outcome. The connector signs in with a certificate, as new OneDrive
        connectors must."""
        cert, key = client_certificate()
        tenant = {**tenant, "app": {**tenant.get("app", {}), "certificate_pem": cert}}
        self.sim.load(tenant, behaviours)
        self.expected = self.sim.expected()
        self.not_pipeshub = set(tenant.get("not_pipeshub", []))
        self.probed = {}
        for user in self.expected["users"]:
            if user["deleted"] or user["key"] in self.not_pipeshub or (user["type"] == "Guest" and not guests):
                continue
            email = (user["mail"] or user["upn"]).lower()
            assert email in self.people, f"no PipesHub account for tenant user {email}"
            self.probed[email] = self.people[email]

        info = self.expected["tenant"]
        config = {"auth": {"clientId": self.expected["app"]["client_id"], "tenantId": info["id"],
                           "hasAdminConsent": True, "certificate": cert, "privateKey": key}}
        self.instance_name = f"onedrive-sim-{uuid.uuid4().hex[:8]}"
        instance = self.client.create_connector(
            connector_type=CONNECTOR_TYPE, instance_name=self.instance_name,
            scope="team", config=config, auth_type="OAUTH_ADMIN_CONSENT",
        )
        assert instance.connector_id, "Connector must have a valid ID"
        self.connector_id = instance.connector_id
        if filters:
            self.client.update_connector_filters_sync_config(self.connector_id, filters=filters)
        if not enable:
            return None
        return await self.enable()

    async def enable(self) -> str | None:
        outcome = await self._run(lambda: self.client.toggle_sync(self.connector_id, enable=True))
        # A name that resolved to the real Microsoft would fail sign-in too; this tells the two apart.
        stats = self.sim.call("GET", "/_sim/sharepoint/stats")
        assert stats["tokens_issued"] > 0, (
            "the connector never signed in at the simulator: Microsoft's host names do not lead to it "
            f"from the PipesHub container (sync outcome {outcome}, simulator stats {stats})"
        )
        return outcome

    def remove_connector(self) -> None:
        """Disable and delete; the graph clean-up runs on its own while the next test starts.
        SHAREPOINT_SIM_KEEP_CONNECTOR=1 leaves it in place, to inspect a failure."""
        if os.getenv("SHAREPOINT_SIM_KEEP_CONNECTOR") == "1":
            logger.warning("Keeping connector %s for inspection", self.connector_id)
            return
        try:
            self.sim.call("DELETE", "/_sim/faults")
        except (requests.RequestException, RuntimeError):
            logger.exception("Could not clear the simulator's faults")
        if not self.connector_id:
            return
        for action in (lambda: self.client.toggle_sync(self.connector_id, enable=False),
                       lambda: self.client.delete_connector(self.connector_id)):
            try:
                action()
            except (PipeshubClientError, requests.RequestException):
                logger.exception("Teardown of connector %s failed", self.connector_id)

    def key(self, key: str) -> str:
        return self.expected["keys"][key]

    def library(self, key: str) -> dict[str, Any]:
        return next(lib for lib in self.expected["libraries"] if lib["key"] == key)

    # ---- the source -----------------------------------------------------------------------------------

    def ops(self, *ops: dict[str, Any]) -> list[dict[str, Any]]:
        return self.sim.call("POST", "/_sim/sharepoint/ops", list(ops))["results"]

    def faults(self, *rules: dict[str, Any]) -> None:
        self.sim.call("DELETE", "/_sim/faults")
        self.sim.call("PUT", "/_sim/faults", {"enabled": True, "rules": list(rules)})
        self._fault_rules = list(rules)

    def clear_faults(self) -> None:
        """Remove the faults, failing when none of them fired: a test whose fault the connector never
        reached proves nothing. Rules given together are alternatives, so one firing is enough."""
        fired = {entry["rule"] for entry in self.sim.call("GET", "/_sim/faults/log")}
        rules, self._fault_rules = self._fault_rules, []
        self.sim.call("DELETE", "/_sim/faults")
        assert not rules or fired, f"no injected fault fired, so the test proves nothing: {rules}"

    def expire_delta_tokens(self, kind: str | None = None) -> None:
        self.sim.call("POST", "/_sim/sharepoint/delta/expire", {"kind": kind} if kind else None)

    # ---- PipesHub -------------------------------------------------------------------------------------

    def _connector(self) -> dict[str, Any]:
        return self.client.get_connector(self.connector_id)

    async def _wait_idle(self) -> None:
        # A connector that never synced has no status yet, which is not a running sync.
        running = {AppStatus.SYNCING.value, AppStatus.FULL_SYNCING.value, AppStatus.QUEUED.value}
        deadline = time.monotonic() + SYNC_TIMEOUT_SEC
        while self._connector().get("status") in running:
            assert time.monotonic() < deadline, f"connector {self.connector_id} never went idle"
            await asyncio.sleep(SYNC_POLL_SEC)

    async def _run(self, trigger: Callable[[], Any]) -> str | None:
        """Trigger a run and wait for the product to record its end; return the outcome it recorded.

        The end time and outcome are written in one update with IDLE, so a newer end time is this run's
        end, whatever the status read in between.
        """
        await self._wait_idle()
        before = self._connector().get("lastSyncFinishedAtTimestamp") or 0
        trigger()
        deadline = time.monotonic() + SYNC_TIMEOUT_SEC
        while True:
            doc = self._connector()
            if (doc.get("lastSyncFinishedAtTimestamp") or 0) > before and doc.get("status") == AppStatus.IDLE.value:
                logger.info("Sync of %s ended: %s", self.connector_id, doc.get("lastSyncOutcome"))
                return doc.get("lastSyncOutcome")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no sync of connector {self.connector_id} ended within {SYNC_TIMEOUT_SEC}s "
                    f"(status {doc.get('status')}, last outcome {doc.get('lastSyncOutcome')})"
                )
            await asyncio.sleep(SYNC_POLL_SEC)

    async def sync(self, *, full: bool = False) -> str | None:
        return await self._run(lambda: self.client.resync_connector(self.connector_id, full_sync=full))

    async def set_filters(self, filters: dict[str, Any]) -> str | None:
        """Save filters the way the UI does; the re-enable that follows runs the full sync they owe."""
        return await self._run(
            lambda: self.client.update_connector_filters_sync_safe(self.connector_id, filters=filters)
        )

    async def records(self) -> list[dict[str, Any]]:
        params = {"c": self.connector_id}
        if os.getenv("TEST_GRAPH_DB_TYPE", "neo4j").lower() == "arango":
            rows = await self.graph.execute_query(_RECORDS_AQL, bind_vars=params)
        else:
            rows = await self.graph.client.execute_query(_RECORDS_CYPHER, params)
        return list(rows or [])

    async def record_of(self, key: str) -> Record | None:
        """The stored record of a tenant item, by its key."""
        item = next(i for i in self.sim.expected()["items"] if i["id"] == self.key(key))
        return await self.graph.get_record_by_external_id(self.connector_id, external_id(item))

    def reader_of(self, key: str) -> SecondUser:
        """A person the source lets read the item: the org admin is not one, and is refused."""
        expected = self.sim.expected()
        email_of = {u["id"]: (u["mail"] or u["upn"]).lower() for u in expected["users"]}
        item = next(i for i in expected["items"] if i["id"] == self.key(key))
        email = next((email_of[r] for r in item["readers"] if email_of.get(r) in self.probed), None)
        assert email, f"nobody probed may read {key}"
        return self.probed[email]

    def stream(self, record_id: str, user: SecondUser) -> tuple[int, bytes]:
        """Stream a record as preview does, as ``user``: (status, body)."""
        resp = requests.get(f"{user.base_url}{self.client._stream_record_path(record_id)}",
                            headers=user.headers, timeout=user.timeout)
        return resp.status_code, resp.content

    async def notices(self, contains: str, timeout: float = 60) -> list[dict[str, Any]]:
        """The creator's notifications from this connector instance (titled "<instance>: ..."), waiting up to
        ``timeout`` for one whose title or message contains ``contains``; returns all of them."""
        prefix = f"{self.instance_name}:"
        deadline = time.monotonic() + timeout
        while True:
            resp = requests.get(f"{self.client.base_url}/api/v1/notifications", params={"limit": 50},
                                headers=self.client._headers(), timeout=30)
            assert resp.status_code == 200, f"GET /api/v1/notifications: HTTP {resp.status_code} {resp.text[:200]}"
            mine = [n for n in resp.json().get("notifications", []) if str(n.get("title", "")).startswith(prefix)]
            if any(contains in f"{n.get('title', '')} {n.get('message', '')}" for n in mine) \
                    or time.monotonic() > deadline:
                return mine
            await asyncio.sleep(SYNC_POLL_SEC)

    # ---- the comparison --------------------------------------------------------------------------------

    def synced_drives(self) -> set[str]:
        """The OneDrives the connector syncs: those (not locked) of tenant users with a PipesHub account. A
        deleted user's OneDrive kept for retention still counts until it is deleted."""
        out = set()
        for user in self.expected["users"]:
            od = user.get("onedrive")
            if od and od["status"] == "ok" and user["key"] not in self.not_pipeshub:
                out.add(od["drive_id"])
        return out

    def locked_drives(self) -> set[str]:
        return {u["onedrive"]["drive_id"] for u in self.expected["users"]
                if u.get("onedrive") and u["onedrive"]["status"] == "locked"}

    def _in_scope(self, item: dict[str, Any]) -> bool:
        if not item["personal"] or item["drive_id"] not in self.synced_drives():
            return False
        if self.scope_extensions is not None and item["kind"] == "file" and \
                item["name"].rsplit(".", 1)[-1].lower() not in self.scope_extensions:
            return False
        return True

    def _expected_synced(self, item: dict[str, Any]) -> bool:
        return self._in_scope(item) and item["kind"] == "file" and "." in item["name"]

    async def _who_opens(self, records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], int]:
        """(email, record id) -> HTTP status of that person opening that record."""
        gate = asyncio.Semaphore(OPEN_CONCURRENCY)

        async def one(email: str, user: SecondUser, record_id: str) -> tuple[tuple[str, str], int]:
            async with gate:
                return (email, record_id), await asyncio.to_thread(record_access_status, user, record_id)

        pairs = [(email, user, r["id"]) for r in records for email, user in self.probed.items()]
        return dict(await asyncio.gather(*(one(*pair) for pair in pairs)))

    async def differences(self, *, allow: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        self.expected = self.sim.expected()
        email_of = {u["id"]: (u["mail"] or u["upn"]).lower() for u in self.expected["users"]}
        items = {external_id(i): i for i in self.expected["items"]}
        page_files: dict[str, dict[str, Any]] = {}
        records = [r for r in await self.records()
                   if not r.get("isDeleted") and not r.get("isPlaceholder") and r.get("mimeType") != FOLDER_MIME]
        problems: list[dict[str, Any]] = []

        by_external: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_external[record["externalRecordId"]].append(record)
        for ext, item in items.items():
            if self._expected_synced(item) and ext not in by_external and item["drive_id"] not in self.locked_drives():
                problems.append({"type": "MISSING", "item": item["name"], "kind": item["kind"], "id": ext})
        for ext, recs in by_external.items():
            if len(recs) > 1:
                problems.append({"type": "DUPLICATE", "item": recs[0]["recordName"], "id": ext,
                                 "records": [r["id"] for r in recs]})

        opened = await self._who_opens(records)
        for record in records:
            ext = record["externalRecordId"]
            item = items.get(ext)
            readers: set[str] = set()
            if item is not None and item["drive_id"] in self.locked_drives():
                # A locked OneDrive: nobody can open it now, and it may be unlocked; left out.
                continue
            if item is None and ext in page_files:
                problems.append({"type": "DUPLICATE", "item": record["recordName"], "record": record["id"],
                                 "detail": "a page also synced as a FILE from the Site Pages library"})
                readers = {email_of[r] for r in page_files[ext]["readers"] if r in email_of}
            elif item is not None and item["kind"] == "folder":
                continue
            elif item is not None:
                readers = {email_of[r] for r in item["readers"] if r in email_of}
            for email in self.probed:
                status = opened[(email, record["id"])]
                if status != 200 and status not in NO_ACCESS_STATUSES:
                    problems.append({"type": "UNSETTLED", "user": email, "item": record["recordName"],
                                     "record": record["id"], "status": status})
                    continue
                can_open = status == 200
                if item is None and ext not in page_files:
                    if can_open:
                        problems.append({"type": "STALE", "user": email, "item": record["recordName"],
                                         "record": record["id"], "detail": "item no longer in the source"})
                elif item is not None and not self._in_scope(item):
                    if can_open:
                        problems.append({"type": "STALE", "user": email, "item": item["name"],
                                         "record": record["id"], "detail": "out of the sync's scope, still opens"})
                elif item is not None and not self._expected_synced(item):
                    if can_open:
                        problems.append({"type": "EXTRA", "user": email, "item": item["name"],
                                         "record": record["id"], "source_can_read": email in readers})
                elif can_open and email not in readers:
                    problems.append({"type": "LEAK", "user": email, "item": record["recordName"],
                                     "record": record["id"]})
                elif email in readers and not can_open and item is not None:
                    problems.append({"type": "DENIED", "user": email, "item": record["recordName"],
                                     "record": record["id"], "status": status})

        problems += [{"type": "SIM_UNMODELED", **e} for e in self.sim.call("GET", "/_sim/sharepoint/unmodeled")]
        problems += [{"type": "SIM_SPEC", **e} for e in self.sim.call("GET", "/_sim/sharepoint/spec-problems")]
        return [p for p in problems if p["type"] not in allow]

    async def assert_matches_source(self, label: str, *, allow: tuple[str, ...] = ()) -> None:
        """Every probed person opens exactly the records the source lets them read, and nothing is
        missing, stale or stored twice."""
        await self._settle(label, lambda: self.differences(allow=allow))

    async def hierarchy_problems(self) -> list[dict[str, Any]]:
        """Each file and folder record hangs under the record of its source parent, and no placeholder
        stands in for a parent that exists."""
        self.expected = self.sim.expected()
        items = {i["id"]: i for i in self.expected["items"]}
        problems: list[dict[str, Any]] = []
        for record in await self.records():
            if record.get("isDeleted"):
                continue
            if record.get("isPlaceholder"):
                problems.append({"type": "PLACEHOLDER", "record": record["id"],
                                 "external": record["externalRecordId"]})
                continue
            item = items.get((record["externalRecordId"] or "").split(":root:")[-1])
            if item is None or item["kind"] == "page" or item["root"]:
                continue
            parent = items.get(item["parent"])
            if parent is None:
                continue
            if record.get("externalParentId") != external_id(parent):
                problems.append({"type": "PARENT", "item": item["name"],
                                 "record_parent": record.get("externalParentId"),
                                 "source_parent": external_id(parent)})
        return problems

    async def assert_hierarchy_matches_source(self, label: str) -> None:
        await self._settle(label, self.hierarchy_problems)

    async def _settle(self, label: str, compare: Callable[[], Any]) -> None:
        deadline = time.monotonic() + SETTLE_TIMEOUT_SEC
        while True:
            problems = await compare()
            if not problems or time.monotonic() >= deadline:
                break
            await asyncio.sleep(SETTLE_POLL_SEC)
        assert not problems, (
            f"{label}: PipesHub differs from SharePoint in {len(problems)} way(s) "
            f"({dict(Counter(p['type'] for p in problems))}): {problems[:25]}"
        )


# ---- fixtures --------------------------------------------------------------------------------------------




@pytest.fixture(scope="session")
def onedrive_sim_people(sharepoint_sim: SharePointSim, pipeshub_client: PipeshubClient) -> dict[str, SecondUser]:
    """A PipesHub account for every person of the OneDrive tenants who should have one. Kept after the session,
    as for the SharePoint suite: an email whose account was deleted cannot be added to the org again."""
    pipeshub_client._ensure_access_token()
    return {email: _account(pipeshub_client, email, f"OneDrive sim {name}") for email, name in ONEDRIVE_PEOPLE.items()}


@pytest_asyncio.fixture(loop_scope="session")
async def onedrive_world(
    sharepoint_sim: SharePointSim,
    onedrive_sim_people: dict[str, SecondUser],
    pipeshub_client: PipeshubClient,
    graph_provider: GraphProviderProtocol,
) -> AsyncGenerator[OneDriveWorld, None]:
    world = OneDriveWorld(sharepoint_sim, pipeshub_client, graph_provider, onedrive_sim_people)
    try:
        yield world
    finally:
        world.remove_connector()
