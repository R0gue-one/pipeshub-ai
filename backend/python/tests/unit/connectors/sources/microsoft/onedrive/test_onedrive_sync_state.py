"""The OneDrive connector's run-state modules: what a run could not read, the notices it sends, which groups
contain which, the OneDrive known for each user, and whether Microsoft reports sharing changes to the app."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from msgraph.generated.models.o_data_errors.o_data_error import ODataError

from app.connectors.sources.microsoft.onedrive.drive_registry import DriveRegistry
from app.connectors.sources.microsoft.onedrive.folder_access import FolderAccess
from app.connectors.sources.microsoft.onedrive.group_nesting import (
    STATE_FIELD,
    GroupNesting,
)
from app.connectors.sources.microsoft.onedrive.notices import OneDriveNotices
from app.connectors.sources.microsoft.onedrive.run_health import (
    OneDriveSyncIncomplete,
    RunHealth,
)
from app.services.notification.types import NotificationSeverity, NotificationType


class _SyncPoint:
    """One sync point with the store's merge-on-update semantics; values must survive a JSON round trip."""

    def __init__(self, stored: dict[str, Any] | None = None) -> None:
        self.stored = stored
        self.writes: list[dict[str, Any]] = []

    async def read(self) -> dict[str, Any] | None:
        return dict(self.stored) if self.stored is not None else None

    async def write(self, data: dict[str, Any]) -> None:
        # A graph property holds scalars and lists of scalars, not maps.
        assert all(not isinstance(v, dict) for v in data.values()), data
        self.writes.append(data)
        self.stored = {**(self.stored or {}), **json.loads(json.dumps(data))}


class TestRunHealth:
    def test_a_run_that_skipped_nothing_is_complete(self) -> None:
        health = RunHealth()

        health.raise_if_incomplete()

        assert health.complete

    def test_a_skipped_drive_names_its_user_and_fails_the_run(self) -> None:
        health = RunHealth()
        health.skipped_drive("ana@acme.com", "HTTP 503")

        with pytest.raises(OneDriveSyncIncomplete, match="OneDrive of ana@acme.com: HTTP 503"):
            health.raise_if_incomplete()
        assert health.skipped_users == ["ana@acme.com"]
        assert not health.complete

    def test_a_skipped_part_that_is_not_a_drive_names_no_user(self) -> None:
        health = RunHealth()
        health.skipped("groups", "some groups could not be read")

        assert health.skipped_users == []
        assert health.problems == ["groups: some groups could not be read"]

    def test_the_summary_names_the_first_parts_and_counts_the_rest(self) -> None:
        health = RunHealth()
        for n in range(7):
            health.skipped(f"part {n}", "x")

        assert health.summary(limit=2) == "part 0: x; part 1: x; and 5 more"
        assert health.summary() == "part 0: x; part 1: x; part 2: x; part 3: x; part 4: x; and 2 more"


class TestOneDriveNotices:
    @staticmethod
    def _notices() -> tuple[OneDriveNotices, AsyncMock, dict[str, Any]]:
        notify = AsyncMock()
        instance = {"name": "OneDrive", "creator": "admin-1"}
        notices = OneDriveNotices(
            notify,
            lambda event: f"{instance['name']}: {event}",
            lambda: {"connector_id": "od-1", "connector_name": instance["name"], "connector_scope": "team"},
            lambda: [instance["creator"]],
        )
        return notices, notify, instance

    async def test_sign_in_failed_has_one_wording(self) -> None:
        notices, notify, _ = self._notices()

        await notices.sign_in_failed()
        await notices.sign_in_failed()

        first, second = (call.kwargs for call in notify.await_args_list)
        assert first == second
        assert first["type"] is NotificationType.CONNECTOR_AUTH_ERROR
        assert first["severity"] is NotificationSeverity.ERROR
        assert first["title"] == "OneDrive: Microsoft refused the sign-in"
        assert "Update the credentials on the connector" in first["message"]
        assert first["recipient_user_ids"] == ["admin-1"]

    async def test_the_instance_name_and_creator_are_read_when_a_notice_is_sent(self) -> None:
        notices, notify, instance = self._notices()
        instance.update(name="Sales OneDrive", creator="admin-2")

        await notices.part_skipped("groups: some groups could not be read")

        sent = notify.await_args.kwargs
        assert sent["title"] == "Sales OneDrive: part of the sync could not be read"
        assert sent["payload"]["connector_name"] == "Sales OneDrive"
        assert sent["recipient_user_ids"] == ["admin-2"]
        assert sent["message"].startswith("groups: some groups could not be read. ")

    async def test_drives_skipped_lists_five_users_and_counts_the_rest(self) -> None:
        notices, notify, _ = self._notices()

        await notices.drives_skipped([f"u{n}@acme.com" for n in range(7)], "HTTP 503")

        sent = notify.await_args.kwargs
        assert sent["type"] is NotificationType.CONNECTOR_RECORD_SYNC_ERROR
        assert sent["severity"] is NotificationSeverity.WARNING
        assert sent["title"] == "OneDrive: some OneDrives could not be synced"
        assert "u0@acme.com, u1@acme.com, u2@acme.com, u3@acme.com, u4@acme.com, and 2 more" in sent["message"]
        assert "u5@acme.com" not in sent["message"]
        assert "(HTTP 503)" in sent["message"]

    @pytest.mark.parametrize("send, title, kind", [
        (lambda n: n.drives_read_again(["ana@acme.com"]), "OneDrive: OneDrives read again from the start",
         NotificationType.CONNECTOR_WARNING),
        (lambda n: n.users_unavailable("OneDrive blocked for some users", "m"), "OneDrive: OneDrive blocked for some users",
         NotificationType.CONNECTOR_WARNING),
        (lambda n: n.nobody_synced(), "OneDrive: no OneDrive users synced", NotificationType.CONNECTOR_RECORD_SYNC_ERROR),
    ])
    async def test_every_notice_names_the_instance(self, send, title, kind) -> None:
        notices, notify, _ = self._notices()

        await send(notices)

        assert notify.await_args.kwargs["title"] == title
        assert notify.await_args.kwargs["type"] is kind


class TestGroupNesting:
    async def test_nesting_survives_a_save_and_load_as_json_text(self) -> None:
        point = _SyncPoint()
        nesting = GroupNesting(point.read, point.write)
        await nesting.load()
        nesting.record("g-eng", {"g-sre", "g-ops"})
        nesting.record("g-all", {"g-sre"})

        await nesting.save()
        again = GroupNesting(point.read, point.write)
        await again.load()

        assert isinstance(point.stored[STATE_FIELD], str)
        assert again.outer_groups("g-sre") == {"g-eng", "g-all"}
        assert again.outer_groups("g-ops") == {"g-eng"}

    async def test_a_stored_map_from_before_is_still_read(self) -> None:
        point = _SyncPoint({STATE_FIELD: {"g-sre": ["g-eng"]}})
        nesting = GroupNesting(point.read, point.write)

        await nesting.load()

        assert nesting.outer_groups("g-sre") == {"g-eng"}

    async def test_a_fresh_read_replaces_what_was_known_of_that_group(self) -> None:
        point = _SyncPoint()
        nesting = GroupNesting(point.read, point.write)
        await nesting.load()
        nesting.record("g-eng", {"g-sre", "g-ops"})
        nesting.record("g-all", {"g-ops"})

        nesting.record("g-eng", {"g-ops"})

        assert nesting.outer_groups("g-sre") == set()
        assert nesting.outer_groups("g-ops") == {"g-eng", "g-all"}

    async def test_a_deleted_group_is_forgotten_inside_and_out(self) -> None:
        point = _SyncPoint()
        nesting = GroupNesting(point.read, point.write)
        await nesting.load()
        nesting.record("g-eng", {"g-sre"})
        nesting.record("g-sre", {"g-db"})

        nesting.forget("g-sre")

        assert nesting.outer_groups("g-sre") == set()
        assert nesting.outer_groups("g-db") == set()

    async def test_nothing_is_written_when_nothing_changed(self) -> None:
        point = _SyncPoint({STATE_FIELD: json.dumps({"g-sre": ["g-eng"]})})
        nesting = GroupNesting(point.read, point.write)
        await nesting.load()
        nesting.record("g-eng", {"g-sre"})
        nesting.forget("g-unknown")

        await nesting.save()

        assert point.writes == []

    async def test_no_stored_state_reads_as_no_nesting(self) -> None:
        point = _SyncPoint()
        nesting = GroupNesting(point.read, point.write)

        await nesting.load()

        assert nesting.outer_groups("g-sre") == set()


class TestDriveRegistry:
    async def test_drives_survive_a_save_and_load_as_json_text(self) -> None:
        point = _SyncPoint()
        drives = DriveRegistry(point.read, point.write)
        await drives.load()
        drives.remember("u-ana", "drive-ana")
        drives.remember("u-ben", "drive-ben")

        await drives.save()
        again = DriveRegistry(point.read, point.write)
        await again.load()

        assert isinstance(point.stored["drives"], str)
        assert again.users() == {"u-ana", "u-ben"}
        assert again.drive_of("u-ana") == "drive-ana"

    async def test_a_stored_map_from_before_is_still_read(self) -> None:
        point = _SyncPoint({"drives": {"u-ana": "drive-ana"}})
        drives = DriveRegistry(point.read, point.write)

        await drives.load()

        assert drives.drive_of("u-ana") == "drive-ana"

    async def test_a_forgotten_user_is_saved_without_their_drive(self) -> None:
        point = _SyncPoint()
        drives = DriveRegistry(point.read, point.write)
        await drives.load()
        drives.remember("u-ana", "drive-ana")
        drives.remember("u-ben", "drive-ben")
        await drives.save()

        drives.forget("u-ana")
        await drives.save()

        assert json.loads(point.stored["drives"]) == {"u-ben": "drive-ben"}

    async def test_nothing_is_written_when_nothing_changed(self) -> None:
        point = _SyncPoint({"drives": json.dumps({"u-ana": "drive-ana"})})
        drives = DriveRegistry(point.read, point.write)
        await drives.load()
        drives.remember("u-ana", "drive-ana")
        drives.forget("u-nobody")

        await drives.save()

        assert point.writes == []
        assert drives.drive_of("u-nobody") is None


def _odata_error(status: int) -> ODataError:
    error = ODataError()
    error.response_status_code = status
    return error


def _graph_client(answer: object) -> object:
    get = AsyncMock(side_effect=answer) if isinstance(answer, BaseException) else AsyncMock(return_value=answer)
    client = MagicMock()
    client.sites.by_site_id.return_value.permissions.get = get
    return client


def _grant(user: str, kind: str = "READ") -> SimpleNamespace:
    return SimpleNamespace(entity_type="USER", external_id=user, email=f"{user}@acme.com", type=kind)


class TestFolderAccess:
    async def test_a_folder_never_remembered_is_not_called_changed(self) -> None:
        access = FolderAccess(_SyncPoint().read, _SyncPoint().write)
        await access.load()

        assert access.changed("d1", [_grant("ben")]) is None

    async def test_a_narrowed_list_is_a_change_and_order_is_not(self) -> None:
        point = _SyncPoint()
        access = FolderAccess(point.read, point.write)
        await access.load()
        access.remember("d1", [_grant("ben"), _grant("cal")])

        assert access.changed("d1", [_grant("cal"), _grant("ben")]) is False
        assert access.changed("d1", [_grant("cal")]) is True
        assert access.changed("d1", [_grant("ben", "WRITE"), _grant("cal")]) is True

    async def test_it_survives_a_restart_and_is_kept_until_remembered(self) -> None:
        point = _SyncPoint()
        access = FolderAccess(point.read, point.write)
        await access.load()
        access.remember("d1", [_grant("ben")])
        await access.save()
        assert access.changed("d1", [_grant("cal")]) is True  # not remembered: still a change

        restarted = FolderAccess(point.read, point.write)
        await restarted.load()

        assert restarted.changed("d1", [_grant("ben")]) is False

    async def test_nothing_is_written_when_nothing_changed(self) -> None:
        point = _SyncPoint({"folderAccess": "not json"})
        access = FolderAccess(point.read, point.write)
        await access.load()

        await access.save()

        assert point.writes == []
        assert access.changed("d1", []) is None

