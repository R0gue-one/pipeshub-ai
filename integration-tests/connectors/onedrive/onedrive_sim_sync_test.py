# pyright: ignore-file

"""
OneDrive on the simulator – syncing
===================================

The real PipesHub stack syncs users' OneDrives from the simulator. After each sync every person opens exactly
the records OneDrive lets them read. The finding each test proves is in brackets (od-audit/REPORT.md); those
fail on upstream main and pass with the audit's fixes.

  TC-ODSIM-SYNC-001  First sync: every file of every PipesHub user's OneDrive, with its access and parents
  TC-ODSIM-SYNC-002  Incremental: a new file, an edit, a rename and a move; the rename keeps the record
  TC-ODSIM-SYNC-003  A drive delta token Graph no longer honours (410): the drive is read again, files
                     added meanwhile arrive and deleted ones go [F-RESYNC]
  TC-ODSIM-SYNC-004  A group delta token Graph no longer honours: groups are read again and syncing goes on
                     [F-GROUP-RESYNC]
  TC-ODSIM-SYNC-005  A file deleted while the connector was off is removed by a full sync
  TC-ODSIM-SYNC-006  Opening a file streams OneDrive's bytes; a file deleted at the source is not found
"""

import logging

import pytest
from connectors.onedrive.sim_tenants import base_tenant
from connectors.onedrive.sim_world import OneDriveWorld, succeeded

logger = logging.getLogger("onedrive-sim-sync-test")


@pytest.mark.integration
@pytest.mark.onedrive_sim
@pytest.mark.asyncio(loop_scope="session")
class TestOneDriveSimSync:

    async def test_tc_odsim_sync_001_first_sync_matches_source(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        outcome = await w.start(base_tenant())
        assert succeeded(outcome), f"the first sync was recorded as {outcome}"
        await w.assert_matches_source("after the first sync")
        await w.assert_hierarchy_matches_source("after the first sync")

    async def test_tc_odsim_sync_002_incremental_changes(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        before = await w.record_of("a-report")
        w.ops({"op": "add_file", "parent": "a-deep", "name": "new.txt", "key": "a-new"},
              {"op": "edit_file", "item": "a-plan", "size": 9000},
              {"op": "rename", "item": "a-report", "name": "report-2027.docx"},
              {"op": "move", "item": "b-notes", "parent": "b-team"})
        assert succeeded(await w.sync()), "the incremental sync failed"
        await w.assert_matches_source("after the changes")
        after = await w.record_of("a-report")
        assert before is not None and after is not None and after.id == before.id, "a rename made a new record"
        await w.assert_hierarchy_matches_source("after the move")

    async def test_tc_odsim_sync_003_expired_drive_token_reads_the_drive_again(
        self, onedrive_world: OneDriveWorld,
    ) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.expire_delta_tokens("drive")
        w.ops({"op": "add_file", "parent": "alice-docs", "name": "after-expiry.txt", "key": "a-after"},
              {"op": "delete", "item": "a-campaign"})
        await w.sync()
        assert succeeded(await w.sync()), "the drives do not sync again after their delta tokens expired"
        await w.assert_matches_source("after the drive delta tokens expired")

    async def test_tc_odsim_sync_004_expired_group_token_reads_the_groups_again(
        self, onedrive_world: OneDriveWorld,
    ) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.expire_delta_tokens("groups")
        w.ops({"op": "remove_member", "group": "eng", "member": "bob"})
        await w.sync()
        assert succeeded(await w.sync()), "syncs keep failing after the group delta token expired"
        await w.assert_matches_source("after the group delta token expired")

    async def test_tc_odsim_sync_005_full_sync_removes_what_was_deleted_while_off(
        self, onedrive_world: OneDriveWorld,
    ) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "delete", "item": "a-plan"}, {"op": "delete", "item": "b-team"})
        assert succeeded(await w.sync(full=True)), "the full sync failed"
        await w.assert_matches_source("after the full sync")

    async def test_tc_odsim_sync_006_streaming(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        record = await w.record_of("a-plan")
        assert record is not None
        status, body = w.stream(record.id, w.reader_of("a-plan"))
        assert status == 200 and body == w.sim.content(w.key("a-plan")), f"streaming answered {status}"
        w.ops({"op": "delete", "item": "a-plan"})
        status, _ = w.stream(record.id, w.reader_of("a-report"))
        assert status == 404, f"a file deleted at the source streamed as {status}"
