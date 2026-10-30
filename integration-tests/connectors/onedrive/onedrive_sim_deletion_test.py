# pyright: ignore-file

"""
OneDrive on the simulator – what leaves OneDrive, or the sync's scope, leaves PipesHub
=====================================================================================

  TC-ODSIM-DEL-001  A deleted file, and a deleted folder with what is under it, whether Graph reports every
                    child or the folder alone
  TC-ODSIM-DEL-002  A user leaves the tenant: what they shared stays while Microsoft keeps their OneDrive, they
                    lose their own access, and once the OneDrive is deleted its files go [F-USER-GONE]
  TC-ODSIM-DEL-003  A OneDrive that is deleted (404) takes its files with it [F-USER-GONE]
  TC-ODSIM-DEL-004  A file renamed out of the extension filter, or to a name without an extension, leaves
                    [F-FILTER-EXIT]
  TC-ODSIM-DEL-005  Narrowing the filter (a full sync) removes access to what it leaves out; widening brings it back
"""

import logging

import pytest
from connectors.onedrive.sim_tenants import base_tenant
from connectors.onedrive.sim_world import OneDriveWorld, succeeded

logger = logging.getLogger("onedrive-sim-deletion-test")


def ext_filter(extensions: list[str]) -> dict:
    return {"file_extensions": {"type": "multiselect", "operator": "in", "value": extensions}}


@pytest.mark.integration
@pytest.mark.onedrive_sim
@pytest.mark.asyncio(loop_scope="session")
class TestOneDriveSimDeletion:

    @pytest.mark.parametrize("children_reported", [True, False], ids=["children-reported", "folder-only"])
    async def test_tc_odsim_del_001_deletes(self, onedrive_world: OneDriveWorld, children_reported: bool) -> None:
        w = onedrive_world
        await w.start(base_tenant(), behaviours={"delta_deleted_folder_reports_children": children_reported})
        w.ops({"op": "delete", "item": "a-tobob"}, {"op": "delete", "item": "a-projects"})
        assert succeeded(await w.sync())
        await w.assert_matches_source("after a file and a folder were deleted")

    async def test_tc_odsim_del_002_user_leaves(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "delete_user", "user": "alice"})
        await w.sync()
        await w.assert_matches_source("after alice left (her OneDrive is kept for retention)")
        w.ops({"op": "remove_onedrive", "user": "alice"})
        await w.sync()
        await w.assert_matches_source("after her OneDrive was deleted")

    async def test_tc_odsim_del_003_onedrive_deleted(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "remove_onedrive", "user": "bob"})
        await w.sync()
        await w.assert_matches_source("after bob's OneDrive was deleted")

    async def test_tc_odsim_del_004_files_leaving_the_filter(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        w.scope_extensions = {"docx", "pdf", "txt"}
        await w.start(base_tenant(), filters={"sync": {"values": ext_filter(["docx", "pdf", "txt"])}})
        w.ops({"op": "rename", "item": "a-report", "name": "report.md"},
              {"op": "rename", "item": "a-plan", "name": "PLAN"})
        await w.sync()
        await w.assert_matches_source("after files were renamed out of the filter")

    async def test_tc_odsim_del_005_filter_narrowed_then_widened(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.scope_extensions = {"docx"}
        await w.set_filters({"sync": {"values": ext_filter(["docx"])}})
        await w.assert_matches_source("after narrowing to docx")
        w.scope_extensions = None
        await w.set_filters({"sync": {"values": {}}})
        await w.assert_matches_source("after widening again")
