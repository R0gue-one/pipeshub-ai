# pyright: ignore-file

"""
OneDrive on the simulator – who may read what
=============================================

  TC-ODSIM-ACL-001  A disabled user loses everything; enabled again, gets it back [F-DISABLED-USER]
  TC-ODSIM-ACL-002  A member of a group nested two levels down reads what the outer group was given; leaving the
                    inner group takes it away [F-NESTED-GROUPS]
  TC-ODSIM-ACL-003  A member removed from a group, and a deleted group, take the group's access away
  TC-ODSIM-ACL-004  A file shared and unshared with one person, and a folder unshared from a group, with the
                    app holding Graph Sites.FullControl.All (Microsoft reports sharing changes then)
  TC-ODSIM-ACL-005  Without Sites.FullControl.All (what the docs ask for) sharing changes reach PipesHub on the next
                    sync, with no notice about the permission [F-SHARING-CHANGES]
  TC-ODSIM-ACL-006  A guest reads only what is shared with them
  TC-ODSIM-ACL-007  A user without a PipesHub account: their OneDrive is not synced, and nothing leaks
  TC-ODSIM-ACL-008  One person removed from a folder still shared with others loses its files [F-SHARING-CHANGES]
  TC-ODSIM-ACL-009  A file shared with a user who has no mailbox (no email on the grant) [F-NO-EMAIL-GRANT]
"""

import logging

import pytest
from connectors.onedrive.sim_tenants import APP, base_tenant
from connectors.onedrive.sim_world import OneDriveWorld, succeeded

logger = logging.getLogger("onedrive-sim-access-test")

FULL_CONTROL = {**APP, "graph_roles": [*APP["graph_roles"], "Sites.FullControl.All"]}


@pytest.mark.integration
@pytest.mark.onedrive_sim
@pytest.mark.asyncio(loop_scope="session")
class TestOneDriveSimAccess:

    async def test_tc_odsim_acl_001_disabled_user_loses_access(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "set_user", "user": "bob", "enabled": False})
        await w.sync()
        await w.assert_matches_source("after bob was disabled")
        w.ops({"op": "set_user", "user": "bob", "enabled": True})
        await w.sync()
        await w.assert_matches_source("after bob was enabled again")

    async def test_tc_odsim_acl_002_nested_groups(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        await w.assert_matches_source("hank, in SRE inside Platform inside Engineering")
        w.ops({"op": "remove_member", "group": "platform", "member": "erin"})
        await w.sync()
        await w.assert_matches_source("after erin left Platform")

    async def test_tc_odsim_acl_003_group_changes(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "remove_member", "group": "eng", "member": "bob"})
        await w.sync()
        await w.assert_matches_source("after bob left Engineering")
        w.ops({"op": "delete_group", "group": "eng"})
        await w.sync()
        await w.assert_matches_source("after Engineering was deleted")

    async def test_tc_odsim_acl_004_sharing_changes(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant(app=FULL_CONTROL))
        w.ops({"op": "grant", "target": "a-report", "kind": "user", "roles": ["Read"], "ref": "carol"},
              {"op": "revoke", "target": "a-tobob", "kind": "user", "ref": "bob"},
              {"op": "revoke", "target": "a-projects", "kind": "secgroup", "ref": "eng"})
        await w.sync()
        await w.assert_matches_source("after the sharing changes")

    async def test_tc_odsim_acl_005_sharing_changes_without_full_control(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        outcome = await w.start(base_tenant())
        assert succeeded(outcome), f"the first sync was recorded as {outcome}"
        w.ops({"op": "grant", "target": "a-report", "kind": "user", "roles": ["Read"], "ref": "carol"},
              {"op": "revoke", "target": "a-tobob", "kind": "user", "ref": "bob"},
              {"op": "revoke", "target": "a-projects", "kind": "secgroup", "ref": "eng"})
        outcome = await w.sync()
        assert succeeded(outcome), f"the sync after the sharing changes was recorded as {outcome}"
        await w.assert_matches_source("after the sharing changes, without Sites.FullControl.All")
        asked = [n for n in await w.notices("Sites.FullControl.All", timeout=10)
                 if "Sites.FullControl.All" in f"{n.get('title', '')} {n.get('message', '')}"]
        assert not asked, f"the admin was asked for a permission PipesHub doesn't need: {[n.get('title') for n in asked]}"

    async def test_tc_odsim_acl_006_guest(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant(), guests=True)
        await w.assert_matches_source("with the guest compared")

    async def test_tc_odsim_acl_007_user_without_account(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        records = {r["externalRecordId"] for r in await w.records()}
        assert w.key("n-file") not in records, "the OneDrive of a user without a PipesHub account was synced"
        await w.assert_matches_source("with nolan outside PipesHub")

    async def test_tc_odsim_acl_008_folder_narrowed_but_still_shared(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "grant", "target": "a-projects", "kind": "user", "roles": ["Read"], "ref": "frank"})
        await w.sync()
        await w.assert_matches_source("after Projects was also shared with frank")
        w.ops({"op": "revoke", "target": "a-projects", "kind": "user", "ref": "frank"})
        await w.sync()
        await w.assert_matches_source("after frank was removed from Projects (Engineering keeps it)")

    async def test_tc_odsim_acl_009_user_without_a_mailbox(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "grant", "target": "a-report", "kind": "user", "roles": ["Read"], "ref": "hal"})
        await w.sync()
        await w.assert_matches_source("after report.docx was shared with hal")
        w.ops({"op": "revoke", "target": "a-report", "kind": "user", "ref": "hal"})
        await w.sync()
        await w.assert_matches_source("after report.docx was unshared from hal")

