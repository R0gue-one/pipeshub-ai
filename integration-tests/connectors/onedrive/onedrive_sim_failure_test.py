# pyright: ignore-file

"""
OneDrive on the simulator – syncs that cannot read everything
==============================================================

What PipesHub knew must survive a failed read, a run that skipped part of the source must not be recorded as a
success, and the admin must be told once.

  TC-ODSIM-FAIL-001  A full sync that cannot read one OneDrive (its delta, or the check of the drive) removes
                     nothing and is not a success [F-SILENT-FAILURE]
  TC-ODSIM-FAIL-002  An incremental sync that cannot read one OneDrive is not a success, the admin is told, and
                     the next sync catches up [F-SILENT-FAILURE, F-NOTIFY]
  TC-ODSIM-FAIL-003  A failed permission read keeps the stored access
  TC-ODSIM-FAIL-004  A throttled tenant (429 with Retry-After) finishes
  TC-ODSIM-FAIL-005  An app without Files.Read.All fails the connection test, so the connector is not enabled
                     to sync nothing [F-CONN-TEST]
  TC-ODSIM-FAIL-006  Clean syncs send no notice [F-NOTIFY]
"""

import logging

import pytest
from connectors.onedrive.sim_tenants import base_tenant
from connectors.onedrive.sim_world import OneDriveWorld, fault, succeeded, user_path
from pipeshub_client import PipeshubClientError  # type: ignore[import-not-found]

logger = logging.getLogger("onedrive-sim-failure-test")


@pytest.mark.integration
@pytest.mark.onedrive_sim
@pytest.mark.asyncio(loop_scope="session")
class TestOneDriveSimFailures:

    @pytest.mark.parametrize("target", ["delta", "drive"], ids=["delta", "drive-check"])
    async def test_tc_odsim_fail_001_full_sync_with_a_failed_drive_removes_nothing(
        self, onedrive_world: OneDriveWorld, target: str,
    ) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        path = user_path(w, "alice") + ("/root/delta" if target == "delta" else "$")
        w.faults(fault(path, 503))
        outcome = await w.sync(full=True)
        w.clear_faults()
        assert not succeeded(outcome), f"a full sync that could not read alice's OneDrive was recorded as {outcome}"
        await w.assert_matches_source("after a full sync that could not read alice's OneDrive")

    async def test_tc_odsim_fail_002_incremental_failure_is_reported(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "add_file", "parent": "alice-docs", "name": "x.txt", "key": "a-x"})
        w.faults(fault(user_path(w, "alice") + "/root/delta", 503))
        outcome = await w.sync()
        w.clear_faults()
        assert not succeeded(outcome), f"a sync that could not read alice's OneDrive was recorded as {outcome}"
        notices = await w.notices("could not be synced")
        assert any("could not be synced" in n.get("title", "") for n in notices), (
            f"the admin was not told: {[n.get('title') for n in notices]}"
        )
        assert succeeded(await w.sync()), "the next sync did not catch up"
        await w.assert_matches_source("after the outage")

    async def test_tc_odsim_fail_003_permission_read_failure_keeps_access(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant())
        w.ops({"op": "edit_file", "item": "a-tobob", "size": 4000})
        w.faults(fault(r"/items/[^/]+/permissions$", 503))
        await w.sync()
        w.clear_faults()
        await w.assert_matches_source("during a permission read outage")

    async def test_tc_odsim_fail_004_throttled_tenant_finishes(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        await w.start(base_tenant(), enable=False)
        w.faults({"match": {"path_regex": r"^/v1\.0/"},
                  "effect": {"type": "rate_limit", "capacity": 6, "refill_per_s": 6}})
        outcome = await w.enable()
        w.clear_faults()
        assert succeeded(outcome), f"a throttled first sync was recorded as {outcome}"
        await w.assert_matches_source("after a throttled first sync")

    async def test_tc_odsim_fail_005_connection_test_names_files_read_all(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        tenant = base_tenant(app={"graph_roles": ["User.Read.All", "Group.Read.All"],
                                  "sharepoint_roles": ["Sites.Read.All"]})
        await w.start(tenant, enable=False)
        with pytest.raises(PipeshubClientError, match="Files.Read.All"):
            w.client.toggle_sync(w.connector_id, enable=True)

    async def test_tc_odsim_fail_006_clean_syncs_send_no_notice(self, onedrive_world: OneDriveWorld) -> None:
        w = onedrive_world
        tenant = base_tenant(app={"graph_roles": ["Files.Read.All", "User.Read.All", "Group.Read.All",
                                                  "Sites.FullControl.All"], "sharepoint_roles": ["Sites.Read.All"]})
        # Every tenant user with a PipesHub account has a OneDrive that opens: frank (none), gina (locked) and
        # the guest gus (none; another test may have given him an account) go, and hank and hal, who have none
        # in the base tenant, get an empty one. Each of those cases is a notice of its own.
        tenant["users"] = [u for u in tenant["users"] if u["key"] not in ("frank", "gina", "gus")]
        tenant["onedrives"] = [o for o in tenant["onedrives"] if o["user"] != "gina"]
        tenant["onedrives"] += [{"user": "hank"}, {"user": "hal"}]
        alice = next(o for o in tenant["onedrives"] if o["user"] == "alice")
        alice["tree"] = [i for i in alice["tree"] if i.get("key") != "a-guest"]  # shared with gus
        await w.start(tenant)
        for _ in range(2):
            assert succeeded(await w.sync())
        notices = await w.notices("\0", timeout=15)
        assert not notices, f"clean syncs sent notices: {[n.get('title') for n in notices]}"
