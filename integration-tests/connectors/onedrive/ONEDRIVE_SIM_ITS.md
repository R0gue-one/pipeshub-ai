# OneDrive simulator integration tests

These tests run the real PipesHub stack against simulated OneDrives. Each test builds a small tenant, syncs it,
changes it, syncs again, and checks that every person can open in PipesHub exactly what OneDrive lets them read.
Marker: `onedrive_sim`.

## How the suite works

It reuses the SharePoint simulator suite's set-up (see `connectors/sharepoint/SHAREPOINT_SIM_ITS.md`): the
`sharepoint` simulator of connectors-simulator also answers Graph's OneDrive routes, and the compose override
`deployment/docker-compose/docker-compose.integration.sharepoint-sim.yml` already sends `graph.microsoft.com`,
`login.microsoftonline.com` and `contoso-my.sharepoint.com` to it through a TLS front. Nothing in PipesHub
changes.

**People.** The connector syncs the OneDrives of tenant users who have a PipesHub account. Every person of
`sim_tenants.py` except `nolan` gets one (fixture `onedrive_sim_people`); `nolan` has a OneDrive and shares a file
with alice, so a test can prove an account-less user's drive is not synced and leaks nothing. No other suite
creates a `nolan` account.

**The check.** As in the SharePoint suite: after each sync, the simulator's oracle says who may read each file;
each person, logged in, opens every record of the connector. LEAK, DENIED, MISSING, STALE, EXTRA, DUPLICATE,
UNSETTLED, SIM_UNMODELED and SIM_SPEC fail the test. A locked OneDrive is left out of the comparison: nobody can
open it now, and it may be unlocked.

**Notices.** `OneDriveWorld.notices()` reads the creator's notifications through `GET /api/v1/notifications`
and keeps those titled `<instance name>: ...`.

**Files.** `sim_world.py` (the world and fixtures; the simulator client and helpers come from the SharePoint
suite), `sim_tenants.py`, and the tests: `onedrive_sim_sync_test.py` (6), `onedrive_sim_access_test.py` (9),
`onedrive_sim_deletion_test.py` (6), `onedrive_sim_failure_test.py` (7).

Each test names the audit finding it guards (od-audit/REPORT.md); the harness scenario that proves the same thing
on the real connector is listed in od-audit/IT_CASES.md.

## Running it

The OneDrive routes are on connectors-simulator branch `sim/onedrive`, not on its `main` yet. Until that is
merged, build the simulator image from that branch:

```bash
cd connectors-simulator && git checkout sim/onedrive && docker build -t pipeshub/connectors-simulator:it .
```

Then bring the stack up with the SharePoint override and run, as for `sharepoint_sim`:

```bash
set -a; . integration-tests/connectors/sharepoint_sim.env; set +a
cd integration-tests && pytest -m onedrive_sim connectors/onedrive
```

## In CI

The suite runs on shard `connectors-3` (`CONN_SHARD_3` in `.github/workflows/integration-tests.yml`), next to
`sharepoint_sim`, which starts the simulator and its TLS front for both. A manual run can pick `onedrive_sim` alone.
`CONNSIM_SHAREPOINT_REF` pins connectors-simulator's `sim/onedrive` branch; move it to that repository's `main` once
the OneDrive routes are merged there.
