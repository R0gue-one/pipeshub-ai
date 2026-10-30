# pyright: ignore-file

"""The OneDrive tenants of the onedrive_sim suite (shape: connectors-simulator's sharepoint/scenario.py,
"onedrives").

- alice, bob, carol, erin: members with a OneDrive and a PipesHub account;
- nolan: a member with a OneDrive and no PipesHub account (no suite creates one); he shares a file with alice;
- frank: no OneDrive (404); gina: her OneDrive is locked (423);
- gus: a guest, a PipesHub account only in tests that ask for guests;
- Engineering (security: bob, carol, the nested group Platform), Platform (security: erin, the nested group SRE),
  SRE (security: hank), Marketing (Microsoft 365: alice, nolan);
- alice's Projects folder is shared with Engineering; inside it Private stops inheriting and keeps alice only;
  single files are shared with bob, with Marketing and with the guest; one file has no extension.

A shared item breaks inheritance with a copy of its parent's access plus the new grant, as OneDrive does.
"""

from __future__ import annotations

import copy
from typing import Any

DOMAIN = "contoso.com"


def _shared(owner: str, *grants: tuple[str, str]) -> dict[str, Any]:
    return {"unique": True, "grants": [{"principal": f"user:{owner}", "roles": ["Full Control"]},
                                       *({"principal": p, "roles": [r]} for p, r in grants)]}


USERS = [
    {"key": "alice", "name": "Alice Admin"},
    {"key": "bob", "name": "Bob Builder"},
    {"key": "carol", "name": "Carol Coder"},
    {"key": "erin", "name": "Erin Platform"},
    {"key": "hank", "name": "Hank SRE"},
    {"key": "nolan", "name": "Nolan Outside"},
    {"key": "frank", "name": "Frank Nodrive"},
    {"key": "gina", "name": "Gina Locked"},
    {"key": "gus", "name": "Gus Guest", "type": "Guest", "mail": "gus@partner.example"},
    # No mailbox (unlicensed, say): Graph shows no email on grants to him.
    {"key": "hal", "name": "Hal Nomail", "mailbox": False},
]

# Who gets a PipesHub account (email -> display name): everyone but nolan.
ONEDRIVE_PEOPLE = {(u.get("mail") or f"{u['key']}@{DOMAIN}").lower(): u["name"] for u in USERS if u["key"] != "nolan"}

GROUPS = [
    {"key": "eng", "name": "Engineering", "members": ["user:bob", "user:carol", "group:platform"]},
    {"key": "platform", "name": "Platform", "members": ["user:erin", "group:sre"]},
    {"key": "sre", "name": "SRE", "members": ["user:hank"]},
    {"key": "marketing", "name": "Marketing", "unified": True, "members": ["user:alice", "user:nolan"],
     "owners": ["alice"]},
]

ONEDRIVES = [
    {"user": "alice", "tree": [
        {"file": "report.docx", "key": "a-report"},
        {"file": "README", "key": "a-noext"},
        {"folder": "Projects", "key": "a-projects", **_shared("alice", ("secgroup:eng", "Read")), "children": [
            {"file": "plan.pdf", "key": "a-plan"},
            {"folder": "Deep", "key": "a-deep", "children": [{"file": "notes.txt", "key": "a-notes"}]},
            {"folder": "Private", "key": "a-private", **_shared("alice"), "children": [
                {"file": "secret.xlsx", "key": "a-secret"}]}]},
        {"file": "to-bob.docx", "key": "a-tobob", **_shared("alice", ("user:bob", "Read"))},
        {"file": "campaign.pptx", "key": "a-campaign", **_shared("alice", ("m365:marketing", "Read"))},
        {"file": "for-guest.pdf", "key": "a-guest", **_shared("alice", ("user:gus", "Read"))},
    ]},
    {"user": "bob", "tree": [
        {"file": "notes.md", "key": "b-notes"},
        {"folder": "Team", "key": "b-team", **_shared("bob", ("user:carol", "Edit")), "children": [
            {"file": "todo.txt", "key": "b-todo"}]},
    ]},
    {"user": "carol", "tree": [{"file": "carol.txt", "key": "c-file"}]},
    {"user": "erin", "tree": [{"file": "erin.csv", "key": "e-file"}]},
    {"user": "nolan", "tree": [{"file": "nolan.docx", "key": "n-file", **_shared("nolan", ("user:alice", "Read"))}]},
    {"user": "gina", "status": "locked", "tree": [{"file": "gina.txt", "key": "g-file"}]},
]

# What docs.pipeshub.com tells customers to grant for OneDrive: Graph Files.Read.All, User.Read.All,
# Group.Read.All, and the SharePoint API's Sites.Read.All for the certificate.
APP = {"graph_roles": ["Files.Read.All", "User.Read.All", "Group.Read.All"], "sharepoint_roles": ["Sites.Read.All"]}


def base_tenant(**overrides: Any) -> dict[str, Any]:
    tenant = {"seed": 23, "tenant": "contoso", "domain": DOMAIN, "users": copy.deepcopy(USERS),
              "groups": copy.deepcopy(GROUPS), "onedrives": copy.deepcopy(ONEDRIVES), "app": copy.deepcopy(APP),
              # Read by the suite, not the simulator.
              "not_pipeshub": ["nolan"]}
    tenant.update(overrides)
    return tenant
