"""The notices a OneDrive connector sends its admin, each naming the connector instance.

One job: word and send them. When to send is the caller's decision; repeats of the same notice are held back
by the base connector's backoff (BaseConnector._suppress_notification). A sync that worked sends nothing: a
notice on every scheduled run (hourly by default) teaches people to ignore the ones that matter.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.services.notification.types import NotificationSeverity, NotificationType

Notify = Callable[..., Awaitable[Any]]


def _listed(names: list[str], limit: int = 5) -> str:
    shown = ", ".join(names[:limit])
    return f"{shown}, and {len(names) - limit} more" if len(names) > limit else shown


class OneDriveNotices:
    def __init__(self, notify: Notify, title: Callable[[str], str], payload: Callable[[], dict[str, Any]],
                 recipients: Callable[[], list[str]]) -> None:
        """`title` prefixes the instance name (BaseConnector._notification_title); `payload` and `recipients` are
        read when a notice is sent, since the instance name and creator are set after construction."""
        self._notify = notify
        self._title = title
        self._payload = payload
        self._recipients = recipients

    async def _send(self, kind: NotificationType, severity: NotificationSeverity, event: str, message: str) -> None:
        await self._notify(type=kind, severity=severity, title=self._title(event), message=message,
                           payload=self._payload(), recipient_user_ids=self._recipients())

    async def sign_in_failed(self) -> None:
        """One wording for every way the sign-in fails (refused at sign-in, or the token refused by Graph), so the
        backoff treats them as one problem; the specifics go to the log."""
        await self._send(NotificationType.CONNECTOR_AUTH_ERROR, NotificationSeverity.ERROR,
                         "Microsoft refused the sign-in",
                         "Microsoft refused the connector's credentials: the client secret or certificate may have "
                         "expired or been removed, or the app's access was revoked. Update the credentials on the "
                         "connector. Nothing was removed; syncing resumes once the sign-in works.")

    async def drives_skipped(self, users: list[str], detail: str) -> None:
        await self._send(NotificationType.CONNECTOR_RECORD_SYNC_ERROR, NotificationSeverity.WARNING,
                         "some OneDrives could not be synced",
                         f"The OneDrives of {_listed(users)} could not be read this time ({detail}). Their files "
                         "keep their last synced state and access, and are read again on the next sync.")

    async def part_skipped(self, detail: str) -> None:
        await self._send(NotificationType.CONNECTOR_RECORD_SYNC_ERROR, NotificationSeverity.WARNING,
                         "part of the sync could not be read",
                         f"{detail}. What was synced before is kept; it is read again on the next sync.")

    async def drives_read_again(self, users: list[str]) -> None:
        await self._send(NotificationType.CONNECTOR_WARNING, NotificationSeverity.INFO,
                         "OneDrives read again from the start",
                         f"Microsoft no longer had the change history of the OneDrives of {_listed(users)}, so "
                         "they were read again in full. Files deleted in the meantime were removed.")

    async def users_unavailable(self, event: str, message: str) -> None:
        await self._send(NotificationType.CONNECTOR_WARNING, NotificationSeverity.WARNING, event, message)

    async def nobody_synced(self) -> None:
        await self._send(NotificationType.CONNECTOR_RECORD_SYNC_ERROR, NotificationSeverity.WARNING,
                         "no OneDrive users synced",
                         "Invite your OneDrive users to PipesHub, and check that the app has the Files.Read.All "
                         "application permission with admin consent.")
