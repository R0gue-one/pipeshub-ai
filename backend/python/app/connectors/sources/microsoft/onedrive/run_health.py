"""What one OneDrive sync run could not read.

A run goes on past a OneDrive it cannot read, so the other users' drives still sync. Two things must then not
happen: the run reporting success, and a full sync removing the records of the drive it did not reach (the
platform unlinks whatever a full sync it is told was complete did not write). This collects those parts so
the connector can say both, and so the admin can be told which ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OneDriveSyncIncomplete(Exception):
    """Raised at the end of a run that skipped part of the source, so the run is recorded as failed."""


@dataclass
class RunHealth:
    # What could not be read, one entry per part ("OneDrive of alice@contoso.com: <error>").
    problems: list[str] = field(default_factory=list)
    # Users whose OneDrive was skipped, for the notice; a subset of what `problems` describes.
    skipped_users: list[str] = field(default_factory=list)

    def skipped(self, what: str, error: object) -> None:
        self.problems.append(f"{what}: {error}")

    def skipped_drive(self, user: str, error: object) -> None:
        self.skipped_users.append(user)
        self.skipped(f"OneDrive of {user}", error)

    @property
    def complete(self) -> bool:
        return not self.problems

    def summary(self, limit: int = 5) -> str:
        shown = "; ".join(self.problems[:limit])
        more = len(self.problems) - limit
        return shown + (f"; and {more} more" if more > 0 else "")

    def raise_if_incomplete(self) -> None:
        if self.problems:
            raise OneDriveSyncIncomplete(
                f"OneDrive sync skipped {len(self.problems)} part(s) of the source: {self.summary()}"
            )
