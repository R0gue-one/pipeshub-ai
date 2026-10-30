"""Unit tests for OneDrive connector notification triggers."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from msgraph.generated.models.o_data_errors.main_error import MainError
from msgraph.generated.models.o_data_errors.o_data_error import ODataError

from app.connectors.core.base.connector.connector_service import ConnectorInitError
from app.connectors.sources.microsoft.onedrive.connector import (
    OneDriveConnector,
    OneDriveUserStatus,
)
from app.connectors.sources.microsoft.onedrive.run_health import OneDriveSyncIncomplete
from app.services.notification.types import NotificationSeverity, NotificationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connector() -> OneDriveConnector:
    logger = logging.getLogger("test.onedrive.notifications")
    data_entities_processor = MagicMock()
    data_entities_processor.org_id = "org-123"
    data_entities_processor.on_new_app_users = AsyncMock()
    data_entities_processor.get_all_active_users = AsyncMock(return_value=[])

    tx_store = AsyncMock()
    tx_store.get_sync_point = AsyncMock(return_value=None)
    data_store_provider = MagicMock()
    data_store_provider.transaction.return_value.__aenter__.return_value = tx_store

    # Patched on the class while constructing: OneDriveNotices keeps the notify it is given.
    with patch.object(OneDriveConnector, "notify", new_callable=AsyncMock) as notify:
        connector = OneDriveConnector(
            logger,
            data_entities_processor,
            data_store_provider,
            MagicMock(),
            "conn-onedrive-notify",
            "team",
            "creator-user-id",
        )
    connector.notify = notify
    return connector


def _make_403_odata_error() -> ODataError:
    err = ODataError()
    err.error = MainError()
    err.error.code = "Authorization_RequestDenied"
    err.response_status_code = 403
    return err


def _make_transient_odata_error() -> ODataError:
    err = ODataError()
    err.error = MainError()
    err.error.code = "ServiceUnavailable"
    err.error.message = "Microsoft Graph is temporarily unavailable"
    err.response_status_code = 503
    return err


def _expected_payload(connector: OneDriveConnector) -> dict:
    return {
        "connector_id": connector.connector_id,
        "connector_name": connector.display_name,
        "connector_scope": connector.scope,
    }


SIGN_IN_FAILED_TITLE = "OneDrive: Microsoft refused the sign-in"
SIGN_IN_FAILED_MESSAGE = (
    "Microsoft refused the connector's credentials: the client secret or certificate may have "
    "expired or been removed, or the app's access was revoked. Update the credentials on the "
    "connector. Nothing was removed; syncing resumes once the sign-in works."
)


def _notify_calls_by_title(connector: OneDriveConnector) -> dict:
    return {call.kwargs["title"]: call.kwargs for call in connector.notify.await_args_list}


# Sentinel: assert recipient_user_ids == [connector.created_by] (the default
# behavior for most notify() call sites in this connector).
_DEFAULT_RECIPIENT = object()


def _assert_notify(
    connector: OneDriveConnector,
    *,
    notification_type: NotificationType,
    severity: NotificationSeverity,
    title: str,
    message: str | None = None,
    message_contains: str | None = None,
    total_calls: int = 1,
    recipient_user_ids: object = _DEFAULT_RECIPIENT,
) -> None:
    assert connector.notify.await_count == total_calls, (
        f"Expected {total_calls} notify call(s), got {connector.notify.await_count}: "
        f"{[c.kwargs.get('title') for c in connector.notify.await_args_list]}"
    )
    kwargs = _notify_calls_by_title(connector).get(title)
    assert kwargs is not None, (
        f"No notify call with title {title!r}; "
        f"got titles={[c.kwargs.get('title') for c in connector.notify.await_args_list]}"
    )
    assert kwargs["type"] is notification_type
    assert kwargs["severity"] is severity
    if message is not None:
        assert kwargs["message"] == message
    if message_contains is not None:
        assert message_contains in kwargs["message"]
    assert kwargs["payload"] == _expected_payload(connector)
    if recipient_user_ids is _DEFAULT_RECIPIENT:
        assert kwargs["recipient_user_ids"] == [connector.created_by]
    elif recipient_user_ids is None:
        # Call site intentionally omits recipient_user_ids so notify()'s own
        # default (creator + last_synced_by) applies.
        assert "recipient_user_ids" not in kwargs
    else:
        assert kwargs["recipient_user_ids"] == recipient_user_ids


# ---------------------------------------------------------------------------
# init()
# ---------------------------------------------------------------------------


class TestInitNotifications:

    @pytest.mark.asyncio
    async def test_auth_failure_notifies_and_raises(self):
        connector = _make_connector()
        connector.config_service.get_config = AsyncMock(return_value={
            "auth": {
                "tenantId": "tenant",
                "clientId": "client",
                "clientSecret": "secret",
                "hasAdminConsent": False,
            }
        })

        with patch(
            "app.connectors.sources.microsoft.onedrive.connector.ClientSecretCredential"
        ) as mock_cred_cls:
            mock_cred = AsyncMock()
            mock_cred.get_token = AsyncMock(side_effect=Exception("token fail"))
            mock_cred_cls.return_value = mock_cred

            with pytest.raises(ValueError, match="Failed to initialize OneDrive credential"):
                await connector.init()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_AUTH_ERROR,
            severity=NotificationSeverity.ERROR,
            title=SIGN_IN_FAILED_TITLE,
            message=SIGN_IN_FAILED_MESSAGE,
        )


# ---------------------------------------------------------------------------
# _sync_users()
# ---------------------------------------------------------------------------


class TestSyncUsersNotifications:

    @pytest.mark.asyncio
    async def test_403_notifies_and_raises(self):
        connector = _make_connector()
        connector.msgraph_client = MagicMock()
        connector.msgraph_client.get_all_users = AsyncMock(
            side_effect=_make_403_odata_error()
        )

        with pytest.raises(ODataError):
            await connector._sync_users()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_USER_SYNC_ERROR,
            severity=NotificationSeverity.ERROR,
            title="OneDrive: users sync failed",
            message=(
                "Please check your authentication credentials are correct "
                "and has User.Read.All API permission."
            ),
        )

    @pytest.mark.asyncio
    async def test_transient_error_does_not_notify(self):
        connector = _make_connector()
        connector.msgraph_client = MagicMock()
        connector.msgraph_client.get_all_users = AsyncMock(
            side_effect=_make_transient_odata_error()
        )

        with pytest.raises(ODataError):
            await connector._sync_users()

        connector.notify.assert_not_called()


# ---------------------------------------------------------------------------
# _sync_user_groups()
# ---------------------------------------------------------------------------


class TestSyncUserGroupsNotifications:

    @pytest.mark.asyncio
    async def test_403_notifies_and_raises(self):
        connector = _make_connector()
        connector.user_group_sync_point = MagicMock()
        connector.user_group_sync_point.read_sync_point = AsyncMock(return_value=None)
        connector._get_initial_delta_link = AsyncMock(side_effect=_make_403_odata_error())

        with pytest.raises(ODataError):
            await connector._sync_user_groups()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_AUTH_ERROR,
            severity=NotificationSeverity.ERROR,
            title="OneDrive: user group sync failed",
            message=(
                "Please check your authentication credentials are correct "
                "and has Group.Read.All API permission."
            ),
        )

    @pytest.mark.asyncio
    async def test_transient_error_does_not_notify(self):
        connector = _make_connector()
        connector.user_group_sync_point = MagicMock()
        connector.user_group_sync_point.read_sync_point = AsyncMock(return_value=None)
        connector._get_initial_delta_link = AsyncMock(side_effect=_make_transient_odata_error())

        with pytest.raises(ODataError):
            await connector._sync_user_groups()

        connector.notify.assert_not_called()


# ---------------------------------------------------------------------------
# _process_users_in_batches()
# ---------------------------------------------------------------------------


class TestProcessUsersInBatchesNotifications:

    @pytest.mark.asyncio
    async def test_user_drive_403_is_reported_once_when_the_run_ends(self):
        connector = _make_connector()
        active_user = MagicMock()
        active_user.email = "active@test.com"
        connector.data_entities_processor.get_all_active_users = AsyncMock(
            return_value=[active_user]
        )
        user = MagicMock()
        user.email = "active@test.com"
        user.source_user_id = "su-1"
        connector.msgraph_client = MagicMock()
        connector.msgraph_client.get_user_drive = AsyncMock(
            side_effect=_make_403_odata_error()
        )

        await connector._process_users_in_batches([user])

        connector.msgraph_client.get_user_drive.assert_awaited_once_with("su-1")
        connector.notify.assert_not_called()
        assert connector._health.skipped_users == ["active@test.com"]
        assert connector.full_sync_coverage().complete is False

        with pytest.raises(OneDriveSyncIncomplete):
            await connector._finish_run()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_RECORD_SYNC_ERROR,
            severity=NotificationSeverity.WARNING,
            title="OneDrive: some OneDrives could not be synced",
            message_contains="active@test.com",
        )

    @pytest.mark.asyncio
    async def test_user_drive_transient_error_is_reported_once_when_the_run_ends(self):
        connector = _make_connector()
        active_user = MagicMock()
        active_user.email = "active@test.com"
        connector.data_entities_processor.get_all_active_users = AsyncMock(
            return_value=[active_user]
        )
        user = MagicMock()
        user.email = "active@test.com"
        user.source_user_id = "su-1"
        connector.msgraph_client = MagicMock()
        connector.msgraph_client.get_user_drive = AsyncMock(
            side_effect=_make_transient_odata_error()
        )

        await connector._process_users_in_batches([user])

        connector.msgraph_client.get_user_drive.assert_awaited_once_with("su-1")
        connector.notify.assert_not_called()
        assert connector._health.skipped_users == ["active@test.com"]
        assert connector.full_sync_coverage().complete is False

        with pytest.raises(OneDriveSyncIncomplete):
            await connector._finish_run()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_RECORD_SYNC_ERROR,
            severity=NotificationSeverity.WARNING,
            title="OneDrive: some OneDrives could not be synced",
            message_contains="read again on the next sync",
        )

    @pytest.mark.asyncio
    async def test_user_without_provisioned_onedrive_warns(self):
        connector = _make_connector()

        active_user = MagicMock()
        active_user.email = "active@test.com"
        connector.data_entities_processor.get_all_active_users = AsyncMock(
            return_value=[active_user]
        )

        user = MagicMock()
        user.email = "active@test.com"
        user.source_user_id = "su-1"
        connector._user_has_onedrive = AsyncMock(
            return_value=OneDriveUserStatus.NOT_PROVISIONED
        )
        connector._run_sync_with_yield = AsyncMock()

        await connector._process_users_in_batches([user])

        # The user is named in this notice, so the "no OneDrive users synced" fallback stays quiet.
        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_WARNING,
            severity=NotificationSeverity.WARNING,
            title="OneDrive: OneDrive not provisioned for some users",
            message_contains="active@test.com",
        )

    @pytest.mark.asyncio
    async def test_no_user_with_a_pipeshub_account_sends_the_fallback(self):
        connector = _make_connector()
        user = MagicMock()
        user.email = "outsider@test.com"
        user.source_user_id = "su-1"

        await connector._process_users_in_batches([user])

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_RECORD_SYNC_ERROR,
            severity=NotificationSeverity.WARNING,
            title="OneDrive: no OneDrive users synced",
            message_contains="Files.Read.All",
        )

    @pytest.mark.asyncio
    async def test_users_with_onedrive_does_not_warn(self):
        connector = _make_connector()

        active_user = MagicMock()
        active_user.email = "active@test.com"
        connector.data_entities_processor.get_all_active_users = AsyncMock(
            return_value=[active_user]
        )

        user = MagicMock()
        user.email = "active@test.com"
        user.source_user_id = "su-1"
        connector._user_has_onedrive = AsyncMock(
            return_value=OneDriveUserStatus.AVAILABLE
        )
        connector._run_sync_with_yield = AsyncMock()

        await connector._process_users_in_batches([user])

        connector.notify.assert_not_called()
        assert connector.onedrive_users_synced == 1

    @pytest.mark.asyncio
    async def test_failure_notifications_are_grouped_and_truncated(self):
        connector = _make_connector()
        failed_users = [
            *[
                (f"missing-{index}@test.com", OneDriveUserStatus.NOT_PROVISIONED)
                for index in range(6)
            ],
            ("locked@test.com", OneDriveUserStatus.LOCKED),
            ("unknown@test.com", OneDriveUserStatus.UNKNOWN_ERROR),
        ]

        await connector._notify_onedrive_user_failures(failed_users)

        # UNKNOWN_ERROR is reported through the run's health, not here.
        assert connector.notify.await_count == 2
        notifications = {
            call.kwargs["title"]: call.kwargs["message"]
            for call in connector.notify.await_args_list
        }
        not_provisioned_message = notifications[
            "OneDrive: OneDrive not provisioned for some users"
        ]
        assert "missing-0@test.com" in not_provisioned_message
        assert "missing-4@test.com" in not_provisioned_message
        assert "missing-5@test.com" not in not_provisioned_message
        assert "and 1 more" in not_provisioned_message
        assert "locked@test.com" in notifications["OneDrive: OneDrive blocked for some users"]
        assert not any("unknown@test.com" in message for message in notifications.values())


# ---------------------------------------------------------------------------
# run_sync()
# ---------------------------------------------------------------------------


class TestRunSyncNotifications:

    @pytest.mark.asyncio
    async def test_success_does_not_notify_when_users_synced(self):
        connector = _make_connector()

        async def _set_synced_users(_users):
            connector.onedrive_users_synced = 2

        connector._reinitialize_credential_if_needed = AsyncMock()
        connector._sync_users = AsyncMock(return_value=[MagicMock()])
        connector._sync_user_groups = AsyncMock()
        connector._process_users_in_batches = AsyncMock(side_effect=_set_synced_users)

        with patch(
            "app.connectors.sources.microsoft.onedrive.connector.load_connector_filters",
            new_callable=AsyncMock,
            return_value=(MagicMock(), MagicMock()),
        ):
            await connector.run_sync()

        connector.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_skipped_drives_send_one_notice_and_fail_the_run(self):
        connector = _make_connector()

        async def _skip_two_drives(_users):
            connector._health.skipped_drive("a@test.com", "HTTP 503")
            connector._health.skipped_drive("b@test.com", "HTTP 503")

        connector._reinitialize_credential_if_needed = AsyncMock()
        connector._sync_users = AsyncMock(return_value=[MagicMock()])
        connector._sync_user_groups = AsyncMock()
        connector._process_users_in_batches = AsyncMock(side_effect=_skip_two_drives)

        with patch(
            "app.connectors.sources.microsoft.onedrive.connector.load_connector_filters",
            new_callable=AsyncMock,
            return_value=(MagicMock(), MagicMock()),
        ), pytest.raises(OneDriveSyncIncomplete):
            await connector.run_sync()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_RECORD_SYNC_ERROR,
            severity=NotificationSeverity.WARNING,
            title="OneDrive: some OneDrives could not be synced",
            message_contains="a@test.com, b@test.com",
        )
        assert connector.full_sync_coverage().complete is False

    @pytest.mark.asyncio
    async def test_skipped_part_that_is_not_a_drive_sends_part_skipped(self):
        connector = _make_connector()

        async def _skip_groups(_users):
            connector._health.skipped("groups of returning users", "some groups could not be read")

        connector._reinitialize_credential_if_needed = AsyncMock()
        connector._sync_users = AsyncMock(return_value=[MagicMock()])
        connector._sync_user_groups = AsyncMock()
        connector._process_users_in_batches = AsyncMock(side_effect=_skip_groups)

        with patch(
            "app.connectors.sources.microsoft.onedrive.connector.load_connector_filters",
            new_callable=AsyncMock,
            return_value=(MagicMock(), MagicMock()),
        ), pytest.raises(OneDriveSyncIncomplete):
            await connector.run_sync()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_RECORD_SYNC_ERROR,
            severity=NotificationSeverity.WARNING,
            title="OneDrive: part of the sync could not be read",
            message_contains="groups of returning users",
        )

    @pytest.mark.asyncio
    async def test_graph_401_during_the_run_sends_the_sign_in_notice_and_drops_the_credential(self):
        connector = _make_connector()
        refused = ODataError()
        refused.error = MainError()
        refused.error.code = "InvalidAuthenticationToken"
        refused.response_status_code = 401
        credential = AsyncMock()
        connector.credential = credential

        connector._reinitialize_credential_if_needed = AsyncMock()
        connector._sync_users = AsyncMock(side_effect=refused)

        with patch(
            "app.connectors.sources.microsoft.onedrive.connector.load_connector_filters",
            new_callable=AsyncMock,
            return_value=(MagicMock(), MagicMock()),
        ), pytest.raises(ODataError):
            await connector.run_sync()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_AUTH_ERROR,
            severity=NotificationSeverity.ERROR,
            title=SIGN_IN_FAILED_TITLE,
            message=SIGN_IN_FAILED_MESSAGE,
        )
        credential.close.assert_awaited_once()
        assert connector.credential is None

    @pytest.mark.asyncio
    async def test_incremental_sync_with_a_skipped_drive_fails_after_one_notice(self):
        connector = _make_connector()

        async def _skip_drive(_users):
            connector._health.skipped_drive("a@test.com", "HTTP 503")

        connector._reinitialize_credential_if_needed = AsyncMock()
        connector._remind_about_certificate = AsyncMock()
        connector.msgraph_client = MagicMock()
        connector.msgraph_client.get_all_users = AsyncMock(return_value=[MagicMock()])
        connector._process_users_in_batches = AsyncMock(side_effect=_skip_drive)

        with pytest.raises(OneDriveSyncIncomplete):
            await connector.run_incremental_sync()

        assert connector.notify.await_count == 1
        assert connector.notify.await_args.kwargs["title"] == "OneDrive: some OneDrives could not be synced"

    @pytest.mark.asyncio
    async def test_success_does_not_notify_when_no_users_synced(self):
        connector = _make_connector()

        async def _set_no_synced_users(_users):
            connector.onedrive_users_synced = 0

        connector._reinitialize_credential_if_needed = AsyncMock()
        connector._sync_users = AsyncMock(return_value=[])
        connector._sync_user_groups = AsyncMock()
        connector._process_users_in_batches = AsyncMock(side_effect=_set_no_synced_users)

        with patch(
            "app.connectors.sources.microsoft.onedrive.connector.load_connector_filters",
            new_callable=AsyncMock,
            return_value=(MagicMock(), MagicMock()),
        ):
            await connector.run_sync()

        connector.notify.assert_not_called()


# ---------------------------------------------------------------------------
# _reinitialize_credential_if_needed()
# ---------------------------------------------------------------------------


class TestReinitializeCredentialNotifications:

    @pytest.mark.asyncio
    async def test_reinit_token_failure_notifies_and_raises(self):
        connector = _make_connector()
        connector.credential = AsyncMock()
        connector.credential.get_token = AsyncMock(side_effect=Exception("transport closed"))
        connector.credential.close = AsyncMock()
        connector.config = {
            "credentials": {
                "auth": {
                    "tenantId": "tenant",
                    "clientId": "client",
                    "clientSecret": "secret",
                }
            }
        }

        with patch(
            "app.connectors.sources.microsoft.onedrive.connector.ClientSecretCredential"
        ) as mock_cred_cls, patch(
            "app.connectors.sources.microsoft.onedrive.connector.GraphServiceClient"
        ), patch(
            "app.connectors.sources.microsoft.onedrive.connector.MSGraphClient"
        ):
            new_cred = AsyncMock()
            new_cred.get_token = AsyncMock(side_effect=Exception("invalid secret"))
            mock_cred_cls.return_value = new_cred

            with pytest.raises(Exception, match="invalid secret"):
                await connector._reinitialize_credential_if_needed()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_AUTH_ERROR,
            severity=NotificationSeverity.ERROR,
            title=SIGN_IN_FAILED_TITLE,
            message=SIGN_IN_FAILED_MESSAGE,
        )

    @pytest.mark.asyncio
    async def test_valid_credential_does_not_notify(self):
        connector = _make_connector()
        connector.credential = AsyncMock()
        connector.credential.get_token = AsyncMock(return_value=MagicMock())

        await connector._reinitialize_credential_if_needed()

        connector.notify.assert_not_called()


# ---------------------------------------------------------------------------
# test_connection_and_access()
# ---------------------------------------------------------------------------


class TestConnectionAndAccessNotifications:

    @pytest.mark.asyncio
    async def test_all_scopes_pass_does_not_notify(self):
        connector = _make_connector()
        connector._probe_users_scope = AsyncMock()
        connector._probe_groups_scope = AsyncMock()
        connector._probe_files_scope = AsyncMock()

        result = await connector.test_connection_and_access()

        assert result is True
        connector.notify.assert_not_called()
        connector._probe_users_scope.assert_awaited_once()
        connector._probe_groups_scope.assert_awaited_once()
        connector._probe_files_scope.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_files_read_all_notifies_with_scope_in_message(self):
        connector = _make_connector()
        connector._probe_users_scope = AsyncMock()
        connector._probe_groups_scope = AsyncMock()
        connector._probe_files_scope = AsyncMock(side_effect=_make_403_odata_error())

        with pytest.raises(ConnectorInitError, match=r"Files\.Read\.All"):
            await connector.test_connection_and_access()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_AUTH_ERROR,
            severity=NotificationSeverity.ERROR,
            title="OneDrive: missing API permissions",
            message_contains="Files.Read.All",
        )

    @pytest.mark.asyncio
    async def test_missing_user_read_all_notifies_with_scope_in_message(self):
        connector = _make_connector()
        connector._probe_users_scope = AsyncMock(side_effect=_make_403_odata_error())
        connector._probe_groups_scope = AsyncMock()
        connector._probe_files_scope = AsyncMock()

        with pytest.raises(ConnectorInitError, match=r"User\.Read\.All"):
            await connector.test_connection_and_access()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_AUTH_ERROR,
            severity=NotificationSeverity.ERROR,
            title="OneDrive: missing API permissions",
            message_contains="User.Read.All",
        )

    @pytest.mark.asyncio
    async def test_missing_group_read_all_notifies_with_scope_in_message(self):
        connector = _make_connector()
        connector._probe_users_scope = AsyncMock()
        connector._probe_groups_scope = AsyncMock(side_effect=_make_403_odata_error())
        connector._probe_files_scope = AsyncMock()

        with pytest.raises(ConnectorInitError, match=r"Group\.Read\.All"):
            await connector.test_connection_and_access()

        _assert_notify(
            connector,
            notification_type=NotificationType.CONNECTOR_AUTH_ERROR,
            severity=NotificationSeverity.ERROR,
            title="OneDrive: missing API permissions",
            message_contains="Group.Read.All",
        )

    @pytest.mark.asyncio
    async def test_all_missing_scopes_single_notification_lists_all(self):
        connector = _make_connector()
        connector._probe_users_scope = AsyncMock(side_effect=_make_403_odata_error())
        connector._probe_groups_scope = AsyncMock(side_effect=_make_403_odata_error())
        connector._probe_files_scope = AsyncMock()

        with pytest.raises(
            ConnectorInitError,
            match=r"User\.Read\.All.*Group\.Read\.All|Group\.Read\.All.*User\.Read\.All",
        ):
            await connector.test_connection_and_access()

        connector.notify.assert_awaited_once()
        message = connector.notify.await_args.kwargs["message"]
        assert "User.Read.All" in message
        assert "Group.Read.All" in message
        assert connector.notify.await_args.kwargs["type"] is NotificationType.CONNECTOR_AUTH_ERROR

    @pytest.mark.asyncio
    async def test_transient_error_propagates_sanitized_message_without_notification(self):
        connector = _make_connector()
        connector._probe_users_scope = AsyncMock(side_effect=_make_transient_odata_error())
        connector._probe_groups_scope = AsyncMock()
        connector._probe_files_scope = AsyncMock()

        with pytest.raises(
            ConnectorInitError,
            match=(
                r"ServiceUnavailable: Microsoft Graph is temporarily unavailable "
                r"\(HTTP 503\)"
            ),
        ):
            await connector.test_connection_and_access()

        connector.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_generic_exception_propagates_safe_message_without_notification(self):
        connector = _make_connector()
        connector._probe_users_scope = AsyncMock(side_effect=RuntimeError("network down"))
        connector._probe_groups_scope = AsyncMock()
        connector._probe_files_scope = AsyncMock()

        with pytest.raises(
            ConnectorInitError,
            match=(
                "Unable to check Microsoft Graph permission "
                "User.Read.All: network down"
            ),
        ):
            await connector.test_connection_and_access()

        connector.notify.assert_not_called()
