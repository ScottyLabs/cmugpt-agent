"""Background worker that periodically polls Gmail for new emails.

Runs as an asyncio task, checking for messages labelled ``cmugpt-unread``
sent to the configured target address at a regular interval.

When Keycloak is configured, the worker fetches Google credentials via
Keycloak's broker token endpoint. Otherwise it falls back to direct
Google OAuth credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from google.oauth2.credentials import Credentials

from email_interface.config import EMAIL_POLL_INTERVAL, EMAIL_TARGET_ADDRESS
from email_interface.handler import EmailHandler
from email_interface.keycloak_auth import get_keycloak_client, keycloak_enabled
from email_interface.token_store import TokenStore

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_ERRORS = 5
_BACKOFF_SECONDS = 300


async def _resolve_credentials() -> Credentials | None:
    """Resolve Google credentials from Keycloak or direct config.

    When Keycloak is enabled, picks the first registered user from the
    token store and fetches their Google token via Keycloak's broker
    endpoint. Returns ``None`` if Keycloak is the sole provider but no
    users have completed OAuth yet.

    Falls back to direct GoogleAuth credentials only when Keycloak is
    not configured.
    """
    if keycloak_enabled():
        store = TokenStore()
        kc = get_keycloak_client()
        keycloak_sub = store.get_first_keycloak_sub()
        if keycloak_sub:
            creds = await kc.get_gmail_credentials(keycloak_sub)
            if creds:
                logger.info("Worker using Keycloak-brokered Google credentials")
                return creds
            logger.warning(
                "Keycloak broker token unavailable for sub %s", keycloak_sub
            )
        else:
            logger.info(
                "Keycloak enabled but no users registered yet — "
                "email worker will start once a user completes /auth/gmail/start"
            )
        return None

    from email_interface.google_auth import get_google_auth

    auth = get_google_auth()
    return auth._get_credentials()


class EmailWorker:
    """Periodically polls Gmail for ``cmugpt-unread`` emails and processes them."""

    def __init__(
        self,
        poll_interval: int | None = None,
        target_address: str | None = None,
    ) -> None:
        self._poll_interval = poll_interval or EMAIL_POLL_INTERVAL
        self._target_address = target_address or EMAIL_TARGET_ADDRESS
        self._handler: EmailHandler | None = None
        self._task: asyncio.Task[Any] | None = None
        self._running = False
        self._consecutive_errors = 0

    async def start(self) -> None:
        if self._running:
            logger.warning("EmailWorker is already running")
            return

        credentials = await _resolve_credentials()
        if credentials is None:
            logger.info(
                "EmailWorker deferred — no credentials available yet. "
                "A user must complete OAuth via /auth/gmail/start first."
            )
            return

        self._handler = EmailHandler()
        await self._handler.initialise(credentials)
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="email-worker")
        logger.info(
            "EmailWorker started — polling %s every %ds",
            self._target_address,
            self._poll_interval,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._handler = None
        logger.info("EmailWorker stopped")

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        # Brief startup delay so other services can initialise first
        await asyncio.sleep(5)

        while self._running:
            try:
                await self._poll_once()
                self._consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                self._consecutive_errors += 1
                logger.exception(
                    "Error during email poll cycle (%d/%d consecutive)",
                    self._consecutive_errors,
                    _MAX_CONSECUTIVE_ERRORS,
                )
                if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "Too many consecutive email poll errors — backing off for %ds",
                        _BACKOFF_SECONDS,
                    )
                    try:
                        await asyncio.sleep(_BACKOFF_SECONDS)
                    except asyncio.CancelledError:
                        raise
                    self._consecutive_errors = 0
                    continue

            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    async def _poll_once(self) -> None:
        assert self._handler is not None

        emails = await self._handler.fetch_unread_emails(self._target_address)
        if not emails:
            return

        logger.info("Found %d new email(s) to process", len(emails))

        for email in emails:
            logger.info(
                'Processing email %s from %s — "%s"',
                email.message_id,
                email.sender_email,
                email.subject,
            )
            try:
                replied = await self._handler.process_email(email)
                if replied:
                    await self._handler.mark_as_read(email.message_id)
                    logger.info("Successfully processed email %s", email.message_id)
                else:
                    logger.info(
                        "No reply sent for email %s — leaving cmugpt-unread",
                        email.message_id,
                    )
            except Exception:
                logger.exception(
                    "Unexpected error processing email %s", email.message_id
                )

        self._handler.cleanup_old_conversations()
