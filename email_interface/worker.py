"""Background worker that periodically polls Gmail for new emails.

Runs as an asyncio task, checking for messages labelled ``cmugpt-unread``
sent to the configured target address at a regular interval.
"""

import asyncio
import contextlib
import logging
from typing import Any

from email_interface.config import EMAIL_POLL_INTERVAL, EMAIL_TARGET_ADDRESS
from email_interface.handler import EmailHandler

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_ERRORS = 5
_BACKOFF_SECONDS = 300


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

        self._handler = EmailHandler()
        await self._handler.initialise()
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
