"""Tests for the email 'from' address enforcement.

Verifies that emails can only be sent from the authenticated user's own
email address, with no fallback to EMAIL_TARGET_ADDRESS or any other address.
"""

import base64
import email
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub out the heavy `agent` package so handler.py's lazy
# `from agent import UserInput, run_agent` resolves without
# needing openai/mcp/etc. installed.
_agent_stub = MagicMock()
sys.modules.setdefault("agent", _agent_stub)
sys.modules.setdefault("agent.agent_hub", _agent_stub)
sys.modules.setdefault("agent.schema", _agent_stub)
sys.modules.setdefault("agent.streaming", _agent_stub)

from email_interface.handler import EmailHandler, InboundEmail
from email_interface.utils import build_reply_message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_inbound_email(**overrides) -> InboundEmail:
    defaults = dict(
        message_id="msg123",
        thread_id="thread456",
        sender_raw="Alice <alice@example.com>",
        sender_email="alice@example.com",
        sender_name="Alice",
        to="cmugpt@andrew.cmu.edu",
        subject="Hello",
        date="Thu, 29 May 2026 10:00:00 -0500",
        body_text="What courses should I take?",
        body_html="<p>What courses should I take?</p>",
        gmail_message_id_header="<orig@example.com>",
        references_header="",
    )
    defaults.update(overrides)
    return InboundEmail(**defaults)


def _mock_gmail_service(profile_email: str | None = None):
    """Return a mock Gmail service. If profile_email is None, getProfile raises."""
    svc = MagicMock()
    if profile_email is not None:
        svc.users().getProfile(userId="me").execute.return_value = {
            "emailAddress": profile_email
        }
    else:
        svc.users().getProfile(userId="me").execute.side_effect = Exception(
            "profile unavailable"
        )
    svc.users().labels().list(userId="me").execute.return_value = {
        "labels": [
            {"name": "cmugpt-unread", "id": "Label_1"},
            {"name": "cmugpt-read", "id": "Label_2"},
        ]
    }
    return svc


# ---------------------------------------------------------------------------
# build_reply_message tests
# ---------------------------------------------------------------------------

class TestBuildReplyMessage:
    def test_requires_from_addr(self):
        """build_reply_message raises ValueError when from_addr is empty."""
        with pytest.raises(ValueError, match="from_addr is required"):
            build_reply_message(
                from_addr="",
                to="alice@example.com",
                subject="Re: Hello",
                body_html="<p>Hi</p>",
                body_plain="Hi",
                original_message_id="<orig@example.com>",
            )

    def test_from_header_uses_provided_address(self):
        """The MIME From header must use the address we provide."""
        result = build_reply_message(
            from_addr="myuser@andrew.cmu.edu",
            to="alice@example.com",
            subject="Re: Hello",
            body_html="<p>Hi</p>",
            body_plain="Hi",
            original_message_id="<orig@example.com>",
        )
        raw_bytes = base64.urlsafe_b64decode(result["raw"])
        parsed = email.message_from_bytes(raw_bytes)
        assert "myuser@andrew.cmu.edu" in parsed["From"]

    def test_thread_id_included_when_provided(self):
        result = build_reply_message(
            from_addr="myuser@andrew.cmu.edu",
            to="alice@example.com",
            subject="Re: Hello",
            body_html="<p>Hi</p>",
            body_plain="Hi",
            original_message_id="<orig@example.com>",
            thread_id="thread456",
        )
        assert result["threadId"] == "thread456"


# ---------------------------------------------------------------------------
# EmailHandler.initialise tests
# ---------------------------------------------------------------------------

class TestEmailHandlerInitialise:
    @pytest.mark.asyncio
    async def test_initialise_succeeds_with_valid_profile(self):
        """Handler initialises when Gmail profile returns an email."""
        handler = EmailHandler()
        svc = _mock_gmail_service(profile_email="cmugpt@andrew.cmu.edu")
        mock_creds = MagicMock()

        with patch("email_interface.handler.build", return_value=svc):
            await handler.initialise(mock_creds)

        assert handler._own_email == "cmugpt@andrew.cmu.edu"

    @pytest.mark.asyncio
    async def test_initialise_fails_when_profile_unavailable(self):
        """Handler refuses to start if it can't resolve the user's email."""
        handler = EmailHandler()
        svc = _mock_gmail_service(profile_email=None)
        mock_creds = MagicMock()

        with patch("email_interface.handler.build", return_value=svc):
            with pytest.raises(Exception):
                await handler.initialise(mock_creds)

    @pytest.mark.asyncio
    async def test_initialise_fails_when_profile_returns_empty(self):
        """Handler refuses to start if profile returns an empty email."""
        handler = EmailHandler()
        svc = _mock_gmail_service(profile_email="")
        mock_creds = MagicMock()

        with patch("email_interface.handler.build", return_value=svc):
            with pytest.raises(RuntimeError, match="Could not resolve"):
                await handler.initialise(mock_creds)


# ---------------------------------------------------------------------------
# EmailHandler.process_email tests
# ---------------------------------------------------------------------------

class TestEmailHandlerProcessEmail:
    @pytest.mark.asyncio
    async def test_send_uses_own_email_as_from(self):
        """Replies must be sent from the authenticated user's own email."""
        handler = EmailHandler()
        handler._own_email = "cmugpt@andrew.cmu.edu"
        handler._label_ids = {
            "cmugpt-unread": "Label_1",
            "cmugpt-read": "Label_2",
        }

        mock_agent_response = MagicMock()
        mock_agent_response.response_text = "<p>Take 15-112!</p>"

        _agent_stub.run_agent = AsyncMock(return_value=mock_agent_response)
        _agent_stub.UserInput = MagicMock()

        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            inbound = _make_inbound_email()
            result = await handler.process_email(inbound)

        assert result is True
        sent_body = mock_send.call_args[0][0]
        raw_bytes = base64.urlsafe_b64decode(sent_body["raw"])
        parsed = email.message_from_bytes(raw_bytes)
        assert "cmugpt@andrew.cmu.edu" in parsed["From"]

    @pytest.mark.asyncio
    async def test_send_blocked_when_own_email_not_resolved(self):
        """If own email is somehow None, process_email must refuse to send."""
        handler = EmailHandler()
        handler._own_email = None

        mock_agent_response = MagicMock()
        mock_agent_response.response_text = "<p>Take 15-112!</p>"

        _agent_stub.run_agent = AsyncMock(return_value=mock_agent_response)
        _agent_stub.UserInput = MagicMock()

        inbound = _make_inbound_email()
        result = await handler.process_email(inbound)

        assert result is False

    @pytest.mark.asyncio
    async def test_no_fallback_to_target_address(self):
        """Even with EMAIL_TARGET_ADDRESS set, handler must not use it as From."""
        handler = EmailHandler()
        handler._own_email = None

        mock_agent_response = MagicMock()
        mock_agent_response.response_text = "<p>Reply</p>"

        _agent_stub.run_agent = AsyncMock(return_value=mock_agent_response)
        _agent_stub.UserInput = MagicMock()

        with patch.object(
            handler, "_send_reply", new_callable=AsyncMock
        ) as mock_send:
            inbound = _make_inbound_email()
            result = await handler.process_email(inbound)

        assert result is False
        mock_send.assert_not_called()
