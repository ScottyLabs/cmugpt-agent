"""Send-email tool for the CMUGPT agent (Keycloak-managed OAuth).

Exposes an ``EmailSender`` that the LLM can invoke as a function-call tool
to compose and send emails. Tokens are stored solely in Keycloak; the agent
fetches the Google token on demand via Keycloak's broker token endpoint.
"""

import asyncio
import logging
import os
import threading
from typing import Any

from googleapiclient.discovery import build

from email_interface.keycloak_auth import get_keycloak_client
from email_interface.token_store import TokenStore
from email_interface.utils import build_new_message, extract_text_from_html

logger = logging.getLogger(__name__)

_gmail_lock = threading.Lock()

SEND_EMAIL_TOOL_NAME = "send_email"

_token_store = TokenStore()

SEND_EMAIL_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEND_EMAIL_TOOL_NAME,
        "description": (
            "Send an email from the user's own Gmail address. "
            "Use this tool to compose and send an email as the user "
            'requested. Set `to` to "self" to send to the user\'s own '
            "inbox. The user must have authorised Gmail first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        'Recipient email address, or "self" to send '
                        "to the user's own inbox."
                    ),
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line.",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Email body as HTML. Use <p>, <b>, <ul>/<li>, "
                        "<a href>, etc. — never raw Markdown."
                    ),
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
}


def _get_oauth_start_url(user_id: str) -> str:
    base = os.getenv(
        "OAUTH_REDIRECT_BASE_URL",
        f"http://localhost:{os.getenv('PORT', '8000')}",
    )
    return f"{base}/auth/gmail/start?user_id={user_id}"


class EmailSender:
    """Sends emails via the Gmail API using Keycloak-managed Google tokens."""

    @property
    def tool_definition(self) -> dict[str, Any]:
        return SEND_EMAIL_TOOL_DEFINITION

    async def send(
        self, *, to: str, subject: str, body: str, user_id: str
    ) -> str:
        """Send an email on behalf of *user_id*. Returns a status string for the LLM."""
        if not user_id:
            return (
                "Error: cannot send email — no user_id in the request. "
                "The Surface client must include a user_id field."
            )

        keycloak_sub = _token_store.get_keycloak_sub(user_id)
        if keycloak_sub is None:
            auth_url = _get_oauth_start_url(user_id)
            return (
                f"GMAIL_AUTH_REQUIRED|{auth_url}|"
                "The user has not authorised Gmail yet. "
                "They need to click the link to grant access, then retry."
            )

        own_email = _token_store.get_email(user_id)
        if not own_email:
            auth_url = _get_oauth_start_url(user_id)
            return (
                f"GMAIL_AUTH_REQUIRED|{auth_url}|"
                "Could not resolve the user's email. "
                "They need to re-authorise via the link."
            )

        # Fetch Google credentials from Keycloak on demand
        kc = get_keycloak_client()
        creds = await kc.get_gmail_credentials(keycloak_sub)
        if creds is None:
            auth_url = _get_oauth_start_url(user_id)
            _token_store.delete_user(user_id)
            return (
                f"GMAIL_AUTH_REQUIRED|{auth_url}|"
                "Gmail authorisation expired or was revoked in Keycloak. "
                "The user needs to re-authorise via the link, then retry."
            )

        recipient = own_email if to.strip().lower() == "self" else to

        body_plain = extract_text_from_html(body) or body
        message = build_new_message(
            from_addr=own_email,
            to=recipient,
            subject=subject,
            body_html=body,
            body_plain=body_plain,
        )

        try:
            svc = build("gmail", "v1", credentials=creds)

            def _send() -> dict:
                with _gmail_lock:
                    return (
                        svc.users()
                        .messages()
                        .send(userId="me", body=message)
                        .execute()
                    )

            result = await asyncio.to_thread(_send)
            msg_id = result.get("id", "unknown")
            logger.info(
                "Email sent to %s for user %s — Gmail ID: %s",
                recipient,
                user_id,
                msg_id,
            )
            return f"Email sent successfully to {recipient}."
        except Exception as exc:
            logger.exception(
                "Failed to send email to %s for user %s", recipient, user_id
            )
            return f"Error sending email: {exc}"
