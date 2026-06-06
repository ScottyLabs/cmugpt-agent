"""Email interface for CMUGPT.

Provides a background worker that polls Gmail for incoming emails
and responds using the same agent pipeline as the HTTP API, plus
per-user OAuth (via Keycloak) so the agent can send email from each
user's own account.
"""

from email_interface.handler import EmailHandler
from email_interface.keycloak_auth import KeycloakClient, get_keycloak_client, keycloak_enabled
from email_interface.send_tool import EmailSender
from email_interface.token_store import TokenStore
from email_interface.worker import EmailWorker

__all__ = [
    "EmailHandler",
    "EmailSender",
    "KeycloakClient",
    "TokenStore",
    "EmailWorker",
    "get_keycloak_client",
    "keycloak_enabled",
]
