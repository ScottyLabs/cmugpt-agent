"""Keycloak identity broker for Gmail OAuth.

Uses Keycloak as the sole identity and token provider. Google tokens are
stored in Keycloak and retrieved on demand via the broker token endpoint.

Flow: service-account token (same realm) → token exchange (impersonate
user) → broker endpoint → Google access token. No tokens are stored
locally; Keycloak manages everything.
"""

import logging
from typing import Any
from urllib.parse import urlencode

import httpx
from google.oauth2.credentials import Credentials

from email_interface.config import (
    KEYCLOAK_ADMIN_PASSWORD,
    KEYCLOAK_ADMIN_USERNAME,
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_CLIENT_SECRET,
    KEYCLOAK_GOOGLE_IDP_ALIAS,
    KEYCLOAK_REALM,
    KEYCLOAK_SERVER_URL,
)

logger = logging.getLogger(__name__)

_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class KeycloakClient:
    """Client for Keycloak OIDC and admin operations related to Gmail OAuth."""

    def __init__(self) -> None:
        self._base = KEYCLOAK_SERVER_URL.rstrip("/")
        self._realm = KEYCLOAK_REALM
        self._client_id = KEYCLOAK_CLIENT_ID
        self._client_secret = KEYCLOAK_CLIENT_SECRET
        self._idp_alias = KEYCLOAK_GOOGLE_IDP_ALIAS

    @property
    def realm_url(self) -> str:
        return f"{self._base}/realms/{self._realm}"

    @property
    def oidc_url(self) -> str:
        return f"{self.realm_url}/protocol/openid-connect"

    @property
    def admin_url(self) -> str:
        return f"{self._base}/admin/realms/{self._realm}"

    # ------------------------------------------------------------------
    # OIDC flow (user-facing auth)
    # ------------------------------------------------------------------

    def get_auth_url(self, redirect_uri: str, state: str) -> str:
        """Build the Keycloak authorization URL that triggers Google IdP brokering."""
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email",
            "state": state,
            "kc_idp_hint": self._idp_alias,
        }
        return f"{self.oidc_url}/auth?{urlencode(params)}"

    async def exchange_code(
        self, code: str, redirect_uri: str
    ) -> dict[str, Any]:
        """Exchange an authorization code for Keycloak tokens."""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.oidc_url}/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            return resp.json()

    async def get_userinfo(self, access_token: str) -> dict[str, Any]:
        """Fetch the authenticated user's info from Keycloak."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.oidc_url}/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Broker token endpoint (requires user's KC access token)
    # ------------------------------------------------------------------

    async def get_broker_token(self, access_token: str) -> dict[str, Any] | None:
        """Retrieve the stored Google token via Keycloak's broker endpoint.

        Requires a valid Keycloak access token for the user.
        """
        url = f"{self.realm_url}/broker/{self._idp_alias}/token"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code in (400, 404):
                logger.warning("Broker token unavailable: %s", resp.text)
                return None
            if resp.status_code == 403:
                logger.warning("Broker token forbidden (missing read-token role?): %s", resp.text)
                return None
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Admin API
    # ------------------------------------------------------------------

    async def _get_admin_token(self) -> str:
        """Obtain an admin access token from the master realm."""
        data = {
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": KEYCLOAK_ADMIN_USERNAME,
            "password": KEYCLOAK_ADMIN_PASSWORD,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/realms/master/protocol/openid-connect/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    async def ensure_broker_read_token(self, keycloak_sub: str) -> None:
        """Grant the ``read-token`` role from the ``broker`` client to a user.

        The broker token endpoint returns 403 unless the user has this
        role. We assign it automatically via the admin API.
        """
        if not KEYCLOAK_ADMIN_USERNAME or not KEYCLOAK_ADMIN_PASSWORD:
            logger.warning(
                "Cannot auto-grant read-token role — "
                "KEYCLOAK_ADMIN_USERNAME/PASSWORD not set"
            )
            return

        try:
            admin_token = await self._get_admin_token()
            headers = {"Authorization": f"Bearer {admin_token}"}

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.admin_url}/clients",
                    params={"clientId": "broker"},
                    headers=headers,
                )
                resp.raise_for_status()
                clients = resp.json()
                if not clients:
                    logger.warning("Built-in 'broker' client not found in realm")
                    return
                broker_client_id = clients[0]["id"]

                resp = await client.get(
                    f"{self.admin_url}/clients/{broker_client_id}/roles/read-token",
                    headers=headers,
                )
                if resp.status_code == 404:
                    logger.warning("read-token role not found on broker client")
                    return
                resp.raise_for_status()
                role = resp.json()

                resp = await client.post(
                    f"{self.admin_url}/users/{keycloak_sub}/role-mappings/clients/{broker_client_id}",
                    json=[role],
                    headers=headers,
                )
                if resp.status_code == 409:
                    logger.debug("read-token role already assigned to %s", keycloak_sub)
                    return
                resp.raise_for_status()
                logger.info("Granted read-token role to user %s", keycloak_sub)

        except Exception:
            logger.exception("Failed to grant read-token role to %s", keycloak_sub)

    # ------------------------------------------------------------------
    # Token exchange — impersonate user via service account
    # ------------------------------------------------------------------

    async def _get_service_account_token(self) -> str:
        """Get a service account token from the same realm (not master).

        Token exchange requires the subject_token to be from the same
        realm as the target user.
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.oidc_url}/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    async def get_google_token_for_user(self, keycloak_sub: str) -> dict[str, Any] | None:
        """Retrieve the Google broker token for a user via token exchange.

        Uses the service account's own token (same realm) to impersonate
        the user, then calls the broker endpoint to get the Google token.
        """
        sa_token = await self._get_service_account_token()

        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "requested_subject": keycloak_sub,
            "subject_token": sa_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.oidc_url}/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                logger.warning(
                    "Token exchange failed for user %s: %s",
                    keycloak_sub,
                    resp.text,
                )
                return None
            user_token = resp.json().get("access_token")

        if not user_token:
            return None

        return await self.get_broker_token(user_token)

    # ------------------------------------------------------------------
    # High-level: get Gmail Credentials for a Keycloak user
    # ------------------------------------------------------------------

    async def get_gmail_credentials(self, keycloak_sub: str) -> Credentials | None:
        """Fetch the Google token from Keycloak and return as Credentials.

        This is the main method the agent uses to get Gmail API access.
        No tokens are stored locally — Keycloak is the sole token store.
        """
        google_token = await self.get_google_token_for_user(keycloak_sub)
        if not google_token:
            return None

        access_token = google_token.get("access_token")
        if not access_token:
            return None

        return Credentials(token=access_token, scopes=_GMAIL_SCOPES)


_keycloak_client: KeycloakClient | None = None


def get_keycloak_client() -> KeycloakClient:
    """Return a module-level singleton KeycloakClient."""
    global _keycloak_client
    if _keycloak_client is None:
        _keycloak_client = KeycloakClient()
    return _keycloak_client


def keycloak_enabled() -> bool:
    """Return True if Keycloak configuration is present."""
    return bool(KEYCLOAK_SERVER_URL and KEYCLOAK_CLIENT_ID)
