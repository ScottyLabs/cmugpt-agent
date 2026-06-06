"""Google OAuth2 authentication for the Gmail API.

Supports service-account credentials (from JSON string or file) and
user-OAuth credentials (from token file or interactive flow).
"""

import json
import logging
import os
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from email_interface.config import (
    GOOGLE_CREDENTIALS_FILE,
    GOOGLE_CREDENTIALS_JSON,
    GOOGLE_TOKEN_JSON,
)

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
]

TOKEN_FILE = "token.json"


class GoogleAuth:
    """Manages OAuth2 credentials and builds cached Gmail service objects."""

    def __init__(self) -> None:
        self._creds: Credentials | None = None
        self._gmail_service: Any = None

    def _get_credentials(self) -> Credentials:
        if self._creds and self._creds.valid:
            return self._creds

        creds = None

        # 1. Service account from JSON env var
        if GOOGLE_CREDENTIALS_JSON:
            try:
                info = json.loads(GOOGLE_CREDENTIALS_JSON)
                creds = service_account.Credentials.from_service_account_info(
                    info, scopes=SCOPES
                )
                logger.info("Loaded service-account credentials from env var")
            except Exception as e:
                logger.warning("Failed to load service-account from env var: %s", e)

        # 2. Service account from file
        has_file = GOOGLE_CREDENTIALS_FILE and os.path.exists(
            GOOGLE_CREDENTIALS_FILE
        )
        if not creds and has_file:
            try:
                creds = service_account.Credentials.from_service_account_file(
                    GOOGLE_CREDENTIALS_FILE, scopes=SCOPES
                )
                logger.info("Loaded service-account credentials from file")
            except ValueError:
                pass  # not a service-account file, fall through

        if not creds:
            # 3. User token from JSON env var
            if GOOGLE_TOKEN_JSON:
                try:
                    info = json.loads(GOOGLE_TOKEN_JSON)
                    creds = Credentials.from_authorized_user_info(info, SCOPES)
                    logger.info("Loaded user credentials from env var")
                except Exception as e:
                    logger.warning("Failed to load user creds from env var: %s", e)

            # 4. User token from file
            if (not creds or not creds.valid) and os.path.exists(TOKEN_FILE):
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

            # Refresh or re-authenticate
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not GOOGLE_CREDENTIALS_FILE or not os.path.exists(
                        GOOGLE_CREDENTIALS_FILE
                    ):
                        raise FileNotFoundError(
                            f"Credentials file not found: {GOOGLE_CREDENTIALS_FILE}"
                        )
                    flow = InstalledAppFlow.from_client_secrets_file(
                        GOOGLE_CREDENTIALS_FILE, SCOPES
                    )
                    creds = flow.run_local_server(port=8090)

                if TOKEN_FILE:
                    with open(TOKEN_FILE, "w") as f:
                        f.write(creds.to_json())
                    logger.info("Saved updated token to %s", TOKEN_FILE)

        self._creds = creds
        return self._creds  # type: ignore[return-value]

    @property
    def gmail(self) -> Any:
        if self._gmail_service is None:
            creds = self._get_credentials()
            self._gmail_service = build("gmail", "v1", credentials=creds)
            logger.info("Built Gmail API v1 service")
        return self._gmail_service


_auth: GoogleAuth | None = None


def get_google_auth() -> GoogleAuth:
    """Return the global GoogleAuth singleton."""
    global _auth
    if _auth is None:
        _auth = GoogleAuth()
    return _auth


def has_google_credentials() -> bool:
    """Check whether Google credentials are available without initialising."""
    return bool(GOOGLE_CREDENTIALS_JSON) or (
        bool(GOOGLE_CREDENTIALS_FILE) and os.path.exists(GOOGLE_CREDENTIALS_FILE)
    )
