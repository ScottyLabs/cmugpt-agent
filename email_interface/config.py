"""Email interface configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

EMAIL_TARGET_ADDRESS: str = os.getenv("EMAIL_TARGET_ADDRESS", "")
EMAIL_POLL_INTERVAL: int = int(os.getenv("EMAIL_POLL_INTERVAL", "60"))
EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)
EMAIL_MODEL: str = os.getenv("EMAIL_MODEL", "openai/gpt-4o")

GOOGLE_CREDENTIALS_FILE: str = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
GOOGLE_CREDENTIALS_JSON: str = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
GOOGLE_TOKEN_JSON: str = os.getenv("GOOGLE_TOKEN_JSON", "")

# --- Keycloak ---
KEYCLOAK_SERVER_URL: str = os.getenv("KEYCLOAK_SERVER_URL", "")
KEYCLOAK_REALM: str = os.getenv("KEYCLOAK_REALM", "cmugpt")
KEYCLOAK_CLIENT_ID: str = os.getenv("KEYCLOAK_CLIENT_ID", "cmugpt-agent")
KEYCLOAK_CLIENT_SECRET: str = os.getenv("KEYCLOAK_CLIENT_SECRET", "")
KEYCLOAK_ADMIN_USERNAME: str = os.getenv("KEYCLOAK_ADMIN_USERNAME", "")
KEYCLOAK_ADMIN_PASSWORD: str = os.getenv("KEYCLOAK_ADMIN_PASSWORD", "")
KEYCLOAK_GOOGLE_IDP_ALIAS: str = os.getenv("KEYCLOAK_GOOGLE_IDP_ALIAS", "google")
