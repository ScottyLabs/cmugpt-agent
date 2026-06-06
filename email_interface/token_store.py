"""SQLite-backed user mapping for Keycloak-managed Gmail OAuth.

Keycloak is the sole token store. This module only persists the mapping
of app-level ``user_id`` → Keycloak ``sub`` (user ID) and resolved email,
so the agent knows which Keycloak user to fetch the Google token for.
No tokens are stored locally.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[1] / "gmail_tokens.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS user_mappings (
    user_id         TEXT PRIMARY KEY,
    keycloak_sub    TEXT NOT NULL,
    email           TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class TokenStore:
    """Maps app user IDs to Keycloak subjects. Keycloak stores all tokens."""

    def __init__(self, db_path: Path | str = _DB_PATH) -> None:
        self._db_path = str(db_path)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def save_user(self, user_id: str, keycloak_sub: str, email: str) -> None:
        """Store or update the user_id → Keycloak mapping."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_mappings (user_id, keycloak_sub, email)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    keycloak_sub = excluded.keycloak_sub,
                    email        = excluded.email,
                    updated_at   = datetime('now')
                """,
                (user_id, keycloak_sub, email),
            )
        logger.info("Saved user mapping: %s → %s (%s)", user_id, keycloak_sub, email)

    def get_keycloak_sub(self, user_id: str) -> str | None:
        """Return the Keycloak subject ID for a user, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT keycloak_sub FROM user_mappings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row[0] if row else None

    def get_email(self, user_id: str) -> str | None:
        """Return the stored email for a user, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT email FROM user_mappings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row[0] if row else None

    def has_user(self, user_id: str) -> bool:
        """Check if a user has completed Gmail auth."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM user_mappings WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row is not None

    def get_first_keycloak_sub(self) -> str | None:
        """Return the Keycloak subject of the most recently updated user, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT keycloak_sub FROM user_mappings ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def delete_user(self, user_id: str) -> None:
        """Remove a user mapping."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM user_mappings WHERE user_id = ?", (user_id,)
            )
        logger.info("Deleted user mapping for %s", user_id)
