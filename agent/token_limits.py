"""Per-user daily token budget.

Usage is persisted to SQLite because the Procfile runs two uvicorn workers.
A per-process counter would give each worker an independent budget and so
double the effective limit.

Enforcement is a soft ceiling: the request that exceeds the limit still
completes, and the request after it is rejected.

`user_id` originates from the request body, so this module depends on the
Surface to authenticate its users and on AGENT_SHARED_SECRET to restrict
direct callers.
"""

import os
import sqlite3
from datetime import UTC, datetime

# Total tokens (input + output) one user may spend per UTC day.
DAILY_TOKEN_LIMIT = 1_000_000

# Requests without a user_id share a single counter. The Surface always
# supplies one.
_ANONYMOUS_KEY = "anonymous"


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS daily_usage ("
    "  day TEXT NOT NULL,"
    "  user_id TEXT NOT NULL,"
    "  tokens INTEGER NOT NULL DEFAULT 0,"
    "  PRIMARY KEY (day, user_id)"
    ")"
)


class DailyTokenLimitExceeded(Exception):
    """Raised when a user has spent the daily token budget."""

    def __init__(self, user_id: str, used: int) -> None:
        self.user_id = user_id
        self.used = used
        super().__init__(
            f"Daily token limit reached ({used}/{DAILY_TOKEN_LIMIT}). "
            "Please try again tomorrow."
        )


def _db_path() -> str:
    # Loss of the default file on redeploy resets accumulated usage. The
    # failure mode is therefore under-counting, never a spurious block.
    return os.getenv("TOKEN_USAGE_DB", "/tmp/cmugpt_token_usage.sqlite3")


def _today() -> str:
    # UTC keeps the day boundary identical across workers.
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=5)
    conn.execute(_SCHEMA)
    return conn


def _user_key(user_id: str | None) -> str:
    return user_id or _ANONYMOUS_KEY


def _read_tokens(key: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT tokens FROM daily_usage WHERE day = ? AND user_id = ?",
            (_today(), key),
        ).fetchone()
    return int(row[0]) if row else 0


def tokens_used_today(user_id: str | None) -> int:
    return _read_tokens(_user_key(user_id))


def ensure_within_daily_limit(user_id: str | None) -> None:
    """Raise DailyTokenLimitExceeded when the user's budget is spent."""
    key = _user_key(user_id)
    used = _read_tokens(key)
    if used >= DAILY_TOKEN_LIMIT:
        raise DailyTokenLimitExceeded(key, used)


def record_usage(user_id: str | None, tokens: int) -> None:
    """Add `tokens` to today's count and prune earlier days."""
    if tokens <= 0:
        return
    today = _today()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO daily_usage (day, user_id, tokens) VALUES (?, ?, ?) "
            "ON CONFLICT (day, user_id) DO UPDATE SET tokens = tokens + ?",
            (today, _user_key(user_id), tokens, tokens),
        )
        conn.execute("DELETE FROM daily_usage WHERE day < ?", (today,))
