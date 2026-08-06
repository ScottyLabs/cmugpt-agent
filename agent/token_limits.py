"""Per-user daily token budget.

Usage lives in a sqlite file because the Procfile runs two uvicorn workers.
A per-process counter would give each worker its own budget, doubling the
effective limit.

The budget is a soft ceiling. The request that crosses the line still
completes and the next one is rejected.
"""

import os
import sqlite3
from datetime import UTC, datetime

# Total tokens (input + output) one user may spend per UTC day.
DAILY_TOKEN_LIMIT = 1_000_000

# Anonymous requests share one bucket. The Surface always sends a user_id.
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
    """Raised when a user has spent their daily token budget."""

    def __init__(self, user_id: str, used: int) -> None:
        self.user_id = user_id
        self.used = used
        super().__init__(
            f"Daily token limit reached ({used}/{DAILY_TOKEN_LIMIT}). "
            "Please try again tomorrow."
        )


def _db_path() -> str:
    # Losing the default file on redeploy only grants a fresh budget. It
    # never wrongly blocks a user.
    return os.getenv("TOKEN_USAGE_DB", "/tmp/cmugpt_token_usage.sqlite3")


def _today() -> str:
    # UTC so both workers agree on the day boundary.
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=5)
    conn.execute(_SCHEMA)
    return conn


def tokens_used_today(user_id: str | None) -> int:
    key = user_id or _ANONYMOUS_KEY
    with _connect() as conn:
        row = conn.execute(
            "SELECT tokens FROM daily_usage WHERE day = ? AND user_id = ?",
            (_today(), key),
        ).fetchone()
    return int(row[0]) if row else 0


def ensure_within_daily_limit(user_id: str | None) -> None:
    """Raise DailyTokenLimitExceeded when the user's budget is spent."""
    used = tokens_used_today(user_id)
    if used >= DAILY_TOKEN_LIMIT:
        raise DailyTokenLimitExceeded(user_id or _ANONYMOUS_KEY, used)


def record_usage(user_id: str | None, tokens: int) -> None:
    """Add `tokens` to today's count and prune earlier days."""
    if tokens <= 0:
        return
    key = user_id or _ANONYMOUS_KEY
    today = _today()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO daily_usage (day, user_id, tokens) VALUES (?, ?, ?) "
            "ON CONFLICT (day, user_id) DO UPDATE SET tokens = tokens + ?",
            (today, key, tokens, tokens),
        )
        conn.execute("DELETE FROM daily_usage WHERE day < ?", (today,))
