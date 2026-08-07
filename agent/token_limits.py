"""Per-user daily token budget with a global backstop.

Usage lives in a sqlite file because the Procfile runs two uvicorn workers.
A per-process counter would give each worker its own budget, doubling the
effective limit.

The budget is a soft ceiling. The request that crosses the line still
completes and the next one is rejected.

user_id comes from the request body, so per-user fairness trusts the Surface
to authenticate its users and AGENT_SHARED_SECRET to gate direct callers.
The global row is the backstop that bounds total spend even when ids are
rotated per request.
"""

import os
import sqlite3
from datetime import UTC, datetime

# Total tokens (input + output) one user may spend per UTC day.
DAILY_TOKEN_LIMIT = 1_000_000

# Anonymous requests share one bucket. The Surface always sends a user_id.
_ANONYMOUS_KEY = "anonymous"

# Every request is also charged to this row so id rotation cannot buy
# unlimited spend.
_GLOBAL_KEY = "__global__"


def _global_limit() -> int:
    return int(os.getenv("GLOBAL_DAILY_TOKEN_LIMIT", "20000000"))


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS daily_usage ("
    "  day TEXT NOT NULL,"
    "  user_id TEXT NOT NULL,"
    "  tokens INTEGER NOT NULL DEFAULT 0,"
    "  PRIMARY KEY (day, user_id)"
    ")"
)


class DailyTokenLimitExceeded(Exception):
    """Raised when a user or the service has spent the daily token budget."""

    def __init__(
        self, user_id: str, used: int, limit: int, scope: str = "user"
    ) -> None:
        self.user_id = user_id
        self.used = used
        self.scope = scope
        if scope == "global":
            message = (
                f"The service has reached its daily token capacity "
                f"({used}/{limit}). Please try again tomorrow."
            )
        else:
            message = (
                f"Daily token limit reached ({used}/{limit}). "
                "Please try again tomorrow."
            )
        super().__init__(message)


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


def _user_key(user_id: str | None) -> str:
    # A caller claiming the global key would double-charge the backstop and
    # trip on the wrong limit, so it maps to the anonymous bucket.
    if not user_id or user_id == _GLOBAL_KEY:
        return _ANONYMOUS_KEY
    return user_id


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
    """Raise DailyTokenLimitExceeded when a budget is spent.

    Checks the per-user budget first, then the global backstop.
    """
    key = _user_key(user_id)
    used = _read_tokens(key)
    if used >= DAILY_TOKEN_LIMIT:
        raise DailyTokenLimitExceeded(key, used, DAILY_TOKEN_LIMIT)
    global_used = _read_tokens(_GLOBAL_KEY)
    if global_used >= _global_limit():
        raise DailyTokenLimitExceeded(
            _GLOBAL_KEY, global_used, _global_limit(), scope="global"
        )


def record_usage(user_id: str | None, tokens: int) -> None:
    """Add `tokens` to today's user and global counts, prune earlier days."""
    if tokens <= 0:
        return
    today = _today()
    with _connect() as conn:
        for key in (_user_key(user_id), _GLOBAL_KEY):
            conn.execute(
                "INSERT INTO daily_usage (day, user_id, tokens) VALUES (?, ?, ?) "
                "ON CONFLICT (day, user_id) DO UPDATE SET tokens = tokens + ?",
                (today, key, tokens, tokens),
            )
        conn.execute("DELETE FROM daily_usage WHERE day < ?", (today,))
