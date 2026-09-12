"""Background learning: distill durable facts from a finished exchange.

Runs off the response path, rate limited per user, and never stores raw
chat turns. Only facts the extractor returns are written, through add_fact.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from ..llm import chat_model
from .facts import add_fact
from .store import FACTS, item_text, search

# Budget for the background learn() pass, which costs one extraction LLM
LEARN_MIN_INTERVAL_SECONDS = 2.0
LEARN_MAX_PER_HOUR = 60


# The extraction model reads the exchange and returns an empty list when
# nothing durable was said.

_EXTRACTION_SYSTEM = (
    "You maintain a long-term memory of durable facts about a user across "
    "chats. Given the latest exchange, output a JSON array of short, "
    "third-person facts about the USER that are stable and useful to remember "
    "in future conversations (identity, role, stable preferences, ongoing "
    "context). Rules:\n"
    "- Only facts the user stated or clearly implied about themselves.\n"
    "- No transient or one-off details (today's plan, a single lookup, the "
    "answer to the current question).\n"
    "- No sensitive data beyond what the user volunteered.\n"
    "- Each fact under 120 characters; do not restate the existing facts.\n"
    "- If nothing qualifies, output []. Output ONLY the JSON array."
)


# Per-user timestamps of recent learn() runs. In-process only. Bounded below.
_learn_history: dict[str, deque[float]] = {}


def _learn_allowed(user_id: str, *, now: float | None = None) -> bool:
    """Per-user budget: minimum gap between runs plus an hourly ceiling."""
    current = time.monotonic() if now is None else now
    if len(_learn_history) > 10_000:  # bound in-process bookkeeping
        stale = [
            uid
            for uid, times in _learn_history.items()
            if not times or current - times[-1] > 3600
        ]
        for uid in stale:
            del _learn_history[uid]
    times = _learn_history.setdefault(user_id, deque())
    while times and current - times[0] > 3600:
        times.popleft()
    if times and current - times[-1] < LEARN_MIN_INTERVAL_SECONDS:
        return False
    if len(times) >= LEARN_MAX_PER_HOUR:
        return False
    times.append(current)
    return True


def _extraction_model_name() -> str:
    return os.getenv("MEMORY_EXTRACTION_MODEL", "qwen/qwen3.7-flash")


def _extractor_model() -> ChatOpenAI:
    return chat_model(_extraction_model_name(), temperature=0.0)


def _parse_facts(raw: str) -> list[str]:
    """Parse the extractor's reply tolerantly: decode the first JSON array.

    ``raw_decode`` from the first ``[`` ignores trailing prose entirely.
    """
    start = raw.find("[")
    if start == -1:
        return []
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:].strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        entry.strip()[:200]
        for entry in data
        if isinstance(entry, str) and entry.strip()
    ]


async def learn(
    store: BaseStore,
    user_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Background write path: distill durable facts from the latest exchange.

    Best-effort, run off the response path, rate-limited per user.
    """
    if not user_text.strip():
        return
    if not _learn_allowed(user_id):
        return

    existing = await search(store, (user_id, FACTS), user_text, 12)
    known = "\n".join(f"- {item_text(item)}" for item in existing if item_text(item))
    payload = (
        f"Latest exchange:\nUser: {user_text}\nAssistant: {assistant_text}\n\n"
        f"Existing known facts:\n{known or '(none)'}"
    )
    response = await _extractor_model().ainvoke(
        [SystemMessage(content=_EXTRACTION_SYSTEM), HumanMessage(content=payload)]
    )
    content = response.content if isinstance(response.content, str) else ""
    for fact in _parse_facts(content):
        await add_fact(store, user_id, fact, source="extraction")
