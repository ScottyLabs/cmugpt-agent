"""One-shot chat title generation from a chat's first user message.

Called once per chat by the surface server, concurrently with the main
agent turn, so it must stay cheap.
"""

import logging
import os
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.moderation import ALLOW, moderate_text

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_MAX_INPUT_CHARS = 600
_MAX_TITLE_CHARS = 48

_SYSTEM_PROMPT = (
    "You write titles for chat conversations. Given the user's first "
    "message, reply with only a title: 2 to 5 words, sentence case, a plain "
    "noun phrase naming the topic. No quotes, no trailing punctuation, no "
    "emojis. Match the message's language."
)


def _title_model_name() -> str:
    return os.getenv("TITLE_MODEL", "qwen/qwen3.7-flash")


@lru_cache(maxsize=4)
def _title_model_for_key(model: str, api_key: str) -> ChatOpenAI:
    # Cached so repeated titles reuse one HTTP client (keep-alive) instead of
    # a new TLS handshake per chat.
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
        temperature=0.0,
        extra_body={"reasoning": {"enabled": False}},
    )


def _clean(raw: str) -> str | None:
    """Normalize model output into a display-safe title, or None if empty."""
    title = raw.strip().split("\n")[0].strip().strip("\"'“”")
    title = title.rstrip(".!?,;:")
    if not title:
        return None
    if len(title) > _MAX_TITLE_CHARS:
        title = title[: _MAX_TITLE_CHARS - 3].rstrip() + "..."
    return title


async def generate_chat_title(first_message: str) -> str | None:
    """Return a short title for the chat, or None on failure."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    text = first_message.strip()
    if not api_key or not text:
        return None
    # A flagged first message must not be echoed into a title. "New chat" is
    # the surface's default title, so returning it both neutralizes the title
    # and lets the next allowed message claim it.
    verdict = await moderate_text(text)
    if verdict.action != ALLOW:
        return "New chat"
    model = _title_model_for_key(_title_model_name(), api_key)
    try:
        reply = await model.ainvoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=text[:_MAX_INPUT_CHARS]),
            ]
        )
        return _clean(str(reply.content))
    except Exception as exc:  # noqa: BLE001 - never break the turn over a title
        logger.warning("title: generation failed (%s)", exc)
        return None
