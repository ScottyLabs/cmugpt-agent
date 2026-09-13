"""One-shot chat title generation from a chat's first user message.

Called once per chat by the surface server, concurrently with the main
agent turn, so it must stay cheap.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import api_key, chat_model
from .moderation import ALLOW, moderate_text
from .settings import get_settings

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 600
_MAX_TITLE_CHARS = 48

_SYSTEM_PROMPT = (
    "You write titles for chat conversations. Given the user's first "
    "message, reply with only a title: 2 to 5 words, sentence case, a plain "
    "noun phrase naming the topic. No quotes, no trailing punctuation, no "
    "emojis. Match the message's language."
)


def _title_model_name() -> str:
    return get_settings().title_model


def _clean(raw: str) -> str | None:
    """Normalize model output into a display-safe title, or None if empty."""
    title = raw.strip().split("\n")[0].strip().strip("\"'" + "\u201c\u201d")
    title = title.rstrip(".!?,;:")
    if not title:
        return None
    if len(title) > _MAX_TITLE_CHARS:
        title = title[: _MAX_TITLE_CHARS - 3].rstrip() + "..."
    return title


async def generate_chat_title(first_message: str) -> str | None:
    """Return a short title for the chat, or None on failure."""
    text = first_message.strip()
    if not api_key() or not text:
        return None
    # A flagged first message must not be echoed into a title. "New chat" is
    # the surface's default title, so returning it both neutralizes the title
    # and lets the next allowed message claim it.
    verdict = await moderate_text(text)
    if verdict.action != ALLOW:
        return "New chat"
    model = chat_model(_title_model_name(), temperature=0.0, reasoning_off=True)
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
