"""Builds the chat models this service reaches through OpenRouter.

Every LangChain model object is created here, so the agent graph, background
memory extraction, and chat titles share one client per configuration. A
cached client keeps its HTTP connection open, which saves a TLS handshake on
every call after the first.
"""

import os
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def api_key() -> str:
    """The OpenRouter key from the environment, or an empty string."""
    return os.getenv("OPENROUTER_API_KEY", "")


@lru_cache(maxsize=32)
def _build(
    model: str,
    key: str,
    temperature: float | None,
    stream_usage: bool,
    reasoning_off: bool,
) -> ChatOpenAI:
    extra: dict[str, Any] = {}
    if stream_usage:
        # The last streamed chunk then carries real token counts, which the
        # daily budget records instead of estimating from text length.
        extra["stream_usage"] = True
    if reasoning_off:
        # Some OpenRouter models think out loud before answering. That is
        # wasted on a one-line title, so a caller can switch it off.
        extra["extra_body"] = {"reasoning": {"enabled": False}}
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(key),
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        **extra,
    )


def chat_model(
    model: str,
    *,
    temperature: float | None = None,
    stream_usage: bool = False,
    reasoning_off: bool = False,
) -> ChatOpenAI:
    """Return a cached client for `model` under the current API key.

    The key is read on every call and is part of the cache key, so a rotated
    OPENROUTER_API_KEY takes effect without a restart.
    """
    return _build(model, api_key(), temperature, stream_usage, reasoning_off)
