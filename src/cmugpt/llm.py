"""Factory for the OpenRouter chat model clients.

All ChatOpenAI instances are created here, so the agent graph, memory
extraction, and title generation share one cached client per configuration.
A reused client keeps its HTTP connection open across calls.
"""

from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .settings import get_settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def api_key() -> str:
    """The OpenRouter key from the environment, or an empty string."""
    return get_settings().openrouter_api_key


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
        # The final streamed chunk then carries measured token counts, which
        # the daily budget records instead of an estimate from text length.
        extra["stream_usage"] = True
    if reasoning_off:
        # Reasoning models emit a thinking phase before the answer. Callers that
        # need a one-line reply, such as title generation, disable it.
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
