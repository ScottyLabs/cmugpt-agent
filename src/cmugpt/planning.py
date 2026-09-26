"""Per-turn planning that runs before the graph.

Decides which tool groups to discover and bind, whether the remember and
forget tools are needed, whether memory recall runs, and how much history to
carry. Every decision is a regex match or a list slice, with no model call,
so a conversational turn spends no tokens on tool schemas.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from .guards import asks_about_tools, should_require_tool
from .maps.tool import build_show_map_tool
from .mcp_tools import (
    filter_tools,
    load_mcp_tools,
    normalize_disabled_groups,
    select_tools_for_query,
)
from .memory import build_memory_tools, ensure_store
from .schema import UserInput

# History is billed on every model pass. Sixty messages is thirty exchanges.
_HISTORY_MAX_MESSAGES = 60
_HISTORY_MAX_MESSAGE_CHARS = 12_000

# User turns scanned for tool-group narrowing. Local regex only, no tokens.
_HISTORY_HINT_TURNS = 20


_MEMORY_TOOL_RE = re.compile(
    r"\b("
    r"remember|don['\u2019]?t\s+forget|forget|delete\s+(?:my\s+)?memory|"
    r"remove\s+(?:that|this|it|my\s+memory)|what\s+do\s+you\s+remember|"
    r"what\s+do\s+you\s+know\s+about\s+me"
    r")\b",
    re.IGNORECASE,
)

_MEMORY_RECALL_RE = re.compile(
    r"\b("
    r"my|me|for\s+me|i['\u2019]?m|im|i\s+am|i\s+have|i\s+need|i\s+prefer|"
    r"i\s+(?:like|love|enjoy|hate|dislike|want|wish|told|said|mentioned)|"
    r"(?:do|did|can|could|would|should|have|am|was)\s+i|"
    r"preference|prefer|allerg|diet|vegetarian|vegan|major|minor|class|"
    r"favorite|favourite|schedule|recommend|suggest|where\s+should|"
    r"what\s+should|about\s+me|know\s+me|remember\s+(?:about\s+)?me|"
    r"based\s+on\s+(?:what|anything)\s+you\s+(?:know|remember)|"
    r"what\s+do\s+you\s+remember|what\s+do\s+you\s+know\s+about\s+me"
    r")\b",
    re.IGNORECASE,
)


def helper_messages(query: str) -> list[dict[str, Any]]:
    """Minimal role/content list for the deterministic helpers."""
    return [{"role": "user", "content": query}]


# A follow-up such as "what about Wean?" rarely repeats a data keyword, so
# the last few user turns are scanned as well. A conversation that needed
# tools then keeps them.
_HISTORY_GATE_TURNS = 4


def needs_data_tools(
    query: str,
    message_history: list[dict[str, str]] | None = None,
) -> bool:
    """True when this turn should pay the MCP/tool-schema latency cost."""
    if asks_about_tools(query):
        return True
    if should_require_tool(helper_messages(query)):
        return True
    recent_user_turns = [
        turn.get("content", "")
        for turn in (message_history or [])
        if turn.get("role") == "user" and isinstance(turn.get("content"), str)
    ][-_HISTORY_GATE_TURNS:]
    return any(should_require_tool(helper_messages(text)) for text in recent_user_turns)


_MEMORY_CONTEXT_TURNS = 5


def _recent_history_texts(
    message_history: list[dict[str, str]] | None,
) -> list[str]:
    if not message_history:
        return []
    return [
        turn["content"]
        for turn in message_history[-_MEMORY_CONTEXT_TURNS:]
        if isinstance(turn.get("content"), str)
    ]


def needs_memory_tools(query: str, history_texts: list[str] | None = None) -> bool:
    """True when the model needs explicit remember/forget tools this turn."""
    if _MEMORY_TOOL_RE.search(query):
        return True
    return any(_MEMORY_TOOL_RE.search(text) for text in history_texts or [])


def needs_memory_recall(query: str, history_texts: list[str] | None = None) -> bool:
    """True when recalled user memory is likely to change the answer."""
    if _MEMORY_RECALL_RE.search(query):
        return True
    return any(_MEMORY_RECALL_RE.search(text) for text in history_texts or [])


def _cap_history_text(content: str) -> str:
    if len(content) <= _HISTORY_MAX_MESSAGE_CHARS:
        return content
    # Retain the head, since answers front-load the substance that
    # follow-ups reference.
    return content[:_HISTORY_MAX_MESSAGE_CHARS] + "\n[earlier turn truncated]"


def sanitize_history(
    message_history: list[dict[str, str]] | None,
) -> list[AnyMessage]:
    """Convert caller history to sanitized LangChain messages.

    We own the system prompt. Smuggled `system`/`tool` turns are an injection
    vector, so only `user` and `assistant` turns are carried over.
    """
    if not message_history:
        return []
    out: list[AnyMessage] = []
    for turn in message_history[-_HISTORY_MAX_MESSAGES:]:
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            out.append(HumanMessage(content=_cap_history_text(content)))
        elif role == "assistant":
            out.append(AIMessage(content=_cap_history_text(content)))
    return out


def _history_hint_texts(
    message_history: list[dict[str, str]] | None,
) -> list[str]:
    """User turns supplied to tool-group narrowing.

    Restricted to user turns because assistant turns reproduce tool data
    verbatim and would therefore match every group.
    """
    if not message_history:
        return []
    texts = [
        turn["content"]
        for turn in message_history
        if turn.get("role") == "user" and isinstance(turn.get("content"), str)
    ]
    return texts[-_HISTORY_HINT_TURNS:]


async def prepare_tools_and_store(
    user_input: UserInput,
    disabled_tools: list[str] | None,
    message_history: list[dict[str, str]] | None,
) -> tuple[list[BaseTool], BaseStore | None, bool, bool]:
    """Plan the turn and prepare only the tools/store it can actually use.

    Ordinary chat skips MCP discovery, schema binding, and store setup for
    latency. Every turn still gets the canonical security policy. Disabled
    tool groups are filtered out first, then the survivors are narrowed to
    the query so a keyword match cannot re-bind a disabled group. Memory
    tools are appended after, so a toggle can never remove them.
    """
    query = user_input.query
    user_id = user_input.user_id

    wants_data_tools = needs_data_tools(query, message_history)
    recent_texts = _recent_history_texts(message_history)
    wants_memory_tools = bool(user_id) and needs_memory_tools(query, recent_texts)
    recall_enabled = bool(user_id) and needs_memory_recall(query, recent_texts)
    maps_enabled = "maps" not in normalize_disabled_groups(disabled_tools)

    tools: list[BaseTool] = []
    if wants_data_tools:
        # Narrowing runs after the disabled-group filter so that a keyword
        # match can never re-bind a disabled group.
        mcp_tools = filter_tools(await load_mcp_tools(), disabled_tools)
        tools.extend(
            select_tools_for_query(
                mcp_tools, query, _history_hint_texts(message_history)
            )
        )
    if maps_enabled:
        # Bound outside the data-tools gate on purpose: the map is the model's
        # decision, so the tool must always be in its hands. Keyword gating here
        # would decide navigation before the model could. The tool is local, so
        # it costs no MCP discovery. The price is the catalog section on every
        # turn. Postprocess still validates every proposal, and query inference
        # remains only a fallback.
        tools.append(build_show_map_tool())

    store: BaseStore | None = None
    if recall_enabled or wants_memory_tools:
        store = await ensure_store()

    if user_id and wants_memory_tools and store is not None:
        tools = [*tools, *build_memory_tools(store, user_id)]

    return tools, store, recall_enabled, maps_enabled
