"""LangGraph implementation of the CMUGPT agent.

A single compiled `StateGraph` is the one source of truth for both the
non-streaming (`/agent/respond`) and streaming (`/agent/respond/stream`) HTTP
endpoints. The model emits plain Markdown. Deterministic nodes compute
`cmu_maps`, `services_used`, and `thought` into graph state.

Graph shape: ``START -> recall -> agent``. From ``agent`` either
``-> tools -> agent`` (when the model requested tool calls) or
``-> postprocess -> END`` (final answer). ``postprocess`` also schedules the
background memory-learn task before emitting ``done``, so a client disconnect
right after the final event cannot cancel it.

Streaming is done with LangGraph's custom stream channel: nodes emit typed
events through the injected `writer`, and the public entrypoints forward them as
``(event_name, data)`` tuples matching the existing SSE contract
(``status`` / ``map`` / ``delta`` / ``done`` / ``error``). When the graph is run
non-streaming via ``ainvoke`` the writes are simply dropped.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import os
import re
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter
from pydantic import SecretStr

from .cmu_maps import _apply_cmu_maps_guard, query_has_map_intent
from .guards import (
    apply_tool_transparency_guard,
    asks_about_tools,
    compute_thought,
    should_require_tool,
)
from .mcp_tools import filter_tools, load_mcp_tools, normalize_disabled_groups
from .memory import (
    FORGET_TOOL,
    REMEMBER_TOOL,
    build_memory_tools,
    ensure_store,
    is_internal_memory_tool,
    learn,
    recall,
)
from .prompts import build_system_prompt
from .schema import ActionType, AgentResponse, CmuMaps, Metadata, Thought, UserInput

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

StreamEvent = tuple[str, dict[str, Any]]


class AgentState(TypedDict):
    """Shared state threaded through the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    user_id: str | None
    memory_block: str
    tool_invocations: Annotated[list[dict[str, Any]], operator.add]
    services_used: Annotated[list[str], operator.add]
    response_text: str
    streamed: bool
    response_payload: dict[str, Any]
    # Tool groups the user switched off. Their tools are already unbound.
    # Postprocess reads this to suppress the CMU Maps embed too.
    disabled_tools: list[str]


# Background memory-extraction tasks are fire-and-forget. Hold references so the
# event loop doesn't garbage-collect them before they finish.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


async def drain_background_tasks(timeout: float = 15.0) -> None:
    """Finish in-flight memory learning before the database pool closes.

    Bounded wait so shutdown cannot hang on a stuck model request.
    """
    tasks = list(_BACKGROUND_TASKS)
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        logger.warning(
            "cancelling %d background memory task(s) after shutdown timeout",
            len(pending),
        )
        for task in pending:
            task.cancel()
    await asyncio.gather(*done, *pending, return_exceptions=True)


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


def _api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "")


@lru_cache(maxsize=16)
def _make_chat_model_for_key(model: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
    )


def _make_chat_model(model: str) -> ChatOpenAI:
    return _make_chat_model_for_key(model, _api_key())


def _message_text(message: AnyMessage | AIMessageChunk | None) -> str:
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    # Some providers return content as a list of parts.
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _helper_messages(query: str) -> list[dict[str, Any]]:
    """Minimal role/content list for the deterministic helpers."""
    return [{"role": "user", "content": query}]


def _fallback_response(text: str, confidence: float = 0.8) -> AgentResponse:
    return AgentResponse(
        thought=Thought(reasoning="Direct response", confidence=confidence),
        action=ActionType.RESPOND,
        tool_calls=[],
        response_text=text,
        metadata=Metadata(),
    )


def _needs_data_tools(query: str) -> bool:
    """True when this turn should pay the MCP/tool-schema latency cost."""
    if asks_about_tools(query):
        return True
    messages = _helper_messages(query)
    return should_require_tool(messages)


def _needs_memory_tools(query: str) -> bool:
    """True when the model needs explicit remember/forget tools this turn."""
    return bool(_MEMORY_TOOL_RE.search(query))


def _needs_memory_recall(query: str) -> bool:
    """True when recalled user memory is likely to change the answer."""
    return bool(_MEMORY_RECALL_RE.search(query))


def _build_agent_node(model: ChatOpenAI, tools: list[BaseTool], maps_enabled: bool):
    bound = model.bind_tools(tools) if tools else model
    bound_required = model.bind_tools(tools, tool_choice="required") if tools else model
    # Forcing a tool is only about CMU data lookups. Memory tools never count.
    # Identity comes from build_memory_tools' metadata marker, not the name,
    # so an MCP tool named "remember" still counts as data.
    has_data_tools = any(not is_internal_memory_tool(tool) for tool in tools)

    async def agent_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        query = state["query"]
        # Force a tool call only while the full toolset is bound: with a group
        # switched off, `tool_choice="required"` could coerce an unrelated
        # tool, wasting a call and misreporting how the answer was sourced.
        force_tool = (
            has_data_tools
            and not normalize_disabled_groups(state.get("disabled_tools"))
            and not state["services_used"]
            and should_require_tool(_helper_messages(query))
        )
        runnable = bound_required if force_tool else bound

        # Inject recalled memory as a second system message. It is never
        # persisted into the Surface's message history.
        call_messages = state["messages"]
        memory_block = state.get("memory_block") or ""
        if memory_block:
            base, *rest = call_messages
            call_messages = [base, SystemMessage(content=memory_block), *rest]

        # Buffer (don't live-stream) passes whose text postprocess may repair:
        # forced tool passes (preamble prose is not the final answer) and map
        # queries (false "couldn't look up" claims get stripped). With CMUMaps
        # off there is no map to contradict, so map queries stream normally.
        suppress_stream = force_tool or (maps_enabled and query_has_map_intent(query))

        gathered: AIMessageChunk | None = None
        saw_tool_call = False
        streamed_any = False
        async for chunk in runnable.astream(call_messages):
            if not isinstance(chunk, AIMessageChunk):
                continue
            gathered = chunk if gathered is None else gathered + chunk
            if chunk.tool_call_chunks:
                saw_tool_call = True
            text = _message_text(chunk)
            if text and not saw_tool_call and not suppress_stream:
                writer({"event": "delta", "data": {"text": text}})
                streamed_any = True

        if gathered is None:
            gathered = AIMessageChunk(content="")

        final_message = AIMessage(
            content=gathered.content,
            tool_calls=gathered.tool_calls,
        )

        if gathered.tool_calls:
            writer({"event": "status", "data": {"text": "Checking CMU tools..."}})
            return {"messages": [final_message]}

        return {
            "messages": [final_message],
            "response_text": _message_text(gathered),
            "streamed": streamed_any,
        }

    return agent_node


def _build_tools_node(tools: list[BaseTool]):
    tools_by_name = {tool.name: tool for tool in tools}
    # Tools this request built via build_memory_tools. Only these are trusted.
    # An MCP tool merely named "remember" stays untrusted below.
    internal_memory_names = {
        tool.name for tool in tools if is_internal_memory_tool(tool)
    }

    async def tools_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = last.tool_calls if isinstance(last, AIMessage) else []

        new_messages: list[AnyMessage] = []
        new_invocations: list[dict[str, Any]] = []
        new_services: list[str] = []

        for call in tool_calls:
            name = call["name"]
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{name}"
            tool = tools_by_name.get(name)
            memory_id: str | None = None
            memory_fact: str | None = None
            is_memory_tool = name in internal_memory_names
            memory_op_failed = False
            if tool is None:
                result = f"Tool '{name}' is not available."
            else:
                try:
                    raw = await tool.ainvoke(args)
                    if (
                        is_memory_tool
                        and name == REMEMBER_TOOL
                        and isinstance(raw, dict)
                    ):
                        raw_message = raw.get("message")
                        raw_memory_id = raw.get("memory_id")
                        raw_fact = raw.get("fact")
                        result = (
                            raw_message
                            if isinstance(raw_message, str)
                            else "Memory saved."
                        )
                        memory_id = (
                            raw_memory_id if isinstance(raw_memory_id, str) else None
                        )
                        memory_fact = raw_fact if isinstance(raw_fact, str) else None
                    else:
                        result = raw if isinstance(raw, str) else str(raw)
                except Exception as exc:  # noqa: BLE001 - surface as tool data
                    if is_memory_tool:
                        # Raw exception text can leak DSN fragments. Log it,
                        # send a generic message onward.
                        logger.warning("memory tool %s failed", name, exc_info=True)
                        result = "The memory operation failed; nothing was changed."
                        memory_op_failed = True
                    else:
                        result = f"Tool '{name}' failed: {exc}"

            if is_memory_tool:
                # Internal tools: results are our own trusted confirmations,
                # never listed as user-facing services. The Surface renders
                # the `memory` event as a chip.
                # Chip only when stored memory actually changed.
                no_op_forget = name == FORGET_TOOL and result.startswith(
                    "No matching memory"
                )
                if not memory_op_failed and not no_op_forget:
                    event_data: dict[str, Any] = {
                        "op": "remove" if name == FORGET_TOOL else "add",
                        "text": result,
                    }
                    if memory_id:
                        event_data["id"] = memory_id
                        event_data["kind"] = "remembered"
                    if memory_fact:
                        event_data["fact"] = memory_fact
                    writer({"event": "memory", "data": event_data})
                new_messages.append(ToolMessage(content=result, tool_call_id=call_id))
                continue

            new_invocations.append({"name": name, "arguments": args, "result": result})
            if name not in state["services_used"] and name not in new_services:
                new_services.append(name)

            # Wrap tool output so the model treats it as untrusted DATA, not as
            # instructions. Defense against prompt-injection from MCP content.
            wrapped = (
                f'<<<TOOL_OUTPUT name="{name}" trust="untrusted-data">>>\n'
                f"{result}\n"
                "<<<END_TOOL_OUTPUT>>>"
            )
            new_messages.append(ToolMessage(content=wrapped, tool_call_id=call_id))

        writer({"event": "status", "data": {"text": "Writing answer..."}})
        return {
            "messages": new_messages,
            "tool_invocations": new_invocations,
            "services_used": new_services,
        }

    return tools_node


async def _postprocess_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
    query = state["query"]
    msgs = _helper_messages(query)
    invocations = state["tool_invocations"]
    services = state["services_used"]

    text = (state.get("response_text") or "").strip()
    if not text:
        text = (
            "I'm sorry, I couldn't generate a response for that. "
            "Please try rephrasing your question."
        )

    parsed = AgentResponse(
        thought=Thought(reasoning="Direct response", confidence=0.5),
        action=ActionType.RESPOND,
        tool_calls=[],
        response_text=text,
        services_used=list(services),
        cmu_maps=CmuMaps(),
        metadata=Metadata(),
    )

    # A switched-off CMUMaps means no map embed either, not just no map tools:
    # the guard below is what attaches the deterministic map to the answer.
    if "maps" not in normalize_disabled_groups(state.get("disabled_tools")):
        parsed = _apply_cmu_maps_guard(parsed, msgs, invocations)
    parsed = apply_tool_transparency_guard(parsed, msgs, services)
    parsed.thought = compute_thought(services, invocations, parsed.response_text)
    parsed.action = ActionType.RETRIEVE if services else ActionType.RESPOND

    # Buffered answers were never streamed. Emit the repaired text now.
    if not state.get("streamed") and parsed.response_text:
        writer({"event": "delta", "data": {"text": parsed.response_text}})

    if parsed.cmu_maps.url:
        writer({"event": "map", "data": parsed.cmu_maps.model_dump()})

    # Schedule the learn task BEFORE emitting `done`: clients often disconnect
    # right after the final event, which cancels the graph, and the task must
    # already exist by then or memories are silently never learned.
    user_id = state.get("user_id")
    if user_id and parsed.response_text:
        task = asyncio.create_task(_safe_learn(user_id, query, parsed.response_text))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    payload = parsed.model_dump()
    writer({"event": "done", "data": payload})
    return {"response_payload": payload, "response_text": parsed.response_text}


def _build_recall_node(store: BaseStore):
    """Read path: fetch top-k relevant memory and stage it for the agent node."""

    async def recall_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        user_id = state.get("user_id")
        if not user_id:
            return {}
        block = await recall(store, user_id, state["query"])
        return {"memory_block": block} if block else {}

    return recall_node


async def _safe_learn(user_id: str, query: str, response_text: str) -> None:
    """Background memory-learn pass. Best-effort, never surfaces a failure."""
    try:
        store = await ensure_store()
        await learn(store, user_id, query, response_text)
    except Exception:
        logger.warning("background memory learn failed", exc_info=True)


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "postprocess"


def build_graph(
    model: ChatOpenAI,
    tools: list[BaseTool],
    store: BaseStore | None,
    *,
    recall_enabled: bool,
    maps_enabled: bool = True,
):
    """Compile the agent graph for one request (model + tools + store captured).

    Shape: ``START -> recall -> agent`` then either ``-> tools -> agent`` or
    ``-> postprocess -> END``. Postprocess schedules the background learn task.
    `tools` must already have the user's disabled groups filtered out: the
    agent node binds exactly this list, so anything missing here is uncallable.
    """
    # ty doesn't yet structurally match TypedDict's synthesized __required_keys__/
    # __optional_keys__ against langgraph's StateLike protocol (confirmed still
    # failing on ty 0.0.65). AgentState is a plain TypedDict, the canonical shape
    # LangGraph expects here.
    graph = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]
    graph.add_node("agent", _build_agent_node(model, tools, maps_enabled))
    graph.add_node("tools", _build_tools_node(tools))
    graph.add_node("postprocess", _postprocess_node)

    if recall_enabled and store is not None:
        graph.add_node("recall", _build_recall_node(store))
        graph.add_edge(START, "recall")
        graph.add_edge("recall", "agent")
    else:
        graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "postprocess": "postprocess"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("postprocess", END)
    if store is None:
        return graph.compile()
    return graph.compile(store=store)


@lru_cache(maxsize=16)
def _build_no_tool_graph(model: str, api_key: str, maps_enabled: bool):
    """Cached graph for no-tool/no-memory turns. Saves compilation time only."""
    return build_graph(
        _make_chat_model_for_key(model, api_key),
        [],
        None,
        recall_enabled=False,
        maps_enabled=maps_enabled,
    )


def _graph_for_request(
    model: str,
    tools: list[BaseTool],
    store: BaseStore | None,
    *,
    recall_enabled: bool,
    maps_enabled: bool,
):
    if not tools and store is None and not recall_enabled:
        return _build_no_tool_graph(model, _api_key(), maps_enabled)
    return build_graph(
        _make_chat_model(model),
        tools,
        store,
        recall_enabled=recall_enabled,
        maps_enabled=maps_enabled,
    )


def _sanitize_history(
    message_history: list[dict[str, str]] | None,
) -> list[AnyMessage]:
    """Convert caller history to LangChain messages, dropping non user/assistant.

    We own the system prompt. Smuggled `system`/`tool` turns are an injection
    vector, so only `user` and `assistant` turns are carried over.
    """
    if not message_history:
        return []
    out: list[AnyMessage] = []
    for turn in message_history:
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _initial_state(
    user_input: UserInput,
    message_history: list[dict[str, str]] | None,
    tools: list[BaseTool],
    disabled_tools: list[str] | None = None,
) -> AgentState:
    prompt = build_system_prompt(tools, disabled_tools)
    messages: list[AnyMessage] = [SystemMessage(content=prompt)]
    messages.extend(_sanitize_history(message_history))
    messages.append(HumanMessage(content=user_input.query))
    return AgentState(
        messages=messages,
        query=user_input.query,
        user_id=user_input.user_id,
        memory_block="",
        tool_invocations=[],
        services_used=[],
        response_text="",
        streamed=False,
        response_payload={},
        disabled_tools=list(disabled_tools or []),
    )


async def _prepare_tools_and_store(
    user_input: UserInput,
    disabled_tools: list[str] | None,
) -> tuple[list[BaseTool], BaseStore | None, bool, bool]:
    """Plan the turn and prepare only the tools/store it can actually use.

    Ordinary chat skips MCP discovery, schema binding, and store setup for
    latency. Every turn still gets the canonical security policy. Disabled
    tool groups are filtered out first. Memory tools are appended after, so a
    toggle can never remove them.
    """
    query = user_input.query
    user_id = user_input.user_id

    needs_data_tools = _needs_data_tools(query)
    needs_memory_tools = bool(user_id) and _needs_memory_tools(query)
    recall_enabled = bool(user_id) and _needs_memory_recall(query)
    maps_enabled = "maps" not in normalize_disabled_groups(disabled_tools)

    tools: list[BaseTool] = []
    if needs_data_tools:
        tools.extend(filter_tools(await load_mcp_tools(), disabled_tools))

    store: BaseStore | None = None
    if recall_enabled or needs_memory_tools:
        store = await ensure_store()

    if user_id and needs_memory_tools and store is not None:
        tools = [*tools, *build_memory_tools(store, user_id)]

    return tools, store, recall_enabled, maps_enabled


async def run_agent(
    user_input: UserInput,
    model: str = "openai/gpt-5.4-mini",
    message_history: list[dict[str, str]] | None = None,
    disabled_tools: list[str] | None = None,
) -> AgentResponse:
    """Non-streaming entry point. Runs the graph and returns the full response.

    `disabled_tools` lists the tool groups the user switched off in the Surface
    (`maps`, `courses`, `eats`, `guide`). Those tools are never bound.
    """
    if not _api_key():
        return _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )

    tools, store, recall_enabled, maps_enabled = await _prepare_tools_and_store(
        user_input, disabled_tools
    )
    graph = _graph_for_request(
        model,
        tools,
        store,
        recall_enabled=recall_enabled,
        maps_enabled=maps_enabled,
    )
    final = await graph.ainvoke(
        _initial_state(user_input, message_history, tools, disabled_tools)
    )

    payload = final.get("response_payload")
    if isinstance(payload, dict) and payload:
        return AgentResponse(**payload)
    return _fallback_response(
        "Unable to complete the request.",
        confidence=0.3,
    )


async def stream_agent_response(
    *,
    user_input: UserInput,
    model: str,
    message_history: list[dict[str, str]] | None,
    disabled_tools: list[str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Streaming entry point: yields ('delta', ...) ... ('done', ...) events."""
    if not _api_key():
        fb = _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )
        yield ("delta", {"text": fb.response_text})
        yield ("done", fb.model_dump())
        return

    tools, store, recall_enabled, maps_enabled = await _prepare_tools_and_store(
        user_input, disabled_tools
    )
    graph = _graph_for_request(
        model,
        tools,
        store,
        recall_enabled=recall_enabled,
        maps_enabled=maps_enabled,
    )

    async for chunk in graph.astream(
        _initial_state(user_input, message_history, tools, disabled_tools),
        stream_mode="custom",
    ):
        if isinstance(chunk, dict) and "event" in chunk:
            yield (chunk["event"], chunk.get("data", {}))
