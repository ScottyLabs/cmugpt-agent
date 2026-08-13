"""LangGraph implementation of the CMUGPT agent.

A single compiled `StateGraph` is the one source of truth for both the
non-streaming (`/agent/respond`) and streaming (`/agent/respond/stream`) HTTP
endpoints. The model emits plain Markdown and proposes the campus map by
calling the local maps_show_map tool. Deterministic nodes validate that
proposal, fall back to query inference, and compute cmu_maps, services_used,
and thought into graph state.

Graph shape: ``START -> recall -> agent``. From ``agent`` either
``-> tools -> agent`` (when the model requested tool calls) or
``-> postprocess -> END`` (final answer). ``postprocess`` also schedules the
background memory-learn task before emitting ``done``, so a client disconnect
right after the final event cannot cancel it.

Streaming uses LangGraph's custom stream channel. Nodes emit typed events
through the injected `writer` and the public entrypoints forward them as
``(event_name, data)`` tuples matching the existing SSE contract
(``status`` / ``map`` / ``delta`` / ``done`` / ``error``). A non-streaming
``ainvoke`` run simply drops the writes.
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
from langchain_core.tools import BaseTool, ToolException
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.types import StreamWriter
from pydantic import SecretStr

from .cmu_maps import SHOW_MAP_TOOL_NAME, _apply_cmu_maps_guard, query_has_map_intent
from .guards import (
    REFUSAL_TEXT,
    StreamScrubber,
    apply_output_guard,
    apply_tool_transparency_guard,
    asks_about_tools,
    canned_refusal_response,
    compute_thought,
    is_flagrant_injection,
    should_require_tool,
)
from .map_tool import build_show_map_tool
from .mcp_tools import (
    filter_tools,
    load_mcp_tools,
    normalize_disabled_groups,
    select_tools_for_query,
)
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
from .token_limits import record_usage

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

StreamEvent = tuple[str, dict[str, Any]]


# Safety limits rather than tuning parameters. The values are set high
# enough that ordinary conversations never reach them, so they engage only
# on anomalous input.

# History is billed on every model pass. Sixty messages is thirty exchanges.
_HISTORY_MAX_MESSAGES = 60
_HISTORY_MAX_MESSAGE_CHARS = 12_000

# User turns scanned for tool-group narrowing. Local regex only, no tokens.
_HISTORY_HINT_TURNS = 20

# Tool results are resent on every subsequent pass. Twelve thousand
# characters accommodates every current CMU tool result, including the 9k
# full dining list, so the cap engages only if a tool begins returning
# substantially more. The marker prevents the model from presenting a
# truncated list as complete.
_TOOL_RESULT_MAX_CHARS = 12_000
_TOOL_RESULT_TRUNCATION_MARKER = (
    "\n[Result truncated. More entries exist beyond this point.]"
)


class AgentState(TypedDict):
    """Shared state threaded through the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    # Memory namespace owner and owner of the daily token budget for this run.
    user_id: str | None
    memory_block: str
    tool_invocations: Annotated[list[dict[str, Any]], operator.add]
    services_used: Annotated[list[str], operator.add]
    response_text: str
    streamed: bool
    response_payload: dict[str, Any]
    # Tool groups the user switched off in the Surface. Their tools are
    # already unbound. Postprocess reads this to keep the map embed off too.
    disabled_tools: list[str]
    # Completed tool rounds. Drives the unbound later passes.
    tool_rounds: Annotated[int, operator.add]
    # Persists across passes so that a subsequent clean pass cannot clear a
    # detection.
    leak_detected: Annotated[bool, operator.or_]


# Background memory-extraction tasks run detached from the response. Python
# discards a task nothing refers to, so references are held here until each
# task completes.
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
        # Report usage on the final stream chunk so the budget records
        # measured consumption rather than estimates.
        stream_usage=True,
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


# A follow-up such as "what about Wean?" rarely repeats a data keyword, so
# the last few user turns are scanned as well. A conversation that needed
# tools then keeps them.
_HISTORY_GATE_TURNS = 4


def _needs_data_tools(
    query: str,
    message_history: list[dict[str, str]] | None = None,
) -> bool:
    """True when this turn should pay the MCP/tool-schema latency cost."""
    if asks_about_tools(query):
        return True
    if should_require_tool(_helper_messages(query)):
        return True
    recent_user_turns = [
        turn.get("content", "")
        for turn in (message_history or [])
        if turn.get("role") == "user" and isinstance(turn.get("content"), str)
    ][-_HISTORY_GATE_TURNS:]
    return any(
        should_require_tool(_helper_messages(text)) for text in recent_user_turns
    )


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


def _needs_memory_tools(query: str, history_texts: list[str] | None = None) -> bool:
    """True when the model needs explicit remember/forget tools this turn."""
    if _MEMORY_TOOL_RE.search(query):
        return True
    return any(_MEMORY_TOOL_RE.search(text) for text in history_texts or [])


def _needs_memory_recall(query: str, history_texts: list[str] | None = None) -> bool:
    """True when recalled user memory is likely to change the answer."""
    if _MEMORY_RECALL_RE.search(query):
        return True
    return any(_MEMORY_RECALL_RE.search(text) for text in history_texts or [])


def _had_tool_round(messages: list[AnyMessage]) -> bool:
    """True once this run has executed any tool round, memory tools included.

    Sanitized history contains only user/assistant turns, so a ToolMessage can
    only come from this run.
    """
    return any(isinstance(message, ToolMessage) for message in messages)


def _record_pass_usage(state: AgentState, gathered: AIMessageChunk) -> None:
    """Charge one model pass to the user's daily budget and log it.

    Falls back to a characters/4 estimate so the budget remains enforceable
    when the stream carries no usage metadata. The log line is what allows
    the caps and thresholds to be tuned from production data rather than
    estimated.
    """
    usage = getattr(gathered, "usage_metadata", None) or {}
    estimated = not usage.get("total_tokens")
    if estimated:
        input_chars = sum(len(_message_text(m)) for m in state["messages"])
        total = (input_chars + len(_message_text(gathered))) // 4
    else:
        total = int(usage["total_tokens"])
    logger.info(
        "pass_usage user=%s round=%s input=%s output=%s total=%s estimated=%s",
        (state.get("user_id") or "anonymous")[:8],
        state.get("tool_rounds", 0),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        total,
        estimated,
    )
    try:
        record_usage(state.get("user_id"), total)
    except Exception:
        # An in-flight answer must not be interrupted, but a silent failure
        # here would disable the budget, so it is logged explicitly.
        logger.exception("token budget recording failed")


def _truncate_tool_result(result: str) -> str:
    if len(result) <= _TOOL_RESULT_MAX_CHARS:
        return result
    return result[:_TOOL_RESULT_MAX_CHARS] + _TOOL_RESULT_TRUNCATION_MARKER


def _build_agent_node(model: ChatOpenAI, tools: list[BaseTool], maps_enabled: bool):
    bound = model.bind_tools(tools) if tools else model
    bound_required = model.bind_tools(tools, tool_choice="required") if tools else model
    # Forcing a tool call applies only to CMU data lookups, so memory tools
    # are not counted. A tool is recognized as a memory tool by the marker
    # build_memory_tools sets, never by its name, so an MCP tool that
    # happens to be named "remember" still counts as a data tool.
    has_data_tools = any(not is_internal_memory_tool(tool) for tool in tools)

    async def agent_node(state: AgentState, writer: StreamWriter) -> dict[str, Any]:
        query = state["query"]
        # A tool call is forced only while the full toolset is bound. With a
        # group disabled, `tool_choice="required"` could push the model into
        # an unrelated tool, wasting a call and misreporting how the answer
        # was sourced. The latch must also count memory-only rounds, since
        # services_used stays empty for those, and a second forced pass
        # would again pick an unrelated tool or loop to the recursion limit.
        force_tool = (
            has_data_tools
            and not normalize_disabled_groups(state.get("disabled_tools"))
            and not _had_tool_round(state["messages"])
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

        # Live deltas cannot be retracted, so they trail the scrubber's
        # holdback. Buffered passes are instead scanned in postprocess.
        scrubber: StreamScrubber | None = None
        if not suppress_stream:
            prompt_text = (
                _message_text(state["messages"][0]) if state["messages"] else ""
            )
            scrubber = StreamScrubber(prompt_text)
        withheld_notice_sent = False

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
            if text and not saw_tool_call and scrubber is not None:
                safe = scrubber.push(text)
                if safe:
                    writer({"event": "delta", "data": {"text": safe}})
                    streamed_any = True
                elif scrubber.tripped and streamed_any and not withheld_notice_sent:
                    writer(
                        {"event": "delta", "data": {"text": "\n\n[Response withheld.]"}}
                    )
                    withheld_notice_sent = True

        if gathered is None:
            gathered = AIMessageChunk(content="")

        # Flush the held tail on every exit path. Otherwise the final
        # characters of a preamble preceding a tool call would be lost.
        if scrubber is not None:
            tail = scrubber.flush()
            if tail:
                writer({"event": "delta", "data": {"text": tail}})
                streamed_any = True
            if scrubber.tripped and streamed_any and not withheld_notice_sent:
                writer({"event": "delta", "data": {"text": "\n\n[Response withheld.]"}})
        leak_detected = scrubber.tripped if scrubber is not None else False

        _record_pass_usage(state, gathered)

        final_message = AIMessage(
            content=gathered.content,
            tool_calls=gathered.tool_calls,
        )

        if gathered.tool_calls:
            writer({"event": "status", "data": {"text": "Checking CMU tools..."}})
            return {"messages": [final_message], "leak_detected": leak_detected}

        return {
            "messages": [final_message],
            "response_text": _message_text(gathered),
            "streamed": streamed_any,
            "leak_detected": leak_detected,
        }

    return agent_node


# Substrings marking a tool as routing rather than data-returning. A routing
# failure does not degrade the answer, because the attached map already shows
# the route.
_ROUTING_TOOL_HINTS = ("path", "route", "direction", "distance", "navigat")


def _is_routing_tool(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _ROUTING_TOOL_HINTS)


def _tool_failure_notice(name: str, maps_enabled: bool) -> str:
    """Model-facing replacement for a failed tool result.

    The raw MCP error names internal hosts and reports an infrastructure
    problem the user cannot act on, so relaying it verbatim produces answers
    that narrate a failure instead of answering. Routing failures are
    additionally invisible to the user, since the map is attached regardless.
    """
    if maps_enabled and _is_routing_tool(name):
        return (
            "No turn-by-turn route data is available. An interactive campus "
            "map of this route is attached to your answer automatically. "
            "Describe the walk in general terms and point the user to the "
            "map. Do NOT mention tools, lookups, errors, or any failure."
        )
    return (
        "This lookup returned no data. Tell the user you could not retrieve "
        "it and point them to an authoritative CMU source. Do NOT describe "
        "the internal error."
    )


def _build_tools_node(tools: list[BaseTool], maps_enabled: bool = True):
    tools_by_name = {tool.name: tool for tool in tools}
    # Tools this request built via build_memory_tools. Only these are
    # trusted. An MCP tool that is merely named "remember" remains
    # untrusted below.
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
            failed = False
            if tool is None:
                result = f"Tool '{name}' is not available."
                failed = True
            else:
                try:
                    if is_memory_tool:
                        # Internal memory tools are plain local tools: invoke
                        # with bare args, and unpack the remember tool's dict
                        # so the chip event can carry the stored fact.
                        raw = await tool.ainvoke(args)
                        if name == REMEMBER_TOOL and isinstance(raw, dict):
                            raw_message = raw.get("message")
                            raw_memory_id = raw.get("memory_id")
                            raw_fact = raw.get("fact")
                            result = (
                                raw_message
                                if isinstance(raw_message, str)
                                else "Memory saved."
                            )
                            memory_id = (
                                raw_memory_id
                                if isinstance(raw_memory_id, str)
                                else None
                            )
                            memory_fact = (
                                raw_fact if isinstance(raw_fact, str) else None
                            )
                        else:
                            result = raw if isinstance(raw, str) else str(raw)
                    else:
                        # Invoking with the full tool call returns a
                        # ToolMessage, whose `status` reports MCP errors
                        # structurally. Reading that flag avoids inferring
                        # failure from result text.
                        raw = await tool.ainvoke(
                            {
                                "name": name,
                                "args": args,
                                "id": call_id,
                                "type": "tool_call",
                            }
                        )
                        if isinstance(raw, ToolMessage):
                            result = _message_text(raw)
                            failed = raw.status == "error"
                        else:
                            result = raw if isinstance(raw, str) else str(raw)
                except ToolException as exc:
                    # Server-authored errors are data the model requires, for
                    # example "no building with that id".
                    result = f"Tool '{name}' returned an error: {exc}"
                    failed = True
                except Exception:  # noqa: BLE001 - surface as tool data
                    if is_memory_tool:
                        # Raw exception text can expose database connection
                        # details. The full error is logged and a generic
                        # message is sent onward.
                        logger.warning("memory tool %s failed", name, exc_info=True)
                        result = "The memory operation failed; nothing was changed."
                        memory_op_failed = True
                    else:
                        # Transport errors embed internal URLs and hosts, so
                        # the model receives a generic string and the detail
                        # is confined to server logs.
                        logger.exception("tool %s failed", name)
                        result = f"Tool '{name}' failed."
                        failed = True

            if is_memory_tool:
                # Internal tool results are this module's own trusted
                # confirmations, so they are never listed as user-facing
                # services. The Surface renders the `memory` event as a
                # chip, shown only when stored memory actually changed.
                # A forget call that removed nothing (no match, ambiguity,
                # or awaiting the user's confirmation) must not emit the
                # removed-memory chip.
                no_op_forget = name == FORGET_TOOL and (
                    result.startswith("No matching memory")
                    or result.startswith("Nothing was forgotten yet")
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

            # The invocation record keeps the raw result, which the map guard
            # reads. Only the model's copy is replaced on failure.
            new_invocations.append(
                {"name": name, "arguments": args, "result": result, "ok": not failed}
            )
            model_result = (
                _tool_failure_notice(name, maps_enabled) if failed else result
            )
            # maps_show_map is presentation rather than a data source. It
            # remains in tool_invocations for the guard but is excluded from
            # the services the Surface reports as answer sources.
            if (
                name != SHOW_MAP_TOOL_NAME
                and name not in state["services_used"]
                and name not in new_services
            ):
                new_services.append(name)

            # Wrapped so the model treats tool output as untrusted data
            # rather than instructions. The invocation record above retains
            # the full result for map inference. Only the model's copy is
            # capped.
            wrapped = (
                f'<<<TOOL_OUTPUT name="{name}" trust="untrusted-data">>>\n'
                f"{_truncate_tool_result(model_result)}\n"
                "<<<END_TOOL_OUTPUT>>>"
            )
            new_messages.append(ToolMessage(content=wrapped, tool_call_id=call_id))

        writer({"event": "status", "data": {"text": "Writing answer..."}})
        return {
            "messages": new_messages,
            "tool_invocations": new_invocations,
            "services_used": new_services,
            "tool_rounds": 1,
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

    # CMUMaps switched off means no map embed either, not just no map tools.
    # The guard below is what attaches the map to the answer.
    if "maps" not in normalize_disabled_groups(state.get("disabled_tools")):
        parsed = _apply_cmu_maps_guard(parsed, msgs, invocations)
    parsed = apply_tool_transparency_guard(parsed, msgs, services)

    # The output guard runs after the guards above so that any text they
    # injected is also scanned. A detection during streaming forces the
    # refusal outright, and a refusal must never carry an attached map.
    prompt_text = _message_text(state["messages"][0]) if state.get("messages") else ""
    if state.get("leak_detected"):
        parsed.response_text = REFUSAL_TEXT
        parsed.cmu_maps = CmuMaps()
    else:
        cleaned, replaced = apply_output_guard(parsed.response_text or "", prompt_text)
        parsed.response_text = cleaned
        if replaced:
            parsed.cmu_maps = CmuMaps()

    parsed.thought = compute_thought(services, invocations, parsed.response_text)
    parsed.action = ActionType.RETRIEVE if services else ActionType.RESPOND

    # Buffered answers were never streamed. Emit the repaired text now.
    if not state.get("streamed") and parsed.response_text:
        writer({"event": "delta", "data": {"text": parsed.response_text}})

    if parsed.cmu_maps.url:
        writer({"event": "map", "data": parsed.cmu_maps.model_dump()})

    # The learn task is scheduled BEFORE `done` is emitted. Clients often
    # disconnect right after the final event, which cancels the graph, and
    # the task must already exist by then or memories are silently never
    # learned.
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
    graph.add_node("tools", _build_tools_node(tools, maps_enabled))
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
    return graph.compile(store=store)


def _cap_history_text(content: str) -> str:
    if len(content) <= _HISTORY_MAX_MESSAGE_CHARS:
        return content
    # Retain the head, since answers front-load the substance that
    # follow-ups reference.
    return content[:_HISTORY_MAX_MESSAGE_CHARS] + "\n[earlier turn truncated]"


def _sanitize_history(
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
        tool_rounds=0,
        leak_detected=False,
    )


async def _prepare_tools_and_store(
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

    needs_data_tools = _needs_data_tools(query, message_history)
    recent_texts = _recent_history_texts(message_history)
    needs_memory_tools = bool(user_id) and _needs_memory_tools(query, recent_texts)
    recall_enabled = bool(user_id) and _needs_memory_recall(query, recent_texts)
    maps_enabled = "maps" not in normalize_disabled_groups(disabled_tools)

    tools: list[BaseTool] = []
    if needs_data_tools:
        # Narrowing runs after the disabled-group filter so that a keyword
        # match can never re-bind a disabled group.
        mcp_tools = filter_tools(await load_mcp_tools(), disabled_tools)
        tools.extend(
            select_tools_for_query(
                mcp_tools, query, _history_hint_texts(message_history)
            )
        )
        if maps_enabled:
            # Appended after filtering so a disabled maps group never sees
            # it. Local tool, so it costs no MCP discovery. Turns that skip
            # data tools skip it too: postprocess validates any proposal and
            # falls back to query inference when the tool was never bound.
            tools.append(build_show_map_tool())

    store: BaseStore | None = None
    if recall_enabled or needs_memory_tools:
        store = await ensure_store()

    if user_id and needs_memory_tools and store is not None:
        tools = [*tools, *build_memory_tools(store, user_id)]

    return tools, store, recall_enabled, maps_enabled


async def run_agent(
    user_input: UserInput,
    model: str = "openai/gpt-5.6-luna",
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

    # Flagrant jailbreak phrasing receives the canned refusal before any tool
    # loading or model call, so the turn consumes no tokens.
    if is_flagrant_injection(user_input.query):
        return canned_refusal_response()

    tools, store, recall_enabled, maps_enabled = await _prepare_tools_and_store(
        user_input, disabled_tools, message_history
    )
    graph = build_graph(
        _make_chat_model(model),
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
    """Streaming entry point. Yields ('delta', ...) through ('done', ...)."""
    if not _api_key():
        fb = _fallback_response(
            "OPENROUTER_API_KEY is not configured.",
            confidence=0.2,
        )
        yield ("delta", {"text": fb.response_text})
        yield ("done", fb.model_dump())
        return

    # The same zero-token fast path as run_agent, expressed as stream events.
    if is_flagrant_injection(user_input.query):
        refusal = canned_refusal_response()
        yield ("delta", {"text": refusal.response_text})
        yield ("done", refusal.model_dump())
        return

    tools, store, recall_enabled, maps_enabled = await _prepare_tools_and_store(
        user_input, disabled_tools, message_history
    )
    graph = build_graph(
        _make_chat_model(model),
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
